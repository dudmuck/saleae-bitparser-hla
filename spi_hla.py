#!/usr/bin/env python3
"""
SPI decoder for Saleae binary exports that feeds data to a High Level Analyzer.

Reads digital_N.bin files (SCLK, MISO, MOSI, nSS) and decodes SPI transactions,
then feeds them to the HLA SPI parser.
"""

import struct
import sys
import os
import mmap
from typing import Iterator, Optional

import numpy as np

# Add the mock saleae module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saleae.analyzers import AnalyzerFrame

SALEAE_MAGIC = b'<SALEAE>'
HEADER_SIZE = 44  # 8 + 4 + 4 + 4 + 8 + 8 + 8

class ChannelData:
    """Holds data for a single digital channel using memory-mapped numpy array."""

    def __init__(self, initial_state: int, timestamps: np.ndarray):
        self.initial_state = initial_state
        self.timestamps = timestamps  # numpy array of float64

    def state_at_indices(self, indices: np.ndarray) -> np.ndarray:
        """Get channel state at given transition indices (vectorized)."""
        # State toggles with each transition, so odd indices flip the initial state
        return (self.initial_state + indices) & 1

def read_saleae_digital(filepath: str) -> Optional[ChannelData]:
    """Read a Saleae v2 digital binary file using memory mapping."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                return None

            magic = header[0:8]
            if magic != SALEAE_MAGIC:
                print(f"Invalid magic in {filepath}: {magic}", file=sys.stderr)
                return None

            version, type_, initial_state = struct.unpack('<iiI', header[8:20])
            begin_time, end_time, num_transitions = struct.unpack('<ddQ', header[20:44])

            if version != 0 or type_ != 0:
                print(f"Unsupported version/type in {filepath}", file=sys.stderr)
                return None

            # Memory-map the timestamps as numpy array
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            timestamps = np.frombuffer(mm, dtype=np.float64, offset=HEADER_SIZE)

            if len(timestamps) != num_transitions:
                print(f"Size mismatch in {filepath}", file=sys.stderr)
                return None

            return ChannelData(initial_state=initial_state, timestamps=timestamps)
    except FileNotFoundError:
        return None

class SPIDecoder:
    """Decodes SPI from raw digital signals using numpy for efficiency."""

    def __init__(self, sclk: ChannelData, miso: ChannelData, mosi: ChannelData, nss: ChannelData,
                 cpol: int = 0, cpha: int = 0):
        self.sclk = sclk
        self.miso = miso
        self.mosi = mosi
        self.nss = nss
        self.cpol = cpol
        self.cpha = cpha

        # Precompute which SCLK edges are sample edges
        # For CPOL=0, CPHA=0: sample on rising edge (odd indices if initial=0)
        # For CPOL=0, CPHA=1: sample on falling edge
        # For CPOL=1, CPHA=0: sample on falling edge
        # For CPOL=1, CPHA=1: sample on rising edge
        sample_on_rising = (cpol == cpha)

        # Determine which SCLK transition indices are sample edges
        n_sclk = len(sclk.timestamps)
        indices = np.arange(n_sclk)
        # After transition i, state = initial ^ ((i+1) & 1) for alternating
        # Rising edge: state goes 0->1, meaning before transition state=0
        # state before transition i = initial ^ (i & 1)
        state_before = (sclk.initial_state + indices) & 1
        is_rising = (state_before == 0)

        if sample_on_rising:
            self.sample_edge_mask = is_rising
        else:
            self.sample_edge_mask = ~is_rising

        # Get sample edge timestamps and their indices
        self.sample_edge_indices = np.where(self.sample_edge_mask)[0]
        self.sample_edge_times = sclk.timestamps[self.sample_edge_indices]

    def decode(self) -> Iterator[AnalyzerFrame]:
        """Decode SPI transactions and yield AnalyzerFrames."""
        nss = self.nss

        # Find nSS falling edges (transaction starts) and rising edges (transaction ends)
        n_nss = len(nss.timestamps)
        if n_nss == 0:
            return

        indices = np.arange(n_nss)
        state_after = (nss.initial_state + indices + 1) & 1

        falling_indices = np.where(state_after == 0)[0]
        rising_indices = np.where(state_after == 1)[0]

        falling_times = nss.timestamps[falling_indices]
        rising_times = nss.timestamps[rising_indices]

        # For each falling edge, find corresponding rising edge
        # Use searchsorted to find the first rising edge after each falling edge
        rising_after_falling = np.searchsorted(rising_times, falling_times, side='right')

        n_transactions = len(falling_times)
        print(f"Processing {n_transactions} SPI transactions...", file=sys.stderr)

        for i in range(n_transactions):
            ts_fall = falling_times[i]

            # Find rising edge
            rise_idx = rising_after_falling[i]
            if rise_idx < len(rising_times):
                ts_rise = rising_times[rise_idx]
            else:
                # No rising edge found, use end of data
                ts_rise = self.sclk.timestamps[-1] if len(self.sclk.timestamps) > 0 else ts_fall + 1

            # Emit 'enable' frame
            yield AnalyzerFrame('enable', ts_fall, ts_fall)

            # Find sample edges within this transaction using binary search
            start_idx = np.searchsorted(self.sample_edge_times, ts_fall, side='right')
            end_idx = np.searchsorted(self.sample_edge_times, ts_rise, side='left')

            if start_idx < end_idx:
                edge_times = self.sample_edge_times[start_idx:end_idx]

                # Sample MISO and MOSI at each edge using binary search
                miso_indices = np.searchsorted(self.miso.timestamps, edge_times, side='right')
                mosi_indices = np.searchsorted(self.mosi.timestamps, edge_times, side='right')

                # State = initial ^ (num_transitions_before & 1)
                miso_bits = (self.miso.initial_state + miso_indices) & 1
                mosi_bits = (self.mosi.initial_state + mosi_indices) & 1

                # Group into bytes (8 bits each)
                n_bits = len(edge_times)
                n_bytes = n_bits // 8

                for b in range(n_bytes):
                    bit_start = b * 8
                    bit_end = bit_start + 8

                    # Convert 8 bits to byte (MSB first)
                    mosi_byte = 0
                    miso_byte = 0
                    for j in range(8):
                        mosi_byte = (mosi_byte << 1) | mosi_bits[bit_start + j]
                        miso_byte = (miso_byte << 1) | miso_bits[bit_start + j]

                    yield AnalyzerFrame('result', edge_times[bit_end - 1], edge_times[bit_end - 1], {
                        'mosi': bytes([mosi_byte]),
                        'miso': bytes([miso_byte])
                    })

            # Emit 'disable' frame
            yield AnalyzerFrame('disable', ts_rise, ts_rise)

            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{n_transactions} transactions...", file=sys.stderr)

def read_channel_names(directory: str) -> dict:
    """Read channel names from digital.csv header and return name->index mapping."""
    csv_path = os.path.join(directory, "digital.csv")
    try:
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
            # Header format: "Time [s],SCLK,MISO,MOSI,nSS,..."
            columns = header.split(',')
            # Skip first column (Time [s]), remaining columns are channel names
            channel_names = {name.strip().upper(): i for i, name in enumerate(columns[1:])}
            return channel_names
    except FileNotFoundError:
        return {}

def find_spi_channels(channel_names: dict) -> dict:
    """Find SPI channel numbers from channel name mapping."""
    # Common variations of SPI signal names
    sclk_names = ['SCLK', 'SCK', 'CLK', 'SPI_SCLK', 'SPI_CLK']
    miso_names = ['MISO', 'SDO', 'DO', 'SPI_MISO', 'DOUT']
    mosi_names = ['MOSI', 'SDI', 'DI', 'SPI_MOSI', 'DIN']
    nss_names = ['NSS', 'CS', 'NCS', 'SS', 'SSEL', 'SPI_NSS', 'SPI_CS', 'ENABLE']

    result = {}
    for signal, names in [('sclk', sclk_names), ('miso', miso_names),
                          ('mosi', mosi_names), ('nss', nss_names)]:
        for name in names:
            if name in channel_names:
                result[signal] = channel_names[name]
                break
    return result

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Decode SPI from Saleae binary exports and run HLA')
    parser.add_argument('directory', help='Directory containing digital_N.bin files')
    parser.add_argument('--hla-path', type=str, default=None,
                        help='Path to HLA directory (required)')
    parser.add_argument('--sclk', type=int, default=None, help='SCLK channel number (auto-detect from CSV if not specified)')
    parser.add_argument('--miso', type=int, default=None, help='MISO channel number (auto-detect from CSV if not specified)')
    parser.add_argument('--mosi', type=int, default=None, help='MOSI channel number (auto-detect from CSV if not specified)')
    parser.add_argument('--nss', type=int, default=None, help='nSS channel number (auto-detect from CSV if not specified)')
    parser.add_argument('--cpol', type=int, default=0, choices=[0, 1], help='Clock polarity')
    parser.add_argument('--cpha', type=int, default=0, choices=[0, 1], help='Clock phase')
    parser.add_argument('--hex', action='store_true', help='Print MOSI/MISO bytes in hex before each decoded line')
    parser.add_argument('--int-pin', type=str, default=None, metavar='NAME',
                        help='Log transitions of interrupt pin (e.g., --int-pin int)')
    parser.add_argument('--extra-pin', type=str, default=None, metavar='NAME',
                        help='Log transitions of an extra pin (e.g., --extra-pin busy)')

    args = parser.parse_args()

    # Check required arguments
    if args.hla_path is None:
        print("Error: --hla-path is required", file=sys.stderr)
        sys.exit(1)

    # Try to auto-detect channel numbers from digital.csv
    channel_names = read_channel_names(args.directory)
    if channel_names:
        print(f"Found channel names in digital.csv: {list(channel_names.keys())}", file=sys.stderr)
        auto_channels = find_spi_channels(channel_names)
    else:
        auto_channels = {}

    # Command-line args override auto-detected values
    spi_channels = {}
    missing = []
    for signal in ['sclk', 'miso', 'mosi', 'nss']:
        arg_val = getattr(args, signal)
        if arg_val is not None:
            spi_channels[signal] = arg_val
        elif signal in auto_channels:
            spi_channels[signal] = auto_channels[signal]
        else:
            missing.append(signal.upper())

    if missing:
        print(f"Error: Could not find channel numbers for: {', '.join(missing)}", file=sys.stderr)
        if not channel_names:
            print(f"No digital.csv found in {args.directory}", file=sys.stderr)
        else:
            print(f"Channel names in CSV: {list(channel_names.keys())}", file=sys.stderr)
            print(f"Could not match SPI signals. Use --sclk, --miso, --mosi, --nss to specify manually.", file=sys.stderr)
        sys.exit(1)

    # Load channel data
    print(f"Loading channels from {args.directory}/digital_*.bin ...", file=sys.stderr)

    channels = {}
    for name, num in spi_channels.items():
        filepath = os.path.join(args.directory, f"digital_{num}.bin")
        data = read_saleae_digital(filepath)
        if data is None:
            print(f"Failed to load {name} from {filepath}", file=sys.stderr)
            sys.exit(1)
        channels[name] = data
        print(f"  {name} (ch{num}): initial={data.initial_state}, transitions={len(data.timestamps)}",
              file=sys.stderr)

    # Load optional logging pins (int-pin, extra-pin)
    log_pins = []  # List of (name, data) tuples
    for pin_arg, pin_label in [(args.int_pin, 'int-pin'), (args.extra_pin, 'extra-pin')]:
        if pin_arg:
            pin_name = pin_arg.upper()
            if pin_name not in channel_names:
                print(f"Error: --{pin_label} '{pin_arg}' not found in CSV", file=sys.stderr)
                print(f"Available channels: {list(channel_names.keys())}", file=sys.stderr)
                sys.exit(1)
            pin_num = channel_names[pin_name]
            filepath = os.path.join(args.directory, f"digital_{pin_num}.bin")
            pin_data = read_saleae_digital(filepath)
            if pin_data is None:
                print(f"Failed to load {pin_arg} from {filepath}", file=sys.stderr)
                sys.exit(1)
            print(f"  {pin_arg} (ch{pin_num}): initial={pin_data.initial_state}, transitions={len(pin_data.timestamps)}",
                  file=sys.stderr)
            log_pins.append((pin_name, pin_data))

    # Create SPI decoder
    print("Initializing SPI decoder...", file=sys.stderr)
    decoder = SPIDecoder(
        sclk=channels['sclk'],
        miso=channels['miso'],
        mosi=channels['mosi'],
        nss=channels['nss'],
        cpol=args.cpol,
        cpha=args.cpha
    )

    # Add HLA path to sys.path and import it
    sys.path.insert(0, args.hla_path)

    try:
        from HighLevelAnalyzer import Hla
    except ImportError as e:
        print(f"Failed to import HLA: {e}", file=sys.stderr)
        sys.exit(1)

    # Create HLA instance and process frames
    hla = Hla()

    # Track MOSI/MISO bytes for --hex option
    mosi_bytes = bytearray()
    miso_bytes = bytearray()

    # Track logging pin transitions - list of [name, data, index, state]
    pin_trackers = [[name, data, 0, data.initial_state] for name, data in log_pins]

    def print_pin_transitions_before(timestamp):
        """Print any pin transitions that occurred before the given timestamp."""
        # Collect all pending transitions across all pins
        while True:
            # Find the earliest pending transition across all pins
            earliest_ts = float('inf')
            earliest_tracker = None
            for tracker in pin_trackers:
                name, data, idx, state = tracker
                if idx < len(data.timestamps) and data.timestamps[idx] < timestamp:
                    if data.timestamps[idx] < earliest_ts:
                        earliest_ts = data.timestamps[idx]
                        earliest_tracker = tracker

            if earliest_tracker is None:
                break

            # Print this transition
            name, data, idx, state = earliest_tracker
            new_state = state ^ 1
            edge = "rising" if new_state == 1 else "falling"
            print(f"{earliest_ts:.9f}: *** {name} {edge} edge ***")

            # Update tracker
            earliest_tracker[2] = idx + 1  # increment index
            earliest_tracker[3] = new_state  # update state

    for frame in decoder.decode():
        if args.hex:
            if frame.type == 'enable':
                mosi_bytes = bytearray()
                miso_bytes = bytearray()
            elif frame.type == 'result':
                mosi_bytes += frame.data['mosi']
                miso_bytes += frame.data['miso']

        result = hla.decode(frame)
        if result is not None:
            # Print any pin transitions before this SPI transaction
            print_pin_transitions_before(result.start_time)

            # Print the decoded message
            msg = result.data.get('string', '')
            if args.hex:
                print(f"  MOSI: {mosi_bytes.hex(' ')}")
                print(f"  MISO: {miso_bytes.hex(' ')}")
            print(f"{result.start_time:.9f}: {msg}")

    # Print any remaining pin transitions
    print_pin_transitions_before(float('inf'))

if __name__ == '__main__':
    main()
