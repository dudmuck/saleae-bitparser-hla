#!/usr/bin/env python3
"""
SPI decoder for Saleae binary exports that feeds data to a High Level Analyzer.

Reads digital_N.bin files (SCLK, MISO, MOSI, nSS) and decodes SPI transactions,
then feeds them to the HLA SPI parser.
"""

import contextlib
import heapq
import struct
import sys
import os
import mmap
from typing import Iterator, Optional

import numpy as np

# Try to import numba for JIT acceleration
try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def _bits_to_bytes_numpy(mosi_bits: np.ndarray, miso_bits: np.ndarray, n_bytes: int):
    """Convert bit arrays to byte arrays using NumPy (fallback)."""
    bits_truncated = n_bytes * 8
    mosi_bits_2d = mosi_bits[:bits_truncated].reshape(n_bytes, 8)
    miso_bits_2d = miso_bits[:bits_truncated].reshape(n_bytes, 8)
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8))
    mosi_bytes = (mosi_bits_2d * weights).sum(axis=1).astype(np.uint8)
    miso_bytes = (miso_bits_2d * weights).sum(axis=1).astype(np.uint8)
    return mosi_bytes, miso_bytes


if HAVE_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _bits_to_bytes_numba(mosi_bits: np.ndarray, miso_bits: np.ndarray, n_bytes: int):
        """Convert bit arrays to byte arrays using Numba JIT (fast path)."""
        mosi_bytes = np.empty(n_bytes, dtype=np.uint8)
        miso_bytes = np.empty(n_bytes, dtype=np.uint8)
        for b in range(n_bytes):
            mosi_byte = 0
            miso_byte = 0
            base = b * 8
            for j in range(8):
                mosi_byte = (mosi_byte << 1) | mosi_bits[base + j]
                miso_byte = (miso_byte << 1) | miso_bits[base + j]
            mosi_bytes[b] = mosi_byte
            miso_bytes[b] = miso_byte
        return mosi_bytes, miso_bytes

    bits_to_bytes = _bits_to_bytes_numba
else:
    bits_to_bytes = _bits_to_bytes_numpy

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

        # Pre-compute MISO/MOSI bit values for ALL sample edges (avoids per-transaction searchsorted)
        miso_indices = np.searchsorted(miso.timestamps, self.sample_edge_times, side='right')
        mosi_indices = np.searchsorted(mosi.timestamps, self.sample_edge_times, side='right')
        self.all_miso_bits = ((miso.initial_state + miso_indices) & 1).astype(np.uint8)
        self.all_mosi_bits = ((mosi.initial_state + mosi_indices) & 1).astype(np.uint8)

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

                # Use pre-computed MISO/MOSI bits (slicing is O(1))
                miso_bits = self.all_miso_bits[start_idx:end_idx]
                mosi_bits = self.all_mosi_bits[start_idx:end_idx]

                # Group into bytes (8 bits each) - vectorized/JIT
                n_bits = len(edge_times)
                n_bytes = n_bits // 8

                if n_bytes > 0:
                    # Convert bits to bytes using Numba JIT (if available) or NumPy
                    mosi_byte_array, miso_byte_array = bits_to_bytes(mosi_bits, miso_bits, n_bytes)

                    # Get timestamps for each byte (last bit of each byte)
                    byte_end_indices = np.arange(8, n_bytes * 8 + 1, 8) - 1
                    byte_timestamps = edge_times[byte_end_indices]

                    # Yield frames for each byte
                    for b in range(n_bytes):
                        yield AnalyzerFrame('result', byte_timestamps[b], byte_timestamps[b], {
                            'mosi': bytes([mosi_byte_array[b]]),
                            'miso': bytes([miso_byte_array[b]])
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

def find_spi_channels(channel_names: dict, nss_name: str = None) -> list:
    """Find SPI channel numbers from channel name mapping.

    Returns a list of dicts, one per SPI port found. Each dict contains:
    - 'name': port name (e.g., 'SPI', 'SPI_B')
    - 'sclk', 'miso', 'mosi', 'nss': channel numbers

    If nss_name is provided, use that specific channel for the chip select
    instead of auto-detecting.
    """
    # Common variations of SPI signal names (base names without suffix)
    sclk_bases = ['SCLK', 'SCK', 'CLK', 'SPI_SCLK', 'SPI_CLK']
    miso_bases = ['MISO', 'SDO', 'DO', 'SPI_MISO', 'DOUT']
    mosi_bases = ['MOSI', 'SDI', 'DI', 'SPI_MOSI', 'DIN']
    nss_bases = ['NSS', 'CS', 'NCS', 'SS', 'SSEL', 'SPI_NSS', 'SPI_CS', 'ENABLE']

    # Common suffixes for additional SPI ports
    suffixes = ['', '_B', '_C', '_D', '_2', '_3', '_4', '2', '3', '4', 'B', 'C', 'D']

    ports = []
    used_channels = set()

    for suffix in suffixes:
        port = {}
        port_name = f"SPI{suffix}" if suffix else "SPI"

        for signal, bases in [('sclk', sclk_bases), ('miso', miso_bases),
                              ('mosi', mosi_bases), ('nss', nss_bases)]:
            # If nss_name was explicitly provided, use it for nss signal
            if signal == 'nss' and nss_name and suffix == '':
                upper = nss_name.upper()
                if upper in channel_names and channel_names[upper] not in used_channels:
                    port[signal] = channel_names[upper]
                    continue
                else:
                    break

            found = False
            for base in bases:
                # Try exact suffix match
                name = f"{base}{suffix}" if suffix else base
                if name in channel_names and channel_names[name] not in used_channels:
                    port[signal] = channel_names[name]
                    found = True
                    break
                # Also try lowercase suffix
                name_lower = f"{base}{suffix.lower()}" if suffix else base
                if name_lower in channel_names and channel_names[name_lower] not in used_channels:
                    port[signal] = channel_names[name_lower]
                    found = True
                    break
            if found:
                continue

            # Fallback: find channel names that end with a known base name
            # This handles names like "TX RADIO NSS" matching the "NSS" base
            if suffix == '':
                for base in bases:
                    candidates = [n for n in channel_names
                                  if (n.endswith(' ' + base) or n.endswith('_' + base))
                                  and channel_names[n] not in used_channels]
                    if len(candidates) == 1:
                        port[signal] = channel_names[candidates[0]]
                        found = True
                        break
            if not found:
                break

        # Only add port if all 4 signals were found
        if len(port) == 4:
            port['name'] = port_name
            used_channels.update(port[s] for s in ['sclk', 'miso', 'mosi', 'nss'])
            ports.append(port)

    return ports

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Decode SPI from Saleae binary exports and run HLA')
    parser.add_argument('directory', help='Directory containing digital_N.bin files')
    parser.add_argument('--hla-path', type=str, default=None,
                        help='Path to HLA directory (required)')
    parser.add_argument('--sclk', type=str, default=None, help='SCLK channel number or name (auto-detect from CSV if not specified)')
    parser.add_argument('--miso', type=str, default=None, help='MISO channel number or name (auto-detect from CSV if not specified)')
    parser.add_argument('--mosi', type=str, default=None, help='MOSI channel number or name (auto-detect from CSV if not specified)')
    parser.add_argument('--nss', type=str, default=None, help='nSS channel number or name (auto-detect from CSV if not specified)')
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

    # Report acceleration status and warm up JIT
    if HAVE_NUMBA:
        print("Numba JIT acceleration: enabled (compiling...)", file=sys.stderr, end='', flush=True)
        # Warmup call to trigger JIT compilation before processing
        _dummy = bits_to_bytes(np.array([0,1,0,1,0,1,0,1], dtype=np.uint8),
                               np.array([1,0,1,0,1,0,1,0], dtype=np.uint8), 1)
        print(" ready", file=sys.stderr)
    else:
        print("Numba JIT acceleration: disabled (install numba for faster processing)", file=sys.stderr)

    # Try to auto-detect channel numbers from digital.csv
    channel_names = read_channel_names(args.directory)
    if channel_names:
        print(f"Found channel names in digital.csv: {list(channel_names.keys())}", file=sys.stderr)
        auto_ports = find_spi_channels(channel_names, nss_name=args.nss)
    else:
        auto_ports = []

    # Resolve a CLI arg (number or channel name) to a channel number
    def resolve_channel(arg_val, signal_name):
        """Resolve a channel arg that may be a number or a name."""
        if arg_val is None:
            return None
        try:
            return int(arg_val)
        except ValueError:
            upper = arg_val.upper()
            if channel_names and upper in channel_names:
                return channel_names[upper]
            print(f"Error: --{signal_name} '{arg_val}' not found in CSV channel names", file=sys.stderr)
            if channel_names:
                print(f"Available channels: {list(channel_names.keys())}", file=sys.stderr)
            sys.exit(1)

    # Command-line args override auto-detected values for the primary port
    spi_ports = []
    if args.sclk is not None or args.miso is not None or args.mosi is not None or args.nss is not None:
        # Manual specification - only one port
        manual_port = {'name': 'SPI'}
        missing = []
        for signal in ['sclk', 'miso', 'mosi', 'nss']:
            arg_val = resolve_channel(getattr(args, signal), signal)
            if arg_val is not None:
                manual_port[signal] = arg_val
            elif auto_ports and signal in auto_ports[0]:
                manual_port[signal] = auto_ports[0][signal]
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
        spi_ports = [manual_port]
    elif auto_ports:
        spi_ports = auto_ports
    else:
        print(f"Error: No SPI channels found", file=sys.stderr)
        if not channel_names:
            print(f"No digital.csv found in {args.directory}", file=sys.stderr)
        else:
            print(f"Channel names in CSV: {list(channel_names.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"Detected {len(spi_ports)} SPI port(s): {[p['name'] for p in spi_ports]}", file=sys.stderr)

    # Load channel data for all ports
    print(f"Loading channels from {args.directory}/digital_*.bin ...", file=sys.stderr)

    all_port_channels = []
    for port in spi_ports:
        channels = {}
        print(f"  {port['name']}:", file=sys.stderr)
        for signal in ['sclk', 'miso', 'mosi', 'nss']:
            num = port[signal]
            filepath = os.path.join(args.directory, f"digital_{num}.bin")
            data = read_saleae_digital(filepath)
            if data is None:
                print(f"Failed to load {signal} from {filepath}", file=sys.stderr)
                sys.exit(1)
            channels[signal] = data
            print(f"    {signal} (ch{num}): initial={data.initial_state}, transitions={len(data.timestamps)}",
                  file=sys.stderr)
        all_port_channels.append((port['name'], channels))

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

    # Create SPI decoders for all ports
    print("Initializing SPI decoder(s)...", file=sys.stderr)
    decoders = []
    for port_name, channels in all_port_channels:
        decoder = SPIDecoder(
            sclk=channels['sclk'],
            miso=channels['miso'],
            mosi=channels['mosi'],
            nss=channels['nss'],
            cpol=args.cpol,
            cpha=args.cpha
        )
        decoders.append((port_name, decoder))
        print(f"  {port_name}: initialized", file=sys.stderr)

    # Add HLA path to sys.path and import it
    sys.path.insert(0, args.hla_path)

    try:
        from HighLevelAnalyzer import Hla
    except ImportError as e:
        print(f"Failed to import HLA: {e}", file=sys.stderr)
        sys.exit(1)

    # Create HLA instance for each port
    hla_instances = {port_name: Hla() for port_name, _ in decoders}

    # Track MOSI/MISO bytes for --hex option (per port)
    mosi_bytes = {port_name: bytearray() for port_name, _ in decoders}
    miso_bytes = {port_name: bytearray() for port_name, _ in decoders}

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

    # Collect all decoded results from all ports with timestamps for interleaving
    # Use a priority queue to interleave results from multiple ports by timestamp
    # Each entry is (timestamp, port_name, result, mosi_hex, miso_hex)
    results_heap = []

    # Redirect stdout to stderr during HLA decode to capture any HLA print statements
    with contextlib.redirect_stdout(sys.stderr):
        for port_name, decoder in decoders:
            hla = hla_instances[port_name]

            for frame in decoder.decode():
                if args.hex:
                    if frame.type == 'enable':
                        mosi_bytes[port_name] = bytearray()
                        miso_bytes[port_name] = bytearray()
                    elif frame.type == 'result':
                        mosi_bytes[port_name] += frame.data['mosi']
                        miso_bytes[port_name] += frame.data['miso']

                try:
                    result = hla.decode(frame)
                except Exception as e:
                    # Report error to stderr and queue for stdout output
                    error_msg = f"{type(e).__name__}: {e}"
                    print(f"Error decoding {port_name} frame at {frame.start_time:.9f}: {error_msg}", file=sys.stderr)
                    mosi_hex = mosi_bytes[port_name].hex(' ') if args.hex else None
                    miso_hex = miso_bytes[port_name].hex(' ') if args.hex else None
                    # Push error as a tuple with None result and error message
                    heapq.heappush(results_heap, (frame.start_time, port_name, None, mosi_hex, miso_hex, error_msg))
                    continue
                if result is not None:
                    mosi_hex = mosi_bytes[port_name].hex(' ') if args.hex else None
                    miso_hex = miso_bytes[port_name].hex(' ') if args.hex else None
                    heapq.heappush(results_heap, (result.start_time, port_name, result, mosi_hex, miso_hex, None))

    # Print results in timestamp order
    show_port_prefix = len(decoders) > 1
    while results_heap:
        timestamp, port_name, result, mosi_hex, miso_hex, error_msg = heapq.heappop(results_heap)

        # Print any pin transitions before this SPI transaction
        print_pin_transitions_before(timestamp)

        # Print hex bytes if requested
        if args.hex:
            prefix = f"  [{port_name}] " if show_port_prefix else "  "
            print(f"{prefix}MOSI: {mosi_hex}")
            print(f"{prefix}MISO: {miso_hex}")

        port_prefix = f"[{port_name}] " if show_port_prefix else ""
        if error_msg is not None:
            # Print error entry
            print(f"{timestamp:.9f}: {port_prefix}*** DECODE ERROR: {error_msg} ***")
        else:
            # Print the decoded message
            msg = result.data.get('string', '')
            print(f"{timestamp:.9f}: {port_prefix}{msg}")

    # Print any remaining pin transitions
    print_pin_transitions_before(float('inf'))

if __name__ == '__main__':
    main()
