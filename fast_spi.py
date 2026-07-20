#!/usr/bin/env python3
"""Vectorized streaming SPI decoder for uniformly sampled logic data.

Replaces the libsigrokdecode SPI protocol decoder in the sigrok_hla.py
pipeline. The PD walks one clock edge per Python wait() call, which runs
roughly 20x slower than real time on dual 10 MHz SPI captured at 25 MSa/s;
this module does the same job with NumPy over whole chunks, fast enough to
keep up with capture (which also prevents the USB overruns that pipeline
back-pressure was causing during bursts).

Input is the raw sample stream libsigrok emits with ``-O binary``: one byte
per sample, bit N = logic channel N (unitsize 1, i.e. up to 8 channels).
Output is the same 'enable'/'result'/'disable' AnalyzerFrame sequence the
Saleae HLAs consume, so it drops into the existing harness unchanged.

Decoding is per SPI port and fully vectorized:
  - nSS edges frame transactions,
  - SCLK sample edges (selected by CPOL/CPHA) index MISO/MOSI directly in
    the sample domain (no searchsorted needed — unlike the transition-list
    input that spi_hla.py works from),
  - bits are packed to bytes with the shared Numba kernel.

State is carried across chunk boundaries so a transaction, a partial byte,
or an edge spanning two chunks decodes identically to a whole-buffer run.
"""

import sys

import numpy as np

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from saleae.analyzers import AnalyzerFrame


def _bits_to_bytes_numpy(mosi_bits, miso_bits, n_bytes):
    """Pack MSB-first bit arrays into bytes (NumPy fallback)."""
    n = n_bytes * 8
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint16))
    mosi = (mosi_bits[:n].reshape(n_bytes, 8) * weights).sum(axis=1).astype(np.uint8)
    miso = (miso_bits[:n].reshape(n_bytes, 8) * weights).sum(axis=1).astype(np.uint8)
    return mosi, miso


if HAVE_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _bits_to_bytes_numba(mosi_bits, miso_bits, n_bytes):
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


# Cap on buffered bits for a transaction that never ends (CS stuck asserted),
# so a wiring fault cannot exhaust memory. 8 Mbit ~= 1 M bytes of payload.
MAX_OPEN_BITS = 8 * 1024 * 1024

# One immutable bytes object per possible byte value: a long capture yields
# millions of result frames, and reusing these avoids as many allocations.
_BYTE = tuple(bytes([i]) for i in range(256))


def find_edges(bits, prev_bit):
    """Indices where ``bits`` differs from the preceding sample."""
    if prev_bit is None:
        # No history: the first sample cannot be an edge.
        return np.flatnonzero(bits[1:] != bits[:-1]) + 1
    rest = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    if bits[0] != prev_bit:
        return np.concatenate((np.array([0], dtype=np.int64), rest))
    return rest


class PinLogger:
    """Reports edges on logic channels that are not part of an SPI port.

    Used to follow an interrupt or BUSY line alongside the decoded traffic:
    the events carry timestamps, so the caller can interleave them with
    decoded transactions and see exactly where a pin asserted.
    """

    def __init__(self, pins, samplerate):
        """pins: sequence of (name, channel index) pairs."""
        self.pins = list(pins)
        self.samplerate = float(samplerate)
        self.abs_pos = 0
        self.prev_sample = None

    def feed(self, chunk):
        """Return [(time, name, 'rising'|'falling'), ...] for this chunk."""
        events = []
        if chunk.size == 0:
            return events
        prev = self.prev_sample
        for name, bit in self.pins:
            bits = (chunk >> bit) & 1
            prev_bit = None if prev is None else (prev >> bit) & 1
            idx = find_edges(bits, prev_bit)
            if idx.size == 0:
                continue
            times = (idx + self.abs_pos) / self.samplerate
            vals = bits[idx]
            for t, v in zip(times.tolist(), vals.tolist()):
                events.append((t, name, 'rising' if v else 'falling'))
        self.abs_pos += chunk.size
        self.prev_sample = int(chunk[-1])
        return events


class SpiPortDecoder:
    """Streaming SPI decoder for one port (CLK/MISO/MOSI/CS channel indices).

    Feed chunks with :meth:`feed`; each call yields the AnalyzerFrames whose
    transactions completed within that chunk. Call :meth:`end` to flush a
    transaction still open at end of stream.
    """

    def __init__(self, name, clk, miso, mosi, cs, samplerate,
                 cpol=0, cpha=0, cs_active_low=True):
        self.name = name
        self.clk = clk
        self.miso = miso
        self.mosi = mosi
        self.cs = cs
        self.samplerate = float(samplerate)
        self.cs_active_low = cs_active_low
        # CPOL==CPHA -> sample on rising edge (mode 0 and 3), else falling.
        self.sample_on_rising = (cpol == cpha)

        self.abs_pos = 0        # absolute sample index of next incoming chunk
        self.prev_sample = None # last sample byte of the previous chunk
        self.cs_asserted = False
        self.open_start = None  # abs sample index where the open transaction began
        self.mosi_bits = []     # buffered bit arrays for the open transaction
        self.miso_bits = []
        self.bit_times = []     # absolute sample index of each buffered bit
        self.open_bit_count = 0
        self.overflowed = False

    # -- helpers ---------------------------------------------------------

    def _t(self, abs_sample):
        return abs_sample / self.samplerate

    def _edges(self, bits, prev_bit):
        return find_edges(bits, prev_bit)

    def _emit_transaction(self, start_abs, end_abs, frames):
        """Close the open transaction, appending its frames."""
        frames.append(AnalyzerFrame('enable', self._t(start_abs), self._t(start_abs)))

        if self.mosi_bits:
            mosi = np.concatenate(self.mosi_bits) if len(self.mosi_bits) > 1 \
                else self.mosi_bits[0]
            miso = np.concatenate(self.miso_bits) if len(self.miso_bits) > 1 \
                else self.miso_bits[0]
            times = np.concatenate(self.bit_times) if len(self.bit_times) > 1 \
                else self.bit_times[0]
            n_bytes = len(mosi) // 8
            if n_bytes:
                mosi_b, miso_b = bits_to_bytes(mosi, miso, n_bytes)
                # Timestamp each byte at its last bit, like the PD does.
                byte_times = times[7::8][:n_bytes] / self.samplerate
                append = frames.append
                for t, mo, mi in zip(byte_times.tolist(),
                                     mosi_b.tolist(), miso_b.tolist()):
                    append(AnalyzerFrame('result', t, t,
                                         {'mosi': _BYTE[mo], 'miso': _BYTE[mi]}))

        frames.append(AnalyzerFrame('disable', self._t(end_abs), self._t(end_abs)))
        self.mosi_bits = []
        self.miso_bits = []
        self.bit_times = []
        self.open_bit_count = 0
        self.overflowed = False

    def _buffer_bits(self, clk_bits, miso_bits, mosi_bits, lo, hi, prev_clk, base):
        """Latch MISO/MOSI at sample edges of CLK within [lo, hi)."""
        if hi <= lo:
            return
        seg = clk_bits[lo:hi]
        pc = prev_clk if lo == 0 else clk_bits[lo - 1]
        idx = self._edges(seg, pc)
        if idx.size == 0:
            return
        # Keep only edges of the sampling polarity: value AT the edge is 1
        # for a rising edge, 0 for a falling edge.
        vals = seg[idx]
        idx = idx[vals == (1 if self.sample_on_rising else 0)]
        if idx.size == 0:
            return
        if self.open_bit_count >= MAX_OPEN_BITS:
            if not self.overflowed:
                print(f"[fast_spi] {self.name}: transaction exceeded "
                      f"{MAX_OPEN_BITS} bits, truncating", file=sys.stderr)
                self.overflowed = True
            return
        self.mosi_bits.append(mosi_bits[lo:hi][idx])
        self.miso_bits.append(miso_bits[lo:hi][idx])
        self.bit_times.append((idx + lo + base).astype(np.float64))
        self.open_bit_count += idx.size

    # -- public API ------------------------------------------------------

    def feed(self, chunk):
        """Process one chunk of packed samples; return a list of frames."""
        frames = []
        if chunk.size == 0:
            return frames

        clk = (chunk >> self.clk) & 1
        miso = (chunk >> self.miso) & 1
        mosi = (chunk >> self.mosi) & 1
        cs = (chunk >> self.cs) & 1

        prev = self.prev_sample
        prev_clk = None if prev is None else (prev >> self.clk) & 1
        prev_cs = None if prev is None else (prev >> self.cs) & 1

        # CS assertion state as a boolean per sample.
        asserted = (cs == 0) if self.cs_active_low else (cs == 1)
        cs_edges = self._edges(cs, prev_cs)

        base = self.abs_pos
        pos = 0
        for e in cs_edges:
            now_asserted = bool(asserted[e])
            if now_asserted and not self.cs_asserted:
                # Transaction starts here; nothing to latch before it.
                self.cs_asserted = True
                self.open_start = base + int(e)
                self.mosi_bits = []
                self.miso_bits = []
                self.bit_times = []
                self.open_bit_count = 0
            elif not now_asserted and self.cs_asserted:
                # Latch the tail of the transaction, then close it.
                self._buffer_bits(clk, miso, mosi, pos, int(e), prev_clk, base)
                self._emit_transaction(self.open_start, base + int(e), frames)
                self.cs_asserted = False
                self.open_start = None
            pos = int(e)

        if self.cs_asserted:
            self._buffer_bits(clk, miso, mosi, pos, chunk.size, prev_clk, base)

        self.abs_pos += chunk.size
        self.prev_sample = int(chunk[-1])
        return frames

    def end(self):
        """Flush a transaction left open at end of stream."""
        frames = []
        if self.cs_asserted and self.open_start is not None:
            self._emit_transaction(self.open_start, self.abs_pos, frames)
            self.cs_asserted = False
            self.open_start = None
        return frames


class MultiPortDecoder:
    """Runs several :class:`SpiPortDecoder` instances over one sample stream."""

    def __init__(self, ports, samplerate, cpol=0, cpha=0):
        """ports: list of dicts with name/clk/miso/mosi/cs channel indices."""
        self.decoders = [
            SpiPortDecoder(p['name'], p['clk'], p['miso'], p['mosi'], p['cs'],
                           samplerate, cpol=cpol, cpha=cpha)
            for p in ports
        ]

    def feed(self, raw):
        """Feed a chunk (bytes or uint8 array); yield (port_name, frame)."""
        chunk = raw if isinstance(raw, np.ndarray) \
            else np.frombuffer(raw, dtype=np.uint8)
        for dec in self.decoders:
            for frame in dec.feed(chunk):
                yield dec.name, frame

    def end(self):
        for dec in self.decoders:
            for frame in dec.end():
                yield dec.name, frame
