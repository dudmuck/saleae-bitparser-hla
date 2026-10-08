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


if HAVE_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _scan_port(chunk, clk, miso, mosi, cs, prev, has_prev, rising):
        """One pass over packed samples: CS edges, and every sampling edge of
        CLK with the MOSI/MISO bits latched there. Matches find_edges: with
        no history the first sample is never an edge."""
        n = chunk.size
        cs_edges = np.empty(n, dtype=np.int64)
        idx = np.empty(n, dtype=np.int64)
        mo = np.empty(n, dtype=np.uint8)
        mi = np.empty(n, dtype=np.uint8)
        ncs = 0
        ne = 0
        want = 1 if rising else 0
        pclk = (prev >> clk) & 1
        pcs = (prev >> cs) & 1
        start = 0
        if not has_prev:
            pclk = (chunk[0] >> clk) & 1
            pcs = (chunk[0] >> cs) & 1
            start = 1
        for t in range(start, n):
            v = chunk[t]
            c = (v >> cs) & 1
            if c != pcs:
                cs_edges[ncs] = t
                ncs += 1
                pcs = c
            k = (v >> clk) & 1
            if k != pclk:
                if k == want:
                    idx[ne] = t
                    mo[ne] = (v >> mosi) & 1
                    mi[ne] = (v >> miso) & 1
                    ne += 1
                pclk = k
        # Compact copies: a transaction held open across chunks keeps its
        # latched slices, which must not pin chunk-sized scratch arrays.
        return cs_edges[:ncs].copy(), idx[:ne].copy(), mo[:ne].copy(), mi[:ne].copy()


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
        if chunk.size == 0:
            return []
        if HAVE_NUMBA:
            return self._feed_scanned(chunk)
        return self._feed_numpy(chunk)

    def _latch(self, idx, mo, mi, lo, hi, base):
        """Buffer the sampling edges with chunk index in [lo, hi)."""
        a, b = np.searchsorted(idx, (lo, hi))
        if b <= a:
            return
        if self.open_bit_count >= MAX_OPEN_BITS:
            if not self.overflowed:
                print(f"[fast_spi] {self.name}: transaction exceeded "
                      f"{MAX_OPEN_BITS} bits, truncating", file=sys.stderr)
                self.overflowed = True
            return
        self.mosi_bits.append(mo[a:b])
        self.miso_bits.append(mi[a:b])
        self.bit_times.append((idx[a:b] + base).astype(np.float64))
        self.open_bit_count += b - a

    def _feed_scanned(self, chunk):
        frames = []
        prev = self.prev_sample
        cs_edges, idx, mo, mi = _scan_port(chunk, self.clk, self.miso, self.mosi, self.cs,
                                           0 if prev is None else prev, prev is not None,
                                           self.sample_on_rising)
        active = 0 if self.cs_active_low else 1
        base = self.abs_pos
        pos = 0
        for e in cs_edges.tolist():
            now_asserted = ((int(chunk[e]) >> self.cs) & 1) == active
            if now_asserted and not self.cs_asserted:
                self.cs_asserted = True
                self.open_start = base + e
                self.mosi_bits = []
                self.miso_bits = []
                self.bit_times = []
                self.open_bit_count = 0
            elif not now_asserted and self.cs_asserted:
                self._latch(idx, mo, mi, pos, e, base)
                self._emit_transaction(self.open_start, base + e, frames)
                self.cs_asserted = False
                self.open_start = None
            pos = e
        if self.cs_asserted:
            self._latch(idx, mo, mi, pos, chunk.size, base)
        self.abs_pos += chunk.size
        self.prev_sample = int(chunk[-1])
        return frames

    def _feed_numpy(self, chunk):
        frames = []

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
        """Feed a chunk (bytes or sample array); yield (port_name, frames).

        One item per port, in port order, holding that port's frames for
        the chunk in time order. Ports are independent, so a consumer can
        hand each list to its own HLA without merging them frame by frame.
        """
        chunk = raw if isinstance(raw, np.ndarray) \
            else np.frombuffer(raw, dtype=np.uint8)
        for dec in self.decoders:
            yield dec.name, dec.feed(chunk)

    def end(self):
        """Flush open transactions; yield (port_name, frames) per port."""
        for dec in self.decoders:
            yield dec.name, dec.end()
