# SPDX-License-Identifier: GPL-2.0-or-later
# Derived from Wayne Roberts' (C) 2026 libsigrok deglitch transform.
"""Streaming port of libsigrok's deglitch transform for physical sample arrays.

Clock periods and minimum periods are in *samples*, not seconds. Only selected
bits change. Like libsigrok, feed() delays output by ``lookahead`` samples and
end() discards that undecidable suffix rather than emitting invented decisions.
Numba is optional; ``use_numba=False`` selects the same NumPy/Python kernel.
"""
import math
import operator

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

HAVE_NUMBA = njit is not None
# chan_state fields: have_level, level, last_edge[0:2], pend_start, run_start,
# run_level, gap_handled, pulse_count, suppressed, splits, inserted, unresolved.
_LAST0, _LAST1, _PEND, _RUN = 2, 3, 4, 5
_COUNT, _SUPPRESSED, _SPLITS, _INSERTED, _UNRESOLVED = 8, 9, 10, 11, 12


def _repair_numpy(data, state, bits, start, tail, first, min_period,
                  period, frame_pulses, pulse_level, split_min, gap_max):
    """Exact state machine; edits only retained history, never future input.

    The channel-outer loop is equivalent to C's sample-outer loop because each
    state machine reads/writes only its own bit. Local state keeps the optional
    compiled hot loop fast; the same function is the correct NumPy fallback.
    """
    for bit in bits:
        bit = int(bit)
        mask = 1 << bit
        have, level = state[bit, 0], state[bit, 1]
        last0, last1 = state[bit, 2], state[bit, 3]
        pending, run_start = state[bit, 4], state[bit, 5]
        run_level, handled = state[bit, 6], state[bit, 7]
        count = state[bit, 8]
        suppressed, splits, inserted, unresolved = state[bit, 9:13]
        half = period / 2.0
        for i in range(first, len(data)):
            n = start + i - first
            x = (int(data[i]) >> bit) & 1
            if min_period >= 2:
                if not have:
                    level = x
                    have = 1
                elif pending >= 0:
                    if x == level:
                        pending = -1
                        suppressed += 1
                    elif n - pending + 1 >= min_period:
                        for m in range(pending, n + 1):
                            pos = m - tail
                            if x:
                                data[pos] = int(data[pos]) | mask
                            else:
                                data[pos] = int(data[pos]) & ~mask
                        if x:
                            last1 = pending
                        else:
                            last0 = pending
                        level = x
                        pending = -1
                        run_start = -1
                    else:
                        if level:
                            data[i] = int(data[i]) | mask
                        else:
                            data[i] = int(data[i]) & ~mask
                elif x != level:
                    previous_edge = last1 if x else last0
                    if n - previous_edge >= min_period:
                        level = x
                        if x:
                            last1 = n
                        else:
                            last0 = n
                    else:
                        pending = n
                        if level:
                            data[i] = int(data[i]) | mask
                        else:
                            data[i] = int(data[i]) & ~mask
            if period <= 0.0:
                continue
            x = (int(data[i]) >> bit) & 1
            if run_start < 0:
                run_start, run_level, handled = n, x, 0
                continue
            if x == run_level:
                if run_level != pulse_level and not handled and n - run_start > gap_max:
                    handled = 1
                    if frame_pulses:
                        deficit = (frame_pulses - count % frame_pulses) % frame_pulses
                        if deficit:
                            if deficit <= 2:
                                for j in range(1, deficit + 1):
                                    pos = run_start + int((2 * j - 1) * half + 0.5)
                                    # GLib CLAMP: compare high first, including low>high.
                                    if pos > n - 1:
                                        pos = n - 1
                                    elif pos < run_start + 1:
                                        pos = run_start + 1
                                    p = pos - tail
                                    if pulse_level:
                                        data[p] = int(data[p]) | mask
                                    else:
                                        data[p] = int(data[p]) & ~mask
                                inserted += deficit
                            else:
                                unresolved += 1
                    count = 0
                continue
            width = n - run_start
            if run_level == pulse_level:
                added = 0
                if split_min <= width <= gap_max:
                    added = int((width / half - 1.0) / 2.0 + 0.5)
                    if added < 1:
                        added = 0
                    for j in range(1, added + 1):
                        pos = run_start + int((2 * j - 1) * half + 0.5)
                        if pos > n - 2:
                            pos = n - 2
                        elif pos < run_start + 1:
                            pos = run_start + 1
                        p = pos - tail
                        if pulse_level:
                            data[p] = int(data[p]) & ~mask
                        else:
                            data[p] = int(data[p]) | mask
                    splits += added
                count += 1 + added
            else:
                if handled:
                    handled = 0
                elif split_min <= width <= gap_max and frame_pulses:
                    deficit = (frame_pulses - count % frame_pulses) % frame_pulses
                    if deficit:
                        cap = int(width / period + 0.5)
                        if cap < 1:
                            cap = 1
                        added = min(deficit, cap)
                        for j in range(1, added + 1):
                            pos = run_start + int((2 * j - 1) * half + 0.5)
                            if pos > n - 2:
                                pos = n - 2
                            elif pos < run_start + 1:
                                pos = run_start + 1
                            p = pos - tail
                            if pulse_level:
                                data[p] = int(data[p]) | mask
                            else:
                                data[p] = int(data[p]) & ~mask
                        inserted += added
                        count += added
            run_start, run_level = n, x
        state[bit, 0], state[bit, 1] = have, level
        state[bit, 2], state[bit, 3] = last0, last1
        state[bit, 4], state[bit, 5] = pending, run_start
        state[bit, 6], state[bit, 7] = run_level, handled
        state[bit, 8] = count
        state[bit, 9], state[bit, 10] = suppressed, splits
        state[bit, 11], state[bit, 12] = inserted, unresolved


# No cache files outside the two assigned source files are needed.
_repair_numba = njit(_repair_numpy) if HAVE_NUMBA else None


def _nonnegative_integer(value, name):
    value = 0 if value is None else operator.index(value)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


class Deglitcher:
    """Repair selected physical clock bits across arbitrary sample chunks.

    ``bits`` is an index or iterable of indices (0..63), never a bit mask.
    None selects all bits of dtype; [] selects none. Like C, selected indices
    beyond the packet width are ignored, but still enable the lookahead delay.
    dtype must be native uint8 or uint16. Input arrays are not modified.
    samplerate is metadata; clock_period/min_period are measured in samples.
    ``stats`` exposes C's per-channel suppressed/split/inserted/unresolved counts.
    end() returns an empty array and drops the final lookahead, exactly like C.
    """
    def __init__(self, bits, samplerate, clock_period=None, frame_pulses=None,
                 min_period=None, pulse_level=1, dtype=np.uint8, *, use_numba=None):
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)) or not self.dtype.isnative:
            raise ValueError("dtype must be native uint8 or uint16")
        self.samplerate = float(samplerate)
        if not math.isfinite(self.samplerate) or self.samplerate <= 0:
            raise ValueError("samplerate must be finite and positive")
        self.clock_period = 0.0 if clock_period is None else float(clock_period)
        if not math.isfinite(self.clock_period) or self.clock_period < 0 or 0 < self.clock_period < 2:
            raise ValueError("clock_period must be zero or finite and >= 2 samples")
        self.frame_pulses = _nonnegative_integer(frame_pulses, "frame_pulses")
        self.min_period = _nonnegative_integer(min_period, "min_period")
        self.pulse_level = int(bool(_nonnegative_integer(pulse_level, "pulse_level")))
        if bits is None:
            bits = range(self.dtype.itemsize * 8)
        else:
            try:
                bits = [operator.index(bits)]
            except TypeError:
                pass
        self.bits = tuple(sorted(set(operator.index(bit) for bit in bits)))
        if any(bit < 0 or bit >= 64 for bit in self.bits):
            raise ValueError("bit indices must be in 0..63")
        self._active_bits = np.array([bit for bit in self.bits if bit < self.dtype.itemsize * 8], dtype=np.int64)
        self.split_min = math.ceil(self.clock_period / 2.0) + 1 if self.clock_period else 0
        self.gap_max = math.floor(4.0 * self.clock_period) if self.clock_period else 0
        if self.gap_max > np.iinfo(np.int64).max - 4:
            raise ValueError("clock_period is too large for signed-64-bit sample indices")
        self.lookahead = max(self.min_period, self.gap_max + 4 if self.clock_period else 0)
        self._state = np.zeros((64, 13), dtype=np.int64)
        self._state[:, 2:4] = -(1 << 62)  # INT64_MIN / 2, as in C.
        self._state[:, 4:6] = -1
        self._carry = np.empty(0, dtype=self.dtype)
        self._tail = self.samples_in = self.samples_out = 0
        self._closed = False
        if use_numba and not HAVE_NUMBA:
            raise RuntimeError("numba acceleration requested but numba is unavailable")
        self.backend = "numba" if HAVE_NUMBA and use_numba is not False else "numpy"
        self._kernel = _repair_numba if self.backend == "numba" else _repair_numpy

    def feed(self, chunk):
        if self._closed:
            raise ValueError("cannot feed after end()")
        chunk = np.asarray(chunk)
        if chunk.ndim != 1 or chunk.dtype != self.dtype:
            raise ValueError(f"chunk must be a one-dimensional {self.dtype} array")
        if not chunk.size:
            return np.empty(0, dtype=self.dtype)
        if not self.lookahead or not self.bits:
            self.samples_in += chunk.size
            self.samples_out += chunk.size
            return chunk.copy()
        first = self._carry.size
        data = np.concatenate((self._carry, chunk))
        self._kernel(data, self._state, self._active_bits, self.samples_in, self._tail,
                     first, self.min_period, self.clock_period, self.frame_pulses,
                     self.pulse_level, self.split_min, self.gap_max)
        self.samples_in += chunk.size
        emit = max(0, data.size - self.lookahead)
        self._carry = data[emit:].copy()
        self._tail += emit
        self.samples_out += emit
        # data owns its buffer, so future feed calls cannot mutate this output.
        return data[:emit]

    def end(self):
        """Discard undecidable tail; do not emit it (libsigrok SR_DF_END)."""
        self._carry = np.empty(0, dtype=self.dtype)
        self._closed = True
        return np.empty(0, dtype=self.dtype)

    @property
    def stats(self):
        return {bit: {name: int(self._state[bit, column]) for name, column in
                      (("suppressed", 9), ("split", 10), ("inserted", 11), ("unresolved", 12))}
                for bit in self.bits}

    @property
    def inserted(self):
        return sum(s["inserted"] for s in self.stats.values())

    @property
    def splits(self):
        return sum(s["split"] for s in self.stats.values())

    @property
    def suppressed(self):
        return sum(s["suppressed"] for s in self.stats.values())

    @property
    def unresolved(self):
        return sum(s["unresolved"] for s in self.stats.values())
