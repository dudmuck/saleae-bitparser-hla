#!/usr/bin/env python3
"""Measure a clock in a META-prefixed dslcap raw file, without hardware access.

Samples are processed in NumPy chunks; only histograms and per-window edge
summaries survive a chunk. Frequency uses first-to-last rising-edge spacing.
Duty and high-time histograms cover complete rising-to-rising periods only.
Glitches use complete high/low runs; clipped first/last runs are excluded.
Long (>1.5 nominal periods) and short (<0.5 nominal) periods flag missing or
extra pulses for inspection. Grid quantization limits short-window precision.
Reports are informational unless --tolerance-ppm is supplied, which checks
the overall frequency error. Glitches and period anomalies are always reported
but do not alone change the exit status. Malformed input or fewer than two
rising edges fails. Nominal source accuracy remains the caller's responsibility.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

CHUNK_BYTES = 1 << 20


def add_histogram(histogram, values):
    if len(values):
        bins, counts = np.unique(values, return_counts=True)
        histogram.update({int(value): int(count) for value, count in zip(bins, counts)})


def measure(path, unitsize, channel, expect_hz, min_run=None, window_ms=1):
    if unitsize not in (1, 2) or not 0 <= channel < unitsize * 8:
        raise ValueError('channel does not fit the sample width')
    if not math.isfinite(expect_hz) or expect_hz <= 0:
        raise ValueError('expect-hz must be finite and positive')
    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError('window-ms must be finite and positive')
    if min_run is not None and (not math.isfinite(min_run) or min_run <= 0):
        raise ValueError('min-run must be finite and positive')
    period_hist, high_hist, low_hist = collections.Counter(), collections.Counter(), collections.Counter()
    window_edges = {}
    total = rises_count = high_sum = periods_sum = 0
    first_rise = last_rise = previous_bit = None
    run_start = 0
    pending_high = None
    glitches = dict(count=0, high=0, low=0, first_sample=None)
    anomalies = dict(long_periods=0, short_periods=0, first_sample=None)
    with Path(path).open('rb') as f:
        match = re.fullmatch(rb'META samplerate: ([1-9][0-9]*)\r?\n', f.readline(128))
        if not match:
            raise ValueError('missing or malformed META samplerate')
        rate = int(match[1])
        if rate > 2**64 - 1:
            raise ValueError('META samplerate exceeds producer limit')
        if f.peek(14).startswith(b'META trigger: '):
            if not re.fullmatch(rb'META trigger: (none|[0-9]+)\r?\n', f.readline(128)):
                raise ValueError('malformed META trigger')
        min_run = rate / expect_hz / 4 if min_run is None else min_run
        nominal_period = rate / expect_hz
        window_samples = max(1, round(rate * window_ms / 1000))
        carry = b''
        while chunk := f.read(CHUNK_BYTES):
            data = carry + chunk
            end = len(data) // unitsize * unitsize
            carry = data[end:]
            samples = np.frombuffer(data[:end], dtype='<u2' if unitsize == 2 else 'u1')
            bits = ((samples >> channel) & 1).astype(np.uint8)
            if not len(bits):
                continue
            change = np.empty(len(bits), dtype=bool)
            change[0] = previous_bit is not None and bits[0] != previous_bit
            change[1:] = bits[1:] != bits[:-1]
            where = np.flatnonzero(change)
            edges = where.astype(np.uint64) + total
            new_levels = bits[where]
            if len(edges):
                old_starts = np.r_[np.uint64(run_start), edges[:-1]]
                widths = edges - old_starts
                complete = old_starts != 0  # initial run may be clipped.
                bad = np.flatnonzero(complete & (widths < min_run))
                if len(bad):
                    glitches['count'] += len(bad)
                    glitches['high'] += int(np.count_nonzero(new_levels[bad] == 0))
                    glitches['low'] += int(np.count_nonzero(new_levels[bad] == 1))
                    if glitches['first_sample'] is None:
                        glitches['first_sample'] = int(old_starts[bad[0]])
                rising = edges[new_levels == 1]
                high_widths = widths[complete & (new_levels == 0)]
                queued_highs = (np.r_[np.uint64(pending_high), high_widths]
                                if pending_high is not None else high_widths)
                rise_path = np.r_[np.uint64(last_rise), rising] if last_rise is not None else rising
                periods = np.diff(rise_path)
                if len(queued_highs) < len(periods) or len(queued_highs) > len(periods) + 1:
                    raise ValueError('inconsistent clock edge pairing')
                highs = queued_highs[:len(periods)]
                pending_high = int(queued_highs[-1]) if len(queued_highs) > len(periods) else None
                add_histogram(period_hist, periods)
                add_histogram(high_hist, highs)
                add_histogram(low_hist, periods - highs)
                high_sum += int(highs.sum())
                periods_sum += int(periods.sum())
                long = periods > 1.5 * nominal_period
                short = periods < .5 * nominal_period
                anomalies['long_periods'] += int(np.count_nonzero(long))
                anomalies['short_periods'] += int(np.count_nonzero(short))
                bad_periods = np.flatnonzero(long | short)
                if len(bad_periods) and anomalies['first_sample'] is None:
                    anomalies['first_sample'] = int(rise_path[bad_periods[0]])
                if len(rising):
                    if first_rise is None:
                        first_rise = int(rising[0])
                    last_rise = int(rising[-1])
                    rises_count += len(rising)
                    bins = rising // window_samples
                    starts = np.r_[0, np.flatnonzero(bins[1:] != bins[:-1]) + 1]
                    ends = np.r_[starts[1:], len(rising)]
                    for start, stop in zip(starts, ends):
                        key = int(bins[start])
                        if key in window_edges:
                            window_edges[key][1] = int(rising[stop - 1])
                            window_edges[key][2] += int(stop - start)
                        else:
                            window_edges[key] = [int(rising[start]), int(rising[stop - 1]), int(stop - start)]
                run_start = int(edges[-1])
            previous_bit = int(bits[-1])
            total += len(bits)
        if carry:
            raise ValueError('truncated two-byte sample')
    if rises_count < 2:
        raise ValueError('fewer than two rising edges; frequency cannot be measured')
    frequency = (rises_count - 1) * rate / (last_rise - first_rise)
    window_hz = np.array([(count - 1) * rate / (last - first)
                          for first, last, count in window_edges.values() if count >= 2])
    window_ppm = (window_hz / expect_hz - 1) * 1e6
    return dict(file=str(path), samplerate=rate, unitsize=unitsize, channel=channel, samples=total,
                expect_hz=expect_hz, rising_edges=rises_count, first_rise=first_rise, last_rise=last_rise,
                frequency_hz=frequency, error_ppm=(frequency / expect_hz - 1) * 1e6,
                window=dict(samples=window_samples, count=len(window_ppm),
                            min_ppm=float(window_ppm.min()) if len(window_ppm) else None,
                            max_ppm=float(window_ppm.max()) if len(window_ppm) else None,
                            std_ppm=float(window_ppm.std()) if len(window_ppm) else None),
                complete_periods=rises_count - 1, duty=high_sum / periods_sum,
                period_histogram=dict(sorted(period_hist.items())),
                high_time_histogram=dict(sorted(high_hist.items())),
                low_time_histogram=dict(sorted(low_hist.items())),
                min_run_samples=min_run, glitches=glitches, period_anomalies=anomalies)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', type=Path)
    p.add_argument('--unitsize', type=int, choices=(1, 2), required=True)
    p.add_argument('--channel', type=int, required=True)
    p.add_argument('--expect-hz', type=float, required=True)
    p.add_argument('--window-ms', type=float, default=1)
    p.add_argument('--min-run', type=float)
    p.add_argument('--tolerance-ppm', type=float)
    a = p.parse_args()
    try:
        if a.tolerance_ppm is not None and (not math.isfinite(a.tolerance_ppm) or a.tolerance_ppm < 0):
            raise ValueError('tolerance-ppm must be finite and nonnegative')
        result = measure(a.raw, a.unitsize, a.channel, a.expect_hz, a.min_run, a.window_ms)
        within = abs(result['error_ppm']) <= a.tolerance_ppm if a.tolerance_ppm is not None else None
        result.update(tolerance_ppm=a.tolerance_ppm, within_tolerance=within)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if within is False else 0
    except (OSError, ValueError, OverflowError) as error:
        print(f'check_clock: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
