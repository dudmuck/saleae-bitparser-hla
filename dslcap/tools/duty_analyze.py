#!/usr/bin/env python3
"""SCLK high/low phase widths inside nSS frames of a dslcap raw capture.

usage: duty_analyze.py RAW --clk BIT --cs BIT [--width 1|2] [--json OUT]
Reads META lines + samples. The sample width comes from RAW.json (a
--raw-out record) or --width; it is never guessed from the selected bits,
since a 12-channel capture is uint16 even when CH0-3 are analyzed. For each
nSS-low frame, measures every SCLK high run and every within-byte low run (low
runs longer than 0.75 of a nominal period are inter-byte gaps or idle and are
excluded). Widths are on the sample grid, so each carries up to one sample
interval of uncertainty. Reports widths in ns, the distribution, and the margin
against one sample at 25 MS/s (40 ns): a phase at or below 40 ns can vanish at
25 MS/s. The whole file is loaded into memory: meant for short buffer captures,
not multi-GB stream files.
"""
import argparse, json, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('raw'); ap.add_argument('--clk', type=int, required=True)
ap.add_argument('--cs', type=int, required=True); ap.add_argument('--sclk', type=float, default=10e6)
ap.add_argument('--width', type=int, choices=(1, 2)); ap.add_argument('--json')
a = ap.parse_args()
width = a.width
side = a.raw + '.json'
try:
    recorded = json.load(open(side)).get('unitsize')
except FileNotFoundError:
    recorded = None
if recorded is not None:
    if width is not None and width != recorded:
        sys.exit(f'--width {width} contradicts {side} unitsize {recorded}')
    width = recorded
if width not in (1, 2):
    sys.exit('sample width unknown: pass --width 1|2 (no RAW.json record)')
if max(a.clk, a.cs) >= 8 * width:
    sys.exit(f'bits {a.clk}/{a.cs} do not fit {width}-byte samples')
data = open(a.raw, 'rb').read()
pos, rate, trig = 0, None, None
while data[pos:pos + 5] == b'META ':
    end = data.index(b'\n', pos)
    line = data[pos:end].decode()
    if line.startswith('META samplerate:'):
        rate = int(line.split()[-1])
    elif line.startswith('META trigger:'):
        trig = line.split()[-1]
    pos = end + 1
if (len(data) - pos) % width:
    sys.exit('partial sample at end of file')
s = np.frombuffer(data[pos:], dtype='<u2' if width == 2 else np.uint8)
clk = ((s >> a.clk) & 1).astype(np.int8)
cs = ((s >> a.cs) & 1).astype(np.int8)
ns = 1e9 / rate
period = rate / a.sclk
# nSS frames
d = np.diff(cs)
falls, rises = np.flatnonzero(d == -1) + 1, np.flatnonzero(d == 1) + 1
frames = []
for f in falls:
    r = rises[rises > f]
    if r.size:
        frames.append((int(f), int(r[0])))
highs, lows, clocks = [], [], []
for f, r in frames:
    c = clk[f:r]
    e = np.flatnonzero(np.diff(c)) + 1
    if e.size < 2:
        clocks.append(0)
        continue
    runs = np.diff(e)
    levels = c[e[:-1]]
    hi = runs[levels == 1]
    lo = runs[(levels == 0) & (runs <= 0.75 * period)]   # longer lows are byte gaps
    highs.extend(hi.tolist()); lows.extend(lo.tolist())
    clocks.append(int(np.count_nonzero(np.diff(c) == 1)))
highs, lows = np.array(highs) * ns, np.array(lows) * ns
def summary(x):
    if not x.size:
        return {}
    vals, counts = np.unique(np.round(x, 2), return_counts=True)
    return dict(n=int(x.size), min_ns=float(x.min()), mean_ns=float(x.mean()), max_ns=float(x.max()),
                p001_ns=float(np.percentile(x, 0.1)), at_or_below_40ns=int((x <= 40.0).sum()),
                histogram_ns={f'{v:g}': int(c) for v, c in zip(vals, counts)})
clock_counts = {}
for c in clocks:
    clock_counts[c] = clock_counts.get(c, 0) + 1
res = dict(file=a.raw, samplerate=rate, trigger=trig, samples=int(s.size), resolution_ns=ns,
           frames=len(frames), clocks_per_frame=dict(sorted(clock_counts.items())),
           high=summary(highs), low_within_byte=summary(lows),
           duty_mean=float(highs.mean() / (highs.mean() + lows.mean())) if highs.size and lows.size else None)
for k in ('high', 'low_within_byte'):
    v = res[k]
    if v:
        print(f"{k:16s} n={v['n']:7d} min={v['min_ns']:.1f} mean={v['mean_ns']:.2f} max={v['max_ns']:.1f} ns  "
              f"<=40ns: {v['at_or_below_40ns']}  margin over 40 ns: {v['min_ns'] - 40:.1f} ns")
print(f"rate {rate} ({ns:g} ns/sample), trigger {trig}, frames {len(frames)}, "
      f"duty {res['duty_mean']}, clocks/frame {res['clocks_per_frame']}")
if a.json:
    json.dump(res, open(a.json, 'w'), indent=1)
