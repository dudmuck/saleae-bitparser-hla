#!/usr/bin/env python3
"""Check W1 raw captures/replays, or SPI --hex output against a JSONL manifest.

Walk/PRBS checks project onto --lanes (default 0xff), collapse consecutive
states cyclically, and anchor on the first complete on-cycle run longer than
ceil(--skew-ns * META_rate / 1e9) samples (default 12 ns). The first run is
clipped and excluded from anchoring. Every expected state must follow in order.
Between A and B, at most popcount(A^B)+1 off-path runs are permitted: X must differ from
A/B and satisfy (A&B) subset X subset (A|B). A&B gaps may have any dwell;
other skew runs may not exceed the sample bound. Gap and skew counts are
reported separately; there is no recovery or silent resynchronization after
a fault. Gray-section A&B gaps remain invalid. The final run is checked with
its observed (possibly clipped) dwell; a valid trailing transient is allowed
but does not count as an expected-state transition.

A full projected cycle is required after the anchor to pass. Identical-state
duplication is indistinguishable from longer dwell, as is whole-period loss.
Transitions entirely inside the skew envelope can be ambiguous, so this is a
bounded waveform consistency check, not a proof against every possible fault.
The expected-transition rate and full-cycle-normalized rate are informational
CPU-generator timebase ratios, not absolute calibration. Walk dwell deviations
are warnings, not failures. check_sequence without a META rate preserves its
strict, sample-index-only interface for legacy callers (zero skew allowance).

No hardware access. SPI raw input is replayed via sigrok_hla.py --dslogic -i;
--decode instead checks saved --hex stdout. Supply one manifest per group, or
one shared manifest when both Pis generated the same frames.
"""
import argparse
import collections
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def cycle_for(pattern, seed=1):
    if pattern == 'walk':
        return [1 << bit for bit in range(8)] + [i ^ (i >> 1) for i in range(256)]
    if not 1 <= seed <= 255:
        raise ValueError('PRBS seed must be 1..255')
    cycle, state = [], seed
    while not cycle or state != seed:
        cycle.append(state)
        feedback = ((state >> 7) ^ (state >> 5) ^ (state >> 4) ^ (state >> 3)) & 1
        state = ((state << 1) | feedback) & 255
        if not state or len(cycle) > 255:
            raise ValueError('invalid PRBS cycle')
    return cycle


def read_states(path, groups, unitsize=None, lanes=0xff):
    side = Path(str(path) + '.json')
    if side.exists():
        record = json.loads(side.read_text())
        if unitsize is not None and unitsize != record['unitsize']:
            raise ValueError('unitsize conflicts with sidecar')
        unitsize = record['unitsize']
        if record.get('exit') != 0:
            raise ValueError('capture record reports a failed or unfinished producer')
        if 'bytes' in record and record['bytes'] != Path(path).stat().st_size:
            raise ValueError('capture size conflicts with sidecar')
    unitsize = unitsize or (2 if 8 in groups else 1)
    if unitsize not in (1, 2) or (8 in groups and unitsize != 2):
        raise ValueError('high group requires two-byte samples')
    states = {g: [] for g in groups}
    indices = {g: [] for g in groups}
    previous = {g: None for g in groups}
    total = 0
    with open(path, 'rb') as f:
        match = re.fullmatch(rb'META samplerate: ([1-9][0-9]*)\r?\n', f.readline(128))
        if not match:
            raise ValueError('missing or invalid META samplerate')
        rate = int(match[1])
        if f.peek(14).startswith(b'META trigger: '):
            if not re.fullmatch(rb'META trigger: (none|[0-9]+)\r?\n', f.readline(128)):
                raise ValueError('invalid META trigger')
        carry = b''
        while chunk := f.read(1 << 20):
            data = carry + chunk
            end = len(data) // unitsize * unitsize
            carry = data[end:]
            samples = np.frombuffer(data[:end], dtype='<u2' if unitsize == 2 else 'u1')
            for group in groups:
                values = ((samples >> group) & lanes).astype(np.uint8)
                if not len(values):
                    continue
                change = np.empty(len(values), dtype=bool)
                change[0] = previous[group] is None or values[0] != previous[group]
                change[1:] = values[1:] != values[:-1]
                where = np.flatnonzero(change)
                states[group].append(values[where])
                indices[group].append(where.astype(np.uint64) + total)
                previous[group] = int(values[-1])
            total += len(samples)
        if carry:
            raise ValueError('truncated two-byte sample')
    return ({g: (np.concatenate(states[g]) if states[g] else np.array([], dtype=np.uint8),
                  np.concatenate(indices[g]) if indices[g] else np.array([], dtype=np.uint64)) for g in groups}, rate, total)


def skew_samples(skew_ns, rate):
    # Decimal CLI input and integer META rate: avoid ceil(3.0000000000000004)
    # accidentally granting an extra sample at an exact sample boundary.
    samples = Fraction(str(skew_ns)) * Fraction(str(rate)) / 1_000_000_000
    return (samples.numerator + samples.denominator - 1) // samples.denominator


def projected_cycle(pattern, seed=1, lanes=0xff):
    """Cyclic repeat collapse, with original transition identity for Gray control."""
    full = cycle_for(pattern, seed)
    values = [v & lanes for v in full]
    positions = [i for i, value in enumerate(values) if value != values[i - 1]]
    return [values[i] for i in positions], [pattern == 'walk' and i >= 9 for i in positions]


def _trace(values, dwells, cycle, gray, anchor, offset, skew_bound, collect=False):
    position, logical, skew, gaps, gray_skew = offset, 1, 0, 0, 0
    pending = max_pending = max_skew = 0
    accepted = [anchor] if collect else None
    for k in range(anchor + 1, len(values)):
        a, b, value = cycle[position], cycle[(position + 1) % len(cycle)], int(values[k])
        if value == b:
            position = (position + 1) % len(cycle)
            logical += 1
            pending = 0
            if collect:
                accepted.append(k)
            continue
        lo, hi = a & b, a | b
        if value == a:
            reason = 'repeated expected state'
        elif value & ~hi or lo & ~value:
            reason = 'out-of-range transient or missing/reordered expected state'
        elif value != lo and int(dwells[k]) > skew_bound:
            reason = f'skew dwell {int(dwells[k])} exceeds bound {skew_bound}'
        elif pending >= (a ^ b).bit_count() + 1:
            reason = f'lost sync: more than {(a ^ b).bit_count() + 1} off-path runs before expected state'
        else:
            pending += 1
            max_pending = max(max_pending, pending)
            if value == lo:
                gaps += 1
            else:
                skew += 1
                max_skew = max(max_skew, int(dwells[k]))
                gray_skew += bool(gray[(position + 1) % len(cycle)])
            continue
        return dict(prefix=k, position=(position + 1) % len(cycle), logical_states=logical,
                    skew=skew, intermediates=gaps, gray_intermediates=0, gray_skew=gray_skew,
                    reason=reason, accepted=accepted, anchor=anchor, max_pending=max_pending, max_skew=max_skew)
    return dict(prefix=len(values), position=(position + 1) % len(cycle), logical_states=logical,
                skew=skew, intermediates=gaps, gray_intermediates=0, gray_skew=gray_skew,
                reason=None, accepted=accepted, anchor=anchor, max_pending=max_pending, max_skew=max_skew)


def wiring_fault(values, cycle):
    # Exact state-only wiring diagnosis remains useful after a path failure.
    # It never accepts the file or resynchronizes the primary checker.
    def matches(transformed):
        distinct = [v for i, v in enumerate(transformed) if v != transformed[i - 1]]
        if len(distinct) < 2 or len(values) < len(distinct):
            return False
        for offset, state in enumerate(distinct):
            if state == values[0]:
                expected = np.resize(np.array(distinct[offset:] + distinct[:offset]), len(values))
                if np.array_equal(values, expected):
                    return True
        return False
    for bit in range(8):
        for level in (0, 1):
            if matches([(v & ~(1 << bit)) | (level << bit) for v in cycle]):
                return f'stuck lane {bit} at {level}'
    for first in range(8):
        for second in range(first + 1, 8):
            if matches([v ^ (((v >> first ^ v >> second) & 1) * ((1 << first) | (1 << second))) for v in cycle]):
                return f'swapped lanes {first}/{second}'
    return None


def check_sequence(values, indices, pattern, seed=1, statistics=None, *, rate=None,
                   total_samples=None, skew_ns=12, lanes=0xff):
    if not isinstance(lanes, int) or not 1 <= lanes <= 255:
        raise ValueError('lanes must be a nonempty 8-bit mask (1..0xff)')
    if not math.isfinite(skew_ns) or skew_ns < 0:
        raise ValueError('skew-ns must be finite and nonnegative')
    if rate is not None and (rate <= 0 or not math.isfinite(rate)):
        raise ValueError('META samplerate must be finite and positive')
    cycle, gray = projected_cycle(pattern, seed, lanes)
    if not len(values):
        return 'empty capture'
    if not cycle:
        return 'lane projection has no transitions'
    values = np.asarray(values) & lanes
    indices = np.asarray(indices, dtype=np.uint64)
    # Direct callers can supply unprojected runs; merge them before computing
    # dwell, just as read_states does when extracting the raw samples.
    change = np.r_[True, values[1:] != values[:-1]]
    values, indices = values[change], indices[change]
    if total_samples is None:
        last_dwell = int(indices[-1] - indices[-2]) if len(indices) > 1 else 1
        total_samples = int(indices[-1]) + last_dwell
    dwells = np.diff(np.r_[indices, np.uint64(total_samples)])
    skew_bound = skew_samples(skew_ns, rate) if rate is not None else 0
    anchor = next((k for k in range(1 if rate is not None else 0, len(values))
                   if int(values[k]) in cycle and int(dwells[k]) > skew_bound), None)
    if statistics is not None:
        statistics.update(skew_bound=skew_bound, projected_length=len(cycle), skew=0,
                          intermediates=0, gray_intermediates=0, gray_skew=0, logical_states=0)
    if anchor is None:
        return 'no stable on-cycle anchor longer than the skew bound'
    candidates = [offset for offset, state in enumerate(cycle) if state == int(values[anchor])]
    # A projected value can occur at several phases. Pick the longest exact
    # path from this single fixed anchor; no candidate can jump past a fault.
    starts = [(anchor, offset) for offset in candidates]
    if rate is None and len(values) > 1:
        # Legacy state-only calls can begin with one observed CLR/SET gap.
        starts.extend((1, offset) for offset, state in enumerate(cycle)
                      if state == values[1] and cycle[offset - 1] & state == values[0]
                      and values[0] not in (cycle[offset - 1], state))
    best = max((_trace(values, dwells, cycle, gray, start, offset, skew_bound, collect=pattern == 'walk')
                for start, offset in starts), key=lambda r: (r['prefix'], r['logical_states']))
    anchor = best['anchor']
    if statistics is not None:
        statistics.update({key: value for key, value in best.items() if key not in ('reason', 'accepted')})
        statistics['anchor_sample'] = int(indices[anchor])
        statistics['transitions'] = best['logical_states'] - 1
        statistics['complete_runs'] = [(int(values[k]), int(indices[k]), int(dwells[k]))
                                       for k in best['accepted'] or [] if k not in (0, len(values) - 1)]
    prefix = best['prefix']
    if prefix == len(values):
        return None if best['logical_states'] >= len(cycle) else 'insufficient coverage: need a full state cycle'
    wiring = wiring_fault(values[anchor:min(len(values), anchor + len(cycle) * 3)], cycle)
    if wiring:
        return f'{wiring}; first mismatch at sample {int(indices[prefix])}'
    position = best['position']
    got, want = int(values[prefix]), cycle[position]
    if prefix + 1 < len(values) and got == cycle[(position + 1) % len(cycle)] and int(values[prefix + 1]) == want:
        kind = 'reordered states'
    elif 'repeated expected' in best['reason'] or (prefix >= 2 and got == int(values[prefix - 2])):
        kind = 'duplicated state block'
    elif got in [cycle[(position + skip) % len(cycle)] for skip in range(1, 17)]:
        kind = 'dropped states'
    elif got in [cycle[(position - back) % len(cycle)] for back in range(1, 17)]:
        kind = 'duplicated state block'
    else:
        kind = 'state mismatch (dropped, duplicated or reordered states)'
    return f'{kind} at sample {int(indices[prefix])}: got 0x{got:02x}, expected 0x{want:02x}; {best["reason"]}'


def check_spi(lines, manifest, bus='SPI'):
    expected = []
    for line in Path(manifest).read_text().splitlines():
        record = json.loads(line)
        if 'frame' in record:
            if record['frame'] != len(expected):
                raise ValueError('manifest frame numbers are not consecutive from zero')
            expected.append((bytes.fromhex(record['mosi']), bytes.fromhex(record['miso'])))
    if not expected:
        return 'empty manifest'
    observed, pending = [], None
    for n, line in enumerate(lines.splitlines(), 1):
        if any(marker in line for marker in ('DECODE ERROR', 'dict-error', 'CMD_FAIL')):
            return f'decode error on line {n}'
        m = re.match(r'\s*(?:\[(\w+)\] )?(MOSI|MISO):\s*(.*)$', line)
        if not m or (m[1] or 'SPI') != bus:
            continue
        data = bytes.fromhex(m[3])
        if m[2] == 'MOSI':
            if pending is not None:
                return f'MOSI without paired MISO at line {n}'
            pending = data
        else:
            if pending is None:
                return f'MISO without paired MOSI at line {n}'
            observed.append((pending, data))
            pending = None
    if pending is not None:
        return 'trailing MOSI without MISO'
    for n, (got, want) in enumerate(zip(observed, expected)):
        if got != want:
            return f'SPI frame {n} mismatch: got {got[0].hex()}/{got[1].hex()}, expected {want[0].hex()}/{want[1].hex()}'
    if len(observed) != len(expected):
        return f'SPI frame count mismatch: got {len(observed)}, expected {len(expected)}'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--pattern', choices=('walk', 'prbs', 'spi'), required=True)
    p.add_argument('--group', type=int, choices=(0, 8), action='append')
    p.add_argument('--unitsize', type=int, choices=(1, 2))
    p.add_argument('--lanes', type=lambda text: int(text, 0), default=0xff,
                   help='nonempty 8-bit mask within each group (default 0xff)')
    p.add_argument('--skew-ns', type=float, default=12,
                   help='maximum non-gap transient duration in ns (default 12)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--step-us', type=float, default=10)
    p.add_argument('--dwell-out', type=Path, help='write per-state walk dwell observations as JSONL')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--decode', action='store_true', help='input is saved sigrok_hla.py --hex stdout')
    p.add_argument('--bus', default='SPI', help='bus label in --decode input')
    p.add_argument('--hla-path', help='HLA directory for replaying SPI raw input')
    a = p.parse_args()
    problems = []
    try:
        groups = sorted(set(a.group or [0]))
        if a.pattern == 'spi':
            if not a.manifest:
                raise ValueError('spi requires --manifest')
            for group in groups:
                if a.decode:
                    lines = a.input.read_text()
                else:
                    if not a.hla_path:
                        raise ValueError('SPI raw replay requires --hla-path (or use --decode)')
                    root = Path(__file__).resolve().parents[2]
                    command = [sys.executable, str(root / 'sigrok_hla.py'), '--dslogic', '-i', str(a.input),
                               '--spi', ','.join(str(group + bit) for bit in range(4)),
                               '--hla-path', a.hla_path, '--hex']
                    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                    if result.returncode:
                        raise ValueError(f'SPI replay failed: {result.stderr[-2000:]}')
                    lines = result.stdout
                error = check_spi(lines, a.manifest, a.bus)
                if error:
                    problems.append(f'group {group}: {error}')
        else:
            if not np.isfinite(a.step_us) or a.step_us < 10:
                raise ValueError('step-us must be finite and at least 10')
            if not 1 <= a.lanes <= 255:
                raise ValueError('lanes must be a nonempty 8-bit mask (1..0xff)')
            if not math.isfinite(a.skew_ns) or a.skew_ns < 0:
                raise ValueError('skew-ns must be finite and nonnegative')
            states, rate, total = read_states(a.input, groups, a.unitsize, a.lanes)
            dwell_records = []
            for group, (values, indices) in states.items():
                statistics = {}
                error = check_sequence(values, indices, a.pattern, a.seed, statistics,
                                       rate=rate, total_samples=total, skew_ns=a.skew_ns, lanes=a.lanes)
                if error:
                    problems.append(f'group {group}: {error}')
                print(f'group {group} lanes=0x{a.lanes:02x}: {len(values)} distinct states, {total} samples, '
                      f'projected cycle {statistics["projected_length"]}, anchor sample {statistics.get("anchor_sample", "none")}')
                print(f'group {group}: skew {statistics["skew"]} (<= {statistics["skew_bound"]} samples), '
                      f'CLR/SET gaps {statistics["intermediates"]} (any dwell)')
                print(f'group {group}: maximum accepted skew dwell {statistics.get("max_skew", 0)} samples; '
                      f'maximum off-path runs per transition {statistics.get("max_pending", 0)}')
                transitions_per_second = statistics.get('transitions', 0) / (total / rate) if total else 0
                full_length = len(cycle_for(a.pattern, a.seed))
                normalized = transitions_per_second * full_length / statistics['projected_length']
                print(f'group {group}: expected transitions per nominal second: {transitions_per_second:.3f}; '
                      f'full-cycle normalized: {normalized:.3f} (informational)')
                if a.pattern == 'walk':
                    print(f'group {group}: Gray-section CLR/SET gaps {statistics["gray_intermediates"]}; '
                          f'skew {statistics["gray_skew"]}')
                if a.pattern == 'walk' and statistics.get('complete_runs'):
                    dwell = np.array([run[2] for run in statistics['complete_runs']])
                    nominal = rate * a.step_us / 1e6
                    outliers = int(np.count_nonzero((dwell < nominal * .5) | (dwell > nominal * 3)))
                    print(f'group {group}: dwell samples {dict(sorted(collections.Counter(map(int, dwell)).items()))}')
                    if a.dwell_out:
                        for state, sample, count in statistics['complete_runs']:
                            dwell_records.append(dict(group=group, sample=sample, state=state, samples=count))
                    if outliers:
                        print(f'warning: {outliers} walk dwells outside 0.5x–3x nominal {nominal:g} samples')
            if a.dwell_out:
                with a.dwell_out.open('x') as output:
                    for record in dwell_records:
                        output.write(json.dumps(record) + '\n')
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        problems.append(str(error))
    for problem in problems:
        print('FAIL:', problem)
    if not problems:
        print('PASS')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
