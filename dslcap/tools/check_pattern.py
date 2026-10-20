#!/usr/bin/env python3
"""Check W1 raw captures/replays, or SPI --hex output against a JSONL manifest.

State checks align anywhere in the walk/PRBS cycle and ignore repeated samples
of a state. GPCLR0-before-GPSET0 permits exactly one extra A&B observation
between A and B, only when distinct from both. The generator emits no
consecutive equal states. Repeated/dropped
USB blocks containing multiple transitions are detectable; an identical-state
duplicate is indistinguishable from longer dwell, as is loss of an entire
period of a periodic pattern. A capture must contain at least one full cycle
of transitions to pass. Walk dwell deviations are warnings, not failures.

No hardware access. SPI raw input is replayed via sigrok_hla.py --dslogic -i;
--decode instead checks saved --hex stdout. Supply one manifest per group, or
one shared manifest when both Pis generated the same frames.
"""
import argparse
import collections
import json
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


def read_states(path, groups, unitsize=None):
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
                values = ((samples >> group) & 255).astype(np.uint8)
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


def alignment(values, cycle):
    """Longest exact path, optionally seeing one A&B state on A→B.

    Return observation index, next cycle position, logical-state count and
    intermediate count. Initial capture may begin on the intermediate itself.
    """
    best = (0, 0, 0, 0)
    length = len(cycle)
    starts = [(offset, 1, 0) for offset, state in enumerate(cycle) if state == values[0]]
    if len(values) > 1:
        for offset, state in enumerate(cycle):
            previous = cycle[(offset - 1) % length]
            if previous & state not in (previous, state) and values[0] == previous & state and values[1] == state:
                starts.append((offset, 2, 1))
    for offset, observed, intermediate in starts:
        position, logical = (offset + 1) % length, 1
        while observed < len(values):
            previous, wanted = cycle[(position - 1) % length], cycle[position]
            got = int(values[observed])
            if got == wanted:
                observed += 1
            elif (previous & wanted not in (previous, wanted) and got == previous & wanted and
                  observed + 1 < len(values) and int(values[observed + 1]) == wanted):
                intermediate += 1
                observed += 2
            else:
                break
            logical += 1
            position = (position + 1) % length
        if observed > best[0]:
            best = observed, position, logical, intermediate
    return best


def wiring_fault(values, cycle):
    # A transformed cycle can collapse transitions. Test only its distinct
    # states, and require a complete observed cycle to diagnose wiring.
    def matches(transformed):
        distinct = [v for i, v in enumerate(transformed) if not i or v != transformed[i - 1]]
        if len(distinct) > 1 and distinct[-1] == distinct[0]:
            distinct.pop()
        return (len(distinct) > 1 and len(values) >= len(distinct) and
                alignment(values, distinct)[0] == len(values))
    for bit in range(8):
        for level in (0, 1):
            if matches([(v & ~(1 << bit)) | (level << bit) for v in cycle]):
                return f'stuck lane {bit} at {level}'
    for first in range(8):
        for second in range(first + 1, 8):
            transformed = [v ^ (((v >> first ^ v >> second) & 1) * ((1 << first) | (1 << second))) for v in cycle]
            if matches(transformed):
                return f'swapped lanes {first}/{second}'
    return None


def check_sequence(values, indices, pattern, seed=1, statistics=None):
    cycle = cycle_for(pattern, seed)
    if not len(values):
        return 'empty capture'
    prefix, position, logical, intermediates = alignment(values, cycle)
    if statistics is not None:
        statistics.update(intermediates=intermediates, logical_states=logical, gray_intermediates=0)
    if prefix == len(values):
        return None if logical >= len(cycle) else 'insufficient coverage: need a full state cycle'
    wiring = wiring_fault(values[:min(len(values), len(cycle) * 3)], cycle)
    if wiring:
        return f'{wiring}; first mismatch at sample {int(indices[prefix])}'
    got, want = int(values[prefix]), cycle[position]
    if (prefix + 1 < len(values) and got == cycle[(position + 1) % len(cycle)] and
            int(values[prefix + 1]) == want):
        kind = 'reordered states'
    elif prefix >= 2 and got == int(values[prefix - 2]):
        kind = 'duplicated state block'
    elif got in [cycle[(position + skip) % len(cycle)] for skip in range(1, 17)]:
        kind = 'dropped states'
    elif got in [cycle[(position - back) % len(cycle)] for back in range(1, 17)]:
        kind = 'duplicated state block'
    else:
        kind = 'state mismatch (dropped, duplicated or reordered states)'
    return f'{kind} at sample {int(indices[prefix])}: got 0x{got:02x}, expected 0x{want:02x}'


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
            states, rate, total = read_states(a.input, groups, a.unitsize)
            dwell_records = []
            for group, (values, indices) in states.items():
                statistics = {}
                error = check_sequence(values, indices, a.pattern, a.seed, statistics)
                if error:
                    problems.append(f'group {group}: {error}')
                print(f'group {group}: {len(values)} distinct states, {total} samples')
                print(f'group {group}: {statistics.get("intermediates", 0)} GPIO clear/set intermediates')
                if a.pattern == 'walk':
                    print(f'group {group}: 0 Gray-section intermediates (single-bit transitions)')
                if a.pattern == 'walk' and len(indices) > 2:
                    dwell = np.diff(indices)[1:]  # first/last may be clipped by capture.
                    nominal = rate * a.step_us / 1e6
                    outliers = int(np.count_nonzero((dwell < nominal * .5) | (dwell > nominal * 3)))
                    print(f'group {group}: dwell samples {dict(sorted(collections.Counter(map(int, dwell)).items()))}')
                    if a.dwell_out:
                        for state, sample, count in zip(values[1:-1], indices[1:-1], dwell):
                            dwell_records.append(dict(group=group, sample=int(sample), state=int(state), samples=int(count)))
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
