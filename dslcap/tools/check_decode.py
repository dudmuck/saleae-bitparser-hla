#!/usr/bin/env python3
"""Compare sigrok_hla.py --hex output with gen_pp.py's expected transfers.

usage: check_decode.py DECODE_STDOUT EXPECTED_JSON
Pairs each '[SPI] MOSI:'/'MISO:' (bus A) and '[SPI_B] ...' (bus B) hex block
in output order and requires a byte-exact match per bus, in order. Fails on
unpaired blocks, short (0/1-byte) transfers, and CMD_FAIL, dict-error or
DECODE ERROR lines.
"""
import json, re, sys

out = open(sys.argv[1]).read().splitlines()
exp = json.load(open(sys.argv[2]))
got = {'A': [], 'B': []}
pending = {}
structure = []
for n, line in enumerate(out, 1):
    m = re.match(r'\s*\[(SPI|SPI_B)\] (MOSI|MISO): ?(.*)$', line)
    if not m:
        continue
    bus = 'A' if m[1] == 'SPI' else 'B'
    data = m[3].replace(' ', '')
    if m[2] == 'MOSI':
        if bus in pending:
            structure.append(f'line {n}: MOSI without MISO before it')
        pending[bus] = data
    elif bus not in pending:
        structure.append(f'line {n}: MISO without MOSI')
    else:
        got[bus].append((pending.pop(bus), data))
structure += [f'bus {bus}: MOSI without MISO at end of output' for bus in pending]
want = {'A': [(e['mosi'], e['miso']) for e in exp if e['bus'] == 'A'],
        'B': [(e['mosi'], e['miso']) for e in exp if e['bus'] == 'B']}
bad = 0
for bus in 'AB':
    g, w = got[bus], want[bus]
    mism = [i for i in range(min(len(g), len(w))) if g[i] != w[i]]
    print(f'bus {bus}: expected {len(w)} transfers, decoded {len(g)}, mismatched {len(mism)}'
          + (f' (first at #{mism[0]})' if mism else ''))
    bad += len(mism) + abs(len(g) - len(w))
short = sum(1 for bus in 'AB' for mo, mi in got[bus] if len(mo) <= 2)
fails = sum(1 for line in out if 'CMD_FAIL' in line or 'dict-error' in line.lower() or 'DECODE ERROR' in line)
print(f'short (<=1 byte) transfers: {short}; CMD_FAIL/dict-error/DECODE ERROR lines: {fails}')
for problem in structure[:10]:
    print('problem:', problem)
ok = bad == 0 and short == 0 and fails == 0 and not structure
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
