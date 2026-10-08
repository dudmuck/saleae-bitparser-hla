#!/usr/bin/env python3
"""Byte-exact check of lr2021_pcycle prbs9 traffic in sigrok_hla.py --hex output.

usage: check_pcycle.py DECODE_STDOUT [--len 511] [--expect N --requester BUS
                       --responder BUS [--echo-lag 2]] [--json OUT]

Structure: every MOSI block must be followed by its MISO block on the same
bus, with equal length; an unpaired, overwritten or trailing block fails.
Frames: a write (MOSI 00 02 + frame) or a read (MOSI 00 01 + zeros, MISO =
2 status bytes + frame). A frame passes as a request if bytes 4.. equal
PRBS9 seeded from its LE32 seq, or as a reply if bytes 8.. do (bytes 4..7
echo a request seq). The only other long transfer accepted is the setup
TX FIFO prefill: 00 02 + exactly 2 x len zero bytes. Cross-bus accounting
counts occurrences (multisets), not just payload membership.

--expect N adds the full-run contract: requests seq 0..N-1 written by the
requester and read by the responder, each exactly once and identical;
replies resp_seq 0..N-1 read by the requester exactly once, identical to a
responder write. Every request (written or read) must match PRBS9 from byte
4; every reply, including unread staged writes, from byte 8 with echo
0xFFFFFFFF for the first echo-lag replies and resp_seq - echo-lag after. The
responder's write seqs must be exactly 0..N+lag-1 once each, optionally plus
one 0..lag-1 prefix staged by a negative control. Without --expect, a trace with no FIFO frames
fails as insufficient coverage.
"""
import argparse, collections, json, re, sys


def prbs9_frame(seq, length):
    out = bytearray(seq.to_bytes(4, 'little'))
    lfsr = (seq & 0x1ff) | 1
    while len(out) < length:
        b = 0
        for _ in range(8):
            fb = ((lfsr >> 8) ^ (lfsr >> 4)) & 1
            b = (b << 1) | (lfsr & 1)
            lfsr = ((lfsr << 1) | fb) & 0x1ff
        out.append(b)
    return bytes(out)


def classify(frame, length):
    if len(frame) != length:
        return 'wrong_length'
    seq = int.from_bytes(frame[:4], 'little')
    ref = prbs9_frame(seq, length)
    if frame[4:] == ref[4:]:
        return 'request'
    if frame[8:] == ref[8:]:
        return 'reply'
    return 'bad'


def parse(lines, stats, problems):
    """Pair MOSI/MISO hex blocks per bus; structural faults are problems."""
    transfers, pending = [], {}
    for n, line in enumerate(lines, 1):
        m = re.match(r'\s*(?:\[(\w+)\] )?(MOSI|MISO): ?(.*)$', line)
        if not m:
            continue
        bus = m[1] or 'SPI'
        try:
            data = bytes.fromhex(m[3].replace(' ', ''))
        except ValueError:
            problems.append(f'line {n}: malformed hex')
            continue
        if m[2] == 'MOSI':
            if bus in pending:
                problems.append(f'line {n}: [{bus}] MOSI without MISO before it')
            pending[bus] = data
        elif bus not in pending:
            problems.append(f'line {n}: [{bus}] MISO without MOSI')
        else:
            mosi = pending.pop(bus)
            if len(mosi) != len(data):
                problems.append(f'line {n}: [{bus}] MOSI {len(mosi)} bytes vs MISO {len(data)}')
            transfers.append((bus, mosi, data))
    for bus in pending:
        problems.append(f'[{bus}] MOSI block without MISO at end of output')
    return transfers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('decode'); ap.add_argument('--len', type=int, default=511)
    ap.add_argument('--expect', type=int, help='exchanges in a complete run (enables the full-run contract)')
    ap.add_argument('--requester', default='SPI'); ap.add_argument('--responder', default='SPI_B')
    ap.add_argument('--echo-lag', type=int, default=2); ap.add_argument('--json')
    a = ap.parse_args()
    lines = open(a.decode, errors='replace').read().splitlines()
    stats, problems, examples = collections.Counter(), [], []
    transfers = parse(lines, stats, problems)
    written = collections.defaultdict(list)    # bus -> frames written, in order
    read = collections.defaultdict(list)
    for bus, mosi, miso in transfers:
        stats[f'{bus} transfers'] += 1
        if len(mosi) <= 1:
            stats['short (<=1 byte) transfers'] += 1
            problems.append(f'[{bus}] {len(mosi)}-byte transfer')
            continue
        op = mosi[:2]
        if op == b'\x00\x02' and len(mosi) == a.len + 2:
            frame, kind = mosi[2:], 'write'
        elif op == b'\x00\x01' and len(mosi) == a.len + 2:
            frame, kind = miso[2:], 'read'
        elif op == b'\x00\x02' and len(mosi) == 2 * a.len + 2 and not any(mosi[2:]):
            stats['zero FIFO prefills'] += 1      # setup: 2 x len zero bytes
            continue
        elif op in (b'\x00\x01', b'\x00\x02'):
            # A FIFO read/write that is neither a frame nor the exact prefill:
            # lost or slipped bits, whatever its length.
            stats['malformed FIFO transfers'] += 1
            problems.append(f'[{bus}] FIFO {op.hex()} transfer of {len(mosi)} bytes')
            continue
        elif len(mosi) >= 64:
            # Only FIFO transfers are this long; a slipped or lost bit changes
            # their length or opcode, so any other long transfer is an error.
            stats['malformed long transfers'] += 1
            problems.append(f'[{bus}] malformed {len(mosi)}-byte transfer starting {mosi[:4].hex()}')
            continue
        else:
            stats['other (short) commands'] += 1
            continue
        verdict = classify(frame, a.len)
        stats[f'{bus} {kind} {verdict}'] += 1
        if verdict in ('bad', 'wrong_length'):
            seq = int.from_bytes(frame[:4], 'little')
            problems.append(f'[{bus}] {kind} seq {seq}: {verdict}')
            if len(examples) < 5 and verdict == 'bad':
                ref = prbs9_frame(seq, a.len)
                first = next(i for i in range(8, a.len) if frame[i] != ref[i])
                examples.append(dict(bus=bus, kind=kind, seq=seq, first_bad_byte=first,
                                     got=frame[first:first + 8].hex(), want=ref[first:first + 8].hex()))
        (written if kind == 'write' else read)[bus].append(frame)
    buses = sorted(set(written) | set(read))
    for src in buses:
        for dst in buses:
            if src == dst:
                continue
            w, r = collections.Counter(written[src]), collections.Counter(read[dst])
            extra_reads = r - w           # read more often than written: always an error
            unread = w - r                # write occurrences never read
            stats[f'{dst} reads not written on {src}'] = sum(extra_reads.values())
            stats[f'{src} write occurrences not read on {dst}'] = sum(unread.values())
            stats[f'{src} distinct payloads never read on {dst}'] = sum(1 for f in w if r[f] == 0)
            if extra_reads:
                problems.append(f'{sum(extra_reads.values())} {dst} reads with no matching {src} write')
            stats[f'{src} unread write seqs on {dst}'] = sorted(
                int.from_bytes(f[:4], 'little') for f in unread.elements())
    for k in ('CMD_FAIL', 'dict-error', 'DECODE ERROR'):
        stats[f'{k} lines'] = sum(1 for l in lines if k in l)
        if stats[f'{k} lines']:
            problems.append(f'{stats[f"{k} lines"]} {k} lines')
    frames = sum(len(v) for v in written.values()) + sum(len(v) for v in read.values())
    if a.expect is not None:
        problems += full_run_contract(a, written, read)
    elif not frames:
        problems.append('no FIFO frames: insufficient coverage')
    for k in sorted(stats):
        print(f'{k}: {stats[k]}')
    for e in examples:
        print('bad frame:', e)
    for p in problems[:20]:
        print('problem:', p)
    verdict = 'FAIL' if problems else 'PASS'
    print(verdict + (f' (full-run contract, {a.expect} exchanges)' if a.expect is not None and not problems else ''))
    if a.json:
        json.dump(dict(stats=dict(stats), problems=problems, bad_examples=examples, verdict=verdict,
                       expect=a.expect, requester=a.requester, responder=a.responder, echo_lag=a.echo_lag),
                  open(a.json, 'w'), indent=1)
    sys.exit(0 if verdict == 'PASS' else 1)


def full_run_contract(a, written, read):
    n, lag, req, resp = a.expect, a.echo_lag, a.requester, a.responder
    if n <= 0 or lag < 0:
        return [f'invalid contract: --expect {n} --echo-lag {lag}']
    seq = lambda f: int.from_bytes(f[:4], 'little')
    echo = lambda f: int.from_bytes(f[4:8], 'little')
    out = []

    def bad_request(f):
        return f[4:] != prbs9_frame(seq(f), a.len)[4:]

    def bad_reply(f):
        s = seq(f)
        return (f[8:] != prbs9_frame(s, a.len)[8:] or
                echo(f) != (0xFFFFFFFF if s < lag else s - lag))

    req_w, req_r = written[req], read[resp]
    rep_w, rep_r = written[resp], read[req]
    # Role-specific payloads: requests from byte 4; replies from byte 8 with the
    # echo rule, for every occurrence including unread staged writes.
    for name, frames, bad in ((f'{req} request write', req_w, bad_request),
                              (f'{resp} request read', req_r, bad_request),
                              (f'{resp} reply write', rep_w, bad_reply),
                              (f'{req} reply read', rep_r, bad_reply)):
        wrong = [seq(f) for f in frames if bad(f)]
        if wrong:
            out.append(f'{len(wrong)} {name}(s) with wrong payload or echo, first seq {wrong[0]}')
    if sorted(map(seq, req_w)) != list(range(n)):
        out.append(f'{req} request writes are not seq 0..{n - 1} exactly once')
    if sorted(map(seq, req_r)) != list(range(n)):
        out.append(f'{resp} request reads are not seq 0..{n - 1} exactly once')
    if collections.Counter(req_w) != collections.Counter(req_r):
        out.append('request frames written and read differ')
    if sorted(map(seq, rep_r)) != list(range(n)):
        out.append(f'{req} reply reads are not resp_seq 0..{n - 1} exactly once')
    if collections.Counter(rep_r) - collections.Counter(rep_w):
        out.append('a reply read has no identical responder write')
    # The responder writes resp_seq 0..n+lag-1 once each (lag replies are staged
    # ahead), plus at most one extra 0..lag-1 prefix staged by a negative control.
    staged = collections.Counter(map(seq, rep_w))
    pipeline = collections.Counter(range(n + lag))
    if staged not in (pipeline, pipeline + collections.Counter(range(lag))):
        out.append(f'{resp} reply write seqs are not 0..{n + lag - 1} once each '
                   f'(optionally plus one negative-control 0..{lag - 1} prefix)')
    if any(written[b] for b in written if b not in (req, resp)) or any(read[b] for b in read if b not in (req, resp)):
        out.append('FIFO frames on an unexpected bus')
    return out


if __name__ == '__main__':
    main()
