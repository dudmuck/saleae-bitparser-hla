#!/usr/bin/env python3
"""Synthetic lr2021_pcycle ping-pong on two SPI buses, as a dslcap raw file.

Bus A (initiator) bits 0..5 = SCLK MISO MOSI nSS BUSY DIO8; bus B (responder)
bits 8..13. Per exchange: A writes request seq k (00 02 + frame), B reads it
(MOSI 00 01 + zeros, MISO = 2 status bytes + frame), then B writes reply
(resp_seq, bytes 4..7 = echoed k) and A reads it. SPI mode 0, MSB first.
SCLK edges are placed in continuous time and sampled at n/rate + phase; a
small ppm offset sweeps every sampling phase across the run.

usage: gen_pp.py OUT --exchanges N [--rate 25e6] [--sclk 10e6] [--ppm 37]
       [--phase 0.0] [--gap-bits 0] [--period-us 4600] [--duty 0.5]
Writes OUT (META samplerate + uint16 samples), OUT.json (dslcap-raw-v1) and
OUT.expected.json (every transfer's MOSI/MISO hex, in time order).
"""
import argparse, json
import numpy as np


def prbs9_frame(seq, length=511):
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


def reply_frame(resp_seq, echo):
    f = bytearray(prbs9_frame(resp_seq))
    f[4:8] = echo.to_bytes(4, 'little')
    return bytes(f)


def check_vectors():
    v = {0: '00000000846139561bd37228569fb24b', 2: '020000008ca34bfa2c759678fba0d6dd',
         511: 'ff01000083df1732094ed1e7cd8a91c6', 512: '00020000846139561bd37228569fb24b'}
    tails = {0: '362a471b', 2: '5a7ec92d', 511: '1219c2f6'}
    for s, h in v.items():
        assert prbs9_frame(s)[:16].hex() == h, s
    for s, h in tails.items():
        assert prbs9_frame(s)[-4:].hex() == h, s


class Bus:
    def __init__(self, base):
        self.clk, self.miso, self.mosi, self.cs, self.busy, self.dio = (base + i for i in range(6))


def render(out_path, exchanges, rate, sclk, ppm, phase, gap_bits, period_us, duty):
    A, B = Bus(0), Bus(8)
    tbit = 1.0 / sclk
    events = []          # (t0, bus, mosi bytes, miso bytes)
    t = 20e-6
    for k in range(exchanges):
        req = prbs9_frame(k)
        rep = reply_frame(k, k)
        status = bytes([0x04, 0x52])
        def xfer(bus, mosi, miso, t0):
            events.append((t0, bus, mosi, miso))
            return t0 + len(mosi) * (8 + gap_bits) * tbit + 2e-6
        tw = xfer(A, b'\x00\x02' + req, status + b'\x00' * len(req), t)
        tr = xfer(B, b'\x00\x01' + b'\x00' * len(req), status + req, t + 0.15 * period_us * 1e-6)
        tw2 = xfer(B, b'\x00\x02' + rep, status + b'\x00' * len(rep), t + 0.45 * period_us * 1e-6)
        xfer(A, b'\x00\x01' + b'\x00' * len(rep), status + rep, t + 0.60 * period_us * 1e-6)
        t += period_us * 1e-6
    total = int((t + 20e-6) * rate)
    samples = np.zeros(total, dtype=np.uint16)
    samples |= np.uint16((1 << A.cs) | (1 << B.cs))       # nSS idle high
    scale = 1.0 + ppm * 1e-6                               # analyzer clock error
    for t0, bus, mosi, miso in events:
        nbits = len(mosi) * 8
        starts = t0 + 0.3e-6 + np.arange(nbits) // 8 * (8 + gap_bits) * tbit + np.arange(nbits) % 8 * tbit
        t_end = starts[-1] + tbit + 0.3e-6
        n0, n1 = int(np.ceil((t0 * rate - phase) / scale)), int(np.floor((t_end * rate - phase) / scale))
        ts = (np.arange(n0, n1 + 1) * scale + phase) / rate
        seg = samples[n0:n1 + 1]
        seg &= np.uint16(~(1 << bus.cs) & 0xffff)          # nSS asserted
        idx = np.searchsorted(starts, ts, side='right') - 1
        inbit = (idx >= 0) & (ts < starts[np.clip(idx, 0, None)] + tbit)
        ph = ts - starts[np.clip(idx, 0, None)]
        mbits = np.unpackbits(np.frombuffer(mosi, np.uint8))
        sbits = np.unpackbits(np.frombuffer(miso, np.uint8))
        i = np.clip(idx, 0, nbits - 1)
        clk = inbit & (ph >= (1 - duty) * tbit)            # mode 0: low then high
        # data holds from bit start until the next bit's start (incl. gaps)
        valid = idx >= 0
        seg |= (clk.astype(np.uint16) << bus.clk)
        seg |= ((valid & (mbits[i] == 1)).astype(np.uint16) << bus.mosi)
        seg |= ((valid & (sbits[i] == 1)).astype(np.uint16) << bus.miso)
        # BUSY pulse after each command, 20 us
        b0, b1 = n1 + 1, min(total, n1 + 1 + int(20e-6 * rate))
        samples[b0:b1] |= np.uint16(1 << bus.busy)
    with open(out_path, 'wb') as f:
        f.write(b'META samplerate: %d\n' % int(rate))
        f.write(samples.tobytes())
    chans = sorted([A.clk, A.miso, A.mosi, A.cs, A.busy, A.dio, B.clk, B.miso, B.mosi, B.cs, B.busy, B.dio])
    json.dump(dict(format='dslcap-raw-v1', command=['synthetic', 'gen_pp.py'], channels=chans, unitsize=2,
                   meta_lines=1, exit=0, synthetic=dict(exchanges=exchanges, rate=rate, sclk=sclk, ppm=ppm,
                   phase=phase, gap_bits=gap_bits, period_us=period_us, duty=duty)),
              open(out_path + '.json', 'w'), indent=2)
    events.sort(key=lambda e: e[0])
    json.dump([dict(t=e[0], bus='A' if e[1] is A else 'B', mosi=e[2].hex(), miso=e[3].hex()) for e in events],
              open(out_path + '.expected.json', 'w'))
    return total


if __name__ == '__main__':
    check_vectors()
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--exchanges', type=int, required=True)
    ap.add_argument('--rate', type=float, default=25e6); ap.add_argument('--sclk', type=float, default=10e6)
    ap.add_argument('--ppm', type=float, default=37.0); ap.add_argument('--phase', type=float, default=0.0)
    ap.add_argument('--gap-bits', type=float, default=0.0); ap.add_argument('--period-us', type=float, default=4600)
    ap.add_argument('--duty', type=float, default=0.5)
    a = ap.parse_args()
    n = render(a.out, a.exchanges, a.rate, a.sclk, a.ppm, a.phase, a.gap_bits, a.period_us, a.duty)
    print(f'{a.out}: {n} samples, {2*n/1e6:.1f} MB')
