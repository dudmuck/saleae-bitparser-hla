#!/usr/bin/env python3
"""Offline raw producer; accepts dslcap/sigrok arguments without USB access."""
import os
import signal
import sys
import time


def spi_samples():
    data = bytearray([8] * 4)
    for mosi, miso in ((0xA5, 0x3C), (0x5A, 0xC3)):
        for bit in range(7, -1, -1):
            sample = ((mosi >> bit) & 1) << 2 | ((miso >> bit) & 1) << 1
            data.extend([sample, sample, sample | 1, sample | 1])
    data.extend([8] * 8)
    for sample in range(20, 40):
        data[sample] |= 16
    return bytes(data)


if __name__ == '__main__':
    scenario = os.environ.get('DSLCAP_FIXTURE_SCENARIO', 'fragmented')
    if scenario == 'empty-failure':
        os.write(2, b'configuration rejected before META\n')
        sys.exit(8)
    if scenario in ('stubborn', 'flood'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if scenario == 'flood':
        while True:
            os.write(1, b'\x08' * 1048576)
    if scenario in ('pressure', 'srd'):
        for i in range(192):
            os.write(2, f'log {i}: '.encode() + b'x' * 1024 + b'\n')
    if scenario == 'srd':
        os.write(1, b'4-68 spi-1: 3C C3\n4-68 spi-1: A5 5A\n')
        sys.exit(0)
    if scenario == 'invalid-meta':
        os.write(1, b'META samplerate: 0\n')
        sys.exit(0)
    if scenario == 'truncated-meta':
        os.write(1, b'META samplerate: 200')
        sys.exit(0)
    if scenario != 'raw':
        for part in (b'M', b'E', b'TA sampler', b'ate: 200', b'0000\n'):
            os.write(1, part)
            time.sleep(.005)
    data = spi_samples()
    if scenario == 'dual':
        base = bytes(value & 15 for value in data)
        idle = bytes([8] * 8)
        data = bytes(low | (high << 4) for low, high in zip(base + idle, idle + base))
    for start in range(0, len(data), 3):
        os.write(1, data[start:start + 3])
        time.sleep(.001)
    if scenario == 'failure':
        os.write(2, b'forced producer failure\n')
        sys.exit(9)
    if scenario == 'stubborn':
        while True:
            time.sleep(1)
