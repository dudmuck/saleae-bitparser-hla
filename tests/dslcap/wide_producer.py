#!/usr/bin/env python3
"""Synthetic DSLogic uint16 producer: no hardware or application access."""
import os
import random
import signal
import struct
import time


def port_samples(pairs):
    samples = [8] * 4
    for mosi, miso in pairs:
        for bit in range(7, -1, -1):
            word = ((mosi >> bit) & 1) << 2 | ((miso >> bit) & 1) << 1
            samples.extend([word, word, word | 1, word | 1])
    return samples + [8] * 8


def wide_samples():
    low = port_samples(((0xA5, 0x3C), (0x5A, 0xC3))) + [8] * 8
    high = [8] * 8 + port_samples(((0x96, 0x69), (0x12, 0x34)))
    words = [a | (b << 8) for a, b in zip(low, high)]
    for sample in range(20, 40):
        words[sample] |= 1 << 15
    for sample in range(28, 48):
        words[sample] |= 1 << 12
    return struct.pack('<' + 'H' * len(words), *words)


if __name__ == '__main__':
    scenario = os.environ.get('DSLCAP_WIDE_SCENARIO', 'odd')
    if scenario == 'stubborn':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    header = b'META samplerate: 2000000\n'
    payload = wide_samples()
    if scenario in ('truncated', 'failed-truncated'):
        payload = payload[:-1]
    # Fragment META and samples together, including odd offsets at the
    # ASCII/binary boundary. Both one-byte and seeded random writes work.
    wire = header + payload
    rng = random.Random(1871)
    position = 0
    while position < len(wire):
        length = rng.randrange(1, 18) if scenario == 'random' else 3
        if scenario == 'one-byte': length = 1
        os.write(1, wire[position:position + length])
        position += length
        time.sleep(.0005)
    if scenario == 'failed-truncated':
        os.write(2, b'upstream wide capture failed\n')
        raise SystemExit(10)
    if scenario == 'stubborn':
        while True:
            time.sleep(1)
