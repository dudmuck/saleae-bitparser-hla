"""The numba single-pass port scan must decode exactly like the NumPy path."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fast_spi


def spi_stream(rng, n_frames, base, wide, gap=3):
    """Random SPI frames on bits base..base+3 (clk, miso, mosi, cs) with jittered
    clock phases of 1-3 samples, so edges land on every chunk alignment."""
    dtype = np.uint16 if wide else np.uint8
    out = [np.full(rng.integers(1, 6), 1 << (base + 3), dtype=dtype)]
    for _ in range(n_frames):
        out.append(np.zeros(rng.integers(1, 4), dtype=dtype))      # cs low, clk idle
        for _ in range(rng.integers(0, 6) * 8 + rng.integers(0, 3)):
            mo, mi = rng.integers(0, 2, 2)
            data = (int(mo) << (base + 2)) | (int(mi) << (base + 1))
            out.append(np.full(rng.integers(1, 4), data, dtype=dtype))
            out.append(np.full(rng.integers(1, 4), data | (1 << base), dtype=dtype))
        out.append(np.full(rng.integers(1, gap + 2), 1 << (base + 3), dtype=dtype))
    return np.concatenate(out)


def decode(chunks, ports, rate=25e6, cpol=0, cpha=0, numba=True):
    with mock.patch.object(fast_spi, 'HAVE_NUMBA', numba):
        dec = fast_spi.MultiPortDecoder(ports, rate, cpol=cpol, cpha=cpha)
        frames = []
        for chunk in chunks:
            for name, batch in dec.feed(chunk):
                frames += [(name, f.type, f.start_time, f.end_time, f.data) for f in batch]
        for name, batch in dec.end():
            frames += [(name, f.type, f.start_time, f.end_time, f.data) for f in batch]
    # Batches interleave ports chunk by chunk; each port's own order is the contract.
    return sorted(frames, key=lambda f: f[0])


def splits(samples, rng, max_len):
    out, pos = [], 0
    while pos < len(samples):
        n = int(rng.integers(1, max_len))
        out.append(samples[pos:pos + n])
        pos += n
    return out


@unittest.skipUnless(fast_spi.HAVE_NUMBA, 'numba not installed')
class ScanEquivalence(unittest.TestCase):
    def check(self, samples, ports, **kw):
        rng = np.random.default_rng(7)
        whole = decode([samples], ports, numba=False, **kw)
        self.assertTrue(any(f[1] == 'result' for f in whole))
        self.assertEqual(decode([samples], ports, numba=True, **kw), whole)
        for max_len in (2, 7, 64, 4096):
            with self.subTest(max_len=max_len):
                chunks = splits(samples, rng, max_len)
                self.assertEqual(decode(chunks, ports, numba=True, **kw), whole)
                self.assertEqual(decode(chunks, ports, numba=False, **kw), whole)

    def test_uint8_single_port_all_modes(self):
        samples = spi_stream(np.random.default_rng(1), 40, 0, wide=False)
        for cpol, cpha in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(cpol=cpol, cpha=cpha):
                self.check(samples, [dict(name='SPI', clk=0, miso=1, mosi=2, cs=3)], cpol=cpol, cpha=cpha)

    def test_uint16_two_independent_ports(self):
        rng = np.random.default_rng(2)
        a = spi_stream(rng, 30, 0, wide=True)
        b = spi_stream(rng, 30, 8, wide=True)
        n = max(len(a), len(b))
        a = np.concatenate([a, np.full(n - len(a), 1 << 3, np.uint16)])
        b = np.concatenate([b, np.full(n - len(b), 1 << 11, np.uint16)])
        ports = [dict(name='SPI', clk=0, miso=1, mosi=2, cs=3),
                 dict(name='SPI_B', clk=8, miso=9, mosi=10, cs=11)]
        self.check(a | b, ports)

    def test_edge_on_first_sample_without_history(self):
        # Without history the first sample is never an edge, in both paths.
        samples = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 8], dtype=np.uint8)
        port = [dict(name='SPI', clk=0, miso=1, mosi=2, cs=3)]
        self.assertEqual(decode([samples], port, numba=True), decode([samples], port, numba=False))


    @unittest.skipUnless(Path('/proc/self/status').exists(), 'needs Linux /proc')
    def test_scan_results_do_not_pin_chunk_buffers(self):
        # A transaction held open keeps its latched slices. Uncompacted, each
        # 2M-sample chunk would pin ~36 MB of scratch (1.8 GB for 50 chunks).
        def vmsize():
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmSize:'):
                    return int(line.split()[1]) * 1024
        chunk = np.zeros(2_000_000, dtype=np.uint16)
        chunk[1_000_000:] = 1
        fast_spi._scan_port(chunk, 0, 1, 2, 3, 0, True, True)    # compile first
        before = vmsize()
        kept = [fast_spi._scan_port(chunk, 0, 1, 2, 3, 0, True, True) for _ in range(50)]
        self.assertEqual(sum(r[1].size for r in kept), 50)
        self.assertLess(vmsize() - before, 200 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
