"""Streaming deglitch parity with the patched file-input libsigrok oracle."""
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

import numpy as np

from deglitch import Deglitcher, HAVE_NUMBA

ORACLE = Path(os.environ.get("DEGLITCH_ORACLE", "/home/wroberts/.local/bin/sigrok-cli"))
SCRATCH = Path("/tmp/dslcap-j-verify")


def oracle(samples, bits, **options):
    """Invoke only sigrok-cli's file-input path; no device selection/scanning."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oracle-", dir=SCRATCH) as directory:
        source, output = Path(directory) / "in.bin", Path(directory) / "out.bin"
        samples.tofile(source)
        transform = "deglitch:channels=" + ",".join(map(str, bits))
        transform += "".join(":" + key + "=" + str(value) for key, value in options.items())
        command = [str(ORACLE), "-l", "4", "-i", str(source), "-I",
                   f"binary:numchannels={samples.dtype.itemsize * 8}:samplerate=25000000",
                   "-T", transform, "-O", "binary", "-o", str(output)]
        result = subprocess.run(command, capture_output=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stderr.decode(errors="replace"))
        data = output.read_bytes()
        if data.startswith(b"META samplerate:"):
            header, data = data.split(b"\n", 1)
            if header != b"META samplerate: 25000000":
                raise AssertionError(header)
        stats = {bit: dict(suppressed=0, split=0, inserted=0, unresolved=0) for bit in bits}
        for fields in re.findall(rb"Channel (\d+): suppressed (\d+), split (\d+), inserted (\d+), unresolved (\d+)", result.stderr):
            bit, *counts = map(int, fields)
            stats[bit] = dict(zip(("suppressed", "split", "inserted", "unresolved"), counts))
        return np.frombuffer(data, dtype=samples.dtype).copy(), stats


def traffic(dtype, phase, corrupt=False, pulse_level=1):
    """25 MS/s raster of 10 MHz SPI, with byte pauses and frame/NSS gaps.

    Both grid and midpoint phases have clean1/2-sample highs. Explicit losses
    model the vanished/merged phases of a marginal sampled capture, rather than
    relying on platform-specific floating-point tie breaking to inject faults.
    """
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(312)
    samples = rng.integers(0, 1 << (dtype.itemsize * 8), size=3500, dtype=dtype)
    bits = [0] if dtype.itemsize == 1 else [0, 8]
    for bit in bits:
        mask = 1 << bit
        samples &= np.array(((1 << (8 * dtype.itemsize)) - 1) ^ mask, dtype=dtype)
        nss = 1 << (bit + 3)
        samples |= np.array(nss, dtype=dtype)
        levels = np.zeros(len(samples), dtype=np.uint8)
        cursor = 40 + (19 if bit else 0)
        for transaction in range(4):
            transaction_start = cursor
            for byte in range(5):
                clock = ((np.arange(20, dtype=np.float64) + phase) % 2.5 < 1.25).astype(np.uint8)
                if corrupt and byte == 1:
                    # Remove one active run from within a byte.
                    rises = np.flatnonzero(np.diff(np.r_[0, clock, 0].astype(np.int8)) == 1)
                    falls = np.flatnonzero(np.diff(np.r_[0, clock, 0].astype(np.int8)) == -1)
                    clock[rises[3]:falls[3]] = 0
                if corrupt and byte == 2:
                    # Remove a low phase, merging two adjacent active runs.
                    rises = np.flatnonzero(np.diff(np.r_[0, clock, 0].astype(np.int8)) == 1)
                    falls = np.flatnonzero(np.diff(np.r_[0, clock, 0].astype(np.int8)) == -1)
                    clock[falls[2]:rises[3]] = 1
                if corrupt and byte == 4:
                    # Tail pulse swallowed by the following NSS/idle gap.
                    rises = np.flatnonzero(np.diff(np.r_[0, clock, 0].astype(np.int8)) == 1)
                    clock[rises[-1]:] = 0
                levels[cursor:cursor + 20] = clock
                cursor += 20 + 6  # legitimate short inter-byte pause
            # nSS stays low through inter-byte pauses, high between transactions.
            samples[transaction_start:cursor] &= np.array(((1 << (8 * dtype.itemsize)) - 1) ^ nss, dtype=dtype)
            cursor += 65  # NSS gap, longer than gap_max
        if pulse_level == 0:
            levels ^= 1
        samples |= (levels.astype(dtype) << bit)
    return samples, bits


def run_stream(samples, bits, splits=None, use_numba=None, **options):
    repair = Deglitcher(bits, 25000000, dtype=samples.dtype, use_numba=use_numba, **options)
    output = []
    offset = 0
    for length in splits or [len(samples)]:
        output.append(repair.feed(samples[offset:offset + length]))
        offset += length
    if offset < len(samples):
        output.append(repair.feed(samples[offset:]))
    output.append(repair.end())
    return np.concatenate(output), repair


class DeglitchTests(unittest.TestCase):
    def assert_oracle(self, samples, bits, **options):
        expected, stats = oracle(samples, bits, **options)
        for backend in ([False, True] if HAVE_NUMBA else [False]):
            actual, repair = run_stream(samples, bits, use_numba=backend, **options)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(repair.stats, stats)
        return expected, stats

    @unittest.skipUnless(ORACLE.is_file(), "patched sigrok-cli file oracle unavailable")
    def test_spi_phases_types_and_injected_losses(self):
        for dtype in (np.uint8, np.uint16):
            for phase in (0.0, 0.5):
                with self.subTest(dtype=dtype, phase=phase):
                    samples, bits = traffic(dtype, phase, corrupt=True)
                    actual, stats = self.assert_oracle(samples, bits, clock_period=2.5, frame_pulses=8)
                    self.assertGreater(sum(s["split"] for s in stats.values()), 0)
                    self.assertGreater(sum(s["inserted"] for s in stats.values()), 0)
                    mask = sum(1 << bit for bit in bits)
                    unchanged = ((1 << (8 * samples.dtype.itemsize)) - 1) ^ mask
                    np.testing.assert_array_equal(actual & unchanged, samples[:len(actual)] & unchanged)

    @unittest.skipUnless(ORACLE.is_file(), "patched sigrok-cli file oracle unavailable")
    def test_clean_do_no_harm(self):
        for dtype in (np.uint8, np.uint16):
            for phase in (0.0, 0.5):
                samples, bits = traffic(dtype, phase)
                actual, stats = self.assert_oracle(samples, bits, clock_period=2.5, frame_pulses=8)
                np.testing.assert_array_equal(actual, samples[:-14])
                self.assertTrue(all(all(count == 0 for count in s.values()) for s in stats.values()))

    def test_arbitrary_splits_match_single_feed(self):
        rng = np.random.default_rng(194)
        for dtype in (np.uint8, np.uint16):
            samples, bits = traffic(dtype, 0.0, corrupt=True)
            for options in (dict(clock_period=2.5, frame_pulses=8), dict(min_period=5),
                            dict(clock_period=3.5, frame_pulses=8, min_period=4)):
                expected, whole = run_stream(samples, bits, use_numba=False, **options)
                for backend in ([False, True] if HAVE_NUMBA else [False]):
                    lengths = rng.integers(1, 48, size=300).tolist()
                    actual, repair = run_stream(samples, bits, lengths, backend, **options)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(repair.stats, whole.stats)
                    one, repair = run_stream(samples[:180], bits, [1] * 180, backend, **options)
                    check, whole_one = run_stream(samples[:180], bits, use_numba=False, **options)
                    np.testing.assert_array_equal(one, check)
                    self.assertEqual(repair.stats, whole_one.stats)

    def test_multibus_independence(self):
        samples, bits = traffic(np.uint16, 0.5, corrupt=True)
        both, repaired = run_stream(samples, bits, clock_period=2.5, frame_pulses=8)
        for bit in bits:
            single, single_repair = run_stream(samples, [bit], clock_period=2.5, frame_pulses=8)
            np.testing.assert_array_equal(both & (1 << bit), single & (1 << bit))
            self.assertEqual(repaired.stats[bit], single_repair.stats[bit])

    @unittest.skipUnless(ORACLE.is_file(), "patched sigrok-cli file oracle unavailable")
    def test_options_random_logic_and_active_low(self):
        rng = np.random.default_rng(511)
        for dtype in (np.uint8, np.uint16):
            samples = rng.integers(0, 1 << (8 * np.dtype(dtype).itemsize), size=2100, dtype=dtype)
            bits = [0] if dtype == np.uint8 else [0, 8]
            for options in (dict(min_period=1), dict(min_period=5),
                            dict(clock_period=2.0, frame_pulses=8),
                            dict(clock_period=2.5, frame_pulses=0),
                            dict(clock_period=3.5, frame_pulses=8, min_period=4),
                            dict(clock_period=2.5, frame_pulses=8, pulse_level=0)):
                with self.subTest(dtype=dtype, options=options):
                    self.assert_oracle(samples, bits, **options)
        samples, bits = traffic(np.uint16, 0.5, True, pulse_level=0)
        self.assert_oracle(samples, bits, clock_period=2.5, frame_pulses=8, pulse_level=0)

    def test_end_short_empty_disabled_and_input_ownership(self):
        for length in (0, 1, 13, 14, 15, 40):
            source = np.arange(length, dtype=np.uint16)
            original = source.copy()
            actual, repair = run_stream(source, [0, 8], clock_period=2.5, frame_pulses=8)
            self.assertEqual(len(actual), max(0, length - 14))
            np.testing.assert_array_equal(source, original)
            self.assertEqual(repair.end().size, 0)
            with self.assertRaises(ValueError):
                repair.feed(source)
            disabled, _ = run_stream(source, [0, 8])
            np.testing.assert_array_equal(disabled, source)
            none, _ = run_stream(source, [], clock_period=2.5)
            np.testing.assert_array_equal(none, source)

    def test_invalid_configuration_and_packet_width(self):
        for options in (dict(clock_period=-1), dict(clock_period=1.9), dict(clock_period=float("nan")),
                        dict(min_period=-1), dict(frame_pulses=-1), dict(dtype=np.int16)):
            with self.assertRaises((ValueError, TypeError)):
                Deglitcher([0], 25000000, **options)
        for bits in ([-1], [64]):
            with self.assertRaises(ValueError):
                Deglitcher(bits, 25000000)
        repair = Deglitcher([0], 25000000)
        with self.assertRaises(ValueError):
            repair.feed(np.zeros(4, dtype=np.uint16))
        with self.assertRaises(ValueError):
            repair.feed(np.zeros((2, 2), dtype=np.uint8))

    @unittest.skipUnless(ORACLE.is_file(), "patched sigrok-cli file oracle unavailable")
    def test_real_saleae_export_rasterized_read_only(self):
        # Original dual_spi is empty on this checkout. The retained 25M capture
        # package contains decode tables only; its9.55GB raw stays on T14s.
        # Supplement with actual local Saleae transitions, explicitly rasterized
        # at25M rather than claiming an original25M uniform acquisition.
        source = Path(__file__).resolve().parents[1] / "board_b"
        paths = [source / f"digital_{bit}.bin" for bit in range(8)]
        if not all(path.is_file() for path in paths):
            self.skipTest("local real Saleae export unavailable")
        times = 1.8848 + np.arange(100000, dtype=np.float64) / 25000000
        samples = np.zeros(len(times), dtype=np.uint8)
        for bit, path in enumerate(paths):
            with path.open("rb") as stream:
                header = stream.read(44)
            self.assertEqual(header[:8], b"<SALEAE>")
            initial = struct.unpack_from("<I", header, 16)[0]
            count = struct.unpack_from("<Q", header, 36)[0]
            transitions = np.fromfile(path, dtype="<f8", offset=44)
            self.assertEqual(len(transitions), count)
            levels = (initial ^ (np.searchsorted(transitions, times, side="right") & 1)).astype(np.uint8)
            samples |= levels << bit
        self.assertGreater(np.count_nonzero(np.diff(samples & 16)), 100)
        self.assert_oracle(samples, [0, 4], clock_period=2.5, frame_pulses=8)


if __name__ == "__main__":
    unittest.main()
