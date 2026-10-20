"""Build the W1 tool locally; all generator invocations are dry runs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'dslcap' / 'tools'
spec = importlib.util.spec_from_file_location('check_pattern', TOOLS / 'check_pattern.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='pi-pattern-offline-')
        cls.binary = Path(cls.directory.name) / 'pi_pattern'
        subprocess.run(['gcc', '-O2', '-Wall', '-Wextra', '-Werror', str(TOOLS / 'pi_pattern.c'),
                        '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def generate(self, pattern, states=1000, *extra):
        r = subprocess.run([str(self.binary), pattern, '--dry-run', '--duration', '1',
                            '--states', str(states), *extra], capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0, r.stderr)
        records = [json.loads(line) for line in r.stdout.splitlines()]
        return [r['state'] for r in records if 'state' in r], records

    def write_raw(self, folder, values, repeats=3, wide=False):
        raw = Path(folder) / 'capture.raw'
        values = np.repeat(np.array(values, dtype='<u2' if wide else 'u1'), repeats)
        raw.write_bytes(b'META samplerate: 1000000\n' + values.tobytes())
        return raw

    def test_sequences_and_dwell(self):
        for pattern in ('walk', 'prbs'):
            values, records = self.generate(pattern)
            cycle = checker.cycle_for(pattern)
            self.assertEqual(values, [cycle[i % len(cycle)] for i in range(1000)])
            self.assertTrue(all(a != b for a, b in zip(values, values[1:])))
            for repeats in (1, 3, 10):
                with tempfile.TemporaryDirectory() as tmp:
                    raw = self.write_raw(tmp, values[7:], repeats)
                    states, rate, total = checker.read_states(raw, [0])
                    self.assertEqual(total, 993 * repeats)
                    self.assertIsNone(checker.check_sequence(*states[0], pattern))
            if pattern == 'walk':
                with tempfile.TemporaryDirectory() as tmp:
                    raw = self.write_raw(tmp, values, 1)
                    run = subprocess.run([sys.executable, str(TOOLS / 'check_pattern.py'), str(raw),
                                          '--pattern', pattern], capture_output=True, text=True, timeout=5)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn('warning:', run.stdout)
                    self.assertIn('dwell samples', run.stdout)

    def test_fault_diagnoses(self):
        for pattern in ('walk', 'prbs'):
            values, _ = self.generate(pattern)
            def swapped(v):
                return v ^ ((((v >> 1) ^ (v >> 6)) & 1) * ((1 << 1) | (1 << 6)))
            faults = {
                'dropped': values[:100] + values[102:],
                'duplicated': values[:100] + values[98:100] + values[100:],
                'reordered': values[:100] + [values[101], values[100]] + values[102:],
                'stuck lane': [v & ~2 for v in values],
                'swapped lanes': [swapped(v) for v in values],
            }
            for diagnosis, bad in faults.items():
                with self.subTest(pattern=pattern, diagnosis=diagnosis), tempfile.TemporaryDirectory() as tmp:
                    raw = self.write_raw(tmp, bad)
                    states, _, _ = checker.read_states(raw, [0])
                    error = checker.check_sequence(*states[0], pattern)
                    self.assertIsNotNone(error)
                    self.assertIn(diagnosis, error)
                    self.assertIn('sample', error)
                    if diagnosis == 'dropped':
                        run = subprocess.run([sys.executable, str(TOOLS / 'check_pattern.py'), str(raw),
                                              '--pattern', pattern], capture_output=True, text=True, timeout=5)
                        self.assertEqual(run.returncode, 1)

    def test_exact_clear_set_intermediates(self):
        for pattern in ('walk', 'prbs'):
            values, records = self.generate(pattern)
            self.assertEqual(records[0]['write_order'], ['GPCLR0', 'GPSET0'])
            observed = [values[0]]
            count = 0
            gray_count = 0
            for i, (a, b) in enumerate(zip(values, values[1:])):
                middle = a & b
                if middle not in (a, b) and i % 2 == 0:
                    observed.append(middle)
                    count += 1
                    if pattern == 'walk' and 9 <= (i + 1) % 264:
                        gray_count += 1
                observed.append(b)
            stats = {}
            error = checker.check_sequence(np.array(observed), np.arange(len(observed)), pattern, statistics=stats)
            self.assertIsNone(error)
            self.assertEqual(stats['intermediates'], count)
            self.assertGreater(count, 0)
            self.assertEqual(gray_count, 0)
            # Start inside a clear/set gap, then continue an exact cycle.
            gap = next(i for i, (a, b) in enumerate(zip(values, values[1:])) if a & b not in (a, b))
            clipped = [values[gap] & values[gap + 1]] + values[gap + 1:]
            self.assertIsNone(checker.check_sequence(np.array(clipped), np.arange(len(clipped)), pattern))
            for wrong in (values[gap] | values[gap + 1], 255):
                bad = values[:gap + 1] + [wrong] + values[gap + 1:]
                self.assertIsNotNone(checker.check_sequence(np.array(bad), np.arange(len(bad)), pattern))
            # An intermediate followed by A again is a reordered/repeated gap.
            bad = values[:gap + 1] + [values[gap] & values[gap + 1], values[gap]] + values[gap + 1:]
            self.assertIsNotNone(checker.check_sequence(np.array(bad), np.arange(len(bad)), pattern))
            # A third state on a single-bit Gray transition is never allowed.
            if pattern == 'walk':
                bad = values[:10] + [255] + values[10:]
                self.assertIsNotNone(checker.check_sequence(np.array(bad), np.arange(len(bad)), pattern))

    def test_two_groups_independent_phase_and_chunk_boundary(self):
        values, _ = self.generate('prbs')
        combined = [v | values[(i + 73) % len(values)] << 8 for i, v in enumerate(values[:800])]
        with tempfile.TemporaryDirectory() as tmp:
            raw = self.write_raw(tmp, combined, repeats=1000, wide=True)
            states, _, total = checker.read_states(raw, [0, 8])
            self.assertEqual(total, 800000)
            for group in (0, 8):
                self.assertIsNone(checker.check_sequence(*states[group], 'prbs'))
            raw.write_bytes(raw.read_bytes() + b'\x01')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                checker.read_states(raw, [0, 8])

    def test_spi_manifest_and_raw_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.jsonl'
            values, records = self.generate('spi', 804, '--manifest', str(manifest))
            for a, b in zip(values, values[1:]):
                if (a ^ b) & 1:
                    self.assertEqual((a ^ b) & 6, 0, 'SCLK change combined with data change')
            frames = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(frames), 2)
            self.assertNotEqual(frames[0]['mosi'], frames[0]['miso'])
            self.assertEqual(len(bytes.fromhex(frames[0]['mosi'])), 16)
            lines = '\n'.join(f'MOSI: {f["mosi"]}\nMISO: {f["miso"]}' for f in frames)
            self.assertIsNone(checker.check_spi(lines, manifest))
            self.assertIn('count mismatch', checker.check_spi(lines.split('MOSI:')[0], manifest))
            self.assertIn('mismatch', checker.check_spi(lines.replace(frames[0]['mosi'], '00'), manifest))
            self.assertIn('frame 0 mismatch', checker.check_spi('\n'.join(
                f'MOSI: {f["mosi"]}\nMISO: {f["miso"]}' for f in frames[::-1]), manifest))
            raw = self.write_raw(tmp, values, repeats=3)
            run = subprocess.run([sys.executable, str(TOOLS / 'check_pattern.py'), str(raw),
                                  '--pattern', 'spi', '--manifest', str(manifest),
                                  '--hla-path', str(ROOT / 'tests' / 'dslcap')],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            decoded = Path(tmp) / 'decode.txt'
            decoded.write_text(lines)
            run = subprocess.run([sys.executable, str(TOOLS / 'check_pattern.py'), str(decoded),
                                  '--pattern', 'spi', '--manifest', str(manifest), '--decode'],
                                 capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_invalid_arguments_never_open_gpio(self):
        for arguments in ([], ['--duration', '0'], ['--duration', 'nan'],
                          ['--duration', '1', '--mask', '1'], ['--duration', '1', '--step-us', '9'],
                          ['--duration', '1', '--seed', '0']):
            r = subprocess.run([str(self.binary), 'walk', '--dry-run', *arguments],
                               capture_output=True, text=True, timeout=5)
            self.assertEqual(r.returncode, 2)
            self.assertNotIn('/dev/gpiomem', r.stderr)


if __name__ == '__main__':
    unittest.main()
