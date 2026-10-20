"""Offline GPCLK0 sequence and sampled-clock metrology tests; no device use."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'dslcap' / 'tools'
spec = importlib.util.spec_from_file_location('check_clock', TOOLS / 'check_clock.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class GpclkDryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='gpclk-offline-')
        cls.binary = Path(cls.directory.name) / 'pi_gpclk'
        subprocess.run(['gcc', '-O2', '-Wall', '-Wextra', '-Werror', str(TOOLS / 'pi_gpclk.c'),
                        '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def dry(self, *args):
        # Every invocation, including invalid options, is explicitly dry-run.
        return subprocess.run([str(self.binary), '--dry-run', *args],
                              capture_output=True, text=True, timeout=5)

    def test_exact_start_stop_sequences(self):
        for source, source_id, divi, expected_hz in [('osc', 1, 2, 27_000_000), ('plld', 6, 75, 10_000_000)]:
            with self.subTest(source=source):
                r = self.dry('--source', source, '--divi', str(divi), '--duration', '0.2')
                self.assertEqual(r.returncode, 0, r.stderr)
                records = [json.loads(line) for line in r.stdout.splitlines()]
                header = records[0]
                self.assertTrue(header['dry_run'])
                self.assertIn('simulated', header['snapshot'])
                self.assertEqual(header['nominal_hz'], expected_hz)
                self.assertEqual(header['cm_page'], '0xfe101000')
                self.assertEqual(int(header['saved_ctl'], 0), 0)
                self.assertEqual(int(header['saved_div'], 0), 0)
                writes = [(e['register'], int(e['value'], 0)) for e in records if e['event'] == 'write']
                password = 0x5a000000
                self.assertEqual(writes, [
                    ('CM_GP0CTL', password | source_id),
                    ('CM_GP0DIV', password | divi << 12),
                    ('CM_GP0CTL', password | source_id),
                    ('CM_GP0CTL', password | source_id | 0x10),
                    ('GPIO_PUP_PDN_CNTRL_REG1', 0),
                    ('GPFSEL2', 2),
                    ('GPFSEL2', 0),
                    ('GPIO_PUP_PDN_CNTRL_REG1', 0x200),
                    ('CM_GP0CTL', password | source_id),
                    ('CM_GP0DIV', password),
                    ('CM_GP0CTL', password)])
                for event in records:
                    if event['event'] == 'write' and event['register'].startswith('CM_'):
                        self.assertEqual(event['address'], '0xfe101070' if event['register'] == 'CM_GP0CTL' else '0xfe101074')
                        self.assertEqual(int(event['value'], 0) >> 24, 0x5a)
                fallback = next(e for e in records if e['event'] == 'conditional_stop_fallback')
                self.assertEqual(int(fallback['kill_ctl'], 0), password | source_id | 0x20)
                self.assertEqual(int(fallback['clear_kill_ctl'], 0), password | source_id)
                self.assertTrue(fallback['restore_only_when_idle'])
                self.assertEqual(records[-2], dict(event='settle', milliseconds=1))
                self.assertEqual(records[-1], dict(event='readback', gpio20_function=0, gpio20_pull=2,
                                                  gpio20_level=0, ctl='0x00000000', div='0x00000000', matches=True))
                self.assertEqual([e['event'] for e in records], [
                    'header', 'write', 'wait_busy_clear', 'write', 'write', 'write',
                    'write', 'write', 'run', 'conditional_stop_fallback', 'write',
                    'write', 'write', 'wait_busy_clear', 'write', 'write', 'settle', 'readback'])

    def test_defaults_and_invalid_options(self):
        r = self.dry('--duration', '.1')
        header = json.loads(r.stdout.splitlines()[0])
        self.assertEqual((header['source'], header['divi']), ('osc', 2))
        for args in [[], ['--duration', '61'], ['--duration', '0'], ['--duration', 'nan'],
                     ['--duration', '1', '--divi', '1'], ['--duration', '1', '--divi', '4096'],
                     ['--duration', '1', '--divi', '-2'], ['--duration', '1', '--source', 'gpclk1']]:
            with self.subTest(args=args):
                bad = self.dry(*args)
                self.assertEqual(bad.returncode, 2, bad.stderr)
                self.assertEqual(bad.stdout, '')
                self.assertNotIn('/dev/', bad.stderr)


class ClockCheckerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='clock-raw-offline-')
        self.raw = Path(self.directory.name) / 'clock.raw'

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def waveform(rate, frequency=27_000_000, duty=.5, duration=.002):
        phase = np.remainder(np.arange(round(rate * duration), dtype=np.float64) * frequency / rate + .271, 1)
        return (phase < duty).astype(np.uint8)

    def save(self, bits, rate, width=1, channel=2, trigger=False):
        samples = bits.astype('<u2' if width == 2 else 'u1') << channel
        # Unrelated pins do not affect the channel measurement.
        samples |= 1
        self.raw.write_bytes(f'META samplerate: {rate}\n'.encode() +
                             (b'META trigger: none\n' if trigger else b'') + samples.tobytes())

    def cli(self, width=1, channel=2, *extra):
        return subprocess.run([sys.executable, str(TOOLS / 'check_clock.py'), str(self.raw),
                               '--unitsize', str(width), '--channel', str(channel),
                               '--expect-hz', '27000000', *extra], capture_output=True, text=True, timeout=15)

    def test_27mhz_all_samplerates_and_widths(self):
        for rate in (125_000_000, 250_000_000, 500_000_000, 1_000_000_000):
            for width, channel in ((1, 2), (2, 10)):
                with self.subTest(rate=rate, width=width):
                    bits = self.waveform(rate)
                    self.save(bits, rate, width, channel)
                    r = self.cli(width, channel)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    data = json.loads(r.stdout)
                    self.assertLess(abs(data['error_ppm']), 5)
                    # A rationally related sample grid can bias a nominal
                    # 50% raster (62/125 at 125M); report the actual samples.
                    sampled_duty = float(bits[data['first_rise']:data['last_rise']].mean())
                    self.assertAlmostEqual(data['duty'], sampled_duty, delta=1e-12)
                    self.assertAlmostEqual(data['duty'], .5, delta=.005)
                    self.assertEqual(data['glitches']['count'], 0)
                    self.assertGreaterEqual(data['window']['count'], 2)
                    self.assertIsNotNone(data['window']['std_ppm'])
                    self.assertGreater(data['rising_edges'], 50_000)
                    self.assertEqual(sum(data['period_histogram'].values()), data['complete_periods'])
                    self.assertEqual(sum(data['high_time_histogram'].values()), data['complete_periods'])
                    self.assertIsNone(data['within_tolerance'])

    def test_positive_100ppm_offset_and_tolerance(self):
        rate = 125_000_000
        self.save(self.waveform(rate, frequency=27_000_000 * 1.0001, duration=.02), rate)
        r = self.cli(1, 2, '--tolerance-ppm', '150')
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertAlmostEqual(data['error_ppm'], 100, delta=1)
        self.assertTrue(data['within_tolerance'])
        rejected = self.cli(1, 2, '--tolerance-ppm', '50')
        self.assertEqual(rejected.returncode, 1)
        self.assertFalse(json.loads(rejected.stdout)['within_tolerance'])

    def test_40_and_50_percent_duty(self):
        for duty in (.4, .5):
            self.save(self.waveform(500_000_000, duty=duty), 500_000_000, 2, 10)
            result = checker.measure(self.raw, 2, 10, 27_000_000)
            self.assertAlmostEqual(result['duty'], duty, delta=.001)

    def test_missing_pulse_and_glitch_are_reported_informationally(self):
        rate = 125_000_000
        bits = self.waveform(rate)
        rises = np.flatnonzero((bits[1:] == 1) & (bits[:-1] == 0)) + 1
        start = int(rises[500])
        end = start + int(np.flatnonzero(bits[start:] == 0)[0])
        bits[start:end] = 0  # Remove one complete pulse.
        self.save(bits, rate)
        r = self.cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        missing = json.loads(r.stdout)
        self.assertEqual(missing['period_anomalies']['long_periods'], 1)
        self.assertEqual(missing['period_anomalies']['first_sample'], int(rises[499]))
        clean = self.waveform(rate)
        low = next(i for i in range(100, 1000) if not np.any(clean[i - 1:i + 2]))
        clean[low] = 1
        self.save(clean, rate)
        r = self.cli(1, 2, '--min-run', '2')
        self.assertEqual(r.returncode, 0, r.stderr)
        glitch = json.loads(r.stdout)
        self.assertGreater(glitch['glitches']['count'], 0)
        self.assertIsNotNone(glitch['glitches']['first_sample'])
        self.assertIn('1', glitch['high_time_histogram'])
        self.assertGreater(glitch['period_anomalies']['short_periods'], 0)

    def test_chunk_boundary_pairing_and_trigger_header(self):
        self.save(self.waveform(125_000_000, duration=.00002), 125_000_000, 2, 10, trigger=True)
        expected = checker.measure(self.raw, 2, 10, 27_000_000)
        for chunk in (7, 32, 101):
            with mock.patch.object(checker, 'CHUNK_BYTES', chunk):
                got = checker.measure(self.raw, 2, 10, 27_000_000)
            self.assertEqual(got, expected)

    def test_malformed_and_insufficient_clock(self):
        for data in (b'', b'no META\n\0', b'META samplerate: 0\n\0',
                     b'META samplerate: 125000000\n\0'):
            self.raw.write_bytes(data)
            r = self.cli()
            self.assertEqual(r.returncode, 1)
        self.raw.write_bytes(b'META samplerate: 125000000\n\0\0\0')
        self.assertEqual(self.cli(2, 10).returncode, 1)
        self.save(self.waveform(125_000_000), 125_000_000)
        self.assertEqual(self.cli(1, 10).returncode, 1)
        self.assertEqual(self.cli(1, 2, '--tolerance-ppm', 'nan').returncode, 1)


if __name__ == '__main__':
    unittest.main()
