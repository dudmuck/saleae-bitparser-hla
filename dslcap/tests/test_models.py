"""Static mode API and pre-activation model fences, fake hardware only."""
import hashlib
from pathlib import Path
import json
import os
import signal
import time
import subprocess
import sys
import unittest

PROGRAM, FIRMWARE, RELOAD = sys.argv[1:4]
del sys.argv[1:]


def run(*args, scenario='u3-super', program=PROGRAM):
    return subprocess.run([program, '--fw-dir', FIRMWARE, *args],
                          env=dict(os.environ, DSLCAP_TEST_SCENARIO=scenario),
                          capture_output=True, text=True, errors="replace", timeout=10)


class Models(unittest.TestCase):
    def test_static_listing(self):
        r = run('--list-modes')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn('FAKE init', r.stderr)
        profiles = json.loads(r.stdout)['profiles']
        self.assertEqual([(p['model'], p['usb_speed']) for p in profiles],
                         [('plus', 'high'), ('u3pro16', 'high'), ('u3pro16', 'super')])
        for p in profiles:
            for m in p['modes']:
                self.assertTrue(m['name'])
                self.assertLessEqual(m['vld_num'], m['num'])
                self.assertEqual(m['validated'], p['model'] == 'plus' or m['stream'] or p['usb_speed'] == 'super')
        super_profile = profiles[-1]
        mode = next(m for m in super_profile['modes'] if m['max_rate'] == 1_000_000_000 and m['stream'])
        self.assertEqual((mode['num'], mode['vld_num']), (8, 3))
        self.assertIn(1_000_000_000, mode['rates'])
        self.assertEqual(len(json.loads(run('--list-modes', '--speed', 'super').stdout)['profiles']), 1)
        self.assertEqual(len(json.loads(run('--list-modes', '--model', 'plus').stdout)['profiles']), 1)
        bad = run('--list-modes', '--model', 'plus', '--speed', 'super')
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, '')
        self.assertIn('plus/super', bad.stderr)
        self.assertNotIn('FAKE init', bad.stderr)

    def test_phase3_fences(self):
        for model in ([], ['--model', 'u3pro16']):
            for extra in (['--mode', 'buffer', '--samples', '2048', '--serial-trigger',
                           'start=3:f,stop=3:r,clock=0:r,data=2,value=0x55,bits=8'],
                          ['--test-pattern'], ['--pattern']):
                with self.subTest(model=model, extra=extra):
                    r = run(*model, *extra)
                    self.assertEqual(r.returncode, 8, r.stderr)
                    self.assertIn('not validated on U3Pro16 (Phase 3c)', r.stderr)
                    self.assertNotIn('FAKE activation', r.stderr)

    def test_real_driver_trigger_images(self):
        golden = json.loads((Path(__file__).resolve().parents[1] /
                             'U3PRO16_TRIGGER_GOLDEN.json').read_text())
        program = str(Path(PROGRAM).with_name('dslcap_config_tests'))
        for case in golden['captures'][:4]:
            trigger = '0:1,11:r' if case['case'] == '4' else '3:f'
            # count is in units of 16 samples; all four are 1,000,448.
            samples = case['fields']['count'] * 16
            cmd = [program, '--model', 'u3pro16', '--mode', 'buffer',
                   '--samplerate', case['rate'], '--channels', case['channels'],
                   '--samples', str(samples), '--trigger', trigger,
                   '--trigger-pos', str(case['position_percent']), '--vth', '1.6']
            with self.subTest(case=case['case']):
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(result.stdout), 372)
                self.assertEqual(result.stdout, bytes.fromhex(case['raw_hex']))
                self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), case['sha256'])

    def test_plus_trigger_and_serial_images_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        program = str(Path(PROGRAM).with_name('dslcap_config_tests'))
        simple = json.loads((root / 'TRIGGER_GOLDEN.json').read_text())
        serial = json.loads((root / 'SERIAL_GOLDEN.json').read_text())
        for case in simple['captures']:
            fields = case['fields']
            mask = int(fields['ch_en_l'], 16)
            count = int(fields['cnt_l'], 16) | (int(fields['cnt_h'], 16) << 16)
            cmd = [program, '--mode', 'buffer', '--samplerate', str(case['rate']),
                   '--channels', ','.join(str(i) for i in range(16) if mask & (1 << i)),
                   '--samples', str(count * 16), '--trigger', '3:f', '--trigger-pos', '10', '--vth', '1.6']
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, bytes.fromhex(case['raw_hex']))
            self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), case['sha256'])
        for case in serial['comparisons']:
            result = subprocess.run([program, *case['command'][1:]], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, bytes.fromhex(case['register_hex']))
            self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), case['reference_sha256'])

    def test_u3_simple_triggers_and_lane_limits(self):
        for rate, channels, trigger in [('100M', '0-15', '3:f'),
                                        ('500M', '0-15', '0:1,11:r'),
                                        ('1G', '0-7', '3:f')]:
            for model in ([], ['--model', 'u3pro16']):
                r = run(*model, '--mode', 'buffer', '--samples', '2048',
                        '--samplerate', rate, '--channels', channels, '--trigger', trigger)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn('META trigger: 8', r.stdout)
        for lane in range(8, 16):
            r = run('--mode', 'buffer', '--samples', '2048', '--samplerate', '1G',
                    '--channels', f'0,{lane}', '--trigger', f'{lane}:r')
            self.assertEqual(r.returncode, 8, r.stderr)
            self.assertNotIn('FAKE activation', r.stderr)

    def test_u3_timeout_fail_and_short_forced_upload(self):
        args = ['--model', 'u3pro16', '--mode', 'buffer', '--samplerate', '500M',
                '--channels', '0-7', '--samples', '2048', '--trigger', '3:f',
                '--trigger-timeout', '0.1s']
        failed = run(*args, scenario='u3-trigger-timeout')
        self.assertEqual(failed.returncode, 16, failed.stderr)
        self.assertNotIn('META trigger:', failed.stdout)
        uploaded = subprocess.run([PROGRAM, '--fw-dir', FIRMWARE, *args, '--on-timeout', 'upload'],
            env=dict(os.environ, DSLCAP_TEST_SCENARIO='u3-trigger-force', DSLCAP_TEST_ACTUAL='1024'),
            capture_output=True, timeout=5)
        self.assertEqual(uploaded.returncode, 0, uploaded.stderr)
        header = b'META samplerate: 500000000\nMETA trigger: none\n'
        self.assertEqual(uploaded.stdout, header + bytes([0x55, 0xaa]) * 512)

    def test_u3_full_depth_pre_header_budgets_and_interrupt(self):
        for scenario in ('u3-trigger-post-hang', 'u3-trigger-forced-hang', 'u3-trigger-false-none'):
            with self.subTest(scenario=scenario):
                cmd = [PROGRAM, '--fw-dir', FIRMWARE, '--model', 'u3pro16',
                       '--mode', 'buffer', '--samplerate', '500M', '--channels', '0-15',
                       '--samples', '134217728', '--trigger', '3:f',
                       '--trigger-timeout', '0.1s', '--on-timeout', 'upload']
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=dict(os.environ, DSLCAP_TEST_SCENARIO=scenario))
                try:
                    # Old post-hit/force bounds expired at ~0.11/0.21 s.
                    # At scale .02 full-depth U3 bounds exceed 1.6 s.
                    time.sleep(.4)
                    self.assertIsNone(proc.poll(), 'full-depth upload allowance expired early')
                    proc.send_signal(signal.SIGINT)
                    _, err = proc.communicate(timeout=3)
                    self.assertEqual(proc.returncode, 130, err)
                    self.assertIn(b'FAKE cleanup', err)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.communicate()

    def test_reload_guard(self):
        r = run('--scan', program=RELOAD)
        self.assertEqual(r.returncode, 8, r.stderr)
        self.assertIn('refusing DSLogic U3Pro16', r.stderr)
        self.assertNotIn('FAKE activation', r.stderr)

    def test_explicit_phase3_before_generic_mode_error(self):
        r = run('--model', 'u3pro16', '--trigger', '3:f', '--samples', '2048')
        self.assertEqual(r.returncode, 2)
        self.assertNotIn('FAKE init', r.stderr)

    def test_stream_rates_and_selected_line(self):
        for rate, channels in [('125M', '0-15'), ('250M', '0-11'),
                               ('250M', '0-5,8-13'), ('500M', '0-5'),
                               ('500M', '0,2,5'), ('1G', '0,2,5')]:
            r = run('--samplerate', rate, '--channels', channels, '--samples', '73')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stderr.count('dslcap: selected 2a0e:002a DSLogic U3Pro16 speed=super'), 1)
        r = run('--model', 'u3pro16', '--samplerate', '1G', '--channels', '8', '--samples', '73')
        self.assertEqual(r.returncode, 8)
        self.assertNotIn('FAKE init', r.stderr)

    def test_untriggered_buffer_rates_depth_and_full_upload(self):
        for rate in ('1M', '2M', '4M', '5M', '10M', '20M', '25M', '50M', '100M', '125M', '250M', '500M', '1G'):
            channels = '0-7' if rate == '1G' else '0-15'
            r = run('--model', 'u3pro16', '--mode', 'buffer', '--samplerate', rate,
                    '--channels', channels, '--samples', '73')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('samples=73', r.stderr)
            self.assertNotIn('META trigger:', r.stdout)
        for rate, channels, depth in [('500M', '0-15', 134217728), ('1G', '0-7', 268435456)]:
            args = ['--model', 'u3pro16', '--mode', 'buffer', '--samplerate', rate,
                    '--channels', channels, '--samples']
            # Real 2-Gibit model limits; discard fake stdout instead of storing
            # a 256 MiB Python string. The frontend/ring/converter all run.
            r = subprocess.run([PROGRAM, '--fw-dir', FIRMWARE, *args, str(depth)],
                env=dict(os.environ, DSLCAP_TEST_SCENARIO='u3-super'), stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f'samples={depth}', r.stderr)
            header_size = len(f'META samplerate: {500000000 if rate == "500M" else 1000000000}\n')
            self.assertIn(f'output bytes={268435456 + header_size}', r.stderr)
            bad = run(*args, str(depth + 1))
            self.assertEqual(bad.returncode, 8, bad.stderr)
            self.assertIn('exceeds channel hardware depth', bad.stderr)
        r = run('--model', 'u3pro16', '--mode', 'buffer', '--samplerate', '1G',
                '--channels', '8', '--samples', '73')
        self.assertEqual(r.returncode, 8)
        self.assertNotIn('FAKE init', r.stderr)
        r = run('--model', 'u3pro16', '--mode', 'buffer', '--continuous')
        self.assertEqual(r.returncode, 2)
        self.assertIn('continuous buffer mode is unsupported', r.stderr)


if __name__ == '__main__':
    unittest.main()
