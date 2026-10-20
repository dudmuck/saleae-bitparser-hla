"""Static mode API and pre-activation model fences, fake hardware only."""
import json
import os
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
                self.assertEqual(m['validated'], p['model'] == 'plus' or m['stream'])
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
            for extra in (['--mode', 'buffer', '--samples', '2048'],
                          ['--mode', 'buffer', '--samples', '2048', '--trigger', '3:f'],
                          ['--mode', 'buffer', '--samples', '2048', '--serial-trigger',
                           'start=3:f,stop=3:r,clock=0:r,data=2,value=0x55,bits=8'],
                          ['--test-pattern'], ['--pattern']):
                with self.subTest(model=model, extra=extra):
                    r = run(*model, *extra)
                    self.assertEqual(r.returncode, 8, r.stderr)
                    self.assertIn('not validated on U3Pro16 (Phase 3)', r.stderr)
                    self.assertNotIn('FAKE activation', r.stderr)

    def test_reload_guard(self):
        r = run('--scan', program=RELOAD)
        self.assertEqual(r.returncode, 8, r.stderr)
        self.assertIn('refusing DSLogic U3Pro16', r.stderr)
        self.assertNotIn('FAKE activation', r.stderr)

    def test_explicit_phase3_before_generic_mode_error(self):
        r = run('--model', 'u3pro16', '--trigger', '3:f', '--samples', '2048')
        self.assertEqual(r.returncode, 8)
        self.assertIn('not validated on U3Pro16 (Phase 3)', r.stderr)
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


if __name__ == '__main__':
    unittest.main()
