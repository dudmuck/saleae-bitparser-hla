# SPDX-License-Identifier: GPL-3.0-or-later
"""Subprocess contracts against real capture/control/ring and fake DSView calls."""
import os
import signal
import subprocess
import sys
import time
import unittest

PROGRAM, FIRMWARE = sys.argv[1:3]
sys.argv[1:] = []

class TriggerContract(unittest.TestCase):
    def command(self, *extra):
        extra = list(extra)
        samples = '2048'
        if '--samples' in extra:
            at = extra.index('--samples')
            samples = extra[at + 1]
            del extra[at:at + 2]
        return [PROGRAM, '--fw-dir', FIRMWARE, '--mode', 'buffer', '--channels', '0-7',
                '--samplerate', '100M', '--samples', samples, '--trigger', '3:f',
                '--trigger-timeout', '0.1s', *extra]

    def capture(self, scenario='trigger-burst', actual=None, extra=(), expected=0):
        env = dict(os.environ, DSLCAP_TEST_SCENARIO=scenario)
        if '--samples' in extra and extra[extra.index('--samples') + 1] == '1':
            env['DSLCAP_TEST_POS'] = '0'
        if actual is not None:
            env['DSLCAP_TEST_ACTUAL'] = str(actual)
        start = time.monotonic()
        run = subprocess.run(self.command(*extra), env=env, capture_output=True, timeout=4)
        self.assertEqual(run.returncode, expected, run.stderr.decode(errors='replace'))
        self.assertLess(time.monotonic() - start, 3)
        self.assertIn(b'FAKE cleanup', run.stderr)
        return run

    def assert_data(self, run, samples, marker=b'8'):
        header = b'META samplerate: 100000000\nMETA trigger: ' + marker + b'\n'
        self.assertEqual(run.stdout, header + bytes([0x55, 0xaa]) * (samples // 2) +
                         (b'\x55' if samples % 2 else b''))

    def test_header_callback_burst_and_owned_copy(self):
        self.assert_data(self.capture(), 2048)
        self.assert_data(self.capture(extra=('--trigger-timeout', '18000000000s')), 2048)
        self.assert_data(self.capture('trigger-grace-header'), 2048)
        self.assert_data(self.capture('trigger-hit-grace'), 2048)
        run = self.capture('trigger-status-fail-header')
        self.assert_data(run, 2048)
        self.assertIn(b'freshness unknown', run.stderr)
        self.assertRegex(run.stderr, rb'cached status failures=[1-9][0-9]*')

    def test_header_failures_and_order(self):
        cases = ('trigger-null', 'trigger-bad-id', 'trigger-packet-status', 'trigger-missing',
                 'trigger-logic-before', 'trigger-duplicate', 'trigger-pos-outside', 'trigger-no-end')
        for scenario in cases:
            with self.subTest(scenario=scenario):
                run = self.capture(scenario, expected=13)
                if scenario not in ('trigger-duplicate', 'trigger-no-end'):
                    self.assertEqual(run.stdout, b'META samplerate: 100000000\n')
        for actual in (0, 1023, 2049):
            with self.subTest(actual=actual):
                self.capture(actual=actual, expected=13)

    def test_timeout_commit_discards_late_header_and_logic(self):
        for scenario in ('trigger-timeout', 'trigger-late-header', 'trigger-status-fail'):
            with self.subTest(scenario=scenario):
                run = self.capture(scenario, expected=16)
                self.assertEqual(run.stdout, b'META samplerate: 100000000\n')
                self.assertIn(b'No trigger within --trigger-timeout', run.stderr)

    def test_force_upload_true_false_and_short_counts(self):
        self.assert_data(self.capture('trigger-force', 1024, ('--on-timeout', 'upload')), 1024, b'none')
        self.assert_data(self.capture('trigger-force-hit', 1024, ('--on-timeout', 'upload')), 1024)
        self.assert_data(self.capture('trigger-false-valid', extra=('--on-timeout', 'upload')), 2048)
        for scenario, code in (('trigger-force-no-header', 13), ('trigger-false-none', 11),
                               ('trigger-false-end', 13), ('trigger-false-short', 13)):
            with self.subTest(scenario=scenario):
                self.capture(scenario, actual=1024, extra=('--on-timeout', 'upload'), expected=code)

    def test_alignment_and_requested_trim(self):
        for requested in (1, 1023, 1024, 1025):
            aligned = (requested + 1023) & ~1023
            marker = b'0' if requested == 1 else b'8'
            for actual in (1024, aligned):
                with self.subTest(requested=requested, actual=actual):
                    run = self.capture('trigger-force-hit', actual,
                        ('--samples', str(requested), '--on-timeout', 'upload'))
                    self.assert_data(run, min(actual, requested), marker)
            if aligned > 1024:
                self.capture('trigger-burst', 1024, ('--samples', str(requested)), expected=13)

    def test_bounded_phases_and_device_precedence(self):
        for scenario in ('trigger-post-hang', 'trigger-upload-hang'):
            with self.subTest(scenario=scenario):
                self.capture(scenario, expected=13)
        for scenario in ('trigger-wait-detach', 'trigger-post-detach', 'trigger-forced-detach', 'trigger-upload-detach'):
            with self.subTest(scenario=scenario):
                self.capture(scenario, extra=('--on-timeout', 'upload'), expected=11)
        for scenario in ('trigger-post-data', 'trigger-forced-data', 'trigger-upload-data'):
            self.capture(scenario, extra=('--on-timeout', 'upload'), expected=13)
        self.capture('trigger-signal-end', expected=143)
        self.capture('trigger-signal-cleanup', expected=143)

    def test_signal_each_phase(self):
        for scenario, delay in (('trigger-wait-hang', .02), ('trigger-post-hang', .02),
                                 ('trigger-forced-hang', .03), ('trigger-upload-hang', .02)):
            with self.subTest(scenario=scenario):
                env = dict(os.environ, DSLCAP_TEST_SCENARIO=scenario)
                cmd = self.command('--on-timeout', 'upload')
                if scenario == 'trigger-wait-hang':
                    cmd = [value for value in cmd] # use long deadline so this remains WAITING
                    cmd[cmd.index('--trigger-timeout') + 1] = '100s'
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(delay)
                proc.send_signal(signal.SIGTERM)
                out, err = proc.communicate(timeout=3)
                self.assertEqual(proc.returncode, 143, err)
                self.assertIn(b'META samplerate:', out)

    def test_buffer_drain_progress_stall_and_signal(self):
        # Untriggered buffer remains exactly one META line; >2s progress must finish.
        cmd = [PROGRAM, '--fw-dir', FIRMWARE, '--mode', 'buffer', '--channels', '0-7',
               '--samplerate', '100M', '--samples', '262144', '--drain-timeout', '1']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunks = []
        while True:
            data = proc.stdout.read(8192)
            if not data:
                break
            chunks.append(data)
            time.sleep(.08)
        err = proc.stderr.read()
        self.assertEqual(proc.wait(timeout=3), 0, err)
        self.assertEqual(b''.join(chunks), b'META samplerate: 100000000\n' + bytes([0x55, 0xaa]) * 131072)
        proc.stdout.close()
        proc.stderr.close()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1.3)
        out, err = proc.communicate(timeout=3)
        self.assertEqual(proc.returncode, 14, err)
        self.assertLess(len(out), 262144)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(.1)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=3)
        self.assertEqual(proc.returncode, 130, err)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, DSLCAP_TEST_SCENARIO='capture-drain-detach'))
        time.sleep(.4)
        _, err = proc.communicate(timeout=3)
        self.assertEqual(proc.returncode, 11, err)

    def test_cli_rejections_before_driver_init(self):
        for args in (('--mode', 'stream'), ('--continuous',), ('--test-pattern',),
                     ('--trigger', '3:r,3:f'), ('--trigger', '16:r'), ('--trigger', '8:f'),
                     ('--trigger', '3:bad'), ('--trigger-pos', '91'), ('--trigger-timeout', '-1'),
                     ('--on-timeout', 'x'), ('--drain-timeout', '0'), ('--drain-timeout', '3601'),
                     ('--trigger-serial', 'x'), ('--t0', 'trigger')):
            with self.subTest(args=args):
                run = subprocess.run(self.command(*args), capture_output=True, timeout=3)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertNotIn(b'FAKE init', run.stderr)
        for modifier in ('--trigger-pos', '--trigger-timeout', '--on-timeout'):
            value = 'fail' if modifier == '--on-timeout' else '1'
            cmd = self.command()
            del cmd[cmd.index('--trigger'):cmd.index('--trigger') + 2]
            del cmd[cmd.index('--trigger-timeout'):cmd.index('--trigger-timeout') + 2]
            run = subprocess.run(cmd + [modifier, value], capture_output=True)
            self.assertEqual(run.returncode, 2, run.stderr)

if __name__ == '__main__':
    unittest.main()
