# SPDX-License-Identifier: GPL-3.0-or-later
"""Real C frontend/fake-driver serial state and shared lifecycle contracts."""
import os
import re
import subprocess
import sys
import unittest

PROGRAM, FIRMWARE = sys.argv[1:3]
sys.argv[1:] = []
SPEC = 'start=3:f,stop=3:r,clock=0:r,data=2,value=0x1c35,bits=16'

class SerialContract(unittest.TestCase):
    def capture(self, specification=SPEC, scenario='trigger-burst', extra=(), expected=0):
        cmd = [PROGRAM, '--fw-dir', FIRMWARE, '--mode', 'buffer', '--channels', '0-7',
               '--samplerate', '100M', '--samples', '2048', '--serial-trigger', specification,
               '--trigger-timeout', '0.1s', *extra]
        run = subprocess.run(cmd, env=dict(os.environ, DSLCAP_TEST_SCENARIO=scenario),
                             capture_output=True, timeout=4)
        self.assertEqual(run.returncode, expected, run.stderr.decode(errors='replace'))
        if expected in (2, 8) and b'FAKE init' not in run.stderr:
            self.assertEqual(run.stdout, b'')
        return run

    def state(self, run):
        matches = re.findall(rb'FAKE serial stage=(\d) value0=(\w+) value1=(\w+) logic=(\d) inv=(\d)/(\d) count=(\d+)/(\d+)', run.stderr)
        self.assertEqual(len(matches), 4, run.stderr)
        return [(int(stage), first.decode(), second.decode(), int(logic), int(inv0), int(inv1), int(count0), int(count1))
                for stage, first, second, logic, inv0, inv1, count0, count1 in matches]

    def test_asymmetric_state_and_packet_order(self):
        run = self.capture()
        # Independent source-derived DSView probe strings, highest bit first.
        self.assertEqual(self.state(run), [
            (0, 'XXXXXXXXXXXXFXXX', 'XXXXXXXXXXXXRXXX', 1, 0, 0, 0, 0),
            (1, 'XXXXXXXXXXXXXXXR', 'XXXXXXXXXXXXXXXX', 1, 0, 0, 1, 0),
            (2, 'XXXXXXXXXXXXX0XX', 'XXXXXXXXXXXXXXXX', 1, 0, 0, 0, 0),
            (3, '0001110000110101', 'XXXXXXXXXXXXXXXX', 1, 0, 0, 15, 0)])
        self.assertEqual(run.stdout, b'META samplerate: 100000000\nMETA trigger: 8\n' + bytes([0x55, 0xaa])*1024)
        self.assertIn(b'MSB-first assumed', run.stderr)

    def test_bit_byte_order_variants_have_distinct_source_state(self):
        for value, expected in (("0x1c35", "0001110000110101"), ("0xac38", "1010110000111000"),
                                ("0x351c", "0011010100011100"), ("0x38ac", "0011100010101100")):
            with self.subTest(value=value):
                self.assertEqual(self.state(self.capture(SPEC.replace('0x1c35', value)))[3][1], expected)

    def test_high_serial_data_role_physical_width(self):
        run = self.capture(SPEC.replace('data=2', 'data=15'), extra=('--channels', '0-3,15'))
        self.assertEqual(self.state(run)[2][1], '0XXXXXXXXXXXXXXX')
        self.assertEqual(run.stdout, b'META samplerate: 100000000\nMETA trigger: 8\n' +
                         bytes([5, 128, 10, 0])*1024)

    def test_binary_short_value_unused_upper_bits_and_reordered_fields(self):
        for specification, expected in (
                ('bits=8,value=0b10001010,data=2,clock=0:F,stop=3:R,start=3:F', 'XXXXXXXX10001010'),
                ('start=3:h,stop=3:l,clock=0:f,data=2,value=0x0,bits=1', 'XXXXXXXXXXXXXXX0'),
                ('start=3:e,stop=3:r,clock=0:r,data=2,value=0B1,bits=01', 'XXXXXXXXXXXXXXX1')):
            with self.subTest(specification=specification):
                stages = self.state(self.capture(specification))
                self.assertEqual(stages[3][1], expected)
                self.assertEqual(stages[1][6], 1)
                self.assertEqual(stages[3][6], len(expected.replace('X', '')) - 1)

    def test_each_serial_role_is_captured_and_physical_lane_bounds(self):
        for role, replacement in (('start=3:f', 'start=8:f'), ('stop=3:r', 'stop=8:r'),
                                  ('clock=0:r', 'clock=8:r'), ('data=2', 'data=8')):
            with self.subTest(role=role):
                run = self.capture(SPEC.replace(role, replacement), expected=2)
                self.assertNotIn(b'FAKE init', run.stderr)
        for rate, channels in (('200M', '0-15'), ('400M', '0-7')):
            run = self.capture(extra=('--samplerate', rate, '--channels', channels), expected=8)
            self.assertNotIn(b'FAKE init', run.stderr)
        self.capture(extra=('--samplerate', '400M', '--channels', '0-3'))

    def test_invalid_fields_prefixes_counts_and_edges_before_hardware(self):
        cases = ['', SPEC + ',', SPEC + ',bits=16', SPEC.replace('stop=3:r,', ''),
                 SPEC.replace('clock=0:r', 'clock=0:h'), SPEC.replace('clock=0:r', 'clock=0:e'),
                 SPEC.replace('data=2', 'data=2:r'), SPEC.replace('bits=16', 'bits=0'),
                 SPEC.replace('bits=16', 'bits=17'), SPEC.replace('bits=16', 'bits=1.0'),
                 SPEC.replace('value=0x1c35', 'value=7221'), SPEC.replace('value=0x1c35', 'value=0x'),
                 SPEC.replace('value=0x1c35', 'value=0b102'), SPEC.replace('value=0x1c35', 'value=-0x1'),
                 SPEC.replace('value=0x1c35', 'value=0x10000'), SPEC.replace('bits=16', 'bits=8'),
                 SPEC.replace('start=3:f', 'start=16:f'), SPEC.replace('stop=3:r', 'stop=-1:r'),
                 SPEC.replace('clock=0:r', 'clock=CLK:r'), SPEC + ',other=1', SPEC.replace('data=2', 'data=2=3'),
                 'start=3:f,,stop=3:r,clock=0:r,data=2,value=0x1,bits=1']
        for specification in cases:
            with self.subTest(specification=specification):
                run = self.capture(specification, expected=2)
                self.assertNotIn(b'FAKE init', run.stderr)
        for extra in (('--trigger', '3:r'), ('--serial-trigger', SPEC), ('--mode', 'stream'),
                      ('--continuous',), ('--test-pattern',), ('--t0', 'trigger')):
            run = self.capture(extra=extra, expected=2)
            self.assertNotIn(b'FAKE init', run.stderr)

    def test_serial_configuration_failures_do_not_arm(self):
        for kind in ('value', 'logic', 'inv', 'count'):
            run = self.capture(scenario=f'serial-config-{kind}-fail', expected=8)
            self.assertEqual(run.stdout, b'')
            self.assertNotIn(b'FAKE serial stage=', run.stderr) # enable was never called
            self.assertIn(b'FAKE cleanup', run.stderr)

    def test_shared_serial_timeout_forced_metadata_and_errors(self):
        for scenario, code, marker in (('trigger-grace-header', 0, b'8'),
                 ('trigger-timeout', 16, None), ('trigger-late-header', 16, None),
                 ('trigger-force', 0, b'none'), ('trigger-force-hit', 0, b'8'),
                 ('trigger-false-valid', 0, b'8'), ('trigger-force-no-header', 13, None),
                 ('trigger-null', 13, None), ('trigger-bad-id', 13, None),
                 ('trigger-duplicate', 13, b'8'), ('trigger-upload-detach', 11, b'8'),
                 ('trigger-signal-end', 143, b'8')):
            with self.subTest(scenario=scenario):
                run = self.capture(scenario=scenario, extra=('--on-timeout', 'upload')
                    if scenario.startswith(('trigger-force', 'trigger-false')) else (), expected=code)
                first = b'META samplerate: 100000000\n'
                if code in (11, 143):
                    # Cancellation may discard queued metadata/data before writer
                    # progress. Already-written bytes must be an exact stream prefix.
                    stream = first + b'META trigger: 8\n' + bytes([0x55, 0xaa])*1024
                    self.assertTrue(stream.startswith(run.stdout), run.stdout)
                    continue
                self.assertTrue(run.stdout.startswith(first))
                if marker is None:
                    self.assertEqual(run.stdout, first)
                else:
                    self.assertTrue(run.stdout.startswith(first + b'META trigger: ' + marker + b'\n'))

if __name__ == '__main__':
    unittest.main()
