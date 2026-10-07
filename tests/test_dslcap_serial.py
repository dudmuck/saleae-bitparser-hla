import contextlib
import io
import sys
import unittest
from unittest import mock

import sigrok_hla as hla
from test_dslcap_backend import arguments, PORTS
from test_dslcap_wide import fixture, ChunkProducer, split_bytes

SPEC = 'start=3:f,stop=3:r,clock=0:r,data=2,value=0x1c35,bits=16'

class SerialPythonTests(unittest.TestCase):
    def options(self, **changes):
        values = dict(dsl_mode='buffer', serial_trigger=SPEC, samples='84')
        values.update(changes)
        return arguments(**values)

    def test_named_fields_normalization_union_and_high_bit_width(self):
        args = self.options(channels='0=CLK,1=DI,2=DO,3=CS,15=DATA',
            serial_trigger='bits=16,value=0b0001110000110101,data=DATA,clock=clk:R,stop=cs:r,start=CS:f',
            trigger_pos=10, trigger_timeout='2ms', on_timeout='upload')
        cmd = hla.build_dslcap_cmd(args, PORTS)
        self.assertEqual(cmd[cmd.index('--serial-trigger') + 1],
            'start=3:f,stop=3:r,clock=0:R,data=15,value=0x1c35,bits=16')
        self.assertEqual(cmd[cmd.index('--channels') + 1], '0,1,2,3,15')
        self.assertNotIn('--trigger', cmd)
        self.assertEqual(hla.dslcap_sample_unitsize(args, PORTS), 2)
        self.assertEqual(hla.dslcap_sample_unitsize(self.options(), PORTS), 1)

    def test_every_role_enters_union_and_fast_lane_checks(self):
        for field, new in (('start=3:f', 'start=15:f'), ('stop=3:r', 'stop=15:r'),
                           ('clock=0:r', 'clock=15:r'), ('data=2', 'data=15')):
            with self.subTest(field=field):
                args = self.options(serial_trigger=SPEC.replace(field, new))
                self.assertEqual(hla.dslcap_sample_unitsize(args, PORTS), 2)
                cmd = hla.build_dslcap_cmd(args, PORTS)
                self.assertEqual(cmd[cmd.index('--channels') + 1], '0,1,2,3,15')
                for rate in ('200M', '400M'):
                    with self.assertRaises(SystemExit):
                        hla.build_dslcap_cmd(self.options(serial_trigger=args.serial_trigger, samplerate=rate), PORTS)
        hla.build_dslcap_cmd(self.options(samplerate='400M'), PORTS)

    def test_invalid_serial_before_producer_launch(self):
        specifications = ['', SPEC + ',', SPEC + ',bits=16', SPEC.replace('stop=3:r,', ''),
            SPEC.replace('clock=0:r', 'clock=0:e'), SPEC.replace('clock=0:r', 'clock=0:h'),
            SPEC.replace('value=0x1c35', 'value=7221'), SPEC.replace('value=0x1c35', 'value=0b102'),
            SPEC.replace('value=0x1c35', 'value=0x10000'), SPEC.replace('value=0x1c35', 'value=0x'),
            SPEC.replace('bits=16', 'bits=8'), SPEC.replace('bits=16', 'bits=0'),
            SPEC.replace('bits=16', 'bits=17'), SPEC.replace('bits=16', 'bits=1.1'),
            SPEC.replace('start=3:f', 'start=16:f'), SPEC.replace('data=2', 'data=2:r'),
            SPEC.replace('data=2', 'data=unknown'), SPEC + ',other=1', SPEC.replace('data=2', 'data=2=3')]
        for specification in specifications:
            with self.subTest(specification=specification), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(self.options(serial_trigger=specification), PORTS)
        for changes in (dict(trigger='3:r'), dict(dsl_mode='stream'), dict(continuous=True),
                         dict(dslogic=False), dict(samples=None), dict(trigger_pos=91)):
            with self.subTest(changes=changes), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(self.options(**changes), PORTS)

    def test_short_binary_zero_and_conditions(self):
        for value, bits in (('0b1', '01'), ('0x0', '1'), ('0X8A', '8')):
            args = self.options(serial_trigger=f'start=3:h,stop=3:l,clock=0:F,data=2,value={value},bits={bits}')
            cmd = hla.build_dslcap_cmd(args, PORTS)
            normalized = cmd[cmd.index('--serial-trigger') + 1]
            self.assertIn(f'bits={int(bits)}', normalized)
            self.assertIn(f'value=0x{int(value, 0):x}', normalized)

    def decode(self, chunks, error=None):
        args = self.options(serial_trigger=SPEC.replace('data=2', 'data=15'))
        out, err = io.StringIO(), io.StringIO()
        producer = ChunkProducer(chunks, error=error)
        with mock.patch.object(hla, 'CaptureProducer', return_value=producer), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            hla.run_sigrok_numpy(args, PORTS, ['SPI'], 1e6, ['fake-serial'])
        self.assertTrue(producer.finished)
        return out.getvalue(), err.getvalue()

    def test_serial_two_line_meta_every_fragment_width_and_chronology(self):
        wire = b'META samplerate: 2000000\nMETA trigger: 80\n' + fixture.wide_samples()
        for sizes in ([1], [17, 3, 1], [len(wire)]):
            with self.subTest(sizes=sizes):
                out, err = self.decode(split_bytes(wire, sizes))
                self.assertIn('Trigger at sample 80 (t = 0.000040000 s)', err)
                self.assertIn('0.000017000: byte a5/3c', out)
                self.assertIn('0.000033000: byte 5a/c3', out)
                self.assertLess(out.index('byte 5a/c3'), out.index('Trigger at sample'))
        out, err = self.decode([b'META samplerate: 2000000\nMETA trigger: none\n' + fixture.wide_samples()])
        self.assertIn('Untriggered capture (forced upload)', err)
        self.assertNotIn('Trigger at sample', out)

    def test_serial_producer_error_precedes_missing_metadata(self):
        args = self.options()
        for status in (16, 143):
            cmd = [sys.executable, '-c', f'import sys; sys.stdout.buffer.write(b"META samplerate: 1000000\\n"); sys.exit({status})']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(hla.CaptureError) as caught:
                hla.run_sigrok_numpy(args, PORTS, ['SPI'], 1e6, cmd)
            self.assertEqual(caught.exception.exit_code, status)
            self.assertNotIn('missing', str(caught.exception))
        with self.assertRaisesRegex(hla.CaptureError, 'META trigger'):
            self.decode([b'META samplerate: 2000000\n'])

if __name__ == '__main__':
    unittest.main()
