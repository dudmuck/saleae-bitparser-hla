import contextlib
import io
import os
import signal
import subprocess
import sys
import unittest
from unittest import mock

import sigrok_hla as hla
from test_dslcap_backend import arguments, PORTS, FIXTURE
from test_dslcap_wide import fixture, ChunkProducer, split_bytes


class TriggerPythonTests(unittest.TestCase):
    def options(self, **changes):
        values = dict(dsl_mode='buffer', trigger='15:r', samples='84')
        values.update(changes)
        return arguments(**values)

    def decode(self, data, options=None, error=None):
        out, err = io.StringIO(), io.StringIO()
        producer = ChunkProducer(data, error=error)
        with mock.patch.object(hla, 'CaptureProducer', return_value=producer), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            hla.run_sigrok_numpy(options or self.options(), PORTS, ['SPI'], 1e6, ['fake'])
        self.assertTrue(producer.finished)
        return out.getvalue(), err.getvalue()

    def test_named_trigger_union_and_width(self):
        args = self.options(channels='0=CLK,1=DI,2=DO,3=CS,15=IRQ', trigger='cs:f,irq:r',
                            trigger_pos=20, trigger_timeout='1ms', on_timeout='upload', drain_timeout='5')
        cmd = hla.build_dslcap_cmd(args, PORTS)
        self.assertEqual(cmd[cmd.index('--channels') + 1], '0,1,2,3,15')
        self.assertEqual(cmd[cmd.index('--trigger') + 1], '3:f,15:r')
        self.assertEqual(cmd[-8:], ['--trigger-pos', '20', '--trigger-timeout', '0.001s',
                                    '--on-timeout', 'upload', '--drain-timeout', '5s'])
        self.assertEqual(hla.dslcap_sample_unitsize(args, PORTS), 2)
        self.assertEqual(hla.dslcap_sample_unitsize(self.options(trigger='3:r'), PORTS), 1)

    def test_invalid_before_spawn(self):
        cases = [dict(dsl_mode='stream'), dict(continuous=True, samples=None), dict(samples=None),
                 dict(trigger='3:r,3:f'), dict(channels='3=CS', trigger='3:r,cs:f'), dict(trigger='16:r'),
                 dict(trigger='3:z'), dict(trigger='3:r,'), dict(trigger=''), dict(trigger_pos=91),
                 dict(trigger_pos=-1), dict(trigger_timeout='nan'), dict(trigger_timeout='-1'),
                 dict(drain_timeout='.5s'), dict(drain_timeout='3601'), dict(on_timeout='x'),
                 dict(samplerate='200M'), dict(samplerate='400M'), dict(samplerate='3M'),
                 dict(dslogic=False)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(self.options(**changes), PORTS)
        for key, value in (('trigger_pos', 0), ('trigger_timeout', '0'), ('on_timeout', 'fail')):
            with self.subTest(key=key), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(self.options(trigger=None, **{key: value}), PORTS)
        hla.build_dslcap_cmd(self.options(trigger='3:f', samplerate='400M'), PORTS)

    def test_exact_two_lines_every_split_and_binary_meta_prefix(self):
        for marker in (b'0', b'80', b'none'):
            raw = b'META trigger: 44\n\xff\x00'
            data = b'META samplerate: 2000000\nMETA trigger: ' + marker + b'\n' + raw
            for split in range(len(data) + 1):
                prefix = hla.MetaPrefix(2e6, required=True, expected_lines=2)
                result = prefix.feed(data[:split]) + prefix.feed(data[split:]) + prefix.feed(b'', eof=True)
                self.assertEqual(result, raw)
                self.assertEqual(prefix.trigger_sample, None if marker == b'none' else int(marker))
            prefix = hla.MetaPrefix(2e6, required=True, expected_lines=2)
            result = b''.join(prefix.feed(bytes([byte])) for byte in data)
            self.assertEqual(result, raw)
        prefix = hla.MetaPrefix(1e6, required=True)
        raw = b'META trigger: 123\n'
        self.assertEqual(prefix.feed(b'META samplerate: 1000000\n' + raw), raw)

    def test_bad_trigger_metadata_bounded(self):
        for tail in (b'', b'META trigger: ', b'raw', b'META trigger: -1\n',
                     b'META trigger: 01\n', b'META trigger: nan\n', b'META trigger: 18446744073709551616\n',
                     b'META trigger: ' + b'9' * 260, b'META trigger: NONE\n'):
            with self.subTest(tail=tail), self.assertRaises(hla.CaptureError):
                prefix = hla.MetaPrefix(1e6, required=True, expected_lines=2)
                prefix.feed(b'META samplerate: 1000000\n' + tail)
                prefix.feed(b'', eof=True)

    def test_markers_are_chronological_with_capture_start_times(self):
        data = b'META samplerate: 2000000\nMETA trigger: 80\n' + fixture.wide_samples()
        for sizes in ([1], [19, 5, 1], [len(data)]):
            with self.subTest(sizes=sizes):
                out, err = self.decode(split_bytes(data, sizes))
                self.assertIn('Trigger at sample 80 (t = 0.000040000 s)', err)
                self.assertIn('0.000017000: byte a5/3c', out)
                self.assertIn('0.000033000: byte 5a/c3', out)
                lines = out.splitlines()
                self.assertLess(next(i for i, line in enumerate(lines) if 'byte 5a/c3' in line),
                                next(i for i, line in enumerate(lines) if 'Trigger at sample' in line))
        out, err = self.decode([b'META samplerate: 2000000\nMETA trigger: none\n' + fixture.wide_samples()])
        self.assertIn('Untriggered capture (forced upload)', err)
        self.assertNotIn('Trigger at sample', out)

    def test_child_error_precedes_missing_line_two(self):
        for code, message in ((16, 'No trigger within --trigger-timeout'), (143, 'interrupted by signal 15')):
            cmd = [sys.executable, '-c', f'import sys; sys.stdout.buffer.write(b"META samplerate: 1000000\\n"); sys.exit({code})']
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                 self.assertRaisesRegex(hla.CaptureError, message) as caught:
                hla.run_sigrok_numpy(self.options(), PORTS, ['SPI'], 1e6, cmd)
            self.assertEqual(caught.exception.exit_code, code)
            self.assertNotIn('missing META', str(caught.exception))

    def test_future_marker_does_not_move_chunk_watermark(self):
        # A transaction begins before marker and finishes after first chunk.
        # Its start-stamped hex output must precede a future marker.
        data = fixture.wide_samples()
        out, _ = self.decode([b'META samplerate: 2000000\nMETA trigger: 80\n' + data[:90], data[90:]],
                             self.options(hex=True))
        self.assertLess(out.index('MOSI:'), out.index('Trigger at sample'))

if __name__ == '__main__':
    unittest.main()
