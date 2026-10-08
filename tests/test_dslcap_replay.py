"""Capture-then-decode: --raw-out writes dslcap output verbatim, -i replays it."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigrok_hla as hla
from test_dslcap_backend import CMD, PORTS, arguments


def produce(scenario='fragmented'):
    env = dict(os.environ, DSLCAP_FIXTURE_SCENARIO=scenario)
    return subprocess.run(CMD, env=env, capture_output=True, check=True).stdout


def replay_args(path, **changes):
    values = dict(input_file=str(path), samples=None, raw_out=None, dsl_triggered=False)
    values.update(changes)
    return arguments(**values)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'cap.raw'

    def tearDown(self):
        self.dir.cleanup()

    def decode(self, args, ports=PORTS, names=('SPI',)):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            hla.validate_backend_args(args, ports)
            hla.run_sigrok_numpy(args, ports, list(names), 1e6)
        return out.getvalue(), err.getvalue()

    def live(self, scenario='fragmented', **changes):
        out = io.StringIO()
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO=scenario), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            hla.run_sigrok_numpy(arguments(**changes), PORTS, ['SPI'], 1e6, CMD)
        return out.getvalue()

    def test_replay_matches_live_decode(self):
        self.path.write_bytes(produce())
        replayed, err = self.decode(replay_args(self.path, int_pin='4', hex=True))
        self.assertEqual(replayed, self.live(int_pin='4', hex=True))
        self.assertIn('byte a5/3c', replayed)
        self.assertIn('Replaying dslcap raw file', err)

    def test_replay_rejects_capture_options(self):
        self.path.write_bytes(produce())
        for change in (dict(samples='76'), dict(time='1s'), dict(continuous=True), dict(vth=1.2),
                       dict(dsl_mode='buffer'), dict(trigger='3:f'), dict(raw_out='x'),
                       dict(input_format='binary'), dict(transform='deglitch')):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                hla.validate_backend_args(replay_args(self.path, **change), PORTS)
        with self.assertRaises(SystemExit):
            hla.build_dslcap_cmd(replay_args(self.path), PORTS)
        with self.assertRaises(SystemExit):
            hla.validate_backend_args(arguments(dsl_triggered=True), PORTS)
        with self.assertRaises(SystemExit):
            hla.validate_backend_args(arguments(dslogic=False, raw_out='x', driver='fx2lafw'), PORTS)

    def test_replay_bad_meta_and_truncation(self):
        self.path.write_bytes(b'raw without meta')
        with self.assertRaises(hla.CaptureError):
            self.decode(replay_args(self.path))
        wide = Path(self.dir.name) / 'wide.raw'
        wide.write_bytes(b'META samplerate: 1000000\n\x08\x00\x08')
        with self.assertRaisesRegex(hla.CaptureError, 'one byte remains'):
            self.decode(replay_args(wide), ports=[hla.parse_spi_port('8,9,10,11')])

    def test_sidecar_checks_channels_and_sets_width(self):
        # A subset mapping (one bus of a wide capture) still reads the file's width.
        record = dict(format='dslcap-raw-v1', channels=[0, 1, 2, 3, 8, 9, 10, 11], unitsize=2, meta_lines=1, exit=0)
        Path(str(self.path) + '.json').write_text(json.dumps(record))
        self.assertEqual(hla.dslcap_replay_layout(replay_args(self.path), PORTS), (2, 1))
        with self.assertRaisesRegex(SystemExit, 'not captured'):
            hla.dslcap_replay_layout(replay_args(self.path, int_pin='5'), PORTS)
        with self.assertRaisesRegex(SystemExit, 'not captured with a trigger'):
            hla.dslcap_replay_layout(replay_args(self.path, dsl_triggered=True), PORTS)
        Path(str(self.path) + '.json').write_text(json.dumps(dict(format='other')))
        with self.assertRaises(SystemExit):
            hla.dslcap_replay_layout(replay_args(self.path), PORTS)

    def test_triggered_file_needs_two_meta_lines(self):
        data = produce()
        newline = data.index(b'\n') + 1
        self.path.write_bytes(data[:newline] + b'META trigger: 12\n' + data[newline:])
        out, err = self.decode(replay_args(self.path, dsl_triggered=True))
        self.assertIn('Trigger at sample 12', out)
        self.assertIn('byte a5/3c', out)

    def test_raw_out_writes_verbatim_and_replays(self):
        args = arguments(raw_out=str(self.path), dsl_triggered=False)
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='fragmented'), \
             mock.patch.object(hla, 'build_dslcap_cmd', return_value=CMD), \
             contextlib.redirect_stderr(io.StringIO()):
            hla.run_dslcap_raw_out(args, PORTS)
        self.assertEqual(self.path.read_bytes(), produce())
        record = json.loads(Path(str(self.path) + '.json').read_text())
        self.assertEqual((record['format'], record['exit'], record['bytes'], record['unitsize'], record['meta_lines']),
                         ('dslcap-raw-v1', 0, self.path.stat().st_size, 1, 1))
        self.assertEqual(record['channels'], [0, 1, 2, 3])
        replayed, _ = self.decode(replay_args(self.path))
        self.assertIn('byte a5/3c', replayed)

    def test_raw_out_refuses_overwrite_and_reports_failure(self):
        self.path.write_bytes(b'keep')
        with self.assertRaises(SystemExit):
            hla.run_dslcap_raw_out(arguments(raw_out=str(self.path)), PORTS)
        self.assertEqual(self.path.read_bytes(), b'keep')
        failed = Path(self.dir.name) / 'failed.raw'
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='failure'), \
             mock.patch.object(hla, 'build_dslcap_cmd', return_value=CMD), \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(hla.CaptureError) as caught:
            hla.run_dslcap_raw_out(arguments(raw_out=str(failed)), PORTS)
        self.assertEqual(caught.exception.exit_code, 9)
        self.assertEqual(json.loads(Path(str(failed) + '.json').read_text())['exit'], 9)

    def test_replay_refuses_failed_unfinished_or_truncated_capture(self):
        self.path.write_bytes(produce())
        base = dict(format='dslcap-raw-v1', channels=[0, 1, 2, 3], unitsize=1, meta_lines=1)
        side = Path(str(self.path) + '.json')
        for extra, reason in ((dict(), 'unfinished'), (dict(exit=9), 'exit 9'),
                              (dict(exit=0, bytes=5), 'record says 5')):
            side.write_text(json.dumps(dict(base, **extra)))
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(SystemExit, reason):
                    self.decode(replay_args(self.path))
                out, err = self.decode(replay_args(self.path, dsl_partial=True))
                self.assertIn('byte a5/3c', out)
                self.assertIn('--dsl-partial', err)
        side.write_text(json.dumps(dict(base, exit=0, bytes=self.path.stat().st_size)))
        self.assertIn('byte a5/3c', self.decode(replay_args(self.path))[0])
        with self.assertRaises(SystemExit):
            hla.validate_backend_args(arguments(dsl_partial=True), PORTS)

    def test_malformed_sidecars_are_input_errors(self):
        self.path.write_bytes(produce())
        side = Path(str(self.path) + '.json')
        good = dict(format='dslcap-raw-v1', channels=[0, 1, 2, 3], unitsize=1, meta_lines=1, exit=0)
        for bad in ([], dict(format='dslcap-raw-v1'), dict(good, unitsize=3), dict(good, unitsize=2),
                    dict(good, meta_lines=3), dict(good, channels=[3, 0]), dict(good, channels=[0, 16]),
                    dict(good, channels=['0']), dict(good, channels=[]), dict(good, exit='0'),
                    dict(good, bytes=1.5), dict(good, unitsize=1.0), dict(good, meta_lines=1.0)):
            side.write_text(json.dumps(bad))
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                hla.dslcap_replay_layout(replay_args(self.path), PORTS)

    def test_results_never_pass_an_open_transaction(self):
        # A transaction from 1 ms to 110 ms spans a quiet chunk; a pin edge at
        # 20 ms and the trigger at 30 ms must still print after its result.
        rate = 1_000_000
        samples = bytearray([8]) * 120_000
        for t in range(1_000, 110_000):
            samples[t] = 0
        for bit in range(8):                     # one byte, 0x81, at the start
            value = 4 if bit in (0, 7) else 0
            base = 1_010 + 4 * bit
            samples[base:base + 2] = bytes([value, value])
            samples[base + 2:base + 4] = bytes([value | 1, value | 1])
        for t in range(20_000, 120_000):
            samples[t] |= 16
        header = f'META samplerate: {rate}\nMETA trigger: 30000\n'.encode()
        txn = str(Path(__file__).parent / 'dslcap_txn')
        for name, data in (('closed', bytes(samples)), ('open-at-eof', bytes(samples[:105_000]))):
            path = Path(self.dir.name) / f'{name}.raw'
            path.write_bytes(header + data)
            with self.subTest(name=name), mock.patch.object(hla.FileProducer, 'CHUNK', 100_000):
                out, _ = self.decode(replay_args(path, dsl_triggered=True, int_pin='4', hla_path=txn))
                lines = [line for line in out.splitlines() if line.strip()]
                self.assertEqual([line.split(': ', 1)[1] for line in lines],
                                 ['txn 81', '[4] rising', 'Trigger at sample 30000 (t = 0.030000000 s)'])
                self.assertTrue(lines[0].startswith('0.001000000: '))


class RawOutTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'cap.raw'

    def tearDown(self):
        self.dir.cleanup()

    def test_existing_record_is_never_replaced(self):
        side = Path(str(self.path) + '.json')
        side.write_text('preserve existing record')
        with self.assertRaises(SystemExit):
            hla.run_dslcap_raw_out(arguments(raw_out=str(self.path)), PORTS)
        self.assertEqual(side.read_text(), 'preserve existing record')
        self.assertFalse(self.path.exists())

    def test_sigterm_is_forwarded_and_status_preserved(self):
        import signal
        import threading
        timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='stubborn'), \
             mock.patch.object(hla, 'build_dslcap_cmd', return_value=CMD), \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(hla.CaptureError) as caught:
            timer.start()
            hla.run_dslcap_raw_out(arguments(raw_out=str(self.path)), PORTS)
        timer.join()
        # The fixture ignores SIGTERM, so the forwarded SIGINT ends it.
        self.assertEqual(caught.exception.exit_code, 128 + signal.SIGINT)
        self.assertEqual(json.loads(Path(str(self.path) + '.json').read_text())['exit'], 128 + signal.SIGINT)
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)


    def test_wait_gives_up_after_kill_grace(self):
        import signal
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [KeyboardInterrupt] + [subprocess.TimeoutExpired('dslcap', 1)] * 3
        with self.assertRaisesRegex(hla.CaptureError, 'even after SIGKILL'):
            hla._wait_raw_producer(proc)
        self.assertEqual([c.args[0] for c in proc.send_signal.call_args_list],
                         [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)


class DeglitchOptionTests(unittest.TestCase):
    T = 'deglitch:channels=SCLK:clock_period=2.5:frame_pulses=8'

    def test_parse_names_and_options(self):
        args = arguments(channels='0=SCLK,8=SCLK_B', transform='deglitch:channels=sclk,SCLK_B:clock_period=2.5:frame_pulses=8')
        self.assertEqual(hla.parse_dslogic_transform(args),
                         dict(bits=[0, 8], clock_period=2.5, frame_pulses=8, min_period=None, pulse_level=1))
        self.assertIsNone(hla.parse_dslogic_transform(arguments()))
        # C's forms: clock_period=0 disables clock recovery; nonzero pulse_level means 1.
        config = hla.parse_dslogic_transform(arguments(transform='deglitch:channels=0:clock_period=0:pulse_level=2'))
        self.assertEqual((config['clock_period'], config['pulse_level']), (0.0, 1))
        for bad in ('invert', 'deglitch', 'deglitch:clock_period=2.5', 'deglitch:channels=0:bogus=1',
                    'deglitch:channels=0:clock_period=nan', 'deglitch:channels=0:frame_pulses=-1',
                    'deglitch:channels=0:clock_period=1.5', 'deglitch:channels=0:clock_period',
                    'deglitch:channels=UNKNOWN'):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                hla.parse_dslogic_transform(arguments(transform=bad))

    def test_channels_must_be_captured(self):
        with self.assertRaisesRegex(SystemExit, 'would not be captured'):
            hla.validate_backend_args(arguments(transform='deglitch:channels=5:clock_period=2.5'), PORTS)
        hla.validate_backend_args(arguments(channels='0=SCLK', transform=self.T), PORTS)

    def test_raw_out_rejects_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.raw'
            with self.assertRaisesRegex(SystemExit, 'apply -T at replay'):
                hla.run_dslcap_raw_out(arguments(raw_out=str(path), channels='0=SCLK', transform=self.T), PORTS)
            self.assertFalse(path.exists())

    def test_replay_with_deglitch_is_harmless_on_clean_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cap.raw'
            path.write_bytes(produce() + bytes([8]) * 32)   # idle beyond the 14-sample lookahead
            outputs = []
            for transform in (None, self.T):
                out, err = io.StringIO(), io.StringIO()
                args = replay_args(path, channels='0=SCLK', transform=transform, hex=True)
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    hla.validate_backend_args(args, PORTS)
                    hla.run_sigrok_numpy(args, PORTS, ['SPI'], 1e6)
                outputs.append((out.getvalue(), err.getvalue()))
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertIn('byte a5/3c', outputs[1][0])
        self.assertIn('deglitch bit 0: suppressed 0, split 0, inserted 0, unresolved 0', outputs[1][1])


if __name__ == '__main__':
    unittest.main()
