import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigrok_hla as hla

FIXTURE = Path(__file__).parent / 'dslcap'
CMD = [sys.executable, str(FIXTURE / 'fake_producer.py')]


def arguments(**changes):
    values = dict(dslogic=True, saleae=False, driver=None, input_file=None,
        input_format=None, transform=None, engine='numpy', channels=None,
        samplerate='1M', samples='76', time=None, continuous=False,
        int_pin=None, extra_pin=[], dslcap='dslcap', vth=None, dslcap_verbose=0,
        cpol=0, cpha=0, hla_path=str(FIXTURE), hex=False)
    values.update(changes)
    return SimpleNamespace(**values)


PORTS = [hla.parse_spi_port('0,1,2,3')]


class BackendTests(unittest.TestCase):
    def test_derived_channel_builder_and_named_pins(self):
        args = arguments(channels='D0=SCLK,1=MISO,2=MOSI,3=CS,4=INT,7=BUSY',
                         int_pin='int', extra_pin=['BUSY'], time='100ms', samples=None)
        ports = [hla.parse_spi_port('sclk,MISO,mosi,CS')]
        cmd = hla.build_dslcap_cmd(args, ports)
        self.assertEqual(cmd, ['dslcap', '--samplerate', '1000000', '--channels',
            '0,1,2,3,4,7', '--vth', '1.6', '--time', '0.1s'])
        self.assertNotIn('=', cmd[cmd.index('--channels') + 1])

    def test_numeric_ports_and_continuous_defaults(self):
        args = arguments(samplerate=None, samples=None, continuous=True)
        cmd = hla.build_dslcap_cmd(args, PORTS)
        self.assertEqual(cmd[cmd.index('--samplerate') + 1], '25000000')
        self.assertEqual(cmd[-1], '--continuous')
        self.assertEqual(hla.resolve_channel_indices(args, PORTS)[0][0]['clk'], 0)

    def test_verbose_producer_forwarding(self):
        self.assertEqual(hla.build_dslcap_cmd(arguments(dslcap_verbose=1), PORTS)[-1], '-v')
        self.assertEqual(hla.build_dslcap_cmd(arguments(dslcap_verbose=2), PORTS)[-1], '-vv')
        with self.assertRaises(SystemExit):
            hla.validate_backend_args(arguments(dslogic=False, dslcap_verbose=1), PORTS)

    def test_unsupported_combinations_before_launch(self):
        cases = [dict(engine='srd'), dict(driver='fx2lafw'), dict(input_file='x'),
            dict(input_format='binary'), dict(saleae=True), dict(transform='deglitch'),
            dict(samples=None), dict(time='1s'), dict(continuous=True),
            dict(vth=float('nan')), dict(vth=2.6), dict(samplerate='nan'),
            dict(samples='0'), dict(samples='1.1'), dict(time='bad', samples=None),
            dict(channels='16=CLK'), dict(int_pin='16'), dict(channels='0=CLK,1=clk')]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(arguments(**change), PORTS)
        with self.assertRaises(SystemExit):
            hla.build_dslcap_cmd(arguments(), [hla.parse_spi_port('16,1,2,3')])

    def test_sigrok_command_regression(self):
        args = arguments(dslogic=False, driver='fx2lafw', channels='0=CLK,1=DI,2=DO,3=CS')
        cmd = hla.build_sigrok_cmd(args, PORTS)
        self.assertEqual(cmd, ['sigrok-cli', '-d', 'fx2lafw', '-C', args.channels,
            '--config', 'samplerate=1M', '--samples', '76', '-O', 'binary'])

    def test_meta_every_split_and_raw_false_prefix(self):
        data = b'META samplerate: 2000000\n' + b'\x00\xff\x08'
        for split in range(1, len(data)):
            with self.subTest(split=split):
                prefix = hla.MetaPrefix(1e6, required=True)
                with contextlib.redirect_stderr(io.StringIO()):
                    result = prefix.feed(data[:split]) + prefix.feed(data[split:]) + prefix.feed(b'', eof=True)
                self.assertEqual(result, b'\x00\xff\x08')
                self.assertEqual(prefix.rate, 2e6)
        prefix = hla.MetaPrefix(1e6)
        self.assertEqual(prefix.feed(b'M') + prefix.feed(b'E\x00'), b'ME\x00')
        prefix = hla.MetaPrefix(1e6)
        self.assertEqual(prefix.feed(b'M') + prefix.feed(b'', eof=True), b'M')

    def test_meta_invalid_and_missing(self):
        for data in (b'META samplerate: 0\n', b'META samplerate: -1\n',
                     b'META samplerate: 2', b'raw', b'', b'META samplerate: ' + b'1' * 260):
            with self.subTest(data=data), self.assertRaises(hla.CaptureError):
                prefix = hla.MetaPrefix(1e6, required=True)
                prefix.feed(data)
                prefix.feed(b'', eof=True)

    def run_stream(self, scenario, **changes):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO=scenario), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            hla.run_sigrok_numpy(arguments(**changes), PORTS, ['SPI'], 1e6, CMD)
        return stdout.getvalue(), stderr.getvalue()

    def test_synthetic_spi_and_meta_pin_timestamps(self):
        output, err = self.run_stream('fragmented', int_pin='4', hex=True)
        # Sampling edges: 6,10,...,34 and 38,42,...,66. The unchanged
        # decoder timestamps each byte at its eighth (last) sample edge.
        self.assertIn('0.000017000: byte a5/3c', output)
        self.assertIn('0.000033000: byte 5a/c3', output)
        self.assertIn('0.000010000: [4] rising', output)
        self.assertIn('0.000020000: [4] falling', output)
        self.assertIn('MOSI: a5 5a', output)
        self.assertIn('MISO: 3c c3', output)
        self.assertLess(output.index('[4] rising'), output.index('byte a5'))
        self.assertLess(output.index('byte a5'), output.index('[4] falling'))
        self.assertLess(output.index('[4] falling'), output.index('byte 5a'))
        self.assertIn('using META for SPI and pin timing', err)

    def test_stderr_more_than_pipe_capacity(self):
        output, err = self.run_stream('pressure')
        self.assertIn('byte a5/3c', output)
        self.assertGreater(len(err), 190000)
        self.assertIn('[dslcap stderr] log 191:', err)

    def test_nonzero_exit_and_bad_meta_rejected(self):
        for scenario in ('failure', 'invalid-meta', 'truncated-meta'):
            with self.subTest(scenario=scenario), self.assertRaises(hla.CaptureError):
                self.run_stream(scenario)
        with self.assertRaisesRegex(hla.CaptureError, 'exited with status 8'):
            self.run_stream('empty-failure')

    def test_raw_sigrok_without_meta_regression(self):
        output, err = self.run_stream('raw', dslogic=False)
        self.assertIn('0.000034000: byte a5/3c', output)
        self.assertNotIn('using META', err)

    def test_sigrok_srd_pressure_and_annotations(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='srd'), \
             mock.patch.object(hla, 'build_sigrok_cmd', return_value=CMD), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            hla.run_sigrok_backend(arguments(dslogic=False, engine='srd'), PORTS, ['SPI'])
        self.assertIn('byte a5/3c', stdout.getvalue())
        self.assertIn('[sigrok stderr] log 191:', stderr.getvalue())

    def test_bounded_cancel_stubborn_child(self):
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='stubborn'), \
             contextlib.redirect_stderr(io.StringIO()):
            producer = hla.CaptureProducer(CMD, 'fixture')
            chunks = producer.chunks()
            next(chunks)
            time.sleep(.1)
            begin = time.monotonic()
            producer.finish(cancel=True)
        self.assertLess(time.monotonic() - begin, 5)
        self.assertEqual(producer.proc.returncode, -signal.SIGKILL)
        self.assertFalse(any(thread.is_alive() for thread in producer.readers))

    def test_cancellation_reaps_child_on_decode_interrupt(self):
        import fast_spi
        created = []
        real = hla.CaptureProducer
        def record(*args):
            producer = real(*args); created.append(producer); return producer
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='stubborn'), \
             mock.patch.object(hla, 'CaptureProducer', side_effect=record), \
             mock.patch.object(fast_spi.MultiPortDecoder, 'feed', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(KeyboardInterrupt):
            hla.run_sigrok_numpy(arguments(), PORTS, ['SPI'], 1e6, CMD)
        self.assertIsNotNone(created[0].proc.poll())
        self.assertFalse(any(thread.is_alive() for thread in created[0].readers))

    def test_reader_errors_surface_and_reap(self):
        with contextlib.redirect_stderr(io.StringIO()), \
             mock.patch.object(hla.CaptureProducer, '_pipe_chunks', side_effect=OSError('read fault')):
            producer = hla.CaptureProducer(CMD, 'fixture')
            with self.assertRaises(hla.CaptureError):
                try:
                    list(producer.chunks())
                    producer.finish()
                finally:
                    producer.finish(cancel=True)
        self.assertIsNotNone(producer.proc.poll())
        self.assertFalse(any(thread.is_alive() for thread in producer.readers))

    def test_saturated_stdout_queue_cancel(self):
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='flood'), \
             contextlib.redirect_stderr(io.StringIO()):
            producer = hla.CaptureProducer(CMD, 'fixture')
            until = time.monotonic() + 3
            while not producer.queue.full() and time.monotonic() < until:
                time.sleep(.01)
            self.assertTrue(producer.queue.full())
            start = time.monotonic()
            producer.finish(cancel=True)
        self.assertLess(time.monotonic() - start, 5)
        self.assertIsNotNone(producer.proc.poll())
        self.assertFalse(any(thread.is_alive() for thread in producer.readers))

    def test_dual_spi_heap_ordering(self):
        output, errors = io.StringIO(), io.StringIO()
        ports = PORTS + [hla.parse_spi_port('4,5,6,7')]
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='dual'), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            hla.run_sigrok_numpy(arguments(), ports, ['SPI', 'SPI_B'], 1e6, CMD)
        self.assertEqual(output.getvalue().splitlines(), [
            '0.000017000: [SPI] byte a5/3c', '0.000021000: [SPI_B] byte a5/3c',
            '0.000033000: [SPI] byte 5a/c3', '0.000037000: [SPI_B] byte 5a/c3'])

    def test_sigrok_numpy_meta_regression(self):
        output, _ = self.run_stream('pressure', dslogic=False)
        self.assertIn('0.000017000: byte a5/3c', output)

    def test_cli_success_and_producer_failure(self):
        with tempfile.TemporaryDirectory(prefix='dslcap-python-') as folder:
            wrapper = Path(folder) / 'fake dslcap'
            wrapper.write_text('#!/usr/bin/env python3\nimport runpy\n'
                f'runpy.run_path({str(FIXTURE / "fake_producer.py")!r}, run_name="__main__")\n')
            wrapper.chmod(0o755)
            command = [sys.executable, str(ROOT / 'sigrok_hla.py'), '--dslogic',
                '--dslcap', str(wrapper), '--spi', '0,1,2,3', '--samples', '76',
                '--samplerate', '1M', '--hla-path', str(FIXTURE)]
            for scenario, expected in (('pressure', 0), ('failure', 1)):
                with self.subTest(scenario=scenario):
                    env = dict(os.environ, DSLCAP_FIXTURE_SCENARIO=scenario)
                    run = subprocess.run(command, capture_output=True, text=True, env=env, timeout=15)
                    self.assertEqual(run.returncode, expected, run.stderr[-2000:])
                    if expected == 0:
                        self.assertIn('0.000017000: byte a5/3c', run.stdout)
                        self.assertIn('[dslcap stderr] log 191:', run.stderr)
                    else:
                        self.assertIn('Capture failed: dslcap exited with status 9', run.stderr)
                    self.assertNotIn('Traceback', run.stderr)
            invalid = subprocess.run(command + ['-d', 'fx2lafw'], capture_output=True,
                                     text=True, timeout=5)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('cannot be combined', invalid.stderr)

    def test_downstream_decode_error_reaps_child(self):
        created = []
        real = hla.CaptureProducer
        def record(*args):
            producer = real(*args); created.append(producer); return producer
        with mock.patch.dict(os.environ, DSLCAP_FIXTURE_SCENARIO='stubborn'), \
             mock.patch.object(hla, 'CaptureProducer', side_effect=record), \
             mock.patch.object(hla, '_flush_results_until', side_effect=BrokenPipeError), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(BrokenPipeError):
            hla.run_sigrok_numpy(arguments(), PORTS, ['SPI'], 1e6, CMD)
        self.assertIsNotNone(created[0].proc.poll())
        self.assertFalse(any(thread.is_alive() for thread in created[0].readers))


if __name__ == '__main__':
    unittest.main()
