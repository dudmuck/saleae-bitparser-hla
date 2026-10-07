import contextlib
import importlib.util
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigrok_hla as hla
from test_dslcap_backend import arguments

FIXTURE = Path(__file__).parent / 'dslcap'
CMD = [sys.executable, str(FIXTURE / 'wide_producer.py')]
PORTS = [hla.parse_spi_port('0,1,2,3'), hla.parse_spi_port('8,9,10,11')]
spec = importlib.util.spec_from_file_location('dslcap_wide_fixture', FIXTURE / 'wide_producer.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

EXPECTED = [
    '0.000010000: [15] rising', '0.000014000: [12] rising',
    '0.000017000: [SPI] byte a5/3c', '0.000020000: [15] falling',
    '0.000021000: [SPI_B] byte 96/69', '0.000024000: [12] falling',
    '0.000033000: [SPI] byte 5a/c3', '0.000037000: [SPI_B] byte 12/34']


class ChunkProducer:
    """Controlled pipe-read boundaries; real subprocess tests follow below."""
    def __init__(self, chunks, error=None):
        self.data = chunks
        self.error = error
        self.finished = False
        self.cancelled = False

    def chunks(self):
        yield from self.data

    def finish(self, cancel=False):
        if self.finished:
            return
        self.finished = True
        self.cancelled = cancel
        if self.error and not cancel:
            raise self.error


def split_bytes(data, sizes):
    result = []
    position = 0
    index = 0
    while position < len(data):
        size = sizes[index % len(sizes)]
        result.append(data[position:position + size])
        position += size
        index += 1
    return result


class WideBackendTests(unittest.TestCase):
    def options(self, **changes):
        return arguments(int_pin='15', extra_pin=['12'], samples='84', **changes)

    def decode(self, producer, options=None):
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(hla, 'CaptureProducer', return_value=producer), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            hla.run_sigrok_numpy(options or self.options(), PORTS, ['SPI', 'SPI_B'], 1e6, CMD)
        return output.getvalue().splitlines(), errors.getvalue()

    def test_exact_producer_union_and_width(self):
        opts = self.options(samplerate='25M')
        cmd = hla.build_dslcap_cmd(opts, PORTS)
        self.assertEqual(cmd[cmd.index('--channels') + 1], '0,1,2,3,8,9,10,11,12,15')
        self.assertEqual(cmd[cmd.index('--samplerate') + 1], '25000000')
        self.assertEqual(hla.dslcap_sample_unitsize(opts, PORTS), 2)
        only_spi = arguments(samplerate='25M')
        self.assertEqual(hla.build_dslcap_cmd(only_spi, PORTS)[4], '0,1,2,3,8,9,10,11')
        self.assertEqual(hla.dslcap_sample_unitsize(only_spi, PORTS), 2)

    def test_unused_high_name_mapping_does_not_widen(self):
        opts = arguments(channels='0=CLK,1=MISO,2=MOSI,3=CS,15=UNUSED')
        ports = [hla.parse_spi_port('CLK,MISO,MOSI,CS')]
        cmd = hla.build_dslcap_cmd(opts, ports)
        self.assertEqual(cmd[cmd.index('--channels') + 1], '0,1,2,3')
        self.assertEqual(hla.dslcap_sample_unitsize(opts, ports), 1)
        opts.extra_pin = ['UNUSED']
        self.assertEqual(hla.dslcap_sample_unitsize(opts, ports), 2)

    def test_named_high_port_resolution(self):
        opts = arguments(channels='0=C0,1=I0,2=O0,3=S0,D8=C1,9=I1,10=O1,11=S1,15=IRQ',
                         int_pin='irq')
        ports = [hla.parse_spi_port('C0,I0,O0,S0'), hla.parse_spi_port('c1,i1,o1,s1')]
        self.assertEqual(hla.resolve_channel_indices(opts, ports)[0][1],
                         {'clk':8, 'miso':9, 'mosi':10, 'cs':11})
        self.assertEqual(hla.build_dslcap_cmd(opts, ports)[4], '0,1,2,3,8,9,10,11,15')

    def test_negative_and_out_of_range_inputs(self):
        for spec in ('-1,1,2,3', '16,9,10,11', '0,1,2,99'):
            with self.subTest(spec=spec), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(arguments(), [hla.parse_spi_port(spec)])
        for mapping in ('-1=CLK', '16=CLK', 'D16=CLK'):
            with self.subTest(mapping=mapping), self.assertRaises(SystemExit):
                hla.build_dslcap_cmd(arguments(channels=mapping), PORTS)
        with self.assertRaises(SystemExit):
            hla.build_dslcap_cmd(arguments(int_pin='-1'), PORTS)
        with self.assertRaises(SystemExit):
            hla.resolve_channel_indices(arguments(dslogic=False), PORTS)

    def test_deterministic_odd_and_random_splits(self):
        wire = b'META samplerate: 2000000\n' + fixture.wide_samples()
        rng = random.Random(3091)
        plans = [[1], [3], [5], [7, 1, 13], [len(wire)],
                 [rng.randrange(1, 24) for _ in range(31)]]
        for sizes in plans:
            with self.subTest(sizes=sizes):
                producer = ChunkProducer(split_bytes(wire, sizes))
                output, errors = self.decode(producer)
                self.assertEqual(output, EXPECTED)
                self.assertIn('using META for SPI and pin timing', errors)
                self.assertTrue(producer.finished)
        # Every possible split of META plus the first binary sample probes
        # carry precisely at the header-to-payload boundary.
        header = b'META samplerate: 2000000\n'
        for split in range(1, len(header) + 3):
            with self.subTest(header_split=split):
                producer = ChunkProducer([wire[:split], wire[split:]])
                self.assertEqual(self.decode(producer)[0], EXPECTED)

    def test_little_endian_uint16_arrays_reach_existing_decoder(self):
        import fast_spi
        seen = []
        real = fast_spi.MultiPortDecoder.feed
        def record(decoder, chunk):
            seen.append((chunk.dtype.str, len(chunk), int(chunk[0])))
            yield from real(decoder, chunk)
        producer = ChunkProducer([b'META samplerate: 2000000\n' + fixture.wide_samples()])
        with mock.patch.object(fast_spi.MultiPortDecoder, 'feed', new=record):
            self.assertEqual(self.decode(producer)[0], EXPECTED)
        self.assertEqual(seen, [('<u2', 84, 0x0808)])

    def test_truncated_sample_and_upstream_failure(self):
        wire = b'META samplerate: 2000000\n' + fixture.wide_samples()[:-1]
        for sizes in ([1], [3, 7], [len(wire)]):
            with self.subTest(sizes=sizes), self.assertRaisesRegex(hla.CaptureError, 'one byte remains'):
                producer = ChunkProducer(split_bytes(wire, sizes))
                self.decode(producer)
            self.assertTrue(producer.finished)
        producer = ChunkProducer([wire], hla.CaptureError('producer exit 10'))
        with self.assertRaisesRegex(hla.CaptureError, 'producer exit 10'):
            self.decode(producer)

    def test_actual_subprocess_odd_random_and_error_cleanup(self):
        created = []
        real = hla.CaptureProducer
        def record(*args):
            p = real(*args); created.append(p); return p
        for scenario in ('odd', 'one-byte', 'random', 'truncated', 'failed-truncated'):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(scenario=scenario), \
                 mock.patch.dict(os.environ, DSLCAP_WIDE_SCENARIO=scenario), \
                 mock.patch.object(hla, 'CaptureProducer', side_effect=record), \
                 contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                if scenario == 'truncated':
                    with self.assertRaisesRegex(hla.CaptureError, 'one byte remains'):
                        hla.run_sigrok_numpy(self.options(), PORTS, ['SPI','SPI_B'], 1e6, CMD)
                elif scenario == 'failed-truncated':
                    with self.assertRaisesRegex(hla.CaptureError, 'status 10'):
                        hla.run_sigrok_numpy(self.options(), PORTS, ['SPI','SPI_B'], 1e6, CMD)
                else:
                    hla.run_sigrok_numpy(self.options(), PORTS, ['SPI','SPI_B'], 1e6, CMD)
                    self.assertEqual(output.getvalue().splitlines(), EXPECTED)
            self.assertIsNotNone(created[-1].proc.poll())
            self.assertFalse(any(thread.is_alive() for thread in created[-1].readers))

    def test_wide_interrupt_reaps_child(self):
        import fast_spi
        created = []
        real = hla.CaptureProducer
        def record(*args):
            p = real(*args); created.append(p); return p
        with mock.patch.dict(os.environ, DSLCAP_WIDE_SCENARIO='stubborn'), \
             mock.patch.object(hla, 'CaptureProducer', side_effect=record), \
             mock.patch.object(fast_spi.MultiPortDecoder, 'feed', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            hla.run_sigrok_numpy(self.options(), PORTS, ['SPI','SPI_B'], 1e6, CMD)
        self.assertIsNotNone(created[0].proc.poll())
        self.assertFalse(any(thread.is_alive() for thread in created[0].readers))


if __name__ == '__main__':
    unittest.main()
