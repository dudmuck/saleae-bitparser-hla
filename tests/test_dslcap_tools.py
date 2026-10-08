"""Negative paths of the dslcap/tools validation checkers: no false PASS."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'dslcap' / 'tools'
sys.path.insert(0, str(TOOLS))
from check_pcycle import prbs9_frame


def hexline(bus, kind, data):
    return f'  [{bus}] {kind}: {data.hex(" ")}'


def transfer(bus, mosi, miso):
    return [hexline(bus, 'MOSI', mosi), hexline(bus, 'MISO', miso)]


def run(n=4, lag=2, length=511):
    """A minimal ping-pong trace in sigrok_hla --hex form: SPI requests, SPI_B replies."""
    status, lines = b'\x04\x52', []
    replies = [bytearray(prbs9_frame(r, length)) for r in range(n + lag)]
    for r, f in enumerate(replies):
        f[4:8] = (0xFFFFFFFF if r < lag else r - lag).to_bytes(4, 'little')
    for k in range(n):
        req = prbs9_frame(k, length)
        lines += transfer('SPI', b'\x00\x02' + req, status + bytes(length))
        lines += transfer('SPI_B', b'\x00\x01' + bytes(length), status + req)
        lines += transfer('SPI_B', b'\x00\x02' + bytes(replies[k + lag]), status + bytes(length))
        lines += transfer('SPI', b'\x00\x01' + bytes(length), status + bytes(replies[k]))
    staged = []
    for r in range(lag):     # replies staged before the first request
        staged += transfer('SPI_B', b'\x00\x02' + bytes(replies[r]), status + bytes(length))
    return staged + lines


class CheckPcycle(unittest.TestCase):
    def check(self, lines, *extra):
        with tempfile.NamedTemporaryFile('w', suffix='.out', delete=False) as f:
            f.write('\n'.join(lines) + '\n')
        result = subprocess.run([sys.executable, str(TOOLS / 'check_pcycle.py'), f.name, *extra],
                                capture_output=True, text=True)
        Path(f.name).unlink()
        return result.returncode, result.stdout

    def test_clean_run_passes_full_contract(self):
        code, out = self.check(run(), '--expect', '4')
        self.assertEqual(code, 0, out)
        self.assertIn('SPI_B unread write seqs on SPI: [4, 5]', out)

    def test_false_pass_cases_fail(self):
        good = run()
        prefill = transfer('SPI_B', b'\x00\x02' + bytes(1021), b'\x04\x52' + bytes(1021))
        cases = {
            'empty trace': ([], ()),
            'wrong-size zero prefill': (good + prefill, ()),
            'trailing MOSI': (good + [hexline('SPI', 'MOSI', b'\x00\x02')], ()),
            'MISO without MOSI': (good + [hexline('SPI', 'MISO', b'\x00\x00')], ()),
            'length mismatch': (good + [hexline('SPI', 'MOSI', b'\x01\x01'), hexline('SPI', 'MISO', b'\x04')], ()),
            'CMD_FAIL line': (good + ['0.1: [SPI] GetStatus (CMD_FAIL)'], ()),
            'DECODE ERROR line': (good + ['0.1: [SPI] *** DECODE ERROR: KeyError ***'], ()),
            # lines 4..11 are exchange 0: request write/read, reply write/read
            'missing request read': ([l for i, l in enumerate(good) if not (6 <= i < 8)], ('--expect', '4')),
            'missing reply read': ([l for i, l in enumerate(good) if not (10 <= i < 12)], ('--expect', '4')),
            'duplicated reply read': (good + good[-2:], ('--expect', '4')),
            'wrong expectation': (good, ('--expect', '5')),
        }
        for name, (lines, extra) in cases.items():
            with self.subTest(name=name):
                code, out = self.check(lines, *extra)
                self.assertEqual(code, 1, out)
                self.assertIn('FAIL', out)

    def test_contract_rejects_bad_payloads_and_staging(self):
        def edit(lines, index, offset, value):
            # Overwrite one byte of the hex block at lines[index] (offset counts from byte 0).
            head, data = lines[index].split(': ', 1)
            raw = bytearray.fromhex(data.replace(' ', ''))
            raw[offset] = value
            lines[index] = head + ': ' + raw.hex(' ')

        staged_write = lambda r: transfer('SPI_B', b'\x00\x02' + bytes(self.reply(r)), b'\x04\x52' + bytes(511))
        # Layout from run(): lines 0..3 staged replies, then 8 lines per exchange:
        # request write MOSI, its MISO, request read MOSI/MISO, reply write, reply read.
        mirrored = run()
        edit(mirrored, 4, 2 + 4, 0x55)                            # request 0 byte 4 on the write...
        edit(mirrored, 7, 2 + 4, 0x55)                            # ...and the same on pi134's read
        bad_echo = run()
        bad_echo_line = [i for i, l in enumerate(bad_echo) if l.startswith('  [SPI_B] MOSI: 00 02 05')][0]
        edit(bad_echo, bad_echo_line, 2 + 4, 0x12)                # unread staged reply 5: echo corrupted
        cases = {
            'request corrupted identically on both buses': mirrored,
            'arbitrary forward-staged reply': run() + staged_write(1000),
            'duplicate final staged reply': run() + staged_write(4),
            'unread staged reply with wrong echo': bad_echo,
            'short FIFO write': run() + transfer('SPI', b'\x00\x02' + bytes(20), bytes(22)),
            'short FIFO read': run() + transfer('SPI', b'\x00\x01' + bytes(20), bytes(22)),
        }
        for name, lines in cases.items():
            with self.subTest(name=name):
                code, out = self.check(lines, '--expect', '4')
                self.assertEqual(code, 1, out)
        code, out = self.check(run() + run()[:4], '--expect', '4')    # negative-control prefix
        self.assertEqual(code, 0, out)
        for bad in (('--expect', '0'), ('--expect', '4', '--echo-lag', '-1')):
            with self.subTest(bad=bad):
                self.assertEqual(self.check(run(), *bad)[0], 1)

    @staticmethod
    def reply(r, lag=2):
        f = bytearray(prbs9_frame(r, 511))
        f[4:8] = (0xFFFFFFFF if r < lag else r - lag).to_bytes(4, 'little')
        return f

    def test_wrong_echo_fails_full_contract(self):
        lines = run()
        bad = lines[-1].split('MISO: ')
        data = bytearray.fromhex(bad[1].replace(' ', ''))
        data[2 + 4] ^= 1                      # echo field of the last reply read
        lines[-1] = bad[0] + 'MISO: ' + data.hex(' ')
        code, out = self.check(lines, '--expect', '4')
        self.assertEqual(code, 1, out)


class CheckDecode(unittest.TestCase):
    def test_error_lines_and_dangling_blocks_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / 'e.json'
            expected.write_text(json.dumps([dict(t=0, bus='A', mosi='0102', miso='0304')]))
            good = transfer('SPI', b'\x01\x02', b'\x03\x04')
            for name, lines, code in (('clean', good, 0), ('CMD_FAIL', good + ['x CMD_FAIL'], 1),
                                      ('DECODE ERROR', good + ['x *** DECODE ERROR: y ***'], 1),
                                      ('dangling MOSI', good + [hexline('SPI', 'MOSI', b'\x01')], 1)):
                out = Path(tmp) / 'd.out'
                out.write_text('\n'.join(lines) + '\n')
                result = subprocess.run([sys.executable, str(TOOLS / 'check_decode.py'), str(out), str(expected)],
                                        capture_output=True, text=True)
                with self.subTest(name=name):
                    self.assertEqual(result.returncode, code, result.stdout)


class DutyAnalyze(unittest.TestCase):
    def test_width_comes_from_record_or_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'w.raw'
            raw.write_bytes(b'META samplerate: 25000000\n' + bytes([8, 0]) * 200)   # uint16, CS idle high
            tool = [sys.executable, str(TOOLS / 'duty_analyze.py'), str(raw), '--clk', '0', '--cs', '3']
            self.assertNotEqual(subprocess.run(tool, capture_output=True).returncode, 0)   # width unknown
            Path(str(raw) + '.json').write_text(json.dumps(dict(unitsize=2)))
            result = subprocess.run(tool, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('frames 0', result.stdout)
            self.assertNotEqual(subprocess.run(tool + ['--width', '1'], capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
