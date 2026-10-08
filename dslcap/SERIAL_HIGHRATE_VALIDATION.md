# Native serial bit order at 200 and 400 MS/s

Bench date: 2026-10-07 PDT / 2026-10-08 UTC. Baseline: `1e9d826`.

The native C.2 bit-order result **holds at 200 MS/s (CH0–7) and 400 MS/s
(CH0–3)**. At each rate, the fixed target `0x1c35/16` triggered on the aligned
intended frame twice, exactly at the last matching clock edge. Bit reversal,
byte reversal and their combination each timed out. No production change was
needed.

| Case | MOSI frame | 200 MS/s | 400 MS/s |
| --- | --- | --- | --- |
| ctl_pos (`3:f`) | `00 02 1C 35 00 00` | hit, K 100054 = NSS fall | hit, K 200074 = NSS fall |
| s_pos | `00 02 1C 35 00 00` | hit, K 100071 = clock 32 | hit, K 200099 = clock 32 |
| s_bit | `00 02 AC 38 00 00` | timeout, exit 16 | timeout, exit 16 |
| s_byte | `00 02 35 1C 00 00` | timeout, exit 16 | timeout, exit 16 |
| s_both | `00 02 38 AC 00 00` | timeout, exit 16 | timeout, exit 16 |
| s_pos_repeat | `00 02 1C 35 00 00` | hit, K 100069 = clock 32 | hit, K 200110 = clock 32 |

The sample delta between K and the measured edge is 0 in all six hits (two
controls, four serial positives). This matches Phase B's zero-sample
simple-trigger result at these rates. Each hit
has two META lines, a valid FPGA header with status 1 and remaining count 0,
and `real_pos` equal to K. Each holds one complete 48-clock frame. The sample
counts are 1,000,448 at 200M and 2,000,896 at 400M, one uint8 per sample;
at 400M only bits 0–3 carry CH0–3. All six negative outputs are the 27-byte
samplerate line only, with no valid pre-abort header.

Settings: frozen binary `/tmp/dslcap-phase-c-build/dslcap`, SHA-256
`5e9be8b773d8f33a74c49323cbf91638f3d68496e322f0939e0b652fae46ce8d`; finite
buffer, position 10%, VTH 1.6 V, timeout 45 s with `fail`. 200M uses CH0–7 with
N=1,000,448; 400M uses CH0–3 with N=2,000,896. Both windows are 5.002 ms long.
Serial: `start=3:f,stop=3:r,clock=0:r,data=2,value=0x1c35,bits=16`. pi133 is the
LR2021 on the existing app path, mode 0 at requested 8 MHz (7.8125 MHz), so a
clock period is 25.6 samples at 200M and 51.2 at 400M.

Every actual 372-byte EP2 arm image equals the direct DSView reference for its
rate and kind. Serial references come from C.1 (`ae1eb644…` at 200M, `701f3589…`
at 400M). Simple references come from Phase B (`0d80e20b…` at 200M, `0e46b442…`
at 400M). The worker's source encoder independently reproduces all four images.

Scope of the negatives: per-variant NSS controls were not repeated at these
rates. The radio emission does not depend on the analyzer rate, and C.2's 100M
controls captured each exact carrier and excluded the other variants at every
alignment. Here the positive control and both positives at each rate show the
carrier on the wire, and each negative has an exact one-frame owner receipt
inside its armed window. The negatives themselves have no contemporaneous
waveform; that limit is stated rather than closed.

Roles: lead (Claude session `saleae-binparser-64`) owned USB, captures and this
record. Radio owner `hydra-develop-26` ran task `dslcap-i-radio-20261008`: twelve
named GOs, each emitting one frame after a verified arm. After each capture it
cleared the TX FIFO and only the newly raised FIFO_TX IRQ, then checked the
baseline. Final state 03:23:22 UTC: STBY_XOSC, IRQ 0, errors 0, TX FIFO 0,
FIFO flags RX 0x03 / TX 0x27, app PID 7973, mode 0 / 8-bit / 8 MHz, pins
unchanged, pi134 untouched. Revision 5 was consumed and acknowledged. A lead
`fuser` check at 03:23:39 UTC found no process holding the analyzer.

Independent verification: `dslcap_worker` (Codex) designed the matrix and
independently checked all twelve captures from the raw artifacts under
`/tmp/dslcap-i-verify`. It covered raw output, register images, valid headers,
complete frames, the measured K edges, HLA decodes, and owner receipts and
restorations against hash-bound mailbox records. Verdict: VERIFIED for the
tested 200M CH0–7 and 400M CH0–3 matrix. The JSON companion preserves all
case records, artifact hashes, register images, receipts and the owner's result.

Limits: only the fixed 16-bit `0x1c35` comparator on an aligned carrier, at
these two rate/channel sets. Arbitrary widths, other alignments, other values
and the full FPGA comparison mechanism remain unproven. An aligned payload word
can match, so this is not an opcode-only filter.
