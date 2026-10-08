# Native C.3 opcode trigger validation

Bench date: 2026-10-07 PDT / 2026-10-08 UTC. Baseline: `bf045bb`.

C.3 **passes at 100 MS/s on CH0–7** for the tested LR2021 opcodes. A 16-bit
serial trigger on an opcode fired exactly on rising clock 16 of the frame that
carries that opcode, whether that frame came first, second or third in the
sequence. Earlier frames with other opcodes did not fire. An opcode absent
from the traffic timed out. As already documented, an aligned payload word
equal to the value also fires, so this is not an opcode-only filter.

| Case | Trigger | MOSI NSS frames, in order | Result | K | Marker |
| --- | --- | --- | --- | --- | --- |
| ctl | `3:f` | `000200000000`; `0101`; `00000000` | NSS control hit | 100039 | first NSS fall |
| s_op | serial `0x0101` | `000200000000`; `0101`; `00000000` | Serial hit | 100062 | frame 2, rising clock 16 |
| s_op2 | serial `0x0002` | `0101`; `00000000`; `000200000000` | Serial hit | 100034 | frame 3, rising clock 16 |
| s_absent | serial `0x011f` | `000200000000`; `0101`; `00000000` | Timeout, exit 16 | none | — |
| s_op_repeat | serial `0x0101` | `0101`; `00000000`; `000200000000` | Serial hit | 100076 | frame 1, rising clock 16 |
| s_alias | serial `0x0101` | `000201010000` | Serial hit | 100075 | payload word, rising clock 32 |

Frames: `000200000000` is WriteTxFifo with 4 zero bytes. `0101` then `00000000`
is GetVersion, request and response read; the radio returned `01 18`.
`000201010000` is WriteTxFifo whose first payload word is `0101`.

Every hit has 1,000,448 uint8 samples, two META lines, a valid FPGA header with
status 1 and remaining count 0, and header `real_pos` equal to META K. The sample
delta from K to the expected edge is 0 in all five hits. Every successful capture held the
complete frame sequence, and the existing LR2021 HLA decodes the frame at K
as GetVersion (s_op, s_op_repeat) or WriteRadioTxFifo (s_op2, s_alias). The
negative output is the 27-byte samplerate line only, with no valid pre-abort header.

Common settings: frozen binary `/tmp/dslcap-phase-c-build/dslcap`, SHA-256
`5e9be8b773d8f33a74c49323cbf91638f3d68496e322f0939e0b652fae46ce8d`; finite
buffer, 100 MS/s, CH0–7, N=1,000,448, position 10%, VTH 1.6 V, timeout 45 s
with `fail`. Serial: `start=3:f,stop=3:r,clock=0:r,data=2,value=<case>,bits=16`.
pi133 is the LR2021 (fw `01 18`) on the existing mcp_radio app path, mode 0 at
requested 8 MHz. Implementation sources hash-match the C.2 record; no
production code changed.

Each actual 372-byte EP2 arm image equals its expected image. The serial
images differ from one another only in the stage-3 value at bytes 118–119. The
worker derived them from DSView 1.3.2 source, and that derivation reproduces
the real C.1 DSView `0x1c35` image and the simple `3:f` image byte for byte.
They are source-derived, not fresh DSView UI captures.

Roles: lead (Claude session `saleae-binparser-64`) owned USB, captures and this
record. Radio owner `hydra-develop-26` ran task `dslcap-h-radio-20261008`: one
named GO per case after a verified arm, one batch per GO, six GOs in total.
After each capture it cleared the TX FIFO and only the newly raised FIFO_TX IRQ,
then checked the baseline. Final state 02:41:52 UTC: STBY_XOSC, IRQ 0,
errors 0, TX FIFO 0, sticky FIFO flags RX 0x03 / TX 0x27, app PID 7973,
mode 0 / 8-bit / 8 MHz, pins unchanged, pi134 untouched. Its terminal
revision 5 was consumed and acknowledged. A lead-run `fuser` check after the
final capture found no process holding the analyzer (recorded in the JSON).

Independent verification: `dslcap_worker` (Codex) designed the matrix and
checked every capture offline with its own verifier under `/tmp/dslcap-h-verify`.
It covered raw output, register images, valid headers, complete frames, K edges,
HLA decodes, and owner receipts checked against hash-bound SENT records and
armed windows. Verdict: VERIFIED for all six cases. The verifier initially
treated dslcap's normal end-of-capture stop request as an abort; that was fixed,
with regression tests from real logs, and negatives stay strict.

Limitations: comparison is on 16-bit words aligned from NSS assertion, so a
matching payload word fires too. The negative case has no contemporaneous
waveform; the control and owner receipts support its stimulus. Host times
place emissions inside the armed wait, not to the sample. Only two opcodes were
tested, at 100 MS/s on CH0–7. Higher-rate live bit order and the full FPGA
comparison mechanism remain unproven. The JSON companion preserves all case
records, artifact hashes, register images, receipts and the owner's result.
