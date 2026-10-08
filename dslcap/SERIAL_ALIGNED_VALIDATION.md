# Native aligned serial bit-order validation

Bench date: 2026-10-07 PDT / 2026-10-08 UTC. Baseline: `cb9008d`.

Native C.2 **passes at 100 MS/s on CH0–7**. With fixed target `0x1c35/16`,
the aligned intended order triggered twice, exactly at the last matching
clock edge. Bit reversal, byte reversal and their combination each timed out.
Four separate NSS controls captured the exact frames and excluded accidental
occurrences of the other variants. This supports the existing MSB-first CLI
mapping; no production change was needed. C.3 opcode validation and live
bit-order checks at higher sample rates remain open.

| Case | MOSI frame | Result | Trigger sample K |
| --- | --- | --- | --- |
| ctl_pos | `00 02 1C 35 00 00` | NSS control passed | 100088 |
| s_pos | `00 02 1C 35 00 00` | Serial hit | 100081 |
| ctl_bit | `00 02 AC 38 00 00` | NSS control passed | 100082 |
| s_bit | `00 02 AC 38 00 00` | Timeout, exit 16 | none |
| ctl_byte | `00 02 35 1C 00 00` | NSS control passed | 100079 |
| s_byte | `00 02 35 1C 00 00` | Timeout, exit 16 | none |
| ctl_both_retry | `00 02 38 AC 00 00` | NSS control passed | 100032 |
| s_both | `00 02 38 AC 00 00` | Timeout, exit 16 | none |
| s_pos_repeat | `00 02 1C 35 00 00` | Serial hit | 100072 |

Each successful capture has exactly 1,000,448 physical uint8 samples, the
correct two META lines, valid FPGA header status 1 and remaining count 0.
The header's real position equals META K. Control K equals NSS assertion;
both serial K values equal rising clock 32 (zero-based bit 31) with zero
sample delta. Every captured frame has 48 clocks and MISO `04 52 00 00 00 00`.
An exhaustive rolling 16-bit search finds its own variant once at bits 16–31
and none of the other three variants anywhere in the frame.

All three negative serial outputs contain only the 27-byte samplerate META
line and exit 16. There are no completed USB IN transfers before abort.
After `bmFORCE_RDY`, cleanup receives a 512-byte all-0x55 dummy header and
data. The dummy's remaining count exceeds the armed sample count, so neither
its magic nor its apparent status bit is valid trigger evidence. Native abort
handling emits no samples from these callbacks. These cases establish native
timeout with no valid pre-abort trigger header; they do not provide a valid
FPGA status-0 header or a contemporaneous negative-case waveform.

One extra attempt, `ctl_both`, expired before its GO was sent following a
context handoff. Its frame was emitted at 02:01:22.222–22.503 UTC, about
131.5 seconds after the process ended at 01:59:10.710. It is retained as
inconclusive scheduling evidence, excluded from the matrix. After restoration,
the explicitly authorized `ctl_both_retry` supplied the valid control above.
There were ten captures and ten emissions, with no overwritten evidence.

Common settings: finite buffer, 100 MS/s, CH0–7, N=1,000,448, position 10%,
VTH 1.6 V, timeout 45 s with `fail`. Serial expression:
`start=3:f,stop=3:r,clock=0:r,data=2,value=0x1c35,bits=16`.
Controls use `--trigger 3:f`. Pi133 uses the existing app's mode-0 SPI at
requested 8 MHz. One WriteTxFifo frame carries the test word at clocks 17–32;
the original clocks-25–40 carrier is deliberately excluded from order testing.
This result and the prior DSView alignment experiment do not establish the
complete FPGA comparison algorithm or all configurations. Aligned payload
words can match too; this is not an opcode-only filter.

The real frozen binary is `/tmp/dslcap-phase-c-build/dslcap`, SHA-256
`5e9be8b773d8f33a74c49323cbf91638f3d68496e322f0939e0b652fae46ce8d`.
Every actual 372-byte EP2 arm image matches the corresponding DSView reference
from `/home/wroberts/DSView-1.3.2`: serial SHA-256
`ba88189cd2d1c83133f3016954ffc6d1e80b09b1501e3b018a90b6cc3eba75bc`,
simple SHA-256 `c63e3ebaae8141b36d045fc5998a07c8870efc90a77b7121e43e56d2de5c0e8c`.

Lead owns USB and the capture records under `/tmp/dslcap-g-live`.
`dslcap_worker` independently checks raw output, actual registers, valid and
invalid headers, complete frames, all possible match offsets and radio receipt
windows under `/tmp/dslcap-g-verify`. Separate controls and the owner's one-frame
receipts support the negative trials; they cannot prove their unrecorded
waveforms. Host timestamps place the valid emissions inside the armed wait,
not at a sample-accurate instant. The JSON companion preserves all ten attempt
reports, hashes, exact register bytes, receipt checks and these limitations.

Radio owner task `dslcap-g-radio-20261008` restores pi133 after each emission:
STBY_XOSC, system IRQ 0, errors 0, TX FIFO empty, original sticky FIFO flags
RX 0x03 / TX 0x27 retained. Only newly raised FIFO_TX is cleared. No RF,
GPIO, reset or configuration change; pi134 is untouched. The owner confirmed
final restoration at 02:07:47 UTC, original app PID 7973, mode-0/8-bit/8-MHz
SPI settings and unchanged pins, including GPIO20 input/pull-down/low.
Terminal revision 5 was consumed and acknowledged. Lead's final USB ownership
check found no holder. The JSON preserves the hash-bound terminal report.

Independent durable-record review passed 4,923 exact field comparisons,
including all 40 artifact hashes, eight unchanged implementation source hashes,
register images, receipts and restoration provenance. No blocking findings.
Its reviewed evidence hash and verdict are embedded in the JSON; only that
review annotation was added afterward. Production code and frozen parsers
were unchanged, so prior implementation tests were not repeated for this
evidence/documentation update.
