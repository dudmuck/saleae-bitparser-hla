# Intermediate serial widths and word alignment

Bench date: 2026-10-07 PDT / 2026-10-08 UTC. Baseline commit: 89648ba.

The original-frame sweep does not show a monotonic width limit. Both 8-bit
controls and the 10-bit match triggered naturally; widths 9 and 11–16 did not.
Three further captures held the 16-bit registers fixed while changing the
target's position: aligned hit, crossing miss, aligned hit. This demonstrates
alignment-dependent matching and working 16-bit comparison on this DSView
setup. It supports complete-word comparison from start, rather than the
previously documented arbitrary sliding-window assumption. The internal FPGA
algorithm and all possible configurations have not been established.

## Original-frame sweep

Each case emits one mode-0 SPI frame `00 02 00 1C 35 00`, at the app's original
requested 8 MHz. The target is the low N bits of `0x1c35`, unique within the
frame and ending at zero-based bit 39 (clock 40 after NSS assertion).

| Case | Width | Value | Natural FPGA hit |
| --- | --- | --- | --- |
| Opening control | 8 | 0x35 | Yes |
| w9 | 9 | 0x35 | No |
| w10 | 10 | 0x35 | Yes |
| w11 | 11 | 0x435 | No |
| w12 | 12 | 0xc35 | No |
| w13 | 13 | 0x1c35 | No |
| w14 | 14 | 0x1c35 | No |
| w15 | 15 | 0x1c35 | No |
| w16 | 16 | 0x1c35 | No |
| Closing control | 8 | 0x35 | Yes |

Positive captures contain the exact 48-clock frame and MISO `045200000000`;
their trigger sample equals the rising edge of bit 39 exactly. Actual arm
images differ only in stage 3 mask/value/count as intended. In particular,
w9 and w10 retain value 0x35 and differ only in mask and count. Both controls
have identical register bytes. This rules out a simple maximum of 8 bits for
these observations. It does not make every tested width/value independent:
width, count and mask change together, and the value changes at 11–13 bits.

## Alignment discriminator

Clock 40 is divisible by 8 and 10. That suggested comparison at complete
N-bit word boundaries counted from start, rather than at every sliding window.
The original frame's 16-bit words are `0002`, `001c`, `3500`; the desired
`1c35` crosses a word boundary. Moving the four FIFO payload bytes to
`1c 35 00 00` produces words `0002`, `1c35`, `0000`, with the unique target
ending at bit 31 / clock 32.

The follow-up sequence is aligned, crossing, aligned, with the same 16-bit
value, mask, count, trigger roles and all other capture settings. It uses the
same WriteTxFifo operation and four payload bytes. No new radio command,
configuration, helper or dependency is needed.

| Case | MOSI frame | Target clocks (1-based) | Natural FPGA hit |
| --- | --- | --- | --- |
| align16_a | `00 02 1C 35 00 00` | 17–32 | Yes |
| cross16_repeat | `00 02 00 1C 35 00` | 25–40 | No |
| align16_b | `00 02 1C 35 00 00` | 17–32 | Yes |

All three actual register images equal the original 16-bit golden SHA-256
`ba88189cd2d1c83133f3016954ffc6d1e80b09b1501e3b018a90b6cc3eba75bc`.
The aligned captures' trigger samples coincide with rising clock 32 exactly.
The crossing repeat returns FPGA status 0 only after forced upload.
This is a direct contrast at fixed width/value/count/mask; it is not evidence
that 16-bit matching is unsupported or that lowering SPI speed is required.
One native inter-agent SENT receipt was dropped during the crossing repeat;
lead relayed the existing receipt, and no SPI frame was retransmitted.

The next native dslcap bit-order test should use the aligned carrier
`00 02 VV VV 00 00`, first with `1C 35`, then the bit/byte-order variants.
Prior negative results used the target across a 16-bit word boundary and
cannot establish a bit-order error. No production trigger mapping was changed.

## Method and evidence limits

Real DSView 1.3.2 from `/home/wroberts/DSView-1.3.2`: 100 MS/s, CH0–7,
N=1,000,448, position 10%, VTH 1.6 V, finite buffer, filter/RLE/loop/instant
off, global stage 0, start CH3 falling, stop CH3 rising, clock CH0 rising,
data CH2. Upper unused compare bits are X. Stage 3 count is Nbits−1.
The validated temporary USB logger preserves actual 372-byte arm settings,
first 512-byte trigger header and full data transfers. FPGA hit status is
the header word at byte 20, bit 0, not the libusb callback status.

The DSView owner and radio owner directly coordinate READY, ARM/register
check, unique named GO, one emission, completion, and restoration. Only the
radio owner emits SPI; cleanup runs while capture is idle. Failed cases wait
at least 20 seconds after emission before authorized Stop. They return FPGA
status 0 and 99,328 forced idle samples; those late windows cannot show the
earlier frame. Emission receipts and successful separate controls support
the failed cases' stimulus. No claim of a contemporaneous waveform in a
failed capture is made.

Independent dslcap_worker verification reuses the frozen E parser, with only
its output-directory allowlist changed to F. It checks arm bytes, true header
status, Stop/force ordering, raw byte totals, cross64 unpacking, SPI bytes,
unique value windows and trigger-edge association. Rates are relative to the
nominal analyzer clock; no new absolute clock calibration is claimed.

Artifacts: `/tmp/dslcap-f-dsview/`, `/tmp/dslcap-f-verify/`.
Tasks: `dslcap-f-dsview-20261008`, `dslcap-f-radio-20261008`.
The companion JSON preserves compact raw register/header evidence and hashes.
Native dslcap bit/byte-order and opcode acceptance remain separate gates.

## Restoration and review

Both owner tasks completed at revision 5; lead consumed and acknowledged the
full reports. DSView restored its original session/configuration and released
USB at 01:24:20.996 UTC. Lead verified no USB holder or DSView/dslcap process.
At 01:24:29.756 UTC, pi133 was back at STBY_XOSC, IRQ/errors zero, empty TX FIFO
and original FIFO flags 03/27. Mode 0, 8 bits, MSB first and default 8 MHz were
unchanged, as was the running mcp_radio process. Pi134 and GPIO configuration
were untouched. No helper, dependency, firmware or production code changed.
The documentation now qualifies the old sliding-window claim and gives an
aligned carrier for the remaining native acceptance tests.
