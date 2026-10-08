# Serial trigger width and clock diagnosis

Live matrix completed, 2026-10-07 PDT / 2026-10-08 UTC.

Real DSView 1.3.2 triggered naturally for a 1-bit value `1` and an 8-bit
value `0x35` at the original requested 8 MHz SPI speed. The 8-bit trigger sample
equals the last rising clock edge of the unique `0x35` byte. The earlier
16-bit value `0x1c35` miss was reproduced at both 1 MHz and 100 kHz.
These results do not establish a maximum supported width or the root cause.
No production implementation change is justified yet.

| Match / control | Requested SPI clock | Natural FPGA hit | Evidence |
| --- | --- | --- | --- |
| 1-bit `1` | 8 MHz | Yes | Status 1, full 1,000,448 samples |
| 8-bit `0x35` | 8 MHz | Yes | Trigger exactly at bit 39, final bit of `0x35` |
| Simple NSS | 1 MHz | Yes | Correct 48-clock frame; measured 1 MHz |
| 16-bit `0x1C35` | 1 MHz | No | No IN for 74 seconds; Stop returned status 0 |
| Simple NSS | 100 kHz | Yes | Correct 48-clock frame; measured 100 kHz |
| 16-bit `0x1C35` | 100 kHz | No | No IN for 66 seconds; Stop returned status 0 |

The previous 16-bit 8 MHz miss is recorded in DSVIEW_SERIAL_RESPONSE.md.
Widths 1 and 8 succeed for these particular values; widths 9–15 have not been
tested. The 8-bit waveform contains the full 16-bit `1c35` window ending on the
same clock edge, but it was armed for 8 bits. This does not establish that
16-bit matching is universally unsupported or that all shorter matches work.
The next discriminating test is a width/value boundary sweep (for example,
9-bit `0x035`, 13-bit `0x1c35`, 15-bit `0x1c35`), with repeated controls. Native dslcap
short-width positives and the original 16-bit order/opcode gates remain open.

## Method

Reference application: `/home/wroberts/DSView-1.3.2`. Analyzer: DSLogic Plus,
HDL 0x0e. All captures use 100 MS/s, CH0–7, N=1,000,448, trigger position 10%,
threshold 1.6 V, finite buffer mode, filter/RLE/instant/loop off. Serial roles:
start CH3 falling, stop CH3 rising, clock CH0 rising, data CH2; global stage 0.
Unused upper match bits are X, not zero-filled. Actual 372-byte arm images
are retained and checked against the existing 16-bit reference. Short widths
change only stage 3 mask, value and count.

Each named case gets exactly one MOSI frame `00 02 00 1C 35 00` (WriteTxFifo,
four data bytes), after DSView reports armed. No RF transmission occurs.
The radio owner restores TX FIFO empty and clears only newly raised system
FIFO_TX after capture completion, preserving original sticky FIFO flags 03/27.
Pi134 is untouched; radio-connected GPIO directions and wiring are unchanged.

At requested 8 MHz, the existing app emits the frame. Its per-transfer speed
override prevents changing speed through the device default. Slow cases use
an independently reviewed temporary stdlib helper, one SPI_IOC_MESSAGE(1),
the same six bytes and a per-transfer speed override. It changes no persistent
mode, bit order or default speed and leaves the app running idle. The owner
excludes other SPI calls during each frame. Mutable ioctl storage permits
checking return 6; MISO is `045200000000`. A read-only BUSY check precedes it.
The helper does not perform restoration; that remains with the radio owner.

## Evidence and interpretation

The true FPGA hit flag is bit 0 of the status word at byte 20 in the first
512-byte header; libusb completion status 0 is unrelated. Natural completion
and a full sample count distinguish the positive cases from forced uploads.
The temporary logger was corrected before the 8-bit capture to classify
header/data by transfer request context, avoiding confusion when ordinary
logic data starts with `55555555`. The original 1-bit header is valid by its
first-IN 512-byte context; that run has no full raw payload.

Raw payload uses 8-channel cross64 packing. Independent worker reconstruction
checks full byte totals, trims to the header-derived aligned sample count,
decodes SPI mode 0 and compares trigger sample to signal edges. Slow controls
measure within-byte periods; an extra idle clock occurs between bytes.
Rates are relative to the nominal analyzer clock, not a new absolute clock
calibration. Pi-side and host timestamps are not assumed tightly synchronized.

Both late forced uploads contain 99,328 idle samples near Stop. They cannot
show whether the earlier frame was present, because that history can have
been overwritten. Correct slow stimulus is supported by the preceding
separate control, helper return count and radio status, not a same-acquisition
waveform in the failed serial case. One case per configuration is diagnostic
evidence, not a reliability or complete bit-order acceptance test.

Artifacts: `/tmp/dslcap-e-dsview/`, independent verifier
`/tmp/dslcap-e-verify/`. Tasks: `dslcap-e-dsview-20261008` and
`dslcap-e-radio-20261008`. Prior comparison: [DSVIEW_SERIAL_RESPONSE.md](DSVIEW_SERIAL_RESPONSE.md).
The companion JSON retains raw header/register records, hashes and restoration
receipts. Independent verification passed six parser/cross-packing tests and
three mocked helper tests, and checked all six captures. These are diagnostic
tool checks, not a new production implementation review cycle.

## Restoration

Both owner tasks completed at revision 5 and were consumed and acknowledged.
DSView exited, restored the original session/configuration byte-for-byte and
released USB at 00:57:36.753 UTC. Lead verified no analyzer USB holder or
DSView/dslcap process remained. Pi133 was restored by approximately 00:58:44 UTC:
STBY_XOSC, system IRQ/errors zero, TX FIFO empty, original FIFO flags 03/27.
SPI readback remains mode 0, 8 bits, MSB first, default 8 MHz; the app's
per-transfer 8 MHz and running process were unchanged. The temporary Pi helper
was removed. Pi134 was untouched. All ten frozen production source hashes match.
