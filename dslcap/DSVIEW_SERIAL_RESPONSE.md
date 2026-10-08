# Live DSView serial-trigger comparison

2026-10-07 PDT / 2026-10-08 UTC. Follow-up to SERIAL_VALIDATION.md, committed
as 01b9694. The user authorized testing real DSView against the same live pattern.

**DSView reproduced the serial no-hit behavior.** Its simple NSS-falling control
triggered naturally. The serial capture remained waiting, then returned FPGA
hit bit 0 when explicitly stopped for a forced upload. This argues against a
dslcap-only cause; it does not identify a firmware, configuration or timing fault.
Hardware bit order remains unproven.

## Setup and handshake

Real DSView 1.3.2, 100 MS/s, CH0–CH7, 1,000,448 samples, 10% position,
VTH 1.6 V, finite buffer mode, filter/RLE/loop/instant off. Serial fields:
start CH3 falling, stop CH3 rising, clock CH0 rising, data CH2, value 0x1c35,
16 bits, global stage 0. Actual USB arm images matched the previously captured
simple and serial references byte-for-byte.

After each confirmed arm, the radio owner sent exactly one documented FIFO-write
frame `00 02 00 1C 35 00` on pi133, using its existing mode-0 SPI and requested
8 MHz clock. No other SPI occurred until capture completion and restoration.
Temporary TX FIFO contents and the newly raised FIFO_TX system IRQ were restored
between runs. Pi134 was untouched; no RF, GPIO, reset or radio configuration
changes were authorized or performed.

## Results

All times below are UTC on 2026-10-08.

| Event | Simple NSS control | Serial 0x1c35 |
|---|---|---|
| Arm | 00:22:46.766 | 00:25:37.855 |
| Owner emission window | 00:23:45.007–45.049 | 00:26:18.580–18.912 |
| Natural completion | yes | no |
| Before manual Stop | no Stop issued | still waiting at 00:27:18.953; no IN callbacks |
| Stop marker | none | 00:27:19.651 |
| Header received | 00:23:45.046 | 00:27:20.305 |
| Actual FPGA status | 1 (triggered) | 0 (not triggered) |
| Header real_pos | 100,051 | 100,063 (not a real trigger event) |
| Remaining sample count | 0 | 900,415 |
| Derived aligned sample count | 1,000,448 | 99,328 |

For serial, the observation lasted more than 60 seconds after the emission window
before Stop. DSView issued `bmFORCE_STOP` at 00:27:20.304991, followed about
0.45 ms later by the header with hit bit 0. It then uploaded data and ended.
The serial register image SHA-256 is
`ba88189cd2d1c83133f3016954ffc6d1e80b09b1501e3b018a90b6cc3eba75bc`,
identical to the reference and dslcap's exact-eight-channel failed trial.

## Evidence quality and correction

The temporary USB logger copied header bytes before invoking DSView's original
callback. Header classification used completed EP6 IN, actual length 512,
first-IN ordering followed by the data transfer, and DSView's receive-header
log. The serial logger additionally records requested length 512, distinct from
the 1 MiB data request. Magic bytes alone are insufficient: sample data can also
start with `55 55 55 55`.

The true FPGA status is little-endian uint32 at header offset 20; real_pos is
at offset 4. Both the old DSView `receive_trigger_pos(): status 0` log and the
logger's outer `status=0` mean **libusb transfer completion**, not the FPGA hit
bit. The DSView owner corrected that wording in its earlier temporary golden
reports. Those earlier logs did not independently establish hit bit 0; their
register-byte comparisons remain unaffected. This comparison uses actual bytes.

No `.dsl` file or full logic payload was saved in this test. Thus there is no
independent waveform decode of these two DSView captures; emission evidence comes
from the radio owner, with the earlier dslcap controls verifying the carrier.
Full header prefixes, register bytes, log hashes and timestamps are preserved in
DSVIEW_SERIAL_RESPONSE.json. Raw logs and logger source are under
/tmp/dslcap-c-dsview-live. Independent worker verification is recorded in
/tmp/dslcap-c-dsview-live-verifier.md.

The worker independently decoded both headers, checked their transfer identity
and ordering, matched both arm images to the references, and confirmed the
serial force-command/header sequence. Its conclusion agrees with the results
above. The serial real_pos exceeds the aligned uploaded count and is not a
valid event marker when the hit bit is clear. Lead additionally verified no
DSView/dslcap process or USB holder remained after cleanup.

## Cleanup and next diagnostic

Both owner tasks completed at revision 5; reports were consumed and acknowledged.
DSView restored its session byte-for-byte, left its configuration unchanged,
exited, and released USB at 00:27:56.273 UTC. The radio owner's final check at
approximately 00:28:18 restored STBY_XOSC, errors 0, system IRQ 0, TX level 0,
and the original FIFO flags rx 0x03 / tx 0x27; mcp_radio is idle. Pi134 was
untouched. No production source was changed.
The serial positive-match gate remains blocked in both frontends. A controlled
width or slower-clock comparison would be a useful next diagnostic; no new bit
mapping, stage-count change or claimed fix follows from this result.
