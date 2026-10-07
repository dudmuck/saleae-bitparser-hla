# Simple-trigger validation — 2026-10-07

Implementation: `fb85f37`, built against `/home/wroberts/DSView-1.3.2`.
Independent review cycle 1 PASS: [TRIGGER_REVIEW.md](TRIGGER_REVIEW.md).
Offline verification independently reproduced 5/5 CTest groups and 36 Python
tests. Review retains a Low duplicate `--on-timeout` action issue and a
Warning about fixed-sleep tests not proving which phase received a signal.
No new package, DSView source modification, radio configuration or NVM write.

Machine-readable evidence: [TRIGGER_VALIDATION.json](TRIGGER_VALIDATION.json).
Phase B gates B.1 through B.6 passed. A2 was separate and open at this handoff;
it subsequently passed with independent review and restoration evidence in
[A2_VALIDATION.md](A2_VALIDATION.md). Phase C has not started.

## B.1 — real DSView register equivalence: PASS

The production binary ran through the same libusb logging shim as the real
DSView application. All three 372-byte FPGA setting images were identical to
the reference bytes in [TRIGGER_GOLDEN.json](TRIGGER_GOLDEN.json):
100M/200M with CH0–7 and N=1000448; 400M with CH0–3 and N=2000896.
All used CH3 falling, position 10%, VTH 1.6. The idle runs exited 16 as expected.
Temporary raw images, decoded fields and logs: `/tmp/dslcap-b-golden-live/`.

## B.4 — idle timeout, forced upload and waiting cancellation

All cases used the production binary, 100M, CH0–3, with no radio traffic.
Every case was followed by an immediate successful `--scan` reopening.

| Case | N | T | Exit | Output samples |
| --- | ---: | ---: | ---: | ---: |
| Fail, CH3 falling | 1000448 | 0.2s | 16 | 0 |
| Upload, CH3 falling | 67108864 | 0s | 0 | 6710272 |
| Upload, CH3 falling | 67108864 | 0.1s | 0 | 6710272 |
| Upload, CH3 falling | 67108864 | 1s | 0 | 6710272 |
| Ctrl-C while waiting | 1000448 | unlimited | 130 | 0 |

Fail and cancellation emitted only the 27-byte samplerate line. All uploads
reported `META trigger: none`, status 0, remain 60398015 and aligned actual 6710272;
their payload lengths matched that count. With 10% position the untriggered
FPGA retained approximately the pre-trigger region, not N samples, even
after the requested capture duration had passed. This is measured behavior,
not an assumption that timeout determines returned length.

The 0s timeout still includes the nominal 340ms grace. It therefore does not
mean an immediate hardware stop. All cached-status reads succeeded; the
diagnostics explicitly reported unchanged-cache freshness uncertainty. Total
process times include roughly initialization and cleanup as well as waiting;
they are not hardware-event-time deadlines.

The post-trigger cancellation test observed cached hit=1 before any trigger
header, then sent SIGINT. It returned 130 with only the samplerate line and
reopened successfully. This supplies live evidence beyond the fixed-sleep
offline tests. Artifacts: `/tmp/dslcap-b-post-cancel/`.

Artifacts: `/tmp/dslcap-b-idle-live/`.

## B.2 — repeated trigger conditions: PASS

Five 100M conditions passed: CH0 falling, rising, high, either edge, and
CH0 rising AND CH3 low. Every condition matched exactly at returned real_pos;
every capture emitted 1000448 samples. The first inspection attempt exposed
a NumPy uint8 mask overflow in the lead's temporary inspector; its width mask
was corrected, the saved capture was re-inspected, and a fresh five-case
smoke completed. No producer change was needed.

The application owner generated 7936 checked GetVersion requests on pi133
over 45.214s with no errors, then returned idle. No GPIO or radio state was
changed. A separately bounded traffic wave supplied the full matrix.

All 300 matrix captures passed: 20 repeats of each of the five conditions at
each of 100M, 200M and 400M. All exited 0 with exactly 1000448 samples.
The requested condition held at the exact returned real_pos in every case;
measured tolerance was zero samples at all three rates. This result applies
to these inputs and captures, not every possible signal timing. Artifacts:
`/tmp/dslcap-b-matrix/`, including each raw capture, header/count diagnostics,
independent decoded SPI frames and incremental summary.

## B.3 — placement and early trigger: PASS

With N=65536 and a rising SCLK trigger, returned positions were:

| Rate | Requested 0% | 10% | 50% | 90% |
| --- | ---: | ---: | ---: | ---: |
| 100M | 125 | 6580 | 32817 | 58987 |
| 200M | 106 | 6567 | 32795 | 58970 |
| 400M | 119 | 6565 | 32813 | 59007 |

Each lies within the 64-sample block starting at the programmed effective
position (64, 6528, 32768, 58944). Every returned sample is the actual
matching rising edge. Thus the 64-sample setting granularity does not imply
that metadata must round the detected edge down to that block boundary.

With already-running periodic SPI traffic, N=10004480 and requested 90%,
early hits returned 153134 / 100515 / 2023021 at 100M / 200M / 400M,
respectively. All were far below 90% and matched the real edge exactly;
META correctly described the shorter pre-trigger region. The source is
bursty periodic SPI, not a continuous reference oscillator.
Artifacts: `/tmp/dslcap-b-positions/`.

## B.5 — timed pulse and timeout races: PASS

Read-only pi133 inspection found GPIO20 unclaimed/input/pull-down/low; the
application owner confirmed it is unused. The operator answered the isolation
and wiring request by confirming CH6→GPIO20, physical pin 38. Only that pin
was driven temporarily; radio-connected pins were never driven directly.

Each run used 100M, CH0–7, N=33554432, position 10%, CH6 rising and T=1s.
An existing Python/SSH/pinctrl path scheduled a 5ms high pulse. No new tool
or service was installed or started. Host/Pi monotonic-clock offset was
estimated using five exchanges and the lowest round-trip time; the evidence
records that RTT and the before/after bounds of each pin-write command.
Offsets below refer to the first host-observed arm/status diagnostic, within
polling/scheduling uncertainty of arming; this is not an FPGA event deadline.

| Pulse offset | `fail` outcome | `upload` outcome |
| --- | --- | --- |
| 0.50s | Complete, triggered, 33554432 samples | Complete, triggered, 33554432 samples |
| 0.98s | Complete, triggered, 33554432 samples | Complete, triggered, 33554432 samples |
| 1.02s | Complete, triggered, 33554432 samples | Complete, triggered, 33554432 samples |
| 1.30s | Exit 16, samplerate line only | Triggered, 7239680 samples, exit 0 |
| 1.36s | Exit 16, samplerate line only | Untriggered, 3354624 samples, exit 0 |

All numeric markers matched the actual CH6 rising edge exactly. Forced-upload
payload counts equaled `min(N, (N-remain)&~1023)` from the hardware header.
The 1.30s upload retained a real trigger marker despite the timeout/force
race; the 1.36s upload correctly reported none. Committed aborts never
reported success or emitted sample data. Every case reopened successfully.

Observed timeout decisions were 1.34010–1.34150s after the arm observation,
consistent with the nominal 340ms grace. All status reads succeeded, while
the cache was still reported as freshness-unknown for deadline decisions.
The result demonstrates the documented best-effort behavior, not a promise
to classify a physical edge strictly before or after T.

An earlier temporary inspector wrongly required a full N for a triggered
forced upload. Its assertion stopped at a valid short capture. The verifier
was corrected to use header-derived counts, decision timestamp observation
was added, and all ten cases were rerun successfully. No producer change
was needed. GPIO20 was restored after every pulse and independently re-read
after the sweep as `20: ip pd | lo // GPIO20 = input`.
Artifacts: `/tmp/dslcap-b-deadline-final/`.

## B.6 — live Python/HLA: PASS

The existing LR2021 HLA decoded the production `--dsl-mode buffer` pipeline:

- pi133 at 400M, CH0–3, N=67108864, CH3 falling, position 10%.
- Both Pis at 100M, physical CH0–3 and CH8–11, N=33554432, CH11 falling,
  position 40% to cover the host-induced skew between independent Pi starts.

The owner completed readiness reads before arming, then emitted exactly 20
GetVersion requests per participating Pi after GO, followed by at least 1.5s
of idle before postflight reads. All 60 application responses across these
two runs were 01 18 with no errors. Both Pis retained their original modes,
IRQ state and error-free status.

Both pipelines exited 0 with full requested sample counts. Single output
contained exactly 20 requests and 20 v1.24 responses. Dual output contained
the same count per port. After removing timestamps and the single-port label,
all 120 lines per port (decoded messages and MOSI/MISO bytes) were identical
to the earlier 25M streaming capture of the same command sequence. Dual
output and its single trigger marker were chronologically ordered.

This compares per-port sequences; cross-port interleaving naturally differs
between independent runs. It is not a simultaneous Saleae reference capture.
Artifacts: `/tmp/dslcap-b-hla-single/`, `/tmp/dslcap-b-hla-dual/` and the
earlier reference `/tmp/dslcap-radio-live1.stdout`.

## Remaining scope

A2 continuous-clock validation subsequently passed; see the linked record.
Serial triggering and
optional trigger-relative timestamps were excluded from this Phase B wave.
The reviewed duplicate C CLI timeout-option issue remains Low; specify
`--on-timeout` once. Historical repeated-Python-interrupt cleanup limitations
are unchanged. No active capture or GPIO generator remains after validation.

The radio owner’s final result (revision 4) was consumed and acknowledged:
79,676 verified GetVersion requests, zero errors, no generator processes or
armed loops, both services idle. Last observed states remain STBY_XOSC on
pi133 and STBY_RC on pi134, with zero IRQ/error status. The application
owner made no GPIO, RF, reset, configuration, driver or service changes.
