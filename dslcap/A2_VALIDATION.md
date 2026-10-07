# A2 continuous-clock validation — 2026-10-07

Status: **PASS**, measurements, restoration and independent review complete.
User moved CH1 from pi133 radio MISO to isolated GPIO20,
physical pin 38, and disconnected CH6. CH1 can be sampled at both 200M and
400M, so no second clock lead is needed. Existing ground remains connected.

Scope: the existing TRIGGER_PLAN.md A2 acceptance, relative clock agreement
within ±100 ppm and clean run lengths. No new dependencies or product changes.
The reference is the Pi clock, not a calibrated laboratory frequency standard.
Machine-readable capture hashes, histograms, timing and cleanup evidence:
[A2_VALIDATION.json](A2_VALIDATION.json).

Owners:

- DS_LEAD: exclusive analyzer access, captures, integration, this record and
  final scoped commits.
- Existing dslcap_worker: temporary offline inspector and synthetic checks;
  no repository/hardware/Git writes. Handoff /tmp/dslcap-a2-worker.md.
- Radio owner hydra-develop-26: secondopinion task dslcap-a2-clock-20261007,
  pi133 preflight, verified integer-divider clock generation, bounded hold,
  daemon/pin restoration. No radio configuration or connected-radio GPIO changes.

Preflight lead readback: GPIO20=input/pull-down/low, GPIO6=input/pull-up/high;
installed pigs/pigpiod, daemon inactive/not running. Owner rechecked before
generation and established the actual clock source/divider. Selected the
plan-permitted 1MHz reference after verifying the PLL and integer divider.
Absence of fractional clock division was established by register readback.

Completed captures: three 200M/CH0–7/N33554432 and three
400M/CH0–3/N67108864, using unchanged reviewed production binary.
Afterward the owner stopped the clock, restored GPIO20 input/pull-down and
stopped the daemon started for this task. User returned CH1 to MISO before
the final idle-level restoration captures, as detailed below.

## Completed measurements

All six captures exited 0 with exact requested counts and matching hashes.
Lead arithmetic and the worker's independently tested inspector agree:

| Rate | Capture 1 ppm | Capture 2 ppm | Capture 3 ppm | Quantization bound |
| --- | ---: | ---: | ---: | ---: |
| 200 MSa/s | +15.109930 | +15.109840 | +15.080037 | <0.029804 ppm |
| 400 MSa/s | +15.109840 | +15.094939 | +15.080037 | <0.014902 ppm |

Frequency = advertised sample rate × complete rising periods / first-to-last
rising-edge span. Each capture spans about 0.168s; the first contains 167773
complete rising periods and the other five contain 167774.
Positive ppm here means the Pi reference appears faster when expressed in the
analyzer's advertised timebase. It does not identify either oscillator's
absolute error. The ±100 ppm relative-agreement criterion passes.

At 200M, every complete high run is 100/101 samples and every complete low
run is 99/100. At 400M, high runs are 200/201 and low runs 199/200. No short
or outlying runs. Each level uses two adjacent integer lengths; combining
both levels produces three lengths because their average widths differ
slightly. Partial runs at capture boundaries are excluded.

Worker verifier: eight synthetic tests passed, including wrong metadata/count,
missing clock, corrupt/glitch runs, bad duty cycle, chunk boundaries and
frequency offset. Six live inspections passed; source/test hashes are frozen.
Artifacts: /tmp/dslcap-a2-worker-handoff.md, -method.md, -results.json and
/tmp/dslcap-a2-live/. Production source and executable are unchanged.

## Reference source and restoration

The owner rechecked isolation/ownership and started the existing local-only
pigpiod service. `pigs hc 20 1000000` selected PLLD_PER, not the raw oscillator:
GP0CTL=0x96 (source 6, enable 1, MASH 0), GP0DIV=0x002ee000 (integer divisor
750, fractional part zero). The kernel models PLLD_PER as 750000023Hz,
giving a nominal 1000000.03Hz reference; that modeled +0.03 ppm is not a
physical calibration. The integer GPCLK divider has no MASH dithering.
No direct clock-register writes, new package or persistent configuration.

STOP cleanup verified enable/busy zero, GPIO20=input/pull-down/low, pigpiod
inactive/disabled with no process, other relevant pins unchanged, same radio
application process and no SPI/RF/configuration/reset activity. The disabled
volatile clock registers retain divisor 750 and stop bits; they are not
byte-identical to baseline, but no clock is running. Owner final result
revision 5 was consumed and acknowledged.

User confirmed CH1 restored to pi133 MISO and CH6 disconnected. Both 200M
and 400M idle restoration captures succeeded with CH0–3 levels 0,0,0,1.
Initial checks expected the historical A0 MISO-high snapshot and failed;
the actual latest pre-A2 capture already had MISO low after the prior status
reads. Comparing that saved pre-test baseline confirms matching restored
levels. Source /tmp/dslcap-b-deadline-final/upload-1.36.raw; restoration
evidence /tmp/dslcap-a2-restored/. Static levels complement user confirmation
and pin/application readback; they alone do not prove wire continuity.

To positively verify restored continuity, the owner performed exactly one
read-only GetVersion on pi133 while a 400M capture was armed. The two complete
NSS frames decoded MOSI/MISO `0101`/`0452` (16 clocks) and
`00000000`/`06520118` (32 clocks), matching the established application
reference. The owner independently returned 01 18, status 0. MISO still idles
low afterward, so an idle-high assertion is not an invariant for this bench.
No radio configuration, GPIO or service changes; no other SPI activity.
Artifacts: /tmp/dslcap-a2-restored-active/. Task
dslcap-a2-restore-spi-20261007 revision 4 was consumed and acknowledged.

## Independent review

Distinct task group A2-clock-validation, wave 1: cycle 1 of 3 allocated before
launching review-1. Sole reviewer-owned file: A2_REVIEW.md. No analysts.
Handoff /tmp/dslcap-a2-review.md; read-only raw data and temporary inspector.
No prior findings in this group; Phase B's review budget is separate.

Cycle 1 PASS: [A2_REVIEW.md](A2_REVIEW.md). The reviewer independently checked
all six raw captures and hashes, eight synthetic tests, exact frequency
arithmetic, per-level run histograms, cleanup records, restored idle baseline
and active SPI bytes. No open A2 live gate remains. One Low documentation
finding and one wording suggestion were reconciled in the final plan and
this record; no production code or verifier changed after review.
