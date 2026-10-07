# A2 independent measurement review

## Cycle 1 - 2026-10-07

Reviewing: **A2-clock-validation**, wave 1, group cycle **1/3**. Independent
synthesizer: review-1. Analysts: none requested or missing. Scope is the six
200M/400M clock captures, temporary offline inspector and tests, reference
preparation/cleanup records, and restoration evidence. This is measurement
validation of the unchanged Phase B implementation, not a product-code review
or a reset of another group's review budget.

### Critical / Severe / High

None.

### Medium / Low / Warning

- **Low — static-verifiable:** `dslcap/TRIGGER_PLAN.md:695` still specifies a
  direct 54 MHz oscillator divider, and `dslcap/TRIGGER_PLAN.md:713` still
  requires the historical MISO-high idle level. Actual owner register readback
  shows PLLD_PER divided by 750 without MASH, and the latest pre-A2 baseline
  already has MISO low. The accurate explanation is now in
  `dslcap/A2_VALIDATION.md:66` and `dslcap/A2_VALIDATION.md:81`. Reconcile the
  plan with that accepted measurement method and contemporary restoration
  criterion so a future operator does not repeat the invalid idle-high check.
  This documentation discrepancy does not invalidate the measured relative
  clock agreement or the active restored-MISO proof.

### Suggestion

- **Static-verifiable:** `dslcap/A2_VALIDATION.md:47` says each capture contains
  167774 rising periods. The first 200M capture contains 167774 rising edges,
  hence 167773 complete rising periods; the other five have 167774 periods.
  Say “about 167774 rising periods” or retain the per-file counts.

### Spec Alignment

The assigned ±100 ppm criterion is agreement between two clocks using the
analyzer's advertised sample rate and a nominal 1 MHz Pi reference. It is not
absolute calibration of either device. Three captures at each required rate
have exact sample counts, uint8 payloads, one samplerate META line, and CH1 as
the reference. Integer-divider PLLD_PER is the explicitly disclosed actual
method, rather than the proposal's raw-oscillator example.

The kernel's modeled PLLD_PER frequency of 750000023 Hz divided by 750 gives
1000000.0306667 Hz; this modeled approximately +0.03 ppm is not a physical
frequency measurement. MASH=0 and DIVF=0 establish no fractional GPCLK divider,
not a measured bound on oscillator/PLL jitter.

### Cross-Task Consistency

Worker results, lead arithmetic, capture manifests, and independent full-array
recomputation agree on all six files. Raw hashes and counts were checked, not
just the reported PASS fields. The inspector's chunk-boundary handling and
exclusion of partial endpoint runs agree with direct transition extraction.

The two restored idle captures match the latest saved pre-A2 baseline, not the
older A0 snapshot. The lead's initial historical MISO-high assertion failed;
that invalid baseline assumption is explicitly retained in the validation
record. A later exploratory post-GetVersion idle-high assertion also failed;
actual MISO remains low after a correctly decoded response. Neither failed
assumption is evidence of miswiring or a changed production implementation.

### Security And Operations

Review performed offline only: no analyzer, Pi, service, GPIO, application, or
Git actions. Source/test frozen hashes matched. No new dependency or product
change was needed.

Owner evidence records isolated GPIO20, existing local-only pigpiod, running
GP0CTL=0x96 and GP0DIV=0x002ee000, then ENAB=BUSY=0, GPIO20 input/pull-down/low,
daemon stopped/inactive/disabled, and unchanged other relevant GPIO/application
state. The disabled volatile clock registers retain divisor/stop state; cleanup
does not claim byte-identical register restoration. The clock-generation task
reported no radio SPI/RF/configuration/reset operations. The later, separately
authorized restoration task performed exactly one read-only GetVersion.

Owner result body hash and readiness-envelope hash were independently checked.
An initial reviewer attempt incorrectly treated the readiness hash as body-only;
the documented six-field message-envelope digest resolved that check. It was a
reviewer schema assumption, not an unresolved integrity failure.

### Verification And Test Adequacy

Executed offline commands:

- `python3 -B /tmp/dslcap-a2-worker-test.py`: exit 0, **8 tests passed**.
- `sha256sum -c /tmp/dslcap-a2-worker-frozen.sha256`: exit 0, **2/2 matched**.
- Repeated for each of the three files at each rate:
  `python3 -B /tmp/dslcap-a2-worker-inspect.py /tmp/dslcap-a2-live/RATE-i.raw --samples N --rate R --frequency 1000000`.
  Substitutions were RATE=200M, N=33554432, R=200000000 and RATE=400M,
  N=67108864, R=400000000, with i=1,2,3. All six exited 0 with PASS and
  `quantization_resolves_ppm_limit` true.
- Independent inline Python/NumPy checks loaded all six complete payloads,
  verified header/count/hash/manifest exit status, extracted CH1 transitions
  with `flatnonzero(bits[1:] != bits[:-1]) + 1`, selected rising edges, and
  recomputed frequency with exact `Fraction` arithmetic. Complete high/low run
  histograms were recomputed from adjacent transition differences. All passed.
- Independent inline Python checks verified pre-A2 and both restored idle
  payload hashes and low-nibble levels, and decoded the supplemental restored
  SPI capture directly from CH0 rising edges, CH3 NSS, CH2 MOSI and CH1 MISO.
  All passed.

Independent frequency results use R × (rising edges − 1) / span:

| Capture | Rising edges | Span, samples | Relative ppm | Endpoint quantization bound, ppm |
| --- | ---: | ---: | ---: | ---: |
| 200M-1 | 167774 | 33554093 | +15.109930106 | 0.029803075 |
| 200M-2 | 167775 | 33554293 | +15.109840043 | 0.029802897 |
| 200M-3 | 167775 | 33554294 | +15.080037148 | 0.029802895 |
| 400M-1 | 167775 | 67108586 | +15.109840043 | 0.014901448 |
| 400M-2 | 167775 | 67108587 | +15.094938596 | 0.014901448 |
| 400M-3 | 167775 | 67108588 | +15.080037148 | 0.014901447 |

For every file, absolute measured ppm plus endpoint quantization bound is
below 100. Every complete high run is 100/101 samples at 200M or 200/201 at
400M; every complete low run is 99/100 or 199/200 respectively. No short or
outlying complete runs were found. Each level has two adjacent lengths;
combining levels naturally produces three. The high/low mean differences
satisfy the inspector's documented one-sample-plus-endpoint allowance.

The eight synthetic tests exercise exact clean results, signed frequency
offsets including out-of-tolerance cases, chunk boundaries/channel selection,
partial endpoints, bad metadata/rate/count, missing clock, glitches, and bad
duty cycle. The inspector separates observed PASS from whether sample
quantization resolves the ppm criterion; short synthetic examples are not
precision evidence. The real captures independently satisfy both conditions.

Restoration evidence independently checked:

- `/tmp/dslcap-b-deadline-final/upload-1.36.raw`: latest pre-A2 saved baseline
  has CH0–3 = 0,0,0,1 throughout. Both 200M and 400M restored idle captures
  contain 1000448 samples with those same levels and successful exit status.
- `/tmp/dslcap-a2-restored-active/capture.raw`: SHA256
  `c7ae7d2109843c42121eccee622993ae7e2633d501867a8cb657194d1900b83e`,
  400M, 2000896 samples, trigger index 200092, capture exit 0. Independently
  decoded two NSS frames: indices 200092–203893, 16 rising clocks,
  MOSI/MISO `0101`/`0452`; indices 274044–277046, 32 rising clocks,
  `00000000`/`06520118`. These agree with the application reference and
  owner-returned version `01 18`, status 0. This is active continuity evidence
  beyond matching static idle levels.

### Open Live Validation

None for the narrowly assigned A2 relative-clock and restoration acceptance:
six real captures, owner cleanup, user wiring confirmation, matching recent
idle baseline, and active restored SPI are present. Hardware state readbacks
remain owner-supplied evidence; the reviewer independently verified saved data
without accessing hardware.

The quantization bounds cover sample endpoint uncertainty only. They do not
bound physical reference-clock accuracy, analog threshold effects, PLL jitter,
environmental drift, or future stability. These captures do not establish
absolute calibration or exclude every possible hardware fault. This verdict
does not close unrelated trigger, capture-mode, or future plan gates.

### Verdict: PASS

No Critical, Severe, High, or Medium findings; **1 Low, 0 Warning, 1 Suggestion**.
All valid executed checks pass. The six measured relative errors and run-length
distributions meet the assigned criterion, and restoration has active evidence.
