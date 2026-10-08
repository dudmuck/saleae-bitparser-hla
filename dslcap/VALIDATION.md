# dslcap hardware validation

This record distinguishes measured behavior from open acceptance gates.
Reference sources: `/home/wroberts/DSView-1.3.2/`. Analyzer: DSLogic Plus
PGL12 USB `2a0e:0034`, firmware 2.2. Tests dated 2026-10-07.

## Phase 0 — reviewed in commit 47d9d6d

- Standalone build and offline CTest passed independently for worker, lead,
  and review-1: 19 driver scenarios and three invalid firmware paths.
- A DSView-owned interface caused exit 5 with empty stdout; the frontend
  rejected the driver's successful return after fallback to the demo device.
- After the operator closed DSView, activation and release/reopen passed.
  Each activation produced security-pass evidence; an explicit HDL read was
  `0x0e`. Lead independently reproduced this from `/tmp/dslcap-lead-review`.
- The explicitly enabled hardware reload test called the established driver
  upload routine with `DSLogicPlus-pgl12-2.bin`. The driver reported 530620
  bytes uploaded. Security and HDL checks passed afterward, and the production
  binary immediately reopened successfully. No NVM writes were performed.
- Invalid firmware directory exited 3 before scanning, followed by successful
  production reactivation.
- G0 review cycle 1 PASS is recorded in REVIEW.md. Cold automatic startup
  upload remains unrun; explicit upload is not equivalent to that branch.

## Phase 1 — G1 cycle 1 review PASS, live gates remain

- First stream capture at 25M, channels 0-7, requested 100001 samples:
  exit 0, META samplerate 25000000, exactly 100001 payload bytes.
- Bounded finite captures of channels 0-11 at 25M, 0-15 at 20M and sparse
  0,3,9,15 at 25M returned exact sample counts and exit 0.
- Internal-pattern capture returned 16777216 uint16 samples; every sample
  matched `sample[i] = i modulo 65536` (no mismatches).
- Lead independently rebuilt the frozen source with RelWithDebInfo and all
  four CTest suites passed. Review-1 independently repeated this and returned
  G1 cycle 1 PASS; details and limitations are in REVIEW.md.
- Worker sanitizer checks passed for conversion/parser/ring. In-memory 8-bit
  conversion measured 714.6 MB/s, which is not a USB throughput measurement.
- Worker live SIGINT returned 130, EPIPE returned 12, and an unconsumed pipe
  caused ring-full exit 9 in about 13.84s including startup and bounded drain.
  Ring high-water reached 268430144 bytes; secondary drain timeout explicitly
  reports discarded output while preserving exit 9. Immediate scan passed
  after each test. The nonaligned 100M buffer test returned exactly 100001
  samples. Unsupported 50M/8, 60M/2, 3M/8 and excess buffer depth were rejected.

## Independent ring wrap verification

After the G1 review identified missing wrap-content coverage, lead compiled
a standalone verifier against the frozen `ring.c`. A concurrent
pipe consumer checked every byte against sequence offset modulo 251. The
producer sent 16777353 bytes in 4090 irregular pushes through a 98317-byte
ring: 170 complete capacity crossings. Both final cursors and total bytes
were checked. Command exited 0:

```sh
cc -std=gnu11 -O2 -pthread -I dslcap dslcap/tests/review_ring_wrap.c dslcap/ring.c -o /tmp/dslcap-ring-wrap
timeout 10s /tmp/dslcap-ring-wrap
```

This closes the immediate byte-preservation evidence gap for wraparound;
adding the case to permanent CTest remains useful. The review's Low finding
about cross-thread `volatile sig_atomic_t` publication remains recorded;
no missed cancellation has been reproduced on this target.

## Pi-generated pulse evidence

The operator connected DSLogic CH0 to pi133 physical pin 12 (GPIO18) and
confirmed analyzer ground connected to Pi ground. Before driving, GPIO18 was
unclaimed input with pull-down. Other pins were not changed.

Lead ran a bounded 120-second Python/gpiod 1.6.3 sequence: high 2ms, low 3ms,
high 5ms, low 7ms, repeated. This is software-timed with scheduling overhead,
not a precision frequency source. The helper logged monotonic timestamp
brackets around every GPIO update. No package installation or daemon was
needed. It generated 27512 edges over approximately 120.002 seconds.

Worker captured each of these configurations while the sequence ran:

| Channels | Rate | Samples | CH0 transitions | Result |
| --- | --- | --- | --- | --- |
| 0-7 | 25M | 5000000 | 45 | Exit 0, alternating expected pulse widths |
| 0-11 | 25M | 5000000 | 46 | Exit 0, alternating expected pulse widths |
| 0-15 | 20M | 4000000 | 46 | Exit 0, alternating expected pulse widths |
| 0,3,9,15 | 25M | 5000000 | 45 | Exit 0, alternating expected pulse widths |

Observed widths were approximately 2.1/5.1ms high and 3.1/7.1ms low,
consistent with the requested sequence and software overhead. Only CH0 was
wired: these tests do not establish physical high-channel mapping or SPI
decoding. Worker artifacts are `/tmp/dslcap-gpio-{8ch,12ch,16ch,sparse}`
with `.raw`, `.json`, and `.stderr` extensions. Lead inspected summaries.

After completion, lead verified `pinctrl get 18` reports `ip pd lo`, and
`gpioinfo gpiochip0` reports GPIO18 unused/input. No signal generator remains
running. Temporary source and timing log: `/tmp/dslcap-pi133-pulses.py` and
`/tmp/dslcap-pi133-pulses.log`.

## Open gates

- Same-traffic Saleae/Logic 2 comparison remains unavailable. Live dual-radio
  LR2021 operation on pi133/pi134 now passes (below), as does 10 MHz
  kernel-driver ping-pong traffic ([10 MHz kernel traffic](#10-mhz-kernel-ping-pong-traffic--2026-10-08)).
  Mixed LR1110/LR2021 and future IRQ/BUSY wiring were not exercised.

**Current wiring (since 2026-10-08 09:38 PDT).** CH0–3 pi133 SPI, CH4 pi134
DIO11 (physical pin 18, GPIO24; RF-switch witness driven by `lr2021_pcycle`),
CH5 pi133 DIO8, CH8–11 pi134 SPI, CH12 pi134 BUSY, CH13 pi134 DIO8; mask
`0x3f3f`. Before that, CH4 was pi133 BUSY (pin 12). Sections below record the
wiring in use at the time of each test.

Raw captures and full temporary logs are intentionally not committed.

## Phase 2 continuous stream — PASS

Lead ran `/tmp/dslcap-lead-review/dslcap --continuous --samplerate 25M
--channels 0-7` with stdout drained directly to `/dev/null` and stderr to a
regular log file, using the existing Python standard library for timing.
No `pv` installation was needed. The process remained alive for 610.063s,
then received SIGINT and exited 130. It reported 15230320128 samples,
15230320154 output bytes (including META), and ring high-water 483904 bytes.
No ring/FPGA overflow or collection error was reported. Immediate verified
`--scan` exited 0 with security passes and HDL 0x0e.

Tested executable SHA-256:
`9ca3447e47a519d0c936628e18f0e4675a0133128b6c33b085c6f6c09f57425f`.
Full result: `/tmp/dslcap-phase2-long.json`; stderr:
`/tmp/dslcap-phase2-long.stderr`. This proves sustained transport and bounded
shutdown on this setup, not the contents of every discarded sample.

## Phase 2 real FPGA overflow — PASS

Lead started a continuous 25M x 8 capture to `/dev/null`, waited three
seconds, sent SIGSTOP to that child only, waited three seconds, and sent
SIGCONT. The resumed frontend received the real driver overflow report and
exited 10 before the 20-second observation timeout. Its diagnostic explicitly
warned that detection is delayed and preceding output may be suspect.
Reported samples: 56025088; output bytes: 56025114; ring high-water: 130560.
Immediate verified scan exited 0. No debug flag, excessive channel
configuration, firmware/NVM change, or dependency was needed.

Full result: `/tmp/dslcap-phase2-overflow.json`; stderr:
`/tmp/dslcap-phase2-overflow.stderr`.

## Phase 3 Python integration — implementation and live smoke PASS

Worker handoff: `/tmp/dslcap-python-handoff.txt`. Lead independently ran
`python3 -m unittest discover -s tests -p 'test_dslcap_backend.py' -v`:
20 tests passed in 14.041s. Review-1 independently passed the same 20 tests
in 14.123s and issued G2 cycle 1 PASS. REVIEW.md retains a Medium finding:
a second interrupt during cleanup can leave a TERM-ignoring child alive and
mark cleanup finished, preventing retry. Single-interrupt cleanup is tested;
unconditional repeated-interrupt cleanup is not claimed. The review workflow
does not initiate an automatic fix wave after PASS.

Lead ran two SPI decoder ports (0,1,2,3 and 4,5,6,7) through `--dslogic
--dslcap /tmp/dslcap-lead-review/dslcap -vv --samplerate 25M --time 30s
--hla-path tests/dslcap`. The pipeline exited 0 after 31.381s, handling
750000000 samples with ring high-water 500224 bytes. Immediate verified
scan exited 0. No generated SPI traffic was active during this throughput
smoke; it does not prove performance under heavy transaction traffic.
Real stderr totaled 4177 bytes, so the approximately 198KiB fake-producer
tests supply the pipe-capacity pressure evidence. Artifacts:
`/tmp/dslcap-python-live.{json,stdout,stderr,scan}` and runner
`/tmp/dslcap-python-live.py`.

The existing GPIO18 pulse sequence was then repeated for ten seconds.
A two-second Python capture at 25M with `--spi 1,2,3,4 --extra-pin 0`
processed 50000000 samples, exited 0, and logged 462 physical CH0 edges.
Intervals ranged from 2.06824ms to 7.19436ms, consistent with the requested
2/3/5/7ms software-timed sequence. GPIO18 was restored and independently
read back as input/pull-down. Artifacts: `/tmp/dslcap-python-pulses.*`.

## Known physical SPI and reference decode — PASS

Operator confirmed CH0→GPIO18/pin12, CH1→GPIO17/pin11,
CH2→GPIO27/pin13, CH3→GPIO24/pin18, plus common ground.
Lead drove these as CLK/MISO/MOSI/CS, CPOL0/CPHA0, with a bounded
20-second gpiod sequence. Each transaction sent MOSI `01 00 55 aa` and
MISO `0f a5 5a c3`. A live two-second 25M Python capture exited 0 and
decoded 24 complete transactions, every byte matching both expected streams.
This verifies the four connected channel roles and live shared decoder.

A separate one-second 25M raw capture exited 0 with exactly 25000000
sample bytes after META removal. The same saved samples were decoded by
the shared NumPy engine and independently by installed sigrok-cli's SPI
protocol decoder:

```sh
sigrok-cli -i /tmp/dslcap-known-spi.bin \
  -I binary:numchannels=8:samplerate=25000000 \
  -P spi:clk=0:miso=1:mosi=2:cs=3 \
  -A spi=miso-transfer:mosi-transfer --protocol-decoder-samplenum
```

Both decoded 11 complete transactions with identical expected MOSI/MISO
bytes. Capture-boundary partial transactions were excluded explicitly:
sigrok emitted a partial starting transaction, and NumPy flushed one byte
at EOF (`01`/`0f`) from the unfinished final transaction. The initial
comparison assertion incorrectly included this EOF partial; the corrected
comparison checks every complete transaction and the expected partial.
This is an independent decoder comparison, not an independent DSView or
Saleae hardware capture, and not real dual-radio traffic.

Generator and both captures exited 0. Independent `pinctrl get 17,18,24,27`
confirmed every driven pin restored to input/pull-down. No GPIO generator
remains active. Artifacts: `/tmp/dslcap-known-spi*` and
`/tmp/dslcap-python-signal-validation.json`; raw captures are not committed.

## Physical CH9/CH15 mapping — PASS

Operator connected separate Pi pins: DSLogic CH9 to pi133 GPIO5/physical
pin29, and CH15 to GPIO6/physical pin31. Existing CH0–CH3 and ground remained
connected. Both additional GPIOs were unclaimed inputs with pull-up before
the test. No lead sharing or extra hardware was required.

Lead drove GPIO5/GPIO6 through `00 -> 10 -> 11 -> 01`, holding these states
for 11/23/37/53ms respectively, for a bounded 20 seconds. Expected CH9
high/low intervals are 60/64ms; CH15 intervals are 90/34ms. This distinguishes
the two physical inputs and detects swapped or incorrectly packed bits.
These are software-timed pulses, not a precision frequency reference.

All three captures used `--time 700ms` and exited 0:

| Channels | Rate | Exact samples | Verified physical inputs |
| --- | --- | --- | --- |
| 0–11 | 25M | 17500000 | CH9: 12 edges |
| 0–15 | 20M | 14000000 | CH9: 11 edges; CH15: 12 edges |
| 0,3,9,15 | 25M | 17500000 | CH9: 11 edges; CH15: 11 edges |

Lead parsed the required META header and little-endian uint16 samples,
checked exact counts and zero disabled bits, and verified every complete
pulse interval within 5ms of its programmed value. Observed CH9 high widths
were 60.127–60.187ms and low widths 64.130–64.217ms; CH15 high widths were
90.130–90.219ms and low widths 34.127–34.188ms. Full and sparse captures
also preserved the exact four-state transition order. The 12-channel case
correctly excludes CH15. This covers representative high physical bits;
it does not claim every one of the 16 inputs was separately wired.

Worker dslcap_worker independently parsed all three immutable captures
without production converter code. It confirmed exact headers/counts,
zero disabled bits, all pulse signatures and four-state ordering with no
mismatches. Every complete pulse was within 2ms of nominal; actual overhead
was approximately 0.13–0.22ms. Its temporary text/JSON reports are
`/tmp/dslcap-high-worker-verification.{txt,json}`; the JSON also records
capture SHA-256 hashes and complete edge intervals.

The GPIO helper exited 0 and restored both pins to input with their original
pull-up settings. Independent `pinctrl get 5,6` confirmed restoration.
Immediate analyzer `--scan` exited 0 with both security checks passing and
HDL 0x0e. No generator or capture remains running.

Temporary evidence: `/tmp/dslcap-high-{12ch,16ch,sparse}.raw`, matching
`.stderr` files, `/tmp/dslcap-high-capture.json`, `/tmp/dslcap-high-analysis.json`,
`/tmp/dslcap-high-gpio.log`, and runner/helper
`/tmp/dslcap-high-{test,gpio}.py`. Raw captures are intentionally not committed.

## Independent DSView stream comparison — PASS

Operator captured the repeating Pi SPI and high-channel pulse signals in
DSView 1.3.2 and saved native `.dsl` archives. The first file, named
`/tmp/dsview-8ch-25M.dsl`, actually included CH9 and was retained as extra
nine-channel evidence. A corrected exact-eight-channel file was supplied.
All four required configurations were then verified against new dslcap
captures using the same generator sources after DSView was closed:

| DSView archive in /tmp | Rate | DSView samples | dslcap samples | Result |
| --- | --- | --- | --- | --- |
| dsview-exact8-25M.dsl | 25M | 25000960 | 25000000 | PASS, CH0–CH7 |
| dsview-12ch-25M.dsl | 25M | 25000960 | 25000000 | PASS, CH0–CH11 |
| dsview-16ch-20M.dsl | 20M | 20000768 | 20000000 | PASS, CH0–CH15 |
| dsview-sparse-25M.dsl | 25M | 25000960 | 25000000 | PASS, CH0/3/9/15 |

Every GUI archive records operation mode 1 (stream), the expected rate and
exact enabled channel set. DSView's requested one-second captures contain
1.0000384 seconds; dslcap trims to exactly one second. This is accounted
for explicitly, rather than requiring identical capture lengths or phases.

Lead verified the native format against the read-only DSView source:
`pv/storesession.cpp` saves per-physical-channel `L-N/block` byte arrays;
`pv/data/logicsnapshot.cpp:get_sample_self` selects sample bits LSB-first.
The archive header/session provides counts, physical indices and sample
rate. Each bitplane's full size matched the declared count. This extraction
does not use dslcap's converter.

In every 8/12/16-channel pair, each capture contains 11 complete CS-bounded
transactions with exactly 32 rising clock edges. All MOSI bytes are
`01 00 55 aa` and MISO bytes `0f a5 5a c3`. Capture-boundary partial frames
are excluded. Initial exact-eight validation assumed 12 complete frames;
inspection showed a clipped initial frame, and the corrected check uses
complete CS boundaries. The exact-eight DSView stream also matched the
independent sigrok decoder and shared NumPy decoder.

Sparse captures intentionally exclude MISO/MOSI, so no SPI-byte claim is
made for that case. Their clock/CS activity and physical CH9/CH15 pulse
signatures agree. Both high channels follow `00 -> 10 -> 11 -> 01 -> 00`
in the 16-channel and sparse pairs. Disabled bits in dslcap output are zero.
CH9 high/low intervals agree with 60/64ms and CH15 with 90/34ms plus Pi
software timing overhead. Across paired captures, clock high/low median
differences are at most 32.84us; high-channel median differences are at most
59.78us. CS-active medians differ by up to 1.983ms, consistent with the
accumulated software-clock differences over a transfer. Comparison limits
were 300us for clock medians and 5ms for other pulse medians. These separate
software-timed runs establish signal structure and mapping, not a precision
timebase calibration or sample-for-sample simultaneity.

The temporary comparison checker initially used a uint16 disabled-bit mask
against uint8 data and raised OverflowError. The mask was corrected to the
input dtype's width; all four comparisons were then rerun successfully with
checked subprocess return codes. No product code changed.

All matching captures exited 0. Both bounded Pi helpers exited 0, and
independent pin readback confirmed GPIO5/6 restored to input/pull-up and
GPIO17/18/24/27 to input/pull-down. Immediate analyzer reopen passed both
security checks and HDL 0x0e. No capture or generator remains active.

Durable counts, pulse summaries and input SHA-256 hashes are recorded in
[DSVIEW_COMPARISON.json](DSVIEW_COMPARISON.json). Raw files remain in /tmp.
Temporary runners/checkers: `/tmp/dslcap-dsview-matched.py`,
`/tmp/dsview-extract-check.py`, `/tmp/dsview-compare-captures.py`; detailed
pair reports: `/tmp/dslcap-comparison-{8ch,12ch,16ch,sparse}.json`.

Worker dslcap_worker independently checked archive CRCs, source-format
references, file stability/hashes and all four matching dslcap payloads.
Its separate bitplane unpacking and CS-bounded byte reconstruction uses
neither fast_spi nor dslcap conversion code. It confirmed matching rates,
physical masks, complete transaction signatures and high-channel pulse
patterns, with no mismatches. Detailed independent evidence is in
`/tmp/dslcap-dsview-worker-verification.json`.

## Cold automatic FPGA startup — PASS

Operator confirmed the DSLogic USB cable was disconnected for approximately
ten seconds and reconnected, with DSView closed. The next analyzer open
was the production frontend's `--scan -vv`, bounded by a 45-second timeout.
It exited 0; the device re-enumerated from bus1/address6 to bus1/address12.
The log explicitly records the automatic branch:

```text
Configure FPGA using "/usr/local/share/DSView/res/DSLogicPlus-pgl12-2.bin"
FPGA configure done: 530620 bytes.
Security check pass!
```

The normal verification reopen also passed security, and explicit HDL
readback was 0x0e. This run used neither the forced-reload test helper nor
any nonvolatile-memory operation. It closes the cold automatic-load gap
retained by the earlier G0 review.

A subsequent production capture, `--samplerate 25M --channels 0-7
--samples 100001`, exited 0 with the correct META header and exactly 100001
sample bytes. GPIO generators were stopped, so this is a post-load capture
and lifecycle test, not another signal-content comparison. Immediate
`--scan` after capture exited 0. No analyzer process remains running.

Artifacts: `/tmp/dslcap-cold-scan.{stdout,stderr}`,
`/tmp/dslcap-cold-capture.{raw,stderr}`,
`/tmp/dslcap-cold-reopen.{stdout,stderr}`,
`/tmp/dslcap-cold-validation.json`.
Cold scan stderr SHA-256:
`19f62576652317883b86f47b206ba81ccfe13055db4d30a7b23ec3fd711c4a1d`.
Capture SHA-256:
`4f05a2e95862ae22a069e5065a06c38cb1d371429c2977cf968d2e6e1d704fa3`.

## G3 wide Python and actual dual-radio operation — PASS

User supplied physical wiring: pi133 CH0/1/2/3 and pi134 CH8/9/10/11,
each SCLK/MISO/MOSI/NSS, and confirmed both grounds connected. The original
optional uint16 integration was activated without a new dependency. Width
is inferred from the same exact channel union as the C command; odd sample
bytes are carried after META parsing, incomplete EOF rejected. Existing
fast_spi, effective timing, HOLD/order and non-DSLogic defaults are unchanged.

Worker handoff: `/tmp/dslcap-wide-handoff.txt`. Lead independently passed
all 29 focused tests in 17.362s. Review-1's G3 cycle 1 PASS independently passed
29 tests, 194 additional split-boundary checks and Saleae dispatch validation.
All five frozen source fingerprints matched afterward. Historical G2
repeated-interrupt cleanup and G1 signal-publication findings remain recorded;
this feature does not claim to fix them.

Application ownership stayed with user-named hydra-develop-9f via the
secondopinion task `dslcap-dual-radio-20261007`. Read-only preflight found
no lr20/lr11/pcycle module loaded and both spi0.0/spi0.1 bound to spidev on
each Pi. Both applications are LR2021 builds, mode0/MSB-first, requested
8 MHz SPI. No GPIO, application, driver, radio-state, firmware or NVM changes
were made. Each GO authorized exactly 20 read-only GetVersion commands per Pi,
approximately 5 ms apart, through the existing application HAL/MCP path.

### RAW1: independent physical-byte reconstruction

The production C frontend captured physical mask0x0f0f at 25 MSa/s in the
12-channel-capable stream profile with two-byte samples. A 90-second bound
stopped it by SIGINT (expected exit 130); it recorded 2228998144 samples with
no overflow, ring high-water 1000448 bytes. Raw gzip output is about 19 MiB;
the 4.46 GB logical stream was inspected incrementally without full expansion.

Each port contained exactly 40 complete NSS frames and 960 rising clock edges,
alternating 16/32 clocks, 120 bytes per direction. All 20 pairs matched:

| Pi | Request MOSI/MISO | Response MOSI/MISO |
| --- | --- | --- |
| pi133 | 0101 / 0452 | 00000000 / 06520118 |
| pi134 | 0101 / 0421 | 00000000 / 06210118 |

Application results independently reported 01 18 for all 20 reads on each Pi,
with no errors. Decoded radio modes were STBY_XOSC on pi133 and STBY_RC
on pi134. Within-byte SCLK estimates were 7.8009 MHz and 7.8154 MHz from
3/4-sample periods. Each byte-boundary period stretched to6/7 samples;
exact expected clock counts show these were not missing edges. Bursts
were sequential, not overlapping, in this run. These measurements validate
this application's timing, not a later 10 MHz clock or arbitrary duty cycle.

An unchanged 3-second sample window was replayed through the wide Python
backend and existing `/home/wroberts/HLA/saleae_lr2021` HLA. It exited 0
with exactly 20 requests and 20 GetVersion v1.24 responses per port, in timestamp
order. Replay window offset 1037500000 samples (41.5s) is recorded separately.

### LIVE1: complete wide Python/HLA pipeline

Lead ran the real producer for 90 seconds with `--dslogic --samplerate 25M
--spi 0,1,2,3 --spi 8,9,10,11 --time 90s --hex -vv`, using that same HLA.
Capture was armed before sending GO. The application agent performed the
same 20 read-only requests per Pi exactly once. The pipeline exited 0 after
91.533 seconds, processing 2250000000 samples / 4500000026 producer bytes,
with ring high-water 2460672 bytes and no overflow. HLA output contained
exactly 20 requests and 20 v1.24 responses per port, chronologically ordered.
Immediate analyzer reopen exited 0. Debug stderr was drained throughout;
large-log pipe-pressure coverage remains supplied by the offline tests.

Application final report revision 7 was consumed, hash-validated and
acknowledged. Both runs completed without error or state changes; no test
loops or captures remain. Normal application services retain their owner.
No direct GPIO signal generation was used for these radio runs.

Durable summary and hashes: [DUAL_RADIO_VALIDATION.json](DUAL_RADIO_VALIDATION.json).
Temporary artifacts: `/tmp/dslcap-radio-run1.{raw.gz,json,stderr}`,
`/tmp/dslcap-radio-run1-inspection.json`, `/tmp/dslcap-radio-run1-window.*`,
`/tmp/dslcap-radio-run1-hla*`, `/tmp/dslcap-radio-live1.*`,
`/tmp/dslcap-radio-live1-validation.json` and reopen stdout/stderr.
The deterministic commands and application-returned bytes are the reference
for these tests; no same-traffic Saleae/Logic2 comparison is claimed because
no Saleae is attached. Sustained heavy transaction load, mixed radio families,
future IRQ/BUSY wiring and 10 MHz kernel traffic remain untested.

## Dual-radio BUSY / DIO8 validation — 2026-10-07

The user added pi133 BUSY/IRQ to CH4/CH5 and pi134 BUSY/IRQ to CH12/CH13.
BUSY is physical Pi pin 12/GPIO18; IRQ is DIO8 on pin 29/GPIO5. The application
owner verified both mappings and active-high polarity against its configuration.
Both grounds were already connected. No synthetic GPIO generators were used.

The documented `--int-pin pi133_dio8` and three `--extra-pin` arguments
selected physical mask `0x3f3f`: exactly 12 inputs, two-byte samples, 25 MSa/s.
The live Python/HLA pipeline completed 90 seconds, 2250000000 samples and
4500000026 producer bytes with exit 0, no overflow and ring high-water 983680.
Analyzer reopen passed. All timestamped output was ordered.

| Signal | Rising / falling edges | Observed high width |
| --- | --- | --- |
| pi133 BUSY / CH4 | 46 / 46 | 16.72–106.20 us |
| pi134 BUSY / CH12 | 48 / 48 | 17.12–143.72 us |
| pi133 DIO8 / CH5 | 1 / 1 | 8.89518344 s |
| pi134 DIO8 / CH13 | 0 / 0 | Not established |

Each port decoded exactly 20 GetVersion requests and 20 v1.24 responses,
matching the application's 20 returned `0118` values. These commands produced
no DIO8 edges. A subsequent bounded receive-only timeout caused pi133 DIO8
to rise 10.09512 ms after SetRx and fall when TIMEOUT was cleared. Its long
high interval reflects the separate application calls used to inspect status
before clearing. There was no RF transmission.

Before RX, both radios reported IRQ 0 and their expected standby modes.
pi133 produced TIMEOUT only. pi134 produced TIMEOUT plus ERROR, with
`RXFREQ_NO_FRONT_END_CALIB` (errors 512) for its existing RX configuration.
The test cleared only TIMEOUT and restored original standby modes; a separate
conditional cleanup then cleared pi134's sole diagnosed error and IRQ bit16.
Final application checks: pi133 IRQ 0/STBY_XOSC; pi134 IRQ 0/errors 0/STBY_RC.
The pi134 error-register baseline was not read before the test, so exact
restoration of that prior latch is unproven. Missing calibration remains;
no calibration, DIO/mask, GPIO, firmware, driver or package changes were made.
RX counters and a host IRQ semaphore may have changed during the test.

CH13 produced no physical edges despite chip IRQ status becoming active.
Its wiring and DIO8 routing/mask remain unvalidated: the existing chip DIO
configuration is write-only and was preserved. PinLogger does not emit an
initial static level, so this log establishes neither stuck-high nor stuck-low.
Do not claim all four status connections passed. Further CH13 testing needs
a known application DIO8 configuration or separate wiring diagnosis.

Native `dslcap_worker` independently checked CLI channel selection; lead
validated the output counts, ordering, widths and producer completion.
This run saved decoded output, not raw samples. Evidence and hashes:
[RADIO_PINS_VALIDATION.json](RADIO_PINS_VALIDATION.json), with temporary
`/tmp/dslcap-radio-pins-live1.{stdout,stderr,json}` and reopen logs.
Secondopinion tasks `dslcap-radio-pins-20261007` revision 4 and
`dslcap-radio-pins-cleanup-20261007` revision 3 were consumed and acknowledged.
No test activity remains. Earlier same-traffic Saleae and 10 MHz limits remain.

## pi134 swapped BUSY / DIO8 leads — 2026-10-07

The user swapped pi134 CH12/CH13: CH13 now connects to BUSY on physical
pin 12/GPIO18, and CH12 to DIO8 on physical pin 29/GPIO5. The README example
reflects this current mapping; the preceding PINS1 evidence retains its
original mapping.

SWAP1 used the same 12 selected channels at 25 MSa/s for 90 seconds. The
application owner ran exactly 20 read-only GetVersion requests on pi134 only.
Lead independently verified all 20 request/response pairs: MOSI 0101/00000000,
MISO 0421/06210118. CH13 logged 40 balanced BUSY pulses, high for 17.04–23.64 us.
CH12 logged no IRQ edges, as expected for GetVersion. pi133 logged no activity.
All 120 timestamped events were ordered. The pipeline exited 0 after processing
2250000000 samples, 4500000026 bytes, ring high-water 3148416, with no overflow.

CH13 demonstrably captures BUSY with the swapped lead. The earlier absent
IRQ edges therefore do not establish a general CH13 capture failure; pi134
DIO8 wiring/routing/mask remains unvalidated. No initial static level was
recorded. No reset, GPIO direction changes, RX/TX, calibration, IRQ clearing,
configuration changes or service stops were performed. No test remains active.
Secondopinion task `dslcap-pi134-swap-20261007` revision 4 was consumed and
acknowledged. Evidence: [PI134_SWAP_VALIDATION.json](PI134_SWAP_VALIDATION.json).

## pi134 configured RX timeout / DIO8 — 2026-10-07

The user requested an RX-timeout interrupt on pi134. The application owner
used installed APIs to establish its standard LoRa configuration and explicit
DIO8 IRQ routing; no new dependencies were needed. Separate status/error
gates before calibration, after calibration and before RX all returned IRQ 0,
errors 0 and STBY_RC. pi133 was untouched.

RXIRQ1 captured the existing 12-input mask 0x3f3f at 25 MSa/s for 90 seconds,
with CH12=DIO8 and CH13=BUSY. It exited 0 with 2250000000 samples,
4500000026 bytes and ring high-water 1787264, no overflow. All timestamped
output was ordered; 33 SPI frames matched 33 BUSY pulses.

| Captured event | Time from acquisition start |
| --- | --- |
| DIO8 falls during SetDioFunction IRQ configuration | 29.878698440 s |
| SetRx, requested 10 ms / decoded 9.979 ms | 33.652763160 s |
| DIO8 rises | 33.663058440 s |
| ClearIrq TIMEOUT frame starts | 37.596009960 s |
| DIO8 falls | 37.596031720 s |

The IRQ rose 10.29528 ms after the SetRx frame began and fell 21.76 us after
ClearIrq began. Its 3.93297328-second high interval includes the application
round trip used to inspect status before clearing. Chip status reported
TIMEOUT only, errors 0; cleanup verified IRQ 0/errors 0/STBY_RC. The initial
fall during DIO configuration shows CH12 was high immediately beforehand;
the earlier quiet IRQ logs alone did not establish a low level.

This validates physical pi134 DIO8 capture on CH12 and BUSY on CH13 with
known IRQ routing. It does not distinguish the exact unknown prior DIO
configuration from any earlier wiring state. Both new status connections
now have observed transitions under controlled radio activity.

Retained on pi134: DIO8 IRQ function with pull-up and mask 0x006C0000
(RX_DONE/TX_DONE/TIMEOUT/CRC_ERROR); LoRa 915 MHz, SF7/BW125/CR4-5,
LDRO off, preamble 8, explicit 16-byte payload, CRC on, standard IQ, sync 0x12;
system calibration blocks 111 and front-end calibration at 915 MHz/path0.
Prior write-only configuration was unknown and was not restored. RX counters
and the host IRQ semaphore may have changed. No TX, reset, PRAM, NVM,
firmware, GPIO, driver, package or source changes occurred; no activity remains.

Lead independently checked the saved SPI/pin output against the application
report. Task `dslcap-pi134-rxirq-20261007` revision 4 was consumed and
acknowledged. Evidence: [PI134_RXIRQ_VALIDATION.json](PI134_RXIRQ_VALIDATION.json).

## pi134 original wiring restored / IRQ repeat — 2026-10-07

The user restored CH12=BUSY and CH13=DIO8. ORIGINALIRQ1 repeated one
10 ms RX timeout using the retained RXIRQ1 calibration and IRQ configuration,
without reconfiguration. Application preflight and the immediate execution
gate showed IRQ 0/errors 0/STBY_RC; no intervening application activity was known.
The write-only settings cannot be read back, but the subsequent physical
interrupt confirms a working source with the restored wiring.

At 25 MSa/s, mask 0x3f3f, CH13 rose 10.28084 ms after the SetRx frame began
and fell 22.12 us after ClearIrq TIMEOUT began. IRQ high duration was 3.67082528 s,
including the application inspection/cleanup round trip. CH12 logged 12 BUSY
pulses matching 12 SPI frames; all timestamps were ordered. The 90-second
pipeline exited 0 with 2250000000 samples/4500000026 bytes, ring high-water
1164032 and no overflow. pi133 logged no activity.

Application result: TIMEOUT only, errors 0, followed by verified cleanup to
IRQ 0/errors 0/STBY_RC. No recalibration, configuration, TX, reset, GPIO, service,
source or dependency changes occurred. RXIRQ1's documented LoRa/DIO8 settings
remain configured. No test remains active. The original CH12 BUSY / CH13 IRQ
wiring now passes; the earlier failure is consistent with unknown IRQ routing,
though that earlier write-only configuration cannot be reconstructed.

Lead checked physical edges and decoded commands against the application report.
Task `dslcap-pi134-originalirq-20261007` revision 4 was consumed and acknowledged.
Evidence: [PI134_ORIGINALIRQ_VALIDATION.json](PI134_ORIGINALIRQ_VALIDATION.json).

## 10 MHz kernel ping-pong traffic — 2026-10-08

**PASS.** At 25 MSa/s (2.5 samples per 10 MHz SCLK period) the DSLogic decoded
two complete `lr2021_pcycle` ping-pong runs on both buses byte for byte,
without deglitch. pi133 (initiator) and pi134 (responder) ran the production
kernel driver with the core pinned at 500 MHz, so SCLK was 10 MHz. The
lr2021_kernel_packet_cycle session ran all traffic; this session ran only the
analyzer, after the user cleared both.

**SCLK phase widths (RUN A).** Six nSS-triggered buffer captures of 41.9 ms
each (0.252 s in total) were taken during about 92 s of continuous traffic:
three of pi133 at 400 MSa/s on CH0–3 (2.5 ns grid), three of pi134 at
100 MSa/s on CH8–11 (10 ns grid). On pi133 every sampled SCLK high was
50.0–52.5 ns and every within-byte low 47.5–50.0 ns, duty 50.9%. That is
7.5 ns above one 40 ns sample at 25 MSa/s on the sample grid; allowing one grid
interval of edge uncertainty, the phases exceed about 45 ns. pi134's means
agree (duty 50.9–51.9%, lows averaging 48–49 ns), but at 10 ns resolution it
recorded thousands of lows at 40 ns. That is consistent with ~49 ns phases
quantized to 40 or 50 ns, but it does not bound pi134 as tightly as pi133. The
empirical evidence for both buses at 25 MSa/s is the byte-exact decode below.
Frames include the 4104-clock (513-byte) FIFO transfers.

**Byte-exact decode (RUN B).** A 90 s `sigrok_hla.py --dslogic --raw-out`
capture of 12 channels (mask `0x3f3f`, uint16) held exactly 2,250,000,000
samples (4.5 GB to local NVMe), dslcap exit 0, ring high-water 1.98 MB. It was
replayed with `--dslogic -i` in 13 s, and in 20 s with
`-T deglitch:channels=0,8:clock_period=2.5:frame_pulses=8`; the deglitch made
zero corrections and its output is byte-identical. `check_pcycle.py --expect
2000` passes the full-run contract:
- All 2000 request frames pi133 wrote (seq 0..1999, each once) match prbs9 from
  byte 4, and are identical to pi134's MISO reads.
- All 2000 replies pi133 read (resp_seq 0..1999, each once) match prbs9 from
  byte 8, and each is identical to a frame pi134 wrote.
- The responder pre-stages two replies, so the first two echoes are
  `0xFFFFFFFF` and the echo is resp_seq − 2 after that.
- pi134 wrote 2004 replies, resp_seq 0,1,0,1,2..2001. Counted by occurrence,
  four are unread: 0 and 1 from the negative control's staging, and 2000 and
  2001 staged after the last exchange.
- There were no CMD_FAIL, dict-error or decode-error lines, and no short
  transfers among 62,185 transfers.
- Three zero-filled 1022-byte TX FIFO prefills during setup decoded as `CMD_OK`.

The kernel's own counts agree: req_rx 2000, reply_rx 2000, 0 missed/CRC/mismatch.

**RUN C**, a second 2000-exchange run with `INIT_RX=auto WITNESS=2`, was
captured the same way after the user moved CH4 to pi134's DIO11 witness (see
the current-wiring note under Open gates). It also passes the full-run contract byte for byte,
with the same staging pattern, and CH4 shows 4004 witness edges.

Scope: two runs, and 0.252 s of high-rate duty sampling, on this bench, wiring
and 1.6 V threshold. Decode was by replay through the same decoder; live decode
of this traffic was not run. Raw captures and their dslcap logs remain on
local disk (`/mnt/foo/dslcap-captures`), not committed; the JSON records their
hashes and the kernel stat file hashes.
Tools: `dslcap/tools/` (duty analysis, prbs9 checker, synthetic generator).
Evidence: [PCYCLE_10MHZ_VALIDATION.json](PCYCLE_10MHZ_VALIDATION.json).
