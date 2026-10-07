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

- Dual-SPI LR1110/LR2021 HLA comparison with Saleae/Logic 2. The known
  single-port Pi burst and independent sigrok decode now pass (below).
- Cold automatic FPGA upload, as recorded in the G0 review.

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
