# Independent implementation reviews

## Cycle 1 - 2026-10-07

Reviewing: Wave 0 / Phase 0 bring-up, task group **G0-bringup**, group cycle **1 of 3**.
Reviewer/synthesizer: **review-1**. Analysts: none requested; no missing analyst results.
Scope: `CMakeLists.txt`, `main.c`, `tests/fake_driver.c`, `tests/bringup.cmake`,
`tests/reload_fpga.c`, `README.md`, `.gitignore`, all under `dslcap/`.
Requirements: current `PLAN.md` and `TASKS.md`. Reference source:
`/home/wroberts/DSView-1.3.2/`, read-only. Existing review file was absent.
This record is beside the plan because repository `.codex/` is read-only.

### Critical / Severe / High

- None.

### Medium / Low / Warning

- **Warning — requires live validation** [`README.md:74`]: Production startup
  with an initially unconfigured FPGA has not been exercised. The worker's
  explicit reload proves the actual upload routine transferred 530620 bytes,
  followed by security success, HDL 0x0e and successful production reopen;
  it does not prove the entire cold automatic-upload branch. Preserve this
  distinction and validate a cold start when physical/device state permits.
- **Warning — static-verifiable** [`README.md:99`]: The pinned driver's FPGA
  polling loops can run indefinitely when USB reads succeed but expected
  status bits never arrive (`hardware/DSL/dsl.c:1366` onward). The optional
  reload instructions correctly use an external timeout; production activation
  can enter the same routine. Keep hardware bring-up commands externally
  bounded, and carry cancellation/watchdog handling into the capture work.
  No abnormal-hardware hang was induced in this review.

### Suggestion

- `TASKS.md` records the durable G0 identifier and consumed cycle. There is
  no separate decisions file; when adding one, retain this same identifier
  and counter rather than starting a new review budget.

### Spec Alignment

Phase 0 is implemented and documented as bring-up only. Missing capture
options, conversion, ring buffering, signal shutdown, and Python integration
are subsequent phases and are not failures of this slice. The build uses
the reference tree directly and matches its library/common source selection;
there is no Qt/Python dependency or copied driver source.

The frontend validates the firmware path before initialization, including
the upstream 500-byte resource buffer constraint. It selects exactly one
USB VID/PID 2a0e:0034 after filtering the pinned model name, guards against
activation's demo fallback through active handle/type and last-error checks,
requires pass and absence of failure logs independently for both activations,
and explicitly reads HDL 0x0e after release/reopen. This matches the pinned
driver's silent-security-failure contract and conditional HDL-check behavior.
Setting firmware resources before initialization is justified because init
itself scans devices.

### Cross-Task Consistency

Diagnostics use stderr; verified scan stdout is deferred until cleanup.
No-option bring-up emits no capture bytes. Actual xlog callback lengths are
honored without assuming a terminating NUL. Atomic flags cover log receiver
access. Test-only reload is excluded from the production target and from
CTest registration. Worker handoff's heading calls this G1-capture Phase 0;
the authoritative TASKS.md and lead assignment identify this review as G0.
No implementation edits, Git operations, or USB access were performed by
review-1. Only this review file was written.

### Security And Operations

Checked `lib_main.c` initialization/exit, device-list allocation, activation
fallback, and release contracts; checked `log.c` shared writer ownership,
`xlog.c` callback construction, and `dsl.c` HDL/security/upload routines.
Failed initialization occurs before the library mutex, scan and hotplug
thread setup, making the immediate-process-exit approach safer than calling
the library exit path. Successful init always reaches exit before shared
logger destruction. The driver's release/exit APIs mask some underlying
close errors, so fake cleanup failures prove frontend handling, not every
physical cleanup outcome. Successful live reopen is separate supporting
evidence. The reload helper uses volatile control registers and USB bulk
transfer, with no NVM-writing operation; security activation reads EEPROM.
No new dependencies or reference-source mutations are introduced.

### Verification And Test Adequacy

Independent offline commands:

```sh
cmake -S dslcap -B /tmp/dslcap-review-1 -DDSVIEW_SRC=/home/wroberts/DSView-1.3.2 -DDSLCAP_BUILD_USB_TESTS=ON
cmake --build /tmp/dslcap-review-1 -j4
ctest --test-dir /tmp/dslcap-review-1 --output-on-failure -V
ctest --test-dir /tmp/dslcap-review-1 -N
ldd /tmp/dslcap-review-1/dslcap
```

Configure/build succeeded; build was also independently rerun with exit 0.
CTest exit 0: all 19 fake-driver scenarios plus three invalid firmware-path
cases passed. Test enumeration exit 0: exactly `bringup_contract`, with no
hardware test registered. ldd exit 0: GLib/libusb/zlib/libm/libc and system
dependencies, no Qt/Python. Configure and build logs are
`/tmp/dslcap-review-1-config.log` and `/tmp/dslcap-review-1-build.log`.

The test runs the real frontend and real reference logger, checks exact
successful stdout and empty failed stdout, independent reopen security,
both failure/pass log rejection, active identity fallback, last-error,
HDL read/version errors, list/init/release/exit failures, ambiguous devices,
and rejection before init for invalid paths. It asserts release before the
second activation and forbids unsafe cleanup after failed initialization.
It provides appropriate Phase 0 frontend regression coverage; it is not
a substitute for physical loading or driver cleanup evidence.

Evaluated worker evidence in `/tmp/dslcap-phase0-handoff.txt`, including
`/tmp/dslcap-phase0-reload.stdout` and `.stderr`: verified-device stdout,
530620-byte upload, initial and reopened security passes, explicit HDL 0x0e,
and hotplug thread completion. Worker also reports invalid-directory exit 3,
held-interface/demo-fallback exit 5, immediate reopen exit 0, and post-reload
production scan exit 0. Lead independently reproduced a fresh build/CTest
and `timeout 30s /tmp/dslcap-lead-review/dslcap --scan -v` with exit 0,
firmware 2.2, both security passes, HDL 0x0e, close/join, and verified stdout.
Lead live evidence is attributed to the lead; review-1 made no USB calls.

### Open Live Validation

Cold automatic startup upload remains unrun as described above. Explicit
volatile upload, warm activation, invalid-path failure, interface contention,
security/HDL and immediate reopen have supporting hardware evidence.
Subsequent sample/capture/stream/Python validation remains outside G0.

### Verdict: PASS

No Critical, Severe, or High findings remain, and all executed build/test
checks passed. Counts: 0 Critical/Severe/High, 0 Medium/Low, 2 Warnings,
1 Suggestion. This verdict accepts Phase 0 bring-up with the recorded
cold-start evidence gap; it does not approve unimplemented later phases.

## G1 Cycle 1 - 2026-10-07

Reviewing: Wave 1, **G1-capture**, group cycle **1 of 3**. G0 remains at
cycle 1 PASS; this is its separately planned capture successor.
Reviewer/synthesizer: **review-1**. Analysts: none requested or missing.
Scope: capture C/build/test changes since bring-up commit `47d9d6d`:
`main.c`, `CMakeLists.txt`, `README.md`, `tests/fake_driver.c`, new
`options`, `convert`, `ring`, `control`, `driver_config`, and `capture`
C/header pairs, `status.h`, `tests/test_core.c`, `tests/capture.cmake`,
and `tests/control.cmake`. Python changes are excluded.
Inputs: current PLAN.md, TASKS.md, VALIDATION.md and
`/tmp/dslcap-capture-handoff.txt`; pinned DSView source read-only.

### Critical / Severe / High

- None.

### Medium / Low / Warning

- **Low — static-verifiable** [`control.c:11`]: `volatile sig_atomic_t`
  protects the signal-handler access pattern but does not provide C11
  inter-thread synchronization. The main/control and watchdog threads both
  read this object, and an unblocked process signal can run its writer on
  any thread. Current target tests pass, and no missed cancellation was
  reproduced; this is a portability/synchronization weakness. Prefer an
  explicitly lock-free atomic signal flag with a build-time guarantee of
  signal-safe atomic access, or a signal-wait/self-pipe design with atomic
  publication to the watchdog. Retain the no-library-work-in-handler rule.
- **Warning — requires live validation** [PLAN.md](PLAN.md): Full Phase 1/2
  evidence remains incomplete. CH0 pulse captures, exact finite lengths,
  internal buffer counter, and shutdown/backpressure evidence are useful,
  but do not establish physically distinct high inputs, the independent
  DSView comparison, actual SPI traffic, or real FPGA overflow detection.
  The lead's 610-second continuous run was still pending at review handoff.
  Preserve these acceptance gates; implementation PASS does not close them.
- **Warning — static-verifiable test gap** [`tests/test_core.c:104`]: The
  byte-for-byte successful ring test writes 200000 bytes into a 262144-byte
  ring, so neither cursor wraps. Saturation tests exercise errors rather
  than validate wraparound output contents. Add a deterministic draining
  consumer test with total data greater than ring capacity, irregular
  pushes and a byte-for-byte oracle. No ring corruption was observed;
  a long run to `/dev/null` alone cannot prove preserved contents.

### Suggestion

- None.

### Spec Alignment

Configuration order matches the prescribed operation/channel/rate/limit/
loop/VTH/enables sequence, with explicit disable and mask/count/rate
readback. Mode selection uses the real 0034 profile, available physical
channels, largest valid-channel count and supported/exactly representable
rates. Compared the divider and 200M/400M packing decisions against
`dsl_fpga_arm`, plus the 0034 profile and channel tables in `dsl.h`.
Rejecting 60M and 3M is justified by actual clock division rather than
trusting the driver's cached samplerate. Buffer limits include 1024-sample
rounding; internal pattern reads and verifies the forced depth/rate.

The converter preserves physical positions, zeroes holes, derives width
from high channel presence, handles partial groups and trims to the exact
finite count. Its independent sample-first bitplane oracle checks both
widths and LSB-first ordering over 3192 split/trim cases. End validation
rejects short finite captures, partial source groups and missing data-END.
META is queued before samples; logging remains on stderr. Configuration
failures occur before META, while subsequent failures are documented as
potentially partial output requiring exit-status checks.

### Cross-Task Consistency

The Python handoff can consume the existing META plus one-byte low-channel
stream without changing the C contract. Standalone high-channel output is
two-byte little-endian as required, while wide Python decoding stays
separate. No Python files were reviewed or changed. Bring-up/security tests
continue to pass. Reload remains opt-in and unregistered with CTest.
Only REVIEW.md was edited; no USB/GPIO access or Git mutation occurred.

### Security And Operations

The USB callback performs conversion and bounded ring copies, never output
I/O or driver stop/join. The main thread stops collection; the fake driver
asserts that stop never runs on its callback thread. The ring releases its
mutex before writing/polling and preserves first error, drains with a
deadline, reports truncation, and restores descriptor flags on normal
completion. Output errors, overflow, device/detach/speed events, malformed
input and signal outcomes have distinct nonzero codes.

Static capture-state lifetime protects late event callbacks from stack
use-after-free, with atomic error/event publication. Reviewed real library
collection-end ordering: data callbacks precede normal END notification.
Inherited library thread-pointer and callback management are not modern
C11 synchronization and remain a pinned dependency limitation. The new
watchdog bounds startup, finite acquisition, stop and cleanup, with a
five-second signal grace and emergency process exit. This addresses the
G0 unbounded polling concern operationally; emergency teardown still
requires a hardware reopen check. Consumers must also drain stderr.

### Verification And Test Adequacy

Independent commands, all exit 0:

```sh
cmake -S dslcap -B /tmp/dslcap-review1-capture -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/dslcap-review1-capture -j4
ctest --test-dir /tmp/dslcap-review1-capture --output-on-failure
```

All four suites passed in 14.15 seconds: capture_core, driver_watchdog,
bringup_contract and capture_contract. Coverage includes 3192 conversion
cases, parser rejection, exact writer bytes, EPIPE, bounded blocked drain,
ring-full error preservation, startup/signal watchdog exits, the previous
22 bring-up cases, and 34 capture/configuration cases. Reviewed assertions
and fakes, including setter order, rate/count readback failures, callback
thread stop prohibition, short/malformed packets and event propagation.
Build logs: `/tmp/dslcap-review1-capture-config.log` and
`/tmp/dslcap-review1-capture-build.log`.

Worker-reported evidence, not independently rerun on hardware: ASan/UBSan
core tests passed; finite 8/12/16/sparse captures have exact lengths;
16777216 internal-pattern uint16 samples matched `i modulo 65536`; buffer
padding/trim passed; unrepresentable rates and excess depth were rejected;
CH0 pulse trains were observed in all required channel configurations;
SIGINT returned 130, closed pipe 12, ring full 9, with immediate reopen 0.
Lead separately rebuilt and ran all four offline suites successfully.
Actual GPIO comparisons only exercise CH0, and software pulse timing is
not a precision-rate reference. The independent reviewer performed no
physical tests while the lead owned the analyzer.

### Open Live Validation

Pending at this review: the lead's 610-second stream result, forced real
FPGA overflow, high physical channel input mapping, independent DSView
capture comparison and known/full SPI comparisons. Fast 200M/400M buffer
modes are implemented but unvalidated. G0 cold automatic FPGA startup
remains unrun. Later lead validation should update VALIDATION.md without
retrospectively treating these gates as having been tested by this reviewer.

### Verdict: PASS

No Critical, Severe, or High finding remains and all executed tests passed.
Counts: 0 Critical/Severe/High, 0 Medium, 1 Low, 2 Warnings, 0 Suggestions.
This accepts the reviewed capture implementation with the explicit test
and live-evidence gaps above; it does not declare all Phase 1/2 acceptance
measurements complete.

## G2 Cycle 1 - 2026-10-07

Reviewing: Wave 1, **G2-integration**, group cycle **1 of 3**.
Reviewer/synthesizer: **review-1**; analysts: none requested or missing.
G0/G1 histories and counters remain unchanged. Scope: `sigrok_hla.py`,
`sigrok_hla_readme.md`, `tests/test_dslcap_backend.py`, and fixtures
`tests/dslcap/{fake_producer.py,DslcapTestHla.py,extension.json}`.
Requirements: PLAN.md Phase 3 and current TASKS.md G2 assignment.
Read the scoped Git diff and all implementation/test files; Git access
was read-only. Only this review file was edited; no USB/GPIO access.

### Critical / Severe / High

- None.

### Medium / Low / Warning

- **Medium — empirically reproduced** [`../sigrok_hla.py:723`]: A repeated
  interrupt during cancellation can leave a child alive and make cleanup
  impossible to retry. If `proc.wait()` raises `KeyboardInterrupt`, the
  `finally` block closes pipes and sets `finished=True` without killing or
  reaping the child. A subsequent `finish(cancel=True)` returns immediately.
  Reproduced with the real TERM-ignoring fixture, injecting KeyboardInterrupt
  at the wait boundary: `child_alive=True, finished=True`, and still alive
  after retry. Reviewer explicitly killed and reaped it afterward. Ensure
  interrupted cleanup kills/reaps before declaring completion, and add a
  repeated-interrupt regression. Normal single-interrupt and timeout paths
  passed; this finding concerns interrupted cleanup itself.
- **Warning — requires live validation**: Offline synthetic dual-SPI frames
  and pin timing establish the shared decoding contract, but the required
  actual dual LR1110/LR2021 HLA comparison with Saleae/Logic 2 remains open.
  A live idle-input or CH0-only smoke cannot replace it. Prior physical
  high-channel/DSView/cold-start gates also remain separately recorded.

### Suggestion

- None.

### Spec Alignment

`--dslogic` uses the shared NumPy engine and explicitly rejects srd and
conflicting input backends. The builder derives a sorted unique physical
capture mask from resolved SPI roles and logged pins, treating `-C` as a
name mapping. Indices above 7 are rejected; standalone two-byte capture
does not silently enter the one-byte Python decoder. Rate/time/sample and
threshold validation occurs before producer launch. Default rate is 25M;
verbosity reaches the producer with `-v`/`-vv`.

META parsing handles fragmented prefixes, bounds the header, requires a
valid positive rate for dslcap, and delays decoder/pin-logger construction
until the effective rate is known. Legacy raw sigrok without META retains
the requested rate. The same effective META rate drives both SPI and pin
timestamps. HOLD, heap merge, HLA calls, and fast_spi interfaces are retained.

### Cross-Task Consistency

The Python command agrees with the reviewed C capture CLI and low-channel
output contract. Prefix diagnostics use stderr; decoded results remain on
stdout. Producer errors propagate before successful final-frame synthesis,
and already emitted data is documented as potentially partial. Existing
sigrok binary and srd command/annotation paths have focused regression
coverage. Saleae acquisition implementation is unchanged; an independent
mocked CLI dispatch check confirmed a valid Saleae time/rate/port invocation
still reaches only `run_saleae_backend` with its arguments preserved.

### Security And Operations

The producer uses argument arrays, not a shell. Independent stdout/stderr
reader threads drain pipes concurrently, with bounded stdout queue and
nonblocking/select reads that can observe cancellation. Stderr is forwarded
live with a producer prefix; sustained downstream stderr blocking is still
the consumer's responsibility. Reader errors become capture failures.
Normal cancellation sends TERM, waits three seconds, then kills/reaps;
normal EOF with a non-exiting child is bounded at 15 seconds. Queue-full,
decoder failure, single interrupt and stubborn-child cleanup are tested.
The repeated-interrupt weakness above qualifies the claim of cleanup on
every path. No dependency, firmware, GPIO, or device-source changes occur.

### Verification And Test Adequacy

Independent command:

```sh
python3 -m unittest discover -s tests -p 'test_dslcap_backend.py' -v
```

Exit 0: all 20 tests passed in 14.123 seconds. Tests use actual subprocess
pipes and check >190KB of forwarded stderr, fragmented META, precise
synthetic SPI bytes and pin timestamps, chronological dual-port output,
producer nonzero exit, malformed/absent META, reader faults, saturated
queue cancellation, TERM-ignoring child kill/reap, and decoder interruption.
The CLI test includes an executable path containing a space. Raw sigrok,
META sigrok and srd annotation behavior are covered.

Additional independent offline checks, both command exits 0:

- Mocked `main()` Saleae CLI dispatch with `--spi 0,1,2,3 --samplerate 4M
  --time 5s`: exactly one Saleae call, no sigrok call, arguments retained.
- Interrupted-finish diagnostic using the real `stubborn` subprocess and
  `mock.patch.object(proc, 'wait', side_effect=KeyboardInterrupt)` during
  cancellation: confirmed the Medium finding and explicitly reaped the
  fixture with kill/wait after the observation. No fixture remains running.

### Open Live Validation

The lead's 30-second verbose 25M x 8 dual-port hardware smoke and reopen
were pending at assignment. The finalized worker handoff
`/tmp/dslcap-python-handoff.txt` was read before completion: it reports the
same 20-test success, successful py_compile/help/diff checks, no hardware
access and no new dependency. It explicitly leaves sustained real pipeline
performance and real dual-SPI comparison open; its unconditional cleanup
claim is qualified by the reproduced repeated-interrupt finding above.
Real wired SPI/HLA/reference comparisons must be recorded separately from
synthetic verification.

### Verdict: PASS

No Critical, Severe, or High finding remains and all executed test commands
passed. Counts: 0 Critical/Severe/High, 1 Medium, 0 Low, 1 Warning,
0 Suggestions. This verdict accepts the reviewed integration with the
repeated-interrupt weakness and physical validation gaps retained explicitly.

## G3 Cycle 1 - 2026-10-07

Reviewing: Wave 1, **G3-wide-dslogic**, group cycle **1 of 3**.
Reviewer/synthesizer: **review-1**; analysts: none requested or missing.
This is the operator-authorized wide-channel feature, not a new cycle for
G2's existing findings. G0/G1/G2 review history is preserved.
Scope: `sigrok_hla.py`, `sigrok_hla_readme.md`,
`tests/test_dslcap_backend.py`, `tests/test_dslcap_wide.py`, and
`tests/dslcap/wide_producer.py`. Requirements: updated PLAN.md dual-radio
extension and TASKS.md G3 contract. Read the scoped diff, implementation,
tests, frozen fingerprint manifest and finalized worker handoff at
`/tmp/dslcap-wide-handoff.txt`.

### Critical / Severe / High

- None.

### Medium / Low / Warning

- **Warning — requires live validation**: The actual RAW1 replay supports
  both low and high SPI ports, but a live wide Python producer/decoder run
  was pending at assignment, and a same-traffic Saleae/Logic 2 comparison
  remains unavailable. The synthetic and replay checks do not prove all
  8MHz/10MHz timing margins at 25M sampling. Preserve those runtime/reference
  gates and distinguish observed GetVersion traffic from arbitrary traffic.
  The earlier G2 repeated-interrupt Medium remains historical and unresolved;
  this feature neither changes that lifecycle code nor claims to fix it.

### Suggestion

- None.

### Spec Alignment

DSLogic-only resolution accepts physical channels 0..15; the non-DSLogic
raw path still limits indices to 0..7. The shared `_used_channel_bits`
function supplies both the producer channel list and sample-width choice,
so unused high `-C` labels cannot widen the stream. CH0/1/2/3 plus
CH8/9/10/11 yields exactly eight selected physical channels and two-byte
samples, retaining the requested 25M rate. No generic width override or
automatic rate reduction is introduced.

META parsing precedes byte alignment. The decoder carries at most one
trailing payload byte, combines it with the next read, and feeds complete
little-endian `<u2` samples to the existing decoder and pin logger.
Successful EOF with a leftover byte fails explicitly; producer failure is
checked first and retains precedence over incidental truncation. Both
decoders initialize from the effective META rate. Existing HOLD, heap
ordering, HLA and hex processing are unchanged.

### Cross-Task Consistency

The width contract matches the C producer's physical mask rule. Tests keep
the low-channel DSLogic path and legacy sigrok raw/srd behavior, updating
only obsolete high-channel rejection limits from 8/9 to 16. An independent
mocked Saleae CLI check with CH8/9/10/11 retained those ports and dispatched
only the Saleae backend; DSLogic width logic did not affect it.
`git diff --exit-code HEAD -- fast_spi.py` returned 0, confirming no
fast_spi edits. Only REVIEW.md was written; no USB/GPIO/application actions,
implementation edits, or Git mutations were performed by review-1.

### Security And Operations

No new process launch, dependency, device operation, or parser trust boundary
is introduced. Negative and out-of-range physical inputs are rejected by
the command-building path. Stderr, cancellation and producer-error handling
reuse G2 infrastructure and its known limitations. Wide truncation cannot
be silently interpreted as an 8-bit final sample, and normal EOF validation
does not supersede a producer's nonzero exit.

### Verification And Test Adequacy

Independent focused command, exit 0:

```sh
python3 -m unittest discover -s tests -p 'test_dslcap*.py' -v
```

All 29 tests passed in **17.341 seconds**; output:
`/tmp/dslcap-g3-review-tests.log`. Existing 20 tests retain low-channel,
sigrok, stderr-pressure and cancellation coverage. Nine wide tests cover
the exact producer union, unused high mappings, names, bounds, distinct
bytes on both buses, high pins 12/15, effective META timing/order, dtype,
deterministic odd/random fragments, truncated EOF, upstream failure
precedence and actual subprocess cleanup.

The wide fixture deliberately assigns different MOSI/MISO bytes to the two
ports. Exact expected interleaving includes high-port result times 21/37us,
low-port times 17/33us, and pin transitions 10/14/20/24us at META 2M versus
requested 1M. A dtype spy asserts `<u2`, 84 samples, initial word 0x0808.
This verifies physical high-bit use and timing rather than only output size.
Controlled chunks establish byte boundaries that OS pipe coalescing could
otherwise hide; real subprocess variants additionally check reader/reap paths.

Additional independent adversarial command, exit 0: used the wide fixture
through the real decode function at **all 194 possible two-chunk split
positions** across its entire META+payload sequence (including empty first
or last chunks), asserting the same exact expected bytes/times/order for
every split. The same command verified mocked Saleae high-channel dispatch.

`sha256sum -c /tmp/dslcap-g3-review.sha256` returned 0 for all five scoped
files; no fingerprint changes were observed. Worker reports 29 tests in
17.626s and lead reports 29 in 17.362s. Worker also reports successful
py_compile, help and diff checks, no added dependencies and no live actions.

### Open Live Validation

Read lead-produced `/tmp/dslcap-radio-run1-hla-validation.json` and the
replayed HLA output: 20 requests and 20 responses per port, including
GetVersion v1.24, matching lead's report of RAW1 bus reconstruction.
This is lead hardware/replay evidence, not reviewer hardware execution.
Live wide pipeline was pending at assignment, and same-traffic Saleae/Logic 2
comparison remains open. The standalone independent DSView reference checks
for all four required channel/rate configurations and cold automatic FPGA
startup were already completed before G3, as recorded in VALIDATION.md
and commits `297c9db` and `4e02d4b`; the cold run automatically uploaded
530620 bytes and passed security/HDL/reopen. Those completed standalone
checks are distinct from the pending live wide Python pipeline. Future high
IRQ/BUSY wiring is deferred. Lead should record subsequent physical outcomes
in VALIDATION.md without recasting this offline review as having performed
them. This gate correction is editorial; the G3 cycle and verdict are unchanged.

### Verdict: PASS

No new Critical, Severe, High, Medium or Low finding in the G3 feature;
one Warning records runtime/reference validation limits. All independently
executed tests and adversarial assertions passed. Counts for this cycle:
0 Critical/Severe/High, 0 Medium/Low, 1 Warning, 0 Suggestions. G2's
unresolved repeated-interrupt Medium is carried forward without resetting
its review history or claiming a fix.
