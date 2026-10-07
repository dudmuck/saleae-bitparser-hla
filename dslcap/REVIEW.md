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
