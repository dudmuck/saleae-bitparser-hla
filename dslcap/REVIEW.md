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
