# Serial-trigger independent review

## Cycle 1 - 2026-10-07

Reviewing: **C-serial-trigger**, wave 1, group cycle **1/3**, baseline b2c5369.
Sole synthesizer: review-1. Analysts: none requested or missing. Scope is the
ten frozen files in `/tmp/dslcap-phase-c-handoff.md`: CMakeLists.txt, options.h,
options.c, driver_config.c, tests/fake_driver.c, tests/test_serial.py and README.md
under dslcap, plus sigrok_hla.py, sigrok_hla_readme.md and
tests/test_dslcap_serial.py. Only this review file was edited.

### Critical / Severe / High

None.

### Medium / Low / Warning

- **Warning — requires live validation:** `dslcap/README.md:220` correctly
  labels MSB-first order as assumed. The fake driver's stage assertions and
  literal asymmetric-value strings establish software configuration, not FPGA
  shift direction, register equivalence with a real DSView session, or opcode
  behavior. Keep the planned real DSView serial golden, four isolated bit/byte
  order variants, actual opcode, and later-payload match checks open. Any
  mapping change prompted by hardware evidence needs review. This is a known
  acceptance boundary for this offline implementation wave, not a claim that
  current hardware behavior is wrong.

### Suggestion

None.

### Spec Alignment

All six fields are required exactly once. Checked decimal channel/width parsing,
hex/binary prefixes, overflow, width 1..16, value fitting width, role sharing,
edge-only clock, mutually exclusive simple/serial specifications, finite buffer
mode, selected physical channels and rate-dependent lane limits. C validates
roles before initialization and main's existing mode preflight rejects impossible
lanes. Python canonicalizes named roles and incorporates all four roles in the
same capture-mask and sample-width union used by the existing decoder.

The fixed C parser output feeds bounded 32-byte stage strings. Each contains
exactly sixteen space-separated probes plus NUL; physical indices and bit counts
cannot escape those arrays after parsing. No user string is passed into DSView's
unchecked stride-two stage setter. Stage 0 contains start/stop, stage 1 clock/X,
stage 2 the physical data-channel zero marker/X, and stage 3 the low-N comparison/X.
Reset, position, serial mode, global stage, values, logic, inversion and counts
precede enable. Configuration failures return without arming.

Global stage 0 and upper 16-N X are explicitly accepted decisions in
TRIGGER_TASKS.md, not undisclosed deviations. Independently inspected DSView
`pv/dock/triggerdock.cpp:306` through the serial count setup and
`libsigrok4DSL/trigger.c:54` through the setter implementations: stage roles,
reverse probe indexing, reset defaults, logic=1, inversion=0 and counts 1/N-1
agree. `hardware/DSL/dsl.c:1181` special-cases serial data stage 3 to avoid
half/quarter replication. Explicit default setters on all four roles preserve
the reset-default values even though the default UI stage selector is one.

Serial selects the existing shared trigger lifecycle and exactly two META lines;
no separate capture/control/ring implementation was added. The Python marker
remains chronological with capture-start timestamps. Optional t0 is excluded.
Both READMEs disclose the last-N-bit sliding comparison, possible later payload
matches, upper-X short-width reference, and unresolved hardware ordering.

### Cross-Task Consistency

C and Python agree on canonical physical roles, condition vocabulary, widths,
prefixed values and mutual exclusion. High trigger-only roles widen uint16 input
even when decoded SPI uses low channels. Previous resolver result arities remain
available; the command builder explicitly requests the extra serial result.
Simple, untriggered, raw sigrok, wide-input and shared subprocess regressions
passed. Source fingerprints matched all ten files both before and after review.

The lead supplied baseline/diff verification; under the assignment's no-Git
restriction this reviewer inspected current full affected code, new tests,
documentation, and the retained worker edit scripts, without independently
running a Git baseline comparison. The preserved Phase B duplicate
`--on-timeout` behavior and historical repeated-interrupt concern were not
claimed fixed or reopened in this group.

### Security And Operations

No USB, SSH, Pi, application, GPIO, external-message or Git actions were used.
Production and fake binaries were built in a separate temporary directory. Only
the fake binary ran through CTest; optional hardware tests remain excluded.
Commands use argument arrays, generated stage data and existing error paths.
No new runtime dependency was introduced; serial tests reuse the existing
Python test dependency. Normal binary stdout and diagnostic stderr separation
is exercised by exact byte oracles. Cancellation tests properly accept a valid
already-written stream prefix plus nonzero status, since queued bytes can be
discarded during the unchanged cancellation path.

### Verification And Test Adequacy

Independent commands from the repository root:

```sh
cmake -S dslcap -B /tmp/dslcap-c-review -DBUILD_TESTING=ON
cmake --build /tmp/dslcap-c-review -j4
ctest --test-dir /tmp/dslcap-c-review --output-on-failure
python3 -m unittest discover -s tests -p 'test_dslcap*.py' -v
sha256sum -c /tmp/dslcap-phase-c-source.sha256
cc -std=c11 -Wall -Wextra -Werror -shared -fPIC dslcap/options.c -lm -o /tmp/dslcap-c-review-options.so
```

All completed with exit 0. Fresh production/fake build succeeded; **6/6 CTest
groups passed in 26.50 seconds**, including the eight serial test methods.
**42 Python tests passed in 17.472 seconds**. Hash checks matched **10/10**.
Build/configure logs are `/tmp/dslcap-c-review-{config,build}.log`; CTest's
log is `/tmp/dslcap-c-review/Testing/Temporary/LastTest.log`.

An additional inline Python/ctypes verifier loaded that separately compiled
parser and compared it with Python parsing: every width 1..16, values zero,
one, maximum and first out-of-range, both hex and binary, deterministic shuffled
field order, high/shared role channels, plus empty, overlong and overflowing
decimal inputs. **132 checks passed**. The initial verifier passed builtin int
as a two-argument channel callback, causing a reviewer harness TypeError before
comparison; replacing it with `lambda value, label: int(value)` corrected that
harness error and the entire check was rerun successfully. This was not a
production or test-suite failure.

Tests assert independent literal strings for 0x1c35 and its three distinct
reversal/swap variants, low-N upper-X placement, channel membership and lane
rejection before fake initialization, API order/defaults, configuration failures,
physical high-bit byte output, timeout/grace/forced/error/cancel paths, fragmented
two-line metadata, uint16 decoding, marker chronology and producer failure
precedence. These are meaningful offline checks; fake traffic does not simulate
the FPGA serial matcher and cannot replace the following live gates.

### Open Live Validation

- Real DSView serial register/EP2 golden with identical 0x1c35/16 setup.
- Hardware intended/reversed/byte-swapped/both traffic, isolated per run.
- Actual LR opcode transaction at the returned trigger, plus later payload
  matching behavior. Short-width comparisons require an upper-X DSView bit-editor
  reference, not its zero-padding hex helper.

No live serial correctness or Phase C hardware completion is claimed. Existing
best-effort cached-status timeout and partial-output limitations remain.

### Verdict: PASS

**0 Critical/Severe/High, 0 Medium, 0 Low, 1 Warning, 0 Suggestions.**
All valid independent checks pass. The frozen offline implementation meets the
assigned contracts; the explicitly separate hardware gates remain open.
