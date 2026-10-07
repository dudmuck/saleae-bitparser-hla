# Independent trigger implementation reviews

## Cycle 1 - 2026-10-07

Reviewing: **B-simple-trigger**, Wave 1, group cycle **1 of 3**.
Synthesizer: **review-1**. Analysts: none requested or missing.
Scope: the 19 frozen implementation/documentation/test files enumerated in
`/tmp/dslcap-phase-b-handoff.md` and `/tmp/dslcap-phase-b-source.sha256`.
Read their current source and tracked diff against `1c485fa`, including
new `dslcap/tests/test_trigger.py` and `tests/test_dslcap_trigger.py`.
Requirements: TRIGGER_PLAN.md, TRIGGER_TASKS.md and the complete review
assignment `/tmp/dslcap-phase-b-review.md`. Read-only driver reference:
`/home/wroberts/DSView-1.3.2/`. Historical REVIEW.md groups are separate.
Only this file was written; no implementation, Git, USB, Pi or application
changes/actions were performed by review-1.

### Critical / Severe / High

- None.

### Medium / Low / Warning

- **Low — empirically reproduced** [`options.c:201`]: Repeating
  `--on-timeout upload --on-timeout fail` retains upload: the `fail` arm
  validates the spelling but never clears `timeout_upload`. Other scalar
  options use the last value, while this one silently retains an earlier
  action. With the real frontend/fake driver, the command below returned
  0, `META trigger: none`, and 1024 payload bytes. A sole `fail` requests
  timeout exit 16. Either reject duplicate timeout actions or explicitly
  assign zero for `fail`, and cover both orderings. The documented
  single-occurrence CLI works; Python argparse forwards only one action.

  ```sh
  DSLCAP_TEST_SCENARIO=trigger-force DSLCAP_TEST_ACTUAL=1024 \
    /tmp/dslcap-b-review/dslcap_fake \
    --fw-dir /tmp/dslcap-b-review/test-firmware \
    --mode buffer --channels 0-7 --samplerate 100M --samples 2048 \
    --trigger 3:f --trigger-timeout 0.1s \
    --on-timeout upload --on-timeout fail
  ```

- **Warning — static-verifiable verifier gap** [`tests/test_trigger.py:105`]:
  Per-phase signal tests sleep 20/30ms, then assert signal exit and the
  samplerate line. They do not observe that WAITING, POST_TRIGGER, FORCED
  or UPLOAD was actually reached before delivering the signal. On a slow
  host a case can pass while cancelling an earlier phase. Add explicit
  phase/readiness synchronization or an independently checked phase
  observation before each signal. These tests passed in this review;
  their cancellation evidence is stronger than their exact-phase claim.
- **Warning — requires live validation**: All B.1–B.6 implementation-side
  hardware gates remain open. DSView golden reference acquisition is
  complete, but no reviewer run establishes dslcap register equivalence,
  actual FPGA trigger position/count behavior, timeout/upload race outcomes,
  physical cancellation/reopen or triggered high-rate HLA captures.
  A2 continuous-clock ppm acceptance also remains open.

### Suggestion

- None.

### Spec Alignment

Simple finite buffer triggers, conditions, captured-lane requirements,
position, timeout policy and buffer drain options match the authorized
scope. Modifiers without a trigger, stream/continuous/pattern triggering,
duplicates within a condition list, invalid positions and unsupported
serial/t0 options are rejected. No Phase C or trigger-relative timestamp
behavior was added.

Configuration resets trigger state after capture readbacks, sets position
and SIMPLE mode, sets each probe, explicitly selects upload policy when
requested and enables last. Compared the pinned driver's private
`SR_BUF_UPLOAD=1`, trigger API and `dsl_fpga_arm` position calculation.
The effective position helper uses the aligned arm count, uint32 cast,
64-sample minimum, aligned channel-depth 90% cap, and 64-sample floor.
Tests check the reference values 100032 and 200064 and the maximum-depth
arithmetic. `TRIGGER_GOLDEN.json` retains actual 372-byte DSView images;
this review does not equate math checks with full image comparison.

The callback copies the transient `ds_trigger_pos` before return, validates
its check ID/status/count/position and packet order, sets the converter's
emitted limit and queues META line 2 before publishing the header under
`packet_mutex`. LOGIC cannot precede the header in triggered captures.
Success requires valid header, exact emitted count and data-END; a header
alone does not finish upload. Trigger metadata depends on returned status,
including a real hit in a forced capture. Forced counts may be shorter,
aligned to 1024 before trimming to N.

### Cross-Task Consistency

Python derives trigger channels through the same name resolver and adds
them to the producer/width union. A trigger-only high bit selects uint16;
unreferenced names do not. Lane/rate checks agree with the pinned supported
profile and representable rates. Durations are normalized before forwarding.
The requested trigger mode fixes the number of META lines at one or two,
and binary bytes beginning `META trigger:` are preserved as payload.
Missing line 2 at EOF does not replace producer timeout 16 or a signal
status. A numeric marker enters the existing heap without advancing the
data watermark; sample and pin timestamps remain capture-relative.
Old stream/wide/sigrok tests remain green. Existing Python repeated-
KeyboardInterrupt cleanup and C signal-publication portability limitations
are historical, remain unresolved, and are not claimed fixed by this wave.

### Security And Operations

Compared DSView's actual `receive_header`, `receive_transfer`,
`SR_CONF_WAIT_UPLOAD`, collection-end and stop paths. The reference frees
the header transfer directly after the callback and suppresses upload once
the abort flag is set. Frontend commitment is therefore irreversible and
mutually exclusive with packet processing; late packets cannot restore
success after timeout commitment. The main thread releases packet_mutex
before any driver call/join. WAIT_UPLOAD intent is published before the
call so callbacks can run during it, then TRUE/FALSE and actual count are
reconciled. FALSE plus a short capture is rejected rather than legitimized
by the transient intent flag.

The 340ms grace is the accepted full nominal 17×20ms buffer interval.
Source `dsl_get_timeout` returns 20ms in buffer mode. Cached status is
polled regardless of verbosity, failures/change observations are reported,
and the code promises no hardware-event-time or cache-age guarantee.
Known terminal signal/device/data facts receive the specified precedence;
signal also wins if raised during library cleanup.

Buffer capacity is checked sample bytes plus exact/max metadata allowance,
including the 512MiB high-bit-only case. Allocation precedes arming.
Writer I/O occurs outside the ring mutex; successful writes reset the
buffer stall and hard watchdog deadlines. Buffer finish uses 50ms timed
condition waits and observes signal/atomic device cancellation; callbacks
do not touch destroyed ring synchronization. Stream retains its 2-second
drain. An emergency watchdog exits the process when inherited driver or
output operations cannot complete; physical reopen remains a live gate.

### Verification And Test Adequacy

Independent commands, all exit 0:

```sh
cmake -S dslcap -B /tmp/dslcap-b-review -DBUILD_TESTING=ON
cmake --build /tmp/dslcap-b-review -j4
ctest --test-dir /tmp/dslcap-b-review --output-on-failure
python3 -m unittest discover -s tests -p 'test_dslcap*.py' -v
sha256sum -c /tmp/dslcap-phase-b-source.sha256
```

CTest: **5/5 suites passed in 23.58s**. Python: **36 tests passed in
17.433s**. All **19 frozen source hashes matched**. Logs:
`/tmp/dslcap-b-review-{config,build,ctest,python}.log`.

Reviewed test assertions and fake-driver timing/events, not only totals.
Tests execute real frontend/control/ring/converter code with actual child
processes and pipes. They cover ordered held-main bursts, immediate header
overwrite/free, header faults/order, grace hit/header observations, committed
late packets, status-read errors, TRUE/FALSE force outcomes, aligned forced
counts and requested trims, soft phase timeouts, detach/data/signal outcomes,
cleanup signal precedence, progress/stall/cancelled drains and byte oracles.
Python tests cover trigger-only channel width, name/argument rejection,
every two-line header split, legal META-like binary prefixes, marker order,
future-marker watermark behavior, timeout/signal exit precedence and prior
backend regressions. The exact-phase signal limitation is recorded above.

Verified generated compiler flags: production `dslcap_core` has no test
time-scale define; only `dslcap_fake` has `DSLCAP_TEST_TIME_SCALE=0.02`
and `-UNDEBUG`. Production timing cannot inherit the fake target's scale.
The slow-reader test uses real unscaled one-second stalls and runs longer
than the former two-second fixed limit. Capacity overflow and 512MiB
arithmetic are tested without requiring a large allocation in the suite.
Python 3 is used for offline tests; production linkage stays free of
Qt/Python and no package installation was performed.

The extra repeated-option diagnostic exited 0 and exposed the Low behavior
above; it was not a failing automated assertion. All executed required
automated tests passed. Worker/lead results are consistent with the fresh
independent runs. No hardware success is inferred from fake-driver tests.

### Open Live Validation

- B.1: dslcap side of the actual 372-byte golden comparisons at
  100M/200M/400M; DSView reference acquisition is complete.
- B.2: repeated real trigger conditions/rates (20 captures per case/rate).
- B.3: positions, returned real_pos and early-hit behavior.
- B.4: fail/upload/cancel count, status, output and immediate reopen.
- B.5: real cached-status/deadline races and nominal-grace behavior.
- B.6: triggered single 400M and dual 100M Python/HLA validation.
- A2: continuous-clock ppm acceptance, not waived by Phase B.

### Verdict: PASS

No Critical, Severe or High finding remains, and all executed automated
tests passed. Counts: 0 Critical/Severe/High, 0 Medium, 1 Low, 2 Warnings,
0 Suggestions. This accepts the reviewed offline implementation with the
explicit Low issue and verification gaps retained; it closes none of the
listed hardware gates. Cycle 1 is consumed; no new cycle is implied.
