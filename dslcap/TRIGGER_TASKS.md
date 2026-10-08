# Trigger implementation tasks

## C-serial-trigger — wave 1

### Live comparison follow-up

After the blocker was committed as01b9694, user authorized testing real DSView
against the same live pattern. Completed tasks: `dslcap-c-dsview-live-20261008`
(DSView owner, exclusiveUSB) and `dslcap-c-dsview-radio-20261008` (radio owner,
baseline then namedGO only). First simple NSS control then serial1c35/16, same
100M CH0-7 settings and padded FIFO frame. No production edits/dependencies.
Response comparison completed: DSView simple control naturally hit (actual
FPGA status1, real_pos100051), serial waited over60s after emission and returned
status0 only after Stop/forced upload. Actual EP2 matched references. Both
tasks completed revision5, consumed/acknowledged; DSView settings restored and
USB released, pi133 baseline restored, pi134 untouched. No production change.
DSVIEW_SERIAL_RESPONSE.md/JSON preserve evidence, including no saved .dsl
waveform and the distinction between USB completion status and FPGA hit bit.
Prior bit-order blocker remains in force and is reproduced in real DSView.

User authorized register comparison then bit-order validation after d6dfd2e.
DSView owner has exclusive USB for real serial reference task
`dslcap-c-golden-20261007` in /home/wroberts/DSView-1.3.2; lead must wait for
explicit release before capturing. Radio owner task `dslcap-c-radio-20261007`
in /mnt/foo/nfs_share_for_pis/hydra_develop prepares safe isolated transactions
and waits for lead GO. dslcap_worker prepares only offline capture verification
under /tmp/dslcap-c-verify; no shared source edits or GPIO generation.
Live results: real372-byte serial register comparison PASS at100/200/400M.
At100M allfour NSS controls passed exactbytes/unique pattern, but allfour
serial trials timed out exit16. Exact-reference CH0-7 retry also timed out,
with its actual arm image byte-identical to DSView. C.2 is blocked, not passed;
C.3 and live higher-rate bit-order gates remain open. SERIAL_GOLDEN.json and
SERIAL_VALIDATION.md/JSON preserve results and limitations. Source mapping and
production code stayed unchanged; no speculative bit-order fix.

Both secondopinion tasks completed revision5, consumed and acknowledged.
DSView settings restored/USB released; radio owner restored pi133 after all9
frames, final modeSTBY_XOSC/errors0/IRQ0/TXlevel0 with original FIFO flags
rx0x03/tx0x27 preserved. Only newly raised FIFO_TX was cleared. Pi134 untouched,
no RF/GPIO/reset/config changes. Worker independently checked waveform and
source/guide; temporary verifier six synthetic tests passed. No new dependency.

User authorized Phase C item 1: serial-trigger implementation and offline
tests. Baseline b2c5369; spec TRIGGER_PLAN.md; read-only reference
/home/wroberts/DSView-1.3.2. No new dependencies or live hardware work.
DSView serial golden comparison, bit/byte-order and opcode hardware gates
remain pending and are not implied by offline implementation acceptance.

Ownership: dslcap_worker owns C/H, CMake, dslcap/tests, sigrok_hla.py,
tests/test_dslcap*.py and the two user READMEs. DS_LEAD owns plan, ledger,
integration and scoped commits. Independent reviewer owns SERIAL_REVIEW.md
after frozen handoff, with no implementation, Git or hardware writes.
Detailed handoff: /tmp/dslcap-phase-c-worker.md.

Review budget for this distinct group: 1/3 consumed. Cycle 1 allocated before
launching independent review-1 on the frozen wave. Allocate each cycle
before launch; only FAIL in cycles 1/2 permits a scoped fix wave; cycle 3
is terminal. Prior B and A2 review budgets remain unchanged.

Status: worker delivered frozen /tmp/dslcap-phase-c-handoff.md; lead verified
10/10 source fingerprints and clean diff whitespace. Worker reports CTest6/6
and Python42 PASS. Independent review cycle1 PASS; no hardware tests run.

review-1 independently built production/fake targets in /tmp/dslcap-c-review,
passed 6/6 CTest groups (26.50s), 42 Python tests (17.472s), 132 additional
C/Python parser comparisons and 10/10 frozen source hashes. Report:
SERIAL_REVIEW.md. No High/Medium/Low findings; one Warning preserves the live
gates. No fix wave needed. Lead inspected the complete baseline diff and
rechecked fingerprints after review. Worker and reviewer are finished.

Source decision: serial global stage value is 0, matching DSView's default
stage combo value 1 minus 1. Serial roles remain stages 0..3, with AND=1,
invert=0, non-contiguous and counts stage1=1, stage3=bits-1. Unused upper
16-bits value positions are X to express a last-N-bit match. The equivalent
DSView reference uses its bit editor with upper X; its hex helper instead
zero-pads all 16 positions. The first future golden uses bits=16/value=0x1c35,
avoiding that distinction. Source: DSView/pv/dock/triggerdock.cpp:306-408,
1041-1065 and libsigrok4DSL/trigger.c reset defaults. Hardware semantics
remain unverified until the separate Phase C live gates.

## B-simple-trigger — wave 1

Started 2026-10-07 by explicit user authorization to start Phase B after
TRIGGER_PLAN.md review corrections. Baseline implementation: 35cd129.
Spec: TRIGGER_PLAN.md, corrected timeout/abort contracts included.
Reference source: /home/wroberts/DSView-1.3.2.

Scope: simple buffer triggers, C lifecycle/metadata/drain policy, Python
buffer/trigger integration, meaningful fake-driver and regression coverage,
documentation and Phase B hardware validation. Serial CLI and optional
--t0 trigger are deferred to later scope; default capture-start timestamps
and chronological trigger marker are required. No new dependencies.

Ownership:
- dslcap_worker: dslcap C/H implementation, CMakeLists.txt and tests;
  sigrok_hla.py, focused Python trigger tests, dslcap/README.md and
  sigrok_hla_readme.md. No edits to plans/task ledgers/review/validation.
- DS_LEAD: this ledger, TRIGGER_PLAN baseline, validation/integration,
  scoped commits, exclusive USB capture and app-owner coordination.
- Independent reviewer: sole author of TRIGGER_REVIEW.md after handoff;
  read-only implementation and no hardware/Git actions.

Worker handoff is /tmp/dslcap-phase-b-worker.md. Worker must deliver code,
exact validation results, deviations, remaining risks and documentation.
Lead waits for that handoff before integrated review/hardware execution.
No direct GPIO generation on radio-connected lines. Radio activity is via
the existing application owner with armed GO handshakes. DSView and dslcap
must never own the analyzer concurrently.

Review budget for this distinct group: 1/3 consumed. Cycle 1 allocated before
launching review-1; follow-ups/fixes stay in this group and budget.
Prior bring-up/G3 implementation and plan reviews are separate completed
scopes. Only FAIL in cycle 1 or 2 permits a scoped fix wave; cycle 3 terminal.

Decisions:
- A1 high-rate packing evidence is recorded in TRIGGER_PLAN.md by the
  DSView agent; lead has not independently rerun that historical evidence.
- A2 continuous-clock ppm acceptance was open when Phase B started, not
  waived. It subsequently passed; see A2_VALIDATION.md and A2_REVIEW.md.
- Trigger timeout uses best-effort cached status/grace; no precise
  hardware-event deadline guarantee. A completed header cannot recover
  data discarded by an abort; success always requires exact count + END.
- Streaming defaults and old one-line META output remain compatible.
- Routine implementation ambiguity is resolved by lead with the accepted
  design; a genuinely new dependency/plan requires operator review.

Status: Phase B implemented, committed as fb85f37, independently reviewed
PASS, and live gates B.1–B.6 passed. The separate A2 check subsequently passed.

## Parallel reference evidence

[x] DSView agent dsview-1-3-2-ec completed task
`dslcap-phase-b-golden-20261007`: actual DSView EP2 register images for
100M/200M simple CH3 falling, 10% position, eight low channels. Temporary
artifacts only, no repository edits or radio activity. It released exclusive
analyzer access before lead captures. Delivery/readiness and exact settings are
tracked in the secondopinion task; accepted delivery is not completion.

[x] dslcap_worker delivered a frozen implementation handoff and is idle.
[x] Lead integration checks after frozen worker handoff.
[x] Independent review cycle 1 PASS.
[x] B.1 golden-image comparison, B.2 condition/position captures,
    B.3 placement, B.4 timeout/cancel/reopen, B.5 deadline races,
    B.6 single/dual-radio Python/HLA checks. Actual evidence and limitations:
    TRIGGER_VALIDATION.md and TRIGGER_VALIDATION.json.

### Wave 1 implementation decisions

Worker chose the full nominal buffer grace window (340 ms from 17 x 20 ms),
explicitly not a hardware-status age bound. Packet/timeout commitment uses
a shared mutex; callback owns copied header validation, converter limit and
ordered META publication. Lead requires releasing that mutex before
ds_stop_collect or callback-dependent driver calls, since stop joins the
driver worker. Irreversible abort gates late packets; exact count/END cannot
erase an explicit signal outcome. Forced-upload intent is published before
WAIT_UPLOAD and reconciled against its TRUE/FALSE result before accepting
short output. Buffer writer progress re-arms the watchdog; test-only timing
scales are compiled in a separate fake core. Serial and t0 remain excluded.

### B.1 reference side complete

[x] DSView agent completed real GUI register logging at 100M/200M/400M.
Task dslcap-phase-b-golden-20261007 revision 5 consumed and acknowledged;
result hash 98c83008f8f40987fb2d5839985aa97c46aa0b5364c1aa360be040e8696eef47.
Lead checked all three 372-byte image hashes and preserved raw hex/settings
in TRIGGER_GOLDEN.json. 100M/200M: CH0-7, N=1000448; 400M: CH0-3,
N=2000896. All use simple3:f, pos10, vth1.6, filter/RLE/loop/instant off.
The reference tpos values are100032/100032/200064. DSView session restored
byte-identical and USB released per owner. Lead now owns USB again.

The dslcap comparison half of B.1 subsequently passed at all three rates.
Temporary shim, raw packets, logs and decoder: /tmp/dslcap-golden/.

### Wave 1 handoff and review cycle 1

Worker handoff: /tmp/dslcap-phase-b-handoff.md. Lead verified all 19 source
fingerprints and clean diff whitespace. Production binary:
/tmp/dslcap-phase-b-build/dslcap (real DSView source, default timing).
Offline CTest: 5/5 groups PASS, 23.76s; Python: 36 tests PASS. Tests exercise
real control/ring/callback code with a fake driver; they do not close live gates.
Independent synthesizer review-1 receives /tmp/dslcap-phase-b-review.md,
whole wave and exact frozen worker scope. No analysts requested. Sole review
file writer: review-1, dslcap/TRIGGER_REVIEW.md. Cycle 1 of 3 allocated here
before launch. No previous findings in B-simple-trigger.

Cycle 1 PASS consumed: 0 High+, 1 Low (duplicate --on-timeout action),
2 Warnings (fixed-sleep phase signal tests and open live gates). Reviewer
independently passed 5/5 CTest, 36 Python tests and 19/19 source fingerprints.
No fix wave was opened. Report: TRIGGER_REVIEW.md. Single-occurrence timeout
options work; retained duplicate-option issue is not claimed corrected.

B.1 implementation side PASS: all three real USB 372-byte setting images
match the saved DSView bytes exactly at100M/200M/400M. Idle captures exited16
as expected, with elapsed1.637/1.637/1.641s including initialization/cleanup.
Artifacts: /tmp/dslcap-b-golden-live/summary.json and per-rate shim/raw/stderr.
This closes register equivalence; subsequent live gates are recorded below.

### Live acceptance complete

B.2: all 300 captures (20 per condition/rate, five conditions, three rates)
passed with full count and exact condition at returned real_pos; zero-sample
tolerance measured even at 200M/400M. B.3: all positions and early-hit cases
passed. B.4: fail/upload/wait and observed-hit cancellation passed, every
immediate reopen succeeded. B.5: ten scheduled isolated GPIO20 pulse cases
passed before/after T and near nominal grace; forced real-hit short output
retained K/count, late untriggered upload retained none/count, abort never
returned success. GPIO20 restored and reverified input/pull-down/low.
B.6: live single400M and dual100M HLA exactly matched prior25M streaming
GetVersion sequences (120 normalized lines per port), chronological output.

Application owner supplied bounded read-only GetVersion waves; no radio
configuration, reset or radio-connected GPIO writes. User confirmed CH6 to
isolated pi133 GPIO20 physical38 for B.5; application owner confirmed no
conflict. Lead restored the original pin state after every pulse. No new
dependency. Temporary verifier issues (NumPy mask width and forced-count
assumption) were corrected and affected checks rerun; implementation stayed
frozen at fb85f37. No additional review cycle consumed.

See TRIGGER_VALIDATION.md/JSON for counts, timing uncertainty and artifact
locations. A2 was subsequently completed in A2_VALIDATION.md, not waived.
Phase C serial
and optional --t0 trigger remain excluded. Retained review Low issue is
documented; no automatic fix wave was opened after PASS.
