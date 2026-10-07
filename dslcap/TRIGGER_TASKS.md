# Trigger implementation tasks

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
- A2 continuous-clock ppm acceptance remains open. Starting Phase B does
  not claim A2 completion or waive remaining hardware validation.
- Trigger timeout uses best-effort cached status/grace; no precise
  hardware-event deadline guarantee. A completed header cannot recover
  data discarded by an abort; success always requires exact count + END.
- Streaming defaults and old one-line META output remain compatible.
- Routine implementation ambiguity is resolved by lead with the accepted
  design; a genuinely new dependency/plan requires operator review.

Status: worker wave 1 delivered; independent review cycle 1 PASS; live validation underway.

## Parallel reference evidence

[-] DSView agent dsview-1-3-2-ec owns task
`dslcap-phase-b-golden-20261007`: actual DSView EP2 register images for
100M/200M simple CH3 falling, 10% position, eight low channels. Temporary
artifacts only, no repository edits or radio activity. It has exclusive
analyzer access until it reports clean release. Lead and worker must not
capture concurrently. Delivery/readiness and resulting exact settings are
tracked in the secondopinion task; accepted delivery is not completion.

[-] dslcap_worker acknowledged implementation handoff and is working offline.
[ ] Lead integration checks after frozen worker handoff.
[ ] Independent review cycle 1, not yet launched.
[ ] B.1 golden-image comparison, B.2 condition/position captures,
    B.3 placement, B.4 timeout/cancel/reopen, B.5 deadline races,
    B.6 single/dual-radio Python/HLA checks. Do not substitute offline
    tests for hardware evidence or mark an unrun gate complete.

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

The dslcap comparison half of B.1 is still pending the worker's executable.
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
This closes register equivalence only; other hardware gates remain open.
