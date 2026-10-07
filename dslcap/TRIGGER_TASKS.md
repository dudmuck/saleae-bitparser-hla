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

Review budget for this distinct group: 0/3 consumed. First synthesizer
launch will consume cycle 1; follow-ups/fixes stay in this group and budget.
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

Status: worker implementation assigned; independent review and live gates pending.
