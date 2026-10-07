# dslcap execution record

## Authority and baseline

The approved requirements and architecture are in [PLAN.md](PLAN.md).
Use `/home/wroberts/DSView-1.3.2/` as the read-only driver/API reference.
The coordinator owns task state, plan changes, review, and Git commits.
The user authorized USB capture and SSH GPIO generation on pi133/pi134.
New dependencies that require a new plan require operator review. Physical
wiring may require the operator. No NVM writes are authorized.

## G1-capture: Phase 0 bring-up assignment

- [!] Implementation owner: existing Claude worker `dslcap_worker`.
  Assignment ID reserved: `dslcap-phase0-20261007`. Not dispatched.
- Scope: create `dslcap/CMakeLists.txt`, C headers/sources under `dslcap/`,
  `dslcap/tests/`, `dslcap/README.md`, and `dslcap/.gitignore` as needed for
  Phase 0. Do not edit PLAN.md, TASKS.md, PLAN_REVIEW.md, sigrok_hla.py,
  unrelated files, or the DSView source tree. Other work may coexist in this
  checkout; preserve it. Do not stage or commit; coordinator owns Git.
- Role: implementation engineer. Consume PLAN.md build, firmware, CLI,
  configuration and logging contracts. Produce a buildable standalone C
  executable with `--scan`, `--fw-dir` and verbosity handling sufficient to
  prove device activation, FPGA/HDL checks and explicit security success.
  Do not claim enumeration alone proves activation.
- Acceptance: no Qt/Python build dependency; no copied DSView sources;
  diagnostics on stderr; reject failed or absent security-pass evidence;
  cleanup on failures; invalid firmware directory exits nonzero.
- Verification: `cmake -S dslcap -B /tmp/dslcap-build
  -DDSVIEW_SRC=/home/wroberts/DSView-1.3.2`, then
  `cmake --build /tmp/dslcap-build`; run the built binary's `--scan` and
  activation path against USB `2a0e:0034`, and exercise an invalid
  `--fw-dir`. Confirm DSView is not using the device before claiming it.
  Report missing existing dependencies before installing anything.
- Handoff: exact changed files, commands/exit codes, activation and security
  evidence, cleanup outcome, source references, blockers and residual risks.
  Coordinator waits for this worker's result, inspects the diff and reruns
  checks before accepting the task or committing implementation.

## Subsequent work

- [ ] G1-capture: implement capture configuration, cross-data conversion,
  bounded ring, writer, finite trimming and shutdown; focused offline tests.
- [ ] G1-capture: Phase 1 signal/reference validation. Resolve the plan's
  unitsize-1 initial scope versus 12/16-channel validation explicitly before
  implementing wider output; do not silently weaken acceptance.
- [ ] G1-capture: Phase 2 continuous, slow-consumer, overflow and reopen tests.
- [ ] G2-integration: Phase 3 Python backend, live stderr drain, documentation
  and real dual-SPI comparison. Assign exact files after capture contract is
  verified. Optional Phase 4 remains optional.
- [ ] Independent review of implementation and evidence; no review cycle has
  started and no implementation PASS is claimed.

## Blocker — 2026-10-07

`secondopinion workers` was queried twice. Neither directory snapshot listed
`dslcap_worker`; no other session was substituted and no task was dispatched.
Start or resume that named worker in `/home/wroberts/src/saleae-binparser`
to unblock the handoff. The installed secondopinion workflow verifies name
and checkout before delegation.

Read-only preflight: `/home/wroberts/DSView-1.3.2/` exists; `lsusb` lists
`2a0e:0034` (USB-based DSL Instrument v2). This proves enumeration only.
The installed `codex-agent-team` skills are available; the sample checkout
also exists at `/mnt/foo/sample-codex-agent-team/`.

This record lives beside the existing plan because repository `.codex/`
is read-only in this session. Unrelated untracked files are preserved.
