# Paused after native aligned bit-order validation

**Resumed 2026-10-07 19:20 PDT. C.3 has since passed; see OPCODE_VALIDATION.md.
This file is retained as the record of the pause.**

User requested pause on 2026-10-07 PDT / 2026-10-08 UTC. Do not start further
tests until the user resumes. Completed evidence is committed in `c198272`
on `main`; see SERIAL_ALIGNED_VALIDATION.md/JSON and TRIGGER_PLAN.md.

C.1 register comparison passed at 100/200/400 MS/s. Native aligned C.2 passed
at 100 MS/s, CH0–7: intended 1c35 matched twice at clock32; three reversals
timed out; all four separate NSS controls passed. One expired control and its
late emission are retained as excluded scheduling evidence. Independent
review passed 4,923 field comparisons. Production mapping remains unchanged.

Next work on resume: C.3 opcode validation. Higher-rate live bit-order checks
remain open; optional trigger-relative timestamps remain unimplemented.
Reference source: `/home/wroberts/DSView-1.3.2`.

Last verified hardware state: native capture exited and USB released. Radio
owner restored pi133 at 02:07:47 UTC: STBY_XOSC, IRQ0, errors0, TX FIFO0,
sticky flags RX03/TX27 retained. App PID7973 and original mode0/8-bit/8MHz SPI
preserved; GPIO20 input/pull-down/low; no GPIO/RF/reset/config changes. Pi134
untouched. These are last verified states, not a guarantee after other agents
or the operator use the hardware during the pause. Recoordinate ownership
and baseline before any resumed test; never drive radio-connected pins.

Wiring: pi133 CH0=SCLK, CH1=MISO, CH2=MOSI, CH3=NSS, CH4=BUSY(pin12),
CH5=DIO8(pin29); CH6 disconnected. Pi134 CH8=SCLK, CH9=MISO, CH10=MOSI,
CH11=NSS, CH12=BUSY(pin12), CH13=DIO8(pin29). Both grounds connected.

Radio task `dslcap-g-radio-20261008` completed revision5, consumed/acknowledged;
owner hydra-develop-26 UUID75e80b13-b652-47db-a4d6-ff7d836a30d4.
Worker `dslcap_worker` thread01a1175b-6787-7232-92a6-a80ffff3dcd4 finished
offline verification and was told to remain idle for the pause. No pending GO.
Rediscover live owners on resume; do not replay old task messages.

Raw capture artifacts and temporary verifiers remain under `/tmp/dslcap-*`.
A local backup outside `/tmp` is saved under ignored `dslcap/local-evidence/`:
`pause-20261008.tar.gz`, with sibling SHA-256 file. It includes all existing
`/tmp/dslcap-*` artifacts and `/tmp/dsview-*.dsl` captures at pause time.
Archive paths retain the `tmp/` prefix; extract into a chosen scratch directory,
not over live files. The archive is local only, not committed or pushed.
Committed JSON retains the current validation's evidence, hashes and reviews.
Unrelated pre-existing untracked files were left untouched.

Archive verification: 475 top-level sources, 5,739 archive members,
67,760,074 compressed bytes. SHA-256:
`96f3d7d1bc7e08205f979f7b3e37b4cdcba39b73f26507e764bd6ba0084ca2aa`.
All 40 current capture artifacts inside the archive match the committed hashes.
