# DSLogic U3Pro16 validation

Plan: [U3PRO16_PLAN.md](U3PRO16_PLAN.md). Raw logs are local only, under
ignored `dslcap/local-evidence/u3pro16-u1-20261020/`, with SHA-256 sums in
`u3pro16-u1-20261020.sha256`.

## Phase 1 implementation review — PASS

Task `dslcap-u3pro16-p1-20261020`, implemented offline by ds-worker (Codex)
on `eb7f8e1`; request SHA-256 `c353db6b…baeff70`. The lead reviewed the diff
and rebuilt independently in a fresh directory: `cmake`, build and
`ctest` all exited 0, 7/7 targets passed (capture_core, driver_watchdog,
driver_config, bringup_contract, serial_contract, trigger_contract,
capture_contract). No blocking findings. Minor: the `u3-no-security` case
duplicates `u3-super` (the fake U3Pro16 never prints a security message),
and `found[2]` and the `--model` indices depend on model-table order by hand.

## U1: bring-up on the U3Pro16 — PASS, forced reload excluded

2026-10-20 PDT, production `dslcap` from the reviewed tree, SHA-256
`39b100087b289d16411fd2d0e8abe1675052a9327962e1d226599f1258515da4`.
Analyzer `2a0e:002a`, SuperSpeed. No probes driven; no capture.

| Step | Result |
|---|---|
| `--scan -v` (address 5) | exit 0; cold FPGA load of `DSLogicU3Pro16.bin`, 465304 bytes; `security=n/a`; HDL 0x0e on reopen |
| `dslcap_reload_fpga --scan -v` (forced volatile reload) | **exit 124**: stalled in `dsl_fpga_config` and was killed at the 30 s timeout |
| `--scan` after the stall | exit 15, driver watchdog; FPGA configure stalled again |
| `--scan -vv` diagnostic | exit 15, stalled at `Configure FPGA` |
| User replugged USB (address 6); `--scan -v` | exit 0; cold FPGA load, 465304 bytes; HDL 0x0e |
| Immediate `--scan -v` (warm) | exit 0; no FPGA load (already configured branch); HDL 0x0e |

Scan line: `2a0e:002a DSLogic U3Pro16 bus=2 address=6 speed=super activated
security=n/a hdl=checked`. No security message from the driver, as expected
without `CAPS_FEATURE_SECURITY`.

**Forced reload on the U3Pro16 is unsafe.** The helper calls DSView's
`dsl_fpga_config` on an already configured FPGA; DSView itself only loads
from `dsl_dev_open` when the FPGA isn't configured. On this model that left
the FX3/FPGA unable to configure again until a USB power cycle. The same
helper passed on the Plus. Don't run `dslcap_reload_fpga` on the U3Pro16;
cold and warm open via the production binary are the validated paths. The
helper should refuse non-Plus models (Phase 2 follow-up).

The Plus was not attached; its regression scan is still owed before merge.
