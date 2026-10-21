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

## Phase 2 implementation review — PASS (offline)

Task `dslcap-u3pro16-p2-20261020` (request SHA-256 `375057fa…2837`),
implemented offline by ds-worker on `c223b79`. The lead rebuilt
independently: ctest 8/8 (adds `model_contract`), `python3 -m unittest
discover -s tests` 87 OK with 8 skipped. The half/quarter factor (Plus ×2/×4,
ADF4360 ×1/×2, stream and buffer) was verified by both against
`dsl.c:1065–1085`, `:1132`, `:1181` and `dslogic.c:1289–1290`. The test
enumerates every 16-bit mask for all three profiles. `--list-modes`
(schema 1) for U3Pro16/super lists 125M×16, 250M×12, 500M×6 and 1G×3
(`num` 8) streams, and buffer 500M×16 / 1G×8 with `validated=false`.
`plus/super` exits 2. U3Pro16 buffer, trigger, serial and pattern paths are
fenced (exit 8) before activation; the reload helper refuses non-Plus models.

`pi_pattern.c` was reviewed in full for Pi safety: W1-only mask, input
precheck, CLR-then-SET order with a single `prev & next` intermediate,
SCLK-only writes, bounded duration, restore and readback on exit and signals.
Non-blocking notes: GPFSEL/pull updates are read-modify-write on shared
registers; `spi` is ~33 kbit/s at the 10 µs minimum step; the `validated`
field means "frontend enabled", not live-validated. No live gate has run
yet (U2–U5 pending).

## Phase 2 live gates — 2026-10-20 16:05–16:35 PDT

Built from `0536d78`: production `dslcap` SHA-256 `fd8252df…1160`;
`pi_pattern.c` SHA-256 `f9a195f1…1ed1`, copied to `~/dslcap-w1/` on each Pi
and built there with `gcc -O2 -Wall -Wextra -Werror` (hashes matched).
Analyzer `2a0e:002a` at bus 2 address 6, SuperSpeed. W1 wiring as in the
plan. Raw captures (3.7 GB) and logs are local under
`dslcap/local-evidence/u3pro16-phase2-20261020/`, with SHA-256 lists
`u3pro16-phase2-20261020{,.raw}.sha256`. The analysis scripts used
(`gate.sh`, `skew_analyze.py`, `proj_analyze.py`) are saved there too.

**Preflight.** Both Pis idle (load 0.00, no radio processes; pi133 had
`lr20xx_module` loaded but unused, pi134 none). `hydra-develop-1a` had
reported no use of either Pi; other sessions were not asked. All 16 W1 lines
were `unused` inputs at default pulls. Full `pinctrl get 0-27` recorded as a
baseline. Target disk sustained 1.6 GB/s (`dd`, 2 GiB, fdatasync).

**Passive baseline.** 25M ×16 with no drive, 12.5 M samples: every sample
`0x2020`, i.e. CH5 and CH13 high (GPIO6 pull-up on each Pi), the rest low.

| Gate | Capture | Result |
|---|---|---|
| U2 walk | 25M ×16, 2 s | PASS both groups (29,094 / 29,038 states); 65 / 64 CLR/SET intermediates; 0 in the Gray section |
| U2 walk | 125M ×16, 1 s | PASS both groups (14,712 / 14,643 states); 194 / 173 intermediates; 0 in Gray |
| U3 prbs | 125M ×16, 2 s, 500 MB | Transport clean (analysis); `check_pattern.py` FAIL (see below) |
| U3 prbs | 250M ×12 CH0–11, 1 s, 500 MB | Transport clean, pi133 |
| U3 prbs | 250M ×12 CH4–15, 1 s, 500 MB | Transport clean, pi134 |
| U4 substitute | 500M ×6 CH0–5, 1 s, 500 MB | Transport clean, pi133 |
| U4 substitute | 1G ×3 CH0–2, 1 s, 1 GB | Transport clean, pi133 |
| U4 substitute | 500M ×6 CH8–13, 0.5 s, 1 GB/s output | Transport clean, pi134 |
| U5 spi | live `sigrok_hla.py`, 25M, both buses | PASS: 100/100 frames per bus byte-exact (`[SPI]` = pi133, `[SPI_B]` = pi134) |

No capture reported overflow; every finite capture returned its exact
sample count. After every run both Pis' generators restored their lines,
and the final `pinctrl get 0-27` was identical to the baseline on both.

**Why `check_pattern.py` fails U3, and why the transport is still clean.**
At ≥125 MS/s, multi-pin transitions are visible part-way: lines set in one
`GPSET0` write don't switch at exactly the same instant, so the analyzer
catches short states between the old and new values. The production
checker later showed these never exceed one sample, even at 1 GS/s, so the
pin-to-pin skew is about 1 ns or less. The CLR/SET gap
(`prev & next`) can also last longer when the CPU is interrupted between its
two writes (seen up to 4.2 µs). The checker accepts only a single
`prev & next` per step, so it fails at the first skew sample. Independent
analysis (`proj_analyze.py`) walks every run against the PRBS8 cycle. It
accepts `prev & next` of any length, plus states bitwise between the old and
new values lasting ≤ 12 ns. It requires every expected state in order, and
fails on anything else or a lost sync. Result: **0 faults** in all seven
prbs captures. For example, at 125M ×16 pi133 had 1,737,166 expected
states in order, 230,487 skew samples and 547,232 CLR/SET gaps. Every
skew state fell within the scratch analysis's 12 ns bound. That bound was
computed in floating point and came out at 2, 4, 7 and 13 samples at 125M,
250M, 500M and 1G; the exact bound is 2, 3, 6 and 12 (found by ds-worker).
The production checker uses the exact bound, and its rerun (below) is the
record.

**Relative timebase (U4 substitute).** GPCLK clocks (P3) are not available
without root, an overlay or PWM, so absolute frequency was not measured.
Instead, the PRBS generator's CPU-bound rate gives a ratiometric check. In
expected states per nominal second, normalized to the full cycle, pi133 gave
868,583 at 125M, 872,310 at 250M, 856,672 at 500M and 868,476 at 1G. pi134
gave 862,084 at 125M, 834,642 at 250M and 830,976 at 500M. A wrong ×2 in the
new 500M/1G factor would show as ~1.7 M or ~430 k. The spread (≤ 4%) is
generator variation, so this proves the factor but not calibration.

**Walk timing.** `--step-us 10` actually produced ~69 µs steps (~1,720
samples at 25M): Linux `nanosleep` overhead. Every walk dwell therefore
raised the warning-only 0.5×–3× band warning; that is expected.

### Checker follow-up `dslcap-u3pro16-p2c-20261020` — PASS

ds-worker updated `check_pattern.py` (request SHA-256 `8bf10f2f…6403`):
- **Skew:** states bitwise between A&B and A|B are accepted up to an exact
  integer bound, `ceil(skew_ns × rate)` with a 12 ns default: 2/3/6/12
  samples at 125M/250M/500M/1G.
- **CLR/SET gap:** `A&B` is accepted at any dwell.
- **Cap:** at most popcount(A^B)+1 off-path runs per transition.
- **Anchoring and lane subsets:** it anchors on the first stable run, and
  `--lanes` projects onto a subset of lanes.
- **Timebase:** it reports transitions per nominal second.

`pi_pattern` now waits 1 ms after restoring the lines before its readback.
The lead reran the Python suite (91 OK, 8 skipped) and the checker on six
saved captures with identical results; ds-worker ran all eight:

| Capture | Args | Result | Max skew dwell | Max off-path / step |
|---|---|---|---|---|
| u2-25m, u2-125m | walk, groups 0 and 8 | PASS | 0 | 1 |
| u3-125m | prbs, groups 0 and 8 | PASS | 1 | 2 |
| u3-250m-lo / -hi | prbs, group 0 / 8 | PASS | 1 | 2 / 3 |
| u4-500m-pi133 | prbs, `--unitsize 1 --lanes 0x3f` | PASS | 1 | 3 |
| u4-1g-pi133 | prbs, `--unitsize 1 --lanes 0x07` | PASS | 1 | 2 |
| u4-500m-pi134 | prbs, `--lanes 0x3f`, group 8 | PASS | 1 | 3 |

Live settle check: `pi_pattern.c` SHA-256 `1c5900bb…7e01`, rebuilt on both
Pis, `walk --duration 0.2`. Both reported `levels=0x00000040` (GPIO6 high
only), and `pinctrl get 0-27` matched the baseline on both.

### U4: absolute frequency against GPCLK0 — PASS

The user approved bounded `sudo` writes to GPCLK0 only (2026-10-20 16:29
PDT). `pi_gpclk.c` (task `dslcap-u3pro16-p2d-20261020`, ds-worker; lead
review PASS; SHA-256 `084c1194…738d`) was built on each Pi. GPCLK0 was idle
on both beforehand (kernel `gp0` enable count 0) and GPIO20 an input with
pull-down. Each run: `sudo -n timeout 60 ./pi_gpclk --source osc --divi 2`,
i.e. the 54 MHz crystal ÷ 2 = 27 MHz on GPIO20, for 45 s (pi133) / 35 s
(pi134), with 1 s single-channel captures taken during it. Checked with
`check_clock.py --expect-hz 27000000`.

| Pi (channel) | Rate | Measured | Error | 1 ms windows min/max | Duty | Glitches / missing |
|---|---|---|---|---|---|---|
| pi133 (CH2) | 125M | 27 000 637.728 Hz | +23.620 ppm | +16 / +27 ppm | 53.19% | 0 / 0 |
| pi133 (CH2) | 250M | 27 000 637.756 Hz | +23.621 ppm | +20 / +27 ppm | 53.22% | 0 / 0 |
| pi133 (CH2) | 500M | 27 000 637.972 Hz | +23.629 ppm | +22 / +25 ppm | 52.99% | 0 / 0 |
| pi133 (CH2) | 1G | 27 000 638.270 Hz | +23.640 ppm | +23 / +24 ppm | 53.00% | 0 / 0 |
| pi134 (CH10) | 125M | 27 000 384.296 Hz | +14.233 ppm | +8 / +19 ppm | 53.53% | 0 / 0 |
| pi134 (CH10) | 250M | 27 000 383.648 Hz | +14.209 ppm | +11 / +16 ppm | 53.56% | 0 / 0 |
| pi134 (CH10) | 500M | 27 000 382.972 Hz | +14.184 ppm | +13 / +16 ppm | 53.34% | 0 / 0 |

- **Rate agreement:** each Pi's clock agrees across all sample rates to
  within 0.05 ppm. That proves the 500M (×1) and 1G (×2, QUAR) stream
  timebases are exactly consistent with 125M/250M, far tighter than the
  ratiometric check.
- **Absolute error:** the analyzer reads the two independent Pi crystals
  +23.6 and +14.2 ppm fast. Their 9.4 ppm difference is Pi-to-Pi. The
  analyzer's own error relative to true time can't be separated from the
  crystals' without a calibrated reference, but it's bounded at tens of ppm.
- **Duty:** ~53% is the clock as seen through the analyzer's input
  threshold, consistent at every rate.
- **Restore:** both runs exited 0 with a matching readback: GPIO20 input,
  pull-down, low; GP0CTL restored to its original `0x200` (MASH field) with
  ENAB clear; DIV 0. `pinctrl get 0-27` was identical before and after, and
  the kernel still showed `gp0` unused.

Raw captures and checker JSON are in the evidence directory, with SHA-256
lists `u3pro16-phase2-20261020.u4c{,.raw}.sha256`.

## Plus regression on the same W1 wiring — PASS

2026-10-20 16:59–17:02 PDT. The U3Pro16 was unplugged and the Plus
(`2a0e:0034`) connected with the same W1 wiring. Its first plug-in on port
1-1 failed USB enumeration (descriptor reads timed out with `-110`); after
the user replugged it, it came up as bus 1 address 63, high speed. Same
production `dslcap` as the Phase 2 gates (SHA-256 `fd8252df…1160`). Both
Pis' `pinctrl get 0-27` baselines were unchanged since the U3Pro16 gates.
Evidence is in `dslcap/local-evidence/plus-regression-20261020/`, with
SHA-256 lists `plus-regression-20261020{,.raw}.sha256`. The run script
`plus_regress.sh` is saved there.

- **Scans.** Automatic and `--model plus` both exit 0:
  `2a0e:0034 DSLogic PLus bus=1 address=63 speed=high activated
  security=pass hdl=checked`. There was a cold FPGA load of
  `DSLogicPlus-pgl12-2.bin` (530 620 bytes), "Security check pass!" on both
  opens, and HDL 0x0e. `--model u3pro16` exits 4 and lists the Plus.
- **Idle baseline.** 20M ×16: every sample `0x2020`, the same mapping as
  on the U3Pro16.
- **Walk.** 20M ×16, 2 s: PASS on both groups, with 0 Gray-section gaps.
- **PRBS** (`check_pattern.py`, all exit 0):

  | Capture | Mode | Lanes | Result |
  |---|---|---|---|
  | 50M ×6, CH0–5 / CH8–13 | stream | 0x3f, both Pis | PASS |
  | 100M ×3, CH0–2 | stream | 0x07 | PASS |
  | 100M ×16, 16 M samples | buffer | 0xff, both groups | PASS |
  | 200M ×8, CH0–7, 16 M samples | buffer (×2) | 0xff | PASS |
  | 400M ×4, CH0–3, 32 M samples | buffer (×4) | 0x0f | PASS |

- **GPCLK0** on pi133, 27 MHz (osc ÷ 2), on CH2:

  | Mode | Measured | Error | Duty | Glitches / missing |
  |---|---|---|---|---|
  | stream 100M | 27 000 378.620 Hz | +14.023 ppm | 52.76% | 0 / 0 |
  | buffer 100M | 27 000 379.391 Hz | +14.052 ppm | 52.81% | 0 / 0 |
  | buffer 200M (half, ×2) | 27 000 379.641 Hz | +14.061 ppm | 52.80% | 0 / 0 |
  | buffer 400M (quarter, ×4) | 27 000 379.641 Hz | +14.061 ppm | 52.81% | 0 / 0 |

  The readings agree within 0.04 ppm, so the rewritten chooser keeps the
  Plus's ×2/×4 factors exact. The same pi133 crystal read +23.6 ppm on the
  U3Pro16, so the two analyzers' timebases differ by about 9.6 ppm. The
  `pi_gpclk` restore readback matched (GP0CTL `0x200`, DIV 0, GPIO20
  input / pull-down / low).
- **SPI.** Live `sigrok_hla.py` at 25M on both buses, which selected the
  Plus: 100/100 frames byte-exact on `[SPI]` (pi133) and `[SPI_B]`
  (pi134).
- **Pins.** After every run, both Pis' `pinctrl get 0-27` was identical to
  the baseline.

**Still open:** Phase 3 (buffer and triggers on the U3Pro16). W2 is
recorded below.

## W2 real radio traffic on the U3Pro16 — PASS

2026-10-20 19:14–19:32 PDT. The user rewired the analyzer passively to
both Pis' radio lines (CH0–7 pi133, CH8–15 pi134, same order per Pi):

| CH | Signal | GPIO | Header pin |
|---|---|---|---|
| 0 / 8 | SCLK | 11 | 23 |
| 1 / 9 | MISO | 9 | 21 |
| 2 / 10 | MOSI | 10 | 19 |
| 3 / 11 | NSS (`spi0-cs,cs0_pin=23`) | 23 | 16 |
| 4 / 12 | BUSY | 18 | 12 |
| 5 / 13 | DIO8 | 5 | 29 |
| 6 / 14 | DIO11 | 24 | 18 |
| 7 / 15 | DIO7 | 12 | 32 |

Ground stays on pin 39. No pin was driven from this host. The user rebooted
both Pis into `mcp_radio` with spidev, so the radio traffic came from
another session (xiao_phy_throughput), at the user's direction. It ran
bounded command bursts with no RF: GetErrors, GetVersion, ClearTxFifo,
WriteTxFifo with an incrementing payload of L bytes, and GetTxLevel. That
is spidev mode 0 at 8 MHz requested, 7.8125 MHz on the bus. Its log has
one line per transfer, and it reported every server result matching
(version 0x0118, TX level = L, no errors).

- **Idle.** 25M×16, 10 s, before traffic: every sample `0x2828` (NSS and
  DIO8 high on both Pis), with no edges on any channel.
- **Window A, live decode.** `sigrok_hla.py --dslogic --dsl-model u3pro16`
  at 25M with all 16 channels, plus a scratch HLA that prints each
  transaction's MOSI and MISO (the LR2021 HLA is no longer on this host).
  pi133 used L=250/64 and pi134 L=200/100.
- **Window B, raw.** `--raw-out` at 125M (`mask=0x0f0f`; raw mode captures
  the SPI channels only), 60 s, 15 GB, then offline replay. pi133 used
  L=180 and pi134 L=240. The ring high-water was 25.8 MB.

| Window | Pi | Transfers decoded / logged | MOSI bytes | MISO read-data fields | Mismatches |
|---|---|---|---|---|---|
| A 25M live | pi133 | 358 / 358 | 8774 | 104 | 0 |
| A 25M live | pi134 | 358 / 358 | 8424 | 104 | 0 |
| B 125M raw | pi133 | 179 / 179 | 4962 | 52 | 0 |
| B 125M raw | pi134 | 179 / 179 | 6462 | 52 | 0 |

- **Order.** Transfers matched in order, with the log's `seq` contiguous.
  MISO matched on the read-data bytes. The log has no command-phase MISO
  or stat bytes; the decode shows those as `0521`, `0721` and `0321`.
- **Timing at 125M.** SCLK rising edges are exactly 8 × the MOSI bytes on
  both Pis (39696 and 51696), so no clock edges were lost. The median
  period is 16 samples (7.8125 MHz), the minimum high phase 64 ns and the
  minimum low phase 56 ns.
- **Window A status lines.** 716 BUSY edges per Pi were decoded live.
- **Limits.**
  - The peer's log was generated from the script it ran plus the server's
    results, not from an independent bus trace.
  - Its timestamps are per burst, so the matching is by order.
  - In both windows, pi134's burst started about 2–10 s after pi133's, so
    the two buses were active one after the other, not concurrently.
  - DIO8, DIO11 and DIO7 did not toggle in this workload.
- **Pins.** `pinctrl get 0-27` on both Pis was identical before and after.

Evidence (git-ignored): `dslcap/local-evidence/u3pro16-w2-20261020/`,
hashed in `u3pro16-w2-20261020.sha256` and `.raw.sha256`. The raw-file
SHA-256 for `b-125m.raw` is `8aa379a7…6cdf`. It includes the peer logs,
`Txn2Hla.py`, `w2cmp.py` and `sclk_stats.py`.
