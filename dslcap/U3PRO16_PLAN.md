# dslcap: DSLogic U3Pro16 (2a0e:002a) support

Status, 2026-10-20: **proposed, not started.** No repository code has
changed. One scan-only bring-up probe ran from a scratch copy (below).

Plan review: session `hydra-develop-1a`, 2026-10-20 14:42 PDT, six points
plus one minor. All were checked against DSView 1.3.2 and accepted: the
stream half of the half/quarter rule moves into Phase 2, `--list-modes`
reports physical lanes and minimum rate, the VTH range is settled, gate
masks and output rates are sized, Plus default selection is preserved, and
the `.fw` note is added.

Goal: `dslcap` and `sigrok_hla.py --dslogic` work with a DSLogic U3Pro16 as
well as the DSLogic Plus PGL12 (`2a0e:0034`). Plus behavior, output format
and validated register images stay unchanged.

Sources: DSView 1.3.2 `libsigrok4DSL/hardware/DSL/dsl.h` (profiles at
lines 589 and 616, `channel_modes`), `dsl.c` (`dsl_fpga_arm`,
`dsl_dev_open`, USB 3 buffer sizing), `dslogic.c` (scan, VTH, `dev_open`).
Pi pin usage: `hydra_develop/examples/modem_pinout_linux.h` and live
read-only `pinctrl`/`gpioinfo` on both Pis.

## The device

`lsusb`: Bus 002 Device 004, `2a0e:002a`, "USB-based DSL Instrument v2",
bcdUSB 3.00, vendor interface, bulk EP2 OUT / EP6 IN at 1024 bytes,
negotiated at 5000M. DSView's profile names it "DSLogic U3Pro16". It is
the only DreamSourceLab device attached; the Plus is unplugged and can be
reconnected later for a regression check.

| | Plus (0034) | U3Pro16 (002a, SuperSpeed) |
|---|---|---|
| Bitstream | `DSLogicPlus-pgl12-2.bin` | `DSLogicU3Pro16.bin` (installed, SHA-256 `0ae5a6e5…a56e`) |
| Features | VTH, BUF, MAX25_VTH, SECURITY | VTH, BUF, USB30, ADF4360 |
| Stream modes | 20M×16, 25M×12, 50M×6, 100M×3 | 125M×16, 250M×12, 500M×6, 1G×3 (CH0–7 only) |
| Buffer modes | 100M×16, 200M×8, 400M×4 | 500M×16, 1G×8 (CH0–7) |
| Minimum rate | 50 kHz | 1 MHz (USB 2 `_3DN2` modes: 100 kHz) |
| Mode clock | hw_max 100M, pre_div 1 | hw_max 500M, pre_div 5 |
| FX3 `.fw` file | not installed; firmware in flash | `DSLogicU3Pro16.fw` not installed either |
| Buffer depth | 256 Mibit | 2 Gibit |
| Rate list | `samplerates400` | `samplerates1000` (adds 125M/250M/500M/1G) |
| Half/quarter rate | 200M / 400M | 500M / 1G |
| VTH register scale | `vth/3.3·(1/2)·255` | `vth/3.3·(1.5/2.5)·255` |
| HDL version | 0x0e | 0x0e (`DSL_HDL_VERSION` is shared) |

DSView also has a USB 2 (`LIBUSB_SPEED_HIGH`) profile for the same PID with
`_3DN2` stream modes (20M×16 … 100M×3). Its scan matches profile by PID
**and** negotiated speed (`dslogic.c:364`).

DSView uploads the `.fw` only when `dsl_check_conf_profile` fails, so a
unit whose FX3 has lost its flash firmware would fail to open here. The
current unit reported firmware 2.1. Mode `num` is the physical lane range
and `vld_num` the usable count: 1G modes have `num` 8, so only CH0–7 work.

## Probe evidence, 2026-10-20 ~14:28 PDT

Scratch copy of `dslcap/` under the session scratchpad, patched only in
`main.c`/`driver_config.c`: PID 0x002a, model "DSLogic U3Pro16", bitstream
`DSLogicU3Pro16.bin`, and the "Security check pass!" requirement dropped
(a failure message would still have rejected). `--scan -v` exited 0:
opened on 2.4 with firmware 2.1 twice, explicit HDL read 0x0e on the
reopen. No capture, no trigger, no probes installed. The probe printed the
hard-coded `2a0e:0034 DSLogic PLus` strings; those lines are not evidence
of model identity. Whether the FPGA load path ran and whether any security
text appeared were not checked; reading the log was denied by the
auto-mode permission classifier. Phase 1 repeats this properly.

## Plus-specific assumptions to remove

1. `main.c`: `TARGET_PID`, `BITSTREAM`, model-name filter, scan/stderr
   strings, and an unconditional security-pass requirement.
2. `driver_config.c` `profile()`: first `0034` entry. Must select by PID
   and negotiated USB speed, like DSView.
3. `driver_config.c` `dsl_choose_mode()`: the half/quarter divider rule is
   written for the Plus and applied only when `!stream` (lines 36–37).
   `dsl_fpga_arm` sets the HALF/QUAR bits whenever the rate equals
   `half_samplerate`/`quarter_samplerate`, in stream mode too
   (`dsl.c:1065–1066`). On the U3Pro16 those are 500M and 1G, both stream
   rates, so today's chooser rejects 1G×3 (500M % 1G ≠ 0). With ADF4360,
   `dsl_fpga_arm` also maps QUAR to `half_trig` and leaves `qutr_trig` false
   (`dsl.c:1132`, `:1181`). Re-derive the divider and packing for 500M/1G
   from source: the **stream** part in Phase 2, the trigger part in Phase 3.
4. `options.c`: `--vth` 0..2.5 V is dslcap's own limit. DSView's UI binds
   0.0–5.0 V for every model (`DSView/pv/prop/binding/deviceoptions.cpp:110`);
   MAX25_VTH changes only the register scale. Keep 2.5 V for both models:
   2.5 V writes register 115 on the U3Pro16 and 96 on the Plus, and the Pi's
   3.3 V logic needs ~1.6 V. Neither scale overflows up to 5 V.
5. Trigger/serial register goldens (`TRIGGER_GOLDEN.json`,
   `SERIAL_GOLDEN.json`, `DSVIEW_COMPARISON.json`) are Plus images.
   U3Pro16 needs its own DSView comparison; Plus goldens stay untouched.
   Its images differ even in stream mode: hw_max 500M and pre_div 5 give
   different `div_h`/`div_l`.
6. Tests: `tests/fake_driver.c` (name, PID), `tests/bringup.cmake`
   (expected scan line), `tests/reload_fpga.c` (bitstream name).
7. `sigrok_hla.py:711–719`: hard-coded Plus rate set (including 50k–500k,
   below the U3Pro16's 1 MHz minimum), per-rate channel counts and the
   200M/400M `max(bits)` lane rule; the module docstring and
   `sigrok_hla_readme.md` say "DSLogic Plus".

## Design

**Model table.** One static dslcap table, `{pid, model string, bitstream,
needs_security}`, for 0034 and 002a. The active DSView profile comes from
the attached device's PID and speed; nothing else hard-codes a model.

**Selection.** New `--model plus|u3pro16` restricts the match. Without it,
the Plus is chosen when present, so today's behavior is unchanged with both
attached; stderr notes the other analyzer. With only a U3Pro16 attached, it
is used. Zero matches exit `EXIT_DEVICE` listing what was found. Multiple
identical units remain unsupported (no serial number; `iSerial` is 0).

**Security.** Require the explicit pass message only when the profile has
`CAPS_FEATURE_SECURITY`, which is exactly when `dsl_dev_open` runs the check
(`dsl.c:1958`). A failure message always rejects. The Plus path is unchanged.

**Scan output.** `<vid:pid> <model> bus=… address=… speed=super|high
activated security=pass|n/a hdl=checked`. This changes the Plus line too
(`speed=`); update `bringup.cmake`. Only that test parses it; Python does not.

**Mode authority.** dslcap's `dsl_choose_mode` already validates against
the profile. Add `dslcap --list-modes [--model M] [--speed super|high]`,
printing the static profile with no USB access: the allowed rate list, and
per mode stream/buffer, min/max rate, `vld_num` (usable channels) and `num`
(highest physical lane + 1). `sigrok_hla.py` replaces its literal rate set,
channel counts and lane rule with that output (one subprocess call per
run); `--dsl-model` is passed through. Replay (`-i`) needs neither.

**Provenance.** `--raw-out` `FILE.json` gains `model` and `usb_speed`.
The stream META header is unchanged, so replay and old files are unaffected.

**Throughput.** Output above ~100–130 MB/s (Python decode) needs
`--raw-out` and offline replay; 250M×12 is ~500 MB/s from dslcap. The 256 MiB
ring gives about 0.5 s of slack at that rate. Document the live-decode limit;
no decoder work is in scope.

## Test signal source: Pi-driven wiring (W1)

The U3Pro16 has no probes installed. The user offered pi133 and pi134 to
drive all 16 inputs. Both are Raspberry Pi 4 Model B Rev 1.5, kernel
6.12.47, `dtoverlay=lr20xx`, gcc present. Read-only inspection on
2026-10-20 found these lines claimed or radio-mapped on both Pis, so they
are **excluded**: GPIO0/1 (HAT ID), 2/3 (I2C, adapter A4), 4 (`radio-reset`),
5 (DIO8), 7/8 (outputs, adapter), 9–11 (SPI0), 12 (DIO7 TX trigger:
every edge transmits), 14/15 (UART), 17/27 (adapter DIO routes), 18 (BUSY),
23 (`spi0 CS0`), 24 (DIO11), 26 (`debug`).

The remaining eight lines are `unused` inputs at default pulls on both Pis.
Proposed map, same order on each Pi; CH0–7 from pi133, CH8–15 from pi134:

| CH (pi133/pi134) | GPIO | Header pin | Role | Hardware function |
|---|---|---|---|---|
| 0 / 8 | 21 | 40 | SCLK | SPI1 SCLK, GPCLK1 |
| 1 / 9 | 19 | 35 | MISO | SPI1 MISO, PWM |
| 2 / 10 | 20 | 38 | MOSI | SPI1 MOSI, GPCLK0 |
| 3 / 11 | 16 | 36 | NSS | SPI1 CE2 |
| 4 / 12 | 13 | 33 | aux | PWM |
| 5 / 13 | 6 | 31 | aux | GPCLK2 |
| 6 / 14 | 22 | 15 | aux | — |
| 7 / 15 | 25 | 22 | aux | — |

Ground: one analyzer ground per Pi, header pin 39 (or 34). The `--spi 0,1,2,3
--spi 8,9,10,11` decoder layout is preserved.

**Before any drive,** the user confirms each wired line has no other
connection on the radio adapter, as was done for GPIO20 in B.5. "Unused"
in software does not prove the copper is free.

**User decisions, 2026-10-20 15:18–15:20 PDT.** W1 wiring is complete on
both Pis, exactly as in the table, ground on pin 39. On isolation, the user
said: "the lr2021 radio boards are connected, but should only be driving its
DIO8 pin and BUSY pin". DIO8 (GPIO5) and BUSY (GPIO18) are not W1 lines, so
no W1 drive contends with a radio output. Read-only check after wiring: all
16 W1 lines still `unused` inputs at default pulls (GPIO6 pull-up/high, the
rest pull-down/low); pi133 has `lr20xx_module` loaded, pi134 none.
Scope: **Phases 1–2 only**; Phase 3 is deferred. **P5 rejected**: SPI tests
use bit-banged P4 only, with no `config.txt` change or reboot.

**Generator.** A small C tool, `dslcap/tools/pi_pattern.c`, built on each
Pi with its own gcc. It writes BCM2711 `GPCLR0` then `GPSET0` through
`/dev/gpiomem`. These are separate writes, so a step that both clears and
sets lines passes through exactly one intermediate state, `prev & next`,
which fast captures can see. The checker accepts only that state between
two expected states (correction from ds-worker review, 2026-10-20). It only
touches the eight W1 lines, refuses any other mask, and on exit or signal
restores them to input with their default pull-down/pull-up. No new
dependency. Patterns:

- **P1 mapping:** walking-one and Gray-code steps, ≥10 µs each. Checks
  channel identity and inter-line skew on each Pi.
- **P2 transport:** a free-running PRBS state sequence at the CPU's toggle
  rate. The checker compares the sequence of distinct 8-bit states, not
  their timing, so it detects dropped, duplicated or reordered data at any
  sample rate.
- **P3 clocks:** GPCLK0/1/2 on GPIO20/21/6 from PLLD (750 MHz on Pi 4) with
  integer dividers, e.g. 25 MHz and 10 MHz. Checks frequency (ppm, as in
  A2), duty, and edge placement at 250M/500M/1G.
- **P4 SPI:** bit-banged frames on CH n+0..3 with prbs9 payloads on both
  MOSI and MISO, decoded through `sigrok_hla.py --dslogic` and checked by
  `tools/check_pcycle.py`-style exact comparison. CPU-timed, so SCLK is a
  few MHz at most and jittery.
- **P5 optional, hardware SPI1 at 10–25 MHz:** needs
  `dtoverlay=spi1-1cs,cs0_pin=16` in `config.txt` and a reboot. That changes
  shared system configuration, so it requires the owners' approval. MISO is a
  Pi input in this mode and stays undriven.

The two Pis are not synchronized; each Pi's eight channels are checked
independently, and cross-Pi ordering is not a gate.

**W2 (optional)** is the old passive layout on the radio's SPI0/BUSY/DIO
lines, for a final real-traffic check. It needs rewiring and coordination
with the radio owners.

## Phases

**Phase 1: bring-up (C only).** Items 1, 2, 6 plus the model table,
selection, security rule and scan output. Offline: fake driver gains a
U3Pro16 device and a both-attached case; existing Plus tests keep passing
byte for byte. Live gate U1 (no wiring needed): `--scan` reports the
correct model, `speed=super`, `security=n/a`, HDL 0x0e, then a cold FPGA
load with `dslcap_reload_fpga` followed by a clean reopen.

**Phase 2: streaming.** Profile-correct stream mode choice up to
125M×16 / 250M×12 / 500M×6 / 1G×3, including the stream half/quarter rule
(item 3), `--list-modes`, Python integration, `pi_pattern` and its checker,
and docs. If the 500M/1G stream rule can't be derived with confidence,
Phase 2 stops at 250M×12, and 500M/1G move to Phase 3. Offline:
mode-chooser table tests for both profiles and both USB speeds, including
500 kHz rejected on 002a/super and accepted on 0034, and 1G rejected for any
channel ≥ 8. Python validation against recorded `--list-modes` output;
checker negative-path tests with `tools/gen_pp.py` style synthetic input.
Live gates on W1. Output rate = rate × unitsize; unitsize is 2 as soon as any
channel ≥ 8 is enabled.
- U2 mapping: P1 on all 16 channels at 25M (50 MB/s) and 125M (250 MB/s).
- U3 transport: P2, 16 channels at 125M and mask `0x3f3f` at 250M
  (500 MB/s), `--raw-out` for a fixed time; no overflow, exact sample count,
  the checker passes on replay. First measure the target disk's sustained
  write rate; if it is below 500 MB/s, keep 250M captures well inside the
  ~0.5 s ring slack.
- U4 clocks: P3 from pi133 only, GPCLK on GPIO21/20/6 = CH0/CH2/CH5, mask
  `0x0025` (unitsize 1), at 250M, 500M (500 MB/s) and 1G (1 GB/s, ×3
  CH0–7); frequency within the analyzer's stated accuracy and duty
  consistent across rates. pi134 clocks (CH8+) repeat at ≤250M only.
- U5 decode: P4 on both buses at 25M live through `sigrok_hla.py`,
  byte-exact on both.
- U6 (optional): W2 real radio traffic. P5 was rejected on 2026-10-20.

**Phase 3: buffer mode and triggers.** Item 3 and the depth limit at 2 Gibit.
Derive the ADF4360 500M/1G divider and packing from `dsl_fpga_arm` and the
receive path. Capture DSView's register images for representative simple
and serial triggers on the U3Pro16, as `DSVIEW_COMPARISON` did for the
Plus, and add them as separate goldens. Live gates U7–U9 on W1: simple
trigger position at 100M/500M/1G on a P1 step; timeout, upload, cancel and
reopen; serial trigger bit order and an opcode spot check on P4 frames.

**Phase 3 started 2026-10-21 04:05 PDT** ("you can start phase three"),
with W1 rewired and `mcp_radio` stopped on both Pis. It is split into:
- **3a:** untriggered buffer mode, 500M×16 and 1G×8, assigned to ds-worker
  as `dslcap-u3pro16-p3a-20261021`.
- **3b:** simple triggers, after DSView reference images for the U3Pro16.
- **3c:** serial triggers.

**W1 isolation finding, same date.** Since the reboot into `mcp_radio`
(~19:20 PDT on 2026-10-20), GPIO22 and GPIO25 (CH6/CH7, CH14/CH15) read high
with pull-down enabled on both Pis, so something off-Pi drives them. They
were low on all earlier W1 baselines. The W1 idle capture now reads `0xa0a0`
(CH5/CH7/CH13/CH15 high), not `0x2020`; CH6/CH14 read low on the analyzer
although the Pis read GPIO22 high. **Resolved at 04:08 PDT:** the user removed the radio boards from both Pis
(with the Pis still running). GPIO22/25 then read low, the idle capture was
`0x2020` again, and a W1 walk at 25M×16 passed on both groups with pins
restored. All eight W1 lines per Pi are back in use; the planned
`pi_pattern --lanes` was withdrawn from 3a.

Phase 3 can be deferred; Phases 1–2 stand alone. A Plus-attached
regression scan and a short 25M×12 stream before merging are required, once
the user reconnects it.

## Hardware rules

- The analyzer only receives; bring-up, VTH and FPGA load drive no pins.
- Pi drive is limited to the eight W1 lines above, after the user confirms
  wiring. Never drive radio-mapped or claimed lines; GPIO12 especially.
- Before each session on a Pi, check who owns it now (other sessions use
  these Pis) and record `pinctrl get` for the W1 lines. On 2026-10-20
  14:43 PDT `hydra-develop-1a` reported it neither owns nor uses pi133/pi134;
  it left them in kernel-module mode (`lr20xx_module`, `lr2021_pcycle`).
  Other sessions were not asked. After it, restore
  them and read them back.
- Generator runs are bounded (`timeout`) and never overlap a capture
  on the same lines from another owner.
- Raw captures go to local disk outside `/tmp`; the 2026-10-18 reboot
  removed `/tmp/dslcap-*`.

## Open questions for the user

None open. The answers to the three original questions are recorded under
"User decisions" in the W1 section.
