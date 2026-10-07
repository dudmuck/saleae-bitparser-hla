# dslcap: DSLogic Plus capture front-end for sigrok_hla.py

Status: authorized for implementation, 2026-10-07. Revised 2026-10-06 after PLAN_REVIEW.md. Every
must-fix item was checked against the DSView 1.3.2 source. Goal: live SPI +
HLA decode on a DreamSourceLab DSLogic Plus (`2a0e:0034`) through the existing
`sigrok_hla.py` numpy engine, the same way the Saleae Logic 8 works through
the `saleae-logic-pro` driver.

## Why not upstream sigrok

- Upstream `dreamsourcelab-dslogic` (origin/master 0bc24877, 2025-11-20) only
  knows the DSLogic Plus as PID `0x0020`. This unit is `0x0034`: the newer
  hardware revision with a Pango PGL12 FPGA instead of the Spartan-6. Its
  bitstream is `DSLogicPlus-pgl12-2.bin`.
- The upstream driver is a 2017-era fork of DSView. It uses separate vendor
  requests `0xb0..0xbc`. The FX2 firmware on the PGL12 boards uses the
  generic `CMD_CTL_WR/RD_PRE/RD` (`0xb0/0xb1/0xb2`) instead, with
  sub-commands. Upstream's old firmware can't drive the new FPGA, so the new
  firmware and therefore the new protocol are required.
- The PGL12 boards also need an HDL version check (`DSL_HDL_VERSION 0x0E`), an
  EEPROM-backed security handshake (`dsl_secuCheck`: 8 words from EEPROM
  `0x3C00` are fed back to the FPGA through `SEC_CTRL/SEC_DATA`), and a
  different VTH scale (`CAPS_FEATURE_MAX25_VTH`).

Bringing all of that into `/mnt/foo/libsigrok` would mean rewriting the
driver's command layer. DSView's own `libsigrok4DSL` already handles this
unit, so we drive it directly instead.

## Architecture

```
dslcap (C, links libsigrok4DSL from /home/wroberts/DSView-1.3.2/)
    --samplerate 25M --channels 0-7 [--time 5s | --continuous] [--vth 1.6]
  -> "META samplerate: N\n" + raw samples on stdout, sigrok "-O binary" layout
     (unitsize 1, bit i = channel i)
  | sigrok_hla.py --dslogic ...  (existing run_sigrok_numpy / fast_spi path)
```

`fast_spi.MultiPortDecoder` and the reader thread / HOLD / heap merge code in
`run_sigrok_numpy` stay as they are. Only the command that produces the byte
stream changes, plus the stderr handling (see below).

## dslcap

### Build

`dslcap/CMakeLists.txt` compiles the library sources straight from the DSView
tree (`-DDSVIEW_SRC=/home/wroberts/DSView-1.3.2`, default). No DSView source is copied:

- `libsigrok4DSL/*.c`, `hardware/DSL/*.c`, `hardware/common/*.c`, `input/`,
  `output/`, `hardware/demo/` (matching `libsigrok4DSL_SOURCES` in DSView's
  CMakeLists.txt, minus the Qt app)
- `common/log/xlog.c`, `common/minizip/*`
- deps: glib-2.0, libusb-1.0, zlib. No Python and no Qt. Python is only used
  by DSView's decoders, which avoids the miniconda libpython mixup that broke
  the DSView install.

### Firmware

Only the FPGA bitstream (`DSLogicPlus-pgl12-2.bin`) ships with DSView. No
`DSLogicPlus-pgl12-2.fw` exists in the tree or in `/usr/local/share/DSView/res`.
The FX2 firmware lives in the unit's EEPROM: it enumerates already running it
("USB-based DSL Instrument v2"), so `dsl_check_conf_profile` skips the upload
(`dslogic.c:403-440`). If that EEPROM is ever wiped, this tree can't recover
the device. Don't experiment with NVM writes.

### CLI

| option | meaning |
|---|---|
| `--scan` | list devices and exit (bring-up check) |
| `--samplerate R` | e.g. `25M`. Read back after configuring; any mismatch is fatal |
| `--channels LIST` | physical channels, e.g. `0-7` or `0,1,2,3`. Any of the 16 inputs; phase 1 output is unitsize 1, so channels 0..7 |
| `--time T` / `--samples N` | finite capture → `SR_CONF_LIMIT_SAMPLES` |
| `--continuous` | unbounded stream → `SR_CONF_LOOP_MODE` (stream mode only, non-zero limit) |
| `--mode stream\|buffer` | `SR_CONF_OPERATION_MODE`, default stream |
| `--vth V` | input threshold, `SR_CONF_VTH` |
| `--fw-dir DIR` | default `/usr/local/share/DSView/res` |
| `--test-pattern` | `LO_OP_INTEST`. **Overrides mode, rate and limit** (see Phase 1) |
| `-v` / `-vv` | raise `ds_log_level`. Default: errors only |

**Channel mode selection.** All four stream modes have `num = 16` probes. The
12/6/3 limit is `vld_num` (`SR_CONF_VLD_CH_NUM`, `dslogic.c:726`), and only
DSView's dialog enforces it, not the driver. Rule: among the modes for
`--mode`, pick the one with the largest `vld_num` whose `max_samplerate` ≥
`--samplerate` and whose `vld_num` ≥ the number of enabled channels. The
DSLogic Plus stream modes are 20M×16, 25M×12, 50M×6 and 100M×3; the buffer
modes are 100M×16, 200M×8 and 400M×4. Dual SPI (8 channels) therefore
streams at up to 25 MSa/s, the same target as SIGROK_HLA_REALTIME_PLAN.md.

### Data path

1. **Open and configure.**
   1. Set up logging: `xlog_new2(0)` + `xlog_add_receiver` → `ds_log_set_context`.
      The receiver forwards lines to stderr at the `-v` level. It also watches
      for "Security check failed!" and "Security check pass!". Both are
      logged with `sr_err`, so they arrive even at the errors-only default.
   2. Run `ds_lib_init` → `ds_set_firmware_resource_dir` → `ds_get_device_list`
      → `ds_active_device`. This does the FPGA load, the HDL version check and
      the security check. **`ds_active_device` returns `SR_OK` even when the
      security check fails** (`dsl.c:1958-1964`), so exit non-zero if the log
      receiver saw "failed", or never saw "pass".
   3. Configure **in this order**, because the setters reset other state:
      operation mode (resets `ch_mode`, `dslogic.c:980-1003`) → channel mode
      (`dsl_adjust_probes` re-enables all 16 probes, `dsl.c:176-178`) →
      samplerate → limit samples → loop mode (only if stream, mirroring
      `sigsession.cpp:582-583`) → VTH → per-channel enables. **Explicitly
      disable** every probe that wasn't requested.
   4. Check before starting:
      - read back `SR_CONF_SAMPLERATE`. `dsl_adjust_samplerate`
        (`dsl.c:218-222`) clamps out-of-range rates silently, so exit
        non-zero if it differs from the request;
      - the enabled count must be ≤ `SR_CONF_VLD_CH_NUM`. Otherwise the
        device streams more channels than USB 2.0 carries and overflows.
   5. Write `META samplerate: N\n` to stdout. `run_sigrok_numpy` already strips
      it, and it lets the Python side cross-check the rate. Then call
      `ds_start_collect`.
2. **Datafeed callback.** It runs on the same thread as
   `libusb_handle_events` (`receive_data`, `dslogic.c:1336`), so any delay
   here directly stalls USB.
   - Transpose only `SR_DF_LOGIC` packets with `format == LA_CROSS_DATA` and
     `status == SR_PKT_OK`. Act on `SR_DF_OVERFLOW` and `SR_DF_END`. Ignore
     header, `SR_DF_TRIGGER` and other types (log them at `-v`).
   - Cross layout: blocks of 8 bytes, cycling through the enabled channels in
     ascending order. Each block holds 64 consecutive samples of one channel
     (`DSLOGIC_ATOMIC_*`, `dsl.h:83-86`; `LogicSnapshot::append_cross_payload`).
     Transpose each group of `n_enabled × 8` bytes into 64 output bytes,
     placing enabled channel k at bit `phys(k)`.
   - Packets aren't aligned to block groups when `n_enabled × 8` doesn't
     divide the transfer size (e.g. 12 ch = 96 B), so carry the leftover bytes
     over to the next packet.
3. **Ring buffer.** Copy the transposed data into a 256 MiB ring and drain it
   to fd 1 from a writer thread. That's about 10 s of backlog at 8 ch × 25 MHz
   (25 MB/s out), or about 5 s at 9-12 ch with unitsize 2 (50 MB/s). If the
   ring fills, report it on stderr and exit non-zero instead of stalling USB.
4. **End of capture / errors.** Exit codes are distinct per cause:
   - `SR_DF_OVERFLOW` (FPGA overflow, `report_overflow`, `dslogic.c:1325`):
     `ds_stop_collect`, drain, exit non-zero. In stream mode it's only polled
     after `MAX_EMPTY_POLL` empty polls (`dslogic.c:1378-1393`), so it arrives
     late. Data written since the real overflow is suspect, and the message
     says so. Don't rely on `logic.data_error`, which is hardwired to 0
     (`dsl.c:2346`).
   - `DS_EV_COLLECT_TASK_END_BY_ERROR`, `END_BY_DETACHED` and
     `DEVICE_SPEED_NOT_MATCH` (`lib_main.c:820-827`): clear message, exit
     non-zero.
   - `SR_DF_END` / `DS_EV_COLLECT_TASK_END`: drain, exit 0.
   - SIGINT/SIGTERM/SIGPIPE: `ds_stop_collect`, drain, `ds_lib_exit`.
   - Finite captures: `actual_samples` is rounded up to `SAMPLES_ALIGN`
     (`dslogic.c:1436`) and the last packet is cut to `actual_bytes`, not N
     (`dsl.c:2398-2401`). dslcap trims the output to exactly N samples.

## sigrok_hla.py changes

- `--dslogic` backend flag (plus `--dslcap PATH`, default `dslcap` on PATH,
  and `--vth`). It reuses `-C` names, `--spi`, `--samplerate`, `--time`,
  `--continuous`, `--int-pin` and `--extra-pin`.
- Split command construction out of `run_sigrok_numpy` (pass `cmd` in) and
  add `build_dslcap_cmd(args, spi_ports)`. Its docstring must say that the
  dslcap `--channels` list is *derived* from the bits named in
  `-C`/`--spi`/pins, not passed through from sigrok_hla.py's
  `-C/--channels`, which is a name mapping.
- **stderr.** `run_sigrok_numpy` currently opens the child with
  `stderr=subprocess.PIPE` and reads it only after exit (`sigrok_hla.py:551`).
  Once about 64 KiB of logs fill that pipe, the child blocks inside a log call
  on its USB thread, the FPGA overflows, and the data is corrupt. Drain
  stderr in a thread (prefix the lines, print them live) for `--dslogic`, and
  for sigrok-cli too. dslcap also defaults to errors-only logging.
- If the META samplerate disagrees with `--samplerate`, warn and use the META
  value. dslcap should already have exited in that case, so this is a
  backstop.
- `--engine srd` with `--dslogic`: optionally pipe through
  `sigrok-cli -i /dev/stdin -I binary:numchannels={8|16}:samplerate=R -P spi...`
  (numchannels derived from dslcap's output unitsize), which keeps the
  reference decoder available for cross-checks. Low priority.
- Update the module docstring examples and sigrok_hla_readme.md.

## Phases

Execution authorization (2026-10-07): the coordinator assigns implementation
to the existing `dslcap_worker` and independently reviews its changes. Continue
through the plan without step-by-step operator approval; USB logic-analyzer
use is authorized. Commit reviewed work as appropriate. Stop for a blocker or
a new dependency requiring a revised plan and operator review. `pi133` and
`pi134` are available over SSH to generate GPIO signals; request operator
wiring when needed. Do not write device NVM. Assignment and validation state
are recorded in `TASKS.md`.

0. **Bring-up.** Build dslcap. Check that `dslcap --scan` finds `2a0e:0034`,
   skips the FX2 upload (firmware already in EEPROM), loads
   `DSLogicPlus-pgl12-2.bin`, passes the HDL version check, and that the log
   receiver caught "Security check pass!". A forced failure (e.g. a wrong
   `--fw-dir`) must produce a non-zero exit. DSView must not be running,
   since it claims the device.
1. **Correct samples.**
   - `--test-pattern`: `LO_OP_INTEST` forces `ch_mode = intest_channel`
     (`DSL_BUFFER100x16` for 0x0034), **buffer** mode, 100 MHz and
     `hw_depth / 16` samples, overriding `--samplerate`/`--samples`
     (`dslogic.c:1013-1031`). It therefore settles only the bit order within
     a block and the channel rotation start for 16-channel buffer data.
     Nothing in DSView checks the pattern, so what it should look like is
     worked out from the first capture.
   - Stream path: finite captures of a known signal (a PWM/clock and a known
     SPI burst from a Nucleo), compared against a DSView capture of the same
     signal, at:
     - **8 ch @ 25M** (64 B groups, always aligned)
     - **12 ch @ 25M** (96 B groups: exercises the partial-group carry)
     - **16 ch @ 20M**
     - a non-contiguous channel set (e.g. 0,3,9,15 → phys bit mapping)
   - Deliberately request an impossible combination (50M on 8 ch) and check
     that dslcap refuses it instead of clamping.
2. **Streaming robustness.** `--continuous` for 10+ minutes at 25M×8 with
   `| pv > /dev/null`. Then a deliberately slow consumer: confirm that the
   ring-full exit happens at about 10 s of backlog. Check that
   `SR_DF_OVERFLOW` is reported (e.g. by forcing more channels than
   `vld_num` with a debug flag). Check Ctrl-C and SIGPIPE shutdown, and that
   the device can be reopened right away.
3. **sigrok_hla.py integration.** Run a dual-SPI LR1110/LR2021 capture
   through `--dslogic` and diff the HLA output against the same traffic on
   the Saleae Logic 8 (or a Logic 2 run). Run at `-vv` for a long capture to
   prove the stderr drain prevents the pipe stall.
4. **Optional.**
   - more than 8 channels: unitsize 2 output plus uint16 support in
     `fast_spi` (`np.frombuffer(..., '<u2')`; the shifts already work)
   - buffer-mode bursts at 100-400 MSa/s, captured then decoded
   - triggers (`ds_trigger_*`)

## Risks / open questions

- **Cross-data details** (rotation start and bit order for stream data, the
  first packet after a trigger position): the test pattern only covers
  16-channel buffer mode, so the stream cases rest on the Phase 1 DSView
  comparisons.
- **Silent driver behavior** the plan has to guard against itself:
  - samplerate clamping (readback check)
  - `vld_num` not enforced (enabled-count check)
  - security failure returning `SR_OK` (log watch)
  - overflow reported late and not stopping capture (`SR_DF_OVERFLOW` handler)
- **stderr backpressure** stalling USB: errors-only logging plus the live
  drain in sigrok_hla.py.
- **Hidden app-side setup:** libsigrok4DSL may expect calls that DSView's
  `DeviceAgent`/`SigSession` make before `ds_start_collect` (e.g.
  `ds_set_user_data_dir`, `SR_CONF_STREAM`, RLE/filter defaults). Mirror
  DSView's call order if anything misbehaves. `DSView/pv/sigsession.cpp` is
  the reference.
- **No FX2 firmware file:** the device depends on its EEPROM firmware (see
  Firmware).
- **License:** libsigrok4DSL is GPLv3. dslcap is a separate GPLv3 program
  invoked as a subprocess, so sigrok_hla.py's licensing is unaffected.
- **USB 2.0 ceiling:** 16×20M and 12×25M are the stream limits. Higher rates
  need `--mode buffer` (256 Mbit on board, finite).
