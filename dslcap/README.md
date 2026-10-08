# dslcap

Standalone GPLv3-or-later frontend for a DSLogic Plus PGL12 analyzer
(`2a0e:0034`). Supports standalone finite/continuous logic capture and
bring-up verification. Python integration and remaining signal comparisons
are tracked separately in [PLAN.md](PLAN.md) and [TASKS.md](TASKS.md).

## Build and offline tests

Prerequisites: CMake 3.16+, a C11 compiler, pkg-config, GLib, libusb 1.0,
zlib, Linux/POSIX threads, and the existing DSView 1.3.2 source tree. The build compiles DSView's
library and common C sources directly, without modifying or copying them.
The production binary needs neither Qt nor Python; offline contract tests use Python 3.

```sh
cmake -S dslcap -B /tmp/dslcap-build -DDSVIEW_SRC=/home/wroberts/DSView-1.3.2
cmake --build /tmp/dslcap-build
ctest --test-dir /tmp/dslcap-build --output-on-failure
```

`DSVIEW_SRC` defaults to `/home/wroberts/DSView-1.3.2`. The implementation
uses this version's device-handle representation, logger ownership and
internal read-only `dsl_hdl_version` helper. Other DSView versions need
compatibility review. `BUILD_TESTING=OFF` disables the offline fake-driver
test executable. Its generated `test-firmware` directory contains fake
data for tests only; never use it with a real analyzer.

New single-configuration builds default to `RelWithDebInfo` for optimized
conversion. An explicitly selected build type is preserved. The converter
uses batches of about 16 KiB per ring push. Measure conversion locally with
`/tmp/dslcap-build/dslcap_core_tests --benchmark`; one observed 8-channel
run processed 714 MB/s in memory (not a USB throughput measurement).

The contract test exercises the real frontend and DSView logger with fake
hardware calls. It verifies security failure and missing-pass rejection,
independent security evidence on reopen, demo fallback, last-error and
identity checks, HDL read/version failures, initialization/list/release/exit
failures, and invalid firmware paths before USB initialization. These tests
do not prove real FPGA loading or USB behavior. Capture tests additionally
cover 3,192 independent sample-oracle packet-split/trim cases, parser bounds,
configuration failures/order, samplerate clamping, impossible/unrepresentable
rates, buffer depth, malformed/truncated packets, overflow/device events,
broken pipe, bounded writer drain and driver-watchdog/signal cancellation. Trigger
contracts compile the real capture/control/ring sources with test-only shorter
deadlines. They exercise freed header payloads, packet ordering, cache/deadline
races, forced counts, device/signal/data precedence, and a progressing
consumer that outlives its watchdog margin. Production timing is unchanged.

## Capture

Close DSView before capture. These examples were exercised with the analyzer:

```sh
timeout 40s /tmp/dslcap-build/dslcap --samplerate 25M --channels 0-7 --samples 100001 > /tmp/capture.raw
timeout 40s /tmp/dslcap-build/dslcap --samplerate 25M --channels 0-11 --time 200ms > /tmp/capture12.raw
timeout 40s /tmp/dslcap-build/dslcap --samplerate 20M --channels 0-15 --samples 100005 > /tmp/capture16.raw
timeout 40s /tmp/dslcap-build/dslcap --samplerate 25M --channels 0,3,9,15 --samples 100007 > /tmp/sparse.raw
timeout 40s /tmp/dslcap-build/dslcap --mode buffer --samplerate 100M --channels 0-7 --samples 100001 > /tmp/buffer.raw
timeout 40s /tmp/dslcap-build/dslcap --test-pattern > /tmp/pattern.raw
```

Exactly one of `--samples N`, `--time T`, or `--continuous` is required for
normal capture. Quantity suffixes are `K`/`k`, `M`, and `G`; time suffixes are
`s`, `ms`, and `us` (plain values mean seconds). Time converts to a rounded-up
sample count. Defaults are `--samplerate 25M --channels 0-7 --mode stream
--vth 1.6`. VTH must be finite and within 0–2.5 V. Channels are physical
indices 0–15, with comma-separated indices or ascending ranges; duplicates
and empty lists fail. Select either `stream` or `buffer`; continuous mode
requires streaming and arms a nonzero sample limit with driver loop mode.

Capture stdout starts with `META samplerate: N\n`, followed immediately by
raw samples. Strip that line before loading the binary payload. If all
enabled channels are below 8, each sample is one byte; otherwise it is two
little-endian bytes. Bit `i` always represents physical channel `i`, and
unselected bits are zero. Width is derived from the highest selected channel,
not the enabled count. Normal finite payloads contain exactly the requested samples; forced triggered uploads can be shorter.
Bring-up without capture arguments keeps stdout empty; `--scan` is a
separate verified-device text mode and cannot be combined with capture flags.

Among supported modes, choose the largest valid-channel count that carries
the requested rate and physical channels: stream limits are 20M×16,
25M×12, 50M×6 and 100M×3. All stream modes expose the 16 physical inputs.
Buffer modes expose 16, 8 or 4 physical inputs at 100M, 200M or 400M.
Rates must appear in the pinned profile's supported list and have an exact
hardware divider (including the fast-buffer packing flags). Samplerate
readback and enabled-channel limits are also checked. Thus 50M×8, 60M×2 and
3M×8 fail instead of advertising a wrong timebase. Buffer counts must fit
the available hardware depth after rounding to 1,024-sample boundaries;
dslcap trims this padding. Signal comparisons at fast buffer rates are
recorded in [VALIDATION.md](VALIDATION.md); trigger-specific live checks
are tracked separately.

`--test-pattern` overrides mode, channels, rate and requested limit. It uses
16-channel buffer mode at 100M and the driver's 16,777,216-sample depth.
The observed full capture is `sample[i] = i modulo 65536`; all samples in
one run matched. This supports LSB-first bit order and ascending channel
rotation for this buffer pattern. CH0 software GPIO pulses were also
observed at 8×25M, 12×25M, 16×20M and sparse-channel settings, with the
expected 2/3/5/7 ms sequence plus software timing overhead. These runs do
not replace a DSView comparison or physically exercise the higher inputs.

The USB callback handles cross-data groups with carry across packet
boundaries. Streaming uses a 256 MiB ring; buffer mode allocates enough
for the requested capture and metadata, with checked arithmetic before
arming. CH15 alone at full hardware depth requires slightly over 512 MiB. A writer drains fd 1 with
nonblocking writes; callbacks never wait on downstream output or stop the
library. Main-thread control handles terminal events and driver joins.
Ring-full, FPGA overflow, malformed/truncated finite data, device errors
and output failure return nonzero. FPGA overflow detection can be delayed;
data emitted since the actual overflow is suspect. A normal finite end
requires the expected samples, a complete source group and a data-END event.

SIGINT/SIGTERM request driver shutdown in normal thread context and return
130/143. Broken pipes return 12. Stream output drains for at most two seconds. Buffer output drains while
writes make progress, with `--drain-timeout S` (1..3600 seconds, default 30)
limiting each stall after collection ends. A stalled consumer exits 14;
SIGINT/SIGTERM or a device error cancels the drain. Truncated output is
reported and an earlier higher-priority failure is preserved.
An unconsumed pipe filled the ring at about 10.7 seconds of sample backlog
and exited 9 in 13.8 seconds including startup and bounded drain; immediate
reopen succeeded after SIGINT, broken pipe and ring-full tests.

A watchdog bounds startup at 30 seconds, finite capture at requested duration
plus 30 seconds after the driver-running event, and normal stop/cleanup at
10 seconds. An interrupt gets a five-second grace period in stream mode and 15 seconds
in buffer mode. Buffered drain re-arms its watchdog after each successful
write, so a progressing slow consumer can take longer than two seconds. If inherited
driver polling or joins cannot cancel, the watchdog terminates the process
with a clear error so the OS releases USB handles; reopen must then be
verified. Continuous collection has no overall duration deadline. An
external `timeout` remains useful for hardware test bounds. The application
draining dslcap must continuously drain stderr too.

## Simple buffer triggers

Simple AND triggers are supported in buffer mode. This syntax is implemented
and covered by offline contracts; live validation is tracked by the lead in
[TRIGGER_PLAN.md](TRIGGER_PLAN.md) and [TRIGGER_TASKS.md](TRIGGER_TASKS.md).

```sh
/tmp/dslcap-build/dslcap --mode buffer --samplerate 100M --channels 0-7 \
  --samples 1000000 --trigger 3:f --trigger-pos 10 \
  --trigger-timeout 5s --on-timeout fail > /tmp/trigger.raw
```

`--trigger CH:COND[,CH:COND...]` requires captured physical channels. Conditions
are `r`/`R` (rising), `f`/`F` (falling), `1`/`h` (high), `0`/`l` (low), and
`e` (either edge). Multiple terms must hold at the same sample. Duplicate
channels, stream or continuous triggers, and triggers with `--test-pattern`
are rejected. Trigger channels obey the fast buffer lane limits: 0..15 at
100M, 0..7 at 200M, and 0..3 at 400M.

`--trigger-pos` is an integer percent from 0 to 90, default 10. The driver
arms N rounded up to 1024 samples. The effective pre-trigger position uses
that aligned count, a 64-sample minimum, a 90% channel-depth cap, and rounds
down to a multiple of 64; stderr logs the effective value. Output is trimmed
to N, and timestamps still begin at capture start.

Without `--trigger-timeout`, waiting lasts until a trigger or interruption.
With a timeout, dslcap allows a full nominal 340 ms observation grace and
reconciles an arrived header or cached trigger hit before acting. Status is
a driver cache: this grace has no hard freshness guarantee. Read failures
and the longest interval without an observed count/hit change appear in the
summary; uncertain decisions say `freshness unknown`. A hit near the deadline
may win or lose. Once a fail timeout commits, later packets cannot recover
discarded data. It returns 16 with only the samplerate header.

`--on-timeout upload` requests a forced upload through DSView's existing
`WAIT_UPLOAD` control. Captures can be shorter than N in 1024-sample steps.
An empty or malformed capture fails with 13. A false control return is
reconciled as a possible natural completion; success still needs a header,
END and the exact emitted count. Metadata always uses the returned header's
trigger status, so a forced capture can still report a real trigger.

Triggered stdout has exactly two lines before binary samples:

```text
META samplerate: 100000000
META trigger: 100032
```

Line two contains the trigger's zero-based sample index, or `none` for an
untriggered forced capture. Untriggered requests retain exactly one line.
Consumers must choose one or two lines from the request, never inspect binary
samples for another header. Any nonzero exit marks output incomplete,
including failures after metadata or samples have already been emitted.
Known terminal conditions use signal > device > data > trigger timeout >
drain stall precedence. Serial triggers share this lifecycle and framing, as
described below. Trigger-relative `--t0` remains unsupported and is rejected.

## Serial buffer triggers

`--serial-trigger` selects a serial trigger instead of a simple AND trigger:

```sh
/tmp/dslcap-build/dslcap --mode buffer --samplerate 400M --channels 0-3 \
  --samples 1M --serial-trigger 'start=3:f,stop=3:r,clock=0:r,data=2,value=0x1c35,bits=16' \
  --trigger-pos 10 --trigger-timeout 5s > /tmp/serial.raw
```

All six fields are required exactly once, in any order. Start and stop use the
same conditions as simple triggers; clock accepts only `r`/`R` or `f`/`F`.
Data is a bare physical channel. Bits must be 1..16; value requires a hexadecimal
`0x` or binary `0b` prefix and must fit that width (uppercase prefixes also
work). All four role channels must be captured and fit the physical rate lanes.
Roles may share a channel, as start/stop normally share nSS. Serial and simple
triggers are mutually exclusive; serial capture requires finite buffer mode
and rejects test patterns. Existing trigger-position, timeout/action, drain,
exact two-line META and exit-status rules apply unchanged.

Configuration follows DSView 1.3.2's serial template: global stage selector0
(default UI stage count1); stage0=start/stop, stage1=clock/all-X,
stage2=data-channel marker/all-X, stage3=compare value/all-X. Logic is AND,
non-contiguous, with no inversion; stage1 count1 and stage3 count(bits-1).
The trigger is enabled last. Fixed sixteen-probe strings put the value LSB in
probe0 and unused upper16-bits positions at X. For bits<16 the matching DSView
reference is the bit editor with upper X positions; its hex helper instead
zero-pads all16 positions.

MSB-first serial order is used: the most recently shifted bit is the value
LSB. The asymmetric16-bit value0x1c35 distinguishes bit reversal0xac38,
byte swap0x351c, and both0x38ac. Offline tests pin the parsed value and generated
stage state; they do not prove hardware shift order or constitute a real DSView
register golden. Actual register comparison passed at 100/200/400 MS/s, and
native aligned bit-order validation passed at 100 MS/s on CH0–7. C.3 opcode
validation also passed there for LR2021 opcodes `0x0101` and `0x0002`; see
[OPCODE_VALIDATION.md](OPCODE_VALIDATION.md). Live bit order also passed at
200 MS/s CH0–7 and 400 MS/s CH0–3; see
[SERIAL_HIGHRATE_VALIDATION.md](SERIAL_HIGHRATE_VALIDATION.md).

Treat the serial comparison as N-bit words aligned from the start condition,
not an arbitrary sliding bit window. In real DSView tests, 16-bit `0x1c35`
matched at clocks 17–32 after NSS assertion but missed at clocks 25–40 with
identical trigger registers. An aligned payload word can also match, so this
is not an opcode-only filter. See [SERIAL_INTERMEDIATE_WIDTHS.md](SERIAL_INTERMEDIATE_WIDTHS.md)
for alignment evidence and [SERIAL_ALIGNED_VALIDATION.md](SERIAL_ALIGNED_VALIDATION.md)
for the native bit-order results and their measured scope.
Timestamps remain relative to capture start.

## Bring-up

Save any current capture and close DSView before running dslcap; only one
program can claim the analyzer. In a sandbox, the authorized USB operation
may need to run outside it to access `/dev/bus/usb` and see host processes.

```sh
timeout 30s /tmp/dslcap-build/dslcap --scan -v
timeout 30s /tmp/dslcap-build/dslcap
timeout 30s /tmp/dslcap-build/dslcap --scan --fw-dir /tmp/dslcap-missing-firmware
```

The last command is an intentional failure check and must exit 3. All
diagnostics go to stderr. With `--scan`, stdout contains one verified-device
line after successful activation and cleanup:

```text
2a0e:0034 DSLogic PLus bus=1 address=6 activated security=pass hdl=checked
```

Bus/address vary. The model spelling matches DSView's profile. Without
`--scan`, bring-up emits only stderr diagnostics. `--help` prints usage.
Default driver verbosity is errors only; `-v` enables information and `-vv`
debug messages. `--scan` actively opens the device and verifies it, rather
than merely enumerating it. Zero or multiple matching units fail; device
selection for multiple analyzers is not implemented.

`--fw-dir DIR` defaults to `/usr/local/share/DSView/res` and must contain a
nonempty readable regular file named `DSLogicPlus-pgl12-2.bin`. Directory
names must contain 1–499 bytes because DSView copies them into a fixed
500-byte buffer. Validate this file even if the FPGA is already configured,
so an invalid explicit directory always fails. Set the resource directory
before library initialization, which itself scans attached devices.

Activation checks the actual active USB handle and driver last-error status;
DSView can otherwise return success after falling back to its demo device.
The stderr receiver must observe `Security check pass!` and no
`Security check failed!` for each activation. Success is logged at the
driver's error level, so the default verbosity retains this evidence.

An unconfigured FPGA loads the bitstream; an already configured FPGA is
reused. After initial activation, dslcap releases and reopens the analyzer
to exercise the driver's configured-FPGA HDL check, then explicitly reads
the HDL version and requires `0x0e`. Security must pass again on reopen.
`-v` shows `Configure FPGA using ...` / `FPGA configure done ...` only when
a load occurs. A warm reopen does not prove a fresh bitstream upload.

### Explicit hardware-only reload test

This optional test forcibly reloads the **volatile FPGA** using DSView's
existing upload routine and the installed bitstream. It then releases and
reopens the analyzer, requires a new security-pass log, explicitly checks
HDL `0x0e`, and cleans up. It performs no NVM writes. Close DSView first.

```sh
cmake -S dslcap -B /tmp/dslcap-build -DDSLCAP_BUILD_USB_TESTS=ON
cmake --build /tmp/dslcap-build --target dslcap_reload_fpga
timeout 30s /tmp/dslcap-build/dslcap_reload_fpga --scan -v
```

`DSLCAP_BUILD_USB_TESTS` defaults to `OFF`. This binary reuses the same
guarded frontend, with the reload helper enabled only for that target;
the production `dslcap` binary does not force a reload. It is never
registered with CTest. `--fw-dir` still validates the installed resource;
never point this hardware binary at the generated fake test firmware.
The external timeout bounds DSView's otherwise unbounded FPGA polling
loops. A timeout is a failure and requires checking device state before retry.

The device normally enumerates running its EEPROM FX2 firmware. DSView's
scan then logs `Found a DSLogic device` and skips FX2 firmware upload.
No matching `.fw` recovery image exists in the referenced installation.
No frontend code requests NVM writes. Security activation reads EEPROM
and writes volatile FPGA registers through the existing driver.

| Exit | Meaning |
| --- | --- |
| 0 | Bring-up or complete finite capture succeeded |
| 2 | Invalid CLI arguments |
| 3 | Invalid/unreadable firmware resource |
| 4 | Missing, ambiguous or unlistable device |
| 5 | Initialization, activation, identity or HDL failure |
| 6 | Failed or absent security-pass evidence |
| 7 | Release, cleanup or output failure |
| 8 | Impossible/invalid configuration, wrong readback or excess buffer depth |
| 9 | Output ring full |
| 10 | FPGA overflow (earlier output may be suspect) |
| 11 | Device collection error, detach or USB speed mismatch |
| 12 | Capture output failure or broken pipe |
| 13 | Malformed/truncated data or incomplete finite capture |
| 14 | Output drain timed out; stream truncated |
| 15 | Driver watchdog timed out |
| 16 | No trigger within the configured timeout and nominal grace |
| 130 / 143 | Interrupted by SIGINT / SIGTERM |

After successful initialization, all exit paths call `ds_lib_exit`, then
release the shared logger. Failed initialization skips `ds_lib_exit` because
DSView can return before initializing the mutex that its exit function uses.
These failure paths precede device scanning and thread creation; immediate
process exit reclaims the partial library context. The offline test asserts
that unsafe partial-initialization cleanup is never called.

Device claim/configuration failures leave stdout empty and return nonzero.
Failures after META or data may leave a partial stream, so consumers must
check the exit status. Completed long streaming, hardware overflow, DSView
signal comparisons, SPI traffic and wide Python validation are recorded in
[VALIDATION.md](VALIDATION.md). New buffer-trigger live gates remain tracked
in [TRIGGER_TASKS.md](TRIGGER_TASKS.md).
