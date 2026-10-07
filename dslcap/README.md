# dslcap

Standalone GPLv3-or-later frontend for a DSLogic Plus PGL12 analyzer
(`2a0e:0034`). Phase 0 implements device bring-up; sample capture and the
Python backend remain future work described in [PLAN.md](PLAN.md).

## Build and offline tests

Prerequisites: CMake 3.16+, a C11 compiler, pkg-config, GLib, libusb 1.0,
zlib, and the existing DSView 1.3.2 source tree. The build compiles DSView's
library and common C sources directly, without modifying or copying them.
It needs neither Qt nor Python.

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

The contract test exercises the real frontend and DSView logger with fake
hardware calls. It verifies security failure and missing-pass rejection,
independent security evidence on reopen, demo fallback, last-error and
identity checks, HDL read/version failures, initialization/list/release/exit
failures, and invalid firmware paths before USB initialization. These tests
do not prove real FPGA loading or USB behavior.

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
| 0 | Activation, security, HDL and cleanup succeeded |
| 2 | Invalid CLI arguments |
| 3 | Invalid/unreadable firmware resource |
| 4 | Missing, ambiguous or unlistable device |
| 5 | Initialization, activation, identity or HDL failure |
| 6 | Failed or absent security-pass evidence |
| 7 | Release, cleanup or output failure |

After successful initialization, all exit paths call `ds_lib_exit`, then
release the shared logger. Failed initialization skips `ds_lib_exit` because
DSView can return before initializing the mutex that its exit function uses.
These failure paths precede device scanning and thread creation; immediate
process exit reclaims the partial library context. The offline test asserts
that unsafe partial-initialization cleanup is never called.

Device claim failures should leave stdout empty and return nonzero. Full
capture configuration, streaming, overflow and signal handling are subsequent
phases.
