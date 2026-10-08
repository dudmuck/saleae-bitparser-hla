# dslcap: triggered buffer-mode capture

Implementation status, 2026-10-07: **Phase B complete**, independently
reviewed and hardware gates B.1–B.6 passed. See
[TRIGGER_VALIDATION.md](TRIGGER_VALIDATION.md) for evidence and retained
limitations. A2 continuous-clock validation also passed, with independent
review and restored wiring; see [A2_VALIDATION.md](A2_VALIDATION.md).
Phase C serial-trigger implementation is now present in C and Python with
offline coverage and independent review PASS; see
[SERIAL_REVIEW.md](SERIAL_REVIEW.md). Real DSView serial register comparison
passed at100M/200M/400M. Native aligned bit-order validation subsequently
**passed at 100 MS/s, CH0–7**; opcode and higher-rate live order gates remain
open. The original crossed-word patterns timed out despite passing NSS controls; see
[SERIAL_VALIDATION.md](SERIAL_VALIDATION.md) and
[TRIGGER_TASKS.md](TRIGGER_TASKS.md). A subsequent real DSView live test also
failed to trigger serially while its NSS control passed; see
[DSVIEW_SERIAL_RESPONSE.md](DSVIEW_SERIAL_RESPONSE.md). A width/clock follow-up
found natural DSView hits for 1-bit and 8-bit matches at requested 8 MHz, but
the 16-bit 0x1c35 match still missed at measured 1 MHz and 100 kHz, with passing
NSS controls. The 8-bit hit lands exactly on the final matching clock edge.
See [SERIAL_WIDTH_CLOCK.md](SERIAL_WIDTH_CLOCK.md). This narrows the diagnosis;
it does not close the original 16-bit bit-order/opcode gates or establish a
general width limit. A subsequent intermediate-width sweep and constant-register
16-bit alignment test produced aligned hit / crossed-word miss / aligned hit.
This demonstrates working 16-bit comparison in DSView and supports word-boundary
matching on this setup; the earlier arbitrary sliding-window assumption was
incorrect. See [SERIAL_INTERMEDIATE_WIDTHS.md](SERIAL_INTERMEDIATE_WIDTHS.md).
Native C.2 then confirmed the intended `1c35` order twice on aligned carriers
and rejected all three reversed variants. See
[SERIAL_ALIGNED_VALIDATION.md](SERIAL_ALIGNED_VALIDATION.md).
Next: C.3 opcode validation. The full FPGA mechanism and untested
configurations, including higher-rate live bit order, remain unproven.
Optional trigger-relative
timestamps remain unimplemented.

Plan review history: proposed 2026-10-07, revised the same day for review
`dslcap-trigger-plan-review-20261007`: all seven findings were accepted and
checked against the source. They are the capture state machine, the
deterministic META header, the buffer drain policy, the forced-upload
length, the Python channel union, C lifecycle tests and validation
precision. A second review round (message `lead-revised-plan-review`,
same task) was also accepted. It covers:
- status-cache staleness and the deadline race
- callback-ordered publication of the trigger record
- partial output after emission
- one exit/watchdog precedence table
- serial clock edges and bit order
- the 1024-sample `SAMPLES_ALIGN` granularity
- `--drain-timeout` rules
- an accurate statement of the user's approval rule

A final readiness check (message `lead-readiness-final`) judged the
design ready to implement after two corrections, both applied:
- an abort is irreversible, so reconciliation happens before
  `ds_stop_collect`, and success after an abort needs header + END + the
  exact count
- `G` is a nominal grace window with no hard timing guarantee, and
  freshness is reported, never assumed

Goal: short, high-rate captures (100/200/400 MSa/s) that start around a
hardware trigger, decoded by the same `sigrok_hla.py --dslogic` path as
streaming. Streaming stays the default and is unchanged.

Sources: DSView 1.3.2 `libsigrok4DSL/trigger.c`, `hardware/DSL/dsl.c`
(`dsl_fpga_arm`, `receive_header`), `hardware/DSL/dslogic.c`, the app's
`pv/dock/triggerdock.cpp`, `pv/view/logicsignal.cpp`, `pv/sigsession.cpp`,
and the DSView User Guide v1.3.0 (`~/Downloads/ds_plus.txt`, §2.3.1, §2.4,
§2.5.1).

## Why buffer mode needs a trigger

Buffer mode samples into the 256 Mibit on-board DRAM, then uploads it over
USB. That's 268,435,456 bits, divided by the number of enabled channels
(`dsl_channel_depth`). The window is therefore short and has to be placed
around the event:

| mode | channels usable | enabled | samples/ch | window |
|---|---|---|---|---|
| 100M×16 | 0-15 | 12 (both radios, current wiring) | 22.4 M | 224 ms |
| 100M×16 | 0-15 | 8 | 33.5 M | 335 ms |
| 200M×8 | **0-7 only** | 6 (pi133 SPI + BUSY + DIO8) | 44.7 M | 224 ms |
| 200M×8 | 0-7 only | 8 | 33.5 M | 168 ms |
| 400M×4 | **0-3 only** | 4 (pi133 SPI) | 67.1 M | 168 ms |

SPI at 8-10 MHz gets 2.5-3 samples per bit at 25M streaming, which is the
marginal case in SPI_DECODE_MARGINAL_REPORT.md. 100M gives 10-12 samples per
bit and can still capture both radios at once. 200M/400M use the half/quarter
packing modes, and those only reach physical channels 0-7 and 0-3
(`dsl_choose_mode`'s `mode->num` check). With the current wiring, pi134
(CH8-13) is only reachable at 100M.

## How DSView drives the trigger (verified in source)

**State** lives in a global `struct ds_trigger`, configured with the public
`ds_trigger_*` API from `trigger.c`. `ds_trigger_reset()` sets it back to
disabled, SIMPLE, pos 0, and all 'X'.

**Simple trigger**, as in the app's `LogicSignal::commit_trig` and
`TriggerDock::commit_trigger`:
- `ds_trigger_set_en(1)`, `ds_trigger_set_mode(SIMPLE_TRIGGER)`,
  `ds_trigger_set_pos(pct)`
- per probe: `ds_trigger_probe_set(index, c, 'X')`, where c is `'R'` rising,
  `'F'` falling, `'1'` high, `'0'` low, `'C'` either edge, or `'X'` don't
  care. This writes the special row `TriggerStages` (16).
- multiple probes are ANDed and must all hold at the same sample (guide
  §2.4.2).

**Advanced triggers** (buffer mode only, guide §2.4.3):
- *Stage trigger:* up to 16 sequential stages, each with two condition
  sets, AND/OR, invert and a count.
- *Serial trigger* (`SERIAL_TRIGGER`, `STRIG_MODE_BIT`): stage 0 is start
  (trigger0) and stop (trigger1). Stage 1 is the clock edge. Stage 2 marks
  the data channel. Stage 3 (`STriggerDataStage`) holds the value. The
  counts are `stage1 = 1` and `stage3 = bits - 1`, with at most 16 bits.
  After start, data is shifted in on each clock edge. Live DSView tests
  support comparison on complete `bits`-wide word boundaries counted from
  start, not every sliding window. Stop clears the serial state. See
  SERIAL_INTERMEDIATE_WIDTHS.md for the tested scope.
- Value strings are `"X X ... X"`, highest channel first, every second
  character (`ds_trigger_stage_set_value`).

**Arming:** `dsl_fpga_arm` (`dsl.c:1060-1205`) builds `struct DSL_setting`
from that global state at `ds_start_collect` time:
- `TRIG_EN_BIT` and `STRIG_MODE_BIT` go in `setting.mode`.
- Trigger position (`dsl.c:1100-1106`), in order:
  1. `v = max((uint32_t)(pos/100.0 × limit_samples), 64)`
  2. `v = min(v, channel_depth × 90/100)` in buffer mode
     (`DS_MAX_TRIG_PERCENT`), or `× 10/100` in stream mode
  3. `tpos_l = v & DSLOGIC_ATOMIC_MASK` (`0xFFFF << 6`) and
     `tpos_h = v >> 16`, so the effective position is rounded **down to a
     multiple of 64**.

  `limit_samples` here is dslcap's aligned-up arm count, not the user's N.
- At 200M/400M the 16-bit trigger masks are replicated across the
  half/quarter lanes (`half_trig`/`qutr_trig`), so trigger channels have to
  stay within 0-7 / 0-3.
- The image is sent as a bulk OUT on EP2 (`dsl.c:1243`).

**Waiting and the trigger report:**
- `dsl_start_transfers` submits a header transfer on EP6 with **no
  timeout** (`dsl.c:2514-2517`), so the driver waits for a trigger forever.
- **The header arrives when capture ends, not when the trigger fires.** It
  carries `remain_cnt`, so the FPGA sends it once sampling is complete and
  the upload is about to start.
- When it completes, `receive_header` (`dsl.c:2441-2480`):
  - sets the status to `DSL_ERROR`, then checks `TRIG_CHECKID` and the
    length
  - in buffer mode, recomputes `actual_samples = (limit_samples -
    remain_cnt) & ~SAMPLES_ALIGN`
  - forwards an `SR_DF_TRIGGER` packet with payload `struct ds_trigger_pos
    {check_id, real_pos, ram_saddr, remain_cnt_l/h, status}`, then moves to
    `DSL_DATA`
  - the payload points **into `transfer->buffer`**, which
    `free_transfer(transfer, 1)` releases right after the callback returns
    (`dsl.c:2483`). Consumers must copy it during the callback and never
    keep the pointer.
- `SAMPLES_ALIGN` is **1023** (`libsigrok.h:105`), so `actual_samples` is a
  multiple of **1024** samples. That's separate from the 64-sample
  (`DSLOGIC_ATOMIC`) granularity of the trigger position.
- **If `remain_cnt >= limit_samples`, no `SR_DF_TRIGGER` is forwarded at
  all** and the status is left at `DSL_ERROR`.
- `status & 1` means triggered. `real_pos` is the trigger's sample index in
  the uploaded data: DSView uses it directly as the trigger cursor index
  (`sigsession.cpp:1166`, `view.cpp:527`).
- The logic data packets follow, so the trigger report arrives before the
  first sample, but after the whole post-trigger window has been sampled.

**Progress while waiting:** while `status == DSL_START` the driver
refreshes `devc->mstatus` over USB, but **only after `MAX_EMPTY_POLL` (16)
consecutive empty polls** of its `receive_data` source. That source runs
on a `dsl_get_timeout(sdi)` period (`dslogic.c:1366-1376, 1498`).
`ds_get_actived_device_status(&st, TRUE)` (`dsl_dev_status_get`,
`dsl.c:2043`) just copies that cache; `prg=TRUE` even bypasses
`mstatus_valid`. It gives `trig_hit & 1` (the trigger has fired;
post-trigger sampling is under way) and `captured_cnt`
(`sigsession.cpp:801-830`).

This is the earliest trigger-hit signal available, but it's stale by a
**nominal** 17 poll periods. That's an estimate, not a bound: host
scheduling and failed USB status reads can stretch it, and dslcap can't
force a refresh through the public API. A fresh read still couldn't atomically close the hardware race
against a later forced stop: that would need a driver/control change,
which is out of scope.

**Stopping before the trigger** has two driver paths:
- `ds_stop_collect` sets `abort` and writes `bmFORCE_RDY`
  (`dsl.c:2015-2018`). `receive_header` ignores the header when `abort` is
  set, so no data comes back.
- DSView's "Upload captured data" works differently. It reads the config
  key `SR_CONF_WAIT_UPLOAD`, which has a side effect: if
  `buf_options == SR_BUF_UPLOAD` (the default) and the status is still
  `DSL_START`, it sets `DSL_ABORT` and writes `bmFORCE_STOP`
  (`dslogic.c:686-694`). The FPGA then reports and uploads what it has. The
  app only does this for single LOGIC captures (`sigsession.cpp:755-759`).
  It gives an "auto trigger": an untriggered capture after a timeout.
  - The `DSL_START` guard is **still true during post-trigger sampling**,
    so this read would also truncate a real triggered capture. dslcap must
    never issue it after `trig_hit` has been observed. Because of cache
    staleness it can still race an unobserved hit; see the deadline rule
    under Capture.
  - It returns FALSE when the status is no longer `DSL_START`, which
    includes a valid header/upload that won the race. FALSE is not by
    itself an error.

## dslcap changes

### CLI

| option | meaning |
|---|---|
| `--trigger CH:COND[,CH:COND...]` | simple trigger, ANDed. COND: `r` rise, `f` fall, `1`/`h` high, `0`/`l` low, `e` either edge |
| `--trigger-pos PCT` | pre-trigger share of the capture, 0-90, default 10 |
| `--trigger-timeout T` | stop waiting after T. Default: wait until Ctrl-C |
| `--on-timeout fail\|upload` | `fail` (default): exit 16 with no samples. `upload`: force-stop and emit whatever was captured (`SR_CONF_WAIT_UPLOAD`); length rule under Capture |
| `--serial-trigger start=CH:COND,stop=CH:COND,clock=CH:EDGE,data=CH,value=V,bits=N` | Phase C. EDGE is `r` or `f` only. SPI example: `start=3:f,stop=3:r,clock=0:r,data=2,value=0x8a,bits=8` |
| `--drain-timeout S` | buffer mode only: exit 14 if the consumer makes no write progress for S seconds after END. 1..3600, default 30 |

Validation rules, checked before USB init where possible:
- trigger options require `--mode buffer` (stream triggering is deferred;
  see Later)
- every trigger channel must be in `--channels`. This keeps triggers inside
  the half/quarter lane limits and lets the output be checked.
- `--trigger-pos` must be 0..90
- the existing buffer-depth check stays
- `--test-pattern` rejects trigger options
- `--trigger` and `--serial-trigger` are mutually exclusive
- `--trigger-pos`, `--trigger-timeout` and `--on-timeout` without a trigger
  are usage errors (exit 2), not silently ignored
- a channel may appear only once in `--trigger`; duplicates are rejected,
  even with the same condition
- serial:
  - `bits` must be 1..16, and `value` (hex `0x` or binary `0b`) must fit in
    `bits`
  - `start` and `stop` take a COND
  - **`clock` takes only an edge, `r` or `f`.** Levels (`1`/`0`) and
    either-edge (`e`) are rejected, because DSView's template and the guide
    only define clock *edges*. Widen this only after a golden-image (B.1)
    and bench check of the other forms.
  - `data` is a bare channel
  - all of them must be captured channels
  - **Bit order (assumed, verified in Phase C):** MSB-first, so the most
    recently shifted bit is the value's LSB. That's the reading of "Data
    Value of most right Data Bits" (guide §2.4.3). Phase C pins it with the
    asymmetric 16-bit golden value **`0x1c35`**. Its bit-reverse (`0xac38`),
    byte-swap (`0x351c`) and both (`0x38ac`) are all distinct, so a wrong
    order can't also match. (`0x8a51`, proposed earlier, is a bit
    palindrome and can't detect reversal.)
- `--drain-timeout` must be 1..3600 and is only valid with `--mode buffer`
  (usage error otherwise). Stream mode keeps its fixed 2 s drain.
- trigger strings are parsed into a fixed 16-entry table, never passed to
  `ds_trigger_stage_set_value`'s pointer arithmetic unchecked

### Configuration (`driver_config.c`)

`dsl_configure` currently calls `ds_trigger_reset()` last, which would wipe
any trigger. New order:

1. Run the existing sequence through the readbacks and the RLE-off set.
2. `ds_trigger_reset()`.
3. If a trigger is requested: `ds_trigger_set_pos` → `ds_trigger_set_mode` →
   per-probe `ds_trigger_probe_set` (simple) or the stage calls in DSView's
   `commit_trigger` order (serial) → `ds_trigger_set_en(1)` last.
4. For `--on-timeout upload`, set `SR_CONF_BUFFER_OPTIONS = SR_BUF_UPLOAD`
   explicitly instead of relying on the default.

Log the requested position next to the source-exact effective value
(max-64 floor → 90%-of-depth cap → rounded down to a multiple of 64; see
Arming), so a clamped or rounded position is visible. Compute it with the
same integer types as `dsl_fpga_arm`, and unit-test it against that
formula.

### Capture (`capture.c`)

**Threads and ownership.**
- **Driver callback thread** (the libusb event thread, which runs
  `receive()`):
  - owns the converter and everything that must happen *in packet order*
  - on `SR_DF_TRIGGER`, synchronously and before returning:
    1. copy `struct ds_trigger_pos` out of the payload (never keep the
       pointer; the driver frees it right after)
    2. validate it (below)
    3. compute `emitted_count` and set `converter.limit`
    4. push `META trigger:` line 2 into the ring
    5. publish the facts (`header_seen`, `trig_status`, `real_pos`,
       `aligned_actual`) as atomics with release ordering
  - only then is `SR_DF_LOGIC` accepted. LOGIC before the header in
    trigger mode is a `DSL_DATA_ERROR`.
  - conversion of LOGIC stays on this thread, as today
- **Main thread:** owns lifecycle only (state transitions, status polling,
  deadlines, `WAIT_UPLOAD`, `ds_stop_collect`, final exit code). It reads
  the callback's facts with acquire ordering and never touches the
  converter or writes META.

**State machine.** States only move forward:

```
ARMING ──RUNNING──▶ WAITING ──trig_hit seen──▶ POST_TRIGGER ──header──▶ UPLOAD ──SR_DF_END──▶ DRAIN ──▶ done
                       │  └──────────────── header (hit not polled) ────────▲
                       └──deadline + grace, no hit, no header──▶ FORCED ──header──┘   (upload mode)
```

The header is published by the callback. The main thread moves to UPLOAD
as soon as it sees `header_seen`, whatever state it was in.

**Status polling and staleness.**
- Poll `ds_get_actived_device_status(&st, TRUE)` about every 50 ms in
  WAITING, POST_TRIGGER and FORCED, at any verbosity. `-v` only controls
  printing.
- Track a **change observation**: the host time at which the cached
  `captured_cnt` (or `trig_hit`) was first seen to change. It's a hint that
  a refresh happened, *not* a hardware refresh timestamp. An unchanged
  value doesn't prove that no refresh happened.
- **Grace window** `G` = 17 × the driver poll period (computed with
  `dsl_get_timeout`'s formula for the configured mode), with a 250 ms
  floor. **It's a nominal observation window, not a staleness bound.**
  There's no hard guarantee on how old a status value is.
- **Freshness is reported, never assumed.** Count status-read failures
  and the longest interval without an observed change. Both go in the `-v`
  log and the final summary. If reads failed or nothing changed during the
  grace window, the timeout decision is logged as "freshness unknown".

**Deadline rule** (best effort, with no hard timing guarantee):
1. At `T = --trigger-timeout`, don't act yet. Keep polling until `T + G`
   or until a change observation after `T` (whichever comes first).
2. **Reconcile before committing anything irreversible:**
   - a published header means acquisition ended and upload is starting.
     Follow it (UPLOAD/DRAIN), and META reflects the header's real
     `status & 1`.
   - `trig_hit` seen means the trigger wins (POST_TRIGGER), even after
     `T`.
3. Only otherwise apply `--on-timeout`. From here a stop is final: see
   "Abort is irreversible".

What's promised: the decision is made only after the grace window and a
final reconcile, and the outcome is always **self-consistent**: META,
length and exit code agree with what the hardware actually delivered. A
hit well before `T` is *expected* to be seen, but isn't guaranteed (a
stretched refresh or failed reads can hide it). A hit near `T` may go
either way. Hardware-event-time semantics would need a driver change;
that's out of scope and listed under Risks.

**Abort is irreversible.**
- `ds_stop_collect` sets `abort` and writes `bmFORCE_RDY`
  (`dsl.c:2015-2017`). From then on `receive_transfer` forces `DSL_STOP`
  for every transfer (`dsl.c:2319-2320`) and forwards data only in
  `DSL_DATA` (`dsl.c:2336`). A header that races the stop **can't**
  bring back the discarded samples.
- After an abort, dslcap reports success only if it holds a valid header,
  a valid END **and** exactly the expected `emitted_count` of samples.
  Anything less is a nonzero aborted/partial outcome: 16 for a `fail`
  timeout, 128+sig for cancel, with the partial-output rules applying if
  line 2 was already pushed.
- dslcap never waits for an upload that an abort has cancelled. After
  abort, only the FORCED/cancel row's bounded wait for END applies.
- END without a valid header is always an error (13), as before.
- This is distinct from `SR_CONF_WAIT_UPLOAD`. That sets `DSL_ABORT`
  status without the `abort` flag, so `receive_header` still moves to
  `DSL_DATA` and the forced upload delivers data.

**Timeout outcomes:**
- `fail`: the reconcile (deadline rule step 2) happens **before** the stop.
  Then `ds_stop_collect` (abort + `bmFORCE_RDY`), a bounded wait for END,
  and exit `DSL_TRIGGER_TIMEOUT = 16`, unless the strict
  header + END + exact-count test above proves completion. A header that
  turns up after the stop doesn't make the capture complete.
- `upload`: read `SR_CONF_WAIT_UPLOAD`.
  - TRUE: FORCED.
  - **FALSE:** the device had already left `DSL_START`, so no force was
    issued and no abort set. Recheck the facts for up to `G`. A valid
    header means a natural completion racing the deadline, so follow it.
    Success still needs END and the exact `emitted_count`. No header means
    `DSL_DEVICE_ERROR` (11), "device left DSL_START without a header".
    END without a header is 13.
  - FORCED with no header before its bound, or END without a header:
    `DSL_DATA_ERROR` (13), "forced upload returned no capture".

**Exit precedence and bounds**, in one table. When several conditions are
known as the main thread finalizes, the highest row wins: **signal >
device error > data error > trigger timeout > drain stall > watchdog**.
Callback-recorded errors keep today's first-wins CAS among themselves.

| state | soft bound (main thread acts) | exit on soft bound | hard bound (control watchdog, `_exit` 15) |
|---|---|---|---|
| WAITING | `T + G` (none without `--trigger-timeout`) | per timeout outcome (16 / continue) | soft + 30 s (none without T) |
| POST_TRIGGER | `(1 - pos) × N / rate + G + 5 s` | 13 "trigger hit but no header" | soft + 10 s |
| FORCED | 10 s for a header or END | 13 | soft + 10 s |
| UPLOAD | `actual_bytes / 4 MB/s + 10 s` | 13 "upload incomplete" | soft + 10 s |
| DRAIN (buffer) | `--drain-timeout` without write progress | 14 (partial output) | re-armed on each write progress: drain timeout + 10 s |
| any, on signal | abort, then up to 10 s for END and drain | 128+sig (partial output) | signal deadline 15 s (> 10 s cancel wait) |
| any, on detach/device error | stop, bounded drain | 11 | 10 s |

Implementation consequences:
- `capture.c:110`'s unconditional `dsl_control_bound(10)` before
  `ds_stop_collect`/`dsl_ring_finish` is replaced in buffer mode by the
  DRAIN row: the bound is re-armed on writer progress.
- `control.c:24`'s 5 s signal deadline becomes 15 s in buffer mode, so it
  can't preempt the 10 s cancel wait.
- The buffer drain wait in `dsl_ring_finish` uses a timed condition wait
  (≤ 100 ms) instead of an unbounded `pthread_join`, rechecking
  `dsl_signal` and device-error facts each time. On a signal it stops
  draining, reports the truncation and returns 128+sig.

**`SR_DF_TRIGGER` validation.** Each failure is a `DSL_DATA_ERROR`:
- payload NULL, `packet.status != SR_PKT_OK`, or `check_id !=
  TRIG_CHECKID`
- a second `SR_DF_TRIGGER`
- `SR_DF_TRIGGER` after any `SR_DF_LOGIC`, or `SR_DF_LOGIC` before it in
  buffer mode
- `status & 1` set while `real_pos >= emitted_count` (see below)

**Trigger value in META.** It comes only from the returned `status & 1`:
`META trigger: <real_pos>` if set, `META trigger: none` if not. The timeout
path that ran doesn't decide it. A forced upload whose header reports a
real trigger (a race) keeps `real_pos`.

**Length contract.**
- `aligned_actual` = the driver's recomputed `actual_samples`, derived from
  `(limit - remain_cnt) & ~SAMPLES_ALIGN`. With `SAMPLES_ALIGN = 1023` it's
  a multiple of **1024**. The arm count is dslcap's N rounded up to a
  multiple of 1024.
- `aligned_actual == 0`, or no header at all: "no capture",
  `DSL_DATA_ERROR` (13).
- Otherwise `emitted_count = min(user N, aligned_actual)`. That can be
  below 1024 only when N itself is below 1024. The callback sets the
  converter limit before the first sample (see Threads).
- Normal triggered captures must have `emitted_count == N`. A short count
  without a force is `DSL_DATA_ERROR`.
- Forced uploads may be shorter, in steps of 1024.
- dslcap logs `remain_cnt`, `aligned_actual` and `emitted_count`.

**Buffer-mode ring and drain policy.**
- The ring holds the whole capture: `capacity = header_bytes +
  emitted_max × unitsize`, using checked `size_t`/`uint64_t` arithmetic.
  Allocation failure or overflow is a config error before arming.
  `unitsize` keeps the existing physical-width rule (2 if any captured
  channel is ≥ 8). The worst case is therefore CH15 alone at 100M:
  268,435,456 × 2 B = **512 MiB**. 400M with one channel is 256 MiB.
- Buffer mode replaces the fixed 2 s `close_locked` drain deadline. Once
  END is received, the writer keeps draining for as long as it makes
  progress, using a **stall timeout** (default 30 s without any successful
  write; `--drain-timeout`, 1..3600). Stream mode keeps today's 2 s. The
  drain stays cancellable; see the precedence table.
- A consumer that stops reading for longer than the stall timeout gets
  exit `DSL_DRAIN_TIMEOUT` (14) with the truncation reported. A slow but
  progressing decoder always gets the whole capture.

### Output format

```
META samplerate: N\n
META trigger: K\n        # iff --trigger or --serial-trigger was given; K or "none"
<samples>
```

The number of header lines is fixed by the request, never by the data:
- 1 line without a trigger option, byte-identical to today
- 2 lines with `--trigger` or `--serial-trigger`

Consumers must not look at sample bytes to detect a second line: raw
samples can legally begin with the bytes `META trigger: 123\n`.

**What's on stdout after a failure depends on how far output got.**
- Failure **before** line 2 is pushed (a `fail` timeout, or cancel or
  error in WAITING, POST_TRIGGER or FORCED): stdout holds only the
  samplerate line.
- Failure **after** line 2 or any samples (cancel, detach, data error or
  drain stall in UPLOAD or DRAIN): bytes already written can't be
  retracted. stdout holds line 1, line 2 and a partial sample run, and the
  nonzero exit code marks it partial.
- In every case, **consumers must treat any nonzero exit as "output is
  incomplete"**, whatever the header lines say.

### C tests (`tests/fake_driver.c`, `tests/capture.cmake`)

Bench runs can't reliably hit callback and error races, so the fake driver
scripts each case:
- **`SR_DF_TRIGGER` faults:** NULL payload, bad `check_id`, packet status
  error, missing (END without it), duplicate, after `SR_DF_LOGIC`, and
  `real_pos >= emitted_count`
- **Ordered publication:** back-to-back `TRIGGER` → `LOGIC` → `END`
  delivered while the main thread is held (e.g. a test hook blocks its
  poll loop). Every sample must follow META line 2 and respect
  `emitted_count`. Also covered: the payload buffer is freed or
  overwritten right after the callback (catches retained pointers), and
  the main-thread timeout decision racing header publication.
- **Stale status (the fake cache refreshes only on a scripted
  schedule):**
  - cached zero with a valid header already published before the deadline
    decision: reconcile follows it, no timeout
  - stale zero with a real hit that the cache shows only within the grace
    window: the trigger wins
  - a hit that first appears after `T + G`: the timeout applies (16 for
    `fail`)
  - status reads failing throughout the grace window: the decision is
    logged "freshness unknown", and the counters appear in the summary
  - **header published immediately before the abort, with LOGIC still
    pending:** the result is nonzero aborted (16), reached within the
    FORCED/cancel bound. Never success, never an unbounded wait.
  - header published *after* the abort: ignored for completion. Still a
    nonzero result.
- **`WAIT_UPLOAD`:**
  - TRUE then no header → 13
  - FALSE with a valid completed capture (header/END within `G`) →
    success
  - FALSE with nothing → 11
  - forced header with `status & 1` → META reports K, not `none`
- **Length (1024-sample alignment):** `aligned_actual` of 0, 1024 and
  2048; user N of 1, 1023, 1024 and 1025; a forced count below N, a
  forced count above N (`emitted = min(N, aligned)`), and a short count
  without force (13)
- **Timeout, cancel and detach** injected in each state (WAITING,
  POST_TRIGGER, FORCED, UPLOAD, DRAIN). Check the exit code against the
  precedence table, the exact stdout bytes (line 1 only before line 2 is
  pushed; partial after) and the bound that fired.
- **These run through the real `control.c` watchdog and `ring.c` writer**
  (no simulated state events), with shortened bounds via test-only
  overrides.
- **Ring/drain:**
  - a slow but progressing consumer completes a capture that's many
    seconds long, with the hard bound re-armed by progress
  - a permanently stalled consumer exits 14 at `--drain-timeout`
  - SIGINT during a blocked drain exits 128+sig within the cancel wait
    (the writer must not block in `pthread_join`)
  - capacity arithmetic overflow, and the 512 MiB CH15 case, are rejected
    or allocated correctly
- **Position:** the logged effective-position formula matches
  `dsl_fpga_arm` for edge values (0%, 64-sample floor, 90% cap, rounding)
- **CLI rejections** from the validation rules above

## sigrok_hla.py changes

- New `[dslogic]` options:
  - `--dsl-mode stream|buffer` (default stream), forwarded as `--mode`
  - `--trigger NAME:COND`, with NAME resolved through `-C`, e.g.
    `--trigger nSS:f`, or a bare channel index. Forwarded as physical
    indices.
  - `--trigger-pos`, `--trigger-timeout`, `--on-timeout`
  - `--serial-trigger` (Phase C), with names resolved the same way
- **Option rules** (mirroring dslcap):
  - `--dsl-mode buffer` rejects `--continuous`, and the samplerate is
    checked against the buffer modes
  - simple and serial triggers are mutually exclusive
  - duplicate trigger channels are rejected
  - trigger modifiers without a trigger are rejected
  - all trigger options require `--dslogic` and `--dsl-mode buffer`
- **Channel union and width.** Resolved trigger channels (simple, and
  serial start/stop/clock/data) join `_used_channel_bits`, the single
  union that feeds both `build_dslcap_cmd --channels` and
  `dslcap_sample_unitsize`. Width and producer therefore can't disagree:
  for example, low-bit SPI plus a CH13 trigger gives `--channels` with 13
  and uint16 samples. Trigger channels outside 0-7 / 0-3 at 200M / 400M
  are rejected before launch, with the reason given.
- **META header.**
  - `MetaPrefix` takes an `expected_lines` argument (1 or 2) from the
    request. It reads exactly that many lines and never inspects sample
    bytes. Line 2 must be `META trigger: (K|none)` with K a decimal integer.
  - The 255-byte bound applies per line. Lines may arrive fragmented
    across reads.
  - If the producer exits after only line 1, report the producer's exit
    code (16 → `No trigger within --trigger-timeout`; 128+sig →
    interrupted) instead of a missing-header error. A missing line 2
    combined with exit 0 is a protocol error.
- **Trigger marker.**
  - Report `Trigger at sample K (t = K/rate s)` on stderr.
  - In the chronological output it's a heap event at K/rate. Pushing it
    must not advance the flush watermark: it's released by the normal
    `chunk_end - HOLD` rule, so earlier frames that are still being
    decoded come out first.
  - `none` prints `Untriggered capture (forced upload)` and adds no marker.
- **`--t0 trigger`** (optional) shifts printed times so the trigger is 0 and
  pre-trigger traffic is negative. The default keeps capture-start timing.
  - The time origin is chosen **once**, when line 2 is parsed: K means
    trigger-relative; `none` means capture-start plus a warning. It never
    changes after that.
  - A later producer failure can't re-time lines already printed. The
    harness prints `Output incomplete: producer exited N` and keeps the
    origin.
  - No output spooling by default.
- **Partial output:** any nonzero producer exit after samples started is
  reported as partial (exit-code mapping as above), never as a clean run.
- **`--dsl-drain-timeout S`:** forwarded as `--drain-timeout`, only with
  `--dsl-mode buffer`; same 1..3600 range check.
- **Tests:**
  - META: 1 vs 2 lines by request; adversarial sample payloads that begin
    with `META trigger: 1\n` (must be kept as samples in 1-line mode);
    fragmented and oversized lines; `none`; malformed K; exit 16 after
    line 1
  - command building: name-resolved triggers, union/width (CH13 →
    uint16), high-channel rejection at 200/400M
  - the option rules
  - marker ordering against the HOLD watermark
  - `--t0` with `none`, and with a failure before any samples
  - **cancel/error after real output** (partial samples, then exit 11/13/
    14/130) in both header modes (1 and 2 lines) and both time origins:
    output is kept, the origin doesn't change, and the partial warning and
    exit mapping are correct
  - `--dsl-drain-timeout` forwarding and rejection outside buffer mode

## Phases

**A. 200M/400M sample correctness (prerequisite).** The README marks fast
buffer capture as unvalidated, and triggers at those rates are meaningless
until the packing is proven. A lane-packing error would reorder samples
within each 2-/4-sample group, showing up as 1-3-sample glitch runs and
distorted periods. If the half/quarter cross layout differs from the 100M
one, fix `convert.c` here.

Bench facts (pi133, checked 2026-10-07 over ssh):
- Raspberry Pi 4B Rev 1.5, kernel 6.12.47. `core_freq=500` and
  `core_freq_min=500`, so the core clock is fixed (`measure_clock core` =
  500.000992 MHz).
- `spi0.0`/`spi0.1` are spidev, driven by `mcp_radio --mcp-http`. The
  application requests 8 MHz, and spi-bcm2835 rounds the divider up to an
  even number: 500 MHz / 64 = **7.8125 MHz**. That matches the 7.80-7.82 MHz
  measured at 25M in VALIDATION.md, and gives exactly 51.2 samples per SCLK
  period at 400M and 25.6 at 200M.
- **GPIO4 is not free.** A userspace process owns it as an output held high,
  and GPIO5 is an input with an IRQ. Both are almost certainly the radio's
  NRESET and DIO8 lines under mcp_radio. Free GPCLK pins for the reference
  clock: **GPIO6 / GPCLK2 (pin 31)** or **GPIO20 / GPCLK0 ALT5 (pin 38)**.
  Avoid GPCLK1.

A0. **Idle bus. Done 2026-10-07.** Buffer captures with no traffic:
- 400M CH0-3 for 160 ms gave exactly 64,000,000 samples; 200M CH0-7 for
  160 ms gave 32,000,000. Both exited 0.
- Levels match an idle mode-0 bus on both rates: SCLK=0, MISO=1, MOSI=0,
  nSS=1, BUSY (CH4) = DIO8 (CH5) = 0, CH6/7 floating low. No edges.
- This proves arm, capture, upload and conversion run end to end at
  200/400M. It doesn't test packing.

A1. **SPI-only pass, no rewiring. PASSED 2026-10-07.**

Setup: hydra-develop-26 ran read-only `GetVersion` on pi133 via mcp_radio
`hal_script`, 5 ms spacing, 90 s (secondopinion task
`dslcap-a1-pi133-getversion-20261007`, READY/GO handshake). Traffic
appeared about 9 s after GO. Six 160 ms buffer captures (22:12:17-22:12:28
UTC) all exited 0.

| capture | frames | clocks/frame | bytes | mean period (samples) | error | runs < 3 | BUSY pulses |
|---|---|---|---|---|---|---|---|
| 400M #1 | 58 | 16×29, 32×29 | 174 | 51.1987 ± 0.0053 | -0.003% | 0 | n/a |
| 400M #2 | 58 | 16×29, 32×29 | 174 | 51.2110 ± 0.0054 | +0.021% | 0 | n/a |
| 400M #3 | 58 | 16×29, 32×29 | 174 | 51.2069 ± 0.0054 | +0.013% | 0 | n/a |
| 200M #1 | 60 | 16×30, 32×30 | 180 | 25.6024 ± 0.0044 | +0.009% | 0 | 60 |
| 200M #2 | 58 | 16×29, 32×29 | 174 | 25.6018 ± 0.0044 | +0.007% | 0 | 58 |
| 200M #3 | 58 | 16×29, 32×29 | 174 | 25.6043 ± 0.0046 | +0.017% | 0 | 58 |

- Every frame decoded exactly: `0101`/`0452` requests and
  `00000000`/`06520118` responses. There's one BUSY pulse per **frame**
  (58 pulses for 29 commands). DIO8 (CH5) stayed flat.
- Worker report (revision 5, sha256 `10e2d8b7…`):
  - 22:12:16.936Z-22:13:47.671Z, 41 `hal_script` calls × 400 = 16,400
    commands (inferred from call duration; `RESULTS_CAP = 128` returns
    only 5,248 results)
  - all 5,248 returned responses were `01 18`, status 0, no errors
  - mean spacing 5.53 ms, so about 29 commands per 160 ms window, which
    matches the 58 frames measured
  - pi133 ended in its preflight state (STBY_XOSC, IRQ 0); nothing else
    changed
- **The run-length criterion was corrected.** "Only 25/26 / 12/13" assumed
  exactly 50% duty. Split by phase, the runs are:
  - 400M: high {26, 27} (mean 26.09) and low {25, 26} (mean 25.11), duty
    **50.96%**
  - 200M: high {13, 14} (mean 13.04) and low {12, 13} (mean 12.57), duty
    **50.91%**

  Each phase spans exactly two adjacent values, which is pure
  quantization. The two rates measure the same duty independently, which
  a lane-packing error couldn't produce.

  Corrected criterion: per phase (high/low), runs take at most two adjacent
  values; there are no runs < 3; the high+low means add up to the nominal
  period; and the duty agrees across rates within 0.2%.
- **Conclusion:** half/quarter (200M/400M) buffer packing and
  `convert.c`'s cross-data transpose are correct on CH0-7 / CH0-3. No
  converter change is needed. ppm-level frequency acceptance remains with
  A2.
- Data and scripts are in the session scratchpad (`phaseA/a1/*.raw|json`,
  `a1_analyze.py`, `a1_capture.sh`, `runstats.py`). Move them into
  `dslcap/tests/` when this is committed.

Original A1 procedure, kept for reference. pi133 SPI is on CH0-3, the only
channels 400M reaches.
- **Traffic.** An untriggered window lasts only 160 ms, so ask the pi133
  application owner for sustained read-only `GetVersion` (about every 5 ms
  for about 30 s). Don't use spidev directly: mcp_radio owns the bus, and
  earlier validations went through the owner with explicit approval. Take
  3+ captures at 400M CH0-3 and 3+ at 200M CH0-7 while the traffic runs.
- **Pass criteria:**
  - within-byte SCLK high/low runs are only 25/26 samples (400M) or 12/13
    (200M), with no runs < 3 samples on any channel. This is the
    packing-error test.
  - **Aggregate within-byte period:** for each byte, measure first-to-last
    rising edge (7 periods; nominal 358.4 samples at 400M, 179.2 at 200M),
    excluding the inter-byte stretch and inter-frame gaps (already seen at
    25M). The mean over all bytes must be 51.2 ± 0.1 samples (400M) or
    25.6 ± 0.05 (200M), with the number of bytes and the standard error
    reported.
    - Uncertainty budget: ±1 sample over 358 is about 2,800 ppm per byte,
      and around 150 bytes per capture only reaches roughly 200-300 ppm.
      So A1 checks the period at the **0.2% level** only.
    - ppm-level frequency acceptance belongs to A2's continuous clock.
  - each nSS frame has 16 or 32 rising SCLK edges (alternating, as at 25M)
  - decoded bytes are `0101`/`0452` (request MOSI/MISO) and
    `00000000`/`06520118` (response), the same as the 25M reference in
    VALIDATION.md
  - BUSY on CH4 pulses between the request and response frames (200M only,
    since CH4 is outside 400M's reach)
- The analysis script is `runstats.py`: per-channel level, edge count and
  run-length histogram from a dslcap file. Move it into `dslcap/tests/`
  when this is committed.

A2. **Continuous reference clock (after A1).**

Completed: three captures each at 200M and 400M measured +15.08 to +15.11 ppm
relative to the nominal 1MHz Pi reference, within ±100 ppm, with clean run
lengths. This is relative clock agreement, not absolute calibration. Clock,
daemon, GPIO and MISO wiring restoration were verified, including a known
GetVersion decode. Independent review: [A2_REVIEW.md](A2_REVIEW.md).

- **Preconditions, rechecked immediately before use.** The gpioinfo result
  above is a snapshot, not evidence of physical isolation.
  - Re-run `gpioinfo` and `pinctrl get 6,20` on pi133.
  - Confirm with the user/bench owner that nothing is wired to the chosen
    header pin besides the new analyzer lead.
  - `pigpiod` is installed but **not running** on pi133. Starting it is a
    reversible service change on a bench the radio owner shares:
    coordinate with the pi133 application owner and stop it afterwards.
  - Installing any new package or tool is a **new dependency**. Under the
    user's standing rule that needs a new plan before use.
- Pi GPCLK on GPIO6 or GPIO20 with a verified integer clock divider and
  no MASH dithering. Do not assume pigpio selects the raw 54MHz oscillator.
  On the tested Pi4, `pigs hc 20 1000000` selects PLLD_PER / 750:
  GP0CTL=0x96 (source6, MASH0), GP0DIV=0x002ee000 (DIVI750, DIVF0).
  Kernel-modeled PLLD_PER=750000023Hz gives nominal 1000000.03Hz; it is not
  a physical frequency calibration. A raw 54MHz source could use /54 for
  1MHz or /6 for 9MHz, but that source was not selected or tested here.
- CH1 works for **both 200M and 400M**, as used in the completed test.
  CH6/CH7 are an alternative for 200M only. For 400M use CH0-3:
  - **unplug the CH1 analyzer lead from MISO completely**, then connect it
    to the GPCLK pin
  - the GPCLK output must never be electrically joined to radio MISO (no
    clip still touching MISO, no shared jumper)
- Check:
  - high/low runs cluster tightly at rate/(2f) (e.g. 9 MHz at 400M →
    22/23 samples), with no short runs
  - the frequency from first-to-last edge over the whole capture (about
    1.5 M periods at 9 MHz over 168 ms) is within ±100 ppm. The
    quantization contribution is about 1e-6 here; the budget is dominated
    by the Pi and DSLogic crystals.
- Restore afterwards and verify: GPCLK off, the pin back to its original
  input/pull state (`pinctrl get`), `pigpiod` stopped if it was started,
  and CH1 back on MISO. Compare idle levels with a contemporary pre-test
  baseline; MISO=1 in A0 was a snapshot, not an invariant. The A2 baseline
  and restored idle are MISO=0. A separately authorized read-only GetVersion
  also verified restored MISO data as 0452 / 06520118.
- The LR2021 HF clock out (32 MHz, via a DIO) is a crystal-accurate
  alternative. It's near the edge of a clean edge at the DSLogic input and
  needs the radio owner to configure it, so it's second choice.

**B. Simple trigger.**
1. **Golden register image.** An `LD_PRELOAD` shim on `libusb_bulk_transfer`
   logs EP2 OUT payloads. Both DSView (static libsigrok4DSL, dynamic
   libusb) and dslcap go through it. For the same settings (buffer 100M,
   8 ch, `3:f`, 10%, and a 200M variant) the `DSL_setting` bytes must be
   identical. This catches order and reset mistakes without any signal
   interpretation.
2. **Self-consistency.** With a periodic signal, run 20 captures per case
   (`f`, `r`, `1`, `e`, a two-channel AND) at 100/200/400M. The trigger
   channel must satisfy COND at output sample `real_pos`. Record the
   tolerance: expect 0 at 100M, and possibly 2 or 4 samples of granularity
   at 200/400M from lane replication. Then document it.
3. **Pre-trigger placement.** Positions 0, 10, 50 and 90%. Check that
   `real_pos ≈ pos × N`. Then the early-trigger case: arm while the event
   is already continuous, with pos 90%. The pre-trigger region won't have
   filled, so check whether `real_pos` comes out smaller, and that META
   reports the real value.
4. **Timeout and cancel.** Trigger on a channel that never toggles:
   - `fail` → exit 16 after the grace window (between `T` and about
     `T + G`, plus the stop's bounded wait), output = samplerate line only,
     and the device reopens immediately. Record the measured decision time
     and the freshness counters.
   - `upload`, varying T against the capture length:
     - immediate (T ≈ 0)
     - T shorter than the time to fill N
     - T longer than the time to fill N
     Expect `META trigger: none`, `emitted_count = min(N, aligned_actual)`,
     the logged `remain_cnt`, or the defined "no capture" error for zero or
     sub-block results. Record what the FPGA actually returns: whether the
     pre-trigger region is a ring and how much of it is uploaded.
   - Ctrl-C while waiting and while in POST_TRIGGER → clean exit, then
     reopen
5. **Trigger near the timeout.** Use a long post-trigger window (pos 10%,
   full depth at 100M ≈ 300 ms post-trigger) with the trigger edge from a
   Pi GPIO pulse timed relative to T. Use a free pin, under the same
   recheck and owner-coordination preconditions as A2.
   - well before T
   - just before T (inside the last poll interval)
   - just after T
   Assertions, per the grace/observation policy:
   - well before T: expected to produce a complete triggered capture
     (length N, `META trigger: K`). A miss is recorded with the freshness
     counters, not silently accepted.
   - just before T, just after T, within the grace window: **either
     outcome is acceptable**, but it must be self-consistent:
     - triggered: length N, META K, exit 0
     - `fail` timeout: exit 16, samplerate line only
     - `upload`: META from the header's real status, and length =
       `min(N, aligned)`
   - an abort that raced a header must never report success.
   - with `upload`, a hit racing the force reports K if the header says
     triggered.
   Record the observed `G`, the decision times and the freshness counters
   for each run.
6. **End to end.**
   - pi133 SPI at 400M on CH0-3 with `--trigger nSS:f`
   - both radios at 100M (12 ch) with a trigger on pi134 nSS
   For both, decode with `sigrok_hla.py --dsl-mode buffer` and diff the
   HLA output against a streaming decode of the same command sequence.

**C. Serial trigger (SPI opcode).** Map `--serial-trigger` to DSView's
serial template (stage roles and counts above; logic = AND, non-contiguous,
`==` invert, as in the UI defaults). The clock is `r`/`f` only.

Validation update: C.1 passed; C.2's original crossed-word tests missed at100M.
Real DSView reproduced those misses, but a later constant-register alignment
test hit twice with the target at clocks 17–32 and missed at clocks 25–40.
The native aligned C.2 run now passes at100M CH0–7: intended order hits twice,
three reversals time out, and all four separate NSS waveform controls pass.
The old crossed-word misses do not establish order. C.3 and higher-rate live
bit-order checks remain pending. See SERIAL_INTERMEDIATE_WIDTHS.md and
SERIAL_ALIGNED_VALIDATION.md; source mapping is unchanged.

1. **Golden image:** B.1-style comparison against DSView's serial tab with
   an identical setup, including value `0x1c35`/`bits=16`, so the
   per-bit placement in the stage-3 value strings is pinned byte for byte.
2. **Bit order on hardware:** arm on `0x1c35` with traffic that contains
   only one of its variants per run: `0x1c35`, the bit-reversed `0xac38`,
   the byte-swapped `0x351c`, or both (`0x38ac`). Only the intended order
   may fire. This confirms or overturns the MSB-first assumption in the CLI
   rules. Generating the traffic needs the radio owner (e.g. a read-only
   command whose MOSI bytes carry the pattern) and is arranged with them,
   like A1.
   Put the test value at a complete 16-bit word boundary after NSS assertion:
   for the existing FIFO carrier use `00 02 VV VV 00 00` (target clocks 17–32),
   not `00 02 00 VV VV 00` (clocks 25–40). First verify the aligned positive,
   then test each reversal with the same alignment. The previous crossed-word
   misses cannot establish bit order.
3. **Opcode trigger:** trigger on a specific LR1110/LR2021 opcode
   (`bits=16`) and check that the decoded transaction at the trigger
   carries it.

Limitation to document: comparison is observed at complete `bits`-wide word
boundaries after start in the tested DSView setup. An opcode value can also
fire on an aligned payload word later in the same transfer. A matching value
crossing a word boundary is not sufficient. Stop (nSS rise) clears serial state.

## Later / not planned

- **Stream-mode trigger:** the same API, but the pre-trigger is capped at
  10% of channel depth and guide §2.5.1 says loop mode ignores triggers. Add
  only if a finite stream capture with a start trigger is needed.
- **Multi-stage advanced trigger CLI:** expose only if simple + serial
  don't cover a real case. A JSON stage file would be the shape.
- **Repeated triggered captures:** a re-arm loop that writes one capture
  per trigger. That needs a framing format beyond the single-capture META
  header.

## Risks / open questions

- **200M/400M packing:** validated by A1 (2026-10-07) with real SPI traffic
  on CH0-7 / CH0-3. A2 (a continuous clock) still owes ppm-level frequency
  acceptance.
- **`real_pos` semantics at half/quarter rates.** DSView uses it directly
  at the effective rate. B.2 confirms it.
- **Trigger on disabled channels:** unknown whether the FPGA sees them. Our
  rule forbids it, which avoids the question.
- **`SR_CONF_WAIT_UPLOAD` partial-upload path:** less exercised (DSView only
  uses it on user stop). B.4 covers its sample count and END handling, and
  B.5 covers its race with a real trigger. It must never be issued after
  `trig_hit`.
- **Trigger hit vs header timing, and cache staleness:** the header comes
  only at capture end, and the status cache is stale by a nominal 17
  driver polls, with no hard bound. The deadline rule gives best-effort,
  always self-consistent semantics, and reports when freshness is unknown.
  An abort can't be undone, so the reconcile happens before it. Hardware-event-time guarantees would need a driver
  change (a synchronous status refresh, or control of `bmFORCE_STOP` versus
  a hit), which is out of scope. B.5 measures the real `G` and race
  behavior.
- **Unknown forced-upload content:** how much pre-trigger data the FPGA
  returns after `bmFORCE_STOP` (a ring or linear fill) is unknown until
  B.4 measures it. The length contract doesn't depend on the answer.
- **Large buffer outputs:** up to 512 MiB in memory (CH15-only width
  case). The capacity check fails cleanly before arming if allocation is
  impossible.
- **Watchdog/trigger interaction:** an unbounded wait has to be explicit
  (no `--trigger-timeout`). A scripted caller should always pass one.
- **Bench safety for A2:** GPCLK pin ownership and wiring are rechecked at
  use time, and the CH1 lead moves rather than bridges. Starting `pigpiod`
  is coordinated with the pi133 owner and reverted afterwards. New
  dependencies need a new plan (the user's rule).
- **Partial output:** cancels or failures after emission leave partial
  stdout. Consumers rely on the exit code, which the harness enforces.
- **Wiring:** dual-radio captures above 100M need pi134 moved onto CH4-7,
  which costs pi133's BUSY/DIO8 on CH4/5. That's a bench decision, not code.
