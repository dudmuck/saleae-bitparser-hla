# sigrok_hla.py

Live SPI capture with HLA decoding, directly from USB hardware. An alternative to `spi_hla.py` (which processes saved binary exports).

## Backends

Three capture backends are supported:

| Backend | Flag | Hardware | How it works |
|---------|------|----------|--------------|
| **Saleae** | `--saleae` | Saleae Logic 8, Pro 8, Pro 16 | Connects to Logic 2 app via automation API (gRPC on port 10430) |
| **sigrok** | `-d DRIVER` | sigrok-supported analyzers | Runs sigrok-cli with the shared NumPy decoder, or the optional srd SPI decoder |
| **DSLogic** | `--dslogic` | DSLogic Plus PGL12 (`2a0e:0034`) | Runs dslcap with the shared NumPy decoder |

Both backends work with the Saleae Logic 8 (21a9:1004). The **sigrok backend** uses the `saleae-logic-pro` driver (requires libsigrok built from source with Logic 8 support). The **Saleae backend** uses the Logic 2 app's automation API.

## Prerequisites

### Saleae backend

- **Saleae Logic 2** desktop app running
- Automation server enabled: Edit -> Settings -> check "Enable Automation Server" (default port 10430)
- Python package: `pip install logic2-automation`

### sigrok backend

- `sigrok-cli` built from source (the Ubuntu/Debian package has a broken `-P` flag)
- `libsigrok` and `libsigrokdecode` built from source
- Hardware supported by sigrok (saleae-logic-pro for Logic 8, fx2lafw, saleae-logic16, etc.)

To build from source:
```bash
# In your sigrok build directory:
cd libsigrok && ./autogen.sh && ./configure && make -j$(nproc)
cd libsigrokdecode && ./autogen.sh && ./configure && make -j$(nproc)
cd sigrok-cli && ./autogen.sh && PKG_CONFIG_PATH=../libsigrok:../libsigrokdecode ./configure && make -j$(nproc)
```

Set `LD_LIBRARY_PATH` to include the local `.libs` directories, or add the local `sigrok-cli` to `PATH`. **Warning:** if the distro's sigrok-cli gets picked up instead, acquisition fails with `g_variant_get_type: assertion 'value != NULL' failed` (ABI mismatch with the local libsigrok).

Better: install the local build to `~/.local/bin` with the library paths baked
in as rpath, so no `PATH`/`LD_LIBRARY_PATH` is ever needed (and rebuilds of
libsigrok/libsigrokdecode are picked up automatically):

```bash
cd sigrok-cli && PKG_CONFIG_PATH=/mnt/foo/libsigrok:/mnt/foo/libsigrokdecode \
LDFLAGS="-Wl,-rpath,/mnt/foo/libsigrok/.libs -Wl,-rpath,/mnt/foo/libsigrokdecode/.libs" \
./configure --prefix=$HOME/.local && make -j$(nproc) && make install
```

### DSLogic backend

Build the standalone C producer using [dslcap's instructions](dslcap/README.md).
Pass its path with `--dslcap PATH`, or put `dslcap` on PATH. The default
firmware resource directory is `/usr/local/share/DSView/res`; the producer
verifies activation/security/HDL before capturing. Save captures and close
DSView before using the analyzer.

The DSLogic Python backend uses the existing NumPy engine and accepts
physical channels 0–15. It reads one byte per sample when every captured
channel is below 8, otherwise two little-endian bytes. Width follows the
exact SPI/pin/trigger channel union passed to dslcap, not unused `-C` name mappings.
This wider input applies only to `--dslogic`; sigrok raw behavior remains
8-bit and Saleae automation is unchanged. `--engine srd`, `-d`, `-I` and
`--saleae` cannot be combined with `--dslogic`; `-i` replays a dslcap raw file
([capture first, decode after](#dslogic-capture-first-decode-after)) and `-T`
accepts only `deglitch` ([with `--dslogic`](#with---dslogic)). For a live
capture, select exactly one of `--samples`, `--time`, or
`--continuous`. The default rate is 25M, threshold 1.6 V; `--vth V` accepts
0–2.5 V. Unsupported rates/channel counts are rejected before the producer
is launched, and dslcap verifies its own configuration and readback.

### DSLogic buffer triggers

Streaming remains the default. Use `--dsl-mode buffer` with an explicit
`--samples` or `--time` for finite captures at up to 100M on physical lanes
0..15, 200M on 0..7, or 400M on 0..3. Buffer mode rejects `--continuous`.

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap \
  --dsl-mode buffer --samplerate 100M --samples 1M \
  --spi SCLK,MISO,MOSI,nSS -C 0=SCLK,1=MISO,2=MOSI,3=nSS,15=IRQ \
  --trigger nSS:f,IRQ:h --trigger-pos 10 \
  --trigger-timeout 5s --on-timeout upload --drain-timeout 30 \
  --hla-path /path/to/HLA
```

`--trigger NAME:COND,...` resolves names through `-C`, case insensitively;
bare physical indices are also accepted. Conditions are rising `r`/`R`,
falling `f`/`F`, high `1`/`h`, low `0`/`l`, and either edge `e`. Terms are
ANDed at one sample. Duplicate physical trigger channels and out-of-range
fast-buffer lanes fail before launch. Trigger inputs join the same channel
union used for producer selection and sample width; a CH15 trigger alone
makes low-channel SPI input uint16 even when CH15 is not logged as a pin.

`--trigger-pos` is an integer 0..90 percent, default 10. Without
`--trigger-timeout` the producer waits until triggered or interrupted.
Timeout action defaults to `fail` (exit 16, `No trigger within
--trigger-timeout`); `upload` can return fewer samples after a forced stop.
The producer uses a nominal 340 ms status observation grace with no hard
cache-age guarantee. `--drain-timeout` is the buffer output stall limit,
1..3600 seconds, default 30; progressing decoding can take longer.

Triggered requests parse exactly two bounded META lines, even across
fragmented reads. Stderr reports `Trigger at sample K (t = K/rate s)`, and
a trigger marker enters the chronological output heap at that time without
advancing the decoder's watermark. Timestamps remain relative to capture
start. An untriggered forced upload reports `Untriggered capture (forced
upload)` and emits no trigger marker. Producer timeout/signal failures
retain their exit codes even if the second META line never arrived. A
successful producer missing that line is a protocol error. Trigger-relative
`--t0` remains unsupported.

### DSLogic serial triggers

Use `--serial-trigger` instead of `--trigger` to match a clocked bit value:

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap \
  --dsl-mode buffer --samplerate 400M --samples 1M \
  --spi CLK,MISO,MOSI,nSS -C 0=CLK,1=MISO,2=MOSI,3=nSS \
  --serial-trigger 'start=nSS:f,stop=nSS:r,clock=CLK:r,data=MOSI,value=0x1c35,bits=16' \
  --trigger-timeout 5s --hla-path /path/to/HLA
```

Fields start, stop, clock, data, value and bits are required exactly once, in
any order. Names resolve through `-C` case insensitively; numeric indices work.
Start/stop accept simple trigger conditions; clock requires rising or falling
edge (`r`/`R`/`f`/`F`), and data is a bare name/index. Value requires `0x` or
`0b` (uppercase prefixes accepted), fitting bits1..16. Shared role channels
are permitted; simple and serial specifications are mutually exclusive.
Resolved start/stop/clock/data join the same producer/width channel union,
including inputs not used by SPI decoding or pin logging. All normal buffer
lane, explicit finite-duration, timeout/action, metadata and drain rules apply.

MSB-first shift order passed native hardware comparison at 100 MS/s on CH0–7:
the most recent bit is the value LSB. Generated stage3 compares the low N bits;
unused upper bits are X (use DSView's bit editor with upper X for a short-width comparison).
Treat matches as N-bit words aligned from the start condition. Real DSView
tests with identical 16-bit settings matched `0x1c35` at clocks 17–32 after
NSS assertion but missed at clocks 25–40. Arbitrary sliding-window matching
is not supported by that evidence. Later aligned payload words can match,
so this is not an opcode-only filter. See
[the alignment evidence](dslcap/SERIAL_INTERMEDIATE_WIDTHS.md).
Source-derived offline stage tests, including0x1c35, are not a real DSView
register golden or proof of hardware bit/byte order. Subsequent actual register
comparison passed at 100/200/400 MS/s; see [native aligned order results](dslcap/SERIAL_ALIGNED_VALIDATION.md)
for the separate 100 MS/s hardware check. Opcode validation then passed at
100 MS/s ([opcode results](dslcap/OPCODE_VALIDATION.md)), and live bit order
passed at 200 MS/s CH0–7 and 400 MS/s CH0–3
([higher-rate results](dslcap/SERIAL_HIGHRATE_VALIDATION.md)). Serial requests use the same exact two-line
META parsing and chronological trigger marker, keeping capture-start timestamps.

### Common prerequisites

- A Saleae High Level Analyzer (HLA) directory containing `HighLevelAnalyzer.py`
- Python 3
- NumPy for the default vectorized engine; optional Numba accelerates byte packing

## Pin Layout

The default pin mapping (matching `bw1/digital.csv`):

| Channel | 0    | 1    | 2    | 3   | 4      | 5      | 6      | 7     |
|---------|------|------|------|-----|--------|--------|--------|-------|
| Signal  | SCLK | MISO | MOSI | nSS | SCLK_B | MISO_B | MOSI_B | nSS_B |

Specified as `--spi CLK,MISO,MOSI,CS` (channel numbers for Saleae, channel names for sigrok).

## Usage

### DSLogic backend

Single SPI port:

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap \
    --hla-path ~/HLA/saleae_lr2021 --spi 0,1,2,3 --samplerate 25M --time 5s
```

Dual-radio wiring, continuous capture with producer debug diagnostics:

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap -vv \
    --hla-path ~/HLA/saleae_lr2021 --spi 0,1,2,3 --spi 8,9,10,11 \
    --samplerate 25M --continuous --hex
```

This maps pi133 SCLK/MISO/MOSI/NSS to CH0/1/2/3 and pi134 to CH8/9/10/11.
The producer captures only these eight physical inputs (`0,1,2,3,8,9,10,11`)
and outputs two bytes per sample, preserving physical bit positions. It keeps
the 12-channel-capable 25M stream profile; it does not lower the requested
rate. Up to four additional selected inputs can fit this profile, subject to
wiring and decoder requirements. Selecting all 16 at 25M fails.

With the bench's BUSY and DIO8 connections, log all four status inputs:

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap \
    --hla-path ~/HLA/saleae_lr2021 --spi 0,1,2,3 --spi 8,9,10,11 \
    -C 4=pi133_busy,5=pi133_dio8,12=pi134_busy,13=pi134_dio8 \
    --int-pin pi133_dio8 --extra-pin pi134_dio8 \
    --extra-pin pi133_busy --extra-pin pi134_busy \
    --samplerate 25M --continuous --hex
```

On each Pi, BUSY is physical pin 12 (GPIO18), and DIO8 is physical pin 29
(GPIO5). With the original pi134 wiring restored, CH12 is pi134 BUSY and
CH13 is pi134 DIO8; pi133 remains CH4 BUSY and CH5 DIO8.
Both Pi grounds connect to the analyzer. This selects exactly 12
inputs, mask `0x3f3f`, with two-byte samples at 25 MSa/s. Both pin options
log edges interleaved with decoded SPI; `--int-pin` is not a hardware trigger.
Static levels produce no initial event. The radio application retains GPIO
ownership; do not run the earlier synthetic GPIO generators on these pins.

Bench validation observed BUSY on both Pis and DIO8 IRQ edges on both Pis.
For pi134, DIO8 transitions were confirmed after the application configured
IRQ routing and calibrated a 915 MHz LoRa RX-timeout test. Those radio settings
remain on pi134; see [validation details](dslcap/VALIDATION.md#pi134-configured-rx-timeout--dio8--2026-10-07).

Named channels and a logged interrupt:

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap \
    --hla-path ~/HLA/saleae_lr2021 \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS,4=INT \
    --spi SCLK,MISO,MOSI,nSS --int-pin INT --samplerate 25M --samples 1M
```

`-C/--channels` maps names to physical bits; it is not a dslcap capture list.
The backend derives a sorted unique producer `--channels` list from the SPI
roles, `--int-pin`/`--extra-pin` references, and resolved trigger conditions. Unreferenced named signals
are not captured and cannot force two-byte input. Duplicate names, negative
indices and references above channel 15 fail before
starting the producer. SPI roles may use numbers or mapped names; name lookup
is case insensitive. `D0=SCLK` mapping syntax is also accepted.

`-v`/`--dslcap-verbose` enables information logs; repeat it or use `-vv` for
debug. These flags require `--dslogic`. The default keeps producer error logs,
including its security-pass evidence.

Both raw producers and the srd subprocess drain stderr concurrently, prefixing
live diagnostics with `[dslcap stderr]` or `[sigrok stderr]`. This prevents
child stderr from filling while decoding waits on stdout. The raw decoder
parses fragmented META headers before initializing SPI and pin timestamps.
If META disagrees with the requested rate, it warns and uses the reported
rate for both. Wide DSLogic input carries an odd trailing sample byte across
reads after META parsing, and rejects an incomplete two-byte sample at EOF.
dslcap requires a valid META header; legacy sigrok raw input
without META retains the configured rate. HOLD/heap ordering and fast_spi
interfaces remain shared.

Producer nonzero exits, malformed/missing required META, and pipe reader
errors fail the command with a diagnostic. Already printed results may be
partial; check the process exit status. Ctrl-C returns 130 after cleanup and
prints already completed queued results. Cancellation sends TERM, waits up
to three seconds, then kills/reaps a stubborn child; queue/select readers
are stopped and pipes closed. Downstream pipe closure also cleans up the
producer. A producer closing stdout but failing to exit gets a bounded
15-second wait before kill. Valid sigrok and Saleae flows remain available.

Offline verification:

```bash
python3 -m unittest discover -s tests -p 'test_dslcap*.py' -v
```

Tests use fake subprocesses only: fragmented META, synthetic single/dual SPI
on low and high physical bits, little-endian uint16, odd/random byte splits,
truncated EOF, high-pin timestamps/order, stderr exceeding pipe capacity, producer
failure, read faults, saturated queues and stubborn-child cancellation.
They also exercise sigrok raw input and srd annotation regressions. The
DSLogic examples describe supported syntax; real dual-SPI HLA comparison
against Saleae/Logic 2 and verbose live capture remain validation gates until
recorded in [dslcap/VALIDATION.md](dslcap/VALIDATION.md).

### DSLogic: capture first, decode after

For long or busy runs, record dslcap's raw output to a local file and decode it
afterwards. `--raw-out FILE` runs the same derived dslcap command with stdout
going straight to FILE, so no Python sits in the sample path, and it writes a
`FILE.json` record (command, captured channels, sample width, META line count,
exit status, byte count). It refuses to start if either FILE or `FILE.json`
already exists, and exits with dslcap's status (128 + signal if dslcap was
killed). Ctrl-C or SIGTERM to sigrok_hla.py is forwarded to dslcap, which then
drains and stops; it is escalated to TERM and KILL if dslcap is still running
after 20 s. Decode with `--dslogic -i FILE` and the same `-C`/`--spi`/pin options:

```bash
./sigrok_hla.py --dslogic --dslcap /tmp/dslcap-build/dslcap \
    --hla-path ~/HLA/saleae_lr2021 --spi 0,1,2,3 --spi 8,9,10,11 \
    -C 4=pi133_busy,5=pi133_dio8,12=pi134_busy,13=pi134_dio8 \
    --extra-pin pi133_busy --extra-pin pi134_busy \
    --samplerate 25M --time 77s --raw-out /local/cell.raw

./sigrok_hla.py --dslogic -i /local/cell.raw \
    --hla-path ~/HLA/saleae_lr2021 --spi 0,1,2,3 --spi 8,9,10,11 \
    -C 4=pi133_busy,5=pi133_dio8,12=pi134_busy,13=pi134_dio8 \
    --extra-pin pi133_busy --extra-pin pi134_busy --hex
```

Replay goes through the same decoder as live capture, and its output is
identical for the same bytes. With a `FILE.json` record, the replay reads the
file's real sample width (so decoding one bus of a 12-channel capture works),
and fails if a mapped channel was not captured or the record is malformed. It
also refuses a capture whose record shows a nonzero exit, no exit status
(unfinished), or a byte count different from the file; `--dsl-partial` decodes
it anyway with a warning, since its last transaction may be cut. A continuous
capture stopped with Ctrl-C records exit 130, so replay it with `--dsl-partial`.
Without a record, the width comes from the mapping as in a live run. A file captured with a trigger has a
second META line: pass `--dsl-triggered`. Capture options (`--samples`,
`--time`, `--vth`, triggers, `--dsl-mode`) are rejected with `-i`. Raw size at
12 channels and 25 MSa/s is 50 MB/s: about 460 MB for 9 s and 11.5 GB for
230 s, so write to local disk, not NFS.

**Sustained decode rate.** Measured by replaying synthetic 12-channel, uint16
dual-bus ping-pong (10 MHz SCLK at 25 MSa/s, 511-byte FIFO writes and reads,
prbs9 payloads; every transfer byte-exact against the expected bytes), as
`sigrok_hla.py --dslogic -i FILE --spi 0,1,2,3 --spi 8,9,10,11 [--hex]`
wall time, startup included. Ranges span medians of three runs from two
independent measurements on this host while other jobs loaded it (load
average up to 11 of 20 CPUs), so treat them as indicative:

| SPI load (CS active per bus) | `--hex` | plain | `--hex` + deglitch | vs 50 MB/s live |
| --- | ---: | ---: | ---: | --- |
| ~18% (planned pcycle ping-pong, 4.6 ms period) | 102-118 MB/s | 130-132 MB/s | 79-85 MB/s | 1.6-2.6x margin |
| ~59% | 33-35 MB/s | 36-41 MB/s | | below real time |

Decode cost is dominated by decoded SPI bytes (the per-byte HLA frame path,
about 0.64 µs per byte), not samples (about 2 ns each). Live decode keeps up
while the two buses together carry under roughly 1-1.5 MB/s of SPI data; heavier
traffic slowly fills dslcap's 256 MiB ring and ends with an overflow. Use
`--raw-out` and replay for those, or for any run where a missed byte matters.
Numbers include about 0.6 s of startup and were taken on this host (20 CPUs,
NumPy 2.2, numba 0.63); without numba the SPI scan falls back to NumPy and is
roughly three times slower.

### Saleae backend

Single SPI port, timed capture:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \
    --spi 0,1,2,3 --samplerate 25M --time 1
```

Dual SPI port:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \
    --spi 0,1,2,3 --spi 4,5,6,7 --samplerate 25M --time 1
```

Manual capture (Ctrl-C to stop recording, then processing begins):
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \
    --spi 0,1,2,3 --samplerate 25M
```

With hex dump of raw bytes:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \
    --spi 0,1,2,3 --spi 4,5,6,7 --samplerate 25M --time 2 --hex
```

Limit capture buffer to 500 MB:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \
    --spi 0,1,2,3 --spi 4,5,6,7 --samplerate 25M --time 5 --buffer-size 500
```

Scripted test capture (stop by creating a file):
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 --saleae \
    --spi 0,1,2,3 --samplerate 25M --stop-file /tmp/stop_capture > results.txt &

# ... run your test ...

touch /tmp/stop_capture
wait
cat results.txt
```

The script removes any stale stop file on startup, polls every 250ms, and cleans up the file after detecting it.

### sigrok backend

Live capture from Saleae Logic 8:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d saleae-logic-pro \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS \
    --spi SCLK,MISO,MOSI,nSS --samplerate 25M --continuous
```

Live capture from fx2lafw device:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d fx2lafw \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS \
    --spi SCLK,MISO,MOSI,nSS --samplerate 4M --continuous
```

Dual SPI port with sigrok (with the deglitch transform recommended for
two 10 MHz buses at 25 MSa/s — see [Deglitch transform](#deglitch-transform-marginal-sample-rates)):
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d saleae-logic-pro \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS,4=SCLK_B,5=MISO_B,6=MOSI_B,7=nSS_B \
    -T "deglitch:channels=SCLK,SCLK_B:clock_period=2.5:frame_pulses=8" \
    --spi SCLK,MISO,MOSI,nSS --spi SCLK_B,MISO_B,MOSI_B,nSS_B \
    --samplerate 25M --continuous
```

From a raw binary file:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 \
    -i capture.bin -I binary:numchannels=4:samplerate=1000000 \
    --spi 0,1,2,3
```

### Decode engines: `--engine numpy` (default) vs `--engine srd`

| Engine | Decoder | Throughput | Use |
|--------|---------|-----------|-----|
| `numpy` (default) | `fast_spi.py`, vectorized (numba scan) | ~100-130 MB/s at planned 12-ch load; see [sustained decode rate](#dslogic-capture-first-decode-after) | live capture, large files |
| `srd` | libsigrokdecode SPI PD | ~1.3 MB/s (~20x slower than real time) | reference / cross-check |

The `numpy` engine reads sigrok-cli's raw sample stream (`-O binary`) and
decodes SPI with NumPy over whole chunks, so **libsigrokdecode does no work at
all** — sigrok-cli runs with no protocol decoder. The libsigrok `-T` transform
is unaffected (it runs in C, upstream of the decode) and should still be used.

The two engines produce the same output, with one deliberate difference: the
srd path emits a garbage decode for a transfer already in progress at sample 0
(a fragment with no valid framing, which is where the reference capture's lone
`dict-error` came from); the numpy engine ignores a transaction with no
observed CS falling edge.

Use `--engine srd` if you need to cross-check a decode against libsigrokdecode,
or for a capture the numpy engine cannot describe (>8 channels, non-8-bit
words, LSB-first).

### Logging interrupt and other pins

Channels that are not part of an SPI port can be followed alongside the
decoded traffic, so you can see exactly where an interrupt or BUSY line
asserted relative to the commands around it. A typical use is disconnecting
one SPI bus and moving those wires to the pins of interest:

```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d saleae-logic-pro \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS,4=INT,5=BUSY \
    -T "deglitch:channels=SCLK:clock_period=2.5:frame_pulses=8" \
    --spi SCLK,MISO,MOSI,nSS --int-pin INT --extra-pin BUSY \
    --samplerate 25M --continuous
```

Pin names come from `-C` (case-insensitive); a bare channel number works too.
`--extra-pin` is repeatable. Output interleaves by timestamp, so a whole
interrupt handshake reads top to bottom (real capture, LR2021 TX path):

```
0.934523880: WriteRadioTxFifo 511 bytes (mode=STBY_XOSC, reset=extPin, CMD_OK)
0.934955440: SetTx timeout-disabled (mode=STBY_XOSC, reset=extPin, CMD_OK)
0.936766600: [INT] rising
0.936788880: GetAndClearIrqStatus (request) (intActive mode=FS, reset=extPin, CMD_OK)
0.936800360: [INT] falling
0.936809640: GetAndClearIrqStatus (response) (TX_TIMESTAMP | TX_DONE  mode=FS, ...)
```

TX is submitted, the IRQ asserts 1.8 ms later, firmware responds 22 us after
that, the pin drops as the handler reads the status, and the response confirms
`TX_DONE`. (With a single `--spi` port the `[SPI]` prefix is omitted, as above;
with two or more, decoded lines carry `[SPI]` / `[SPI_B]` and pin lines carry
the pin name.)

Both flags need the default `numpy` engine — libsigrokdecode only reports
protocol-decoder output, so `--engine srd` rejects them.

Output is fully chronological: a chunk's transactions and pin edges are merged
by timestamp, and printing is held back briefly so a transaction spanning a
chunk boundary settles first (an HLA timestamps a result at the *start* of a
transaction but only returns it at the end). Measured 0 out-of-order lines on
the 3 s reference capture, against 21 for the `srd` engine.

Note that a pin pointed at a fast signal (e.g. a still-connected second SPI
clock) emits an edge event per transition and will swamp the output — pointing
`--int-pin` at an SPI clock produced 155,000 events in 12 s. These flags are
meant for low-rate control lines; a real interrupt line in the same setup gave
40 edges in 15 s.

### High-throughput traffic

With `--engine srd`, live decode cannot keep up with saturated dual-SPI
traffic: the pipeline back-pressures sigrok-cli, USB transfer resubmission
falls behind, and the FX2 overruns — whole chunks are lost and the channel
demux rotates, so clock bitstreams land on the wrong channels. The symptom is
a flood of unparseable `[sigrok] N-N spi-X:` lines (empty CS transfers 1-2
samples or whole 16384-sample chunks wide). This is a throughput limit, not a
decode bug, and it happens with or without `-T`.

**The `numpy` engine (the default) is the fix** — it sustains ~36 MB/s against
the 25 MB/s capture rate, so live `--continuous` decode of burst traffic works
directly. Measured on 20 s of live iperf3 burst traffic, same command and same
`-T deglitch` on both:

| | numpy | srd |
|---|---|---|
| transactions decoded | 145,313 | 9,207 |
| timespan covered (of 20 s) | 19.2 s | 1.4 s |
| 511-byte FIFO transfers | 14,285 | 572 |
| CMD_FAIL / xferLen1 / dict-error | 0 / 0 / 0 | 551 / 398 / 267 |

numpy's 7,586 transactions/s matches the offline reference rate across the full
window; srd covered only the first 1.4 s before falling irrecoverably behind,
and corrupted even that.

The capture-first/decode-after workflow below remains useful when you want to
archive the raw samples, re-decode with different options, or run on a slower
machine:

```bash
# 1. Capture raw while the burst runs (Logic 8 has no sample limit;
#    'q' on stdin stops a continuous capture)
(sleep 20; echo q) | sigrok-cli \
    -d saleae-logic-pro \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS,4=SCLK_B,5=MISO_B,6=MOSI_B,7=nSS_B \
    --config samplerate=25m --continuous -o burst.bin -O binary

# 2. Decode offline: same sigrok_hla.py command, -i instead of -d
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 \
    -i burst.bin -I binary:numchannels=8:samplerate=25000000 \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS,4=SCLK_B,5=MISO_B,6=MOSI_B,7=nSS_B \
    -T "deglitch:channels=SCLK,SCLK_B:clock_period=2.5:frame_pulses=8" \
    --spi SCLK,MISO,MOSI,nSS --spi SCLK_B,MISO_B,MOSI_B,nSS_B \
    --samplerate 25M
```

Note: at 25 MSa/s x 8 channels the raw file grows at 25 MB/s (~500 MB per
20 s). The `-o` capture writes a plain 1-byte-per-sample stream readable by
`-I binary:numchannels=8`.

## Options

### Common options

| Option | Description |
|--------|-------------|
| `--spi CLK,MISO,MOSI,CS` | SPI port definition (repeatable for multiple ports) |
| `--hla-path PATH` | Path to HLA directory (required) |
| `--samplerate RATE` | Sample rate, e.g., `25M`, `4M`, `1000000` |
| `--time DURATION` | Capture duration, e.g., `1`, `5s`, `100ms` |
| `--cpol {0,1}` | Clock polarity (default: 0) |
| `--cpha {0,1}` | Clock phase (default: 0) |
| `--hex` | Print MOSI/MISO hex bytes before each decoded transaction |

### Saleae-specific options

| Option | Description |
|--------|-------------|
| `--saleae` | Use Saleae Logic 2 automation backend |
| `--saleae-port PORT` | Automation server port (default: 10430) |
| `--buffer-size MB` | Capture buffer size limit in MB |
| `--stop-file PATH` | Stop capture when this file appears (for scripted tests) |

### sigrok-specific options

| Option | Description |
|--------|-------------|
| `-d DRIVER` | sigrok driver (e.g., `saleae-logic-pro`, `fx2lafw`, `saleae-logic16`) |
| `-i FILE` | Input file instead of live capture |
| `-I FORMAT` | Input format (e.g., `binary:numchannels=4:samplerate=1000000`) |
| `-C CHANNELS` | Channel list (e.g., `0=SCLK,1=MISO,2=MOSI,3=nSS`) |
| `--samples N` | Number of samples to capture |
| `--continuous` | Continuous streaming capture |
| `-T MODULE[:OPT=VAL...]` | libsigrok transform module applied to the sample stream before decoding (see [Deglitch transform](#deglitch-transform-marginal-sample-rates)) |
| `--engine numpy\|srd` | SPI decode engine (default `numpy`, real-time capable; see [Decode engines](#decode-engines---engine-numpy-default-vs---engine-srd)) |
| `--int-pin NAME` | Log transitions of an interrupt pin, interleaved with decoded traffic (numpy engine only) |
| `--extra-pin NAME` | Log transitions of an extra pin (repeatable; numpy engine only) |

## Sample Output

```
0.001398000: [SPI] wakeup 1.2250000000000108e-05
0.001434750: [SPI_B] FIFO_RX | PREAMBLE_DETECTED | SYNC_WORD_HEADER_VALID ...
0.001545250: [SPI] 0x3dbe, dict-error:KeyError(15806) (mode=SLEEP, reset=cleared, CMD_FAIL)
0.002005250: [SPI_B] wakeup 1.1499999999999792e-05
```

With `--hex`:
```
  [SPI] MOSI: 84 00
  [SPI] MISO: 00 00
0.001545250: [SPI] SetSleep WARM, 0 (mode=STBY_RC, reset=NA, CMD_OK)
```

## Sample Rate Selection

The sample rate must be high enough to capture the SPI clock. For an 8 MHz SPI SCLK, use at least 25 MSa/s (the Nyquist minimum is 16 MSa/s, but oversampling is needed for reliable decoding).

| SPI SCLK | Minimum sample rate | Recommended |
|----------|-------------------|-------------|
| 1 MHz | 4 MSa/s | 4-10 MSa/s |
| 4 MHz | 10 MSa/s | 10-25 MSa/s |
| 8 MHz | 25 MSa/s | 25 MSa/s |
| 10 MHz | 25 MSa/s | 25 MSa/s + `-T deglitch` (see below) |

## Deglitch transform (marginal sample rates)

With both SPI buses captured on the Logic 8, all 8 channels are active and the
FX2 caps out at 25 MSa/s — only 2.5 samples per 10 MHz SPI clock period. In
this regime a one-sample clock phase can collapse to zero width whenever the
signal edges drift into alignment with the sample instants, silently deleting a
clock cycle and bit-slipping the rest of the transfer (symptoms: empty
`CMD_FAIL` transactions, `xferLen1`, `dict-error` garbage opcodes, clustered in
high-throughput bursts). Logic 2 tolerates the same data; sigrok's sample-based
decoder does not.

The libsigrok `deglitch` transform repairs this before the SPI decoder runs.
**Recommended options for the dual 10 MHz SPI / 25 MSa/s use case:**

```
-T "deglitch:channels=SCLK,SCLK_B:clock_period=2.5:frame_pulses=8"
```

- `channels=SCLK,SCLK_B` — apply only to the two clock lines (names as
  assigned by `-C`; indices `0,4` also work)
- `clock_period=2.5` — nominal clock period in samples (25 MSa/s / 10 MHz);
  enables splitting of merged pulses (a high run longer than one half-period
  is provably two pulses whose separating low phase collapsed)
- `frame_pulses=8` — SPI bytes carry 8 clock pulses; pulses are counted
  between idle gaps and a vanished pulse is re-inserted where the count comes
  up short of a multiple of 8
- `min_period=N` (not needed here) — classic glitch suppression for spurious
  pulses shorter than N samples; only useful at ≥4 samples/period

Validated results: on a worst-case-alignment stream the transform restores
98.6% of vanished clock edges and cuts protocol-level decode failures by ~95%;
on a clean capture (128M pulses) its output is byte-identical to its input, so
it is safe to leave enabled permanently — it only acts when the alignment
actually degrades.

Requirements: libsigrok with the `deglitch` module (commit `c09d7129`+) and,
for file-input runs (`-i`), sigrok-cli with the `-T`-on-file-input fix
(commit `54eaaf5`+). Both are in the local source builds under `/mnt/foo/`.
Note the transform delays the stream by a small look-ahead window and drops
its final few samples (≤ ~10) at end of capture.

For sustained burst traffic, decode from a recorded file rather than live —
see [High-throughput traffic: capture first, decode after](#high-throughput-traffic-capture-first-decode-after).

### With `--dslogic`

The DSLogic path has no sigrok-cli, so `-T deglitch:...` runs in-process:
`deglitch.py` is a streaming port of the same transform, applied to the
physical sample array before SPI decode and pin logging. Same syntax and
options, except that `channels=` is required (name the SPI clocks; the C
default of every logic channel is not offered). Channel names resolve through
`-C`, and numbers are physical bits:

```
-T "deglitch:channels=0,8:clock_period=2.5:frame_pulses=8"
```

Its output and counters match libsigrok's transform byte for byte (oracle
tests against the patched sigrok-cli on uint8 and uint16 data, both sampling
phases, random chunk splits). Per-clock counters are printed to stderr at the
end. It applies at replay too (`--dslogic -i FILE -T ...`); `--raw-out` refuses
`-T`, so the recorded file is always the unmodified producer stream. Lookahead
at these options is 14 samples (560 ns), which are dropped at end of stream as
in C. It needs numba (about 690 MB/s alone; the NumPy fallback runs at about
6 MB/s and warns). With it, the 12-channel planned-load decode runs at
79-85 MB/s with `--hex`, and the output on clean data is identical with zero
corrections.

At 25 MSa/s a sample is 40 ns, and each 10 MHz SCLK phase is nominally 50 ns.
On synthetic traffic without deglitch, decode is byte-exact down to a 41% duty
cycle and starts slipping bits at 40%, where a phase shrinks to one sample.
With deglitch at 40% the corrupted transfers fall from 59 to 8 of 400; at 39%
they fall from 321 to 127 of 400, so many remain wrong: the framing is
restored, but data at a lost pulse cannot always be recovered. Treat deglitch as a mitigation; the margin
that matters is the effective SCLK duty at the analyzer threshold. Whether
the DSLogic sees 10 MHz SCLK with enough margin is not yet validated live.

## Buffer and Memory

At 25 MSa/s with 8 digital channels, Logic 2 consumes significant memory. This is the same limitation as when using the Logic 2 GUI manually. The `--buffer-size` option can cap memory usage:

```bash
--buffer-size 500   # limit to 500 MB
```

To reduce memory usage:
- Only enable channels you need (single port = 4 channels instead of 8)
- Use shorter capture durations
- Lower the sample rate if your SPI clock allows it

## How It Works

### Saleae backend flow

1. Connects to Logic 2 automation API via gRPC
2. Configures and starts a capture on the Logic 8 hardware
3. Waits for capture to complete (timed) or for Ctrl-C (manual)
4. Adds Saleae's built-in SPI analyzer to the capture (one per port)
5. Exports the SPI data table to a temporary CSV
6. Parses the CSV rows (enable/result/disable frames) and feeds them to the HLA
7. Prints decoded output interleaved by timestamp

### sigrok backend flow

With `--engine numpy` (default):

1. Launches `sigrok-cli` as a subprocess emitting raw samples (`-O binary`,
   plus the `-T` transform if given) — no protocol decoder is instantiated
2. A reader thread feeds 4 MiB chunks through a deep queue, so a GC pause
   never back-pressures the capture
3. `fast_spi.py` decodes each chunk with NumPy: nSS edges frame transactions,
   CPOL/CPHA select the SCLK edges that latch MISO/MOSI, and a Numba kernel
   packs bits to bytes; state carries across chunk boundaries
4. Feeds AnalyzerFrame objects to the HLA in real time
5. Prints decoded output interleaved by timestamp

With `--engine srd`, steps 1-3 are replaced by sigrok-cli's SPI protocol
decoder and per-transfer annotation parsing (one annotation per CS#-asserted
transfer, paired MISO/MOSI by sample range).

## Comparison with spi_hla.py

| Feature | spi_hla.py | sigrok_hla.py |
|---------|-----------|---------------|
| Input | Saved binary exports | Live USB capture or files |
| SPI decoding | Custom NumPy/Numba decoder | Saleae or sigrok built-in |
| Speed | Very fast (vectorized) | Depends on backend |
| Pin logging | `--int-pin`, `--extra-pin` | `--int-pin`, `--extra-pin` (numpy engine) |
| Workflow | Export from Logic, then run | One command, captures and decodes |
