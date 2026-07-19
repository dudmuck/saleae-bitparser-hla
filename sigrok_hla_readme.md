# sigrok_hla.py

Live SPI capture with HLA decoding, directly from USB hardware. An alternative to `spi_hla.py` (which processes saved binary exports).

## Backends

Two capture backends are supported:

| Backend | Flag | Hardware | How it works |
|---------|------|----------|--------------|
| **Saleae** | `--saleae` | Saleae Logic 8, Pro 8, Pro 16 | Connects to Logic 2 app via automation API (gRPC on port 10430) |
| **sigrok** | `-d DRIVER` | sigrok-supported analyzers | Runs sigrok-cli as subprocess with SPI protocol decoder |

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

### Both backends

- A Saleae High Level Analyzer (HLA) directory containing `HighLevelAnalyzer.py`
- Python 3

## Pin Layout

The default pin mapping (matching `bw1/digital.csv`):

| Channel | 0    | 1    | 2    | 3   | 4      | 5      | 6      | 7     |
|---------|------|------|------|-----|--------|--------|--------|-------|
| Signal  | SCLK | MISO | MOSI | nSS | SCLK_B | MISO_B | MOSI_B | nSS_B |

Specified as `--spi CLK,MISO,MOSI,CS` (channel numbers for Saleae, channel names for sigrok).

## Usage

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

### High-throughput traffic: capture first, decode after

Live `--continuous` decode is fine for light traffic (status polling), but the
Python decode pipeline runs roughly 20x slower than real time on saturated
dual-SPI data. During sustained bursts (e.g. back-to-back 511-byte FIFO
transfers) the pipeline back-pressures sigrok-cli, USB transfer resubmission
falls behind, and the FX2 overruns — whole chunks are lost and the channel
demux rotates, so clock bitstreams land on the wrong channels. The symptom is
a flood of unparseable `[sigrok] N-N spi-X:` lines (empty CS transfers 1-2
samples or whole 16384-sample chunks wide). This is a USB/CPU throughput
limit, not a decode bug, and it happens with or without `-T`.

For burst captures, record raw first — capture-to-file keeps up easily — and
decode offline:

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

1. Launches `sigrok-cli` as a subprocess with SPI protocol decoder(s) (and the
   `-T` transform, if given, applied to the sample stream before decoding)
2. Streams and parses the SPI decoder's per-transfer annotations line-by-line
   (one annotation per real CS#-asserted transfer, with sample numbers)
3. Pairs MISO/MOSI transfers by sample range
4. Feeds AnalyzerFrame objects to the HLA in real-time
5. Prints decoded output interleaved by timestamp

## Comparison with spi_hla.py

| Feature | spi_hla.py | sigrok_hla.py |
|---------|-----------|---------------|
| Input | Saved binary exports | Live USB capture or files |
| SPI decoding | Custom NumPy/Numba decoder | Saleae or sigrok built-in |
| Speed | Very fast (vectorized) | Depends on backend |
| Pin logging | `--int-pin`, `--extra-pin` | Not supported |
| Workflow | Export from Logic, then run | One command, captures and decodes |
