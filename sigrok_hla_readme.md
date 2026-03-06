# sigrok_hla.py

Live SPI capture with HLA decoding, directly from USB hardware. An alternative to `spi_hla.py` (which processes saved binary exports).

## Backends

Two capture backends are supported:

| Backend | Flag | Hardware | How it works |
|---------|------|----------|--------------|
| **Saleae** | `--saleae` | Saleae Logic 8, Pro 8, Pro 16 | Connects to Logic 2 app via automation API (gRPC on port 10430) |
| **sigrok** | `-d DRIVER` | fx2lafw-compatible analyzers | Runs sigrok-cli as subprocess with SPI protocol decoder |

The **Saleae backend** is the primary one. The Saleae Logic 8 (21a9:1004) is not supported by sigrok, so the automation API is used instead. Logic 2 handles the USB communication and SPI decoding, then the script exports the decoded data and feeds it through the HLA.

## Prerequisites

### Saleae backend

- **Saleae Logic 2** desktop app running
- Automation server enabled: Edit -> Settings -> check "Enable Automation Server" (default port 10430)
- Python package: `pip install logic2-automation`

### sigrok backend

- `sigrok-cli` installed (`sudo apt install sigrok sigrok-firmware-fx2lafw`)
- Hardware supported by sigrok (fx2lafw, saleae-logic16, etc.)

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

Live capture from sigrok-supported device:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d fx2lafw \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS \
    --spi SCLK,MISO,MOSI,nSS --samplerate 4M --continuous
```

Dual SPI port with sigrok:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 -d fx2lafw \
    -C 0=SCLK,1=MISO,2=MOSI,3=nSS,4=SCLK_B,5=MISO_B,6=MOSI_B,7=nSS_B \
    --spi SCLK,MISO,MOSI,nSS --spi SCLK_B,MISO_B,MOSI_B,nSS_B \
    --samplerate 4M --continuous
```

From a raw binary file:
```bash
./sigrok_hla.py --hla-path ~/HLA/saleae_lr2021 \
    -i capture.bin -I binary:numchannels=4:samplerate=1000000 \
    --spi 0,1,2,3
```

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
| `-d DRIVER` | sigrok driver (e.g., `fx2lafw`, `saleae-logic16`) |
| `-i FILE` | Input file instead of live capture |
| `-I FORMAT` | Input format (e.g., `binary:numchannels=4:samplerate=1000000`) |
| `-C CHANNELS` | Channel list (e.g., `0=SCLK,1=MISO,2=MOSI,3=nSS`) |
| `--samples N` | Number of samples to capture |
| `--continuous` | Continuous streaming capture |

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

1. Launches `sigrok-cli` as a subprocess with SPI protocol decoder(s)
2. Streams and parses annotation output line-by-line (MISO/MOSI data with sample numbers)
3. Pairs MISO/MOSI by sample range, detects CS transaction boundaries from gaps
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
