# spi_hla.py

SPI decoder for Saleae Logic 2 binary exports that feeds data to a High Level Analyzer.

## Description

In Logic we can get the decoding from high level analyzer using "Export Table" under "Analyzers" with the menu available on the right side of the Data search box. But this script offers additional features, such as logging the activity of another pin alongside the decoder output strings. For example, when a debug pin is asserted or an interrupt pin.

this was based off the project at https://github.com/znuh/saleae-binparser

## Exporting Data from Logic

1. **Export binary data:** File -> Export Raw Data. For Export Format, select binary and choose time range of "all time" or "visible screen".

2. **Export channel names:** File -> Export Raw Data, select CSV Export Format and use time range "Visible Screen" because we only need the channel names from the header of this CSV. Save as `digital.csv` in the same directory as the binary files.

## Requirements

- Python 3
- NumPy (`pip install numpy`)
- A Saleae High Level Analyzer (HLA) directory

## Usage

```
python3 spi_hla.py <directory> --hla-path <path-to-hla>
```

### Options

| Option | Description |
|--------|-------------|
| `directory` | Directory containing digital_N.bin files |
| `--hla-path PATH` | Path to HLA directory (required) |
| `--sclk N` | SCLK channel number (auto-detect from CSV if not specified) |
| `--miso N` | MISO channel number (auto-detect from CSV if not specified) |
| `--mosi N` | MOSI channel number (auto-detect from CSV if not specified) |
| `--nss N` | nSS channel number (auto-detect from CSV if not specified) |
| `--cpol {0,1}` | Clock polarity (default: 0) |
| `--cpha {0,1}` | Clock phase (default: 0) |
| `--hex` | Print MOSI/MISO bytes in hex before each decoded line |
| `--int-pin NAME` | Log transitions of interrupt pin (e.g., `--int-pin int`) |
| `--extra-pin NAME` | Log transitions of an extra pin (e.g., `--extra-pin busy`) |

### Examples

Basic usage:
```
python3 spi_hla.py spislave_tx --hla-path ~/HLA/saleae_spislave
```

With hex output and interrupt pin logging:
```
python3 spi_hla.py spislave_tx --hla-path ~/HLA/saleae_spislave --hex --int-pin int
```

With multiple pin logging:
```
python3 spi_hla.py spislave_tx --hla-path ~/HLA/saleae_spislave --int-pin int --extra-pin busy
```

### Sample Output

```
0.977247075: GetVersion (request) (mode=STBY_RC, reset=extPin, CMD_OK)
0.977275725: GetVersion v1.24 (mode=STBY_RC, reset=extPin, CMD_DAT)
0.977312275: SetRegMode SIMO_TX_ONLY (mode=STBY_RC, reset=extPin, CMD_OK)
0.977532975: *** INT falling edge ***
0.977553525: SetDioFunction DIO9, DIO_FUNCTION_NONE, DRIVE_NONE ...
```

With `--hex`:
```
  MOSI: 01 01
  MISO: 04 21
0.977247075: GetVersion (request) (mode=STBY_RC, reset=extPin, CMD_OK)
```

## Multiple SPI Ports

The script automatically detects multiple SPI ports from the `digital.csv` header. It looks for signal names with common suffixes like `_B`, `_C`, `_2`, etc.

For example, with a CSV header:
```
Time [s],SCLK,MISO,MOSI,nSS,sclk_b,miso_b,mosi_b,nss_b
```

The script detects two ports:
- **SPI**: SCLK, MISO, MOSI, nSS (channels 0-3)
- **SPI_B**: sclk_b, miso_b, mosi_b, nss_b (channels 4-7)

Each port gets its own HLA instance, and results from all ports are interleaved by timestamp. When multiple ports are present, output lines are prefixed with the port name:

```
2.108120280: [SPI] SetSleep WARM, 0 (mode=STBY_RC, reset=NA, CMD_OK)
2.108138498: [SPI_B] SetSleep WARM, 0 (mode=STBY_RC, reset=NA, CMD_OK)
2.108181213: [SPI] SetStandby STBY_XOSC (mode=SLEEP, reset=NA, CMD_OK)
2.108198963: [SPI_B] SetStandby STBY_XOSC (mode=SLEEP, reset=NA, CMD_OK)
```

With `--hex`, the port prefix also appears on the hex lines:
```
  [SPI] MOSI: 84 00
  [SPI] MISO: 00 00
2.108120280: [SPI] SetSleep WARM, 0 (mode=STBY_RC, reset=NA, CMD_OK)
```

When only one SPI port is detected, no prefix is shown (backwards compatible).
