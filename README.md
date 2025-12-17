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
