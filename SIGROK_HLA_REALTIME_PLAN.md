# sigrok_hla.py real-time decoding plan

Status: proposal, 2026-07-19. Goal: live dual-SPI decode at 25 MSa/s x 8
channels (25 MB/s) keeping up with capture, eliminating both the ~20x
real-time deficit and the USB-overrun corruption it causes.

## Where the time goes today

Pipeline: `saleae-logic-pro` driver (C) → optional `deglitch` transform (C) →
**libsigrokdecode SPI PD (pure Python)** → annotation text → harness regex →
HLA (Python).

Measured: a 19.4 s / 484 MB capture takes ~6.5 min to decode (~20x slower
than real time). The dominant cost is the libsigrokdecode SPI decoder: it
processes one clock edge per Python `wait()` iteration — ~64 M SCLK edges per
bus in that capture — with per-edge Python/C transitions. The harness's own
parsing is negligible (per-transfer annotation lines only), and the HLA sees
at most ~1.6 k transactions/s during iperf3 bursts (fine for Python).

Consequence of being slow live: the blocked pipe back-pressures sigrok-cli,
USB transfer resubmission falls behind, the FX2 overruns, and the channel
demux rotates — the `[sigrok] N-N spi-X:` empty-transfer flood. Real-time
decode fixes correctness, not just latency.

## Why real-time is achievable (measured)

- NumPy edge extraction on the same 484 MB capture (bit unpack, diff,
  flatnonzero, searchsorted per channel) runs in seconds — >50x the 25 MB/s
  capture rate, leaving ample headroom for a few passes per chunk.
- `spi_hla.py` already proves the decode math vectorizes: precomputed
  sample-edge bit arrays via one `searchsorted`, Numba `bits_to_bytes`
  kernel. It decodes whole captures in seconds; it just reads Saleae
  transition exports instead of a live stream.
- HLA cost: ~1.6 k transactions/s peak x ~50-100 us each ≈ 10-15% of one
  core.

## Recommended approach: vectorized streaming engine

Keep sigrok-cli as the capture front-end (driver + `-T deglitch` both stay,
both are C and fast). Replace only the SPI-decode stage:

```
sigrok-cli -d saleae-logic-pro -C ... -T "deglitch:..." --continuous \
    -o /dev/stdout -O binary
  | sigrok_hla.py  (new numpy engine: chunk → bits → transfers → HLA)
```

### Phase 1 — streaming SPI engine (new module, e.g. `fast_spi.py`)

Per ~4 MB chunk of the 1-byte-per-sample stream (unitsize 1, bit i =
channel i):

1. Strip the leading `META samplerate: N\n` text line that `-O binary`
   emits on stdout (first chunk only).
2. Per SPI port: extract SCLK/MISO/MOSI/nSS bit arrays (`(chunk >> ch) & 1`).
3. nSS falling/rising edges frame transactions (`np.diff`/`flatnonzero`).
4. SCLK sample edges per CPOL/CPHA (`sample_on_rising = (cpol == cpha)`,
   as in `spi_hla.py`); latch MISO/MOSI values at those edge indices
   directly (sample-domain indexing — no searchsorted needed).
5. Group bits per CS window into bytes with the existing Numba
   `bits_to_bytes` kernel (import from / share with `spi_hla.py`).
6. Emit the same enable/result/disable `AnalyzerFrame` sequence the HLA
   already consumes (reuse `_load_hla_class` / `_format_hla_result` /
   the timestamp-interleaved printer from the harness).

Chunk-boundary handling (the only tricky part):
- Carry an open transaction across chunks: buffer bits since the last nSS
  fall until the nSS rise arrives; cap the buffer (e.g. 1 M samples) to
  bound memory if CS sticks low.
- Carry the last partial byte's bits and the trailing channel state
  (1 sample) for edge detection continuity.

### Phase 2 — wire into sigrok_hla.py

- `--engine numpy|srd` option, default `numpy` for the sigrok backend when
  `--spi` ports are given; `srd` keeps the current libsigrokdecode path as
  a reference/fallback.
- The numpy engine consumes `-O binary` output; the srd path keeps `-P spi`.
  Build the sigrok-cli command accordingly.
- Reader thread with a large buffer between the pipe and the decoder so
  short Python GC pauses never back-pressure sigrok-cli.

### Phase 3 — validation

1. Offline equivalence: decode `dual_slice_mid.bin` (clean reference) with
   both engines → transfer byte streams must match exactly; then
   `cleaned_grid` + deglitch → equal or better than the srd path.
2. Live light traffic (status polling): both engines side by side, same
   transaction stream.
3. Live iperf3 burst (the real target): numpy engine must produce zero
   `[sigrok]` unparseable lines and zero empty-transfer flood for a full
   burst window, at <100% of one core (measure with `pidstat`).
4. Throughput margin test: decode a saved 484 MB capture from file and
   confirm ≥2x real-time end-to-end including HLA.

### Phase 4 (optional, later)

- Vectorized deglitch in Python (port of the libsigrok transform's rules)
  so the Saleae-export path (`spi_hla.py`) and file decodes can use it
  without sigrok-cli. The transform's split/mod-8 logic maps naturally to
  run-length arrays.
- Multiprocess chunk decode for offline files (split at CS boundaries) if
  even faster batch turnaround is wanted; streaming real-time makes this
  mostly moot.

## Constraints / notes

- Assumes wordsize 8, MSB-first (matches current PD usage and the LR2021
  HLA). CPOL/CPHA supported as in `spi_hla.py`.
- `-O binary` on stdout prepends the `META samplerate` text line (quirk);
  `-o file` does not. The chunk reader must handle both.
- The srd engine remains the arbiter for any decode disagreement during
  validation; keep it selectable.
- Not in scope: replacing the Saleae-backend flow (already fast via
  Logic 2's own analyzer + CSV export).

## Effort estimate

Phase 1+2: one focused session (the decode math already exists in
`spi_hla.py` and in the diagnostic scripts from the deglitch work);
Phase 3: one session with hardware + iperf3 window.
