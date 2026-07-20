# Plan: >8 channel support in the numpy decode engine

Status: proposal, 2026-07-20. Goal: decode two SPI ports **plus** interrupt /
status pins simultaneously on a logic analyzer with more than 8 channels
(e.g. Saleae Logic Pro 16, or a 16-channel fx2lafw device). Today that needs
10-12 channels and the numpy engine caps out at 8.

## Current state

`fast_spi.py` reads sigrok-cli's `-O binary` stream as `uint8` and extracts a
channel with `(chunk >> ch) & 1`, so channel indices above 7 are impossible;
`resolve_channel_indices()` in `sigrok_hla.py` rejects them explicitly. The
8-channel Logic 8 forces a choice today: two SPI ports **or** one port plus
pins, never both.

## What already works, and does not need touching

- **The `deglitch` transform.** `src/transform/deglitch.c` was written
  unitsize-aware: a 64-bit channel mask, `get_bit()`/`put_bit()` addressing
  `frame[c >> 3]`, `unitsize` latched from the first `SR_DF_LOGIC` packet, and
  a `(c >> 3) >= unitsize` bounds check. A 16-channel stream needs no change.
- **The wire format.** `src/output/binary.c:40` dumps `logic->data` verbatim,
  so a wide capture is simply `unitsize` bytes per sample, little-endian,
  channel N = bit N. No framing, no escaping.
- **Everything downstream of the decoder** — HLA feeding, the ordering
  watermark, pin logging, `--hex` — is channel-count agnostic.

## The one hard problem: knowing the sample width

The binary stream does not carry `unitsize`, and the `META` line gives only the
samplerate. Worse, drivers disagree on how the width is chosen:

| Driver | Rule | Consequence |
|---|---|---|
| `fx2lafw` | `sample_wide = channel_mask > 0xff \|\| num_analog > 0` (`protocol.c:568`) | width follows the **enabled** channels: naming only D0-D7 with `-C` gives 1 byte even on a 16-channel device |
| `saleae-logic-pro` | `devc->unit_size = model->unit_size`, fixed per model | a Logic Pro 16 emits **2 bytes always**, even when decoding only channels 0-3 |

So inferring the width from "highest channel I was asked to decode" is correct
for fx2lafw and wrong for a Logic Pro 16 used on a few low channels. Guessing
wrong does not fail loudly — it silently decodes interleaved garbage.

### Resolution

1. **Infer** a default: `width = (max referenced channel index) // 8 + 1`,
   where "referenced" spans all `--spi` roles and all `--int-pin`/`--extra-pin`
   channels. For file input, prefer `numchannels` parsed from
   `-I binary:numchannels=N` (`(N + 7) // 8`), which is authoritative — the
   binary input module derives its unitsize the same way
   (`src/input/binary.c:64`).
2. **Allow an explicit override**: `--sample-width {1,2}` (bytes per sample),
   which wins over inference. Required for a Logic Pro 16 decoding only low
   channels.
3. **Validate loudly**, because a wrong width is otherwise silent:
   - error if the total byte count is not a multiple of the width;
   - after the first chunk, warn if any *named* channel shows zero transitions
     (a width mismatch shuffles bits, typically flatlining some channels), or
     if a CS line is asserted for an implausible fraction of the window;
   - print the resolved width and its source (inferred vs explicit) at startup,
     next to the existing per-port bit mapping.

## Implementation

### `fast_spi.py`

- `SpiPortDecoder`, `PinLogger`, `MultiPortDecoder` take a `sample_width`
  (default 1). Where they currently do `np.frombuffer(raw, uint8)`, view the
  buffer as `<u2` when width is 2: `np.frombuffer(raw, dtype='<u2')`. All the
  `(chunk >> bit) & 1` extraction, edge finding, and CS framing then work
  unchanged on a uint16 array.
- **Odd-byte carry.** A pipe read can end mid-sample. Keep a `bytes` remainder
  in `MultiPortDecoder.feed()`, prepend it to the next chunk, and only convert
  a whole number of samples. (The 4 MiB reads are even-sized, but a short read
  at end of stream or a partial write is not — this is the one genuinely new
  failure mode.)
- `prev_sample` is stored as `int(chunk[-1])`, which already works for uint16.
- Keep `MAX_OPEN_BITS` and the `_BYTE` intern table as they are.

### `sigrok_hla.py`

- `--sample-width {1,2}` argparse option (default: infer).
- `resolve_channel_indices()`: bound indices by `sample_width * 8 - 1` instead
  of the hardcoded 7; report the offending flag name in the error as it does
  now.
- `run_sigrok_numpy()`: resolve the width, log it, pass it to the decoder and
  the pin logger, and run the sanity checks above.
- The `-C` string must name every channel the device should enable, since
  sigrok-cli enables exactly the listed channels — document that naming the
  extra channels is what makes fx2lafw switch to wide sampling.

### Docs

- `sigrok_hla_readme.md`: a "More than 8 channels" subsection with a worked
  two-ports-plus-two-interrupts example, the per-driver width table above, and
  the `--sample-width` override.

## Validation

1. **Synthetic 16-channel** (no hardware needed, and the main correctness
   gate): extend the generator already used for the 8-channel unit test to
   emit `<u2` samples with SPI A on channels 0-3, SPI B on 8-11, and pins on
   12-13. Assert decoded bytes are exact and that randomized chunk splits —
   including odd byte offsets — match a whole-buffer decode.
2. **Regression**: re-run the 8-channel equivalence check against
   `dual_slice_mid.bin`; output must stay byte-identical to today's
   (`mid_ordered3.txt`), and throughput within ~5%.
3. **Width-mismatch behaviour**: feed 8-channel data while claiming
   `--sample-width 2` and confirm the validation fires instead of emitting
   garbage.
4. **Hardware** (requires a 16-channel analyzer, not currently on hand):
   two SPI ports plus two interrupt pins live; check transaction rate against
   the same buses captured 4-channels-at-a-time on the Logic 8, and confirm
   the interrupt/command causal ordering (as in the Logic 8 test where 20/20
   INT rising edges preceded a status read). **Note the Logic Pro 16's fixed
   2-byte width is exactly the case most likely to surprise — treat the first
   hardware run as the real test of the width logic.**

## Scope and risk

~30-40 lines across the two files plus the synthetic test. Low risk to the
existing 8-channel path (width 1 keeps the current code path; the uint16 view
and odd-byte carry are only exercised when width is 2). The residual risk is
entirely in width resolution, which is why the validation above is weighted
toward detecting a mismatch rather than assuming one cannot happen.

## Not in scope

- Analog channels (fx2lafw sets `sample_wide` when analog is enabled, which
  would change the stream layout in ways this plan does not model).
- More than 16 channels: the code would generalise to `<u4` for 17-32
  channels, but no such device is in use here and the extra width would go
  untested.
- The `srd` engine, which has no channel-count limit of its own and remains
  available for cross-checking.
