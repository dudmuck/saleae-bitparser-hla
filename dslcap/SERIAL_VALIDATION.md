# Serial-trigger hardware validation

2026-10-07 PDT (2026-10-08 UTC for the traffic tests), implementation d6dfd2e.

**Register comparison PASS. Bit-order validation BLOCKED:** no positive serial
match was observed. This is not evidence for a different bit order and is not
Phase C hardware acceptance. Production code was not changed during testing.

Follow-up: real DSView also missed the same serial pattern while its simple
NSS control triggered naturally. See DSVIEW_SERIAL_RESPONSE.md for actual FPGA
header status evidence and restoration. The root cause remains unproven.

## Register comparison

Real DSView 1.3.2 GUI arm/cancel sessions used the auto-loaded device profile,
with USB bulk-OUT logging. All 372 bytes matched dslcap at each rate:

| Rate | Channels | Samples | Position | Result |
|---|---|---:|---:|---|
| 100 MS/s | CH0–CH7 | 1,000,448 | 10% | identical |
| 200 MS/s | CH0–CH7 | 1,000,448 | 10% | identical |
| 400 MS/s | CH0–CH3 | 2,000,896 | 10% | identical |

All used serial start=3:f, stop=3:r, clock=0:r, data=2, value=0x1c35,
bits=16, VTH=1.6, buffer mode, filter/RLE/loop/instant off and global stage0.
Stage3 remains unreplicated at the higher rates, as DSView specifies.
Full register hex, hashes, commands and comparison results: SERIAL_GOLDEN.json.
Reference task `dslcap-c-golden-20261007`, revision5, consumed/acknowledged.
Raw reference: /tmp/dslcap-c-golden; dslcap: /tmp/dslcap-c-register-live.

Evidence caveat: the DSView owner could not read back the saved AT-SPI widget
trees because its permission classifier denied that read. No bypass was used.
The reference rests on the seeded profile, profile-loaded/security-pass logs,
and actual GUI-generated EP2 bytes. DSView session/settings were restored and
USB released before dslcap captures. No real DSView *response to the waveform*
was tested; these reference captures establish register serialization only.

## Isolated traffic tests at 100 MS/s

The application owner used documented LR20xx WriteTxFifo (opcode 0x0002) on
pi133, existing mode0 SPI, requested8MHz (approximately7.8125MHz on the wire).
Each run sent one frame `00 02 00 XX YY 00`; four bytes entered an initially
empty TX FIFO. No SetTx/RF, radio configuration, reset or GPIO changes.

Each pattern first had a separate simple NSS-falling control capture, then a
serial capture with target fixed at0x1c35/16. Captures selected CH0–CH3 and
1,000,448 samples. All controls exited0, had exactly48 clocks and the expected
six MOSI bytes, one matching variant at bit offset24, and no other variant at
any bit alignment. No ambiguous edges or partial frames were found.

| Wire pattern | Meaning | NSS control | Serial target0x1c35 |
|---|---|---|---|
| 0x1c35 | intended | exact bytes, PASS | timeout, exit16 |
| 0xac38 | bit reverse | exact bytes, PASS | timeout, exit16 |
| 0x351c | byte swap | exact bytes, PASS | timeout, exit16 |
| 0x38ac | both | exact bytes, PASS | timeout, exit16 |

The first serial timeout was120s; the remaining ones were45s. A final positive
retry enabled CH0–CH7 to match the100M reference exactly. Its *actual test-arm*
372-byte image was logged and matched the DSView reference, yet it also timed
out after45s. Every serial run emitted only the samplerate META line, no trigger
header or sample payload. All nine owner emission windows fall strictly after
the corresponding observed arm and before process exit; owner operations
reported success. Exact manifests, hashes, decoder output and owner receipts:
SERIAL_VALIDATION.json; raw files /tmp/dslcap-c-live.

The failed serial runs did not yield simultaneous raw waveform evidence.
Separate controls plus owner receipts establish the test stimulus as far as
these artifacts permit. Cached hit0 is not a hardware event log and has no
hard freshness bound. No false bit-order conclusion follows from five misses.

## Independent checks and remaining work

dslcap_worker prepared an offline decoder with six passing synthetic tests for
all variants, non-byte-aligned matches, NSS reset, partial frames, metadata and
truncation. Lead reran those tests. The worker independently confirmed the first
control using direct edge extraction and NumPy packbits: target at bits24–39,
4.87us NSS setup before the first clock, 3.45us from target end to NSS release,
and at least six sampled100M ticks of MOSI setup/hold. These are sampled margins,
not analog guarantees. The source and guide agree with F/R/R and global stage0;
no serial clock limit was found in the inspected sections. Cause is unproven.
Temporary verifier/diagnosis: /tmp/dslcap-c-verify.

The final worker cross-check independently verified all four controls with a
direct NumPy bit oracle, all five exit16/samplerate-only results, and the exact
eight-channel retry's register image. It agrees with the evidence above and
does not assign a cause or a bit-order PASS. Report:
/tmp/dslcap-c-verify/final-check.md. All ten frozen implementation source hashes
remained unchanged. Lead verified no DSView/dslcap process or USB holder remains.

The100M positive-match gate must work before bit order can be accepted. Live
200M/400M bit-order and opcode/payload gates remain open. Useful next diagnosis
would compare a real DSView serial response on the same stimulus, then isolate
width and clock-rate dependence. Changing stage counts or reversing the value
without positive evidence would be speculative; no such fix was made.

## Restoration

Radio owner completed task `dslcap-c-radio-20261007`, revision5, consumed and
acknowledged. Final verification at approximately00:17:13Z restored the full
baseline and left mcp_radio idle. Per-run restoration verified
TX level0, mode STBY_XOSC, errors0, and original FIFO flags rx0x03/tx0x27.
Each write/clear sequence raised system FIFO_TX (0x2); only that newly raised
bit was cleared, restoring system IRQ0. Existing FIFO flags were preserved.
Pi134 was untouched. Wiring was unchanged; CH5 remains DIO8 and CH6 disconnected.
