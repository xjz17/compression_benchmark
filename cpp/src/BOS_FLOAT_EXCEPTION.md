# BOS float exception frontend (BFE1)

`bos_float_exception` is an experimental, lossless frontend for dense IEEE-754
`float32` and `float64` blocks. It does not replace the integer BOS core. The
caller supplies the real BOS `int64` encoder and decoder through callbacks, and
BFE1 applies those callbacks independently to:

1. decimal-scaled integers `q`;
2. gaps between repair positions; and
3. IEEE prediction-XOR repair masks.

The current public interface is declared in
[`bos_float_exception.hpp`](bos_float_exception.hpp):

```cpp
Bytes encode(std::span<const std::uint8_t> ieee_little_endian,
             DType dtype,
             const EncodeInt& encode_int,
             Stats* stats = nullptr,
             const EncodeOptions& options = {});

Bytes decode(std::span<const std::uint8_t> bfe1,
             const DecodeInt& decode_int,
             Stats* stats = nullptr);
```

Both input and decoded output are dense, little-endian IEEE words. A block may
contain at most 10,000 values. Missing-value positions and dataset metadata are
outside this interface. NaN payloads, infinities, negative zero, subnormals and
values that cannot be scaled into `int64` are retained by XOR repairs rather
than normalized or discarded.

## Precision selection and timing boundary

The default precision candidates are `{0,1,2,3,4,6}`. For blocks larger than
`pilot_values` (default 256), BFE1 uniformly samples values with
`floor(i*(N-1)/(m-1))`, including the first and last value. Every precision is
actually passed through the integer callback on the pilot. The smallest
complete pilot exception frame selects one precision, which is then encoded
once on the full block. Pilot raw fallback is deliberately ignored during this
choice. The selected full exception frame is finally compared with a complete
BFE1 raw frame.

`pilot_values=0` performs exhaustive full-block selection. A block no larger
than the pilot evaluates every full candidate and reuses the winning wire.
`fixed_precision` evaluates exactly one full candidate and skips the pilot.

A fair encoding measurement must surround the complete `encode` call. That
includes sampling, decimal parsing, all pilot BOS calls, selection, the full
BOS call and the full raw-fallback comparison. Rejected pilot bytes are search
work and therefore contribute time, but they are not stored in the selected
wire. Decode timing covers the selected frame only.

## What the BFE1 byte count includes

The BFE1 wire contains a 32-byte header with version, dtype, mode, precision,
value and exception counts, and all stream lengths. Its reported size includes
that header and every byte returned by the BOS callbacks for `q`, gaps and XOR
masks. Raw mode includes the same header and every original IEEE byte. No
unpriced precision, exception-position or repair metadata is kept outside the
wire.

## Required production container boundary

BFE1 is a compact inner payload. It does **not** contain the benchmark's NGF
graph description, graph/operator identifiers, missing bitmap, outer stream
lengths or checksum. Consequently, a bare BFE1 byte count is not directly
comparable with an existing complete NGF frame. Production comparisons must
put BFE1 and the baseline payload behind the same NGF outer container and
charge the same graph, routing, missing-bitmap, checksum and framing fields.
Only that uniformly wrapped size can be attributed to the exception frontend.

For a dense column, the NGF missing bitmap may be an empty bitmap, but its
standard representation must still follow the same container contract on both
arms. The production decoder should route through NGF and then invoke BFE1;
the `BFE1` magic alone is only the inner-payload discriminator.

## Portability and floating-point audit

- BFE1 reads and writes explicit little-endian words, so host byte order does
  not change the inner wire. It assumes 32-bit/64-bit IEC 60559 binary floating
  representations; production builds should enforce that assumption.
- Scaling parses the implementation's shortest round-trippable decimal and
  uses checked integer powers of ten with half-even rounding. No out-of-range
  floating-to-integer cast is performed. Overflow falls back to `q=0` plus an
  exact XOR repair.
- Reconstruction rounds the exact rational `q / 10^p` to the target IEEE
  representation with integer arithmetic and round-to-nearest, ties-to-even.
  This replaces the former formatting/`from_chars` path without changing its
  wire representation. The independent old/new oracle covers 4,800,350
  float32/float64 conversions, including integer limits and halfway cases.
  Decimal parsing is shared across precision candidates within one call;
  small adjacent-value caches are also local to that call.
- Special values are detected before decimal arithmetic. Under the benchmark's
  default non-trapping IEEE environment, classifying a signaling NaN does not
  interrupt encoding and the original payload bits are restored from the XOR
  stream. A process that explicitly unmasks floating-point traps needs a
  bit-level special-value classifier before using this frontend.
- BFE1 has structural validation but no inner checksum. Integrity remains the
  responsibility of the uniform NGF outer container.

The standalone tests cover float32/float64 special bit patterns, signaling and
quiet NaN payloads, infinities, negative zero, subnormals, extreme values,
10,000 random words, pilot/exhaustive/fixed selection, sparse repairs, raw
fallback, truncation and malformed frames.
