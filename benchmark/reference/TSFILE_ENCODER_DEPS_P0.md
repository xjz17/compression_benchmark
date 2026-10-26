# P0.1 — TSFile six encodings: Java source map (benchmark parity)

Bench CSV keys: `TS_2DIFF`, `RLE`, `GORILLA`, `CHIMP`, `SPRINTZ`, `RLBE`.
Java entry: `TSEncodingBuilder.getEncodingBuilder(encoding)` in `TSEncodingBuilder.java` + `CompressionUtils.testEncoding` (`Processor.processCsvSimple` loop).

Base path (all paths below relative to repo root):

`external/compression/tsfile/java/tsfile/src/main/java/org/apache/tsfile/encoding/`

## Encoder class by (encoding × dtype)

| CSV key | INT32 / DATE | INT64 / TIMESTAMP | FLOAT | DOUBLE |
|---------|--------------|-------------------|-------|--------|
| **TS_2DIFF** | `DeltaBinaryEncoder.IntDeltaEncoder` | `DeltaBinaryEncoder.LongDeltaEncoder` | `FloatEncoder(TS_2DIFF, FLOAT, …)` | `FloatEncoder(TS_2DIFF, DOUBLE, …)` |
| **RLE** | `IntRleEncoder` | `LongRleEncoder` | `FloatEncoder(RLE, FLOAT, …)` | `FloatEncoder(RLE, DOUBLE, …)` |
| **GORILLA** | `IntGorillaEncoder` | `LongGorillaEncoder` | `SinglePrecisionEncoderV2` | `DoublePrecisionEncoderV2` |
| **CHIMP** | `IntChimpEncoder` | `LongChimpEncoder` | `SinglePrecisionChimpEncoder` | `DoublePrecisionChimpEncoder` |
| **SPRINTZ** | `IntSprintzEncoder` | `LongSprintzEncoder` | `FloatSprintzEncoder` | `DoubleSprintzEncoder` |
| **RLBE** | `IntRLBE` | `LongRLBE` | `FloatRLBE` | `DoubleRLBE` |

Note: benchmark uses **INT32** and **FLOAT** only (`Processor` / `CppBenchmarkCompanion`). **GORILLA** in IoTDB enum is **Gorilla V2** (`TSEncoding.GORILLA` → `GorillaV2` builder), not `GORILLA_V1`.

## Encoder / decoder files (direct)

**Shared**

- `encoder/Encoder.java`, `decoder/Decoder.java`
- `encoder/DeltaBinaryEncoder.java`, `decoder/DeltaBinaryDecoder.java`
- `encoder/FloatEncoder.java`, `decoder/FloatDecoder.java`
- `bitpacking/IntPacker.java`, `bitpacking/LongPacker.java`

**RLE**

- `encoder/IntRleEncoder.java`, `encoder/LongRleEncoder.java`, `encoder/RleEncoder.java`
- `decoder/IntRleDecoder.java`, `decoder/LongRleDecoder.java`, `decoder/RleDecoder.java`

**GORILLA (V2)**

- `encoder/IntGorillaEncoder.java`, `encoder/LongGorillaEncoder.java`
- `encoder/SinglePrecisionEncoderV2.java`, `encoder/DoublePrecisionEncoderV2.java`
- `encoder/GorillaEncoderV2.java` (if referenced)
- `decoder/IntGorillaDecoder.java`, `decoder/LongGorillaDecoder.java`
- `decoder/SinglePrecisionDecoderV2.java`, `decoder/DoublePrecisionDecoderV2.java`
- `decoder/GorillaDecoderV2.java`

**CHIMP**

- `encoder/IntChimpEncoder.java`, `encoder/LongChimpEncoder.java`
- `encoder/SinglePrecisionChimpEncoder.java`, `encoder/DoublePrecisionChimpEncoder.java`
- `decoder/IntChimpDecoder.java`, `decoder/LongChimpDecoder.java`
- `decoder/SinglePrecisionChimpDecoder.java`, `decoder/DoublePrecisionChimpDecoder.java`

**SPRINTZ**

- `encoder/IntSprintzEncoder.java`, `encoder/LongSprintzEncoder.java`
- `encoder/FloatSprintzEncoder.java`, `encoder/DoubleSprintzEncoder.java`
- `encoder/SprintzEncoder.java`
- `decoder/IntSprintzDecoder.java`, `decoder/LongSprintzDecoder.java`
- `decoder/FloatSprintzDecoder.java`, `decoder/DoubleSprintzDecoder.java`
- `decoder/SprintzDecoder.java`

**RLBE**

- `encoder/IntRLBE.java`, `encoder/LongRLBE.java`
- `encoder/FloatRLBE.java`, `encoder/DoubleRLBE.java`
- `decoder/IntRLBEDecoder.java`, `decoder/LongRLBEDecoder.java`
- `decoder/FloatRLBEDecoder.java`, `decoder/DoubleRLBEDecoder.java`

## C++ port next steps (P0.2+)

1. Port **INT32 + FLOAT** paths first (match benchmark).
2. Pull in **TSFileConfig** defaults used by `FloatEncoder` / `Rle` (`maxPointNumber`, float precision) — align with `TSFileDescriptor.getInstance().getConfig()` in Java when generating goldens.
3. Add goldens under `benchmark/reference/golden/tsfile/` (see `CPP_MIGRATION.md` P0.3). **P0.2–P0.4 status:** `tsfile_encode_p02`…`p04` cover all six encodings; golden test compares **12** `.bin` files. **Note:** `FloatRLBE` in `tsfile` **2.0.2** (Maven) differs from older `FloatRLBE extends IntRLBE` sources: float deltas are IEEE `fsub`, not `intBits(a)-intBits(b)`.

## Changelog

| Date | Change |
|------|--------|
| 2026-04-02 | P0.1 dependency map from vendored TSFile Java |
| 2026-04-02 | C++ `encode_rlbe_float` verified vs **tsfile 2.0.2** `FloatRLBE` (not vendored `IntRLBE`-style float); goldens under `benchmark/reference/golden/tsfile/` |
| 2026-04-04 | CI: `.github/workflows/cpp-tsfile-goldens.yml` exercises `verify_tsfile_goldens` + `tsfile_p02_golden_test` (commit `*.bin` + `manifest.tsv`) |
