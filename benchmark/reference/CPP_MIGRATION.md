# Pure C++ benchmark migration (tracking)

This document splits **removing JVM companion code** from `benchmark_cpp` into milestones with clear acceptance criteria. Each section can become a **GitHub Issue** or internal requirement ticket.

**Current state:** `benchmark_cpp` runs native byte codecs, **column encodings (P1 ✅)**, **Java-compatible FastPFOR+VariableByte (`fastpfor_int` / `fastpfor_float`, P2.1 ✅)**, **paper Chimp32 (`chimp_int` / `chimp_float`, P2.2 ✅)** in `cpp/src/chimp32_paper_codec.*`, **LeCo / ALP float (P3 ✅)** in `cpp/src/leco_float_codec.*` and `cpp/src/alp_float_codec.*`, **ELF-on-Chimp and Elf+ float (P4 ✅)** in `cpp/src/elf_float_codec.*`, **Camel + Sim-Piece float (P5 ✅)** in `cpp/src/camel_float_codec.*` and `cpp/src/simpiece_float_codec.*`, **SZ / SZ3 / Serf float research rows (P6 ✅)** — **CLI by default**, **optional in-process** via **`BENCHMARK_CPP_EMBED_{SZ,SZ3,SERF}`** (Serf still gated by **`WEB_COMPRESSION_SERF_BENCH=1`**), and **IoTDB six TSFile encodings** via `tsfile_bench_tsfile`. Hybrid **`CppBenchmarkCompanion`** is reserved for **future** JAR-only rows (default env skips duplicate `ExternalCompressionBench.printRows` float research codecs via `WEB_COMPRESSION_CPP_NATIVE_*` in `scripts/cpp_java_companion.py`).

**North star:** All compression/encoding rows produced without invoking the JVM, with **golden parity** against today’s Java/TSFile behavior (bit-exact encoded bytes and lossless round-trip for lossless codecs).

**P0.1 done in-repo:** Java encoder/decoder file map → [`TSFILE_ENCODER_DEPS_P0.md`](./TSFILE_ENCODER_DEPS_P0.md).

---

## Git workflow + GitHub Issue 标题（直接复制）

### 推荐 Git 命令

```bash
# 新建跟踪文档分支（示例）
git checkout main
git pull
git checkout -b docs/cpp-migration-issues

# 提交跟踪文档与 P0.1 依赖清单
git add benchmark/reference/CPP_MIGRATION.md benchmark/reference/TSFILE_ENCODER_DEPS_P0.md
git commit -m "docs: CPP migration tracking + P0.1 TSFile encoder dependency map"

# 每个 Issue 单独功能分支（命名示例）
git checkout -b feat/cpp-p0-tsfile-golden-harness
# …开发…
git push -u origin feat/cpp-p0-tsfile-golden-harness

# 合并或直接在 main 上开发时，提交后记得推远程
git push origin main
```

### Issue 标题（复制到 GitHub New Issue）

按优先级从高到低：

1. `P0.6 [CPP] TSFile encodings — remove JVM ts-only companion` ✅ native `tsfile_bench_tsfile` + decoders; companion `external-only`
2. `P0.5 [CPP] TSFile encodings — golden harness + CI` ✅ `.github/workflows/cpp-tsfile-goldens.yml` (`verify_tsfile_goldens` + `tsfile_p02_golden_test`; requires goldens committed under `benchmark/reference/golden/tsfile/`)
3. `P0.4 [CPP] TSFile encodings — port SPRINTZ + RLBE (int32/float)` ✅ `tsfile_encode_p04.*` + `tsfile_p02_golden_test` (12 blobs)
4. `P0.3 [CPP] TSFile encodings — port GORILLA + CHIMP (int32/float)` ✅ encode/decode + goldens + `tsfile_p02_golden_test`
5. `P0.2 [CPP] TSFile encodings — port TS_2DIFF + RLE (int32/float)` ✅ same
6. `P0.1 [CPP] TSFile encodings — dependency audit for six encodings` ✅ 见 `TSFILE_ENCODER_DEPS_P0.md`

7. `P1.5 [CPP] Column encodings — encode-after-decode involution on goldens` ✅ `column_encodings_golden_test`
8. `P1.4 [CPP] Column encodings — one-shot export + manifest sync` ✅ `scripts/export_column_encodings_golden_java.sh`; `ColumnarGoldenExport` writes `manifest.tsv`
9. `P1.3 [CPP] Column encodings — native decoders + golden manifest` ✅
10. `P1.2 [CPP] Column encodings — fix deltas to match Java byte-for-byte` ✅
11. `P1.1 [CPP] Column encodings — golden fixtures vs Java` ✅

12. `P2.2 [CPP] Chimp32 int/float — golden + native port` ✅ `chimp32_golden_test`, native CSV rows; `scripts/export_chimp32_golden_java.sh`
13. `P2.1 [CPP] FastPFOR int/float — golden + native port` ✅ `external/compression/FastPFor`, `fastpfor_golden_test`, native CSV rows; export `scripts/export_fastpfor_golden_java.sh`

14. `P3.2 [CPP] alp_float — parity with Java ExternalCompressionBench` ✅ `AlpFloatGoldenExport`, `alp_float_golden_test`, native `benchmark_cpp` row; `scripts/export_alp_float_golden_java.sh`
15. `P3.1 [CPP] leco_float — parity with Java ExternalCompressionBench` ✅ `LeCoFloatGoldenExport`, `leco_float_golden_test`, native row; `scripts/export_leco_float_golden_java.sh`

16. `P4.2 [CPP] elf_plus_float — parity + remove JVM row` ✅ `elf_float_codec.*`, `ElfFloatGoldenExport`, `scripts/export_elf_float_golden_java.sh` (builds `external/compression/elf` + `dependency:copy-dependencies`)
17. `P4.1 [CPP] elf_on_chimp_float — feasibility + golden` ✅ same goldens layout as `benchElfVariant32Float` (`writtenBits` truncation; 2048 chunks)

18. `P5.2 [CPP] Sim-Piece presets — native path + golden` ✅ `simpiece_float_codec.*`, `SimPieceFloatGoldenExport`, `golden/simpiece_float/`, `export_simpiece_float_golden_java.sh`
19. `P5.1 [CPP] camel_float — native path or binary contract` ✅ `camel_float_codec.*`, `CamelFloatGoldenExport`, `scripts/export_camel_float_golden_java.sh`; goldens under `golden/camel_float/`

20. `P6.2 [CPP] Serf — embed vs CLI` ✅ `-DBENCHMARK_CPP_EMBED_SERF=ON` → `Serf/src` shared **`serf`**, `research_cli_serf_embed.*`; `WEB_COMPRESSION_SERF_BENCH=1` prefers in-process before `sfz`
21. `P6.1 [CPP] SZ/SZ3 — optional link vs CLI` ✅ **SZ:** `-DBENCHMARK_CPP_EMBED_SZ=ON`; **SZ3:** `-DBENCHMARK_CPP_EMBED_SZ3=ON` + system **pkg-config libzstd**; default remains CLI

### 建议优先级（简表）

| 优先级 | 里程碑 | 说明 |
|--------|--------|------|
| P0 ✅ | TSFile 六种 | 去掉 companion 最大一块 JVM |
| P1 ✅ | 列式编码 | 与 `ColumnarEncodings` 字节对齐；`col_*` 原生 CSV |
| P2 | FastPFOR + Chimp32 | FastPFOR ✅；Chimp32 ✅（金标 + 原生行） |
| P3 ✅ | LeCo + ALP | float：`leco_float` / `alp_float` 金标 + 原生行；companion 跳过重复 JVM |
| P4 ✅ | ELF-on-Chimp + Elf+ | `elf_on_chimp_float` / `elf_plus_float` 金标 + 原生行；companion 跳过重复 JVM |
| P5 ✅ | Camel + Sim-Piece | 原生 + 金标；`WEB_COMPRESSION_CPP_NATIVE_CAMEL_FLOAT` / `WEB_COMPRESSION_CPP_NATIVE_SIMPIECE_FLOAT` |
| P6 ✅ | SZ / SZ3 / Serf 可选进程内 | 同上；验收见下文 **Milestone P6** 核对表 |

### 仓库路径索引（快捷）

| 用途 | 路径 |
|------|------|
| C++ 主流程 | `cpp/src/main.cpp`, `cpp/src/column_encodings.cpp`, `cpp/src/native_streams.cpp`, `cpp/src/research_cli.cpp`, `cpp/src/run_java_companion.cpp` |
| JVM companion | `java/.../CppBenchmarkCompanion.java`, `scripts/cpp_java_companion.py` |
| Java 编码参考 | `java/.../CompressionUtils.java`, `external/compression/tsfile/java/tsfile/.../encoding/` |
| P0.1 依赖清单 | `benchmark/reference/TSFILE_ENCODER_DEPS_P0.md` |
| P0.2 TS_2DIFF+RLE (C++) | `cpp/src/tsfile_encoding/tsfile_encode_p02.hpp` |
| P0.3 GORILLA+CHIMP (C++) | `cpp/src/tsfile_encoding/tsfile_encode_p03.hpp` |
| P0.4 SPRINTZ+RLBE (C++) | `cpp/src/tsfile_encoding/tsfile_encode_p04.hpp` |
| TSFile + golden CI | `.github/workflows/cpp-tsfile-goldens.yml` (also `fastpfor_*`, `chimp32_*`, `leco_float_*`, `alp_float_*`, `elf_*float*`, `camel_float*`, `simpiece_float*` targets) |
| P1 column goldens | `benchmark/reference/golden/column_encodings/`, `java/.../ColumnarGoldenExport.java`, `scripts/export_column_encodings_golden_java.sh`, `cpp/tests/column_encodings_golden_test.cpp`, `cpp/tests/verify_column_encodings_goldens.cpp` |
| P2.1 FastPFOR goldens | `benchmark/reference/golden/fastpfor/`, `java/.../FastPFORGoldenExport.java`, `scripts/export_fastpfor_golden_java.sh`, `cpp/src/java_fastpfor_codec.*`, `cpp/tests/fastpfor_golden_test.cpp`, `cpp/tests/verify_fastpfor_goldens.cpp`, local clone `external/compression/FastPFor`; **SIMDe** at `cpp/third_party/simde` via `bash scripts/init_compression_submodules.sh` for arm64 / Apple Silicon — x86_64 uses `-msse4.2`, MSVC x64 uses `/arch:AVX` (`cpp/CMakeLists.txt`) |
| P3 LeCo / ALP float | `benchmark/reference/golden/leco_float/`, `benchmark/reference/golden/alp_float/`, `java/.../LeCoFloatGoldenExport.java`, `AlpFloatGoldenExport.java`, `scripts/export_leco_float_golden_java.sh`, `export_alp_float_golden_java.sh`, `cpp/src/leco_float_codec.*`, `alp_float_codec.*`, `cpp/tests/leco_float_golden_test.cpp`, `alp_float_golden_test.cpp`, verify helpers |
| P4 ELF float | `golden/elf_on_chimp_float/`, `golden/elf_plus_float/`, `java/.../ElfFloatGoldenExport.java`, `scripts/export_elf_float_golden_java.sh`, `cpp/src/elf_float_codec.*`, `cpp/tests/elf_*_float_golden_test.cpp`, verify helpers; ELF JAR `external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar` + `target/dependency/*.jar` for export |
| P5.1 Camel float | `golden/camel_float/`, `java/.../CamelFloatGoldenExport.java`, `scripts/export_camel_float_golden_java.sh`, `cpp/src/camel_float_codec.*`, `cpp/tests/camel_float_golden_test.cpp`, `verify_camel_float_goldens`; build camel `external/compression/camel` |
| P5.2 Sim-Piece float | `golden/simpiece_float/`, `java/.../SimPieceFloatGoldenExport.java`, `scripts/export_simpiece_float_golden_java.sh`, `cpp/src/simpiece_float_codec.*`, `simpiece_float_golden_test`, `verify_simpiece_float_goldens`; native ports in Java/Python/C++ (upstream submodule kept for reference only) |
| P6 ✅ (CMake 可选) | **`cpp/CMakeLists.txt`:** `BENCHMARK_CPP_EMBED_SZ` / `SZ`, `WEB_COMPRESSION_EMBED_SZ` + `research_cli.cpp`; **`BENCHMARK_CPP_EMBED_SZ3`** / `SZ3c`, `WEB_COMPRESSION_EMBED_SZ3` + `research_cli_sz3_embed.*`, **pkg-config `libzstd`**; **`BENCHMARK_CPP_EMBED_SERF`** / `Serf/src` → **`serf`**, `WEB_COMPRESSION_EMBED_SERF` + `research_cli_serf_embed.*`, **`BUILD_RPATH`**, **`WEB_COMPRESSION_SERF_BENCH=1`** 仍必选才能出 Serf 行。任一开：Linux 上关闭 **`benchmark_cpp` `-static`**。 **`cmake_minimum` 3.18**（SZ3） |
| 上传 / 静态 | `backend/myapp/views.py`, `scripts/regenerate_staticdata_cpp.py` |
| 前端 Available Algorithms | `frontend/src/utils/benchmarkAlgorithmCatalog.js`, `frontend/src/views/Home.vue`, `frontend/src/views/Upload.vue` |

### Issue 描述模板（changelog 风格一句）

```text
Goal: <P0.x 具体交付>
Acceptance: C++ encode bytes == golden; round-trip == source column; companion/JVM removed for these rows.
Refs: benchmark/reference/CPP_MIGRATION.md, benchmark/reference/TSFILE_ENCODER_DEPS_P0.md
```

---

## Principles

1. **Golden source of truth:** Per milestone, fix CSV column packing (`int32` / `float`, little-endian) to match Java. Golden artifacts are **encoded byte blobs** from Java (or `sha256` manifests) plus round-trip value equality.
2. **Incremental merge:** Ship C++ behind CMake flags or env toggles until parity is proven; then default pure C++ and delete the JVM path for that milestone.
3. **Acceptance template (per Issue):**
   - **Inputs:** Named dataset slice + dtype + column selection (must match Java harness).
   - **Artifacts:** `golden/<milestone>/<algorithm>_<dtype>_<fixture>.bin` (or hash list).
   - **Tests:** C++ encode output `memcmp` to golden; decode vs source column `==` element-wise.
   - **Integration:** Row appears from `benchmark_cpp` only; `WEB_COMPRESSION_CPP_SKIP_JAVA_COMPANION` no longer needed for those algorithms.
   - **Docs:** Algorithm CSV key mapping, known limits (e.g. float-only).

---

## Milestone P0 — TSFile six encodings (highest priority)

Align with Java report keys: `TS_2DIFF`, `RLE`, `GORILLA`, `CHIMP`, `SPRINTZ`, `RLBE` for **INT32** and **FLOAT** columns.

| ID    | Task | Deliverable |
|-------|------|-------------|
| P0.1  | Map minimal dependency cut from `external/compression/tsfile/java/tsfile` encoders/decoders (six types + `Encoder`/`Decoder`/bit streams). | Dependency graph / file list in Issue or appendix |
| P0.2  | Implement `cpp/src/tsfile_encoding/` (or similar) with semantics matching Java, no “close enough” shortcuts. | ✅ encoders + decoders + `tsfile_p02_golden_test` round-trip (float quant rules where applicable) |
| P0.3  | Export Java golden blobs for fixtures (small columns + edge cases: constant, ramp, NaN policy for float per Java). | ✅ `benchmark/reference/golden/tsfile/*.bin` (+ export script / `TsFileGoldenExport`) |
| P0.4  | CI: C++ vs golden binary compare (+ decode check when decoders exist). | ✅ `.github/workflows/cpp-tsfile-goldens.yml` |
| P0.5  | Remove TS encodings from `CppBenchmarkCompanion` (`--phase ts-only` deleted). C++ **decoders** + `tsfile_bench_tsfile` decompression timing. | ✅ done (companion `external-only`) |

**Risks:** `GORILLA` vs `GORILLA_V1`, CHIMP float/int variants, RLBE/Sprintz corner cases — mirror `TSEncoding` choice in Java exactly.

**Suggested Issue titles:**

- `P0.1 [CPP] TSFile encodings — dependency audit for six encodings`
- `P0.2 [CPP] TSFile encodings — port TS_2DIFF + RLE (int/float)`
- `P0.3 [CPP] TSFile encodings — port GORILLA + CHIMP (int/float)`
- `P0.4 [CPP] TSFile encodings — port SPRINTZ + RLBE (int/float)`
- `P0.5 [CPP] TSFile encodings — golden harness + CI`
- `P0.6 [CPP] TSFile encodings — remove JVM ts-only companion`

---

## Milestone P1 — Columnar encodings (parity with Java `ColumnarEncodings`)

**Status: ✅ complete (P1.1–P1.5).** `cpp/src/column_encodings.*` is the native implementation; goldens and CI lock **bit-level alignment** with Java where required (BSS+LZ4 non-empty: semantic parity only — see P1.2).

**Acceptance summary:** deterministic Java blobs `memcmp` in C++; every golden decodes to the export fixture; `encode(decode(blob))` memcmp for deterministic layouts + empty LZ4 frame; `manifest.tsv` + `verify_column_encodings_goldens`; `benchmark_cpp` emits `DELTA_BINARY_PACKED`, `col_dictionary`, `col_rle_int`, `col_bss_lz4` without the JVM.

| ID    | Task |
|-------|------|
| P1.1  | Golden for `delta_binary_packed`, `col_dictionary`, `col_rle_int` (int); `col_dictionary`, `col_bss_lz4` (float). **Done:** int/float dictionary + delta-zigzag + RLE + **planar BSS** use **memcmp** vs Java (`ColumnarGoldenExport`); extended edge fixtures (empty, dupes, RLE runs, negative deltas). |
| P1.2  | **Byte parity audit vs `ColumnarEncodings` (2026-04):** All **deterministic** layouts match Java on exported goldens. **BSS+LZ4:** Apache writes LZ4 **frame** with DEFAULT prefs (4MB max block, block-independent, content checksum). C++ **must not** use `LZ4F_compressFrame` alone — it calls `LZ4F_optimalBSID` and rewrites BD to 64KB for small inputs; use **`LZ4F_compressBegin` / `Update` / `End`** with those prefs so the **frame header** matches Java. **LZ4 block bodies** still differ: Commons uses its Java LZ77 block encoder; bundled liblz4 uses the native fast codec — both decompress to the same BSS, but **full framed blob memcmp for non-empty columns is not required**. **Empty float column:** framed blob **memcmp** vs Java ✅. **Non-empty:** Java golden → C++ decode → floats; C++ encode → C++ decode round-trip. |
| P1.3  | **Decoders + manifest:** C++ `decodeDeltaZigZagVarint`, `decodeDictionaryInt`/`Float`, `decodeRleInt`, `floatsFromPlanarByteStreamSplit` mirror Java layouts; `column_encodings_golden_test` decodes every int/float/planar golden back to the fixture column. `benchmark/reference/golden/column_encodings/manifest.tsv` lists `values_len` + `bytes_len`; CI runs `verify_column_encodings_goldens`. |
| P1.4  | **Export + integration:** `scripts/export_column_encodings_golden_java.sh` runs `ColumnarGoldenExport`, which writes all `.bin` files and **regenerates `manifest.tsv`** (sorted, sizes from actual bytes). `benchmark_cpp` appends `DELTA_BINARY_PACKED`, `col_dictionary`, `col_rle_int`, `col_bss_lz4` from **native** `colenc::bench*` (no JVM for these rows); Java companion remains **external-only** for other codecs. |
| P1.5  | **Involution on goldens:** `column_encodings_golden_test` checks `encode(decode(blob)) == blob` (memcmp) for every deterministic golden (delta, dict, RLE, planar BSS; **empty** BSS+LZ4 frame only). Skips non-empty `col_bss_lz4_float_default.bin` (LZ4 block bytes may differ from Java). |

**Suggested Issue titles:**

- `P1.1 [CPP] Column encodings — golden fixtures vs Java`
- `P1.2 [CPP] Column encodings — fix deltas to match Java byte-for-byte`
- `P1.3 [CPP] Column encodings — native decoders + golden manifest`
- `P1.4 [CPP] Column encodings — one-shot export + manifest sync`
- `P1.5 [CPP] Column encodings — encode-after-decode involution on goldens`

---

## Milestone P2 — FastPFOR + Chimp32 (paper path) ✅

Java reference: `ExternalCompressionBench` (Lemire `Composition(FastPFOR, VariableByte)` + reflection Chimp32). **P2.1 + P2.2 are done in-tree** (native rows + goldens + companion env skips).

| ID | Task |
|----|------|
| **P2.1** | **Done:** Lemire C++ `CompositeCodec<FastPFor<8>, VariableByte>` in `cpp/src/java_fastpfor_codec.*`, linked into `benchmark_cpp`; goldens under `benchmark/reference/golden/fastpfor/` (`FastPFORGoldenExport`, LE int32 **words**); `fastpfor_golden_test` memcmp + round-trip; `WEB_COMPRESSION_CPP_NATIVE_FASTPFOR=1` (set in `cpp_java_companion.py`) skips JVM duplicate. **Build:** static lib compiled from `external/compression/FastPFor/src/*` (no upstream CMake — avoids CPM download). **aarch64:** needs [SIMDe](https://github.com/simd-everywhere/simde) headers → e.g. `brew install simde`, or clone that repo into `cpp/third_party/simde`, or set `SIMDE_ROOT` to the parent of `simde/`. **x86_64:** `-msse4.2` only. |
| **P2.2** | **Done:** Vigna bitstream + Chimp32 paper path in `cpp/src/chimp32_paper_codec.*` (Java `getSize()` truncation for golden bytes); goldens under `benchmark/reference/golden/chimp32/` (`Chimp32GoldenExport`, `scripts/export_chimp32_golden_java.sh`); `chimp32_golden_test` memcmp + round-trip; `WEB_COMPRESSION_CPP_NATIVE_CHIMP32=1` skips JVM `chimp_*` in `ExternalCompressionBench.printRows` when using `cpp_java_companion.py`. |

**Suggested Issue titles:**

- `P2.1 [CPP] FastPFOR int/float — golden + native port`
- `P2.2 [CPP] Chimp32 int/float — golden + native port`

---

## Milestone P3 — LeCo + ALP (float) ✅

Java reference: `ExternalCompressionBench.benchLeCoFloatScaledIntegersFastPFor`, `AlpFloatBench.benchAlpFloat` (int stage `Composition(FastPFOR, VariableByte)`). **Done in-tree:** goldens from `LeCoFloatGoldenExport` / `AlpFloatGoldenExport`; C++ encode `memcmp` + round-trip vs fixtures; `WEB_COMPRESSION_CPP_NATIVE_LECO_FLOAT` / `WEB_COMPRESSION_CPP_NATIVE_ALP_FLOAT` skip duplicate JVM rows when using `cpp_java_companion.py`; CI runs `verify_*` + `*_golden_test` like P2.

| ID | Task |
|----|------|
| **P3.1** | **Done:** `cpp/src/leco_float_codec.*`; goldens `benchmark/reference/golden/leco_float/`; `leco_float_golden_test`; export `scripts/export_leco_float_golden_java.sh`. |
| **P3.2** | **Done:** `cpp/src/alp_float_codec.*`; goldens `benchmark/reference/golden/alp_float/`; `alp_float_golden_test`; export `scripts/export_alp_float_golden_java.sh`. |

**Suggested Issue titles:**

- `P3.1 [CPP] leco_float — parity with Java ExternalCompressionBench`
- `P3.2 [CPP] alp_float — parity with Java ExternalCompressionBench`

---

## Milestone P4 — ELF (Elf-on-Chimp, Elf+) ✅

Java reference: `ExternalCompressionBench.benchElfVariant32Float` (2048-value chunks, compressed length from `OutputBitStream.writtenBits()`, isolated ELF JAR classpath in the bench). **Done in-tree:** C++ port in `cpp/src/elf_float_codec.*` (erasing stage + Chimp32 XOR or Elf+XOR32 per Java); goldens from `ElfFloatGoldenExport` (`mvn -Dmaven.test.skip=true dependency:copy-dependencies package` in `external/compression/elf`); `WEB_COMPRESSION_CPP_NATIVE_ELF_ON_CHIMP_FLOAT` / `WEB_COMPRESSION_CPP_NATIVE_ELF_PLUS_FLOAT` skip duplicate JVM rows when using `cpp_java_companion.py`.

| ID | Task |
|----|------|
| **P4.1** | **Done:** `elf_on_chimp_float` native row + goldens + `elf_on_chimp_float_golden_test`. |
| **P4.2** | **Done:** `elf_plus_float` native row + goldens + `elf_plus_float_golden_test`. |

**Suggested Issue titles:**

- `P4.1 [CPP] elf_on_chimp_float — feasibility + golden`
- `P4.2 [CPP] elf_plus_float — parity + remove JVM row`

---

## Milestone P5 — Camel + Sim-Piece

**P5.1 ✅** Native Camel in `cpp/src/camel_float_codec.*` (Vigna bit stream; 1000-point chunks). Goldens from `CamelFloatGoldenExport` use **Camel-stable** ramps (`1.0f + i`, `100.0f + i * 0.5f`) because the upstream JAR does not round-trip every float sequence (e.g. `1.0f`, `1.03125f` fails `parityCompressedChunksCamel`). `WEB_COMPRESSION_CPP_NATIVE_CAMEL_FLOAT` skips the JVM row when using `cpp_java_companion.py`.

**P5.2 ✅** Native Sim-Piece in `cpp/src/simpiece_float_codec.*` (segment logic matching `simpiece_python` / `SimPieceFloatNative`; prelude + **Zstd level 3**). Goldens from `SimPieceFloatGoldenExport` (fixtures align with Camel-stable ramps). Codec is **lossy**: golden test **`memcmp`** on compressed blob + decompress tolerance vs Java bench. `WEB_COMPRESSION_CPP_NATIVE_SIMPIECE_FLOAT` skips JVM `simpiece_*` rows when using `cpp_java_companion.py`.

**Suggested Issue titles:**

- `P5.1 [CPP] camel_float — native path or binary contract` ✅
- `P5.2 [CPP] Sim-Piece presets — native path + golden` ✅

---

## Milestone P6 — SZ / SZ3 / Serf embedding (optional strictness)

**Status: ✅ closed (product deliverable).** Default remains **subprocess** to upstream CLIs (no JVM). Optional CMake targets let **`benchmark_cpp`** call **native APIs** instead of `sz` / `sz3` / `sfz` when subprocess-free or packaging constraints matter.

**Delivered:**

- **P6.1 — Classic SZ 2.x:** `-DBENCHMARK_CPP_EMBED_SZ=ON` — nested **`external/compression/SZ`** → shared **`libSZ`**. `research_cli.cpp`: `SZ_Init` + `SZ_compress_args` + `SZ_decompress` for `sz_float_*`; fallback to **`sz`** CLI if init fails.

- **P6.1 — SZ3:** `-DBENCHMARK_CPP_EMBED_SZ3=ON` — nested **`external/compression/SZ3`** → **`SZ3c`**. `research_cli_sz3_embed.cpp`: `SZ3::Config(n)`, **`loadcfg(sz.config)`**, `SZ_compress` / `SZ_decompress<float>` for `sz3_float_*` presets aligned with **`sz3`** CLI; fallback to **`sz3`** CLI. Requires **pkg-config `libzstd`** (bundled zstd **off** — avoids CMake **`zstd`** target clash with **`cpp/third_party/zstd`**).

- **P6.2 — Serf:** `-DBENCHMARK_CPP_EMBED_SERF=ON` — **`Serf/src`** only → shared **`serf`**. When **`WEB_COMPRESSION_SERF_BENCH=1`**, **`bench_serf_inproc`** (window **1000**, bounds **1e-1 … 1e-4**) runs before **`sfz`** subprocess. **`BUILD_RPATH`** / **`INSTALL_RPATH`** point at **`libserf`**.

**Shared constraints:** Any embed option disables Linux **`-static`** on **`benchmark_cpp`**. **`cmake_minimum` 3.18** under **`cpp/`** (SZ3).

### P6 closed — checklist

| Item | Status |
|------|--------|
| **SZ** | `BENCHMARK_CPP_EMBED_SZ` + `WEB_COMPRESSION_EMBED_SZ`; CLI fallback; submodule **`external/compression/SZ`**. |
| **SZ3** | `BENCHMARK_CPP_EMBED_SZ3` + `WEB_COMPRESSION_EMBED_SZ3` + **`libzstd`** via pkg-config; CLI fallback; submodule **`external/compression/SZ3`**. |
| **Serf** | `BENCHMARK_CPP_EMBED_SERF` + `WEB_COMPRESSION_EMBED_SERF`; **`WEB_COMPRESSION_SERF_BENCH=1`** unchanged; RPATH for **`serf`**; submodule **`external/compression/Serf`**. |
| **Docs / index** | Issue list, priority table, path index, references, changelog updated. |

**Suggested Issue titles (historical):**

- `P6.1 [CPP] SZ/SZ3 — optional static link vs CLI (product decision)` ✅
- `P6.2 [CPP] Serf — embed sfz logic vs CLI` ✅

---

## Priority order (recommended)

1. **P0** — Largest JVM removal, table-visible core encodings.
2. **P1** ✅ — Lock column encodings to Java bytes (`ColumnarEncodings`); done.
3. **P2** — FastPFOR ✅; Chimp32 ✅.
4. **P3 ✅** — LeCo / ALP float native + goldens.
5. **P4 ✅** — ELF float native + goldens.
6. **P5 ✅** — Camel + Sim-Piece float native + goldens.
7. **P6 ✅** — SZ / SZ3 / Serf optional in-proc CMake flags (`WEB_COMPRESSION_SERF_BENCH` still gates Serf rows).

---

## References in this repo

- Hybrid driver: `cpp/src/main.cpp` (`processCsvSimple`), `cpp/src/run_java_companion.cpp`
- JVM companion: `java/src/main/java/org/example/CppBenchmarkCompanion.java`
- Launcher: `scripts/cpp_java_companion.py`
- Java golden reference: `java/src/main/java/org/example/CompressionUtils.java` (`testEncoding`), `external/compression/tsfile/java/tsfile/.../encoding/`
- **P1 column encodings:** `java/.../ColumnarEncodings.java`, `java/.../ColumnarGoldenExport.java`, `cpp/src/column_encodings.hpp`, golden dir `benchmark/reference/golden/column_encodings/`
- **P2.1 FastPFOR:** `java/.../FastPFORGoldenExport.java`, `cpp/src/java_fastpfor_codec.hpp`, golden dir `benchmark/reference/golden/fastpfor/`
- **P3 LeCo / ALP float:** `java/.../LeCoFloatGoldenExport.java`, `AlpFloatGoldenExport.java`, `cpp/src/leco_float_codec.hpp`, `alp_float_codec.hpp`, golden dirs `benchmark/reference/golden/leco_float/`, `golden/alp_float/`
- **P4 ELF float:** `java/.../ElfFloatGoldenExport.java`, `cpp/src/elf_float_codec.hpp`, golden dirs `benchmark/reference/golden/elf_on_chimp_float/`, `golden/elf_plus_float/`
- **P5.1 Camel float:** `java/.../CamelFloatGoldenExport.java`, `cpp/src/camel_float_codec.hpp`, `golden/camel_float/`
- **P5.2 Sim-Piece float:** `java/.../SimPieceFloatGoldenExport.java`, `cpp/src/simpiece_float_codec.hpp`, `golden/simpiece_float/`
- **P6 ✅ optional in-proc:** `cpp/CMakeLists.txt` (`BENCHMARK_CPP_EMBED_{SZ,SZ3,SERF}`, CMake **3.18**), `research_cli.cpp`, `research_cli_sz3_embed.*`, `research_cli_serf_embed.*`, submodules `external/compression/{SZ,SZ3,Serf}`

---

## Submodule hygiene (`external/compression/*`)

When `git status` shows `m external/compression/camel` (or chimp) or `?` under SZ/SZ3, the dirt is **inside the submodule**, not in the parent commit unless you bump the submodule pointer after committing there.

### What was dirty (typical causes)

| Submodule | Symptom | Cause |
|-----------|---------|--------|
| **camel** | `M pom.xml`, `M CamelUtils.java` | Local `pom.xml` adds `maven-dependency-plugin` `copy-dependencies`; `CamelUtils.java` drops an erroneous `javax.jws.Oneway` import (compile fix). |
| **chimp** | `M Chimp*.java` (5 files) | Local workaround: constructors use `new OutputBitStream(new byte[16 * 1024 * 1024])` instead of small default buffers (bench / large-series workaround). |
| **SZ** | `?? example/sz` | **Built SZ CLI** copied or produced under `example/sz` (untracked binary). Parent `.gitignore` already lists this path for the superproject; the submodule itself still shows `??` until the file is removed or ignored. |
| **SZ3** | `?? sz.config` | **Local/runtime config** at repo root of SZ3; not part of upstream tree. |
| **elf** | `?? target/dependency/` (sometimes ignored in parent) | **`mvn dependency:copy-dependencies`** for `ElfFloatGoldenExport` / golden regeneration; safe to delete; parent `.gitignore` lists `external/compression/elf/target/dependency/`. |

### Cleanup (discard local submodule edits — restore vendor defaults)

```bash
# Camel: drop Maven + Java edits (verify you do not need copy-dependencies for a local JAR flow)
git -C external/compression/camel restore .

# Chimp: drop oversized OutputBitStream buffers (restore paper defaults)
git -C external/compression/chimp restore .
```

### Cleanup (artifacts only — keep camel/chimp edits)

```bash
# SZ: remove built CLI (rebuild with your usual CMake flow when needed)
rm -f external/compression/SZ/example/sz

# SZ3: remove local config (regenerate or pass config path explicitly when running tests)
rm -f external/compression/SZ3/sz.config
```

### If you intend to keep camel/chimp changes

Commit **inside** the submodule on a branch, push to your fork, then in the parent repo update the submodule commit pointer. Do **not** leave committed pointers at `-dirty` (invalid).

```bash
cd external/compression/chimp
git checkout -b local/larger-bitstream-buffers
git add -A && git commit -m "bench: enlarge OutputBitStream buffers"
# git remote add ... && git push ...
cd ../../..
git add external/compression/chimp
git commit -m "chore: bump chimp submodule for bench buffers"
```

### P1 closed — column encodings checklist (P1.1–P1.5)

| Item | Status |
|------|--------|
| **P1.1** | Golden blobs + `ColumnarGoldenExport`; `column_encodings_golden_test` memcmp on deterministic encodings. |
| **P1.2** | Byte parity audit (LZ4 frame streaming, planar BSS, edge fixtures); see P1 table. |
| **P1.3** | Native decoders + decode-all-goldens checks + `manifest.tsv` + `verify_column_encodings_goldens` in CI. |
| **P1.4** | `export_column_encodings_golden_java.sh`; `ColumnarGoldenExport` auto-writes `manifest.tsv`; `benchmark_cpp` `col_*` rows are native-only. |
| **P1.5** | Golden byte involution `encode(decode(blob))` for deterministic layouts (+ empty LZ4 frame). |

**Next:** **P2.2** (Chimp32 native + goldens). P2.1 (FastPFOR) is shipped.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-02 | Initial tracking doc from migration roadmap |
| 2026-04-02 | Added Git/GitHub copy-paste section, path index, Issue template; **P0.1** `TSFILE_ENCODER_DEPS_P0.md` |
| 2026-04-02 | **P0.2a** Java `TsFileGoldenExport`, `benchmark/reference/golden/tsfile/`, `scripts/export_tsfile_golden_java.sh`, optional CMake `ENABLE_TSFILE_GOLDEN_TESTS` → `verify_tsfile_goldens` |
| 2026-04-04 | **P0.2 (partial)** C++ `cpp/src/tsfile_encoding/tsfile_encode_p02.*` — TS_2DIFF + RLE for INT32/FLOAT bit-exact vs Java goldens; CMake target `tsfile_p02_golden_test` (same option as manifest checker) |
| 2026-04-04 | **P0.3 (partial)** C++ `cpp/src/tsfile_encoding/tsfile_encode_p03.*` — GORILLA V2 + CHIMP INT32/FLOAT bit-exact vs Java goldens; exercised by same `tsfile_p02_golden_test` (now 8 blobs) |
| 2026-04-04 | **P0.4 (partial)** C++ `cpp/src/tsfile_encoding/tsfile_encode_p04.*` — SPRINTZ + RLBE; **Float RLBE** via `FloatRlbEncoder` matches Maven `tsfile` 2.0.2 (`FloatRLBE`: float `fsub` deltas, not int-bit delta). `tsfile_p02_golden_test`: 12 blobs, all `memcmp` OK |
| 2026-04-04 | **CI:** `.github/workflows/cpp-tsfile-goldens.yml` — `verify_tsfile_goldens` + `tsfile_p02_golden_test` on push/PR |
| 2026-04-05 | **P0 complete (runtime):** `tsfile_bench_tsfile`, native decoders, RLE inner-read + Chimp float/Java case-1 width fixes; JVM `ts-only` removed; doc: Issue list + P0 table checkmarks, **Submodule hygiene** section (camel/chimp/SZ/SZ3), **Next: P1** pointer |
| 2026-04-05 | **P1.3:** `decodeDeltaZigZagVarint` / `decodeDictionary*` / `decodeRleInt` / `floatsFromPlanarByteStreamSplit`; golden decode checks; `column_encodings/manifest.tsv` + `verify_column_encodings_goldens`; CI step |
| 2026-04-05 | **P1.4:** `ColumnarGoldenExport` rewrites `manifest.tsv` from written blobs; `scripts/export_column_encodings_golden_java.sh`; doc: native `col_*` CSV rows vs companion |
| 2026-04-05 | **P1.5:** `column_encodings_golden_test` — memcmp `encode(decode(Java golden))` for delta/dict/RLE/planar + empty BSS+LZ4; CI workflow also triggers on column golden export script |
| 2026-04-05 | **P1 milestone closed (docs):** top-level status + Issue list P1.5; Milestone P1 acceptance block; path index + References; priority table P1 ✅ |
| 2026-04-05 | **P2.1 FastPFOR:** Java `FastPFORGoldenExport` + `golden/fastpfor/`; C++ `java_fastpfor_codec` + Lemire submodule; `fastpfor_golden_test` / CI; native `fastpfor_int`/`fastpfor_float` in `benchmark_cpp`; companion env skips JVM FastPFOR duplicate |
| 2026-04-02 | **P5.1 Camel float:** `CamelFloatGoldenExport` + `golden/camel_float/` (stable ramps); C++ `camel_float_codec`; `camel_float_golden_test` + CI; native `camel_float` CSV row; `WEB_COMPRESSION_CPP_NATIVE_CAMEL_FLOAT` |
| 2026-04-02 | **P5.2 Sim-Piece float:** `SimPieceFloatGoldenExport` + `golden/simpiece_float/`; C++ `simpiece_float_codec` (Zstd-3 blobs match Java); `simpiece_float_golden_test` + CI; native `simpiece_float_relRange_*` rows; `WEB_COMPRESSION_CPP_NATIVE_SIMPIECE_FLOAT` |
| 2026-04-04 | **P6.1 SZ embed:** CMake `BENCHMARK_CPP_EMBED_SZ` + nested `external/compression/SZ`; `research_cli` in-process `sz_float_*` via `sz_api.h`; CLI default unchanged |
| 2026-04-04 | **SIMDe for FastPFOR (arm64 + Intel):** submodule `cpp/third_party/simde`; `cpp/CMakeLists.txt` uses explicit header check + `-msse4.2` (GCC/Clang x86_64) or `/arch:AVX` (MSVC) or SIMDe + `SIMDE_ENABLE_NATIVE_ALIASES` on AArch64; backend dev: `backend/.venv` + `pip install -r requirements.txt` |
| 2026-04-04 | **P6.1 SZ3 in-proc:** `BENCHMARK_CPP_EMBED_SZ3` + `SZ3c` / `research_cli_sz3_embed.cpp` (`Config::loadcfg` + compress/decompress); system **pkg-config libzstd** only; `cmake_minimum` 3.18 for `cpp/`; doc + path index updates |
| 2026-04-04 | **P6.2 Serf in-proc:** `BENCHMARK_CPP_EMBED_SERF` + `Serf/src` → shared **`serf`**, `research_cli_serf_embed.cpp`, `WEB_COMPRESSION_EMBED_SERF`, `BUILD_RPATH`; `bench_serf` tries in-proc before `sfz` when `WEB_COMPRESSION_SERF_BENCH=1` |
| 2026-04-04 | **P6 milestone closed (docs):** Current state + Issues **#20–#21** ✅, priority table **P6 ✅**, Milestone P6 **closed** + acceptance checklist, path index / References / suggested titles aligned |
