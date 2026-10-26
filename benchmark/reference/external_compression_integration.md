# external/compression ↔ benchmark

Third-party codec sources under `external/compression/` and
`external/submodules/` are pinned by [`.gitmodules`](../../.gitmodules). Initialize
them with `git submodule update --init --recursive`. Project-specific changes
live under [`patches/`](../../patches/) and are documented below.

| Upstream (local path) | Report key(s) | Stack | int32 column | float32 column | Notes |
|------------|---------------|-------|--------------|----------------|--------|
| FastPFor / JavaFastPFOR | `ext_javafastpfor`, `ext_fastpfor_float` | Java | `ext_javafastpfor` (integer array) | `ext_fastpfor_float` (`floatToRawIntBits` → same integer codec; bitwise lossless) | Ships in shaded `benchmark-java` JAR (`me.lemire.integercompression:JavaFastPFOR`). |
| buff ([Tranway1/buff](https://github.com/Tranway1/buff)) | `buff_float_scalar`, `buff_float_simd` | Java / Python / C++ | — | optional | Rust nightly `comp_profiler` in `external/compression/buff/database` (`cargo +nightly build --release --bin comp_profiler`). Rows emit when `WEB_COMPRESSION_BUFF_BENCH=1` or when the binary exists (Java/Python helpers set this). On x86_64, `buff_float_simd` uses the upstream `buff-simd` CLI mode (AVX2); on AArch64 (Apple Silicon) the same key reuses portable `buff` totals unless `WEB_COMPRESSION_BUFF_SIMD_MODE=buff-simd` (e.g. Rosetta x86 binary). |
| chimp | (legacy paper rows omitted) | Java / Python / C++ | — | — | Git submodule `external/compression/chimp` remains for optional golden export (`scripts/export_chimp32_golden_java.sh`). **Benchmark classpath does not use it.** Ranked int/float row is TsFile **`CHIMP`** (shaded JAR / `tsfile_encoding/` / `chimp32_paper_codec.*`). **`elf_on_chimp`** uses Chimp XOR classes **bundled in the Elf JAR**, not this submodule. |
| zfp | `ext_zfp_zfpy`, `ext_zfp_tol_1e-*` | C++ / Java / Python | — | optional | Submodule `external/compression/zfp`; **C++** in-process (`zfp::zfp` via `cpp/`); **Java** standalone `zfp` CLI (same cmake as README Research codecs); **Python** pip **`zfpy`** (`requirements.txt`). Override CLI with `WEB_COMPRESSION_ZFP`; skip `WEB_COMPRESSION_ZFP_BENCH=0`. |
| c-blosc | `blosc_blosclz` | C++ / Java / Python | int64 wire | float32 wire | string UTF-8 bytes | Submodule `external/compression/c-blosc`. `python scripts/build_blosc_col_bench.py` → `build/bin/blosc_col_bench` for Java subprocess; C++ Blosc1 in-process via `cpp/`. Skip `WEB_COMPRESSION_BLOSC_BENCH=0`. |
| c-blosc2 | `blosc2_blosclz` | C++ / Java / Python | int64 wire | float32 wire | string UTF-8 bytes | Submodule `external/compression/c-blosc2`. `python scripts/build_blosc2_col_bench.py` → `build/bin/blosc2_col_bench`; C++ subprocess. Skip `WEB_COMPRESSION_BLOSC2_BENCH=0`. |
| bitshuffle | `ext_bitshuffle_lz4` | Python | — | optional | `bitshuffle` + LZ4. |
| ALP | — (planned: native glue) | C++ | planned | planned | Header-driven column codec (`alp.hpp`); needs a small JNI or subprocess helper that runs ALP encode/decode on one column buffer. |
| [cmix](https://github.com/byronknoll/cmix) | `ext_cmix` | Java / Python / C++ | — | float32 column as raw bytes | **CPU-only** (no GPU). Submodule `external/compression/cmix`; build `cmix` (`make`). Opt-in **`WEB_COMPRESSION_CMIX_BENCH=1`**. Optional **`WEB_COMPRESSION_CMIX`**, **`WEB_COMPRESSION_CMIX_TIMEOUT_SEC`**, **`WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES`** (regenerate defaults **65536** prefix), **`WEB_COMPRESSION_CMIX_NO_PREPROCESS=1`** (uses `cmix -n` for faster runs). High RAM. |
| [MSDZip](https://github.com/huidong-ma/MSDZip) | `ext_msdzip_www25` | Python | — | float (needs **PyTorch**; CUDA recommended) | Optional clone `external/compression/MSDZip` (not a git submodule); **`WEB_COMPRESSION_MSDZIP_BENCH=1`**. Install **`torch`** in the backend venv. Optional **`WEB_COMPRESSION_MSD_MAX_COLUMNS`** (only first *N* columns, for shorter staticdata runs), **`WEB_COMPRESSION_MSD_SUBPROCESS_TIMEOUT_SEC`**, **`WEB_COMPRESSION_MSD_MIN_COLUMN_BYTES`** (defaults to 200000 on CPU when CUDA is absent, else 500000). |
| [MSDLC](https://github.com/huidong-ma/MSDLC) | `ext_msdlc_icassp25` | Python | — | float (needs **PyTorch**; CUDA recommended) | Submodule `external/compression/MSDLC`; **`WEB_COMPRESSION_MSDLC_BENCH=1`**. Same env vars as MSDZip. |
| [OAR DAC'23](https://github.com/mynotwo/Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework) | — | — | — | — | Submodule `external/compression/Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework`; upstream is library / `absl` entrypoints only — **no benchmark row** until a file in/out CLI is wired. |
| FastLanes | — | C++ | — | — | SIMD layout; not a drop-in for raw CSV columns here. |
| fsst | — | C++ | — | — | String-oriented. |
| btrblocks | — | C++ | — | — | Full engine bundle. |
| SZ | — (planned: CLI/JNI) | C | `testint_compress` style | `sz` CLI `-f` | Default SZ is **error-bounded (often lossy)**. Int path uses `SZ_INT32` examples; float path uses `sz -z -f -1 <n>`. Wire via a small native helper or `ProcessBuilder` once `sz.config` and binaries from `external/compression/SZ` build are available. |
| ndzip | — (planned) | C++/SYCL | — | primary | nd-array GPU/CPU codec; needs 1D column adapter + build. |
| LogShrink | — | Python/C++ | — | — | Log text pipeline; optional clone `external/compression/LogShrink` (not a submodule). |
| FPCompress | — | CUDA | — | — | GPU research code. |

Java launches with `java -cp … org.example.Main`, so `WEB_COMPRESSION_JAVA_CLASSPATH_APPEND` (or `backend/lib/java_classpath.append`, one path per line) can add research JARs (e.g. Elf, Camel).

### Serf (optional clone + hang fix)

Clone [Spatio-Temporal-Lab/Serf](https://github.com/Spatio-Temporal-Lab/Serf) to `external/compression/Serf` when you need `sfz` or **`BENCHMARK_CPP_EMBED_SERF`**. If `sfz` or embedded Serf benches hang on malformed bitstreams, apply the repo-local diff:

`git -C external/compression/Serf apply "$(git rev-parse --show-toplevel)/patches/serf-decompressor-hang-fix.patch"`

**Tracked in-repo deltas (upstream vs our bench):** `patches/serf-decompressor-hang-fix.patch` (Serf C++ sources listed in the patch).

### Paper-derived GitHub links

The development workspace can build a candidate repository inventory with
[`scripts/ingest_github_from_paper_pdfs.py`](../../scripts/ingest_github_from_paper_pdfs.py).
The generated inventory and source PDFs are not included in this code-only
repository. Treat such an inventory as an initial screen only: citation links
and keyword heuristics can miss or misidentify the canonical repository.

Before adding or prioritizing a new codec checkout, **manually confirm** the paper’s **artifact**, **code availability**, **supplementary material**, or **GitHub** callout (often a named paragraph or footnote) matches the repository you intend to clone. Add the pinned dependency to [`.gitmodules`](../../.gitmodules) and record any local patch under `patches/` with a short note in this file.

### Staticdata scripts hang or never finish

`scripts/regenerate_staticdata_{java,python,cpp}.py` default **`WEB_COMPRESSION_CMIX_BENCH` to `0`** so regeneration does not run cmix (avoids long runs). Enable **`export WEB_COMPRESSION_CMIX_BENCH=1`** when you want **`ext_cmix`** rows; optional **`WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES`**, **`WEB_COMPRESSION_CMIX_TIMEOUT_SEC`**, **`WEB_COMPRESSION_CMIX_NO_PREPROCESS`**. MSDZip / MSDLC default to **`0`** (no **`ext_msdzip_www25`** / **`ext_msdlc_icassp25`** rows in `*_python_compression_result.csv` until enabled). Set **`WEB_COMPRESSION_MSDZIP_BENCH=1`** and **`WEB_COMPRESSION_MSDLC_BENCH=1`** after **`pip install torch`** (and optional clones under `external/compression/MSDZip` / `MSDLC`). GPU is strongly recommended; CPU is supported but slow. Use **`WEB_COMPRESSION_MSD_MAX_COLUMNS`** if a full run would take too long.

If you previously exported overrides in the same shell, run **`unset WEB_COMPRESSION_MSDZIP_BENCH WEB_COMPRESSION_MSDLC_BENCH`** (or open a fresh terminal) before `python3 scripts/regenerate_staticdata_*.py` (or `python` if that is your project interpreter).

Staticdata scripts skip cmix by default (**`WEB_COMPRESSION_CMIX_BENCH=0`**). To run cmix: **`export WEB_COMPRESSION_CMIX_BENCH=1`**.

If **`sfz`** / Serf is slow or hangs, use **`WEB_COMPRESSION_SERF_BENCH=0`**, **`WEB_COMPRESSION_SERF_SHELL_TIMEOUT_SEC`**, or **`WEB_COMPRESSION_CPP_SKIP_SZ_SERF=1`** (C++ path); see `regenerate_staticdata_cpp.py` header.

### Regenerating static rankings

After changing Java codecs, rebuild the JAR, copy to `backend/lib/`, and re-run the batch that writes `backend/dataset/**/_compression_result.csv` so the Home page picks up new `algorithm` rows.
