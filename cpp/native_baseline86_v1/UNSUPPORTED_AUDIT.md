> Historical v2 integration audit. The v3 migration is implemented in
> `upstream_general_a`, `upstream_general_b`, `upstream_legacy`, and
> `upstream_specialists`. This file explains the old 25 missing adapters; it
> is not the current v3 build inventory. Query `native_baseline86_cli inventory`
> and read the new qualification receipt for current status.

# Unsupported baseline label audit

Inventory profile: `baseline86-fixed-profiles-20260919-v2`.

The catalog retains all 86 labels. This build links 61; 53 are direct same-label primary comparisons, 8 are proxy/alias rows, and 25 remain unsupported.

These rows are kept in every campaign output with null primary metrics. The reason below describes the current source/build gap; source presence alone is not implementation evidence.

## FPCmp-R

Source status: `external/compression/FPCompress/double_src/ratio-*.cpp`.

Vendored code is a standalone OpenMP file program with main and separate compressor/decompressor executables, not a reusable bounded in-process API. It has not been refactored into an independently decodable frame adapter.

## FPCmp-S

Source status: `external/compression/FPCompress/double_src/speed-*.cpp`.

Vendored code is a standalone OpenMP file program with main and separate compressor/decompressor executables, not a reusable bounded in-process API. It has not been refactored into an independently decodable frame adapter.

## Density

Source status: `external/compression/lzbench/misc/density`.

The checkout contains Rust cdylib source and a cached macOS dylib. A reproducible Linux Rust build, C ABI integration and bounded cold-decode adapter are not part of this CMake target.

## Gipfeli

Source status: `external/compression/lzbench/lz/gipfeli`.

Vendored sources exist, but the lzbench wrapper is kept in buggy_codecs.cpp and writes through an unchecked sink. No bounded adapter with malformed-stream and exact-length checks is admitted.

## GLZA

Source status: `external/compression/lzbench/misc/glza`.

Vendored multi-file GLZA 0.12 sources exist, but the legacy API and global state have not been isolated or validated for bounded in-process complete-frame cold decode.

## LZO1

Source status: `external/compression/lzbench/lz/lzo`.

Vendored LZO 2.10 has multiple LZO1 variants and levels. No fixed public-label profile and no isolated full-source adapter have been admitted; historical best-level selection is intentionally not reused.

## LZO2A

Source status: `external/compression/lzbench/lz/lzo`.

Vendored LZO2A 2.10 sources exist, but its large legacy build and work-memory/init path are not integrated into this isolated target or cold-decode matrix.

## LZSSE

Source status: `external/compression/lzbench/lz/lzsse`.

The public label spans LZSSE2/4/8, fast/optimal parsers and multiple levels. No non-post-hoc fixed profile or bounded parse-state adapter has been admitted.

## Pithy

Source status: `not present in current checkout`.

A stale ablation test references lz/pithy/pithy.h, but the Pithy source/header is absent from this checkout; there is no source that this component can build.

## Shrinker

Source status: `not present in current checkout`.

A stale ablation test references lz/shrinker/shrinker.h, but the Shrinker source/header is absent from this checkout; there is no source that this component can build.

## NRV

Source status: `external/compression/lzbench/lz/ucl`.

Vendored UCL exposes NRV2B/2D/2E plus multiple levels. The label has no predeclared variant here, and the legacy multi-file UCL build has not passed the full-frame cold-decode matrix.

## XPACK

Source status: `not present in current checkout`.

A stale ablation test references an XPACK API, but the XPACK source/header is absent from this checkout; there is no source that this component can build.

## BSC

Source status: `external/compression/lzbench/bwt/libbsc`.

Vendored libbsc has many modes and optional internal multithreading/CUDA. A single-thread fixed profile and isolated full-source build have not been admitted.

## Crush

Source status: `external/compression/lzbench/lz/crush`.

The source uses large process-global mutable match tables and an unbounded source decoder. It needs isolation and explicit safety/serialization tests before admission.

## CSC

Source status: `external/compression/lzbench/lz/libcsc`.

The only wrapper is classified by lzbench as buggy_codecs and uses custom stream callbacks without the strict truncation/error contract required by this frame runner.

## FastLZMA2

Source status: `external/compression/lzbench/lz/fast-lzma2`.

Vendored Fast-LZMA2 is a large threaded/platform-specific build including x86 assembly. No threads=1 isolated build and cross-platform cold-decode adapter has been qualified.

## Kanzi

Source status: `external/compression/lzbench/misc/kanzi-cpp`.

Vendored Kanzi is a large stream framework whose numeric levels select different transform/entropy pipelines. No fixed pipeline and single-thread memory-stream adapter has been qualified.

## Lizard

Source status: `external/compression/lzbench/lz/lizard`.

Vendored Lizard 2.1 includes its own entropy/xxhash symbols that require isolation from the graph runtime, and the public label spans levels 10-49. No fixed isolated adapter is admitted.

## LZHAM

Source status: `external/compression/lzbench/lz/lzham`.

Vendored LZHAM contains 55 C++ source files and dictionary/level choices. No fixed single-thread memory profile and bounded cold-decode adapter has been qualified.

## Lzlib

Source status: `external/compression/lzbench/lz/lzlib`.

Vendored lzlib 1.15 is a streaming API. Its required feed/drain/member-finish and trailing-input validation loops are not yet implemented in this complete-frame adapter.

## PPMD8

Source status: `external/compression/lzbench/misc/7-zip/Ppmd8*.c`.

PPMd8 requires an explicit order, memory size, restore method, allocator and range stream. No fixed single-thread profile and bounded memory-stream adapter has been admitted.

## Tornado

Source status: `external/compression/lzbench/lz/tornado/Tornado.cpp`.

Vendored Tornado is a monolithic legacy source with many method levels and command-oriented shared state. It has not been isolated behind a fixed bounded memory profile.

## Zlib-ng

Source status: `external/compression/lzbench/lz/zlib-ng`.

Vendored zlib-ng requires its generated build configuration and symbol-isolated native-mode library. That isolated target and cold-decode matrix are not present here.

## Zling

Source status: `external/compression/lzbench/lz/libzling`.

Vendored Zling is disabled by default in lzbench. Its current memory wrapper can silently truncate output, so a strict bounded stream adapter must be written before admission.

## ZXC

Source status: `external/compression/lzbench/lz/zxc`.

Vendored ZXC has a multi-file architecture-dispatch build and stateful contexts. No isolated default-level target with Linux/macOS cold-decode validation is present.

