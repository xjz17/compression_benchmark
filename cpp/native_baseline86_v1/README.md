# Native baseline-86 complete-frame runner

This component preserves the fixed 28 specialist + 58 general public labels,
while separating a label from the implementation that is actually linked.
`native_baseline86_cli inventory` is the machine-readable authority for the
current build.  Every row is one of:

- `direct_current`: an admitted current same-label implementation or a directly
  linked upstream library profile;
- `direct_generic`: a fixed direct implementation of a generic coding label;
- `direct_exact_subset`: a direct source branch/profile with a bit-exact input restriction;
- `shared_alias`: a public label intentionally sharing another method's kernel;
- `project_proxy`: a reversible project implementation that is not claimed to
  be the named upstream algorithm;
- `unsupported`: no admitted in-process whole-method adapter in this build.

Linked `direct_current`, `direct_generic`, and `direct_exact_subset` rows
can enter the main comparison only for inputs they actually restore bit-exactly.
`direct_generic` names a fixed concrete implementation of a generic label;
`direct_exact_subset` explicitly limits CORAD and BUFF to eligible inputs.

Profile `baseline86-fixed-profiles-20260919-v3` replaces the v2 missing/proxy
adapters with 23 migrated general-purpose sources, upstream Chimp and two
FPCompress CPU recipes, five explicit generic coding profiles, and two exact
subsets. The old v2 build and its results keep their original identity. See
`UPSTREAM_SPECIALISTS_SOURCE_AUDIT.md` for source-level equivalence and the
CORAD/BUFF limitations. Build availability is not proof of a single-thread
profile or of full-dataset applicability; the final Linux binary needs both
complete-frame cold-decode and thread-conformance qualification.

## Fixed profiles

Profiles are declared in `baseline.cpp::make_inventory()` and serialized into
every NGF1 recipe.  They are independent of the current train, validation and
test inputs.  In particular, this runner does not import
`BENCHMARK_86_GLOBAL_BEST_GENERAL_KEYS` or `EXACT_NATIVE_METHOD_KEYS`, whose
levels were selected from historical 20-column benchmark results.  Examples of
fresh fixed profiles are zlib level 6, Zstd level 3, Brotli quality 5, XZ preset
6, LZ4 frame level 0, LZ4 Fast acceleration 1 and LZ4HC level 9.

The baseline-owned vendored adapters also link libdeflate 1.25, FastLZ 0.5.0,
LZF, Apple LZFSE/LZVN, BriefLZ 1.3.0, LZAV 5.7, liblzg 1.0.10,
OpenSolaris LZJB 2010, TAMP 2.1.0, Bzip3 1.5.2, FSE 0.9.0 and Huff0.
The same isolated adapter library also links LZMAT 1.01, LZRW1, wfLZ Fast,
Yappy default level 10, YALZ77's default 12/64 KiB profile, QuickLZ 1.5.1 beta 7 level
1, and SLZ gzip level 1.
Bzip3 uses a fixed 1 MiB high-level block size; FSE and Huff0 use fixed 128 KiB
chunks.  Each one-byte adapter tag selects either the real upstream stream or verbatim bytes;
the verbatim branch is selected only when the upstream result is not smaller
(and for FastLZ inputs shorter than its documented 16-byte minimum).  This
fixed framing rule is included in complete-frame size and timing and is not a
historical per-column parameter choice.

## Complete-frame and timing contract

The API starts from an already resident prepared block: compact valid numeric
bytes in their original little-endian dtype, the original row count and a
one-bit-means-missing LSB-first bitmap.  NGF1 retains the method label, fixed
configuration, dtype, row count, full missing bitmap, compressed payload, graph
metadata, lengths and checksums.  Decode needs only NGF1 and the linked frozen
implementation.

`original_numeric_bytes` is the byte size of the valid numeric payload before
any transform.  Complete-frame bytes are divided by that same value for
compression ratio.  Complete C++ encode and decode nanoseconds are each divided
by that value for `ns/byte`.  A zero denominator produces JSON `null` normalized
metrics and status `N/A`.

For an all-missing block, the method payload is the canonical empty byte string.
This adapter rule keeps methods whose upstream API rejects empty input comparable:
NGF1 still records the selected method, dtype, row count and complete missing
bitmap, and cold decode reconstructs the empty numeric payload from the frame.

The timed range includes validation, per-block initialization, allocation,
compression/decompression, graph scheduling, envelope parsing/serialization and
checksums.  It excludes source-file I/O, process startup, one-time library load
and the byte comparison used for correctness.  The caller must pin each runner
to one CPU; every admitted profile is configured single-threaded.

## CLI

```text
native_baseline86_cli inventory
native_baseline86_cli encode METHOD DTYPE ROWS VALUES MISSING FRAME
native_baseline86_cli decode FRAME VALUES_OUT MISSING_OUT
native_baseline86_cli measure METHOD DTYPE ROWS VALUES MISSING WARMUPS REPEATS FRAME
native_baseline86_cli benchmark-manifest MANIFEST JSONL WARMUPS REPEATS [--frame-dir DIR] [METHOD ...]
```

`benchmark-manifest` understands the prepared dataset manifest's `blocks`
array and resolves `values_path` and `missing_bitmap_path` relative to that
manifest. With `--frame-dir`, every successful or N/A complete NGF1 frame is
published after timing at `<sha256-prefix>/<sha256>.ngf1`; each JSONL row stores
that relative path and SHA-256. Existing content-addressed files are verified
before reuse. Frame file I/O is outside the native encode/decode timing interval.
It verifies declared SHA-256 values, emits one JSONL record for every
requested label/block pair, keeps unsupported/failure rows explicit, and flushes
each record so a stopped run remains auditable.  Passing no methods preserves all
86 labels.  `decode` is the independent cold-decoder entry point; CTest encodes
and decodes its fixture in separate processes before comparing original bytes.
The cold-matrix CTest additionally encodes and decodes every linked method on
each admitted int64/float64 fixture in distinct process invocations.
