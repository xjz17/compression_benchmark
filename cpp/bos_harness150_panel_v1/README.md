# Frozen-150 public harness panel

This standalone C++ runner measures the three explicit public Registry routes
`algorithm:BOS`, `algorithm:LZ4`, and `algorithm:Zstd` over frozen dense column
segment manifests.  It does not alter the unified runtime or a root CMake file.

The runner accepts the two frozen clean150 measurement-manifest schemas:
`native-baseline86-scaled-column-segment-manifest-v1` for the float150 and
old135 integer inputs, and
`project150-new15-algorithm85-entity-aware-column-manifest-v1` for the 25
new15 integer inputs. Their blocks identify the canonical dtype and exact
`values_path`, `values_offset`, and `values_length`. The accepted dtypes are
`int64`, `float32`, and `float64`; the runner performs no cast or widening.
Every dense block must contain at most 10,000 values. It is passed to
`unified_compression_v1::Registry` as one complete public `UCC1` frame with
W1/R3/T1. The public decoder is then called independently and the recovered
column is compared byte for byte.

The result contract is intentionally explicit:

- `compressed_to_original_ratio` is complete UCC1 bytes divided by source
  canonical numeric bytes; `compression_factor` is the inverse. The new15
  widened-integer path also reports
  `compressed_to_source_native_reference_ratio`. That second value is an
  accounting reference, not a measured source-native codec benefit: execution
  remains the frozen canonical `int64` typed plane and the panel performs no
  cast.
- encode and decode samples are the sample-wise sums over the column's frames;
  the reported time is their median.
- `strict_path` and `eligible_for_strict_selector` are derived from Registry
  execution audits.
- a BOS-native raw block remains a valid bit-exact explicit measurement and is
  reported as `lossless_native_raw`; it is not relabeled as a harness bypass.
- `results.jsonl` is an append ledger.  Each outcome is first committed as an
  independent JSON file by temporary-file rename.  `--resume` reconstructs the
  ledger from independent files after checking every file against the exact
  run configuration. For every BOS result it also fails closed unless the
  recorded Registry contract has profile/float route
  `bos-bfe-ngf-hybrid-v2` and integer route
  `project_bos2_partitioned`.
- the accepted clean150 `FEATURES.jsonl` is read directly. Every row must use
  `solid-dtype-correct-column-features-v1`; its `dataset_id`, canonical
  `dtype`, `valid_count`, `width`, unique `manifest_path`, and
  `manifest_sha256` are checked. `RUN_CONFIG.json` binds the complete list
  file path and SHA-256, so resume cannot silently substitute another list.
  Its `runtime_contract` records the exact BOS profile, float route, integer
  route, implementation, and configurations; these are deployment execution
  identities, not substitutes for a native benchmark row's measurement-route
  provenance.
  This input form must be the sole manifest source and requires
  `--expected-columns` (3,644 for the frozen clean150 file).
- the executable fixes common native thread-control variables to one and, on
  a Linux benchmark host, enforces a 2 GiB `RLIMIT_AS` ceiling by default. Local
  non-Linux validation records that this kernel mechanism was unavailable.

Build and self-test:

```sh
cmake -S cpp/bos_harness150_panel_v1 -B /tmp/bos-harness150-panel-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/bos-harness150-panel-build -j2 \
  --target bos_harness150_panel
ctest --test-dir /tmp/bos-harness150-panel-build \
  -R '^bos_harness150_panel_selftest$' --output-on-failure
```

Full frozen-manifest usage (single process and one native thread):

```sh
/tmp/bos-harness150-panel-build/bos_harness150_panel \
  --manifest-list /path/to/FEATURES.jsonl \
  --output-dir /path/to/output/panel \
  --expected-columns 3644 \
  --address-space-limit-bytes 2147483648 \
  --methods algorithm:BOS,algorithm:LZ4,algorithm:Zstd
```

Use `--canary 3` for the first three sorted manifests and `--resume` only with
the same SHA-bound manifest-list source, manifest set, candidate list, W1/R3
settings, and output directory.
`--manifest PATH` may be repeated. `--manifest-list PATH` accepts either a
plain newline-delimited path list or a JSON plan containing `jobs[].manifests`.
The 3,644-column list must be the union of the 3,158 frozen float manifests,
461 old135 integer manifests, and 25 new15 entity-aware integer manifests.

This runner deliberately has no selector or model code.  It produces the
same-contract three-candidate cost panel that later recommendation evaluation
can consume.  NNCP, PPMd/PPMD8, DMC, PAQ8, CMIX, and LSTM are outside this
runner's closed candidate allowlist.
