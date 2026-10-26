# Full-135 streaming HCB30 driver

This directory contains the phase-1 full-column driver for the existing
`harness_combo_pilot_v1` runtime. It does not change that runtime or its
catalog. The measured candidate set is exactly its 30 fixed recipes:

- `zstd3` and `lz4` are fixed baselines;
- the 18 entries with `new_combination=false` form the local control set;
- the 12 entries with `new_combination=true` are the new-combination set.

This is not the frozen 85-algorithm, 97-operator, or 182-observed-pipeline
oracle, and it is not an operator-by-algorithm Cartesian product. The driver
never combines independent oracle optima.

## Build

Compile this `main.cpp` with the unchanged pilot `runtime.cpp` and the same
include paths and libraries already used by the deployed pilot executable.
For example, add this source in place of the pilot `main.cpp` in that build:

```text
cpp/harness_combo_full135_v1/main.cpp
cpp/harness_combo_pilot_v1/runtime.cpp
```

C++20, OpenSSL crypto, Zstd, LZ4, and the existing native graph/OpenZL
dependencies are required. No additional library is introduced by this
driver.

## CLI

```bash
harness_combo_full135 catalog --output CATALOG.json
harness_combo_full135 run-column --task TASK.json --out RESULT_DIR \
  --warmups 1 --repeats 3
harness_combo_full135 verify --input COLUMN.hcf --output RECEIPT.json
harness_combo_full135 selftest --out SELFTEST_DIR
```

`selftest` creates its own two-block float64 fixture. It covers nonzero source
file offsets, a non-byte-aligned missing bitmap, signed zeros, an exact NaN
payload, infinities, a subnormal, full HCF1 commit, and fresh-process decode.

## Column task v1

The production task is one row from the frozen `COLUMNS.json`, plus split and
provenance bindings. It contains no oracle label.

```json
{
  "schema": "harness-combo-full135-column-task-v1",
  "scope": "legacy_current135",
  "phase": 1,
  "column": {
    "dataset_id": "Air-Quality",
    "column_id": "...",
    "column": "C6H6(GT)",
    "dtype": "float64",
    "manifest_path": "/absolute/path/to/column-manifest.json",
    "manifest_sha256": "...",
    "original_numeric_bytes": 74856,
    "valid_value_count": 9357
  },
  "split": {
    "name": "development",
    "fold": 0,
    "source_group": "..."
  },
  "bindings": {
    "columns_path": ".../COLUMNS.json",
    "columns_sha256": "...",
    "split_path": ".../split_config.json",
    "split_sha256": "...",
    "oracle_reference_path": "benchmark/reference/recommendation_ground_truth.json",
    "oracle_reference_sha256": "...",
    "dataset_set_sha256": "..."
  }
}
```

`split` and `bindings` are copied to the result as provenance. They are never
used as labels. A confirm task is executable for final validation; the driver
does not read any confirm oracle label or tune a rule.

## Streaming and measurement contract

The manifest array is the column order. Within every entity, `block_index`
must be consecutive from zero. The driver reads each values/bitmap region once
per run and verifies its manifest SHA-256 before any candidate sees it. It
holds one source block and one candidate frame at a time, so a large logical
column has no whole-column allocation or 8 MiB sampling limit. A single source
region retains the pilot runtime's 256 MiB stream bound. The fresh verifier has
an address-space limit of 2 GiB.

All pending candidates are evaluated during the same source pass. For each
candidate and repeat, encode/decode wall and thread-CPU nanoseconds are summed
over every source block; the reported median is taken over those complete
column sums. Source IO/SHA, container writes, cold fresh-exec verification,
and total parent execution are separate fields.

Any ineligible block, domain rejection, encode/decode error, nondeterministic
frame, same-process mismatch, or fresh-process mismatch makes the candidate
ineligible for strict ranking. No fallback result enters the ranking.

## HCF1 container and receipts

Every successful candidate produces one self-contained HCF1 column container:

1. deterministic column/candidate/manifest header;
2. one complete HCB1 frame and its source binding for every manifest block;
3. a deterministic footer binding the ordered frame sequence.

The actual HCF1 file size is `container_bytes`, including its metadata. A
fresh `exec` of `verify` receives only that container and writes a receipt with
the ordered restored-block digest. `run-column` compares that digest against
the digest independently accumulated from verified source regions.

Successful candidate checkpoints include task, catalog, container, receipt,
and ordered-source hashes. A rerun skips only checkpoints whose files and
hashes still match. Final `RESULTS.jsonl` and `RESULT.json` are written through
temporary files and atomic rename. All successful containers remain on disk.

`RESULT.json` reports the bytes/encode-time Pareto set, separate best entries
for the full HCB30 set, control18, and new12 set, plus fixed Zstd3/LZ4 rows.
Best-by-bytes and best-by-time are independent summaries and are explicitly
not presented as a joint optimum.
