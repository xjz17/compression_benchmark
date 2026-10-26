# Operator harness v2

This is the file-backed native harness used for bounded real-data validation.
It reuses `native_graph_v1::GraphSession` and the source-exact
`Operator102Dispatcher`.  The timed path is C++; JSON is only the control and
receipt envelope.

The process reads one `operator-harness-request-v2` JSON object per stdin line
and writes exactly one independent response line.  A failed case does not stop
later cases.  The durable batch coordinator invokes `run` and `decode` in
different processes and compares the committed restored file with the original
sample.

`A106` through `A110` and every PPMd, DMC, PAQ8, CMIX, LSTM, or NNCP action are
permanently excluded in both directions.  CMake also forces NNCP off.  There is
no override flag.

## Run request

```json
{
  "schema": "operator-harness-request-v2",
  "command": "run",
  "request_id": "dataset__A013",
  "implementation_ids": ["A013"],
  "catalog_path": "/absolute/operator_harness_batch_catalog_v2.json",
  "catalog_sha256": "raw-file-sha256",
  "input_path": "/absolute/sample.bin",
  "initial_kind": "numeric",
  "initial_width": 8,
  "input_view": "int64",
  "dtype": "int64",
  "element_count": 128,
  "source_byte_width": 8,
  "original_numeric_payload": true,
  "recipe": {
    "policy_registry": "operator-harness-batch-v2",
    "steps": [
      {
        "kind": "stage",
        "inputs": [0],
        "stage": {
          "action": "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
          "profile": "int64",
          "parameters": {}
        }
      },
      {"kind": "stop"}
    ]
  },
  "warmups": 0,
  "repeats": 1,
  "frame_path": "/absolute/case.ngf"
}
```

The input file is never numerically converted. `initial_width` may equal
`source_byte_width`, or may divide it to expose the exact bytes as narrower raw
lanes.  Wider, non-dividing, extending, and truncating conversions return a
`gap` receipt.  A canonical source-schema JSON record is stored as an auxiliary
NGF root so dtype/count/adapter provenance is charged in the complete frame and
survives cold decode.

Every run is fail-closed against the generated v2 catalog. `catalog_path` must
be absolute and `catalog_sha256` is the SHA-256 of the exact catalog file bytes.
The binding layer separately verifies the catalog's canonical content hash and
the selected row's action, profile, normalized string parameters, input
shape/view, edge chain, and final stop.  Its exact implementation ID, catalog
hashes, and recipe hash are embedded in the charged source-schema root.

## Decode request

```json
{
  "schema": "operator-harness-request-v2",
  "command": "decode",
  "request_id": "dataset__A013__cold",
  "implementation_ids": ["A013"],
  "catalog_sha256": "raw-file-sha256",
  "recipe_sha256": "binding-recipe-sha256",
  "frame_path": "/absolute/case.ngf",
  "output_path": "/absolute/restored.bin",
  "warmups": 0,
  "repeats": 1
}
```

`decode` never accepts or reads the original input path.  `GraphSession`
validates the reconstructed root against the SHA-256 stored in NGF1, then the
harness atomically commits only the data root to `output_path`.  The batch
coordinator independently compares that file with the source sample.
The decode request must select the same single implementation ID.  Its catalog
and recipe hashes are checked against the binding embedded in the frame; decode
does not need the catalog file.

The native timing samples cover fixed-recipe GraphSession scheduling, kernels,
and complete NGF1 framing/checksum on encode, and NGF1 parse/checksum plus
inverse scheduling and kernels on decode.  File I/O, JSON handling, process
startup, recipe resolution, hashing for receipts, and the untimed fallback
audit are outside the samples.  The run response records strict-wrapper and
raw-fallback counts separately.

Build the ordinary closure with:

```sh
cmake -S cpp/operator_harness_v2 -B cpp/operator_harness_v2/build-local \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/operator_harness_v2/build-local \
  --target operator_harness_v2_cli operator_harness_v2_protocol_test -j2
ctest --test-dir cpp/operator_harness_v2/build-local \
  -R operator_harness_v2_protocol_test --output-on-failure
```

The seven residual implementations need the separately isolated codec closure:

```sh
cmake -S cpp/operator_harness_v2 -B cpp/operator_harness_v2/build-residual \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPERATOR_HARNESS_V2_ENABLE_SUPPORT_DEPS=ON
```

Unsupported or unresolved special bridges are reported by the catalog/batch
coordinator as gaps.  This driver never calls `run_special_bridge`, caches its
`target_before`, or reconstructs input from external state during decode.

The `capabilities` command's action list is the base
`native_stage_capabilities()` registry.  Dispatcher and v2 special-stage
actions extend that list, so it is diagnostic information rather than an exact
inventory of the 97 eligible catalog rows.  Catalog-bound run/decode receipts
are the support evidence.
