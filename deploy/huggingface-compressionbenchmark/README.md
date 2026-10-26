---
title: Time Series Compression Benchmark
emoji: 📊
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Compression rates and compress/decompress costs
---

# Static benchmark site

This directory contains the source for the static benchmark interface. Measured
site payloads are intentionally not part of the code repository.

## Build a local payload

Generate a result snapshot from your own accepted measurements, apply the
desired dataset scope, and build the compact browser payload:

```bash
python3 deploy/huggingface-compressionbenchmark/export_results.py
python3 deploy/huggingface-compressionbenchmark/apply_dataset_scope.py
python3 deploy/huggingface-compressionbenchmark/build_ui_payload.py
```

The builders expect their input paths and output paths through their command
line interfaces or project configuration. Do not commit generated `data.json`,
`ui-data.json`, result streams, manifests, or verification receipts.

## Interface source

- `index.html`, `style.css`, `app.js`, and `ranking.js` implement the browser UI.
- `export_results.py` converts accepted benchmark records into a site snapshot.
- `apply_dataset_scope.py` filters a snapshot to an explicit dataset contract.
- `build_ui_payload.py` creates the browser-facing payload.

Missing, unsupported, partial, failed, and non-lossless measurements must remain
distinct. Only measurements that satisfy the caller's acceptance contract may
enter rankings.
