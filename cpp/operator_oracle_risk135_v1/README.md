# C++ development byte-risk recommender

This directory contains the C++20 feature extraction, weighted ridge/kNN
fitting, serialized-model prediction, and ranking implementation. Python stages
immutable caller-provided inputs and enforces resource limits. Historical run
receipts and result audits are intentionally not included in this repository.

```
risk_features COLUMNS INPUT_METADATA OUTPUT_DIR
risk_experiment selftest
risk_experiment tune DEVELOPMENT_COSTS PREFIX_FEATURES MULTI_FEATURES PROTOCOL OUTPUT_DIR
risk_experiment predict MODEL_INDEX PREFIX_FEATURES MULTI_FEATURES OUTPUT_JSONL
```

`risk_features` emits `prefix24` and `multi3` subdirectories, each containing `FEATURES.jsonl` and `FEATURES_RECEIPT.json`. It uses only development source payloads. Prefix vectors preserve all prior 24 values; multi3 has 98 values, adding deterministic whole-column head/middle/tail statistics and two column-size features. Every sample binds masks, source-region hashes, value coordinates and payload bits. Native tests include long single-block offsets, block boundaries, nulls, empty blocks and numeric extremes.

The formal design compares 18 feature/weight/scorer policies against frozen prefix controls under the frozen five-fold source split and legacy-alias purges. It minimizes total OOF charged frame bytes subject to dataset, worst-source and absolute-tail byte-risk constraints. If no feasible improvement exists, the old control remains selected. Confirm costs are neither accepted nor staged this round. OOF used for selection is not a final generalization estimate.

Scopes remain separate: algorithm84, operator97 with common Zstd3, and 181 namespaced observed slots. Full frozen reference gaps retain 85/97/182. No algorithm/operator Cartesian joint optimum was measured. Operator times remain null. The online PreparedRegistry still executes a separate 23-configuration pool. NNCP and all previously excluded actions/algorithms remain disabled.
