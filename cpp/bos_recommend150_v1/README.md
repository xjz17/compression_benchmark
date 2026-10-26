# BOS recommendation evaluation for frozen150

This isolated C++ driver adds the accepted replacement `BOS` measurement to
the algorithm recommender without changing the old current135 implementation.
It consumes the accepted clean150 `FEATURES.jsonl`, an immutable old135
source-group split, and partitioned column-method feeds.

## Candidate and data contract

- `experiment_scope_full85_algorithm_oracle`: all 85 measured algorithms. It is
  reference-only and includes `PPMD8`.
- `primary84`: the selectable 85-method catalog with `PPMD8` removed. The
  driver also rejects NNCP, PPMd, DMC, PAQ8, CMIX, and LSTM from every
  selectable catalog. `BOS` remains one in-place ID; its feed SHA, binaries,
  accepted upstream source/result SHA, input/timing profile, measurement route,
  and profile version are checked fail-closed. Deployment runtime routes are
  recorded separately and are never used as benchmark provenance. The old and
  new implementations are not treated as jointly measured candidates.
- `restricted3`: `BOS`, fixed `LZ4`, and fixed level-3 `Zstd` (`Zstd` in the
  accepted method feed). This is an ablation, not the primary selector.
- Space, encode time, and decode time are trained independently. The complete
  85-method oracle for one objective never supplies labels to another.

The canonical frozen current135 oracle is not loaded by this C++ trainer. It is
read only by the reporting aggregator after that aggregator independently
verifies `LOCK.json`, the CV SHA, `MODEL_INDEX.json`, and all six model hashes.
Its `best_algorithm`, `best_operator_id` under the common Zstd-3 backend, and
`best_observed_pipeline` targets remain separate.

The 43 feature-map keys are sorted lexicographically and frozen by the model
index. The 12 decimal/nonfinite fields that are inapplicable to integer rows
use a frozen value-plus-mask encoding (`0` plus an explicit missing indicator),
so no integer column is discarded. Six label-free physical features are then
appended: three dtype one-hot values, byte width, `log1p(row_count)`, and
`log1p(valid_count)`. The resulting 61 names and transform contract are stored
in `MODEL_INDEX.json`.

Feature/feed identity uses one fail-closed rule: the exact pair
`(dataset_id, column_id)` in `FEATURES.jsonl` must equal
`(dataset_id, stable_id)` in every method-feed row. `column_key`, column names,
manifest contents, and row order never act as fallbacks. Duplicate or
ambiguous identities are rejected, and the joined rows must still agree on
valid-value count and canonical input bytes.

Candidate eligibility is computed on development only. For each physical
dtype, a method must have an accepted lossless measurement on every
development column of that dtype. These dtype-specific pools are stored in the
model index and used verbatim at final inference. Confirm/external `status` and
`rank_eligible` fields can contribute to hindsight oracle accounting and
post-selection validation, but never remove or add an inference candidate.

## Leakage gate

`tune` accepts only datasets whose old135 assignment is `development` and
uses the frozen five source-group folds and the frozen cohort/legacy-group
mapping. Every fit purges legacy groups represented in confirm; each OOF fit
also purges every source and legacy group represented in its held-out fold. It
writes all six final models, the
complete OOF grid, and `LOCK.json`. A row assigned to `confirm` is rejected
before it can become a training cost. `final-eval` verifies the lock, split,
features, model index, and every model hash before opening its input feed.
New15 datasets are absent from the old split and are accepted only under the
separate `external-final` partition. The lock binds the split, cohorts,
candidates, features, development feed, CV summary, model index, and models.

For every OOF grid point the report includes total objective cost, dataset-equal
geometric regret, worst source-group regret, and CVaR95 of per-column relative
excess. Space selects the minimum-total-byte grid point that is no worse than
the frozen `lambda=1,k=0,ridge=1` baseline on all four statistics. Encode and
decode select their own minimum-total-time grid point. The complete small grid
is ridge lambda `{0.1,1,10}` crossed with ridge weight `{0,0.5,1}` at fixed
`k=8` for kNN (`k=0` for the ridge-only serialization). Component scores from
one prepared kNN ranking are reused for all three blend weights.

The selected development result is a strict 3,433-column OOF evaluation, not
an in-sample replay of the final model. Every selector and independent oracle
reports compression ratio, compression time, and decompression time for the
same chosen candidate. It also retains physical `source_native_input_bytes`
and a compressed-to-source-native reference ratio; that ratio is accounting
only and never treats int8/int16-to-int64 widening as compression gain. Frozen
candidate budgets `{1,3,5,10}` additionally
simulate the closed loop: the recommender supplies a ranked prefix, accepted
harness measurements choose the best target cost in that prefix, and probe
encode/decode costs are charged. Restricted3 naturally caps the effective
budget at three. These OOF sufficient statistics can be added to confirm and
external-final statistics for a descriptive 150-dataset aggregate, while
confirm remains the only formal final validation partition.

## Build and test

```sh
cmake -S cpp/bos_recommend150_v1 -B /tmp/bos-recommend150-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/bos-recommend150-build -j2
ctest --test-dir /tmp/bos-recommend150-build --output-on-failure
```

## Production commands

The feeds must already be partitioned without changing any measurement row.
The development feed has 3,433 old135 development columns; confirm has 86
old135 confirm columns; external-final has the 125 new15 columns.

```sh
/tmp/bos-recommend150-build/bos_recommend150 tune \
  --features /accepted/clean150/FEATURES.jsonl \
  --split /frozen/old135/split_config.json \
  --cohorts /frozen150/SPLIT_COHORTS.json \
  --candidates /frozen150/CANDIDATES.json \
  --development-feed /accepted/clean150/DEVELOPMENT_METHODS.jsonl \
  --out /isolated/run/models

/tmp/bos-recommend150-build/bos_recommend150 final-eval \
  --features /accepted/clean150/FEATURES.jsonl \
  --split /frozen/old135/split_config.json \
  --cohorts /frozen150/SPLIT_COHORTS.json \
  --candidates /frozen150/CANDIDATES.json \
  --feed /accepted/clean150/CONFIRM_METHODS.jsonl \
  --models /isolated/run/models --partition confirm \
  --out /isolated/run/CONFIRM_REPORT.json --timing-repeats 1000

/tmp/bos-recommend150-build/bos_recommend150 final-eval \
  --features /accepted/clean150/FEATURES.jsonl \
  --split /frozen/old135/split_config.json \
  --cohorts /frozen150/SPLIT_COHORTS.json \
  --candidates /frozen150/CANDIDATES.json \
  --feed /accepted/clean150/NEW15_METHODS.jsonl \
  --models /isolated/run/models --partition external-final \
  --out /isolated/run/NEW15_REPORT.json --timing-repeats 1000
```

Each final report gives aggregate compression ratio, encode/decode ns per
value, fixed LZ4 and Zstd-3 baselines, primary/restricted candidate-pool
oracles, the experiment-scope full85 algorithm oracle, decomposed
selection/pool/total gaps, and
hot `PreparedModel` ranking overhead cycled across held-out columns (warmup 1,
configurable repeats >= 5). A minimal `PreparedRegistry` contains only the
development-locked per-dtype algorithm IDs. The report separately times
single-`FeatureRow` validation plus registry resolution, hot full ranking, and
validation+registry+rank end to end. The legacy `wall`/`thread_cpu` fields keep
the rank-only boundary. All three phases start from an already-materialized
61-value `FeatureRow`: raw-column feature extraction and JSON feature reading or
generation are explicitly outside the boundary and are never represented as
online feature-extraction time. The operator97 registry and any unrelated
feature extractor are not imported. Harness probe costs remain a separate
measurement-derived accounting item.
