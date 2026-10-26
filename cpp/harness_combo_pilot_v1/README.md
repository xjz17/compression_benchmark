# Harness-guided combination pilot v1

This additive C++20 pilot reuses the deployed `GraphSession`,
`PolicyDispatcher`, source-exact typed operators, Zstd3, LZ4 frame and complete
NGF framing. It does not modify the 102-entry catalog, the unified deployment,
the frozen oracle or any existing experiment binary.

The 30 explicitly registered recipes contain 18 controls and 12 new-to-this-pilot
combinations. Here "new" is a comparison-pool label, not a literature novelty
claim. Examples are semantic XOR -> bitpack/A048 -> terminal codecs and
dictionary -> bitpack -> terminal codecs. Deliberately raw first ports are
part of a declared strict recipe, not an implicit raw fallback.

`catalog()` describes exact recipes, dtype eligibility and backend assignment;
`encode(id, Block)` returns a complete self-contained frame and strict/fallback
audit; `decode(frame)` validates the bound recipe before reversing it. A
literal carrier XOR -> columnize chain is not registered because its physical
types do not compose. NNCP and the previously excluded model/probability
families remain disallowed in both directions. Raw harness fallback is rejected.

## Frozen experiment

`scripts/prepare_harness_combo_pilot_v1.py` selects only metadata from the
frozen 135-dataset source-group split, without reading oracle outcomes.
The pilot has 41 complete columns, 60,049,932 numeric bytes, three source
dtypes, each 16 KiB..8 MiB. It retains all rows, null bits and source value
bits and joins the existing physical blocks into one complete-column frame.
The source manifests and individual payload regions are SHA-checked.

* Discovery: development folds 0..2, 23 columns. Fit a simple dtype-specific
  candidate prior by median of source-group median bytes/Zstd3 ratios;
  candidates must succeed on all applicable discovery columns.
* Tuning: development fold 3, 9 columns. Compare the predeclared prefix-k8,
  multi-window-k4/k8/k16 strategies and three direct baselines. The isolated
  second run also compares 8192/32768-value windows; this follow-up was motivated
  by first-run tuning results, not validation or confirm labels.
* Locked development validation: fold 4, 9 columns. Freeze one strategy from
  tuning before reading its validation outcomes or running its same-pool
  full-column oracle. This is not external validation or confirm evaluation.

The 45-cell target has four retained coverage gaps in fold 0. Confirm is not
used to fill them. The immutable plan records repeated source groups within
cells. Its existing source groups do not cross discovery/tuning/validation.

The proposal prior is deliberately simple and is **not** the earlier
`PreparedModel`. The question is whether measured feedback improves this
prior, and whether the expanded recipe pool improves the old control pool.

## Online feedback, constraints and accounting

Each strategy always proposes Zstd3 and LZ4 plus candidates in the frozen
prior order. Budget k means a hard upper bound on candidate count, not elapsed
milliseconds. A candidate is tried on one prefix or three nonoverlapping
head/middle/tail valid-value windows of up to 1024, 8192 or 32768 values. Actual candidate and
window counts are saved. The minimum total complete sample-frame bytes / total
sample numeric bytes wins; only strict bit-exact results are eligible.

Tuning selects minimum aggregate bytes subject to total observed online CPU
at most 3x fixed Zstd3 and worst-source byte ratio no worse than the unprobed
prior. CPU breaks ties. The prior is an explicit fallback if no feasible
strategy exists; the receipt records this case. These are pilot development
selection rules, not a theoretical guarantee on unseen columns. Validation
reports realized violations separately.

After selection, the complete column is actually encoded and decoded; only a
committed frame verified in a fresh exec receives a successful result receipt.
Full-column failure or bypass causes an explicitly charged Zstd3 fallback.
No policy reads the evaluation cost matrix to make its online decision.

Full matrices use one warmup and three timed encode/decode repetitions; the
median and every timing sample are retained. Encode includes input validation,
strict auditing, recipe preflight and complete framing. Online runs are
one-shot and report sample preparation, real encode probes, probe decode
verification, final encoding/decoding, frame commit and fresh verification.
Parent and verifier CPU and actual wall time are included in the online total.
Input file loading and parent startup are outside that total; verifier startup
is inside. Baselines receive the same commit/verification boundary and run
before/after policies in a deterministic rotation across columns. Small one-shot
timing differences are exploratory; they are not robust throughput claims.

Complete-frame size includes roots, null bitmap, binding, inverse metadata and
checksums, not only terminal payloads. The pilot's HCB1 binding and merged
column boundary differ from the frozen 3519-column historical byte contract.
Its same-pool oracle and 18-control oracle are separately named. Historical
85/97/182 reference costs may be reported as context with this mismatch
explicit, never silently replaced or treated as equivalent. Historical
operator ranking times remain null. No full operator-by-algorithm product is
claimed.

## Isolated server execution

The coordinator is `scripts/run_harness_combo_pilot_v1.py`. It compiles only
the new runtime and driver objects, linking the immutable dependency list from
the existing unified deployment's `link.txt`. No CMake target, installed package
or old dependency is rebuilt. The environment spec, exact commands, logs,
source/binary hashes and preservation snapshots live in the new run root.

On the target host, the resource wrapper is:

```sh
chrt --idle 0 taskset -c 62 nice -n 19 ionice -c 3 \
  prlimit --as=2147483648 --cpu=900 python3 source/scripts/run_harness_combo_pilot_v1.py \
  --root /path/to/run \
  --deployment /path/to/deployment \
  --mode experiment
```

The complete-column payloads stay in their original read-only locations.
Build and selftest must pass, and an independent agent must execute the saved
`WITNESS.zh.md` verbatim, before this command is accepted. The coordinator
verifies source, binary, plan and test bindings. Its serial phase order is
discovery matrix -> prior -> online tuning -> policy lock -> online validation
-> tuning/validation evaluation matrices. Running processes and settings are
observed before/after; the pilot never changes or terminates old processes.
Low priority and snapshots limit interference and expose changes, but cannot
prove zero shared-host resource contention.

Final results must be read from the completion receipt plus the independent
report, not inferred from process visibility or partial logs.
