# Operator-86 required paths v1

`operator86_required_paths_v1.py` builds a deterministic, JSON-serializable
requirement row for each of the fixed 86 benchmark labels.  It reads the
Methods-page mechanism catalog, the 86-to-A audit, the current 102 handbook,
and `handbook_content.py`; every consumed core input receives a fresh SHA-256
receipt.

The historical audit order is also checked against the current
`BENCHMARK_86_SPECIALISTS` and `BENCHMARK_86_GENERALS` tuple literals by parsing
`primitive_operator_action_runtime.py` with `ast.literal_eval`.  This does not
import the runtime or its dependencies, and prevents the audit from validating
only itself.

The output deliberately keeps four evidence layers separate:

1. `documentary_path` is the ordered mechanism-family description from
   `method_operator_pipelines.json`.  It is useful as a required candidate
   shape, but has no byte-exact typed-port claim.
2. `audited_runtime_sequence` preserves the existing ordered runtime actions,
   branch/topology note, historical A-ID mapping, coverage grade, and source
   locators for each logical type.  It is explicitly the historical 106-era
   audit snapshot; fresh file hashes are provenance and do not revalidate its
   behavior against current code.
3. `direct_method_evidence` records an upstream/baseline or project adapter.
   `proves_decomposed_graph_path` is always false: a whole-codec wrapper is not
   an independently schedulable stage path.  Current `native_baseline86`
   behavior is marked `not_refreshed_by_this_builder` unless a separate audit
   supplies that evidence.
4. `policy_graph_registration` remains a gap until a separate current-registry
   check proves the whole path.  Historical A catalog membership never implies
   registration in the 6341-action graph.

All 86 methods have `considered_as_required_candidate=true`.  They are not
silently admitted: typed ports, decoder-visible state, inverse behavior,
source-exact decomposition, and current graph registration remain explicit
gaps.  Zstd additionally lists Huff0 literals, LL/ML/OF generation and FSE
coding, repcode history, and cross-block/frame state as required exactness gaps.
Huff0 and FSE also carry their real entropy-table/state/bitstream requirements;
their historical native-LZ proxy actions stay only in
`historical_comparison_runtime_actions` and are never emitted as candidate seed
actions.

Run the focused test with:

```bash
python3 scripts/test_operator86_required_paths_v1.py
```

Generate a review document with:

```bash
python3 scripts/operator86_required_paths_v1.py --output /tmp/operator86_required_paths_v1.json
```
