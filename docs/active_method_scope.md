# Active method scope

As of 2026-10-10, new method comparisons use 85 active methods under `paper164-active85-glza-retired-v1`. The canonical policy is `benchmark/reference/methods/active_method_policy_v1.json`, SHA-256 `f48677a529247a165898bce319f3e20ef6ba9c305552676344f3ead39d0bfc5d`. GLZA is retired uniformly for engineering stability and unbounded runtime; its recorded ratios are preserved.

Use these entry points for new work:

- `scripts/benchmark_active_combinations.py` for the active composition benchmark. It binds policy provenance in its sidecar, column results and manifest.
- `scripts/active_primitive_operator_runtime.py` for active method-to-pipeline projections.
- `scripts/physical_operator_runtime_v2.py` for current physical operator search. Its active compressor selection excludes GLZA while historical wire IDs remain decodable.
- The current Python/C++/Java product benchmark gates and catalog views, which apply the same retirement policy.

The following remain historical compatibility or replay surfaces: `benchmark_86_combinations.py`, `primitive_operator_action_runtime.py`, `prepared_block_baseline86_*`, `benchmark_canonical206_fixed_algorithms.py`, and the existing canonical206 recipe/depth10 commands that consume a frozen registry. They may still expose an old GLZA label or a shared native-LZ parameter profile. Their output does not establish an active85 experiment. New comparisons must derive and record the current policy before using historical results or recipe catalogs.

Canonical206 atomic search, exhaustive graph search, and `unified_action_runtime.py` operate on shared underlying operators. A generic LZ stage or a window parameter shared with GLZA is not the upstream GLZA compressor and remains available to other algorithms. Do not remove shared actions or change historical wire IDs when retiring a method.

Frozen source versions, 86-method manifests, receipts, binaries, raw results, and third-party GLZA source remain immutable historical evidence. An archived partial frame is not an admitted successful result. The active V9 recovery controller uses a separate namespace and excludes GLZA; it preserves migration, smoke, budget, cold verification, guardian, and terminal-result gates.
