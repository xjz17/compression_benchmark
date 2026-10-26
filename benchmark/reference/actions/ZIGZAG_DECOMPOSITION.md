# Explicit ZigZag stages (2026-09-15)

The current implementation-equivalence directory is `operator_implementations_106.json` (30 families, 106 unique A IDs). It is separate from the executable action inventory and graph slots. The preceding 108-item manifest is retained as a historical version.

| Historical entry | Current composition | Boundary |
|---|---|---|
| A016 / LECO predictor | A016 → A046 | Regression produces `float64` coefficients and `int64` residuals; ZigZag transforms only residuals into `uint64`. |
| retired A017 / REGER | A028 → A046 | Pure lag-1 delta, then ZigZag; first value and reorder indices bypass the mapping. |
| retired A017 / BOS64 | A028 → A046 | Pure lag-1 delta, then ZigZag; first value and optional scale bypass. |
| A071 / signed reference integer code | A046 → A023 | Signed-to-unsigned mapping, then unsigned ULEB128. A071 is no longer an independent implementation. |
| A074 / Snappy length varint | A023 | The same unsigned ULEB128 core; a scalar `uint32` raw length is an I/O profile. |

The ULEB128 raw-value, DBP-residual and Snappy-length adapters share `cpp/src/unsigned_leb128.hpp`. The ZigZag adapters share `cpp/src/zigzag.hpp`. Reused A IDs retain their equivalence meaning. A071 and A074 are retired crosswalk entries, not additional selectable items.

The current policy registry is `cpp-canonical-127-v3`. It preserves slots 0–126, reserves 5, 18, 35, 87 and 90, exposes 122 current actions, and retains STOP=127 with a 128-value policy output. The graph registry is `operator-taxonomy-30-v6`; it preserves all 138 graph slots, with 125 codec descriptors, five retired descriptors, five probability providers, and three special operations. Old selectors migrate through ordered recipes carrying I/O profiles; old model logits are not automatically interchangeable with new semantics.

Historical frames remain decodable. New selection rejects the retired identity and fused signed-varint reference actions. Compatibility entry points in the LECO/REGER/BOS native code compose the separated steps and preserve existing public wire bytes. The new generic A046→A023 graph carrier is not claimed to be byte-identical to the old composite reference OPA1 carrier.

Verification evidence is in `analysis_outputs/zigzag_decomposition_20260915/`: `cpp_leco_reger_bos/golden_report.json`, `shared_core_validation/report.json`, and the Python/graph validation reports. Native before/after checks cover empty input, signed extremes, multiple blocks, scaling, byte equality and cold inverse decoding. No server benchmark or compression/decompression timing experiment is claimed by these checks.

The accompanying PDF appends all 86 method labels and their current source-audited action chains, with integer/float differences. Reference/proxy stages, method residual cores and unsupported exact type variants are explicitly marked. The FastPFor A111→A020 semantic decomposition still has a fused executable stage and is not presented as independently wired.
