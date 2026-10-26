# Frozen GRPO policy inference, version 1

This standalone C++20 library executes the existing float64 policy
`tanh(tanh(x W1 + b1) W2 + b2) W3 + b3`, with exactly two 128-wide hidden layers.
It uses only the C++ standard library. Loading, forward evaluation, legal masks,
stable softmax and deterministic action ranking do not invoke Python, subprocesses,
libtorch or a GPU. No third-party source was copied into this component.

This component consumes **already constructed** task features, state features and
three objective weights. It does not construct graph features, select a complete
operator combination, execute graph edges, train a policy, or certify a complete
compression pipeline. The complete C++ pipeline must integrate those components
and measure their time under its declared common boundary.

## Build and export

```sh
cmake -S cpp/native_policy_v1 -B /tmp/native-policy-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/native-policy-build --parallel 2
ctest --test-dir /tmp/native-policy-build --output-on-failure
python3 scripts/export_weighted_grpo_native_policy_v1.py \
  --checkpoint /absolute/path/policy.json --output /absolute/path/policy.wgpn
python3 scripts/test_export_weighted_grpo_native_policy_v1.py
```

The exporter uses the existing checkpoint loader and its canonical payload SHA256,
shape, configuration and finite-parameter checks. Current policy checkpoints are
JSON with base64 little-endian float64 arrays. Optimizer NPZ state is not an inference
checkpoint and is not exported. The exporter preserves all PolicyIdentity fields
and records both original checkpoint file and payload SHA256. It never overwrites
an existing native model. Training metadata remains recoverable from that pinned
original checkpoint; optimizer state is not required for inference.

The tests build real C++ unless `NATIVE_POLICY_CLI` names an existing binary. Tests
use synthetic frozen parameters, not a trained model or compression benchmark.

## C++ API

```cpp
#include "policy.hpp"
namespace np = weighted_grpo::native_policy_v1;
// Values come from the integrating runtime's independently frozen contract.
np::RuntimeIdentity expected{
    registry_fingerprint, np::feature_schema, action_count, stop_action, 18, 28
};
const auto policy = np::Policy::load(model_path, expected);
// features: task[18], state[28], objective weights[3], in that order.
// legal_mask: unchanged registry positions, each byte exactly 0 or 1.
const np::Result result = policy.infer(features, legal_mask);
```

The static CMake target is `weighted_grpo_native_policy_v1`, with its include path
exported for consumers. `Policy` exposes immutable identity/origin accessors; each
inference owns its scratch memory, so a loaded `const Policy` can be shared across
threads. There is no environment-based fallback or implicit identity migration.
All runtime identity fields must match, including action count, STOP ID and both
feature dimensions. Data/root fingerprints remain in the model identity for
upstream provenance checks.

Features must be finite; the final three entries must be nonnegative and sum to
one within absolute tolerance `1e-12`, matching the reference weight contract.
An empty, wrong-width or nonbinary mask is rejected. Masked actions have probability
exactly zero and never appear in the ranking. STOP is an ordinary masked action:
the caller must explicitly permit it when legal. Finite affine overflow is an error.

Ranking uses descending **logit** and ascending unchanged action ID for exact ties.
This retains the mathematical ordering when a very small softmax probability
underflows to zero. No epsilon tie rule or random sampling is introduced. Binary
repeats with the same build and inputs are deterministic; cross-platform libm/BLAS
rounding need not be bit-identical, so numerical comparison uses stated tolerances.
The library disables fast-math and floating-point contraction for GCC/Clang; MSVC
uses strict floating-point mode. It does not assert ranking stability for distinct
logits separated by less than cross-platform numerical error.

The model's SHA256 detects corruption, not authorship. Integrations should pin the
export receipt's full-file hash alongside their runtime registry/feature contract.
Model loading/verification is initialization; callers determine and disclose whether
it is outside their repeated measurement boundary. `infer` includes affine layers,
tanh, mask validation, softmax and sorting. It does not supply a separate compression
time or denominator and does not add its time to another inclusive timer.

## Binary model format

All integers and IEEE754 float64 values are little-endian. There is no native struct
padding. A text field is `uint32 byte_length` then nonempty strict UTF-8, without NUL.

1. Eight-byte magic `WGPNAT1\0`, followed by `uint64 body_bytes`.
2. Five text fields: checkpoint schema, checkpoint payload SHA256, checkpoint file
   SHA256, registry fingerprint, feature schema.
3. Six uint32 fields: action count, STOP ID, task feature count, state feature count,
   first hidden width, second hidden width.
4. Training data fingerprints, then root fingerprints. Each is `uint32 count` and
   that many `(text task_id, text fingerprint)` pairs, sorted with unique task IDs.
5. `uint32 parameter_count = 6`. For each parameter in `W1,b1,W2,b2,W3,b3` order:
   `uint32 rank`, `rank` uint32 dimensions, `uint64 element_count`, then that many
   row-major float64 values. Shapes must exactly match identity, including 128×128 W2.
6. SHA256 of all preceding bytes, as 32 raw bytes. No trailing bytes are allowed.

The parser caps model size at 256 MiB, input/action dimensions at 65536, identity
text at 1 MiB per field and fingerprint lists at 65536 pairs. It checks lengths
before allocation, architecture, schemas, sorted unique fingerprints, checksum,
finite parameters and exact end-of-file. Unsupported versions fail explicitly.

## Comparison CLI

`weighted_grpo_native_policy_cli inspect MODEL` emits the full validated model
identity and checkpoint provenance as JSON. Inspection cannot run inference.

`weighted_grpo_native_policy_cli infer MODEL` requires eight named options:
`--registry`, `--features`, `--task-dim`, `--state-dim`, `--actions`, `--stop`,
`--input`, `--output`. The expected identity must come from the runtime. Output is
created only after successful validation/inference, and existing output is refused.

Input: eight-byte `WGPINP1\0`, uint32 row count and input width, followed per row by
input-width float64 features and action-count uint8 mask entries. Output: eight-byte
`WGPRES1\0`, uint32 row count and action count, followed per row by action-count
float64 logits, action-count float64 probabilities, uint32 legal count and that many
uint32 ranked IDs. Input is capped at 64 MiB and output allocation at 128 MiB.
These files are comparison fixtures, not a graph frame or a production hot-path API.

See `VALIDATION.json` for the recorded build, numerical evidence and test scope.
