# Native composite candidate v1

This component exposes five directly linked whole-algorithm candidates from
`native_baseline86_v1`: XPACK, Blosc2, Blosc, Libdeflate and Zstd.  They are coarse
complete backends, not atomic operators, and this registry assigns no graph
action IDs.

`Selector::select(task_features, weights)` is the C++ boundary for a separately
frozen weight/feature-conditioned policy.  The versioned task vector has 18
features sampled from at most 3 KiB of the compact numeric root. `measure()`
extracts it when requested and calls the selector inside each complete encode
sample, so feature extraction and inference are charged to compression time.
The default wire is the existing complete NGF1 frame. Decode reads the selected
backend, type, row count, null bitmap and configuration from NGF1 and never
reruns the selector. The policy identity and weights belong in the experiment
receipt because they are not needed for recovery. An optional NCC1 audit mode
embeds those fields and the candidate-registry SHA, paying another envelope and
SHA-256; it is not the default benchmark path.

Candidate source, implementation version, fixed configuration, linked status
and one-thread contract are reported by `native_composite_candidate_cli
inventory`.  The registry SHA-256 covers those live baseline records and is
stored in every NCC1 frame.  Blosc/Blosc2 explicitly configure one thread;
XPACK and Libdeflate use synchronous one-context APIs with no worker pool.
The panel's same-run Zstd comparator uses `measure-direct`, which invokes
baseline86 encode/decode directly and bypasses selector, weight validation,
virtual dispatch and the composite decoder's extra identity inspection. Its
complete NGF1 bytes are asserted equal to the fixed-selector wire in tests.

`train_only_objective_linear_v1()` transcribes the five three-term coefficients
from the train-only policy with SHA-256
`c76f74104ec651705f3e772137280e45d1f9746a44c2714c5f5df50c422cca5b`.
It performs the five-arm dot products and argmin inside timed encode. It does
not use task features; feature extraction is therefore skipped for this exact
model. A later feature-conditioned model sets `uses_task_features()` true and
receives the 18 features plus the same three weights.

The CLI accepts `train-linear-v1` in place of a fixed backend name for `encode`
and `measure`. Its JSON receipt records the exact selector identity while its
output remains the minimal complete NGF1 wire.

`DataAwareLinearSelector::load()` verifies the exact model-file SHA and its
train-only, feature, candidate-order, scale, clamp and matrix contracts before
storing parameters. Loading is outside the timer. The 18-feature extraction,
five 3x18 residual predictions, clamping, weighted objective and tie-ordered
argmin execute inside timed encode. The frozen model can be selected in the CLI
as `data-aware-v1@/absolute/path/to/model.json`.
The `select` diagnostic command exposes the exact feature vector, five predicted
three-cost vectors, weighted scores and chosen backend for cross-language
audits; it is outside the compression timing path.

This prototype does not connect the candidates to the 6,373-action policy,
does not decompose them into the PDF102 operator inventory, and supplies no
claim about validation, Web20 or independent-test performance.  A production
learned selector must have a new frozen identity and must be trained/evaluated
without using the Web20 development probe for teacher labels or checkpoint
selection.
