# Operator harness v1

This is a local harness over `native_graph_v1`, with a versioned registration
record for all 102 historical A ids. Nine representative mechanisms have
explicit, executable bindings: A013, A020, A023, A031, A046, A048, A077, A080 and
A111. Registration does not qualify the remaining entries for execution.
The 67 mechanisms / 7 profiles / 14 recipes / 14 adapters remain distinct.

The harness does not connect to servers, alter the existing benchmark queue, launch
tmux workers, or read/write the historical operator102 campaign. NNCP/A110 is
disabled in registry validation, native forward/inverse dispatch and the new
build. There is no NNCP enable override. Existing native builds retain their
previous default through the default-ON `NATIVE_GRAPH_ENABLE_NNCP` build option;
the harness forces this option OFF in its separate build tree.

## Build and run

From the repository root:

```sh
cmake -S cpp/operator_harness_v1 -B cpp/operator_harness_v1/build-local \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/operator_harness_v1/build-local \
  --target operator_harness_v1_cli operator_harness_v1_protocol_test -j2
ctest --test-dir cpp/operator_harness_v1/build-local \
  -R '^operator_harness_v1_protocol_test$' --output-on-failure

python3 scripts/operator_harness_v1.py registry \
  --out /tmp/operator-harness-registry.json
python3 scripts/operator_harness_v1.py demo-suite \
  --out /tmp/operator-harness-suite.json
python3 scripts/operator_harness_v1.py run \
  --suite /tmp/operator-harness-suite.json \
  --out analysis_outputs/operator_harness_v1_my_run \
  --warmups 1 --repeats 5 --timeout 30
```

The same `run` command resumes the same immutable run: committed results are
checked and retained, and only cases without a committed receipt are executed.
For changed source, binary, suite, registry, timeout or repetition settings,
choose a new output directory. Failed and timed-out receipts are retained and
are not silently retried; create a separate suite containing the desired retry
cases and use a new output directory. A nonempty unrelated output directory,
the workspace root and its ancestors are rejected.

## Registration and interface

`scripts/operator_harness_registry_v1.py` exports `load_registry()`,
`registry_by_id()`, `get_spec()` and `validate_case(case, registry=None)`.
`OpSpec` separates the historical trace id from executable action bindings,
ports, parameters, state, resource declarations and gating. Unqualified fields
are marked explicitly rather than inferred from the presence of source code.
The registry includes catalog and resolved-source hashes and can be exported as
deterministic JSON. Execution receipts provide evidence for particular inputs
and IO variants; the registry itself is not correctness or performance evidence.

A suite uses schema `operator-harness-suite-v1` and contains independent cases.
Each case names its `implementation_ids`, roots and an exact recipe. Native
roots use `kind`, `width`, `logical_view`, `original_numeric_payload` and
canonical little-endian `data_hex`. Stage parameters are string pairs. Every
claimed implementation must occur in the recipe through its registered binding.
The existing runtime remains responsible for actual graph execution, limits,
frame validation, checksums and inverse operations.

`cpp/operator_harness_v1/operator_harness_v1_cli` is the JSONL native protocol
(the compiled executable is in `build-local/`). It supports `capabilities`,
`run` and `decode`. One failed request produces a failed response while the
process remains available to handle the next request. The Python runner adds
a process deadline and restarts its own local executor after a failure. It
does not use remote process control.

### A048 typed IO bridge

The existing BUFF graph action consumes a BUFF packet and returns a packet.
The new `harness.a048.byte_planes.v1` binding instead calls the existing
`buff_byte_planes_split/join` C++ kernels with raw unsigned offsets and an
explicit `bit_width`. It performs no FOR or scaling. It returns one Serial/1
stream per byte plane, in high-plane-first order, with `N` and width recorded
for inverse reconstruction. Minimum and any upstream state remain separate
graph leaves or upstream metadata.

At width zero there are zero semantic planes. Because the existing graph ABI
requires at least one physical output, the bridge returns one empty physical
carrier and declares this in inverse metadata. Tests cover empty input,
constant input, invalid width, invalid offsets, malformed reconstruction count
and the zero-carrier contract. This is a new IO variant, not another mechanism
or a claim of compatibility with the existing BUFF packet wire format.

## Independent commits and provenance

Each case is one independently encoded block/recipe. The runner shares a
resident native driver across cases, and commits each case immediately after
its native and fresh-process checks complete. A failing case cannot retract
successful siblings. There is no 97-implementation group transaction.

Outputs:

```text
RUN.json                 immutable source/binary/registry/suite/timer identity
REGISTRY.json            frozen 102-entry registration snapshot
SUITE.json               exact inputs and recipes
events.jsonl             fsynced start/native-complete/commit events
cases/<case-key>/
  REQUEST.json
  NATIVE_RESPONSE.json
  frame.ngf              complete recoverable NGF1 representation
  COLD_DECODE.json        response from a new decoder OS process
  RESULT.json            atomic independently committed receipt
operators/<A-id>/<case-key>.json
                         independently persisted trace-specific evidence
SUMMARY.json             coverage, statuses and receipt locations
```

The trace-specific records refer to the full case receipt. A multi-operator
case supplies graph evidence for its participating ids, not standalone kernel
timing. The demo includes an independent single-operator case for every one of
the nine bound mechanisms as well as compositions and edge cases.
`independently_verified_implementation_ids` counts passing single-operator
cases. `implementation_ids_in_passing_cases` counts all participating ids.
Neither field means that an operator's entire input domain has been proven.

Resume verifies receipt and frame digests before reusing results. Immutable
identity checks reject stale results after source or timer-contract changes.
Result files are committed with a temporary file, fsync and atomic rename;
an exclusive output-directory lock prevents concurrent writers. If interrupted
before commit, only that uncommitted case is rerun. Missing trace-specific
records can be reconstructed from a previously committed case.

The runner does not arbitrarily subdivide stateful streams. Adjacent XOR runs
in original element order and each fixture is a declared independent block.
Within-block state checkpointing, an automatic distributed scheduler and
full-dataset campaigns are outside this first version.

## Measurement boundaries

The native runtime measures with `std::chrono::steady_clock`:

| Metric | Included | Excluded |
|---|---|---|
| Fixed-recipe encode | Graph scheduling, native stages, inverse metadata, complete NGF1 framing and checksums | JSON/hex conversion, driver IO, recipe construction/selection, process startup |
| Fixed-recipe decode | Frame parsing/checksums, inverse scheduling and native inverse stages | JSON/hex conversion, driver IO, final root comparison |
| Fresh-process decode | Same native decode boundary in a new OS process, with zero warmups | OS process startup and protocol overhead |
| Complete bytes | Entire emitted NGF1 frame, including all leaves and reconstruction information | Nothing is removed from the emitted frame |

Every timed graph iteration is checked bit-for-bit outside the timed regions.
The Python runner additionally decodes the complete frame in a separate process
and compares root type, logical view, denominator flag and exact bytes. The
timer arrays and their median/min/max are retained. A zero-byte original has
null byte-normalized ratios and times, not division by zero.

The demo's 30 bounded synthetic cases are interface and measurement-path
qualification. Their timings are local, unpinned samples and must not be used
as a dataset-wide codec ranking. Single-operator graphs still include graph
and frame costs; this version does not report isolated kernel microbenchmarks,
peak RSS, or online selector cost. H0/H3 and the historical Zstd-3 ablation
remain in the existing quality harness and are not inserted into these timers.
Any later online selector must report its analysis/selection cost separately
and include it in an explicitly defined end-to-end metric.

## Verification

```sh
python3 -m unittest discover -s scripts -p 'test_operator_harness_registry_v1.py' -v
OPERATOR_HARNESS_DRIVER="$PWD/cpp/operator_harness_v1/build-local/operator_harness_v1_cli" \
  python3 -m unittest discover -s scripts -p 'test_operator_harness_v1.py' -v
```

Native tests exercise real graph round trips, protocol failure recovery,
malformed frames and the NNCP forward/inverse policy gate. Runner tests cover
success/failure/success commits, immutable resume, tamper detection, process
deadlines and refusal to write into unrelated experiment directories. The fake
protocol driver used by runner unit tests is a test double; only the native
integration and saved local demo contain C++ execution evidence.
