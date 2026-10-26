# In-process layout bridge: development scope

This opt-in shared library calls the existing `openzl_layout_stages.cpp`
kernels without launching an operator process. It preserves the current OLS1
wire representation. The active `typed_graph_runtime.py`, registry, action
fingerprint, checkpoints, and formal experiment gates are unchanged.

## Build and verify

```sh
cmake -S cpp/native_layout_bridge_v1 -B cpp/build-native-layout-v1 -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-native-layout-v1 -j 2
python3 scripts/test_native_layout_bridge_v1.py -v
```

The standalone build requires C++20 and does not rebuild or modify the running
campaign's native artifacts. Set `NATIVE_LAYOUT_BRIDGE_V1` to use an explicitly
selected shared library. The bridge fails when the library is missing; there is
no process or Python implementation fallback.

`scripts/native_layout_bridge_v1.py::NativeLayoutBridge.run` mirrors the
arguments to `openzl_layout_stages_cli`. It supports the existing Transpose,
SplitStruct, FloatDeconstruct, BitSplit profiles, Bitunpack, Concat, continuous
Split, and SplitByRange modes. These modes are not new operator groups or new
catalog items. STOP belongs to the graph layer and is not provided by this
stage bridge.

The C ABI (`cpp/src/native_layout_bridge_v1.h`) owns each result separately;
callers release it through `wc_layout_result_free_v1`. Errors cannot cross the
C boundary as C++ exceptions. No shared last-error buffer or encoder cache is
used. The existing kernel API remains available directly to a future C++ graph
executor, which should pass typed data flows in memory instead of repeatedly
serializing an OLS1 packet for every edge.

## Timing boundary

`benchmark_stage` measures one layout stage and its inverse inside the native
process, including the stage's OLS1 serialization/parsing and allocations. It
excludes foreign-function transfer, process startup, graph scheduling, TGF1
framing/hashing, and verification comparisons. Samples and the complete OLS1
size are returned; they are **not whole-graph compression/decompression times
or a compression ratio**. Reports explicitly set `whole_graph_measurement` and
`formal_eligible` to false. No formal training gate accepts this report.

Local development verification on 2026-09-18: seven tests passed, covering 13
CLI wire-parity profiles, float special bit patterns, independent child-data-
flow re-encoding with subprocess execution forbidden, fresh-process decode,
all truncations of a sample frame, invalid parameters, and concurrent result
ownership. This is not a 192 timing result or an all-operator graph receipt.

## Remaining full-native work

1. Port graph replay, TGF1 creation/parsing, SHA256 checks, and strict dependency
   validation out of `scripts/typed_graph_runtime.py` into a single native
   executor. Preserve old-frame decoding; a new frame schema requires an
   explicit migration and new measurements.
2. Bind legacy action/semantic-port edges, minimum FOR, byte-plane Split,
   adjacent XOR, A111, residual-window packing, and predictor/entropy paths
   through direct native APIs. The layout bridge alone covers none of these.
3. Preserve every root, missing mask, recovery metadata field, output type and
   length, and leaf. Prove bit-exact cold decoding and malformed-frame
   rejection for whole graphs, not only stage packets.
4. Replace `weight_conditioned_graph_task_v1.py::_evaluate`'s Python replay and
   child-process timing only after candidate execution and complete-frame
   measurements have a common in-process native boundary. The time scales and
   final baseline measurements must use that same boundary and original
   numeric-byte denominator.
5. Publish a new native source/artifact fingerprint and rerun graph correctness,
   operator coverage, and measurement gates before formal training. Do not
   reuse or relabel measurements from current Python/CLI snapshots.
