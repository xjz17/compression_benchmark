# A110: Genuine NNCP Symbol Probability

The standalone provider calls the original Fabrice Bellard NNCP 3.3 model,
optimizer, and LibNC numerical implementation. It does not implement a proxy
network. Its C++ CLI calls the C model bridge. The official upstream files are
included unchanged, with the upstream executable's `main` renamed at compilation.

## Source And Platform

- Official archive: https://bellard.org/nncp/nncp-2024-06-05.tar.gz
- Version: NNCP 3.3, release 2024-06-05.
- `nncp.c` SHA256: `379a52a3624aa3661b0bb327c95fa7c650f701f133d2f25264eb34b9ca919dd4`.
- `libnc.so` SHA256: `1836cdfde987885e542cb88847cc58c9abefb0ef59a511ea9540dcbe46ac6d3e`.
- Upstream C sources: MIT. The upstream README separately permits redistribution
  of the binary LibNC library. LibNC's binary is not claimed to be MIT source.
- Bundled LibNC requires Linux x86-64. No native macOS/ARM library is substituted.
- A local Apple ARM host was verified using its existing Docker Desktop and
  the `gcc:14.3.0-bookworm` Linux amd64 image. This is local functional validation,
  not a native CPU performance benchmark or a server result.

## Frozen Profile

Profile `transformer-stream1-v1` keeps the upstream `default` Transformer network:
4 layers, model width 256, 8 attention heads, inner width 512, memory 32,
relative-position width 32, float32 CPU computation, and the original Adam
configuration and learning-rate schedule. Vocabulary is all 256 raw bytes,
seed is 123, there is one thread, and there is no tokenizer or pretraining.

The profile explicitly sets `batch_size=1` and `seg_len=1`. Each input segment
contains only the already observed previous byte; the initial previous byte is
0, as in the upstream block loop. It uses the upstream complete-segment
`model_eval(-1, input)` path and performs the original gradient and optimizer
update after observing the current byte. Because a segment has exactly one
known prior symbol, the complete-segment evaluation is causal. Attention memory
continues across observations. A session accepts at most 4096 bytes, then must
end or reset explicitly as a new block/session.

This profile is not the upstream default file schedule (`batch_size=16`,
`seg_len=32`), enwik8/enwik9 profile, or an NNCP-native file-format promise.
The different schedule is an explicit model configuration, not a new algorithm.

## Contract

`NNCPSymbolSession.predict()` returns 256 float32 probabilities in symbol order
0 through 255. The sum is within 1e-4 of 1. It predicts the next byte before
observation and returns the cached vector on repeated calls.

`observe(symbol)` accepts an integer in 0..255 only after a prediction. The
caller must first encode or decode the current event using that distribution.
Observe supplies the event as `expected_output`, calls upstream
`model_eval_gradient`, then `model_update`, and advances the schedule. The next
prediction can depend on this event. Calls out of order are rejected.

The output is a byte-symbol distribution, not a bit P(1), a logit, a diagnostic
submodel vector, or compressed data. The existing bit API rejects NNCP and
points callers to the separate symbol API. It must not be fed to a static
byte histogram coder or serialized as the compression result. A decoder must
reconstruct the same profile, seed, byte order, reset boundaries, numerical
implementation, and model updates.

The C API is in `cpp/include/nncp_probability.h`; the genuine model bridge is
`cpp/src/nncp_probability_bridge.c`. The standalone C++ executable exposes:

| Command | Input | Output |
|---|---|---|
| `D` | no current symbol | 256 little-endian IEEE float32 values |
| `O` | one observed raw byte | one acknowledgement byte, value 1 |
| `T` | uint64 little-endian count and raw bytes, count <=4096 | one probability vector before each observed byte |
| `Q` | no payload | clean session termination |

One Python session owns one fresh process. The source model owns its parameters,
attention memory, optimizer, and random state; no trained state is shared between
independent sessions.

## Matched Entropy Validation

The `encode` and `decode` CLI modes use upstream `write_sym` and `read_sym`, with
the same 256 probabilities, original 15-bit binary arithmetic coder, and model
update sequence. `decode` initializes a fresh model without access to the input
or encoder state. This is a bounded independent path for validating the provider
and matched entropy coding.

The new `NNP1` frame contains 24 bytes of fixed overhead: magic (4), profile ID
(4), original byte count (8), CRC32 of raw input (4), and arithmetic body byte
count (4), followed by the complete arithmetic body. Count is <=4096. Unknown
profiles, wrong lengths, appended/truncated bytes and checksum mismatches are
rejected. The decoder also reconstructs the canonical arithmetic body using
the same probabilities and checks exact bytes, so an attacker cannot bypass
trailing/truncated-data checks by changing the declared body length. This
also validates the empty-input arithmetic body. Profile 1 freezes the model
and initialization described above.
Full frame size, not just the arithmetic body, is the compressed byte count.
No performance or compression benefit is claimed by these functional tests.

## Typed-Graph Identity

`scripts/nncp_entropy_codec.py` exposes A110 to the typed graph only as the
complete `NNP1` terminal entropy path. The graph step records the exact action
name `op.prediction_probability.nncp.provider_v1`, provider `nncp`, and profile
`transformer-stream1-v1`. Cold restoration starts a fresh native decoder from
the `NNP1` leaf and the graph metadata; it does not reuse encoder/model state.

This integration does **not** relabel the 256-value probability provider as an
atomic transform, and a probability vector is never counted as compressed
bytes. The fixed GRPO catalog masks A110 unless both the standalone validation
and typed-graph cold-decode receipts match the current sources and native
binary. A110's existing 4096-byte adapter bound still applies.

## Build And Use

On Linux x86-64 with a C compiler, C++20 compiler and zlib development headers:

```sh
python3 scripts/build_nncp_probability.py
PYTHONPATH=scripts python3 scripts/test_nncp_probability.py -v
PYTHONPATH=scripts python3 scripts/validate_nncp_entropy_graph.py
```

The optional project CMake module is `cpp/cmake/NNCPProbability.cmake`; enable
`WEB_COMPRESSION_BUILD_NNCP=ON` after including that module. Its default is off
because the supplied LibNC library has a platform restriction.

For explicit local Docker execution on a different host:

```sh
WEB_COMPRESSION_NNCP_USE_DOCKER=1 python3 scripts/build_nncp_probability.py
WEB_COMPRESSION_NNCP_USE_DOCKER=1 PYTHONPATH=scripts python3 scripts/test_nncp_probability.py -v
```

Docker mode is never selected silently. It uses a Linux amd64 container with
network disabled, a 2 GiB memory limit, 2 CPUs, and a 180-second session limit.
The library stays in an isolated build folder with a source/compiler fingerprint.
`WEB_COMPRESSION_NNCP_DRIVER` may select an explicit prebuilt executable.
The read-only `nncp_availability()` helper compares source, binary and LibNC
fingerprints against the successful validation receipt. For Docker it also
checks the running daemon and exact image ID; configuration alone does not
establish availability. It never compiles or starts a container/model.

```python
from prediction_probability import NNCPSymbolSession

with NNCPSymbolSession() as model:
    for symbol in b"ABA":
        distribution = model.predict()
        # Encode/decode this event with the matching entropy coder here.
        model.observe(symbol)
```

## Verification

On 2026-09-17, the actual original model passed six bounded tests in Docker:
probability normalization and nonuniform output, deterministic prefix causality,
cached prediction and batch/interactive parity, isolated interleaved sessions,
event ordering and input bounds, and independent cold arithmetic decoding.
The cold-decoding sample includes byte zero, byte 255 and repeated letters;
empty input and malformed frames are also covered. A later native Linux x86-64
run on the authorized 192 host passed the same standalone suite and three
typed-graph tests: direct graph round trip, fresh-interpreter/fresh-process cold
restore, and action/profile/corruption identity checks. The existing eight
PPMd/DMC/PAQ tests passed after the symbol API was added.

This establishes the independent A110 provider, its bounded matched entropy
path, and the explicit complete-path graph bridge. It does not establish
completed dataset evaluation, a compression advantage, default NNCP benchmark
equivalence, or formal readiness of the other operator mechanisms.
