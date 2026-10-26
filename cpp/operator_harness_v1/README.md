# Native operator harness v1

`operator_harness_v1_cli` is a native JSONL service. It reads one
`operator-harness-request-v1` object per stdin line and emits exactly one
`operator-harness-response-v1` object per stdout line. A failed request returns
`status: "failed"`; the process continues with later requests.

Commands:

- `capabilities`: returns the linked native actions exposed by the harness.
  NNCP is absent and permanently disabled. PAQ and CMIX are reported but are
  disabled by default because they are context-heavy.
- `run`: accepts the native graph `roots` and `recipe` structures, benchmarks a
  fixed recipe, emits the complete NGF1 frame and verifies a bit-exact decode
  with a distinct dispatcher instance.
- `decode`: decodes `frame_hex`. With `measure_decode: true`, `decode_ns`
  includes NGF1 parse, checksum/validation, inverse scheduling and all inverse
  kernels. It excludes process startup, JSON/hex conversion and driver I/O.

`warmups` defaults to 1 and `repeats` to 3 for `run`. Decode measurement
defaults to off; measured decode defaults to 0 warmups and 1 repeat.

The harness-only action `harness.a048.byte_planes.v1` exposes A048 as a typed
multi-output graph stage. It accepts one `numeric` width-8 stream with
`profile: "uint64"` and one string parameter `bit_width` in `[0,64]`. Each
semantic byte plane is a separate serial output. At width zero, GraphSession's
required physical output is an explicit empty carrier; metadata records that it
represents zero semantic planes.

Build without touching the existing native graph build directory:

```sh
cmake -S cpp/operator_harness_v1 -B cpp/operator_harness_v1/build-local \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/operator_harness_v1/build-local \
  --target operator_harness_v1_cli operator_harness_v1_protocol_test -j2
ctest --test-dir cpp/operator_harness_v1/build-local \
  -R operator_harness_v1_protocol_test --output-on-failure
```
