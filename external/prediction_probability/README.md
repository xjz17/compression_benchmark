# Prediction-only source provenance

These algorithms are OpenZL-related literature, **not OpenZL built-in codecs**.
The adapters return probabilities before observation. Probability output alone
is neither a compressed representation nor an independently reversible codec.

## Existing cmix source (not modified)

- Repository: <https://github.com/byronknoll/cmix>
- Local source commit: `38e2a9e63ebaedab5a072266e89ca9810fbfe41b` (cmix v21).
- Sources: `external/compression/cmix/src/`.
- License: upstream `COPYING` is GPL version 3; retain that file and all notices.
  The PAQ implementation carries an explicit GPL-2.0-or-later notice at
  `models/paq8.cpp:4-18`. The combined probability executable is GPL-3.0.
- PPMd: `models/ppmd.cpp:1-3` identifies Dmitry Shkarin's PPMd, adapted by
  Eugene Shelwien as mod_ppmd_v2 and then by cmix. `PPMD::ByteUpdate()` at
  1328-1337 updates the observed byte, prepares the next full 256-symbol
  distribution, floors source weights, applies the vocabulary, and normalizes.
  `ByteModel::Predict()` converts that distribution to the next MSB-first
  conditional bit probability. Our vocabulary is all 256 symbols; no future
  input is inspected to construct a vocabulary mask. Defaults: order 6, 16 MiB.
- DMC: `models/paq8.cpp:7598-7613` explicitly documents differences from
  Cormack's original DMC. Our adapter calls the actual `dmcModel::pr1()`
  (7733-7736) and `update()` (7682-7729), using 65536 extra nodes and threshold
  1024 in upstream fixed-point units. This is the **count-probability branch**
  of the cmix PAQ DMC model, not its StateMap combination or ten-model forest.
- PAQ: source line 1 identifies a PAQ8l/PAQ8pxd/PAQ8px hybrid. The adapter
  exposes the actual final `paq8::Predictor::p()` (8230), divided by 4096, then
  observes via `paq8::y=bit; predictor.update()`. `PAQ8::Predict()` at 8373
  instead exposes a vector of submodel features and must not be substituted
  for the final PAQ probability. Default memory level 0 is an original source
  parameter setting, not a reimplementation. Each session gets a new process
  because upstream PAQ/DMC use global/static state.
- cmix: real `Predictor::Predict()` / `Perceive(bit)` at `predictor.cpp:361-468`
  is connected. All-byte vocabulary, no preprocessing, no dictionary and no
  pretraining. The upstream full configuration is retained, **not run by
  default or in smoke tests**: `README:8` recommends at least 32 GB RAM.
  Running requires explicit `allow_large_memory=True`. It is build/link
  verified but not runtime verified in this implementation task.

PPMd, DMC and PAQ adapt online; they require no downloaded pretrained weights.
cmix includes learned mixing, including an LSTM, and updates online. Different
parameter choices affect probability traces; always record adapter parameters,
source revision, and compiler/float mode with an experiment. These adapters
never invoke the original entropy coder, parser or file preprocessor.

## NNCP official source retained, executable adapter unavailable

- Official page: <https://bellard.org/nncp/>
- Downloaded source release: <https://bellard.org/nncp/nncp-2024-06-05.tar.gz>
- Archive SHA-256: `7b4be2a5871186b82cd5f1c6137a8f6fed0d0c6b2bb281793db1f0be65831119`
- Archive and extracted tree are retained in this directory unchanged.
- `nncp.c:1-23` contains the MIT notice; `readme.txt:55-57` distinguishes
  MIT source from the freely redistributable binary LibNC library.
- Both supplied LibNC libraries are Linux x86-64 ELF `.so` files, not native
  macOS/arm64 libraries. `readme.txt:20-25` describes Linux compilation and
  Windows cross-compilation. CUDA is optional; large profiles benefit from it.
- Precise source boundary: `NNCPModelClass::model_eval` at `nncp.c:61`;
  Transformer softmax at 1007 (LSTM at 2048); `process_block` at 3002-3010
  materializes a float probability vector and passes it to `write_sym`.
  Capture this vector **before** `write_sym`, then observe the true symbol
  at 3018. Gradient and optimizer updates occur after a full segment/batch
  at 3051-3052, not after each bit or arbitrary byte. Partial final batches
  deliberately do not train (`3079`, `3141`). Historical retraining is in
  `retrain_block` (2820-2940).
- Default profile (`2463-2510`) uses seed 123, 16 streams, segments of 32
  symbols, 256 raw-byte symbols, a four-layer Transformer, and online Adam.
  Optional tokenization changes the symbol vocabulary (`3381`). It is not
  an off-the-shelf large language model and need not download pretrained
  weights. Decoder-side training/order/seed must match the encoder.
- The independent entropy boundary is `cp_utils.c:299-323`: `write_sym`
  reduces the symbol distribution to conditional binary probabilities and
  calls `put_bit`. A future adapter can replace this call with a callback
  while preserving model state, stream order and update schedule.
- The current Python provider intentionally raises `PredictorUnavailable`.
  Downloaded source and an audited cut point are **not** a completed adapter.

## Original DMC licensing caution

The historical Cormack DMC source retrieved for inspection at
<https://www.jjj.de/crs4/dmc.c> identifies copyright 1987/1993 and restricts
inclusion in sold/royalty-bearing software (`15-21`). It is not bundled here
and is not used by the adapters. Its `predict()` returns P(0), unlike the
P(1) convention used by this interface. The actual cmix PAQ implementation
and its explicit variant label avoid silently replacing it with a toy model.

## Build and protocol

Run `python3 scripts/prediction_probability.py build` from the repository.
Only the isolated `build/prediction_probability/prediction_probability_cli`
binary is built. Upstream sources and their build products remain untouched.
Use `scripts/test_prediction_probability.py` for bounded tests.

Native stdin commands: `P` predicts (stdout little-endian IEEE-754 f64 P(1));
`O` followed by one byte 0/1 observes, with no output; `D` outputs 256 f64
probabilities (PPMd at a byte boundary only); `T` followed by u64le byte count
and raw input outputs one f64 per input bit, MSB first; `Q` exits. Repeated
`P` before `O` returns the cached prediction, avoiding double forward calls.
Every `O` must follow `P`. `T` internally uses the same predict/observe order.
No probability/CDF quantization or entropy coding is hidden in the provider.
