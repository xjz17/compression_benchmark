# BOS versioned route source binding

This directory vendors the reviewed BOS components into the unified harness so
the replacement can be built without modifying the frozen baseline runtime or
its historical deployments.

## Pinned inputs

| Component | Reviewed source SHA-256 | Local treatment |
| --- | --- | --- |
| `bos_float_exception.cpp` | `745df45d88ed4884f6f920874465cf7c280577c6b58c07bf874d74ce5ec9e285` | copied unchanged |
| `bos_float_exception.hpp` | `5dd9796e9276f06286dbeeaa05740f80e5591b6952244d467dc6ea13c46a8529` | copied unchanged |
| `dsc3_refined.cpp` | `2b641300da0f1baf6be860c66ee32e1347e0867905291f63d269499e5bfedd80` | copied unchanged |
| `decimal_scaling_v2.hpp` | `86d750fed6ddf050ec0a2713c03f862bb706cac78620664c7551ed51a1c6d0c4` | copied unchanged |
| `decimal_scaling_v2_legacy.cpp` | `0a032e1f5f0292087e00e7706549fc8c096c625fa314d4a1d569c4521919b755` | copied unchanged |
| Project-150 `bos_refined.cpp` | `54d7cae932d21e4a18d9e407a4daeb056aec8c57ea654a3e603c1dddb4e0ff97` | renamed `bos2_refined.cpp`; only include/namespace binding changed to avoid an ODR collision with the legacy BOS implementation |
| reviewed `ucc_bos_bfe_bridge.cpp` | `feb25a3b344f18a01eacbe1a2a91596fb45a50262332375f2288339cad9350e7` | extended with the versioned integer dispatcher, multi-block envelope, and inspectable audit counts |
| reviewed `ucc_bos_bfe_bridge.hpp` | `ed2405dee66dcaa045be487337df363929c89aa4772e3f45d8f51b767e1615a4` | extended with the integer route and audit fields |

The reviewed sources come from
`analysis_outputs/bos_float_exception_150_20261023_v1/integration/` and the
Project-150 runtime referenced by its `install_ucc_bos_bfe_overlay.py` source
checks.

## Route contract

- `int64`: `project_bos2_partitioned`, serialized with the Project-150
  baseline86 BOS action/configuration identity. New frames contain the isolated
  BOS2 kernel; historical `adapter_compose_int64` BOS64 frames remain accepted
  by the legacy decoder. Empty and all-missing public integer blocks retain the
  baseline86 canonical empty method payload; this shortcut does not change the
  BFE/DSC3 integer subpayloads used inside floating routes.
- `float32` and `float64`: `bos-bfe-ngf-hybrid-v2`, selecting the smaller
  complete NGF frame between DSC3+BOS2 and BFE1+BOS2 for each block. Ties select
  DSC3. Precision is computed once for the whole column. More than 10,000 valid
  values use the checksummed BHC1 envelope.
- BFE1 native raw mode is an explicit lossless child candidate and is reported
  through `raw_block_count`. DSC3 and BFE1 exception mode are strict encoded
  paths.

This binding records implementation and compatibility scope only. It does not
claim evaluation on the 135-dataset benchmark.
