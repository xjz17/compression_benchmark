# Experiment harnesses

This directory contains reusable experiment code extracted from the active
development campaigns. Run receipts, remote launch state, result streams, and
machine-specific commands were deliberately omitted.

## `operator_timing`

Measures operator encode/decode time and emits caller-selected output files.
`smoke_test.py` checks the runner on a minimal deterministic workload.
The `a015_a018_repair/` directory contains the source-level compatibility fix
and focused probe for those operator implementations.

## `operator_algorithm_cartesian`

Builds and evaluates explicitly measured operator/algorithm compositions. The
Python tests cover the carrier frame, joint frame, and runner integration. The
`algorithm_routes/` subdirectory contains the active algorithm route builder
and its tests.

These harnesses require the root `cpp/` tree and initialized submodules. They do
not bundle input datasets or previously generated results.
