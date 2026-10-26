# Frozen 206-registry I/O compatibility fixtures

These 39 interface fixtures were captured before the 182-registry runtime edit.
Do not regenerate the expected bytes with the new runtime.

`manifest.json` identifies all 15 merged groups, each old canonical interface,
the old runtime source hashes, and SHA-256 for every binary. Each interface has
five files: root input, immediate action input, encoded UAR1 result, inverse
result, and bare codec wire. All 39 original forward/inverse pairs were checked
byte for byte during capture, and both runtime source hashes remained unchanged.

`migration_expectations.json` independently maps the 206 and 225 historical
policy spaces to the audited document order (A001 maps to policy ID 0). Both
STOP IDs map to 182. Non-STOP entries retain the old 206 canonical interface name.

Representative inputs include repeated/negative integers, exact two-decimal
floating values, a 21-word Sprintz input (complete 9-word blocks and short tail),
21-word FPC/pFPC inputs (odd tail), typed prefix packets, reference envelopes,
and frozen valid gzip/zlib/XZ container wires. Dependent stages preserve their
original nested UAR1 headers. CORAD framing uses an actual preceding CZL2 chain.

The fixture scope is interface dispatch and wire compatibility; one captured
case per interface is not exhaustive validation of each codec's numeric domain.
Existing codec tests remain responsible for broader edge and failure domains.

Run the compatibility suite from the repository root:

```sh
python3 scripts/test_canonical182_io_runtime.py
```

The matching stage executables in `cpp/build` must already be available.
The test never builds C++ targets or rewrites its fixtures.
