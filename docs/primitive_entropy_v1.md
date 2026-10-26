# Primitive byte entropy coder v1

This is an independent entropy-coding primitive for the new atomic-operator
experiment. It does not invoke zlib, Bzip2, XZ, Zstd, LZ4, or another complete
compression algorithm. It is not a renamed wrapper around a complete codec.
Legacy source files and binaries remain unchanged.

## Commands

Binary: `cpp/build/primitive_entropy_v1/primitive_entropy_v1`.
Each command reads bytes from stdin and writes bytes to stdout:

| Encode command | Decode command | Model |
|---|---|---|
| `range0-encode` | `range0-decode` | Adaptive order-0, all byte frequencies initially one |
| `range1-encode` | `range1-decode` | Adaptive previous-byte context, frequencies initially one |
| `sparse0-encode` | `sparse0-decode` | Adaptive order-0, initially only an escape symbol |
| `sparse1-encode` | `sparse1-decode` | Adaptive previous-byte context, initially only escape |

The initial previous-byte context is zero. Each sparse model emits escape and
an arithmetic-coded uniform byte when encountering a new symbol, then adds that
symbol at frequency one. Escape remains at frequency one until all 256 bytes
have occurred, when it is removed. This is one explicit modeling primitive, not
an LZ dictionary. Every observed byte increments its frequency once. Models
rescale at total 32768 by halving frequencies, rounding upward; zero stays zero.

The coder uses 32-bit inclusive arithmetic intervals with 64-bit intermediate
products. All interval products are below 2^47. It outputs individual normalized
bits, not an LZ/BWT transformation. The context model and arithmetic coder are
fixed by the command and wire mode; no trained model is loaded externally.

## Complete size accounting

Every invocation returns a complete PEV1 frame, not payload bytes alone:

| Offset | Bytes | Field |
|---|---:|---|
| 0 | 4 | `PEV1` magic/version |
| 4 | 1 | Mode: 0 range0, 1 range1, 2 sparse0, 3 sparse1 |
| 5 | 3 | Zero reserved flags |
| 8 | 8 | Raw byte count, little endian |
| 16 | 8 | Arithmetic body byte count, little endian |
| 24 | 8 | FNV-1a 64-bit checksum of raw bytes |
| 32 | 8 | FNV-1a 64-bit checksum of arithmetic body |
| 40 | variable | Arithmetic bitstream including termination and final-byte padding |

Thus all 40 header bytes, arithmetic termination, padding, and any outer
operator-chain metadata must be included in the experiment's stored size.
Frequency tables need no storage because the decoder recreates them from the
fixed initial state and previously decoded bytes. An empty input occupies 41
bytes. FNV checksums detect accidental corruption; they are not cryptographic
authentication and do not establish security against a malicious party.

## Safety and validation

Maximum decoded input/output is 256 MiB per invocation. The CLI caps encoded
stdin at 512 MiB + 40 bytes, rejects a wrong mode, nonzero flags, length mismatch,
truncation, trailing data, bad checksums, invalid arithmetic intervals, and
noncanonical sparse escapes. The bit reader permits at most 32 zero padding
bits after the encoded body for arithmetic termination. Failed commands return
2 and an explanation on stderr; successful commands return 0.

Build and test with `python3 scripts/test_primitive_entropy_v1.py --benchmark`.
The test suite covers empty and short data, every byte and byte-pair context,
random full-byte data, biased alphabets, repeated runs, model rescaling,
deterministic encoding, exact wire lengths, damaged headers and bodies,
rehashed invalid bodies, oversized output, trailing data, wrong mode, and
absence of complete-codec dynamic dependencies. Source and binary hashes are
reported with throughput so the experiment can freeze the implementation.

These are empirical search candidates, not a guarantee of beating every fixed
algorithm. Sparse context models can be better on low-alphabet transformed
streams and worse on nearly uniform bytes; all choices need actual measurement.
