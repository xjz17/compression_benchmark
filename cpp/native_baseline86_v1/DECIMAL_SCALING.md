# Lossless decimal scaling for integer codecs

The opt-in profile `baseline86-decimal-scaling-int64-exact-v1` admits
`float32` and `float64` to 2DIFF, BOS, REGER, SVB, FastPFor, Simple8b, DBP,
RLBE and LECO. The original inventory, default commands, configurations and
integer frames remain separate. Existing measurements are not rewritten.

## Representation

For each original-dtype finite value, the adapter obtains its canonical shortest
round-trip decimal representation. The frozen column parameter is
`P = max(required fractional decimal places)` across every entity and block of
that scientific column. Thus `1.23, 6.5` use `P=2` and become `123, 650`.
The column identity is dataset/table/role/source-column/column/dtype; entity and
chunk coordinates do not choose separate scales. Precision refers to the typed
numeric values, not to insignificant trailing zeros in the original CSV text.

The implementation parses decimal coefficients and exponents and performs
checked exact integer arithmetic. It does not use floating multiplication by
`pow(10,P)` or rounding to decide whether a value fits. `P` can be 0..324.
Every scalable value is converted back to its original float dtype during
encoding and checked bit for bit. The decoder needs only the complete frame.

NaN payloads, infinities, negative zero and values that cannot fit the common
`int64` scale retain their original bits in an exception stream. NULL rows stay
in the original missing bitmap. The integer codec sees the compact remaining
integer sequence. A nonempty block with no scalable values is `unsupported`,
rather than a fabricated raw-only integer-codec result. The integer backend's
own support boundaries still apply, including 2DIFF adjacent-delta overflow.

The DSC1 metadata records the dtype, `P`, original/scaled/exception counts,
exception bitmap and raw exception bytes inside the checksummed NGF1 frame.
Compression size is the complete NGF1 frame, including DSC1, integer payload,
NULL bitmap and every other recovery byte. The denominator remains the original
valid numeric bytes (`4*N` or `8*N`), never the expanded integer buffer size.
Receipts disclose exception counts/fraction and metadata/raw-exception bytes.

## Commands

Build with the standalone component CMake file in a new build directory:

```sh
cmake -S cpp/native_baseline86_v1 -B cpp/native_baseline86_v1/build-decimal-scaling-v1 -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/native_baseline86_v1/build-decimal-scaling-v1 -j 8
ctest --test-dir cpp/native_baseline86_v1/build-decimal-scaling-v1 --output-on-failure -R 'decimal_scaling|column_scales|roundtrip|cold_|zero_denominator'
```

The binary supports:

```text
inventory-scaled
encode-scaled METHOD DTYPE ROWS VALUES MISSING P FRAME
measure-scaled METHOD DTYPE ROWS VALUES MISSING P WARMUPS REPEATS FRAME
decode FRAME VALUES_OUT MISSING_OUT
scan-column-scales MANIFEST OUT [WARMUPS REPEATS]
benchmark-manifest-scaled MANIFEST JSONL WARMUPS REPEATS --column-scales OUT [--frame-dir DIR] [METHOD ...]
benchmark-manifest-scaled MANIFEST JSONL WARMUPS REPEATS --column-scales OUT --frame-pack PACK [METHOD ...]
verify-scaled-pack MANIFEST JSONL PACK SCALES RECEIPT METHOD [METHOD ...]
```

`scan-column-scales` defaults to one warmup and three measurements. It requires
a frozen block manifest with `dataset_id` and `blocks`. Each block carries a
unique `block_id`, dtype, row count, role and column identity, values and missing
bitmap paths, both SHA-256 digests, and `original_numeric_bytes`. It scans one
resident block at a time; source I/O, SHA verification and input validation are
outside the precision timer. The measured column sample is the sum of the block
precision scans and their maximum updates. All blocks participate in `P`.

The precision receipt binds the exact manifest, block union and hashes, running
CLI binary hash, host, CPU affinity, warmups/repeats and raw timing samples. The
benchmark revalidates inputs and recomputes `P` before creating JSONL. It rejects
changed precision, inputs, binary, host/affinity or timing contract. Existing
precision and scaled JSONL paths are rejected; use a new isolated attempt.

## Timing and admission

`measure-scaled` includes the complete per-block precision check, exact
conversion, integer codec, exception handling and frame construction in encode
time; decode includes all inverse conversion and framing. Its explicit `P`
does not by itself measure whole-column precision discovery. This command alone
is therefore insufficient for a complete compression-time comparison.

`benchmark-manifest-scaled` adds the measured whole-column precision samples
once per method and column, on the first successful nonempty block. Every later
block receives a zero additional selection charge. It retains
`compression_encode_only_ns_samples`, `precision_derivation_ns_samples` and
their samplewise sum in `compression_ns_samples`; column totals must include
all blocks. All-NULL blocks remain `N/A` and do not absorb the selection charge.
Original `int64` blocks retain the original profile; other integer widths are
not silently widened by this float adapter.

Formal campaign acceptance must freeze the Linux build/runtime and precision
receipt, qualify the input and same-host single-thread CPU/resource profile,
and independently cold-decode the stored full frames. Local regression tests
establish correctness, not a published performance result. Scanner timing
samples must be protected by the campaign's frozen receipt SHA and process
evidence; validating JSON structure alone cannot authenticate elapsed time.
No running producer, old accepted result, paper figure or website ranking is
changed by enabling this new profile.

## Shared input shards and batch cold verification

For existing offset catalogs, a block can add `values_offset`, `values_length`,
`missing_bitmap_offset` and `missing_bitmap_length`. Offset/length pairs must be
unsigned, bounded and complete. SHA fields refer to the exact slices, which
are read and verified before timing. The source payload is neither widened nor
materialized as one file per block. Scientific columns remain whole when
rendering large catalogs into one bounded manifest per column.

`--frame-pack` exclusively creates a custody archive containing repeated
8-byte little-endian lengths followed by the exact NGF1 frames. JSONL records
bind the record offset, frame length and frame SHA. The original NGF1 bytes and
timing API are identical to `--frame-dir`; the pack header is custody framing,
like a tar member header. Any comparison against a different scientific outer
container requires explicit byte-boundary compatibility rather than silently
merging the sizes. Pack and directory custody options are mutually exclusive.
Metadata writes are buffered and pack persistence occurs outside codec timing.

The batch verifier starts as a separate process, reads each complete frame,
decodes it before reading reference value bytes, then compares original dtype
bits and NULL masks. It also inspects the actual framed method/precision/counts,
checks whole-column precision charges and the exact requested block-method
union, reproduces unsupported boundaries, and rejects extra/truncated members.
Two independent verifier processes can validate a whole column-method pack;
this avoids millions of process starts while retaining complete cold recovery
evidence. These receipts verify bytes and semantics; final campaign timing and
resource acceptance remains separate.

For comparison with accepted native aggregate sizes, production adds
`--compound-frame` to `--frame-pack`. This mode requires exactly one method
and one whole logical column. The actual pack is the accepted P172B861
container: eight magic bytes, an eight-byte member count, then each eight-byte
length and its exact NGF1 child. `complete_frame_bytes` charges the child plus
its length, and charges the global 16 bytes on the first materialized child
only. Their sum equals the actual compound file size; all-NULL child bytes
must remain in the column total even though their individual ratios are N/A.
`complete_ngf1_frame_bytes` and the child SHA separately identify the exact
NGF1 child. The full compound SHA is bound by each cold-verification receipt.

The in-memory construction/parsing of each length is timed once and added to
all repeated samples, matching the accepted aggregate adapter. Its separate
unsigned timing fields and the original NGF1-only samples are retained and
arithmetically checked by the cold verifier. Fixed global-header/count file
I/O remains outside timing, as in the accepted adapter; whole-column precision
derivation remains an additional once-per-method-column scientific cost.
This preserves the storage boundary but does not turn this new float-adapter
timing profile into historical integer timings. No custody-only pack can be
used as a P172B861 production result.
