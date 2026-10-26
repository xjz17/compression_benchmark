"""One max-value-width packing algorithm with source-format I/O adapters."""
ALGORITHM_NAME = "op.bitpack_int.max_value_width.cpp_v1"
PLAIN_VARIANT = "op.bitpack_int.max_value_width.uint64_le.cpp_v1"
SOURCE_VARIANTS = ("op.bitpack_int.ts_2diff.cpp_v1", "op.bitpack_int.leco.cpp_v1",
                   "op.bitpack_int.alp.cpp_v1", "op.bitpack_int.reger.cpp_v1")
INPUT_MAGICS = dict(zip(SOURCE_VARIANTS, (b"S2F1", b"LAR1", b"AFO1", b"RFO1")))
SOURCE = "cpp/src/max_value_bitpack.hpp"

PRODUCER_VARIANTS = dict(zip(SOURCE_VARIANTS, ("op.frame_of_reference.ts_2diff.cpp_v1",
    "op.zigzag.leco_residual.cpp_v2", "op.frame_of_reference.alp.cpp_v1", "op.frame_of_reference.reger.cpp_v1")))
