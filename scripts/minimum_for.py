"""One minimum-plus-offsets transform; codecs retain only format adapters."""
ALGORITHM_NAME = "op.frame_of_reference.minimum.cpp_v1"
PLAIN_VARIANT = "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1"
SOURCE_VARIANTS = ("op.frame_of_reference.reference_v1", "op.frame_of_reference.bos.cpp_v1",
    "op.frame_of_reference.reger.cpp_v1", "op.frame_of_reference.ts_2diff.cpp_v1",
    "op.frame_of_reference.bos.post_outlier.cpp_v1", "op.frame_of_reference.alp.cpp_v1")
PREVIOUS_WHOLE_NAME = "op.frame_of_reference.whole_stream_min.cpp_v2"
SOURCE = "cpp/src/minimum_for.hpp"
# Raw bytes are always uint64 LE; private packets require an explicit profile
# or a declared upstream producer, never a coincidental magic prefix.
PRODUCER_VARIANTS = {
    "op.frame_of_reference.bos.cpp_v1": ("op.delta.bos.cpp_v1", b"BDL1"),
    "op.frame_of_reference.ts_2diff.cpp_v1": ("op.delta.ts_2diff.cpp_v1", b"S2D1"),
    "op.frame_of_reference.reger.cpp_v1": ("op.zigzag.reger_residual.cpp_v2", b"RAR1"),
    "op.frame_of_reference.bos.post_outlier.cpp_v1": ("op.outlier_separation.bos.cpp_v1", b"BOS1"),
    "op.frame_of_reference.alp.cpp_v1": ("op.scaling.alp.cpp_v1", b"ASC1"),
}
