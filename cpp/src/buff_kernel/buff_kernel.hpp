#pragma once

/**
 * In-process BUFF scalar kernel (bounded double + split column pack), aligned with upstream
 * ``external/compression/buff/database`` (``prec_double.rs``, ``split_double.rs`` early encode path).
 *
 * SIMD column helpers (AVX2 on x86_64, NEON on AArch64) and a minimal Roaring-style id set live in
 * ``buff_simd_ops.hpp`` / ``buff_simd_native.cpp`` and ``buff_roaring_lite.*``. Full Rust SIMD
 * encode and slice filters are not ported here; see ``buff_reconstruct.hpp`` for float reconstruction
 * notes.
 */

#include "buff_precision_bound.hpp"
#include "buff_bit_pack.hpp"
#include "buff_split_double_codec.hpp"
#include "buff_reconstruct.hpp"
