#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** pFPC v1.0 memory compress/decompress (Burtscher, DCC 2009). Returns bytes written. */
size_t pfpc_compress_memory(const void *in_stream, size_t in_bytes, void *out_stream, size_t out_cap,
                            unsigned char pred_size_lg2, unsigned char threads, int chunk_size);

/** Returns bytes consumed from {@code in_stream}, or 0 on failure. */
size_t pfpc_decompress_memory(const void *in_stream, size_t in_bytes, void *out_stream, size_t out_cap);

enum pfpc_ablation_mode {
  PFPC_ABLATE_XOR = 1,
  PFPC_ABLATE_TRAILING_ZERO_PACK = 2
};

/**
 * Paired lossless formats used only by the operator-ablation benchmark.
 * The encoder omits exactly one pFPC representation operator, and the
 * decoder consumes the corresponding format. Returns bytes written/consumed.
 */
size_t pfpc_compress_memory_ablation(const void *in_stream, size_t in_bytes,
                                    void *out_stream, size_t out_cap,
                                    unsigned char pred_size_lg2,
                                    enum pfpc_ablation_mode mode);
size_t pfpc_decompress_memory_ablation(const void *in_stream, size_t in_bytes,
                                      void *out_stream, size_t out_cap);

#ifdef __cplusplus
}
#endif
