#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct wc_native_graph_v1_result wc_native_graph_v1_result;
uint32_t wc_native_graph_v1_abi_version(void);
wc_native_graph_v1_result* wc_native_graph_v1_encode_zlib(const uint8_t*, size_t);
wc_native_graph_v1_result* wc_native_graph_v1_decode(const uint8_t*, size_t);
wc_native_graph_v1_result* wc_native_graph_v1_benchmark_zlib(const uint8_t*, size_t, uint32_t, uint32_t);
const uint8_t* wc_native_graph_v1_result_data(const wc_native_graph_v1_result*);
size_t wc_native_graph_v1_result_size(const wc_native_graph_v1_result*);
const char* wc_native_graph_v1_result_error(const wc_native_graph_v1_result*);
const uint64_t* wc_native_graph_v1_encode_ns(const wc_native_graph_v1_result*);
size_t wc_native_graph_v1_encode_ns_count(const wc_native_graph_v1_result*);
const uint64_t* wc_native_graph_v1_decode_ns(const wc_native_graph_v1_result*);
size_t wc_native_graph_v1_decode_ns_count(const wc_native_graph_v1_result*);
uint64_t wc_native_graph_v1_original_numeric_bytes(const wc_native_graph_v1_result*);
void wc_native_graph_v1_result_free(wc_native_graph_v1_result*);
#ifdef __cplusplus
}
#endif
