#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI-owned results must be released by the library that allocated them. */
typedef struct wc_layout_result_v1 wc_layout_result_v1;
uint32_t wc_layout_abi_version_v1(void);
wc_layout_result_v1* wc_layout_run_v1(
    size_t argc, const char* const* argv, const uint8_t* data, size_t size);
wc_layout_result_v1* wc_layout_benchmark_v1(
    size_t argc, const char* const* argv, const uint8_t* data, size_t size,
    uint32_t warmups, uint32_t repeats);
const uint8_t* wc_layout_result_data_v1(const wc_layout_result_v1* result);
size_t wc_layout_result_size_v1(const wc_layout_result_v1* result);
const char* wc_layout_result_error_v1(const wc_layout_result_v1* result);
void wc_layout_result_free_v1(wc_layout_result_v1* result);

#ifdef __cplusplus
}
#endif
