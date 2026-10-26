set(_B86_LEGACY "${CMAKE_CURRENT_LIST_DIR}/../../external/operator_ablation_sources/lzbench_legacy/upstream")
add_library(native_baseline86_upstream_legacy STATIC
  "${CMAKE_CURRENT_LIST_DIR}/upstream_legacy.cpp"
  "${_B86_LEGACY}/lz/pithy/pithy.cpp"
  "${_B86_LEGACY}/lz/shrinker/shrinker.c"
  "${_B86_LEGACY}/lz/xpack/lib/xpack_common.c"
  "${_B86_LEGACY}/lz/xpack/lib/x86_cpu_features.c"
  "${_B86_LEGACY}/lz/xpack/lib/xpack_compress.c"
  "${_B86_LEGACY}/lz/xpack/lib/xpack_decompress.c")
target_include_directories(native_baseline86_upstream_legacy PUBLIC "${CMAKE_CURRENT_LIST_DIR}" PRIVATE "${_B86_LEGACY}" "${_B86_LEGACY}/lz")
target_compile_features(native_baseline86_upstream_legacy PUBLIC cxx_std_20)
set_target_properties(native_baseline86_upstream_legacy PROPERTIES POSITION_INDEPENDENT_CODE ON)
