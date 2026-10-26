cmake_minimum_required(VERSION 3.18)
project(native_baseline86_v1 LANGUAGES C CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(ExternalProject)
# Reuse the shared graph frame and already-linked native kernels; do not fork it.
if(NOT TARGET native_graph_v1_stages)
  add_subdirectory(../native_graph_v1 ${CMAKE_CURRENT_BINARY_DIR}/native_graph EXCLUDE_FROM_ALL)
endif()
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_path(BASELINE86_JSON_INCLUDE_DIR nlohmann/json.hpp PATHS /opt/homebrew/include /usr/local/include /usr/include REQUIRED)
add_library(native_baseline86_v1 STATIC baseline.cpp decimal_scaling.cpp)
target_include_directories(native_baseline86_v1 PUBLIC . ../native_graph_v1 ../src PRIVATE ${BASELINE86_JSON_INCLUDE_DIR})
target_link_libraries(native_baseline86_v1 PUBLIC native_graph_v1_stages ZLIB::ZLIB)
set_target_properties(native_baseline86_v1 PROPERTIES POSITION_INDEPENDENT_CODE ON)
include(upstream_general_a.cmake)
native_baseline86_add_upstream_general_a(native_baseline86_upstream_general_a)
include(upstream_general_b.cmake)
include(upstream_legacy.cmake)
include(upstream_specialists.cmake)
native_baseline86_attach_upstream_specialists(native_baseline86_v1)
target_link_libraries(native_baseline86_v1 PUBLIC
  native_baseline86_upstream_general_a native_baseline86_general_b
  native_baseline86_upstream_legacy)
native_baseline86_add_upstream_specialist_tests(native_baseline86_v1)
add_executable(native_baseline86_upstream_legacy_test upstream_legacy_test.cpp)
target_link_libraries(native_baseline86_upstream_legacy_test PRIVATE native_baseline86_upstream_legacy)
add_test(NAME native_baseline86_upstream_legacy COMMAND native_baseline86_upstream_legacy_test)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(native_baseline86_v1 PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()
# Each optional upstream backend is independent; absence is explicit unsupported.
function(optional_backend key header)
  find_path(BASELINE86_${key}_INCLUDE NAMES ${header} PATHS /opt/homebrew/include /usr/local/include /usr/include)
  set(found ${BASELINE86_${key}_INCLUDE})
  set(libs "")
  foreach(name IN LISTS ARGN)
    find_library(BASELINE86_LIB_${name} NAMES ${name} PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
    if(NOT BASELINE86_LIB_${name})
      set(found FALSE)
    else()
      list(APPEND libs ${BASELINE86_LIB_${name}})
    endif()
  endforeach()
  if(found)
    target_include_directories(native_baseline86_v1 PRIVATE ${BASELINE86_${key}_INCLUDE})
    target_link_libraries(native_baseline86_v1 PUBLIC ${libs})
    target_compile_definitions(native_baseline86_v1 PRIVATE BASELINE86_${key}=1)
  endif()
endfunction()
optional_backend(ZSTD zstd.h zstd)
optional_backend(BROTLI brotli/encode.h brotlienc brotlidec brotlicommon)
if(TARGET native_graph_v1_lz4)
  target_link_libraries(native_baseline86_v1 PUBLIC native_graph_v1_lz4)
  target_compile_definitions(native_baseline86_v1 PRIVATE BASELINE86_LZ4=1)
else()
  optional_backend(LZ4 lz4frame.h lz4)
endif()
if(TARGET native_graph_v1_snappy)
  target_link_libraries(native_baseline86_v1 PUBLIC native_graph_v1_snappy)
  target_compile_definitions(native_baseline86_v1 PRIVATE BASELINE86_SNAPPY=1)
else()
  optional_backend(SNAPPY snappy.h snappy)
endif()
optional_backend(LZMA lzma.h lzma)
optional_backend(BZIP2 bzlib.h bz2)

# Baseline-owned adapters for lightweight upstream sources already vendored by
# lzbench.  They are isolated from native_graph_v1: graph dispatch and action
# identity do not depend on this library.
set(_B86_LZBENCH_LZ "${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/lzbench/lz")
set(LIBDEFLATE_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_GZIP OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${_B86_LZBENCH_LZ}/libdeflate"
  "${CMAKE_CURRENT_BINARY_DIR}/libdeflate" EXCLUDE_FROM_ALL)

set(_B86_ENTROPY "${_B86_LZBENCH_LZ}/lizard/entropy")
add_library(native_baseline86_vendored_entropy STATIC
  vendored_entropy_codecs_v1.cpp
  "${_B86_ENTROPY}/entropy_common.c"
  "${_B86_ENTROPY}/hist.c"
  "${_B86_ENTROPY}/fse_compress.c"
  "${_B86_ENTROPY}/fse_decompress.c"
  "${_B86_ENTROPY}/huf_compress.c"
  "${_B86_ENTROPY}/huf_decompress.c")
target_include_directories(native_baseline86_vendored_entropy PUBLIC . PRIVATE
  "${_B86_ENTROPY}")
# This older standalone FSE/Huff0 source already prefixes its shared core as
# NGEF/NGEH.  Prefix the remaining exported detail entry points too so it can
# coexist in one process with native_graph_v1's newer Zstd entropy internals.
target_compile_definitions(native_baseline86_vendored_entropy PRIVATE
  FSE_buildCTable=B86F_buildCTable
  FSE_buildCTable_raw=B86F_buildCTable_raw
  FSE_compress=B86F_compress
  FSE_compress2=B86F_compress2
  FSE_compress_wksp=B86F_compress_wksp
  FSE_createCTable=B86F_createCTable
  FSE_freeCTable=B86F_freeCTable
  FSE_buildDTable=B86F_buildDTable
  FSE_buildDTable_raw=B86F_buildDTable_raw
  FSE_buildDTable_rle=B86F_buildDTable_rle
  FSE_createDTable=B86F_createDTable
  FSE_decompress=B86F_decompress
  FSE_decompress_usingDTable=B86F_decompress_usingDTable
  FSE_decompress_wksp=B86F_decompress_wksp
  FSE_freeDTable=B86F_freeDTable
  HUF_buildCTable=B86H_buildCTable
  HUF_compress1X=B86H_compress1X
  HUF_compress1X_wksp=B86H_compress1X_wksp
  HUF_compress2=B86H_compress2
  HUF_compress4X_wksp=B86H_compress4X_wksp
  HUF_getNbBits=B86H_getNbBits
  HUF_writeCTable=B86H_writeCTable
  HUF_decompress1X1=B86H_decompress1X1
  HUF_decompress1X1_DCtx=B86H_decompress1X1_DCtx
  HUF_decompress1X1_DCtx_wksp_bmi2=B86H_decompress1X1_DCtx_wksp_bmi2
  HUF_decompress1X1_usingDTable=B86H_decompress1X1_usingDTable
  HUF_decompress1X2=B86H_decompress1X2
  HUF_decompress1X2_DCtx=B86H_decompress1X2_DCtx
  HUF_decompress1X2_usingDTable=B86H_decompress1X2_usingDTable
  HUF_decompress1X_DCtx=B86H_decompress1X_DCtx
  HUF_decompress1X_usingDTable_bmi2=B86H_decompress1X_usingDTable_bmi2
  HUF_decompress4X1=B86H_decompress4X1
  HUF_decompress4X1_DCtx=B86H_decompress4X1_DCtx
  HUF_decompress4X1_DCtx_wksp=B86H_decompress4X1_DCtx_wksp
  HUF_decompress4X1_usingDTable=B86H_decompress4X1_usingDTable
  HUF_decompress4X2=B86H_decompress4X2
  HUF_decompress4X2_DCtx=B86H_decompress4X2_DCtx
  HUF_decompress4X2_DCtx_wksp=B86H_decompress4X2_DCtx_wksp
  HUF_decompress4X2_usingDTable=B86H_decompress4X2_usingDTable
  HUF_decompress4X_DCtx=B86H_decompress4X_DCtx
  HUF_decompress4X_hufOnly=B86H_decompress4X_hufOnly
  HUF_decompress4X_hufOnly_wksp_bmi2=B86H_decompress4X_hufOnly_wksp_bmi2
  HUF_decompress4X_usingDTable_bmi2=B86H_decompress4X_usingDTable_bmi2
  HUF_readDTableX1=B86H_readDTableX1
  HUF_readDTableX2=B86H_readDTableX2)
set_target_properties(native_baseline86_vendored_entropy PROPERTIES
  POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden)

add_library(native_baseline86_vendored_lightweight STATIC
  vendored_lightweight_codecs_v1.cpp
  "${_B86_LZBENCH_LZ}/fastlz/fastlz.c"
  "${_B86_LZBENCH_LZ}/lzf/lzf_c_very.c"
  "${_B86_LZBENCH_LZ}/lzf/lzf_d.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzfse_encode.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzfse_encode_base.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzfse_decode.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzfse_decode_base.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzfse_fse.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzvn_encode_base.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzvn_decode.c"
  "${_B86_LZBENCH_LZ}/lzfse/lzvn_decode_base.c"
  "${_B86_LZBENCH_LZ}/brieflz/brieflz.c"
  "${_B86_LZBENCH_LZ}/brieflz/depacks.c"
  "${_B86_LZBENCH_LZ}/liblzg/encode.c"
  "${_B86_LZBENCH_LZ}/liblzg/decode.c"
  "${_B86_LZBENCH_LZ}/liblzg/checksum.c"
  "${_B86_LZBENCH_LZ}/liblzg/version.c"
  "${_B86_LZBENCH_LZ}/lzjb/lzjb2010.c"
  "${_B86_LZBENCH_LZ}/lzmat/lzmat_enc.c"
  "${_B86_LZBENCH_LZ}/lzmat/lzmat_dec.c"
  "${_B86_LZBENCH_LZ}/lzrw/lzrw1.c"
  "${_B86_LZBENCH_LZ}/wflz/wfLZ.c"
  "${_B86_LZBENCH_LZ}/yappy/yappy.cpp"
  "${_B86_LZBENCH_LZ}/quicklz/quicklz151b7.c"
  "${_B86_LZBENCH_LZ}/slz/src/slz.c"
  "${_B86_LZBENCH_LZ}/tamp/common.c"
  "${_B86_LZBENCH_LZ}/tamp/compressor.c"
  "${_B86_LZBENCH_LZ}/tamp/decompressor.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/lzbench/bwt/bzip3/src/libbz3.c")
target_include_directories(native_baseline86_vendored_lightweight PUBLIC . PRIVATE
  "${_B86_LZBENCH_LZ}/libdeflate"
  "${_B86_LZBENCH_LZ}/fastlz"
  "${_B86_LZBENCH_LZ}/lzf"
  "${_B86_LZBENCH_LZ}/lzfse"
  "${_B86_LZBENCH_LZ}/brieflz"
  "${_B86_LZBENCH_LZ}/liblzg"
  "${_B86_LZBENCH_LZ}/lzjb"
  "${_B86_LZBENCH_LZ}/lzmat"
  "${_B86_LZBENCH_LZ}/lzrw"
  "${_B86_LZBENCH_LZ}/wflz"
  "${_B86_LZBENCH_LZ}/yappy"
  "${_B86_LZBENCH_LZ}/yalz77"
  "${_B86_LZBENCH_LZ}/quicklz"
  "${_B86_LZBENCH_LZ}/slz/src"
  "${_B86_LZBENCH_LZ}/tamp"
  "${_B86_LZBENCH_LZ}/lzav"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/lzbench/bwt/bzip3/include")
target_compile_definitions(native_baseline86_vendored_lightweight PRIVATE
  TAMP_STREAM=0 VERSION="1.5.2")
target_link_libraries(native_baseline86_vendored_lightweight PUBLIC
  libdeflate::libdeflate_static native_baseline86_vendored_entropy ZLIB::ZLIB)
set_target_properties(native_baseline86_vendored_lightweight PROPERTIES
  POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden)
target_link_libraries(native_baseline86_v1 PUBLIC
  native_baseline86_vendored_lightweight)

# Blosc1/2 already have exact one-buffer C++ adapters in cpp/src.  Build the
# vendored upstream libraries in isolated trees so their bundled dependency
# target names cannot collide with native_graph_v1's LZ4/Zstd targets.
set(_B86_BLOSC1_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/c-blosc")
if(EXISTS "${_B86_BLOSC1_SRC}/CMakeLists.txt")
  set(_B86_BLOSC1_EP_SRC "${CMAKE_CURRENT_BINARY_DIR}/blosc1-src")
  set(_B86_BLOSC1_EP_BUILD "${CMAKE_CURRENT_BINARY_DIR}/blosc1-build")
  set(_B86_BLOSC1_LIB "${_B86_BLOSC1_EP_BUILD}/blosc/libblosc.a")
  ExternalProject_Add(baseline86_blosc1_ep
    SOURCE_DIR "${_B86_BLOSC1_EP_SRC}" BINARY_DIR "${_B86_BLOSC1_EP_BUILD}"
    DOWNLOAD_COMMAND "" UPDATE_COMMAND ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory "${_B86_BLOSC1_SRC}" "${_B86_BLOSC1_EP_SRC}"
      COMMAND ${CMAKE_COMMAND} -Dsrc=${_B86_BLOSC1_EP_SRC}/blosc/blosc.c
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/patch_c_blosc_mingw_pthread.cmake
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_STATIC=ON
      -DBUILD_SHARED=OFF -DBUILD_TESTS=OFF -DBUILD_FUZZERS=OFF
      -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_LZ4=ON -DDEACTIVATE_SNAPPY=ON
      -DDEACTIVATE_ZLIB=ON -DDEACTIVATE_ZSTD=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target blosc_static
    INSTALL_COMMAND "" BUILD_BYPRODUCTS "${_B86_BLOSC1_LIB}")
  add_library(baseline86_blosc1 STATIC IMPORTED GLOBAL)
  add_dependencies(baseline86_blosc1 baseline86_blosc1_ep)
  set_target_properties(baseline86_blosc1 PROPERTIES IMPORTED_LOCATION "${_B86_BLOSC1_LIB}")
  add_dependencies(native_baseline86_v1 baseline86_blosc1_ep)
  target_sources(native_baseline86_v1 PRIVATE ../src/blosc_codec.cpp)
  target_include_directories(native_baseline86_v1 PRIVATE "${_B86_BLOSC1_SRC}/blosc")
  target_link_libraries(native_baseline86_v1 PUBLIC baseline86_blosc1)
  target_compile_definitions(native_baseline86_v1 PRIVATE BASELINE86_BLOSC1=1)
endif()

set(_B86_BLOSC2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/c-blosc2")
if(EXISTS "${_B86_BLOSC2_SRC}/CMakeLists.txt")
  set(_B86_BLOSC2_EP_BUILD "${CMAKE_CURRENT_BINARY_DIR}/blosc2-build")
  set(_B86_BLOSC2_LIB "${_B86_BLOSC2_EP_BUILD}/blosc/libblosc2.a")
  ExternalProject_Add(baseline86_blosc2_ep
    SOURCE_DIR "${_B86_BLOSC2_SRC}" BINARY_DIR "${_B86_BLOSC2_EP_BUILD}"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_LITE=ON
      -DBUILD_STATIC=ON -DBUILD_SHARED=OFF -DBUILD_TESTS=OFF -DBUILD_FUZZERS=OFF
      -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PLUGINS=OFF
      -DBLOSC_ENABLE_ZFP=OFF
      -DBLOSC_LZ4_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/lz4
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target blosc2_static
    INSTALL_COMMAND "" BUILD_BYPRODUCTS "${_B86_BLOSC2_LIB}")
  add_library(baseline86_blosc2 STATIC IMPORTED GLOBAL)
  add_dependencies(baseline86_blosc2 baseline86_blosc2_ep)
  set_target_properties(baseline86_blosc2 PROPERTIES IMPORTED_LOCATION "${_B86_BLOSC2_LIB}")
  add_dependencies(native_baseline86_v1 baseline86_blosc2_ep)
  target_sources(native_baseline86_v1 PRIVATE ../src/blosc2_codec.cpp)
  target_include_directories(native_baseline86_v1 PRIVATE "${_B86_BLOSC2_SRC}/include")
  target_link_libraries(native_baseline86_v1 PUBLIC baseline86_blosc2)
  target_compile_definitions(native_baseline86_v1 PRIVATE BASELINE86_BLOSC2=1)
endif()
add_executable(native_baseline86_cli cli.cpp column_scales.cpp)
target_include_directories(native_baseline86_cli PRIVATE ${BASELINE86_JSON_INCLUDE_DIR})
target_link_libraries(native_baseline86_cli PRIVATE native_baseline86_v1)
add_executable(native_baseline86_test test.cpp)
target_link_libraries(native_baseline86_test PRIVATE native_baseline86_v1)
add_test(NAME native_baseline86_roundtrip COMMAND native_baseline86_test)
add_executable(native_baseline86_decimal_scaling_test
  ../tests/decimal_scaling_test.cpp decimal_scaling.cpp)
target_include_directories(native_baseline86_decimal_scaling_test PRIVATE .)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(native_baseline86_decimal_scaling_test PRIVATE
    -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME native_baseline86_decimal_scaling
  COMMAND native_baseline86_decimal_scaling_test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME native_baseline86_decimal_scaling_cold_cli
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_decimal_scaling_cli.py
    --cli $<TARGET_FILE:native_baseline86_cli>)
add_executable(native_baseline86_column_scales_test
  ../tests/column_scales_test.cpp column_scales.cpp decimal_scaling.cpp)
target_include_directories(native_baseline86_column_scales_test PRIVATE
  . ../native_graph_v1 ../src ${BASELINE86_JSON_INCLUDE_DIR})
target_link_libraries(native_baseline86_column_scales_test PRIVATE OpenSSL::Crypto)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(native_baseline86_column_scales_test PRIVATE
    -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME native_baseline86_column_scales
  COMMAND native_baseline86_column_scales_test)
add_test(NAME native_baseline86_decimal_scaling_manifest
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_decimal_scaling_manifest.py
    --cli $<TARGET_FILE:native_baseline86_cli>)
add_test(NAME native_baseline86_decimal_scaling_segments_pack
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_decimal_scaling_segments_pack.py
    --cli $<TARGET_FILE:native_baseline86_cli>)
add_executable(native_baseline86_vendored_lightweight_test
  vendored_lightweight_test.cpp)
target_link_libraries(native_baseline86_vendored_lightweight_test PRIVATE
  native_baseline86_vendored_lightweight)
add_test(NAME native_baseline86_vendored_lightweight
  COMMAND native_baseline86_vendored_lightweight_test)

# A fixture encoded and decoded by distinct executable invocations proves that
# decode uses only the complete frame.  The original values are supplied only
# to the final byte comparison, never to the decode process.
set(_B86_COLD "${CMAKE_CURRENT_BINARY_DIR}/cold_fixture")
add_test(NAME native_baseline86_cold_prepare
  COMMAND native_baseline86_test write-fixture "${_B86_COLD}")
set_tests_properties(native_baseline86_cold_prepare PROPERTIES FIXTURES_SETUP b86_input)
add_test(NAME native_baseline86_cold_encode
  COMMAND native_baseline86_cli encode Zlib int64 13
    "${_B86_COLD}/values.bin" "${_B86_COLD}/missing.bitmap" "${_B86_COLD}/frame.ngf")
set_tests_properties(native_baseline86_cold_encode PROPERTIES
  FIXTURES_REQUIRED b86_input FIXTURES_SETUP b86_frame)
add_test(NAME native_baseline86_cold_decode
  COMMAND native_baseline86_cli decode "${_B86_COLD}/frame.ngf"
    "${_B86_COLD}/decoded.values.bin" "${_B86_COLD}/decoded.missing.bitmap")
set_tests_properties(native_baseline86_cold_decode PROPERTIES
  FIXTURES_REQUIRED b86_frame FIXTURES_SETUP b86_decoded)
add_test(NAME native_baseline86_cold_check
  COMMAND native_baseline86_test check-fixture "${_B86_COLD}")
set_tests_properties(native_baseline86_cold_check PROPERTIES FIXTURES_REQUIRED b86_decoded)
add_test(NAME native_baseline86_zero_denominator
  COMMAND native_baseline86_cli measure Zlib int64 3
    "${_B86_COLD}/empty.values.bin" "${_B86_COLD}/all-missing.bitmap" 0 1
    "${_B86_COLD}/empty.frame.ngf")
set_tests_properties(native_baseline86_zero_denominator PROPERTIES
  FIXTURES_REQUIRED b86_input PASS_REGULAR_EXPRESSION "\\\"status\\\":\\\"N/A\\\"")
add_test(NAME native_baseline86_cold_matrix_encode
  COMMAND native_baseline86_test encode-matrix "${_B86_COLD}")
set_tests_properties(native_baseline86_cold_matrix_encode PROPERTIES
  FIXTURES_REQUIRED b86_input FIXTURES_SETUP b86_matrix)
add_test(NAME native_baseline86_cold_matrix_decode
  COMMAND native_baseline86_test decode-matrix "${_B86_COLD}")
set_tests_properties(native_baseline86_cold_matrix_decode PROPERTIES
  FIXTURES_REQUIRED b86_matrix)
