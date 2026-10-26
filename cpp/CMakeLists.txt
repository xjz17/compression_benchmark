cmake_minimum_required(VERSION 3.18)
project(myapp)
enable_language(C)

# Minimal Xcode CLT installs may leave CMAKE_OSX_SYSROOT unset; clang then cannot find libc++
# headers (`fatal error: 'vector' file not found`). Prefer the active SDK from `xcrun`.
if(APPLE AND NOT CMAKE_OSX_SYSROOT)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE _WC_MACOS_SDK
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _WC_XCRUN_RC
  )
  if(_WC_XCRUN_RC EQUAL 0 AND _WC_MACOS_SDK)
    set(CMAKE_OSX_SYSROOT "${_WC_MACOS_SDK}" CACHE PATH "macOS SDK for system libc++ headers")
    message(STATUS "CMAKE_OSX_SYSROOT (auto): ${CMAKE_OSX_SYSROOT}")
  endif()
endif()

option(BENCHMARK_CPP_EMBED_SZ
       "Link classic SZ (external/compression/SZ) for in-process sz_float_* rows; disables -static on Linux" OFF)
option(BENCHMARK_CPP_EMBED_SZ3
       "Link SZ3c (external/compression/SZ3) for in-process sz3_float_* rows; disables -static on Linux" OFF)
option(BENCHMARK_CPP_EMBED_SERF
       "Link Serf shared lib for in-process serf_float_* when WEB_COMPRESSION_SERF_BENCH=1; disables -static on Linux"
       OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_TESTING OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(BUILD_UTILITIES OFF CACHE BOOL "")

set(BROTLI_BUILD_TOOLS OFF)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/brotli"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_brotli"
)

# fpzip upstream CMakeLists uses a very old `cmake_minimum_required`; recent CMake
# needs an explicit compatibility floor when consuming it as a subdirectory.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/fpzip"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_fpzip"
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_ZFPY OFF CACHE BOOL "" FORCE)
set(BUILD_UTILITIES OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_CFP OFF CACHE BOOL "" FORCE)

# benchmark_cpp links with -static on Linux (below). zfp may enable OpenMP, and FindOpenMP
# defaults to libgomp.so — incompatible with -static. Prefer libgomp.a from the same GCC.
if(UNIX AND NOT APPLE AND NOT BENCHMARK_CPP_EMBED_SZ AND NOT BENCHMARK_CPP_EMBED_SZ3 AND
   NOT BENCHMARK_CPP_EMBED_SERF)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.a
    OUTPUT_VARIABLE _WC_STATIC_GOMP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_WC_STATIC_GOMP AND EXISTS "${_WC_STATIC_GOMP}" AND NOT IS_DIRECTORY "${_WC_STATIC_GOMP}")
    set(OpenMP_gomp_LIBRARY "${_WC_STATIC_GOMP}" CACHE FILEPATH
        "Static libgomp for benchmark_cpp -static link" FORCE)
  else()
    set(ZFP_WITH_OPENMP OFF CACHE BOOL "Disable zfp OpenMP when static libgomp is unavailable" FORCE)
  endif()
endif()

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/zfp"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_zfp"
)

set(LZ4_BUILD_CLI OFF)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lz4/build/cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_lz4"
)

# Blosc2 (blosclz): build c-blosc2 in an isolated CMake tree (ExternalProject) so its
# bundled LZ4 does not clash with ext_lz4 targets in this project.
set(_WC_BLOSC2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/c-blosc2")
if(NOT EXISTS "${_WC_BLOSC2_SRC}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "c-blosc2 missing at ${_WC_BLOSC2_SRC}; from repo root run:\n"
      "  git submodule update --init external/compression/c-blosc2"
  )
endif()
set(_WC_BLOSC2_EP_BUILD "${CMAKE_CURRENT_BINARY_DIR}/ext_blosc2_ep-build")
set(_WC_BLOSC2_LIB "${_WC_BLOSC2_EP_BUILD}/blosc/libblosc2.a")

include(ExternalProject)
ExternalProject_Add(
    ext_blosc2_ep
    SOURCE_DIR "${_WC_BLOSC2_SRC}"
    BINARY_DIR "${_WC_BLOSC2_EP_BUILD}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_LITE=ON
        -DBUILD_STATIC=ON
        -DBUILD_SHARED=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_FUZZERS=OFF
        -DBUILD_BENCHMARKS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_PLUGINS=OFF
        -DBLOSC_ENABLE_ZFP=OFF
        -DBLOSC_LZ4_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lz4
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target blosc2_static
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_WC_BLOSC2_LIB}"
    USES_TERMINAL_BUILD TRUE
)

add_library(wc_blosc2_static STATIC IMPORTED GLOBAL)
add_dependencies(wc_blosc2_static ext_blosc2_ep)
set_target_properties(
    wc_blosc2_static
    PROPERTIES
        IMPORTED_LOCATION "${_WC_BLOSC2_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_WC_BLOSC2_SRC}/include"
)
add_library(Blosc2::blosc2_static ALIAS wc_blosc2_static)

# Blosc1 (c-blosc): isolated ExternalProject; blosclz-only build (no extra codecs).
set(_WC_BLOSC1_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/c-blosc")
if(NOT EXISTS "${_WC_BLOSC1_SRC}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "c-blosc missing at ${_WC_BLOSC1_SRC}; from repo root run:\n"
      "  git submodule update --init external/compression/c-blosc"
  )
endif()
set(_WC_BLOSC1_EP_BUILD "${CMAKE_CURRENT_BINARY_DIR}/ext_blosc_ep-build")
set(_WC_BLOSC1_EP_SRC "${CMAKE_CURRENT_BINARY_DIR}/ext_blosc_ep-src")
set(_WC_BLOSC1_LIB "${_WC_BLOSC1_EP_BUILD}/blosc/libblosc.a")

ExternalProject_Add(
    ext_blosc_ep
    SOURCE_DIR "${_WC_BLOSC1_EP_SRC}"
    BINARY_DIR "${_WC_BLOSC1_EP_BUILD}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND
        ${CMAKE_COMMAND} -E copy_directory "${_WC_BLOSC1_SRC}" "${_WC_BLOSC1_EP_SRC}"
    COMMAND
        ${CMAKE_COMMAND}
        -Dsrc=${_WC_BLOSC1_EP_SRC}/blosc/blosc.c
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_c_blosc_mingw_pthread.cmake
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_STATIC=ON
        -DBUILD_SHARED=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_FUZZERS=OFF
        -DBUILD_BENCHMARKS=OFF
        -D
        DEACTIVATE_LZ4=ON
        -D
        DEACTIVATE_SNAPPY=ON
        -D
        DEACTIVATE_ZLIB=ON
        -D
        DEACTIVATE_ZSTD=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target blosc_static
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_WC_BLOSC1_LIB}"
    USES_TERMINAL_BUILD TRUE
)

add_library(wc_blosc1_static STATIC IMPORTED GLOBAL)
add_dependencies(wc_blosc1_static ext_blosc_ep)
set_target_properties(
    wc_blosc1_static
    PROPERTIES
        IMPORTED_LOCATION "${_WC_BLOSC1_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_WC_BLOSC1_SRC}/blosc"
)
add_library(Blosc1::blosc_static ALIAS wc_blosc1_static)

set(XZ_TOOL_XZ OFF)
set(XZ_TOOL_XZDEC OFF)
set(XZ_TOOL_LZMADEC OFF)
set(XZ_TOOL_LZMAINFO OFF)
set(XZ_TOOL_SCRIPTS OFF)
set(XZ_DOC OFF)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/xz"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_xz"
)

set(SNAPPY_BUILD_TESTS OFF CACHE BOOL "")
set(SNAPPY_BUILD_BENCHMARKS OFF CACHE BOOL "")
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/snappy"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_snappy"
)

set(ZLIB_BUILD_EXAMPLES OFF)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/zlib"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_zlib"
)

set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_BUILD_TESTS OFF)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/zstd/build/cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ext_zstd"
)

# Lemire FastPFOR + VariableByte (Java `Composition(FastPFOR, VariableByte)` parity).
# Build only the static lib from vendored sources — do not use upstream CMakeLists (CPM download on configure).
# - x86_64 / amd64: native -msse4.2 (GCC, Clang, AppleClang) or /arch:AVX (MSVC x64).
# - Apple Silicon / Linux arm64 / Windows on Arm: SIMDe translates SSE intrinsics.
set(FP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/FastPFor")

set(FP_USE_SIMDE FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|^arm|ARM")
  set(FP_USE_SIMDE TRUE)
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
  set(FP_USE_SIMDE TRUE)
endif()

set(FP_SIMDE_ROOT "")
if(FP_USE_SIMDE)
  if(DEFINED ENV{SIMDE_ROOT} AND EXISTS "$ENV{SIMDE_ROOT}/simde/x86/sse4.1.h")
    set(FP_SIMDE_ROOT "$ENV{SIMDE_ROOT}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../external/submodules/simde/simde/x86/sse4.1.h")
    set(FP_SIMDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/submodules/simde")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/simde/simde/x86/sse4.1.h")
    set(FP_SIMDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/simde")
  else()
    foreach(_d IN ITEMS /opt/homebrew/include /usr/local/include)
      if(EXISTS "${_d}/simde/x86/sse4.1.h")
        set(FP_SIMDE_ROOT "${_d}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT FP_SIMDE_ROOT)
    message(
      FATAL_ERROR
        "FastPFOR on ${CMAKE_SYSTEM_PROCESSOR} needs SIMDe. From repo root run:\n"
        "  git submodule update --init external/submodules/simde\n"
        "Or install simde (e.g. brew install simde) and set SIMDE_ROOT to the parent of the simde/ include directory."
    )
  endif()
endif()

add_library(
  FastPFOR STATIC
  ${FP_ROOT}/src/bitpacking.cpp
  ${FP_ROOT}/src/bitpackingaligned.cpp
  ${FP_ROOT}/src/bitpackingunaligned.cpp
  ${FP_ROOT}/src/horizontalbitpacking.cpp
  ${FP_ROOT}/src/simdunalignedbitpacking.cpp
  ${FP_ROOT}/src/codecfactory.cpp
  ${FP_ROOT}/src/simdbitpacking.cpp
  ${FP_ROOT}/src/varintdecode.c
  ${FP_ROOT}/src/streamvbyte.c
)
target_include_directories(FastPFOR PUBLIC ${FP_ROOT}/headers)
if(FP_USE_SIMDE)
  target_include_directories(FastPFOR PUBLIC ${FP_SIMDE_ROOT})
  target_compile_definitions(FastPFOR PUBLIC SIMDE_ENABLE_NATIVE_ALIASES)
elseif(MSVC)
  target_compile_options(FastPFOR PRIVATE /arch:AVX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(FastPFOR PRIVATE -msse4.2)
else()
  message(
    FATAL_ERROR
      "FastPFOR: unsupported toolchain (${CMAKE_CXX_COMPILER_ID}) for CPU ${CMAKE_SYSTEM_PROCESSOR}. "
      "Use GCC, Clang, AppleClang, or MSVC."
  )
endif()
set_target_properties(FastPFOR PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

find_package(Threads REQUIRED)

add_executable(
    benchmark_cpp
    src/spdp.c
    src/fpc.c
    src/pfpc.c
    src/main.cpp
    src/ans_rans.cpp
    src/huffman_bytes.cpp
    src/arithmetic_bytes.cpp
    src/simple8b_int_codec.cpp
    src/subcolumn_codec.cpp
    src/grouping_codec.cpp
    src/ts2diff_bos_native.cpp
    src/ts2diff_operator_ablation.cpp
    src/column_encodings.cpp
    src/java_fastpfor_codec.cpp
    src/spdp_codec.cpp
    src/fpc_codec.cpp
    src/pfpc_codec.cpp
    src/zfp_codec.cpp
    src/fpzip_codec.cpp
    src/fpcompress_codec.cpp
    src/blosc_codec.cpp
    src/blosc2_codec.cpp
    src/streamvbyte_int_codec.cpp
    src/most_float_codec.cpp
    src/machete_float_codec.cpp
    src/corad_tricklet_float_codec.cpp
    src/chimp32_paper_codec.cpp
    src/pongo_codec.cpp
    src/leco_float_codec.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/alp_float_codec.cpp
    src/elf_float_codec.cpp
    src/elf_double_codec.cpp
    src/camel_float_codec.cpp
    src/simpiece_float_codec.cpp
    src/native_streams.cpp
    src/native_lz_family.cpp
    src/reger_codec.cpp
    src/research_cli.cpp
    src/buff_kernel/buff_precision_bound.cpp
    src/buff_kernel/buff_bit_pack.cpp
    src/buff_kernel/buff_split_double_codec.cpp
    src/buff_kernel/buff_simd_native.cpp
    src/buff_kernel/buff_roaring_lite.cpp
    src/research_cli_sz3_embed.cpp
    src/research_cli_serf_embed.cpp
    src/native_cli_subprocess.cpp
    src/run_java_companion.cpp
    src/column_info_loader.cpp
    src/process_csv_patch_bench.cpp
    src/algorithm_allow_list.cpp
    src/ts2diff_bos_java.cpp
    src/elf_double_java.cpp
    src/tsfile_bench_tsfile.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_encode_p03.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
    src/tsfile_encoding/tsfile_encode_p04.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_decode_p03.cpp
    src/tsfile_encoding/tsfile_decode_p04.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
)

target_include_directories(benchmark_cpp
    PRIVATE
    ../external/compression/xz/src/liblzma/api
    "${CMAKE_SOURCE_DIR}/src"
)

add_executable(
    reger_codec_test
    tests/reger_codec_test.cpp
    src/reger_codec.cpp
)
target_include_directories(reger_codec_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
enable_testing()
add_test(NAME reger_codec_test COMMAND reger_codec_test)

add_executable(
    reger_profile_bench
    tests/reger_profile_bench.cpp
    src/reger_codec.cpp
)
target_include_directories(reger_profile_bench PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    pongo_codec_test
    tests/pongo_codec_test.cpp
    src/pongo_codec.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(pongo_codec_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME pongo_codec_test COMMAND pongo_codec_test)

add_executable(
    pongo_exact_stages_test
    tests/pongo_exact_stages_test.cpp
    src/pongo_codec.cpp
)
target_include_directories(pongo_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME pongo_exact_stages_test COMMAND pongo_exact_stages_test)

add_executable(
    pongo_exact_stages_cli
    src/pongo_exact_stages_cli.cpp
    src/pongo_codec.cpp
)
target_include_directories(pongo_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    lossless_target_codecs_test
    tests/lossless_target_codecs_test.cpp
    src/simple8b_int_codec.cpp
    src/corad_tricklet_float_codec.cpp
    src/native_lz_family.cpp
    src/subcolumn_codec.cpp
    src/leco_double_codec.cpp
    src/leco_float_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/dbp_exact_stages.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_encode_p04.cpp
    src/tsfile_encoding/tsfile_decode_p04.cpp
    src/tsfile_encoding/tsfile_encode_p03.cpp
    src/tsfile_encoding/tsfile_decode_p03.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
    src/buff_kernel/buff_precision_bound.cpp
    src/buff_kernel/buff_bit_pack.cpp
    src/buff_kernel/buff_split_double_codec.cpp
)
target_include_directories(lossless_target_codecs_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(lossless_target_codecs_test PRIVATE FastPFOR libzstd_static zlibstatic)
add_test(NAME lossless_target_codecs_test COMMAND lossless_target_codecs_test)

add_executable(
    remaining_codec_action_catalog_test
    tests/remaining_codec_action_catalog_test.cpp
    src/remaining_codec_action_catalog.cpp
)
target_include_directories(remaining_codec_action_catalog_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME remaining_codec_action_catalog_test COMMAND remaining_codec_action_catalog_test)

add_executable(
    bos_exact_stages_test
    tests/bos_exact_stages_test.cpp
    src/bos_exact_stages.cpp
    src/bos_operator_ablation.cpp
    src/ts2diff_bos_native.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/dbp_exact_stages.cpp
)
target_include_directories(bos_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(bos_exact_stages_test PRIVATE FastPFOR libzstd_static zlibstatic)
add_test(NAME bos_exact_stages_test COMMAND bos_exact_stages_test)

add_executable(
    bos_exact_stages_cli
    src/bos_exact_stages_cli.cpp
    src/bos_exact_stages.cpp
    src/ts2diff_bos_native.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/dbp_exact_stages.cpp
)
target_include_directories(bos_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(bos_exact_stages_cli PRIVATE FastPFOR libzstd_static zlibstatic)

add_executable(
    fastpfor_exact_stages_test
    tests/fastpfor_exact_stages_test.cpp
    src/fastpfor_exact_stages.cpp
    src/dbp_exact_stages.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(fastpfor_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(fastpfor_exact_stages_test PRIVATE FastPFOR)
add_test(NAME fastpfor_exact_stages_test COMMAND fastpfor_exact_stages_test)

add_executable(
    fastpfor_exact_stages_cli
    src/fastpfor_exact_stages_cli.cpp
    src/fastpfor_exact_stages.cpp
    src/dbp_exact_stages.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(fastpfor_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(fastpfor_exact_stages_cli PRIVATE FastPFOR)

add_executable(
    elf_exact_stages_test
    tests/elf_exact_stages_test.cpp
    src/elf_float_codec.cpp
)
target_include_directories(elf_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME elf_exact_stages_test COMMAND elf_exact_stages_test)

add_executable(
    elf_exact_stages_cli
    src/elf_exact_stages_cli.cpp
    src/elf_float_codec.cpp
)
target_include_directories(elf_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(elf64_exact_stages_cli
    src/elf64_exact_stages_cli.cpp
    src/elf_double_codec.cpp
)
target_include_directories(elf64_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    camel_exact_stages_test
    tests/camel_exact_stages_test.cpp
    src/camel_exact_stages.cpp
)
target_include_directories(camel_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME camel_exact_stages_test COMMAND camel_exact_stages_test)

add_executable(
    camel_exact_stages_cli
    src/camel_exact_stages_cli.cpp
    src/camel_exact_stages.cpp
)
target_include_directories(camel_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    snappy_exact_stages_test
    tests/snappy_exact_stages_test.cpp
    src/snappy_exact_stages.cpp
)
target_include_directories(snappy_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(snappy_exact_stages_test PRIVATE snappy)
add_test(NAME snappy_exact_stages_test COMMAND snappy_exact_stages_test)

add_executable(
    snappy_exact_stages_cli
    src/snappy_exact_stages_cli.cpp
    src/snappy_exact_stages.cpp
)
target_include_directories(snappy_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(snappy_exact_stages_cli PRIVATE snappy)

add_executable(
    lz4_exact_stages_test
    tests/lz4_exact_stages_test.cpp
    src/lz4_exact_stages.cpp
)
target_include_directories(lz4_exact_stages_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/../external/compression/lz4/lib"
)
target_link_libraries(lz4_exact_stages_test PRIVATE lz4)
add_test(NAME lz4_exact_stages_test COMMAND lz4_exact_stages_test)

add_executable(
    lz4_exact_stages_cli
    src/lz4_exact_stages_cli.cpp
    src/lz4_exact_stages.cpp
)
target_include_directories(lz4_exact_stages_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/../external/compression/lz4/lib"
)
target_link_libraries(lz4_exact_stages_cli PRIVATE lz4)

add_executable(
    zstd_exact_stages_test
    tests/zstd_exact_stages_test.cpp
    src/zstd_exact_stages.cpp
)
target_include_directories(zstd_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(zstd_exact_stages_test PRIVATE libzstd_static)
add_test(NAME zstd_exact_stages_test COMMAND zstd_exact_stages_test)

add_executable(
    zstd_exact_stages_cli
    src/zstd_exact_stages_cli.cpp
    src/zstd_exact_stages.cpp
)
target_include_directories(zstd_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(zstd_exact_stages_cli PRIVATE libzstd_static)

add_executable(
    deflate_container_stages_test
    tests/deflate_container_stages_test.cpp
    src/deflate_container_stages.cpp
)
target_include_directories(deflate_container_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(deflate_container_stages_test PRIVATE zlibstatic)
add_test(NAME deflate_container_stages_test COMMAND deflate_container_stages_test)

add_executable(
    deflate_container_stages_cli
    src/deflate_container_stages_cli.cpp
    src/deflate_container_stages.cpp
)
target_include_directories(deflate_container_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    xz_container_stage_test
    tests/xz_container_stage_test.cpp
    src/xz_container_stage.cpp
)
target_include_directories(xz_container_stage_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME xz_container_stage_test COMMAND xz_container_stage_test)

add_executable(
    xz_container_stage_cli
    src/xz_container_stage_cli.cpp
    src/xz_container_stage.cpp
)
target_include_directories(xz_container_stage_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    fpzip_exact_stages_test
    tests/fpzip_exact_stages_test.cpp
    src/fpzip_exact_stages.cpp
)
target_include_directories(fpzip_exact_stages_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/../external/compression/fpzip/src"
)
target_link_libraries(fpzip_exact_stages_test PRIVATE fpzip)
add_test(NAME fpzip_exact_stages_test COMMAND fpzip_exact_stages_test)

add_executable(
    fpzip_exact_stages_cli
    src/fpzip_exact_stages_cli.cpp
    src/fpzip_exact_stages.cpp
)
target_include_directories(fpzip_exact_stages_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/../external/compression/fpzip/src"
)
target_link_libraries(fpzip_exact_stages_cli PRIVATE fpzip)

add_executable(
    ts2diff_exact_stages_test
    tests/ts2diff_exact_stages_test.cpp
    src/ts2diff_exact_stages.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
)
target_include_directories(ts2diff_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME ts2diff_exact_stages_test COMMAND ts2diff_exact_stages_test)

add_executable(
    ts2diff_exact_stages_cli
    src/ts2diff_exact_stages_cli.cpp
    src/ts2diff_exact_stages.cpp
)
target_include_directories(ts2diff_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    svb_exact_stages_test
    tests/svb_exact_stages_test.cpp
    src/svb_exact_stages.cpp
    src/dbp_exact_stages.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(svb_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(svb_exact_stages_test PRIVATE FastPFOR libzstd_static)
add_test(NAME svb_exact_stages_test COMMAND svb_exact_stages_test)

add_executable(
    svb_exact_stages_cli
    src/svb_exact_stages_cli.cpp
    src/svb_exact_stages.cpp
    src/dbp_exact_stages.cpp
)
target_include_directories(svb_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(svb_exact_stages_cli PRIVATE FastPFOR)

add_executable(
    simple8b_exact_stages_test
    tests/simple8b_exact_stages_test.cpp
    src/simple8b_exact_stages.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(simple8b_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(simple8b_exact_stages_test PRIVATE FastPFOR libzstd_static)
add_test(NAME simple8b_exact_stages_test COMMAND simple8b_exact_stages_test)

add_executable(
    simple8b_exact_stages_cli
    src/simple8b_exact_stages_cli.cpp
    src/simple8b_exact_stages.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(simple8b_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(simple8b_exact_stages_cli PRIVATE FastPFOR libzstd_static)

add_executable(
    ans_rans_test
    tests/ans_rans_test.cpp
    src/ans_rans.cpp
)
target_include_directories(ans_rans_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME ans_rans_test COMMAND ans_rans_test)

add_executable(
    ans_rans_cli
    src/ans_rans_cli.cpp
    src/ans_rans.cpp
)
target_include_directories(ans_rans_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    remaining_codec_action_catalog_cli
    src/remaining_codec_action_catalog_cli.cpp
    src/remaining_codec_action_catalog.cpp
)
target_include_directories(remaining_codec_action_catalog_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    learning_codec_exact_stages_test
    tests/learning_codec_exact_stages_test.cpp
    src/learning_codec_exact_stages.cpp
    src/alp_float_codec.cpp
    src/reger_codec.cpp
    src/leco_operator_ablation.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/buff_kernel/buff_split_double_codec.cpp
    src/buff_kernel/buff_bit_pack.cpp
    src/buff_kernel/buff_precision_bound.cpp
)
target_include_directories(learning_codec_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(learning_codec_exact_stages_test PRIVATE FastPFOR libzstd_static)
add_test(NAME learning_codec_exact_stages_test COMMAND learning_codec_exact_stages_test)

add_executable(
    learning_codec_exact_stages_cli
    src/learning_codec_exact_stages_cli.cpp
    src/learning_codec_exact_stages.cpp
    src/alp_float_codec.cpp
    src/reger_codec.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(learning_codec_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(learning_codec_exact_stages_cli PRIVATE FastPFOR libzstd_static)

add_executable(max_value_bitpack_cli src/max_value_bitpack_cli.cpp)
add_executable(residual_window_pack_cli src/residual_window_pack_cli.cpp)
target_include_directories(residual_window_pack_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(max_value_bitpack_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(max_value_bitpack_test tests/max_value_bitpack_test.cpp)
target_include_directories(max_value_bitpack_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME max_value_bitpack_test COMMAND max_value_bitpack_test)

add_executable(
    max_value_bitpack_adapter_test
    tests/max_value_bitpack_adapter_test.cpp
    src/ts2diff_exact_stages.cpp
    src/learning_codec_exact_stages.cpp
    src/alp_float_codec.cpp
    src/reger_codec.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
target_include_directories(max_value_bitpack_adapter_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(max_value_bitpack_adapter_test PRIVATE FastPFOR libzstd_static)
add_test(NAME max_value_bitpack_adapter_test COMMAND max_value_bitpack_adapter_test)

add_executable(minimum_for_cli src/minimum_for_cli.cpp)
target_include_directories(minimum_for_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_executable(minimum_for_test tests/minimum_for_test.cpp)
target_include_directories(minimum_for_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME minimum_for_test COMMAND minimum_for_test)
add_executable(minimum_for_adapter_test
    tests/minimum_for_adapter_test.cpp
    src/reference_operator_actions.cpp
    src/ts2diff_exact_stages.cpp
    src/bos_exact_stages.cpp
    src/learning_codec_exact_stages.cpp
    src/ts2diff_bos_native.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/alp_float_codec.cpp
    src/reger_codec.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/dbp_exact_stages.cpp)
target_include_directories(minimum_for_adapter_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(minimum_for_adapter_test PRIVATE FastPFOR libzstd_static zlibstatic)
add_test(NAME minimum_for_adapter_test COMMAND minimum_for_adapter_test)
add_executable(minimum_for_legacy_test tests/minimum_for_legacy_test.cpp)
target_include_directories(minimum_for_legacy_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME minimum_for_legacy_test COMMAND minimum_for_legacy_test)

set(BUFF_NATIVE_EXACT_STAGE_SOURCES
    src/buff_native_exact_stages.cpp
    src/buff_kernel/buff_split_double_codec.cpp
    src/buff_kernel/buff_bit_pack.cpp
    src/leco_double_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
)
add_executable(
    buff_native_exact_stages_cli
    src/buff_native_exact_stages_cli.cpp
    ${BUFF_NATIVE_EXACT_STAGE_SOURCES}
)
target_include_directories(buff_native_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(buff_native_exact_stages_cli PRIVATE FastPFOR libzstd_static)

add_executable(
    buff_native_exact_stages_test
    tests/buff_native_exact_stages_test.cpp
    ${BUFF_NATIVE_EXACT_STAGE_SOURCES}
)
target_include_directories(buff_native_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(buff_native_exact_stages_test PRIVATE FastPFOR libzstd_static)
add_test(NAME buff_native_exact_stages_test COMMAND buff_native_exact_stages_test)

add_executable(
    corad_lossless_counterexample_test
    tests/corad_lossless_counterexample_test.cpp
)
target_include_directories(corad_lossless_counterexample_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(corad_lossless_counterexample_test PRIVATE zlibstatic)
add_test(NAME corad_lossless_counterexample_test COMMAND corad_lossless_counterexample_test)

add_executable(
    general_codec_reference_stages_test
    tests/general_codec_reference_stages_test.cpp
    src/general_codec_reference_stages.cpp
    src/huffman_bytes.cpp
    src/ans_rans.cpp
    src/arithmetic_bytes.cpp
)
target_include_directories(general_codec_reference_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME general_codec_reference_stages_test COMMAND general_codec_reference_stages_test)

add_executable(
    general_codec_reference_stages_cli
    src/general_codec_reference_stages_cli.cpp
    src/general_codec_reference_stages.cpp
    src/huffman_bytes.cpp
    src/ans_rans.cpp
    src/arithmetic_bytes.cpp
)
target_include_directories(general_codec_reference_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

set(GENERAL_CODEC_RESIDUAL_COMMON_SOURCES
    src/general_codec_residual_stages.cpp
    src/general_codec_reference_stages.cpp
    src/deflate_container_stages.cpp
    src/xz_container_stage.cpp
    src/blosc_codec.cpp
    src/blosc2_codec.cpp
    src/bzip2_operator_ablation.cpp
    src/huffman_bytes.cpp
    src/ans_rans.cpp
    src/arithmetic_bytes.cpp
    ../external/compression/lzbench/bwt/bzip2/blocksort.c
    ../external/compression/lzbench/bwt/bzip2/huffman.c
    ../external/compression/lzbench/bwt/bzip2/crctable.c
    ../external/compression/lzbench/bwt/bzip2/randtable.c
    ../external/compression/lzbench/bwt/bzip2/compress.c
    ../external/compression/lzbench/bwt/bzip2/decompress.c
    ../external/compression/lzbench/bwt/bzip2/bzlib.c
)

add_executable(
    general_codec_residual_stages_test
    tests/general_codec_residual_stages_test.cpp
    ${GENERAL_CODEC_RESIDUAL_COMMON_SOURCES}
)
target_include_directories(general_codec_residual_stages_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/xz/src/liblzma/api"
    "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lzbench/bwt/bzip2"
)
target_link_libraries(general_codec_residual_stages_test PRIVATE
    brotlienc brotlidec brotlicommon libzstd_static zlibstatic liblzma
    Blosc1::blosc_static Blosc2::blosc2_static
)
add_test(NAME general_codec_residual_stages_test
         COMMAND general_codec_residual_stages_test)

add_executable(
    general_codec_residual_stages_cli
    src/general_codec_residual_stages_cli.cpp
    ${GENERAL_CODEC_RESIDUAL_COMMON_SOURCES}
)
target_include_directories(general_codec_residual_stages_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/xz/src/liblzma/api"
    "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lzbench/bwt/bzip2"
)
target_link_libraries(general_codec_residual_stages_cli PRIVATE
    brotlienc brotlidec brotlicommon libzstd_static zlibstatic liblzma
    Blosc1::blosc_static Blosc2::blosc2_static
)

add_executable(
    rle_int64_action_cli
    src/rle_int64_action_cli.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
)
target_include_directories(rle_int64_action_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    rle_exact_stages_cli
    src/rle_exact_stages.cpp
    src/rle_exact_stages_cli.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
)
target_include_directories(rle_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    rle_exact_stages_test
    src/rle_exact_stages.cpp
    tests/rle_exact_stages_test.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
)
target_include_directories(rle_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME rle_exact_stages_test COMMAND rle_exact_stages_test)

add_executable(
    p04_exact_stages_test
    tests/p04_exact_stages_test.cpp
    src/p04_exact_stages.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(p04_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME p04_exact_stages_test COMMAND p04_exact_stages_test)

add_executable(
    p04_exact_stages_cli
    src/p04_exact_stages_cli.cpp
    src/p04_exact_stages.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(p04_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    dbp_exact_stages_test
    tests/dbp_exact_stages_test.cpp
    src/dbp_exact_stages.cpp
)
target_include_directories(dbp_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME dbp_exact_stages_test COMMAND dbp_exact_stages_test)

add_executable(
    dbp_exact_stages_cli
    src/dbp_exact_stages_cli.cpp
    src/dbp_exact_stages.cpp
)
target_include_directories(dbp_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    gorilla_exact_stages_test
    tests/gorilla_exact_stages_test.cpp
    src/gorilla_exact_stages.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(gorilla_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME gorilla_exact_stages_test COMMAND gorilla_exact_stages_test)

add_executable(
    gorilla_exact_stages_cli
    src/gorilla_exact_stages_cli.cpp
    src/gorilla_exact_stages.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(gorilla_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    chimp128_exact_stages_test
    tests/chimp128_exact_stages_test.cpp
    src/chimp128_exact_stages.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(chimp128_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME chimp128_exact_stages_test COMMAND chimp128_exact_stages_test)

add_executable(
    chimp128_exact_stages_cli
    src/chimp128_exact_stages_cli.cpp
    src/chimp128_exact_stages.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
)
target_include_directories(chimp128_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    spdp_exact_stages_test
    tests/spdp_exact_stages_test.cpp
    src/spdp_exact_stages.cpp
)
target_include_directories(spdp_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME spdp_exact_stages_test COMMAND spdp_exact_stages_test)

add_executable(
    spdp_exact_stages_cli
    src/spdp_exact_stages_cli.cpp
    src/spdp_exact_stages.cpp
)
target_include_directories(spdp_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    fpc_exact_stages_test
    tests/fpc_exact_stages_test.cpp
    src/fpc_exact_stages.cpp
    src/fpc.c
)
target_include_directories(fpc_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME fpc_exact_stages_test COMMAND fpc_exact_stages_test)

add_executable(
    fpc_exact_stages_cli
    src/fpc_exact_stages_cli.cpp
    src/fpc_exact_stages.cpp
)
target_include_directories(fpc_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    pfpc_exact_stages_test
    tests/pfpc_exact_stages_test.cpp
    src/pfpc_exact_stages.cpp
    src/pfpc.c
)
target_include_directories(pfpc_exact_stages_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(pfpc_exact_stages_test PRIVATE pthread)
add_test(NAME pfpc_exact_stages_test COMMAND pfpc_exact_stages_test)

add_executable(
    pfpc_exact_stages_cli
    src/pfpc_exact_stages_cli.cpp
    src/pfpc_exact_stages.cpp
)
target_include_directories(pfpc_exact_stages_cli PRIVATE "${CMAKE_SOURCE_DIR}/src")

add_executable(
    bos_native_microbench
    tests/bos_native_microbench.cpp
    src/ts2diff_bos_native.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/leco_double_codec.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
)
target_include_directories(bos_native_microbench PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(bos_native_microbench PRIVATE FastPFOR libzstd_static)

add_executable(
    operator_ablation_bench
    tests/operator_ablation_bench.cpp
    src/ans_rans.cpp
    src/alp_float_codec.cpp
    src/camel_float_codec.cpp
    src/elf_double_codec.cpp
    src/fpc.c
    src/fpc_codec.cpp
    src/pfpc.c
    src/fastpfor_operator_ablation.cpp
    src/spdp.c
    src/native_lz_family.cpp
    src/lzf_operator_ablation.cpp
    ../external/compression/lzbench/lz/lzf/lzf_c_ultra.c
    ../external/compression/lzbench/lz/lzf/lzf_c_very.c
    ../external/compression/lzbench/lz/lzf/lzf_d.c
    ../external/operator_ablation_sources/lzf_small_window.c
    src/native_streams.cpp
    src/blosc_codec.cpp
    src/blosc2_codec.cpp
    src/bos_operator_ablation.cpp
    src/bzip2_operator_ablation.cpp
    src/lzjb_operator_ablation.cpp
    src/column_encodings.cpp
    src/pongo_codec.cpp
    src/reger_codec.cpp
    src/ts2diff_bos_native.cpp
    src/ts2diff_operator_ablation.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
    src/leco_double_codec.cpp
    src/leco_operator_ablation.cpp
    src/buff_kernel/buff_bit_pack.cpp
    src/buff_kernel/buff_precision_bound.cpp
    src/buff_kernel/buff_split_double_codec.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
    ../external/compression/lzbench/bwt/bzip2/blocksort.c
    ../external/compression/lzbench/bwt/bzip2/huffman.c
    ../external/compression/lzbench/bwt/bzip2/crctable.c
    ../external/compression/lzbench/bwt/bzip2/randtable.c
    ../external/compression/lzbench/bwt/bzip2/compress.c
    ../external/compression/lzbench/bwt/bzip2/decompress.c
    ../external/compression/lzbench/bwt/bzip2/bzlib.c
)
target_include_directories(operator_ablation_bench PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(operator_ablation_bench PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lzbench/lz/lzf"
                           "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lzbench/bwt/bzip2")
target_link_libraries(operator_ablation_bench PRIVATE FastPFOR libzstd_static lz4 snappy zlibstatic
                                                        Blosc1::blosc_static Blosc2::blosc2_static fpzip)

add_executable(
    lzbench_easy_source_air_bench
    tests/lzbench_easy_source_air_bench.cpp
    ../external/compression/lzbench/lz/brieflz/brieflz.c
    ../external/compression/lzbench/lz/brieflz/depack.c
    ../external/compression/lzbench/lz/brieflz/depacks.c
    ../external/compression/lzbench/lz/crush/crush.cpp
    ../external/compression/lzbench/lz/fastlz/fastlz.c
    ../external/compression/lzbench/lz/lzmat/lzmat_enc.c
    ../external/compression/lzbench/lz/lzmat/lzmat_dec.c
    ../external/compression/lzbench/lz/lzrw/lzrw1.c
    ../external/compression/lzbench/lz/lzrw/lzrw1-a.c
    ../external/compression/lzbench/lz/lzrw/lzrw2.c
    ../external/compression/lzbench/lz/lzrw/lzrw3.c
    ../external/compression/lzbench/lz/lzrw/lzrw3-a.c
    ../external/compression/lzbench/lz/wflz/wfLZ.c
    ../external/compression/lzbench/lz/yappy/yappy.cpp
    ../external/compression/lzbench/lz/liblzg/checksum.c
    ../external/compression/lzbench/lz/liblzg/decode.c
    ../external/compression/lzbench/lz/liblzg/encode.c
    ../external/compression/lzbench/lz/liblzg/version.c
    ../external/compression/lzbench/lz/ucl/alloc.c
    ../external/compression/lzbench/lz/ucl/n2b_99.c
    ../external/compression/lzbench/lz/ucl/n2b_d.c
    ../external/compression/lzbench/lz/ucl/n2b_ds.c
    ../external/compression/lzbench/lz/ucl/n2b_to.c
    ../external/compression/lzbench/lz/ucl/n2d_99.c
    ../external/compression/lzbench/lz/ucl/n2d_d.c
    ../external/compression/lzbench/lz/ucl/n2d_ds.c
    ../external/compression/lzbench/lz/ucl/n2d_to.c
    ../external/compression/lzbench/lz/ucl/n2e_99.c
    ../external/compression/lzbench/lz/ucl/n2e_d.c
    ../external/compression/lzbench/lz/ucl/n2e_ds.c
    ../external/compression/lzbench/lz/ucl/n2e_to.c
    ../external/compression/lzbench/lz/ucl/ucl_crc.c
    ../external/compression/lzbench/lz/ucl/ucl_init.c
    ../external/compression/lzbench/lz/ucl/ucl_ptr.c
    ../external/compression/lzbench/lz/ucl/ucl_str.c
    ../external/compression/lzbench/lz/ucl/ucl_util.c
)
target_include_directories(lzbench_easy_source_air_bench PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lzbench"
                           "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/lzbench/lz")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  if(MSVC)
    set_property(SOURCE src/buff_kernel/buff_simd_native.cpp APPEND PROPERTY COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_property(SOURCE src/buff_kernel/buff_simd_native.cpp APPEND PROPERTY COMPILE_OPTIONS "-mavx2" "-mfma")
  endif()
endif()

add_library(CLI11 INTERFACE)
target_include_directories(CLI11 INTERFACE ../external/submodules/CLI11/include)

add_library(csv2 INTERFACE)
target_include_directories(csv2 INTERFACE ../external/submodules/csv2/single_include)

if(NOT APPLE AND NOT BENCHMARK_CPP_EMBED_SZ AND NOT BENCHMARK_CPP_EMBED_SZ3 AND NOT
                       BENCHMARK_CPP_EMBED_SERF)
  target_link_options(benchmark_cpp PRIVATE -static)
endif()

target_link_libraries(benchmark_cpp
    PRIVATE
    Threads::Threads
    CLI11
    csv2
    brotlienc brotlidec brotlicommon
    zfp::zfp
    lz4
    liblzma
    snappy
    fpzip
    Blosc1::blosc_static
    Blosc2::blosc2_static
    FastPFOR
    # libzstd_shared
    libzstd_static
    # zlib
    zlibstatic
)
if(BENCHMARK_CPP_EMBED_SZ)
  set(_WC_SZ_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/SZ")
  if(NOT EXISTS "${_WC_SZ_SRC}/CMakeLists.txt")
    message(FATAL_ERROR "BENCHMARK_CPP_EMBED_SZ is ON but ${_WC_SZ_SRC} is missing (init SZ submodule)")
  endif()
  # Shared libSZ + bundled zstd/zlib inside the DSO avoids duplicate static zlib/zstd symbols with benchmark_cpp.
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(SZ_FIND_DEPS OFF CACHE BOOL "" FORCE)
  set(BUILD_SZ_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory("${_WC_SZ_SRC}" "${CMAKE_BINARY_DIR}/wc_sz_embed" EXCLUDE_FROM_ALL)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  target_link_libraries(benchmark_cpp PRIVATE SZ)
  target_compile_definitions(benchmark_cpp PRIVATE WEB_COMPRESSION_EMBED_SZ=1)
endif()

if(BENCHMARK_CPP_EMBED_SZ3)
  set(_WC_SZ3_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/SZ3")
  if(NOT EXISTS "${_WC_SZ3_SRC}/CMakeLists.txt")
    message(FATAL_ERROR "BENCHMARK_CPP_EMBED_SZ3 is ON but ${_WC_SZ3_SRC} is missing (init SZ3 submodule)")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(_WC_EMBED_SZ3_ZSTD IMPORTED_TARGET libzstd)
  if(NOT _WC_EMBED_SZ3_ZSTD_FOUND)
    message(
      FATAL_ERROR
        "BENCHMARK_CPP_EMBED_SZ3 needs system libzstd (pkg-config module 'libzstd'). "
        "e.g. macOS: brew install zstd | Ubuntu: apt install libzstd-dev"
    )
  endif()
  # Bundled SZ3/tools/zstd defines target `zstd`, which clashes with external/compression/zstd's `zstd` CLI target.
  set(SZ3_USE_BUNDLED_ZSTD OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  # tools/sz3c is added only when BUILD_SZ3_BINARY is ON
  set(BUILD_SZ3_BINARY ON CACHE BOOL "" FORCE)
  add_subdirectory("${_WC_SZ3_SRC}" "${CMAKE_BINARY_DIR}/wc_sz3_embed" EXCLUDE_FROM_ALL)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  target_link_libraries(benchmark_cpp PRIVATE SZ3c)
  target_compile_definitions(benchmark_cpp PRIVATE WEB_COMPRESSION_EMBED_SZ3=1)
endif()

if(BENCHMARK_CPP_EMBED_SERF)
  set(_WC_SERF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/Serf")
  if(NOT EXISTS "${_WC_SERF_SRC}/src/CMakeLists.txt")
    message(FATAL_ERROR "BENCHMARK_CPP_EMBED_SERF is ON but ${_WC_SERF_SRC}/src is missing (init Serf submodule)")
  endif()
  add_subdirectory("${_WC_SERF_SRC}/src" "${CMAKE_BINARY_DIR}/wc_serf_embed" EXCLUDE_FROM_ALL)
  target_link_libraries(benchmark_cpp PRIVATE serf)
  target_include_directories(benchmark_cpp PRIVATE "${_WC_SERF_SRC}/src")
  target_compile_definitions(benchmark_cpp PRIVATE WEB_COMPRESSION_EMBED_SERF=1)
  set_target_properties(
    benchmark_cpp
    PROPERTIES
      BUILD_RPATH "$<TARGET_FILE_DIR:serf>"
      INSTALL_RPATH "$<TARGET_FILE_DIR:serf>"
  )
endif()

option(ENABLE_TSFILE_GOLDEN_TESTS "Build verify_tsfile_goldens manifest/size checker" OFF)
if(ENABLE_TSFILE_GOLDEN_TESTS)
  set(TSFILE_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/tsfile")
  add_executable(verify_tsfile_goldens tests/verify_tsfile_goldens.cpp)
  target_compile_definitions(
    verify_tsfile_goldens
    PRIVATE
    "TSFILE_GOLDEN_DIR=\"${TSFILE_GOLDEN_ROOT}\""
  )

  add_executable(
    tsfile_p02_golden_test
    tests/tsfile_p02_golden_test.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_encode_p03.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp
    src/tsfile_encoding/tsfile_encode_p04.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_decode_p03.cpp
    src/tsfile_encoding/tsfile_decode_p04.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
  )
  target_include_directories(tsfile_p02_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(
    tsfile_p02_golden_test
    PRIVATE
    "TSFILE_GOLDEN_DIR=\"${TSFILE_GOLDEN_ROOT}\""
  )

  set(COLUMN_ENCODINGS_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/column_encodings")
  add_executable(verify_column_encodings_goldens tests/verify_column_encodings_goldens.cpp)
  target_compile_definitions(
    verify_column_encodings_goldens
    PRIVATE
    "COLUMN_ENCODINGS_GOLDEN_DIR=\"${COLUMN_ENCODINGS_GOLDEN_ROOT}\""
  )

  add_executable(
    column_encodings_golden_test
    tests/column_encodings_golden_test.cpp
    src/column_encodings.cpp
  )
  target_include_directories(column_encodings_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(column_encodings_golden_test PRIVATE lz4)
  target_compile_definitions(
    column_encodings_golden_test
    PRIVATE
    "COLUMN_ENCODINGS_GOLDEN_DIR=\"${COLUMN_ENCODINGS_GOLDEN_ROOT}\""
  )

  set(FASTPFOR_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/fastpfor")
  add_executable(verify_fastpfor_goldens tests/verify_fastpfor_goldens.cpp)
  target_compile_definitions(
    verify_fastpfor_goldens
    PRIVATE
    "FASTPFOR_GOLDEN_DIR=\"${FASTPFOR_GOLDEN_ROOT}\""
  )

  add_executable(
    fastpfor_golden_test
    tests/fastpfor_golden_test.cpp
    src/java_fastpfor_codec.cpp
    src/int64_column_codec.cpp
  )
  target_include_directories(fastpfor_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(fastpfor_golden_test PRIVATE FastPFOR)
  target_compile_definitions(
    fastpfor_golden_test
    PRIVATE
    "FASTPFOR_GOLDEN_DIR=\"${FASTPFOR_GOLDEN_ROOT}\""
  )

  set(CHIMP32_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/chimp32")
  add_executable(verify_chimp32_goldens tests/verify_chimp32_goldens.cpp)
  target_compile_definitions(
    verify_chimp32_goldens
    PRIVATE
    "CHIMP32_GOLDEN_DIR=\"${CHIMP32_GOLDEN_ROOT}\""
  )

  add_executable(
    chimp32_golden_test
    tests/chimp32_golden_test.cpp
    src/chimp32_paper_codec.cpp
  )
  target_include_directories(chimp32_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(
    chimp32_golden_test
    PRIVATE
    "CHIMP32_GOLDEN_DIR=\"${CHIMP32_GOLDEN_ROOT}\""
  )

  set(LECO_FLOAT_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/leco_float")
  add_executable(verify_leco_float_goldens tests/verify_leco_float_goldens.cpp)
  target_compile_definitions(
    verify_leco_float_goldens
    PRIVATE
    "LECO_FLOAT_GOLDEN_DIR=\"${LECO_FLOAT_GOLDEN_ROOT}\""
  )

  add_executable(
    leco_float_golden_test
    tests/leco_float_golden_test.cpp
    src/leco_float_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
  )
  target_include_directories(leco_float_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(leco_float_golden_test PRIVATE FastPFOR)
  target_compile_definitions(
    leco_float_golden_test
    PRIVATE
    "LECO_FLOAT_GOLDEN_DIR=\"${LECO_FLOAT_GOLDEN_ROOT}\""
  )

  set(ALP_FLOAT_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/alp_float")
  add_executable(verify_alp_float_goldens tests/verify_alp_float_goldens.cpp)
  target_compile_definitions(
    verify_alp_float_goldens
    PRIVATE
    "ALP_FLOAT_GOLDEN_DIR=\"${ALP_FLOAT_GOLDEN_ROOT}\""
  )

  add_executable(
    alp_float_golden_test
    tests/alp_float_golden_test.cpp
    src/alp_float_codec.cpp
    src/int64_column_codec.cpp
    src/java_fastpfor_codec.cpp
  )
  target_include_directories(alp_float_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(alp_float_golden_test PRIVATE FastPFOR)
  target_compile_definitions(
    alp_float_golden_test
    PRIVATE
    "ALP_FLOAT_GOLDEN_DIR=\"${ALP_FLOAT_GOLDEN_ROOT}\""
  )

  set(ELF_ON_CHIMP_FLOAT_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/elf_on_chimp_float")
  add_executable(verify_elf_on_chimp_float_goldens tests/verify_elf_on_chimp_float_goldens.cpp)
  target_compile_definitions(
    verify_elf_on_chimp_float_goldens
    PRIVATE
    "ELF_ON_CHIMP_FLOAT_GOLDEN_DIR=\"${ELF_ON_CHIMP_FLOAT_GOLDEN_ROOT}\""
  )

  add_executable(
    elf_on_chimp_float_golden_test
    tests/elf_on_chimp_float_golden_test.cpp
    src/elf_float_codec.cpp
  )
  target_include_directories(elf_on_chimp_float_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(
    elf_on_chimp_float_golden_test
    PRIVATE
    "ELF_ON_CHIMP_FLOAT_GOLDEN_DIR=\"${ELF_ON_CHIMP_FLOAT_GOLDEN_ROOT}\""
  )

  set(ELF_PLUS_FLOAT_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/elf_plus_float")
  add_executable(verify_elf_plus_float_goldens tests/verify_elf_plus_float_goldens.cpp)
  target_compile_definitions(
    verify_elf_plus_float_goldens
    PRIVATE
    "ELF_PLUS_FLOAT_GOLDEN_DIR=\"${ELF_PLUS_FLOAT_GOLDEN_ROOT}\""
  )

  add_executable(
    elf_plus_float_golden_test
    tests/elf_plus_float_golden_test.cpp
    src/elf_float_codec.cpp
  )
  target_include_directories(elf_plus_float_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(
    elf_plus_float_golden_test
    PRIVATE
    "ELF_PLUS_FLOAT_GOLDEN_DIR=\"${ELF_PLUS_FLOAT_GOLDEN_ROOT}\""
  )

  set(CAMEL_FLOAT_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/camel_float")
  add_executable(verify_camel_float_goldens tests/verify_camel_float_goldens.cpp)
  target_compile_definitions(
    verify_camel_float_goldens
    PRIVATE
    "CAMEL_FLOAT_GOLDEN_DIR=\"${CAMEL_FLOAT_GOLDEN_ROOT}\""
  )

  add_executable(
    camel_float_golden_test
    tests/camel_float_golden_test.cpp
    src/camel_float_codec.cpp
  )
  target_include_directories(camel_float_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(
    camel_float_golden_test
    PRIVATE
    "CAMEL_FLOAT_GOLDEN_DIR=\"${CAMEL_FLOAT_GOLDEN_ROOT}\""
  )

  set(SIMPIECE_FLOAT_GOLDEN_ROOT "${CMAKE_SOURCE_DIR}/../benchmark/reference/golden/simpiece_float")
  add_executable(verify_simpiece_float_goldens tests/verify_simpiece_float_goldens.cpp)
  target_compile_definitions(
    verify_simpiece_float_goldens
    PRIVATE
    "SIMPIECE_FLOAT_GOLDEN_DIR=\"${SIMPIECE_FLOAT_GOLDEN_ROOT}\""
  )

  add_executable(
    simpiece_float_golden_test
    tests/simpiece_float_golden_test.cpp
    src/simpiece_float_codec.cpp
  )
  target_include_directories(simpiece_float_golden_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(simpiece_float_golden_test PRIVATE libzstd_static)
  target_compile_definitions(
    simpiece_float_golden_test
    PRIVATE
    "SIMPIECE_FLOAT_GOLDEN_DIR=\"${SIMPIECE_FLOAT_GOLDEN_ROOT}\""
  )
endif()

# Bit-preserving int64/float64 action with legacy 64-bit dictionary indices.
add_executable(dictionary64_action_cli src/dictionary64_action_cli.cpp)

# Graph-level columnization and special stream operations (not OpenZL framing).
add_executable(openzl_layout_stages_cli
    src/openzl_layout_stages.cpp src/openzl_layout_stages_cli.cpp)
add_executable(openzl_layout_stages_test
    src/openzl_layout_stages.cpp tests/openzl_layout_stages_test.cpp)
target_include_directories(openzl_layout_stages_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME openzl_layout_stages_test COMMAND openzl_layout_stages_test)

# Real causal predictors extracted from the vendored cmix source. Do not enable
# fast-math: prediction probabilities depend on reproducible floating arithmetic.
set(PREDICTION_CMIX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/compression/cmix/src")
file(GLOB PREDICTION_CMIX_SOURCES CONFIGURE_DEPENDS
    "${PREDICTION_CMIX_ROOT}/contexts/*.cpp"
    "${PREDICTION_CMIX_ROOT}/mixer/*.cpp"
    "${PREDICTION_CMIX_ROOT}/states/*.cpp"
    "${PREDICTION_CMIX_ROOT}/models/*.cpp")
list(FILTER PREDICTION_CMIX_SOURCES EXCLUDE REGEX "/(paq8|ppmd)\\.cpp$")
add_library(prediction_probability_model STATIC src/prediction_probability_model.cpp
    "${PREDICTION_CMIX_ROOT}/predictor.cpp"
    "${PREDICTION_CMIX_ROOT}/context-manager.cpp" ${PREDICTION_CMIX_SOURCES})
target_include_directories(prediction_probability_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(prediction_probability_model PRIVATE -fno-fast-math -ffp-contract=off)
add_executable(prediction_probability_cli src/prediction_probability_cli.cpp)
target_link_libraries(prediction_probability_cli PRIVATE prediction_probability_model)
target_compile_options(prediction_probability_cli PRIVATE -fno-fast-math -ffp-contract=off)
add_executable(prediction_entropy_codec_cli src/prediction_entropy_codec_cli.cpp)
target_link_libraries(prediction_entropy_codec_cli PRIVATE prediction_probability_model)
target_compile_options(prediction_entropy_codec_cli PRIVATE -fno-fast-math -ffp-contract=off)

add_executable(reference_operator_actions_cli
    src/reference_operator_actions.cpp src/reference_operator_actions_cli.cpp)
target_link_libraries(reference_operator_actions_cli PRIVATE zlibstatic)
target_compile_options(reference_operator_actions_cli PRIVATE -ffp-contract=off)

# Frozen zlib level-6 timing scale for E0/D0.  The helper builds a complete
# self-describing frame and times only in-process encode/decode, not CLI startup.
add_executable(zlib6_reference_benchmark_cli
    src/zlib6_reference_benchmark_cli.cpp)
target_link_libraries(zlib6_reference_benchmark_cli PRIVATE zlibstatic)
target_compile_options(zlib6_reference_benchmark_cli PRIVATE -ffp-contract=off)

add_executable(unsigned_leb128_cli src/unsigned_leb128_cli.cpp)

add_executable(native_lz_action_stages_cli
    src/native_lz_action_stages.cpp src/native_lz_action_stages_cli.cpp)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    PATHS /opt/homebrew/include /usr/local/include REQUIRED)
add_executable(legacy_numeric_action_stages_cli src/native_lz_action_stages.cpp
    src/legacy_numeric_action_stages.cpp src/legacy_numeric_action_stages_cli.cpp)
target_include_directories(legacy_numeric_action_stages_cli PRIVATE "${NLOHMANN_JSON_INCLUDE_DIR}")
target_compile_options(legacy_numeric_action_stages_cli PRIVATE -ffp-contract=off)

add_executable(native_legacy_action_stages_test
    tests/native_legacy_action_stages_test.cpp
    src/native_lz_action_stages.cpp src/legacy_numeric_action_stages.cpp)
target_include_directories(native_legacy_action_stages_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${NLOHMANN_JSON_INCLUDE_DIR}")
target_compile_options(native_legacy_action_stages_test PRIVATE -ffp-contract=off)
add_test(NAME native_legacy_action_stages_test COMMAND native_legacy_action_stages_test)

add_executable(corad_lossless_cpp_cli src/corad_lossless_cpp.cpp)
target_link_libraries(corad_lossless_cpp_cli PRIVATE zlibstatic)
target_compile_options(corad_lossless_cpp_cli PRIVATE -ffp-contract=off)
add_executable(reger_legacy_cpp_cli src/reger_legacy_cpp.cpp)
target_compile_options(reger_legacy_cpp_cli PRIVATE -ffp-contract=off)

# The frozen Blosc2 action is version-pinned, distinct from the vendored codec.
# Point this at a C-Blosc2 3.3.3 distribution (include/ and lib/); no Python is
# linked or executed. The Python wheel can supply these native artifacts.
set(WEB_COMPRESSION_BLOSC2_BINDING_ROOT "" CACHE PATH "C-Blosc2 3.3.3 native distribution")
if(WEB_COMPRESSION_BLOSC2_BINDING_ROOT)
    add_executable(blosc2_binding_cpp_cli src/blosc2_binding_cpp.cpp)
    target_include_directories(blosc2_binding_cpp_cli PRIVATE
        "${WEB_COMPRESSION_BLOSC2_BINDING_ROOT}/include")
    find_library(WEB_COMPRESSION_BLOSC2_BINDING_LIBRARY NAMES blosc2
        PATHS "${WEB_COMPRESSION_BLOSC2_BINDING_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(blosc2_binding_cpp_cli PRIVATE "${WEB_COMPRESSION_BLOSC2_BINDING_LIBRARY}")
endif()

# Shared current RLE kernel. Historical stage entry points keep their old wire
# format; current numeric codecs and current operator profiles use this core.
add_library(optimal_rle_core STATIC src/optimal_rle.cpp)
target_include_directories(optimal_rle_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(unified_rle_numeric_core STATIC src/unified_rle_numeric.cpp)
target_link_libraries(unified_rle_numeric_core PUBLIC optimal_rle_core)
target_compile_options(unified_rle_numeric_core PRIVATE -fno-fast-math -ffp-contract=off)
add_executable(optimal_rle_cli src/optimal_rle_cli.cpp)
target_link_libraries(optimal_rle_cli PRIVATE optimal_rle_core)
add_executable(optimal_rle_test tests/optimal_rle_test.cpp)
target_link_libraries(optimal_rle_test PRIVATE optimal_rle_core)
add_test(NAME optimal_rle_test COMMAND optimal_rle_test)
add_executable(unified_rle_numeric_cli src/unified_rle_numeric_cli.cpp
    src/tsfile_encoding/tsfile_encode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p02_int64.cpp
    src/tsfile_encoding/tsfile_decode_p02.cpp
    src/tsfile_encoding/tsfile_encode_p04.cpp
    src/tsfile_encoding/tsfile_encode_p04_int64.cpp
    src/tsfile_encoding/tsfile_decode_p04.cpp
    src/tsfile_encoding/tsfile_decode_p04_int64.cpp
    src/tsfile_encoding/tsfile_encode_p03.cpp
    src/tsfile_encoding/tsfile_decode_p03.cpp
    src/tsfile_encoding/tsfile_p03_int64_codec.cpp)
target_link_libraries(unified_rle_numeric_cli PRIVATE unified_rle_numeric_core)
target_compile_options(unified_rle_numeric_cli PRIVATE -fno-fast-math -ffp-contract=off)

# p02/p04 are also compiled directly into existing benchmark and stage targets.
# Attach the shared dependency to every such target, including optional tests.
get_property(_rle_link_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_rle_target IN LISTS _rle_link_targets)
    get_target_property(_rle_sources ${_rle_target} SOURCES)
    if("${_rle_sources}" MATCHES "tsfile_encoding/tsfile_(encode|decode)_p0[24]")
        target_link_libraries(${_rle_target} PRIVATE unified_rle_numeric_core)
    endif()
    if("${_rle_sources}" MATCHES "src/general_codec_residual_stages.cpp")
        target_link_libraries(${_rle_target} PRIVATE optimal_rle_core)
    endif()
endforeach()

add_executable(exception_separation_cli src/exception_separation_cli.cpp)
target_include_directories(exception_separation_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_executable(exception_separation_test tests/exception_separation_test.cpp)
target_include_directories(exception_separation_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME exception_separation_test COMMAND exception_separation_test)

# Opt-in IEEE-exact floating front end; integer-codec callbacks bind BOS at use.
add_library(bos_float_exception STATIC src/bos_float_exception.cpp)
target_include_directories(bos_float_exception PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(bos_float_exception PUBLIC cxx_std_20)
add_executable(bos_float_exception_test tests/bos_float_exception_test.cpp)
target_link_libraries(bos_float_exception_test PRIVATE bos_float_exception)
add_test(NAME bos_float_exception_test COMMAND bos_float_exception_test)

add_executable(operator_consolidation_test tests/operator_consolidation_test.cpp
    src/openzl_layout_stages.cpp src/reference_operator_actions.cpp
    src/native_lz_action_stages.cpp src/spdp_exact_stages.cpp)
target_include_directories(operator_consolidation_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(operator_consolidation_test PRIVATE zlibstatic)
target_compile_options(operator_consolidation_test PRIVATE -ffp-contract=off)
add_test(NAME operator_consolidation_test COMMAND operator_consolidation_test)

include(cmake/NNCPProbability.cmake)
