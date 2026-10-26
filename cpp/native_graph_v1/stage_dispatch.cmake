find_package(ZLIB REQUIRED)
find_path(NATIVE_GRAPH_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include /usr/include REQUIRED)
set(_NG_SRC "${CMAKE_CURRENT_LIST_DIR}/../src")
set(_NG_EXT "${CMAKE_CURRENT_LIST_DIR}/../../external/compression")

# These are the same vendored native dependencies used by cpp/CMakeLists.txt.
# Keeping them in this self-contained target is required for in-process graph
# execution: no CLI/subprocess fallback is allowed in the timed path.
add_library(native_graph_v1_lz4 STATIC
  "${_NG_EXT}/lz4/lib/lz4.c" "${_NG_EXT}/lz4/lib/lz4frame.c"
  "${_NG_EXT}/lz4/lib/lz4hc.c" "${_NG_EXT}/lz4/lib/xxhash.c")
target_include_directories(native_graph_v1_lz4 PUBLIC "${_NG_EXT}/lz4/lib")
set_target_properties(native_graph_v1_lz4 PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(native_graph_v1_snappy STATIC
  "${_NG_EXT}/snappy/snappy.cc" "${_NG_EXT}/snappy/snappy-sinksource.cc"
  "${_NG_EXT}/snappy/snappy-stubs-internal.cc")
include(CheckIncludeFile)
check_include_file("sys/uio.h" _NG_SNAPPY_HAVE_SYS_UIO_H)
set(HAVE_SYS_UIO_H_01 0)
if(_NG_SNAPPY_HAVE_SYS_UIO_H)
  set(HAVE_SYS_UIO_H_01 1)
endif()
# The vendored public header is a configure-time artifact in upstream Snappy.
# Generate it in this target's build tree as well so a pristine build does not
# accidentally depend on external/compression/snappy/build.
set(_NG_SNAPPY_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/native_graph_v1_snappy_generated")
file(MAKE_DIRECTORY "${_NG_SNAPPY_GENERATED}")
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 2)
set(PROJECT_VERSION_PATCH 2)
configure_file("${_NG_EXT}/snappy/snappy-stubs-public.h.in"
  "${_NG_SNAPPY_GENERATED}/snappy-stubs-public.h")
unset(PROJECT_VERSION_MAJOR)
unset(PROJECT_VERSION_MINOR)
unset(PROJECT_VERSION_PATCH)
target_include_directories(native_graph_v1_snappy PUBLIC
  "${_NG_SNAPPY_GENERATED}" "${_NG_EXT}/snappy")
target_compile_definitions(native_graph_v1_snappy PRIVATE
  HAVE_SYS_UIO_H=${HAVE_SYS_UIO_H_01})
set_target_properties(native_graph_v1_snappy PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_NG_FASTPFOR "${_NG_EXT}/FastPFor")
add_library(native_graph_v1_fastpfor STATIC
  "${_NG_FASTPFOR}/src/bitpacking.cpp"
  "${_NG_FASTPFOR}/src/bitpackingaligned.cpp"
  "${_NG_FASTPFOR}/src/bitpackingunaligned.cpp"
  "${_NG_FASTPFOR}/src/horizontalbitpacking.cpp"
  "${_NG_FASTPFOR}/src/simdunalignedbitpacking.cpp"
  "${_NG_FASTPFOR}/src/codecfactory.cpp"
  "${_NG_FASTPFOR}/src/simdbitpacking.cpp"
  "${_NG_FASTPFOR}/src/varintdecode.c"
  "${_NG_FASTPFOR}/src/streamvbyte.c")
target_include_directories(native_graph_v1_fastpfor PUBLIC
  "${_NG_FASTPFOR}/headers")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|^arm|ARM")
  target_include_directories(native_graph_v1_fastpfor PUBLIC
    "${_NG_EXT}/../submodules/simde")
  target_compile_definitions(native_graph_v1_fastpfor PUBLIC
    SIMDE_ENABLE_NATIVE_ALIASES)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(native_graph_v1_fastpfor PRIVATE -msse4.2)
endif()
set_target_properties(native_graph_v1_fastpfor PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

if(NOT TARGET libzstd_static)
  set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "Only the vendored Zstd library is embedded" FORCE)
  set(ZSTD_BUILD_TESTS OFF CACHE BOOL "Project graph tests cover the embedded Zstd path" FORCE)
  add_subdirectory("${_NG_EXT}/zstd/build/cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/native_graph_v1_zstd" EXCLUDE_FROM_ALL)
endif()
set_target_properties(libzstd_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_NG_FPZIP "${_NG_EXT}/fpzip")
add_library(native_graph_v1_fpzip STATIC
  "${_NG_FPZIP}/src/error.cpp" "${_NG_FPZIP}/src/rcdecoder.cpp"
  "${_NG_FPZIP}/src/rcencoder.cpp" "${_NG_FPZIP}/src/rcqsmodel.cpp"
  "${_NG_FPZIP}/src/version.cpp")
target_include_directories(native_graph_v1_fpzip PUBLIC "${_NG_FPZIP}/include" "${_NG_FPZIP}/src")
target_compile_definitions(native_graph_v1_fpzip PUBLIC FPZIP_FP=FPZIP_FP_FAST
  PRIVATE FPZIP_BLOCK_SIZE=4096)
set_target_properties(native_graph_v1_fpzip PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_NG_CMIX "${_NG_EXT}/cmix/src")
file(GLOB _NG_CMIX_SOURCES CONFIGURE_DEPENDS
  "${_NG_CMIX}/contexts/*.cpp" "${_NG_CMIX}/mixer/*.cpp"
  "${_NG_CMIX}/states/*.cpp" "${_NG_CMIX}/models/*.cpp")
list(FILTER _NG_CMIX_SOURCES EXCLUDE REGEX "/(paq8|ppmd)\\.cpp$")
add_library(native_graph_v1_probability STATIC
  "${_NG_SRC}/prediction_probability_model.cpp"
  "${_NG_CMIX}/predictor.cpp" "${_NG_CMIX}/context-manager.cpp"
  ${_NG_CMIX_SOURCES})
target_include_directories(native_graph_v1_probability PUBLIC "${_NG_SRC}")
target_compile_options(native_graph_v1_probability PRIVATE -fno-fast-math -ffp-contract=off)
set_target_properties(native_graph_v1_probability PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(native_graph_v1_probability PRIVATE -fno-gnu-unique)
endif()

# Linux CMIX is built from a private copy of the predictor objects so the few
# direct malloc/calloc/free spellings in upstream fxcmv1.cpp and paq8.cpp can be
# redirected to throwing, DSO-local arena functions.  The ordinary probability
# target remains unchanged for the in-process PPM/DMC/PAQ dispatchers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(native_graph_v1_cmix_probability STATIC
    "${_NG_SRC}/prediction_probability_model.cpp"
    "${_NG_CMIX}/predictor.cpp" "${_NG_CMIX}/context-manager.cpp"
    ${_NG_CMIX_SOURCES})
  target_include_directories(native_graph_v1_cmix_probability PUBLIC "${_NG_SRC}")
  target_compile_options(native_graph_v1_cmix_probability PRIVATE
    -fno-fast-math -ffp-contract=off
    -include "${CMAKE_CURRENT_LIST_DIR}/cmix_allocation_redirect.hpp")
  set_target_properties(native_graph_v1_cmix_probability PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(native_graph_v1_cmix_probability PRIVATE -fno-gnu-unique)
  endif()
endif()

# Full cmix includes PAQ/PPMd implementation state that is not reset by merely
# destroying Predictor.  Keep the complete predictor + arithmetic edge in an
# unloadable hidden-symbol DSO, exactly as for the standalone PAQ provider.
add_library(native_graph_v1_cmix_isolated SHARED
  "${CMAKE_CURRENT_LIST_DIR}/cmix_isolated_plugin.cpp")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(native_graph_v1_cmix_isolated PRIVATE
    native_graph_v1_cmix_probability)
else()
  target_link_libraries(native_graph_v1_cmix_isolated PRIVATE
    native_graph_v1_probability)
endif()
set_target_properties(native_graph_v1_cmix_isolated PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(native_graph_v1_cmix_isolated PRIVATE
  -fno-fast-math -ffp-contract=off)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(native_graph_v1_cmix_isolated PRIVATE -fno-gnu-unique)
  target_link_options(native_graph_v1_cmix_isolated PRIVATE -Wl,-Bsymbolic)
endif()

# PAQ8's upstream implementation has pervasive mutable global and
# function-static state.  Keep it in an unloadable hidden-symbol DSO: each
# codec call dlopen/dlclose cycles the DSO, giving independent frame state
# without a subprocess or a replacement predictor.
add_library(native_graph_v1_paq_isolated SHARED
  "${CMAKE_CURRENT_LIST_DIR}/paq_isolated_plugin.cpp")
set_target_properties(native_graph_v1_paq_isolated PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(native_graph_v1_paq_isolated PRIVATE
  -fno-fast-math -ffp-contract=off)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # PAQ8 keeps its model in function statics.  GNU unique symbols pin DSOs in
  # memory across dlclose, which would silently defeat per-frame state reset.
  target_compile_options(native_graph_v1_paq_isolated PRIVATE -fno-gnu-unique)
endif()

# The bundled upstream NNCP LibNC binary is intentionally platform-specific.
# On the 166 Linux x86-64 host, compile the real model and arithmetic coder
# directly into the dispatcher.  Other platforms omit the capability so the
# action fails closed rather than being mapped to another predictor.
option(NATIVE_GRAPH_ENABLE_NNCP "Build optional NNCP provider" ON)
set(_NG_HAVE_NNCP FALSE)
if(NATIVE_GRAPH_ENABLE_NNCP AND
   CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_NG_NNCP "${_NG_EXT}/../prediction_probability/nncp-2024-06-05")
  if(EXISTS "${_NG_NNCP}/libnc.so")
    find_package(Threads REQUIRED)
    add_library(native_graph_v1_nncp STATIC
      "${_NG_SRC}/nncp_probability_bridge.c"
      "${_NG_NNCP}/cmdopt.c" "${_NG_NNCP}/cp_utils.c"
      "${_NG_NNCP}/arith.c" "${_NG_NNCP}/preprocess.c"
      "${_NG_NNCP}/cutils.c")
    target_include_directories(native_graph_v1_nncp PUBLIC
      "${CMAKE_CURRENT_LIST_DIR}/../include" PRIVATE "${_NG_NNCP}")
    target_compile_definitions(native_graph_v1_nncp PRIVATE
      LIBNC_CONFIG_FULL CONFIG_VERSION="2024-06-05")
    target_compile_options(native_graph_v1_nncp PRIVATE -ffp-contract=off)
    target_link_libraries(native_graph_v1_nncp PUBLIC
      "${_NG_NNCP}/libnc.so" ZLIB::ZLIB Threads::Threads m)
    set_target_properties(native_graph_v1_nncp PROPERTIES
      POSITION_INDEPENDENT_CODE ON BUILD_RPATH "${_NG_NNCP}")
    set(_NG_HAVE_NNCP TRUE)
  endif()
endif()

add_library(native_graph_v1_stages STATIC
  "${CMAKE_CURRENT_LIST_DIR}/stage_dispatch.cpp"
  "${_NG_SRC}/openzl_layout_stages.cpp"
  "${_NG_SRC}/reference_operator_actions.cpp"
  "${_NG_SRC}/native_lz_action_stages.cpp"
  "${_NG_SRC}/legacy_numeric_action_stages.cpp"
  "${_NG_SRC}/general_codec_reference_stages.cpp"
  "${_NG_SRC}/huffman_bytes.cpp" "${_NG_SRC}/ans_rans.cpp" "${_NG_SRC}/arithmetic_bytes.cpp"
  "${_NG_SRC}/pongo_codec.cpp"
  "${_NG_SRC}/elf_float_codec.cpp" "${_NG_SRC}/elf_double_codec.cpp"
  "${_NG_SRC}/camel_exact_stages.cpp"
  "${_NG_SRC}/buff_native_exact_stages.cpp"
  "${_NG_SRC}/fpc_exact_stages.cpp" "${_NG_SRC}/pfpc_exact_stages.cpp"
  "${_NG_SRC}/dbp_exact_stages.cpp" "${_NG_SRC}/ts2diff_exact_stages.cpp"
  "${_NG_SRC}/gorilla_exact_stages.cpp" "${_NG_SRC}/chimp128_exact_stages.cpp"
  "${_NG_SRC}/tsfile_encoding/tsfile_p03_int64_codec.cpp"
  "${_NG_SRC}/spdp_exact_stages.cpp" "${_NG_SRC}/optimal_rle.cpp"
  "${_NG_SRC}/bos_exact_stages.cpp" "${_NG_SRC}/svb_exact_stages.cpp"
  "${_NG_SRC}/simple8b_exact_stages.cpp" "${_NG_SRC}/fastpfor_exact_stages.cpp"
  "${_NG_SRC}/p04_exact_stages.cpp" "${_NG_SRC}/rle_exact_stages.cpp"
  "${_NG_SRC}/learning_codec_exact_stages.cpp"
  "${_NG_SRC}/alp_float_codec.cpp" "${_NG_SRC}/reger_codec.cpp"
  "${_NG_SRC}/leco_operator_ablation.cpp" "${_NG_SRC}/leco_double_codec.cpp"
  "${_NG_SRC}/buff_kernel/buff_split_double_codec.cpp"
  "${_NG_SRC}/buff_kernel/buff_bit_pack.cpp"
  "${_NG_SRC}/buff_kernel/buff_precision_bound.cpp"
  "${_NG_SRC}/lz4_exact_stages.cpp" "${_NG_SRC}/snappy_exact_stages.cpp"
  "${_NG_SRC}/fpzip_exact_stages.cpp" "${_NG_SRC}/corad_lossless_cpp.cpp"
  "${_NG_SRC}/int64_column_codec.cpp" "${_NG_SRC}/java_fastpfor_codec.cpp"
  "${_NG_SRC}/tsfile_encoding/tsfile_encode_p02.cpp"
  "${_NG_SRC}/tsfile_encoding/tsfile_encode_p02_int64.cpp"
  "${_NG_SRC}/tsfile_encoding/tsfile_decode_p02.cpp"
  "${_NG_SRC}/tsfile_encoding/tsfile_encode_p04_int64.cpp"
  "${_NG_SRC}/tsfile_encoding/tsfile_decode_p04_int64.cpp"
  "${_NG_SRC}/unified_rle_numeric.cpp")
# corad_lossless_cpp.cpp exposes real stage functions but also contains its CLI.
# Rename only that translation unit's entry point so it can be linked in-process.
set_source_files_properties("${_NG_SRC}/corad_lossless_cpp.cpp"
  PROPERTIES COMPILE_DEFINITIONS "main=corad_lossless_cli_main")
set_target_properties(native_graph_v1_stages PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(native_graph_v1_stages PUBLIC "${CMAKE_CURRENT_LIST_DIR}" "${_NG_SRC}"
  PRIVATE "${NATIVE_GRAPH_JSON_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/../third_party"
  "${_NG_FASTPFOR}/headers" "${_NG_EXT}/../submodules/simde"
  "${_NG_EXT}/lz4/lib" "${_NG_EXT}/snappy"
  "${_NG_FPZIP}/include" "${_NG_FPZIP}/src")
target_link_libraries(native_graph_v1_stages PUBLIC ZLIB::ZLIB native_graph_v1_lz4
  native_graph_v1_snappy native_graph_v1_fpzip native_graph_v1_probability
  native_graph_v1_fastpfor native_graph_v1_chimp_history_xor_v2
  libzstd_static ${CMAKE_DL_LIBS})
target_compile_definitions(native_graph_v1_stages PRIVATE
  NATIVE_GRAPH_V1_PAQ_PLUGIN_PATH="$<TARGET_FILE:native_graph_v1_paq_isolated>"
  NATIVE_GRAPH_V1_CMIX_PLUGIN_PATH="$<TARGET_FILE:native_graph_v1_cmix_isolated>")
add_dependencies(native_graph_v1_stages
  native_graph_v1_paq_isolated native_graph_v1_cmix_isolated)
if(_NG_HAVE_NNCP)
  target_compile_definitions(native_graph_v1_stages PRIVATE
    NATIVE_GRAPH_V1_HAVE_NNCP=1)
  target_link_libraries(native_graph_v1_stages PUBLIC native_graph_v1_nncp)
endif()
target_compile_options(native_graph_v1_stages PRIVATE -ffp-contract=off)
