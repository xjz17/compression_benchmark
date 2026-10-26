# Direct same-label C/C++ backends for the general-B migration batch.  This
# file is intentionally included by the caller so the running/frozen v5b tree
# is not changed while the new target is qualified.

set(_B86_GB_DIR "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(_B86_CPP_DIR "${_B86_GB_DIR}/.." ABSOLUTE)
get_filename_component(_B86_ROOT "${_B86_CPP_DIR}/.." ABSOLUTE)
set(_B86_LZ "${_B86_ROOT}/external/compression/lzbench")
set(_B86_DENSITY "${_B86_ROOT}/external/operator_ablation_sources/density_c_0_14_2")

set(_B86_DENSITY_SOURCES
  "${_B86_DENSITY}/src/algorithms/algorithms.c"
  "${_B86_DENSITY}/src/algorithms/chameleon/core/chameleon_decode.c"
  "${_B86_DENSITY}/src/algorithms/chameleon/core/chameleon_encode.c"
  "${_B86_DENSITY}/src/algorithms/cheetah/core/cheetah_decode.c"
  "${_B86_DENSITY}/src/algorithms/cheetah/core/cheetah_encode.c"
  "${_B86_DENSITY}/src/algorithms/dictionaries.c"
  "${_B86_DENSITY}/src/algorithms/lion/core/lion_decode.c"
  "${_B86_DENSITY}/src/algorithms/lion/core/lion_encode.c"
  "${_B86_DENSITY}/src/algorithms/lion/forms/lion_form_model.c"
  "${_B86_DENSITY}/src/buffers/buffer.c"
  "${_B86_DENSITY}/src/globals.c"
  "${_B86_DENSITY}/src/structure/header.c")

set(_B86_KANZI_SOURCES
  "${_B86_LZ}/misc/kanzi-cpp/src/io/CompressedOutputStream.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/io/CompressedInputStream.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/EntropyUtils.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/ExpGolombEncoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/FPAQEncoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/ANSRangeEncoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/ANSRangeDecoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/BinaryEntropyDecoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/BinaryEntropyEncoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/ExpGolombDecoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/HuffmanEncoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/FPAQDecoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/TPAQPredictor.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/CMPredictor.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/HuffmanCommon.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/RangeDecoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/RangeEncoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/entropy/HuffmanDecoder.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/bitstream/DefaultInputBitStream.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/bitstream/DebugOutputBitStream.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/bitstream/DebugInputBitStream.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/bitstream/DefaultOutputBitStream.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/Event.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/Global.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/AliasCodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/BWT.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/RLT.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/TextCodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/EXECodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/SBRT.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/ROLZCodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/LZCodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/SRT.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/DivSufSort.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/BWTBlockCodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/BWTS.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/UTFCodec.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/ZRLT.cpp"
  "${_B86_LZ}/misc/kanzi-cpp/src/transform/FSDCodec.cpp")

set(_B86_LIZARD_SOURCES
  "${_B86_LZ}/lz/lizard/lizard_compress.c"
  "${_B86_LZ}/lz/lizard/lizard_decompress.c"
  "${_B86_LZ}/lz/lizard/entropy/huf_compress.c"
  "${_B86_LZ}/lz/lizard/entropy/huf_decompress.c"
  "${_B86_LZ}/lz/lizard/entropy/entropy_common.c"
  "${_B86_LZ}/lz/lizard/entropy/fse_compress.c"
  "${_B86_LZ}/lz/lizard/entropy/fse_decompress.c"
  "${_B86_LZ}/lz/lizard/entropy/hist.c"
  "${_B86_LZ}/lz/lizard/xxhash/xxhash.c")

set(_B86_LZHAM_SOURCES
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_assert.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_checksum.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_huffman_codes.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_lzdecomp.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_lzdecompbase.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_mem.cpp"
  "${_B86_GB_DIR}/upstream_general_b_lzham_platform.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_prefix_coding.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_symbol_codec.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_timer.cpp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp/lzham_vector.cpp"
  "${_B86_LZ}/lz/lzham/lzhamlib/lzham_lib.cpp"
  "${_B86_LZ}/lz/lzham/lzhamcomp/lzham_lzbase.cpp"
  "${_B86_LZ}/lz/lzham/lzhamcomp/lzham_lzcomp.cpp"
  "${_B86_LZ}/lz/lzham/lzhamcomp/lzham_lzcomp_internal.cpp"
  "${_B86_LZ}/lz/lzham/lzhamcomp/lzham_lzcomp_state.cpp"
  "${_B86_LZ}/lz/lzham/lzhamcomp/lzham_match_accel.cpp"
  "${_B86_LZ}/lz/lzham/lzhamcomp/lzham_pthreads_threading.cpp")
if(APPLE)
  # This old upstream release used libc++'s removed internal spelling
  # std::__is_pod<T>::__value.  Restore the trait only in its isolated
  # translation units; the vendored source and platform selection stay intact.
  set_source_files_properties(${_B86_LZHAM_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-include;${_B86_GB_DIR}/upstream_general_b_lzham_compat.hpp")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    # LZHAM 1.0 treats every macOS target as x86 and emits the x86 `pause`
    # instruction.  Its documented ANSI port selects portable atomics and is
    # sufficient here because this adapter fixes the helper-thread count at 0.
    set_source_files_properties(${_B86_LZHAM_SOURCES} PROPERTIES
      COMPILE_DEFINITIONS "LZHAM_ANSI_CPLUSPLUS=1")
  endif()
endif()

set(_B86_ZLIBNG_SOURCES
  "${_B86_LZ}/lz/zlib-ng/adler32.c" "${_B86_LZ}/lz/zlib-ng/crc32.c"
  "${_B86_LZ}/lz/zlib-ng/deflate_medium.c" "${_B86_LZ}/lz/zlib-ng/deflate_stored.c"
  "${_B86_LZ}/lz/zlib-ng/inftrees.c" "${_B86_LZ}/lz/zlib-ng/uncompr.c"
  "${_B86_LZ}/lz/zlib-ng/compress.c" "${_B86_LZ}/lz/zlib-ng/deflate.c"
  "${_B86_LZ}/lz/zlib-ng/deflate_quick.c" "${_B86_LZ}/lz/zlib-ng/functable.c"
  "${_B86_LZ}/lz/zlib-ng/insert_string.c" "${_B86_LZ}/lz/zlib-ng/zutil.c"
  "${_B86_LZ}/lz/zlib-ng/cpu_features.c" "${_B86_LZ}/lz/zlib-ng/deflate_fast.c"
  "${_B86_LZ}/lz/zlib-ng/deflate_rle.c" "${_B86_LZ}/lz/zlib-ng/infback.c"
  "${_B86_LZ}/lz/zlib-ng/insert_string_roll.c" "${_B86_LZ}/lz/zlib-ng/crc32_braid_comb.c"
  "${_B86_LZ}/lz/zlib-ng/deflate_huff.c" "${_B86_LZ}/lz/zlib-ng/deflate_slow.c"
  "${_B86_LZ}/lz/zlib-ng/inflate.c" "${_B86_LZ}/lz/zlib-ng/trees.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/adler32_c.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/chunkset_c.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/crc32_braid_c.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/slide_hash_c.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/adler32_fold_c.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/compare256_c.c"
  "${_B86_LZ}/lz/zlib-ng/arch/generic/crc32_fold_c.c")

set(_B86_ZXC_COMMON
  "${_B86_LZ}/lz/zxc/src/lib/zxc_common.c"
  "${_B86_LZ}/lz/zxc/src/lib/zxc_driver.c"
  "${_B86_LZ}/lz/zxc/src/lib/zxc_dispatch.c"
  "${_B86_LZ}/lz/zxc/src/lib/zxc_pstream.c"
  "${_B86_LZ}/lz/zxc/src/lib/zxc_seekable.c")
set(_B86_ZXC_DEFAULT
  "${_B86_LZ}/lz/zxc/src/lib/zxc_compress.c"
  "${_B86_LZ}/lz/zxc/src/lib/zxc_decompress.c"
  "${_B86_LZ}/lz/zxc/src/lib/zxc_huffman.c")

add_library(native_baseline86_general_b STATIC
  "${_B86_GB_DIR}/upstream_general_b.cpp"
  ${_B86_DENSITY_SOURCES} ${_B86_KANZI_SOURCES} ${_B86_LIZARD_SOURCES}
  ${_B86_LZHAM_SOURCES}
  "${_B86_LZ}/lz/lzlib/lzlib.c"
  "${_B86_LZ}/misc/7-zip/Ppmd8.c"
  "${_B86_LZ}/misc/7-zip/Ppmd8Dec.c"
  "${_B86_LZ}/misc/7-zip/Ppmd8Enc.c"
  "${_B86_LZ}/lz/tornado/tor_test.cpp"
  ${_B86_ZLIBNG_SOURCES}
  "${_B86_LZ}/lz/libzling/libzling.cpp"
  "${_B86_LZ}/lz/libzling/libzling_huffman.cpp"
  "${_B86_LZ}/lz/libzling/libzling_lz.cpp"
  "${_B86_LZ}/lz/libzling/libzling_utils.cpp"
  ${_B86_ZXC_COMMON} ${_B86_ZXC_DEFAULT})

target_include_directories(native_baseline86_general_b PUBLIC "${_B86_GB_DIR}" PRIVATE
  "${_B86_LZ}" "${_B86_LZ}/lz/lizard" "${_B86_LZ}/lz/lizard/entropy"
  "${_B86_LZ}/lz/lzham/include" "${_B86_LZ}/lz/lzham/lzhamcomp"
  "${_B86_LZ}/lz/lzham/lzhamdecomp" "${_B86_LZ}/lz/zlib-ng"
  "${_B86_LZ}/lz/zlib-ng/arch/generic" "${_B86_LZ}/lz/zxc/include"
  "${_B86_LZ}/lz/zxc/src/lib" "${_B86_LZ}/lz/zxc/src/lib/vendors"
  "${_B86_DENSITY}/src")

target_compile_definitions(native_baseline86_general_b PRIVATE
  # Kanzi's stream classes submit even a jobs=1 block through std::async when
  # concurrency support is compiled in.  Disable that path so the benchmark's
  # jobs=1 profile executes the block synchronously in the calling thread.
  CONCURRENCY_DISABLED
  THREAD_MODEL_POSIX ZXC_STATIC_DEFINE ZXC_DISABLE_SIMD XXH_NAMESPACE=B86GB_)
set_source_files_properties(${_B86_ZXC_DEFAULT} PROPERTIES
  COMPILE_DEFINITIONS "ZXC_STATIC_DEFINE;ZXC_DISABLE_SIMD;ZXC_FUNCTION_SUFFIX=_default")

# Prefix old Lizard entropy detail symbols so this target can coexist with the
# newer Zstd/FSE implementation already linked by native_graph_v1.
target_compile_definitions(native_baseline86_general_b PRIVATE
  FSE_buildCTable=B86GB_FSE_buildCTable FSE_buildCTable_raw=B86GB_FSE_buildCTable_raw
  FSE_compress=B86GB_FSE_compress FSE_compress2=B86GB_FSE_compress2
  FSE_compress_wksp=B86GB_FSE_compress_wksp FSE_createCTable=B86GB_FSE_createCTable
  FSE_freeCTable=B86GB_FSE_freeCTable FSE_buildDTable=B86GB_FSE_buildDTable
  FSE_buildDTable_raw=B86GB_FSE_buildDTable_raw FSE_buildDTable_rle=B86GB_FSE_buildDTable_rle
  FSE_createDTable=B86GB_FSE_createDTable FSE_decompress=B86GB_FSE_decompress
  FSE_decompress_usingDTable=B86GB_FSE_decompress_usingDTable
  FSE_decompress_wksp=B86GB_FSE_decompress_wksp FSE_freeDTable=B86GB_FSE_freeDTable
  HUF_buildCTable=B86GB_HUF_buildCTable HUF_compress1X=B86GB_HUF_compress1X
  HUF_compress1X_wksp=B86GB_HUF_compress1X_wksp HUF_compress2=B86GB_HUF_compress2
  HUF_compress4X_wksp=B86GB_HUF_compress4X_wksp HUF_getNbBits=B86GB_HUF_getNbBits
  HUF_writeCTable=B86GB_HUF_writeCTable HUF_decompress1X1=B86GB_HUF_decompress1X1
  HUF_decompress1X1_DCtx=B86GB_HUF_decompress1X1_DCtx
  HUF_decompress1X1_DCtx_wksp_bmi2=B86GB_HUF_decompress1X1_DCtx_wksp_bmi2
  HUF_decompress1X1_usingDTable=B86GB_HUF_decompress1X1_usingDTable
  HUF_decompress1X2=B86GB_HUF_decompress1X2
  HUF_decompress1X2_DCtx=B86GB_HUF_decompress1X2_DCtx
  HUF_decompress1X2_usingDTable=B86GB_HUF_decompress1X2_usingDTable
  HUF_decompress1X_DCtx=B86GB_HUF_decompress1X_DCtx
  HUF_decompress1X_usingDTable_bmi2=B86GB_HUF_decompress1X_usingDTable_bmi2
  HUF_decompress4X1=B86GB_HUF_decompress4X1
  HUF_decompress4X1_DCtx=B86GB_HUF_decompress4X1_DCtx
  HUF_decompress4X1_DCtx_wksp=B86GB_HUF_decompress4X1_DCtx_wksp
  HUF_decompress4X1_usingDTable=B86GB_HUF_decompress4X1_usingDTable
  HUF_decompress4X2=B86GB_HUF_decompress4X2
  HUF_decompress4X2_DCtx=B86GB_HUF_decompress4X2_DCtx
  HUF_decompress4X2_DCtx_wksp=B86GB_HUF_decompress4X2_DCtx_wksp
  HUF_decompress4X2_usingDTable=B86GB_HUF_decompress4X2_usingDTable
  HUF_decompress4X_DCtx=B86GB_HUF_decompress4X_DCtx
  HUF_decompress4X_hufOnly=B86GB_HUF_decompress4X_hufOnly
  HUF_decompress4X_hufOnly_wksp_bmi2=B86GB_HUF_decompress4X_hufOnly_wksp_bmi2
  HUF_decompress4X_usingDTable_bmi2=B86GB_HUF_decompress4X_usingDTable_bmi2
  HUF_readDTableX1=B86GB_HUF_readDTableX1 HUF_readDTableX2=B86GB_HUF_readDTableX2
  NGEF_NCountWriteBound=B86GB_NGEF_NCountWriteBound
  NGEF_buildCTable_rle=B86GB_NGEF_buildCTable_rle
  NGEF_buildCTable_wksp=B86GB_NGEF_buildCTable_wksp
  NGEF_compressBound=B86GB_NGEF_compressBound
  NGEF_compress_usingCTable=B86GB_NGEF_compress_usingCTable
  NGEF_getErrorName=B86GB_NGEF_getErrorName NGEF_isError=B86GB_NGEF_isError
  NGEF_normalizeCount=B86GB_NGEF_normalizeCount
  NGEF_optimalTableLog=B86GB_NGEF_optimalTableLog
  NGEF_optimalTableLog_internal=B86GB_NGEF_optimalTableLog_internal
  NGEF_readNCount=B86GB_NGEF_readNCount NGEF_versionNumber=B86GB_NGEF_versionNumber
  NGEF_writeNCount=B86GB_NGEF_writeNCount
  NGEH_buildCTable_wksp=B86GB_NGEH_buildCTable_wksp
  NGEH_compress=B86GB_NGEH_compress
  NGEH_compress1X_repeat=B86GB_NGEH_compress1X_repeat
  NGEH_compress1X_usingCTable=B86GB_NGEH_compress1X_usingCTable
  NGEH_compress4X_repeat=B86GB_NGEH_compress4X_repeat
  NGEH_compress4X_usingCTable=B86GB_NGEH_compress4X_usingCTable
  NGEH_compressBound=B86GB_NGEH_compressBound NGEH_decompress=B86GB_NGEH_decompress
  NGEH_decompress1X1_DCtx_wksp=B86GB_NGEH_decompress1X1_DCtx_wksp
  NGEH_decompress1X2_DCtx_wksp=B86GB_NGEH_decompress1X2_DCtx_wksp
  NGEH_decompress1X_DCtx_wksp=B86GB_NGEH_decompress1X_DCtx_wksp
  NGEH_decompress1X_usingDTable=B86GB_NGEH_decompress1X_usingDTable
  NGEH_decompress4X_hufOnly_wksp=B86GB_NGEH_decompress4X_hufOnly_wksp
  NGEH_decompress4X_usingDTable=B86GB_NGEH_decompress4X_usingDTable
  NGEH_estimateCompressedSize=B86GB_NGEH_estimateCompressedSize
  NGEH_getErrorName=B86GB_NGEH_getErrorName NGEH_isError=B86GB_NGEH_isError
  NGEH_optimalTableLog=B86GB_NGEH_optimalTableLog
  NGEH_readCTable=B86GB_NGEH_readCTable
  NGEH_readDTableX1_wksp=B86GB_NGEH_readDTableX1_wksp
  NGEH_readDTableX2_wksp=B86GB_NGEH_readDTableX2_wksp
  NGEH_readStats=B86GB_NGEH_readStats NGEH_selectDecoder=B86GB_NGEH_selectDecoder
  NGEH_validateCTable=B86GB_NGEH_validateCTable
  NGES_count=B86GB_NGES_count NGES_countFast=B86GB_NGES_countFast
  NGES_countFast_wksp=B86GB_NGES_countFast_wksp NGES_count_simple=B86GB_NGES_count_simple
  NGES_count_wksp=B86GB_NGES_count_wksp NGES_isError=B86GB_NGES_isError)

find_package(Threads REQUIRED)
target_link_libraries(native_baseline86_general_b PUBLIC Threads::Threads)
set_target_properties(native_baseline86_general_b PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(native_baseline86_general_b PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-fno-fast-math;-ffp-contract=off>)
endif()

if(BUILD_TESTING)
  add_executable(native_baseline86_general_b_test "${_B86_GB_DIR}/upstream_general_b_test.cpp")
  target_link_libraries(native_baseline86_general_b_test PRIVATE native_baseline86_general_b)
  add_test(NAME native_baseline86_general_b_roundtrip COMMAND native_baseline86_general_b_test)
endif()
