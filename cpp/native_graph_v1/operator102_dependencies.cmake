# Isolated source-accurate dependencies for the 102-entry measurement driver.
# The general residual and frozen 3.3.3 binding use distinct private Blosc DSOs.
include(ExternalProject)
find_package(Threads REQUIRED)

set(_O2_SRC "${CMAKE_CURRENT_LIST_DIR}/../src")
set(_O2_EXT "${CMAKE_CURRENT_LIST_DIR}/../../external/compression")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Keep support codec closure private" FORCE)
set(OPERATOR102_BLOSC333_SOURCE "${_O2_EXT}/c-blosc2-3.3.3" CACHE PATH "Pinned C-Blosc2 3.3.3 source")
set(BROTLI_DISABLE_TESTS ON CACHE BOOL "" FORCE)
set(BROTLI_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory("${_O2_EXT}/brotli" "${CMAKE_CURRENT_BINARY_DIR}/operator102_brotli" EXCLUDE_FROM_ALL)
set(XZ_TOOL_XZ OFF CACHE BOOL "" FORCE)
set(XZ_TOOL_XZDEC OFF CACHE BOOL "" FORCE)
set(XZ_TOOL_LZMADEC OFF CACHE BOOL "" FORCE)
set(XZ_TOOL_LZMAINFO OFF CACHE BOOL "" FORCE)
set(XZ_TOOL_SCRIPTS OFF CACHE BOOL "" FORCE)
set(XZ_DOC OFF CACHE BOOL "" FORCE)
add_subdirectory("${_O2_EXT}/xz" "${CMAKE_CURRENT_BINARY_DIR}/operator102_xz" EXCLUDE_FROM_ALL)

function(operator102_private_blosc variant source target_name library_name)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/operator102_${target_name}_build")
  set(library "${build_dir}/blosc/${library_name}")
  if(variant EQUAL 1)
    set(options -DDEACTIVATE_LZ4=ON -DDEACTIVATE_SNAPPY=ON
      -DDEACTIVATE_ZLIB=ON -DDEACTIVATE_ZSTD=ON)
    set(include_dir "${source}/blosc")
    set(codec_source "${_O2_SRC}/blosc_codec.cpp")
    set(codec_target blosc_static)
  else()
    set(options -DBUILD_LITE=ON -DBLOSC_ENABLE_ZFP=OFF
      -DBLOSC_LZ4_SOURCE_DIR=${_O2_EXT}/lz4)
    set(include_dir "${source}/include")
    set(codec_target blosc2_static)
    if(variant EQUAL 2)
      set(codec_source "${_O2_SRC}/blosc2_codec.cpp")
    else()
      set(codec_source "${_O2_SRC}/blosc2_binding_cpp.cpp")
      set_source_files_properties("${codec_source}" PROPERTIES
        COMPILE_DEFINITIONS "main=operator102_blosc_binding_cli_main")
    endif()
  endif()
  ExternalProject_Add(${target_name}_ep SOURCE_DIR "${source}"
    BINARY_DIR "${build_dir}" DOWNLOAD_COMMAND "" UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DBUILD_STATIC=ON -DBUILD_SHARED=OFF -DBUILD_TESTS=OFF
      -DBUILD_FUZZERS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF
      -DBUILD_PLUGINS=OFF ${options}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target ${codec_target}
    INSTALL_COMMAND "" BUILD_BYPRODUCTS "${library}")
  add_library(${target_name}_static STATIC IMPORTED GLOBAL)
  set_target_properties(${target_name}_static PROPERTIES IMPORTED_LOCATION "${library}")
  add_dependencies(${target_name}_static ${target_name}_ep)
  add_library(${target_name} SHARED operator102_blosc_plugin.cpp "${codec_source}")
  target_compile_definitions(${target_name} PRIVATE OPERATOR102_BLOSC_VARIANT=${variant})
  target_include_directories(${target_name} PRIVATE "${include_dir}" "${_O2_SRC}")
  target_link_libraries(${target_name} PRIVATE ${target_name}_static Threads::Threads ${CMAKE_DL_LIBS})
  target_compile_options(${target_name} PRIVATE -ffp-contract=off)
  set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(exports "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.exports")
    file(WRITE "${exports}" "{ global: operator102_blosc_call; local: *; };\n")
    target_link_options(${target_name} PRIVATE "-Wl,--version-script=${exports}" -Wl,-Bsymbolic)
  endif()
endfunction()

operator102_private_blosc(1 "${_O2_EXT}/c-blosc" operator102_blosc1 libblosc.a)
operator102_private_blosc(2 "${_O2_EXT}/c-blosc2" operator102_blosc2 libblosc2.a)
operator102_private_blosc(3 "${OPERATOR102_BLOSC333_SOURCE}" operator102_blosc333 libblosc2.a)

set(_O2_BZ "${_O2_EXT}/lzbench/bwt/bzip2")
add_library(operator102_support_deps STATIC
  "${_O2_SRC}/general_codec_residual_stages.cpp"
  "${_O2_SRC}/deflate_container_stages.cpp"
  "${_O2_SRC}/xz_container_stage.cpp"
  "${_O2_SRC}/bzip2_operator_ablation.cpp"
  "${_O2_SRC}/reger_legacy_cpp.cpp"
  operator102_blosc_host.cpp
  "${_O2_BZ}/blocksort.c" "${_O2_BZ}/huffman.c" "${_O2_BZ}/crctable.c"
  "${_O2_BZ}/randtable.c" "${_O2_BZ}/compress.c" "${_O2_BZ}/decompress.c"
  "${_O2_BZ}/bzlib.c")
set_source_files_properties("${_O2_SRC}/reger_legacy_cpp.cpp" PROPERTIES
  COMPILE_DEFINITIONS "main=operator102_reger_legacy_cli_main")
target_include_directories(operator102_support_deps PUBLIC "${_O2_SRC}"
  PRIVATE "${_O2_BZ}" "${_O2_EXT}/brotli/c/include" "${_O2_EXT}/xz/src/liblzma/api")
target_link_libraries(operator102_support_deps PUBLIC native_graph_v1_stages
  brotlienc brotlidec brotlicommon liblzma ZLIB::ZLIB libzstd_static ${CMAKE_DL_LIBS})
target_compile_definitions(operator102_support_deps PRIVATE
  OPERATOR102_BLOSC1_PLUGIN="$<TARGET_FILE:operator102_blosc1>"
  OPERATOR102_BLOSC2_PLUGIN="$<TARGET_FILE:operator102_blosc2>"
  OPERATOR102_BLOSC333_PLUGIN="$<TARGET_FILE:operator102_blosc333>")
add_dependencies(operator102_support_deps operator102_blosc1 operator102_blosc2 operator102_blosc333)
target_compile_options(operator102_support_deps PRIVATE -ffp-contract=off)
