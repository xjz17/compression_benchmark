# Direct upstream adapters for ten general-purpose baseline labels.  Include
# this fragment and call native_baseline86_add_upstream_general_a(<target>).
# The target is static and can be linked into the complete baseline86 process;
# no executable launch or IPC is introduced.

set(_native_baseline86_general_a_dir "${CMAKE_CURRENT_LIST_DIR}")

function(native_baseline86_add_upstream_general_a target_name)
  set(_lzbench "${_native_baseline86_general_a_dir}/../../external/compression/lzbench")
  set(_general_a_sources "${_native_baseline86_general_a_dir}/upstream_general_a.cpp")
  set(_general_a_definitions)
  set(_general_a_includes "${_native_baseline86_general_a_dir}")

  list(APPEND _general_a_sources
    "${_lzbench}/lz/gipfeli/decompress.cc"
    "${_lzbench}/lz/gipfeli/entropy.cc"
    "${_lzbench}/lz/gipfeli/entropy_code_builder.cc"
    "${_lzbench}/lz/gipfeli/gipfeli-internal.cc"
    "${_lzbench}/lz/gipfeli/lz77.cc")
  list(APPEND _general_a_includes "${_lzbench}/lz/gipfeli")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_GIPFELI=1)

  list(APPEND _general_a_sources
    "${_lzbench}/misc/glza/GLZAcomp.c"
    "${_lzbench}/misc/glza/GLZAformat.c"
    "${_lzbench}/misc/glza/GLZAcompress.c"
    "${_lzbench}/misc/glza/GLZAencode.c"
    "${_lzbench}/misc/glza/GLZAdecode.c"
    "${_lzbench}/misc/glza/GLZAmodel.c")
  list(APPEND _general_a_includes "${_lzbench}/misc/glza")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_GLZA=1)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    option(NATIVE_BASELINE86_GLZA_SERIAL_SCHEDULER
      "Run GLZA's internal worker graph on one cooperative OS thread" ON)
    if(NATIVE_BASELINE86_GLZA_SERIAL_SCHEDULER)
      list(APPEND _general_a_sources
        "${_native_baseline86_general_a_dir}/glza_serial_scheduler.c")
      list(APPEND _general_a_definitions
        BASELINE86_GENERAL_A_GLZA_SERIAL_SCHEDULER=1)
      set_source_files_properties(
        "${_lzbench}/misc/glza/GLZAcompress.c"
        PROPERTIES COMPILE_OPTIONS
          "-include;${_native_baseline86_general_a_dir}/glza_serial_pthread_shim.h")
    endif()
  endif()

  list(APPEND _general_a_sources
    "${_lzbench}/lz/lzo/lzo1.c"
    "${_lzbench}/lz/lzo/lzo2a_9x.c"
    "${_lzbench}/lz/lzo/lzo2a_d1.c"
    "${_lzbench}/lz/lzo/lzo2a_d2.c"
    "${_lzbench}/lz/lzo/lzo_crc.c"
    "${_lzbench}/lz/lzo/lzo_init.c"
    "${_lzbench}/lz/lzo/lzo_ptr.c"
    "${_lzbench}/lz/lzo/lzo_str.c"
    "${_lzbench}/lz/lzo/lzo_util.c")
  list(APPEND _general_a_includes "${_lzbench}/lz")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_LZO=1)

  list(APPEND _general_a_sources
    "${_lzbench}/lz/ucl/alloc.c"
    "${_lzbench}/lz/ucl/n2b_99.c"
    "${_lzbench}/lz/ucl/n2b_ds.c"
    "${_lzbench}/lz/ucl/ucl_crc.c"
    "${_lzbench}/lz/ucl/ucl_init.c"
    "${_lzbench}/lz/ucl/ucl_ptr.c"
    "${_lzbench}/lz/ucl/ucl_str.c"
    "${_lzbench}/lz/ucl/ucl_util.c")
  list(APPEND _general_a_includes "${_lzbench}/lz/ucl")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_NRV=1)

  set(_bsc "${_lzbench}/bwt/libbsc/libbsc")
  list(APPEND _general_a_sources
    "${_bsc}/bwt/libsais/libsais.c"
    "${_bsc}/adler32/adler32.cpp"
    "${_bsc}/bwt/bwt.cpp"
    "${_bsc}/coder/coder.cpp"
    "${_bsc}/coder/qlfc/qlfc.cpp"
    "${_bsc}/coder/qlfc/qlfc_model.cpp"
    "${_bsc}/filters/detectors.cpp"
    "${_bsc}/filters/preprocessing.cpp"
    "${_bsc}/libbsc/libbsc.cpp"
    "${_bsc}/lzp/lzp.cpp"
    "${_bsc}/platform/platform.cpp"
    "${_bsc}/st/st.cpp")
  list(APPEND _general_a_includes "${_bsc}")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_BSC=1)

  list(APPEND _general_a_sources "${_lzbench}/lz/crush/crush.cpp")
  list(APPEND _general_a_includes "${_lzbench}/lz/crush")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_CRUSH=1)

  list(APPEND _general_a_sources
    "${_lzbench}/lz/libcsc/csc_analyzer.cpp"
    "${_lzbench}/lz/libcsc/csc_coder.cpp"
    "${_lzbench}/lz/libcsc/csc_dec.cpp"
    "${_lzbench}/lz/libcsc/csc_enc.cpp"
    "${_lzbench}/lz/libcsc/csc_encoder_main.cpp"
    "${_lzbench}/lz/libcsc/csc_filters.cpp"
    "${_lzbench}/lz/libcsc/csc_lz.cpp"
    "${_lzbench}/lz/libcsc/csc_memio.cpp"
    "${_lzbench}/lz/libcsc/csc_mf.cpp"
    "${_lzbench}/lz/libcsc/csc_model.cpp"
    "${_lzbench}/lz/libcsc/csc_profiler.cpp"
    "${_lzbench}/lz/libcsc/csc_default_alloc.cpp")
  list(APPEND _general_a_includes "${_lzbench}/lz/libcsc")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_CSC=1)

  file(GLOB _fast_lzma2_sources CONFIGURE_DEPENDS
       "${_lzbench}/lz/fast-lzma2/*.c")
  list(APPEND _general_a_sources ${_fast_lzma2_sources})
  list(APPEND _general_a_includes "${_lzbench}/lz/fast-lzma2")
  list(APPEND _general_a_definitions BASELINE86_GENERAL_A_FASTLZMA2=1
                                      FL2_SINGLETHREAD=1 NO_XXHASH=1)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    list(APPEND _general_a_sources
      "${_lzbench}/lz/lzsse/lzsse8/lzsse8.cpp")
    list(APPEND _general_a_includes "${_lzbench}/lz/lzsse/lzsse8")
    list(APPEND _general_a_definitions BASELINE86_GENERAL_A_LZSSE=1)
    set_source_files_properties(
      "${_lzbench}/lz/lzsse/lzsse8/lzsse8.cpp"
      PROPERTIES COMPILE_OPTIONS "-msse4.1")
  endif()

  add_library(${target_name} STATIC ${_general_a_sources})
  target_compile_features(${target_name} PUBLIC cxx_std_20)
  target_include_directories(${target_name} PUBLIC ${_general_a_includes})
  target_compile_definitions(${target_name} PRIVATE ${_general_a_definitions})
  set_target_properties(${target_name} PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON)
endfunction()
