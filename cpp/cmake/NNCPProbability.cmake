# Optional genuine NNCP 3.3 predictor. The upstream LibNC binary is Linux x86-64.
option(WEB_COMPRESSION_BUILD_NNCP "Build the original NNCP probability adapter" OFF)
if(WEB_COMPRESSION_BUILD_NNCP)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "Bundled NNCP LibNC requires Linux x86-64")
  endif()
  enable_language(C)
  find_package(ZLIB REQUIRED)
  find_package(Threads REQUIRED)
  get_filename_component(NNCP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../external/prediction_probability/nncp-2024-06-05" ABSOLUTE)
  add_library(nncp_probability_core STATIC
    "${CMAKE_CURRENT_LIST_DIR}/../src/nncp_probability_bridge.c"
    "${NNCP_ROOT}/cmdopt.c" "${NNCP_ROOT}/cp_utils.c"
    "${NNCP_ROOT}/arith.c" "${NNCP_ROOT}/preprocess.c" "${NNCP_ROOT}/cutils.c")
  target_include_directories(nncp_probability_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../include" PRIVATE "${NNCP_ROOT}")
  target_compile_definitions(nncp_probability_core PRIVATE LIBNC_CONFIG_FULL CONFIG_VERSION="2024-06-05")
  target_compile_options(nncp_probability_core PRIVATE -ffp-contract=off)
  target_link_libraries(nncp_probability_core PUBLIC "${NNCP_ROOT}/libnc.so" ZLIB::ZLIB Threads::Threads m)
  add_executable(nncp_probability_cli "${CMAKE_CURRENT_LIST_DIR}/../src/nncp_probability_cli.cpp")
  target_compile_features(nncp_probability_cli PRIVATE cxx_std_20)
  target_link_libraries(nncp_probability_cli PRIVATE nncp_probability_core)
  set_target_properties(nncp_probability_cli PROPERTIES BUILD_RPATH "${NNCP_ROOT}")
endif()
