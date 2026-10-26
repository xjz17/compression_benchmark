cmake_minimum_required(VERSION 3.18)
project(unified_compression_v1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# A separate build tree; never mutate the binaries of an existing experiment.
set(NATIVE_GRAPH_ENABLE_NNCP OFF CACHE BOOL "NNCP remains disabled" FORCE)
option(UNIFIED_COMPRESSION_ENABLE_OPERATOR_SUPPORT_DEPS
  "Link the existing residual-operator codec dependencies" ON)
option(UNIFIED_COMPRESSION_FETCH_MISSING_DEPENDENCIES
  "Download the hash-pinned Blosc 3.3.3 source if no local source is present" ON)
set(OPERATOR_HARNESS_V2_ENABLE_SUPPORT_DEPS
  ${UNIFIED_COMPRESSION_ENABLE_OPERATOR_SUPPORT_DEPS} CACHE BOOL "" FORCE)
# Reuse the verified pinned source retained by the prior harness deployment
# when the historical external/ placeholder is empty. Only this build cache
# changes; existing harness binaries and source checkouts remain untouched.
set(_ucc_pinned_blosc "${CMAKE_CURRENT_SOURCE_DIR}/../../analysis_outputs/operator102_current135_20261017_v1/deployment/pinned/c-blosc2-3.3.3")
if(UNIFIED_COMPRESSION_ENABLE_OPERATOR_SUPPORT_DEPS AND
   (NOT DEFINED OPERATOR102_BLOSC333_SOURCE OR
    NOT EXISTS "${OPERATOR102_BLOSC333_SOURCE}/CMakeLists.txt") AND
   EXISTS "${_ucc_pinned_blosc}/CMakeLists.txt")
  set(OPERATOR102_BLOSC333_SOURCE "${_ucc_pinned_blosc}" CACHE PATH
    "Pinned C-Blosc2 3.3.3 source for this isolated build" FORCE)
endif()
if(UNIFIED_COMPRESSION_ENABLE_OPERATOR_SUPPORT_DEPS AND
   (NOT DEFINED OPERATOR102_BLOSC333_SOURCE OR
    NOT EXISTS "${OPERATOR102_BLOSC333_SOURCE}/CMakeLists.txt") AND
   NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../external/compression/c-blosc2-3.3.3/CMakeLists.txt")
  if(NOT UNIFIED_COMPRESSION_FETCH_MISSING_DEPENDENCIES)
    message(FATAL_ERROR "Provide -DOPERATOR102_BLOSC333_SOURCE=/path/to/c-blosc2-3.3.3 or enable pinned dependency download")
  endif()
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  FetchContent_Declare(unified_blosc333
    URL https://codeload.github.com/Blosc/c-blosc2/tar.gz/refs/tags/v3.3.3
    URL_HASH SHA256=3ab395116eae3ce0b7488e994224f85cbc858a7a71663af1f938cc5997a7dddd
    SOURCE_SUBDIR __ucc_download_only__)
  # Download only; the existing isolated ExternalProject owns compilation.
  FetchContent_MakeAvailable(unified_blosc333)
  set(OPERATOR102_BLOSC333_SOURCE "${unified_blosc333_SOURCE_DIR}" CACHE PATH
    "Hash-pinned C-Blosc2 3.3.3 source for this isolated build" FORCE)
endif()
add_subdirectory(../operator_harness_v2 operator_harness EXCLUDE_FROM_ALL)
add_subdirectory(../native_baseline86_v1 baseline86 EXCLUDE_FROM_ALL)
if(TARGET native_graph_v1_nncp)
  message(FATAL_ERROR "Unified compression must not build NNCP")
endif()

find_package(OpenSSL REQUIRED)
find_path(UNIFIED_COMPRESSION_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include /usr/include REQUIRED)
add_library(unified_compression_v1 STATIC api.cpp algorithm_adapter.cpp operator_adapter.cpp
  selection.cpp recommender.cpp ../operator_recommend135_v1/model.cpp
  ../harness_combo_pilot_v1/runtime.cpp
  bos_bfe_v2/ucc_bos_bfe_bridge.cpp
  bos_bfe_v2/bos_float_exception.cpp
  bos_bfe_v2/bos2_refined.cpp
  bos_bfe_v2/dsc3_refined.cpp
  bos_bfe_v2/decimal_scaling_v2_legacy.cpp)
target_include_directories(unified_compression_v1 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${UNIFIED_COMPRESSION_JSON_INCLUDE_DIR})
target_link_libraries(unified_compression_v1 PUBLIC
  operator_harness_v2 native_baseline86_v1 OpenSSL::Crypto)
target_compile_definitions(unified_compression_v1 PRIVATE
  UNIFIED_COMPRESSION_OPERATOR_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/../../benchmark/reference/actions/operator_harness_tree_catalog_v3.json")
target_compile_options(unified_compression_v1 PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
if(UNIFIED_COMPRESSION_ENABLE_OPERATOR_SUPPORT_DEPS)
  target_compile_definitions(unified_compression_v1 PRIVATE UNIFIED_COMPRESSION_HAS_OPERATOR_SUPPORT_DEPS=1)
endif()

add_executable(unified_compression_cli cli.cpp)
target_link_libraries(unified_compression_cli PRIVATE unified_compression_v1)
add_executable(unified_compression_api_test api_test.cpp)
target_link_libraries(unified_compression_api_test PRIVATE unified_compression_v1)
add_test(NAME unified_compression_api COMMAND unified_compression_api_test)
add_executable(unified_tree_selection_test tree_selection_test.cpp)
target_link_libraries(unified_tree_selection_test PRIVATE unified_compression_v1)
add_test(NAME unified_tree_selection COMMAND unified_tree_selection_test)
add_executable(unified_recommender_test recommender_test.cpp)
target_link_libraries(unified_recommender_test PRIVATE unified_compression_v1)
set(_ucc_frozen_model "${CMAKE_CURRENT_SOURCE_DIR}/../../analysis_outputs/operator_prepared135_192_20261020_v1/remote_evidence/models/MODEL_confirm.json")
set(_ucc_frozen_split "${CMAKE_CURRENT_SOURCE_DIR}/../../analysis_outputs/operator_selection_execution_20261020_v1/data_protocol/split_config.json")
if(EXISTS "${_ucc_frozen_model}" AND EXISTS "${_ucc_frozen_split}")
  add_test(NAME unified_recommender COMMAND unified_recommender_test
    "${_ucc_frozen_model}" "${_ucc_frozen_split}")
endif()
add_executable(unified_algorithm_adapter_test algorithm_adapter_test.cpp)
target_link_libraries(unified_algorithm_adapter_test PRIVATE unified_compression_v1)
add_test(NAME unified_algorithm_adapter COMMAND unified_algorithm_adapter_test)
add_executable(bos_harness_integration_test bos_harness_integration_test.cpp)
target_link_libraries(bos_harness_integration_test PRIVATE unified_compression_v1)
add_test(NAME bos_harness_integration COMMAND bos_harness_integration_test)
add_executable(unified_operator_adapter_test operator_adapter_test.cpp)
target_link_libraries(unified_operator_adapter_test PRIVATE unified_compression_v1)
add_test(NAME unified_operator_adapter COMMAND unified_operator_adapter_test
  "${CMAKE_CURRENT_SOURCE_DIR}/../../benchmark/reference/actions/operator_harness_tree_catalog_v3.json")
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME unified_compression_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cli_integration_test.py
            $<TARGET_FILE:unified_compression_cli>)
endif()
