cmake_minimum_required(VERSION 3.18)
project(native_composite_candidate_v1 LANGUAGES C CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reuse the directly linked baseline86 implementation.  This component adds a
# distinct coarse-candidate envelope and selector boundary; it does not add or
# alias any atomic graph action.
if(NOT TARGET native_baseline86_v1)
  add_subdirectory(../native_baseline86_v1
                   ${CMAKE_CURRENT_BINARY_DIR}/native_baseline86 EXCLUDE_FROM_ALL)
endif()
find_package(OpenSSL REQUIRED)
find_path(NATIVE_COMPOSITE_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS
    ${CMAKE_CURRENT_LIST_DIR}/../../external/compression/FastLanes/src/include/fls/json
    /opt/homebrew/include /usr/local/include /usr/include
  REQUIRED)

add_library(native_composite_candidate_v1 STATIC composite.cpp)
target_include_directories(native_composite_candidate_v1 PUBLIC
  . ../native_baseline86_v1)
target_include_directories(native_composite_candidate_v1 PRIVATE
  ${NATIVE_COMPOSITE_JSON_INCLUDE_DIR})
target_link_libraries(native_composite_candidate_v1 PUBLIC
  native_baseline86_v1 OpenSSL::Crypto)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(native_composite_candidate_v1 PRIVATE
    -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()

add_executable(native_composite_candidate_cli cli.cpp)
target_link_libraries(native_composite_candidate_cli PRIVATE
  native_composite_candidate_v1)
add_executable(native_composite_candidate_test test.cpp)
target_link_libraries(native_composite_candidate_test PRIVATE
  native_composite_candidate_v1)
add_test(NAME native_composite_candidate_roundtrip
         COMMAND native_composite_candidate_test)

# Encode and decode happen in separate executable invocations.  The decoder is
# given only the complete NCC1 frame; expected bytes are used by a third process.
set(_NCC_COLD ${CMAKE_CURRENT_BINARY_DIR}/cold_fixture)
add_test(NAME native_composite_candidate_cold_prepare
  COMMAND native_composite_candidate_test write-fixture ${_NCC_COLD})
set_tests_properties(native_composite_candidate_cold_prepare PROPERTIES
  FIXTURES_SETUP ncc_input)
add_test(NAME native_composite_candidate_cold_encode
  COMMAND native_composite_candidate_cli encode train-linear-v1 0.5 0 0.5 float64 1030
    ${_NCC_COLD}/values.bin ${_NCC_COLD}/missing.bitmap ${_NCC_COLD}/frame.ncc)
set_tests_properties(native_composite_candidate_cold_encode PROPERTIES
  FIXTURES_REQUIRED ncc_input FIXTURES_SETUP ncc_frame)
add_test(NAME native_composite_candidate_cold_decode
  COMMAND native_composite_candidate_cli decode ${_NCC_COLD}/frame.ncc
    ${_NCC_COLD}/decoded.values.bin ${_NCC_COLD}/decoded.missing.bitmap)
set_tests_properties(native_composite_candidate_cold_decode PROPERTIES
  FIXTURES_REQUIRED ncc_frame FIXTURES_SETUP ncc_decoded)
add_test(NAME native_composite_candidate_cold_check
  COMMAND native_composite_candidate_test check-fixture ${_NCC_COLD})
set_tests_properties(native_composite_candidate_cold_check PROPERTIES
  FIXTURES_REQUIRED ncc_decoded)
add_test(NAME native_composite_candidate_zero_denominator
  COMMAND native_composite_candidate_cli measure Blosc2 0.5 0 0.5 int64 3
    ${_NCC_COLD}/empty.values.bin ${_NCC_COLD}/all-missing.bitmap 0 1
    ${_NCC_COLD}/empty.ngf)
set_tests_properties(native_composite_candidate_zero_denominator PROPERTIES
  FIXTURES_REQUIRED ncc_input PASS_REGULAR_EXPRESSION "\\\"status\\\":\\\"N/A\\\"")
