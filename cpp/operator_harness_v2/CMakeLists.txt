cmake_minimum_required(VERSION 3.16)
project(operator_harness_v2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The v2 harness is an execution policy boundary.  The shared upstream runtime
# still compiles some ordinary probability support, but the wrapper rejects all
# five excluded families before every forward/inverse call.  NNCP itself is
# additionally kept out of the build even when the host can provide it.
set(NATIVE_GRAPH_ENABLE_NNCP OFF CACHE BOOL
    "NNCP is permanently disabled in operator_harness_v2" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../native_graph_v1"
                 "${CMAKE_CURRENT_BINARY_DIR}/native_graph_v1" EXCLUDE_FROM_ALL)
if(TARGET native_graph_v1_nncp)
  message(FATAL_ERROR "operator_harness_v2 must not build native_graph_v1_nncp")
endif()

find_package(OpenSSL REQUIRED)
find_path(OPERATOR_HARNESS_V2_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include /usr/include REQUIRED)

option(OPERATOR_HARNESS_V2_ENABLE_SUPPORT_DEPS
  "Enable the optional heavy residual codec closure" OFF)

add_library(operator_harness_v2 STATIC
  a048_typed_planes.cpp
  binding_contract.cpp
  policy_dispatch.cpp
  protocol.cpp
  special_stages.cpp
  "${CMAKE_CURRENT_LIST_DIR}/../native_graph_v1/operator102_dispatcher.cpp")
target_include_directories(operator_harness_v2 PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/../native_graph_v1"
  "${CMAKE_CURRENT_LIST_DIR}/../src"
  "${OPERATOR_HARNESS_V2_JSON_INCLUDE_DIR}")
target_link_libraries(operator_harness_v2 PUBLIC
  native_graph_v1_stages OpenSSL::Crypto)
target_compile_options(operator_harness_v2 PRIVATE
  -Wall -Wextra -Wpedantic -ffp-contract=off)

if(OPERATOR_HARNESS_V2_ENABLE_SUPPORT_DEPS)
  if(NOT TARGET operator102_support_deps)
    message(FATAL_ERROR "operator102_support_deps target is unavailable")
  endif()
  target_compile_definitions(operator_harness_v2 PRIVATE
    OPERATOR102_ENABLE_SUPPORT_DEPS=1)
  target_link_libraries(operator_harness_v2 PUBLIC operator102_support_deps)
endif()

add_executable(operator_harness_v2_cli main.cpp)
target_link_libraries(operator_harness_v2_cli PRIVATE operator_harness_v2)

# batch_runner.cpp is owned by the durable coordinator.  Keep the native
# runner independent from codec libraries: it only forks the CLI and hashes
# committed files.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/batch_runner.cpp")
  add_executable(operator_harness_v2_batch_runner batch_runner.cpp)
  target_include_directories(operator_harness_v2_batch_runner PRIVATE
    "${OPERATOR_HARNESS_V2_JSON_INCLUDE_DIR}")
  target_link_libraries(operator_harness_v2_batch_runner PRIVATE OpenSSL::Crypto)
  target_compile_options(operator_harness_v2_batch_runner PRIVATE
    -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

enable_testing()
add_executable(operator_harness_v2_protocol_test protocol_test.cpp)
target_link_libraries(operator_harness_v2_protocol_test PRIVATE operator_harness_v2)
target_compile_definitions(operator_harness_v2_protocol_test PRIVATE
  OPERATOR_HARNESS_V2_TEST_CATALOG="${CMAKE_CURRENT_LIST_DIR}/../../benchmark/reference/actions/operator_harness_batch_catalog_v2.json"
  OPERATOR_HARNESS_V3_TEST_CATALOG="${CMAKE_CURRENT_LIST_DIR}/../../benchmark/reference/actions/operator_harness_tree_catalog_v3.json")
add_test(NAME operator_harness_v2_protocol_test
         COMMAND operator_harness_v2_protocol_test)

# Input preparation only; payload extraction and deterministic boundary bits
# remain native C++. It neither invokes kernels nor alters the prior runtime.
add_executable(operator_harness_v2_stress_input_builder stress_input_builder.cpp)
target_include_directories(operator_harness_v2_stress_input_builder PRIVATE
  "${OPERATOR_HARNESS_V2_JSON_INCLUDE_DIR}")
target_link_libraries(operator_harness_v2_stress_input_builder PRIVATE OpenSSL::Crypto)
target_compile_options(operator_harness_v2_stress_input_builder PRIVATE
  -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(operator_harness_v2_a048_typed_planes_test
  a048_typed_planes_test.cpp)
target_link_libraries(operator_harness_v2_a048_typed_planes_test PRIVATE
  operator_harness_v2)
add_test(NAME operator_harness_v2_a048_typed_planes_test
         COMMAND operator_harness_v2_a048_typed_planes_test)
