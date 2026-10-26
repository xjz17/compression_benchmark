cmake_minimum_required(VERSION 3.16)
project(operator_harness_v1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This harness must never compile or expose NNCP. Keep the native runtime's
# default ON for its existing standalone builds, but fail closed here.
set(NATIVE_GRAPH_ENABLE_NNCP OFF CACHE BOOL "NNCP is disabled in operator_harness_v1" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../native_graph_v1"
                 "${CMAKE_CURRENT_BINARY_DIR}/native_graph_v1" EXCLUDE_FROM_ALL)
if(TARGET native_graph_v1_nncp)
  message(FATAL_ERROR "operator_harness_v1 must not build native_graph_v1_nncp")
endif()

find_path(OPERATOR_HARNESS_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include /usr/include REQUIRED)

add_library(operator_harness_v1 STATIC harness_dispatch.cpp protocol.cpp)
target_include_directories(operator_harness_v1 PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/../native_graph_v1"
  "${CMAKE_CURRENT_LIST_DIR}/../src"
  "${OPERATOR_HARNESS_JSON_INCLUDE_DIR}")
target_link_libraries(operator_harness_v1 PUBLIC native_graph_v1_stages)
target_compile_options(operator_harness_v1 PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(operator_harness_v1_cli main.cpp)
target_link_libraries(operator_harness_v1_cli PRIVATE operator_harness_v1)

enable_testing()
add_executable(operator_harness_v1_protocol_test protocol_test.cpp)
target_link_libraries(operator_harness_v1_protocol_test PRIVATE operator_harness_v1)
add_test(NAME operator_harness_v1_protocol_test COMMAND operator_harness_v1_protocol_test)
