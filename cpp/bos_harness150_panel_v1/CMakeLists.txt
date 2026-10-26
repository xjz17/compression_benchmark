cmake_minimum_required(VERSION 3.18)
project(bos_harness150_panel_v1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The reused baseline inventory includes audited legacy GNU C++ translation
# units. Keep the runtime's existing extension mode instead of changing it from
# this standalone consumer.
set(CMAKE_CXX_EXTENSIONS ON)

# This is an isolated build entry point.  It reuses the public UCC Registry but
# does not add this campaign runner to the repository's root build.
set(NATIVE_GRAPH_ENABLE_NNCP OFF CACHE BOOL "NNCP remains disabled" FORCE)
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../unified_compression_v1"
  "${CMAKE_BINARY_DIR}/unified_compression_v1"
  EXCLUDE_FROM_ALL)

add_executable(bos_harness150_panel main.cpp)
target_link_libraries(bos_harness150_panel PRIVATE unified_compression_v1)
target_compile_options(bos_harness150_panel PRIVATE
  -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)

enable_testing()
add_test(NAME bos_harness150_panel_selftest
         COMMAND bos_harness150_panel --selftest)
