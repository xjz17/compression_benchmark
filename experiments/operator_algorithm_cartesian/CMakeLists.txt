cmake_minimum_required(VERSION 3.20)
project(operator_algorithm_cartesian200_bundle LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenSSL REQUIRED)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# Reuse the exact native graph source closure.  Building this isolated target
# does not replace the accepted operator102 executables.
add_subdirectory("${REPO_ROOT}/cpp/native_graph_v1"
                 "${CMAKE_BINARY_DIR}/native_graph_v1" EXCLUDE_FROM_ALL)

set(BUNDLE_SOURCES
    source/o2r1_bundle_cli.cpp
    "${REPO_ROOT}/cpp/native_graph_v1/operator102_measure.cpp"
    "${REPO_ROOT}/cpp/native_graph_v1/operator102_dispatcher.cpp"
    "${REPO_ROOT}/cpp/native_graph_v1/operator102_tmpfile.cpp")
if(TARGET operator102_support_deps)
  list(APPEND BUNDLE_SOURCES
       "${REPO_ROOT}/cpp/native_graph_v1/operator102_special_bridges.cpp")
endif()

add_executable(o2r1_bundle_cli ${BUNDLE_SOURCES})
target_include_directories(o2r1_bundle_cli PRIVATE
    "${REPO_ROOT}/cpp/native_graph_v1" "${REPO_ROOT}/cpp/src"
    "${NATIVE_GRAPH_JSON_INCLUDE_DIR}")
target_link_libraries(o2r1_bundle_cli PRIVATE native_graph_v1_stages
    OpenSSL::Crypto libzstd_static)
if(TARGET operator102_support_deps)
  target_link_libraries(o2r1_bundle_cli PRIVATE operator102_support_deps)
  target_compile_definitions(o2r1_bundle_cli PRIVATE
      OPERATOR102_ENABLE_SPECIAL_BRIDGES=1 OPERATOR102_ENABLE_SUPPORT_DEPS=1)
endif()
