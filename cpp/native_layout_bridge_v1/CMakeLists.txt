cmake_minimum_required(VERSION 3.16)
project(native_layout_bridge_v1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(native_layout_bridge_v1 SHARED
    ../src/native_layout_bridge_v1.cpp ../src/openzl_layout_stages.cpp)
target_include_directories(native_layout_bridge_v1 PUBLIC ../src)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(native_layout_bridge_v1 PRIVATE -Wall -Wextra -Wpedantic
      -fno-fast-math -ffp-contract=off)
endif()
