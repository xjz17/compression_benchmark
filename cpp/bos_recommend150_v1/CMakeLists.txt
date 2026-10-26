cmake_minimum_required(VERSION 3.20)
project(bos_recommend150_v1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
  HINTS /opt/homebrew/include /usr/local/include /usr/include)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
  message(FATAL_ERROR "nlohmann/json.hpp was not found")
endif()

add_library(bos_recommend150_core
  recommend150.cpp
  ../operator_oracle_quality135_v1/quality_model.cpp)
target_include_directories(bos_recommend150_core PUBLIC
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ..
  ../operator_oracle_quality135_v1
  ../operator_recommend135_v1)
target_link_libraries(bos_recommend150_core PUBLIC OpenSSL::Crypto ZLIB::ZLIB)
target_compile_options(bos_recommend150_core PRIVATE
  -O3 -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(bos_recommend150 main.cpp)
target_link_libraries(bos_recommend150 PRIVATE bos_recommend150_core)

add_executable(bos_recommend150_selftest selftest.cpp)
target_link_libraries(bos_recommend150_selftest PRIVATE bos_recommend150_core)

enable_testing()
add_test(NAME bos_recommend150_selftest COMMAND bos_recommend150_selftest)
