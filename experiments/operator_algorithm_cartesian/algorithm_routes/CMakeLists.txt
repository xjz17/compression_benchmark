cmake_minimum_required(VERSION 3.20)
project(corad_carrier_measure_cli LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
set(CORAD_SOURCE_DIR "" CACHE PATH "Path containing corad_lossless.cpp/.hpp")
if(NOT EXISTS "${CORAD_SOURCE_DIR}/corad_lossless.cpp")
  message(FATAL_ERROR "CORAD_SOURCE_DIR must contain corad_lossless.cpp")
endif()
add_executable(corad_carrier_measure_cli
  corad_measure_cli.cpp
  "${CORAD_SOURCE_DIR}/corad_lossless.cpp")
target_include_directories(corad_carrier_measure_cli PRIVATE
  "${CORAD_SOURCE_DIR}"
  "${CORAD_SOURCE_DIR}/third_party")
target_link_libraries(corad_carrier_measure_cli PRIVATE OpenSSL::Crypto ZLIB::ZLIB)
