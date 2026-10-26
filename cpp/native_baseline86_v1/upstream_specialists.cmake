# Native direct implementations for the baseline86 specialist/generic labels
# audited in upstream_specialists.cpp.  This file is intentionally separate
# from the frozen baseline86 CMakeLists so a caller can build and qualify the
# replacements before changing the formal inventory.

function(native_baseline86_define_upstream_specialists target_name)
  set(_usp_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  set(_usp_root "${_usp_dir}/../..")
  find_package(ZLIB REQUIRED)
  if(TARGET "${target_name}")
    message(FATAL_ERROR "target already exists: ${target_name}")
  endif()
  add_library("${target_name}" STATIC
    "${_usp_dir}/upstream_specialists.cpp"
    "${_usp_root}/cpp/src/ans_rans.cpp"
    "${_usp_root}/cpp/src/huffman_bytes.cpp"
    "${_usp_root}/cpp/src/arithmetic_bytes.cpp")
  target_compile_features("${target_name}" PUBLIC cxx_std_20)
  target_include_directories("${target_name}" PUBLIC
    "${_usp_dir}"
    "${_usp_root}/cpp/src"
    "${_usp_root}/external/compression/ALP/publication/source_code/include")
  target_link_libraries("${target_name}" PUBLIC ZLIB::ZLIB)
endfunction()

# Use this form when ans_rans.cpp, huffman_bytes.cpp and arithmetic_bytes.cpp
# are already sources of an existing target (as in native_baseline86_v1).
function(native_baseline86_attach_upstream_specialists target_name)
  set(_usp_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  set(_usp_root "${_usp_dir}/../..")
  find_package(ZLIB REQUIRED)
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR "unknown target: ${target_name}")
  endif()
  target_sources("${target_name}" PRIVATE
    "${_usp_dir}/upstream_specialists.cpp")
  target_compile_features("${target_name}" PUBLIC cxx_std_20)
  target_include_directories("${target_name}" PUBLIC
    "${_usp_dir}"
    "${_usp_root}/cpp/src"
    "${_usp_root}/external/compression/ALP/publication/source_code/include")
  target_link_libraries("${target_name}" PUBLIC ZLIB::ZLIB)
endfunction()

function(native_baseline86_add_upstream_specialist_tests library_target)
  set(_usp_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  add_executable(native_baseline86_upstream_specialists_test
    "${_usp_dir}/upstream_specialists_test.cpp")
  target_link_libraries(native_baseline86_upstream_specialists_test PRIVATE
    "${library_target}")
  add_test(NAME native_baseline86_upstream_specialists
    COMMAND native_baseline86_upstream_specialists_test)

  set(_cold_dir "${CMAKE_CURRENT_BINARY_DIR}/upstream_specialists_cold")
  add_test(NAME native_baseline86_upstream_specialists_cold_write
    COMMAND native_baseline86_upstream_specialists_test write-fixtures
            "${_cold_dir}")
  set_tests_properties(native_baseline86_upstream_specialists_cold_write
    PROPERTIES FIXTURES_SETUP baseline86_upstream_specialists_cold)
  add_test(NAME native_baseline86_upstream_specialists_cold_decode
    COMMAND native_baseline86_upstream_specialists_test decode-fixtures
            "${_cold_dir}")
  set_tests_properties(native_baseline86_upstream_specialists_cold_decode
    PROPERTIES FIXTURES_REQUIRED baseline86_upstream_specialists_cold)
endfunction()
