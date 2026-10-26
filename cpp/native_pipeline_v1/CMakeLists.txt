cmake_minimum_required(VERSION 3.18)
project(native_pipeline_v1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
add_subdirectory(../native_features_v1 native_features)
add_subdirectory(../native_policy_v1 native_policy)
add_subdirectory(../native_graph_v1 native_graph)
find_path(NATIVE_PIPELINE_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /opt/homebrew/include /usr/local/include /usr/include REQUIRED)
add_library(wc_native_pipeline_v1 STATIC pipeline.cpp catalog.cpp task.cpp search.cpp online.cpp)
target_include_directories(wc_native_pipeline_v1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(wc_native_pipeline_v1 PRIVATE ${NATIVE_PIPELINE_JSON_INCLUDE_DIR})
target_link_libraries(wc_native_pipeline_v1 PUBLIC wc_native_features_v1
  weighted_grpo_native_policy_v1 native_graph_v1_core)
target_compile_options(wc_native_pipeline_v1 PRIVATE -Wall -Wextra -Wpedantic
  -fno-fast-math -ffp-contract=off)
add_executable(native_pipeline_v1_server server.cpp)
target_include_directories(native_pipeline_v1_server PRIVATE ${NATIVE_PIPELINE_JSON_INCLUDE_DIR})
target_link_libraries(native_pipeline_v1_server PRIVATE wc_native_pipeline_v1 native_graph_v1_stages)
add_executable(native_online_policy_server online_server.cpp)
target_include_directories(native_online_policy_server PRIVATE ${NATIVE_PIPELINE_JSON_INCLUDE_DIR})
target_link_libraries(native_online_policy_server PRIVATE wc_native_pipeline_v1 native_graph_v1_stages)
add_executable(native_online_training_server online_training_server.cpp)
target_include_directories(native_online_training_server PRIVATE ${NATIVE_PIPELINE_JSON_INCLUDE_DIR})
target_link_libraries(native_online_training_server PRIVATE wc_native_pipeline_v1 native_graph_v1_stages)
if(BUILD_TESTING)
  add_executable(native_pipeline_v1_online_contract online_contract.cpp)
  target_link_libraries(native_pipeline_v1_online_contract PRIVATE wc_native_pipeline_v1 native_graph_v1_stages)
  add_test(NAME native_pipeline_v1_online_contract COMMAND native_pipeline_v1_online_contract
    ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json
    ${CMAKE_CURRENT_SOURCE_DIR}/experiment_no_cmix_nncp_v1.json
    ${CMAKE_CURRENT_BINARY_DIR}/online-contract-fixtures)
  add_executable(native_pipeline_v1_cache_accounting_contract cache_accounting_contract.cpp)
  target_link_libraries(native_pipeline_v1_cache_accounting_contract PRIVATE wc_native_pipeline_v1 native_graph_v1_stages)
  add_test(NAME native_pipeline_v1_cache_accounting_contract COMMAND native_pipeline_v1_cache_accounting_contract
    ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json)
  add_executable(native_pipeline_v1_online_semantic_mask_contract
    online_semantic_mask_contract.cpp)
  target_link_libraries(native_pipeline_v1_online_semantic_mask_contract PRIVATE
    wc_native_pipeline_v1 native_graph_v1_stages)
  add_test(NAME native_pipeline_v1_online_semantic_mask_contract
    COMMAND native_pipeline_v1_online_semantic_mask_contract
      ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json
      ${CMAKE_CURRENT_SOURCE_DIR}/experiment_no_cmix_nncp_v1.json)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME native_online_training_server_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_online_training_server.py
      --server $<TARGET_FILE:native_online_training_server>
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json
      --experiment-profile ${CMAKE_CURRENT_SOURCE_DIR}/experiment_no_cmix_nncp_v1.json)
  add_executable(native_pipeline_v1_smoke smoke.cpp)
  target_link_libraries(native_pipeline_v1_smoke PRIVATE wc_native_pipeline_v1
    native_graph_v1_stages)
  add_executable(native_pipeline_v1_cold_decode cold_decode.cpp)
  target_link_libraries(native_pipeline_v1_cold_decode PRIVATE native_graph_v1_stages)
  add_executable(native_pipeline_v1_action_safety action_safety.cpp)
  target_link_libraries(native_pipeline_v1_action_safety PRIVATE wc_native_pipeline_v1
    native_graph_v1_stages)
  add_executable(native_pipeline_v1_pruning_contract pruning_contract.cpp)
  target_link_libraries(native_pipeline_v1_pruning_contract PRIVATE wc_native_pipeline_v1
    native_graph_v1_stages)
  set(_FIXTURES "${CMAKE_CURRENT_BINARY_DIR}/test-fixtures")
  add_test(NAME native_pipeline_v1_smoke COMMAND native_pipeline_v1_smoke "${_FIXTURES}")
  set_tests_properties(native_pipeline_v1_smoke PROPERTIES FIXTURES_SETUP pipeline_frames)
  add_test(NAME native_pipeline_v1_cold_decode COMMAND native_pipeline_v1_cold_decode "${_FIXTURES}")
  set_tests_properties(native_pipeline_v1_cold_decode PROPERTIES FIXTURES_REQUIRED pipeline_frames)
  add_test(NAME native_pipeline_v1_server_contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_server.py
      --server $<TARGET_FILE:native_pipeline_v1_server>
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json
      --experiment-profile ${CMAKE_CURRENT_SOURCE_DIR}/experiment_no_cmix_nncp_v1.json
      --graph-cli $<TARGET_FILE:native_graph_request_cli>)
  add_test(NAME native_pipeline_v1_action_safety
    COMMAND native_pipeline_v1_action_safety
      ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json)
  add_test(NAME native_pipeline_v1_pruning_contract
    COMMAND native_pipeline_v1_pruning_contract
      ${CMAKE_CURRENT_SOURCE_DIR}/frozen_complete_action_catalog_v1.json)
  add_test(NAME native_pipeline_v1_failure_continuation
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../scripts
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_native_failure_continuation_v1.py)
  add_test(NAME native_pipeline_v1_frozen_resume
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../scripts
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_native_frozen_evaluation_v1.py)
  add_test(NAME native_pipeline_v1_end_to_end_preflight_contract
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../scripts
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_validate_native_end_to_end_preflight_v1.py)
  add_test(NAME native_pipeline_v1_validation_worker_lifetime
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../scripts
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_native_validation_worker_lifetime_v1.py)
  add_test(NAME native_pipeline_v1_runtime_freeze_contract
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../scripts
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_freeze_native_runtime_v1.py)
endif()
