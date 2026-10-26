#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace bos_recommend150_v1 {

struct TuneRequest {
  std::filesystem::path features_jsonl;
  std::filesystem::path split_json;
  std::filesystem::path cohorts_json;
  std::filesystem::path candidates_json;
  std::filesystem::path development_feed_jsonl;
  std::filesystem::path output_directory;
  std::size_t expected_samples = 3433;
  std::size_t expected_methods = 85;
};

struct FinalRequest {
  std::filesystem::path features_jsonl;
  std::filesystem::path split_json;
  std::filesystem::path cohorts_json;
  std::filesystem::path candidates_json;
  std::filesystem::path feed_jsonl;
  std::filesystem::path model_directory;
  std::filesystem::path output_json;
  std::string partition;  // confirm or external-final
  std::size_t expected_samples = 0;
  std::size_t expected_methods = 85;
  std::size_t timing_repeats = 1000;
};

nlohmann::json tune(const TuneRequest& request);
nlohmann::json final_evaluate(const FinalRequest& request);

}  // namespace bos_recommend150_v1
