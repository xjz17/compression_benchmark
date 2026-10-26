#include "quality_data.hpp"

#include "../operator_recommend135_v1/common.hpp"
#include "../operator_recommend135_v1/model.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
using Json = nlohmann::json;
using Bytes = std::vector<std::uint8_t>;
namespace fs = std::filesystem;
namespace d = operator_recommend135_v1::detail;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function&& function, const std::string& expected) {
  try {
    function();
  } catch (const std::exception& error) {
    check(std::string(error.what()).find(expected) != std::string::npos,
          "wrong rejection: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected rejection: " + expected);
}

void write(const fs::path& path, const Bytes& bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  check(bool(stream), "cannot create fixture");
  if (!bytes.empty())
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  check(bool(stream), "cannot write fixture");
}

void write(const fs::path& path, const std::string& bytes) {
  write(path, Bytes(bytes.begin(), bytes.end()));
}

Bytes words(std::initializer_list<std::uint64_t> values) {
  Bytes result;
  for (const auto value : values)
    for (unsigned byte = 0; byte < 8; ++byte)
      result.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
  return result;
}

Bytes bitmap(std::uint64_t rows,
             std::initializer_list<std::uint64_t> missing_rows) {
  Bytes result((rows + 7) / 8, 0);
  for (const auto row : missing_rows)
    result.at(row / 8) |= static_cast<std::uint8_t>(1U << (row % 8));
  return result;
}

struct Fixture {
  fs::path root;
  fs::path columns;
  fs::path payload;
  Bytes payload_bytes;

  Fixture() {
    root = fs::temp_directory_path() /
           ("operator-oracle-quality135-test-" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
  }
  ~Fixture() { fs::remove_all(root); }
};

Fixture make_fixture() {
  Fixture fixture;
  const auto column_id =
      d::sha(std::string("fixture-column-id"));
  const auto empty_values = Bytes{};
  const auto empty_missing = bitmap(3, {0, 1, 2});
  const auto values = words({10, 11, 12, 13, 14, 15, 16, 17});
  const auto missing = bitmap(12, {0, 2, 5, 10});
  const Bytes pad = {0xaa, 0xbb, 0xcc};
  fixture.payload_bytes = pad;
  const auto empty_values_offset = fixture.payload_bytes.size();
  fixture.payload_bytes.insert(fixture.payload_bytes.end(), empty_values.begin(),
                               empty_values.end());
  const auto empty_missing_offset = fixture.payload_bytes.size();
  fixture.payload_bytes.insert(fixture.payload_bytes.end(), empty_missing.begin(),
                               empty_missing.end());
  const auto values_offset = fixture.payload_bytes.size();
  fixture.payload_bytes.insert(fixture.payload_bytes.end(), values.begin(), values.end());
  const auto missing_offset = fixture.payload_bytes.size();
  fixture.payload_bytes.insert(fixture.payload_bytes.end(), missing.begin(), missing.end());
  fixture.payload = fixture.root / "payload.bin";
  write(fixture.payload, fixture.payload_bytes);

  auto block = [&](const char* id, std::uint64_t index, std::uint64_t rows,
                   std::uint64_t valid, std::uint64_t missing_count,
                   std::uint64_t value_offset, const Bytes& block_values,
                   std::uint64_t bitmap_offset, const Bytes& block_missing) {
    return Json{{"block_id", id},
                {"block_index", index},
                {"dataset_id", "dataset-dev"},
                {"column", "metric"},
                {"dtype", "int64"},
                {"row_count", rows},
                {"valid_count", valid},
                {"missing_count", missing_count},
                {"original_numeric_bytes", block_values.size()},
                {"values_path", fixture.payload.string()},
                {"values_offset", value_offset},
                {"values_length", block_values.size()},
                {"values_sha256", d::sha(block_values)},
                {"missing_bitmap_path", fixture.payload.string()},
                {"missing_bitmap_offset", bitmap_offset},
                {"missing_bitmap_length", block_missing.size()},
                {"missing_bitmap_sha256", d::sha(block_missing)}};
  };
  Json blocks = Json::array();
  blocks.push_back(block("empty", 0, 3, 0, 3, empty_values_offset,
                         empty_values, empty_missing_offset, empty_missing));
  blocks.push_back(block("data", 1, 12, 8, 4, values_offset, values,
                         missing_offset, missing));
  Json manifest{{"schema", "native-baseline86-scaled-column-segment-manifest-v1"},
                {"dataset_id", "dataset-dev"},
                {"column_identity",
                 {{"dataset_id", "dataset-dev"},
                  {"column", "metric"},
                  {"dtype", "int64"}}},
                {"blocks", blocks}};
  const auto manifest_path = fixture.root / (column_id + ".json");
  const auto manifest_text = manifest.dump();
  write(manifest_path, manifest_text);
  Json columns{{"schema", "operator-prepared135-source-columns-v1"},
               {"scope", "fixture"},
               {"oracle_reference_jsonl_sha256", d::sha("fixture-oracle")},
               {"columns",
                Json::array({{{"dataset_id", "dataset-dev"},
                              {"column_id", column_id},
                              {"column", "metric"},
                              {"dtype", "int64"},
                              {"manifest_path", manifest_path.string()},
                              {"manifest_sha256", d::sha(manifest_text)},
                              {"original_numeric_bytes", values.size()},
                              {"valid_value_count", 8}}})}};
  fixture.columns = fixture.root / "COLUMNS.json";
  write(fixture.columns, columns.dump());
  return fixture;
}

Json read_first_jsonl(const fs::path& path) {
  std::ifstream stream(path);
  std::string line;
  std::getline(stream, line);
  check(bool(stream), "cannot read feature row");
  return Json::parse(line);
}

void test_first_nonempty_prefix_preserves_null_layout() {
  auto fixture = make_fixture();
  const auto output = fixture.root / "features.jsonl";
  operator_oracle_quality135_v1::FeatureBuildOptions options;
  options.max_valid_values = 4;
  options.require_frozen_current135 = false;
  const auto receipt =
      operator_oracle_quality135_v1::build_features(fixture.columns, output,
                                                    options);
  check(receipt.at("column_count") == 1 &&
            receipt.at("oracle_labels_read") == false,
        "receipt scope or label boundary mismatch");
  const auto& timings = receipt.at("timings");
  for (const auto* phase : {"source_validation_and_read", "feature_extraction",
                            "jsonl_serialization", "output_commit", "total"}) {
    check(timings.contains(phase) && timings.at(phase).contains("cpu_ns") &&
              timings.at(phase).contains("wall_ns"),
          std::string("missing timing phase ") + phase);
  }
  check(receipt.at("timing_contract").at("oracle_label_io_included") == false,
        "timing contract crosses oracle-label boundary");
  const auto row = read_first_jsonl(output);
  check(row.at("sample_source").at("block_id") == "data",
        "builder did not choose first nonempty block");
  check(row.at("input_dtype") == "int64" &&
            row.at("column_id").is_string() &&
            row.at("dataset_id") == "dataset-dev",
        "canonical row identity differs");
  check(row.at("sample").at("row_count") == 7 &&
            row.at("sample").at("valid_count") == 4 &&
            row.at("sample").at("missing_count") == 3 &&
            row.at("sample").at("truncated") == true,
        "prefix row/null/value shape differs");
  const auto expected_values = words({10, 11, 12, 13});
  const auto expected_missing = bitmap(7, {0, 2, 5});
  check(row.at("sample").at("values_sha256") == d::sha(expected_values) &&
            row.at("sample").at("missing_sha256") == d::sha(expected_missing),
        "sample bit binding differs");
  const auto expected = operator_recommend135_v1::extract_features(
      expected_values, "int64", expected_missing, 7);
  check(row.at("features").get<std::vector<double>>() == expected,
        "feature vector differs from existing extractor");
}

void test_selected_region_sha_is_mandatory() {
  auto fixture = make_fixture();
  fixture.payload_bytes.back() ^= 0x01;
  write(fixture.payload, fixture.payload_bytes);
  operator_oracle_quality135_v1::FeatureBuildOptions options;
  options.max_valid_values = 4;
  options.require_frozen_current135 = false;
  rejects([&] {
    operator_oracle_quality135_v1::build_features(
        fixture.columns, fixture.root / "features.jsonl", options);
  }, "selected source missing bitmap SHA256 mismatch");
}

void test_formal_scope_gate_rejects_fixture() {
  auto fixture = make_fixture();
  rejects([&] {
    operator_oracle_quality135_v1::build_features(
        fixture.columns, fixture.root / "features.jsonl");
  }, "column count differs from 3519");
}

}  // namespace

int main() {
  try {
    test_first_nonempty_prefix_preserves_null_layout();
    test_selected_region_sha_is_mandatory();
    test_formal_scope_gate_rejects_fixture();
    std::cout << "quality_data_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "quality_data_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
