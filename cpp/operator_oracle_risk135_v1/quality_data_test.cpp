#include "quality_data.hpp"

#include "../operator_recommend135_v1/common.hpp"
#include "../operator_recommend135_v1/model.hpp"

#include <algorithm>
#include <cmath>
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

Bytes repeated_word(std::uint64_t value, std::size_t count) {
  Bytes result;
  result.reserve(count * 8);
  for (std::size_t index = 0; index < count; ++index)
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

Fixture make_single_block_fixture(const std::string& dataset_id,
                                  const std::string& dtype,
                                  const Bytes& values,
                                  std::uint64_t row_count,
                                  const Bytes& missing) {
  Fixture fixture;
  const std::size_t dtype_width = dtype == "float32" ? 4 : 8;
  check(values.size() % dtype_width == 0, "bad fixture value width");
  const auto valid_count = values.size() / dtype_width;
  const auto column_id = d::sha(dataset_id + ":metric:" + dtype);
  fixture.payload_bytes = values;
  const auto missing_offset = fixture.payload_bytes.size();
  fixture.payload_bytes.insert(fixture.payload_bytes.end(), missing.begin(),
                               missing.end());
  fixture.payload = fixture.root / "payload.bin";
  write(fixture.payload, fixture.payload_bytes);
  const Json block{{"block_id", "data"},
                   {"block_index", 0},
                   {"dataset_id", dataset_id},
                   {"column", "metric"},
                   {"dtype", dtype},
                   {"row_count", row_count},
                   {"valid_count", valid_count},
                   {"missing_count", row_count - valid_count},
                   {"original_numeric_bytes", values.size()},
                   {"values_path", fixture.payload.string()},
                   {"values_offset", 0},
                   {"values_length", values.size()},
                   {"values_sha256", d::sha(values)},
                   {"missing_bitmap_path", fixture.payload.string()},
                   {"missing_bitmap_offset", missing_offset},
                   {"missing_bitmap_length", missing.size()},
                   {"missing_bitmap_sha256", d::sha(missing)}};
  const Json manifest{
      {"schema", "native-baseline86-scaled-column-segment-manifest-v1"},
      {"dataset_id", dataset_id},
      {"column_identity",
       {{"dataset_id", dataset_id}, {"column", "metric"}, {"dtype", dtype}}},
      {"blocks", Json::array({block})}};
  const auto manifest_path = fixture.root / (column_id + ".json");
  const auto manifest_text = manifest.dump();
  write(manifest_path, manifest_text);
  const Json columns{
      {"schema", "operator-prepared135-source-columns-v1"},
      {"scope", "fixture"},
      {"oracle_reference_jsonl_sha256", d::sha("fixture-oracle")},
      {"columns",
       Json::array({{{"dataset_id", dataset_id},
                     {"column_id", column_id},
                     {"column", "metric"},
                     {"dtype", dtype},
                     {"manifest_path", manifest_path.string()},
                     {"manifest_sha256", d::sha(manifest_text)},
                     {"original_numeric_bytes", values.size()},
                     {"valid_value_count", valid_count}}})}};
  fixture.columns = fixture.root / "COLUMNS.json";
  write(fixture.columns, columns.dump());
  return fixture;
}

fs::path write_development_metadata(const Fixture& fixture) {
  const auto columns = d::read_json(fixture.columns);
  const auto& source = columns.at("columns").at(0);
  const auto path = fixture.root / "INPUT_METADATA.json";
  const Json metadata{
      {"schema", "operator-oracle-quality135-input-metadata-v1"},
      {"oracle_sha256", columns.at("oracle_reference_jsonl_sha256")},
      {"split_sha256", d::sha("fixture-split")},
      {"legacy_groups_sha256", d::sha("fixture-groups")},
      {"columns",
       Json::array({{{"column_id", source.at("column_id")},
                     {"sample_id", source.at("column_id")},
                     {"dataset_id", source.at("dataset_id")},
                     {"dtype", source.at("dtype")},
                     {"manifest_path", source.at("manifest_path")},
                     {"manifest_sha256", source.at("manifest_sha256")},
                     {"original_numeric_bytes",
                      source.at("original_numeric_bytes")},
                     {"valid_value_count", source.at("valid_value_count")},
                     {"source_group", d::sha("fixture-source-group")},
                     {"legacy_group", "source:fixture"},
                     {"split", "development"},
                     {"dev_fold", 0}}})}};
  write(path, metadata.dump());
  return path;
}

Json read_first_jsonl(const fs::path& path) {
  std::ifstream stream(path);
  std::string line;
  std::getline(stream, line);
  check(bool(stream), "cannot read feature row");
  return Json::parse(line);
}

std::size_t jsonl_line_count(const fs::path& path) {
  std::ifstream stream(path);
  check(bool(stream), "cannot read JSONL fixture output");
  std::size_t count = 0;
  std::string line;
  while (std::getline(stream, line)) count += line.empty() ? 0 : 1;
  return count;
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

void test_variant_builder_requires_development_source_sha() {
  auto fixture = make_fixture();
  const auto metadata = write_development_metadata(fixture);
  fixture.payload_bytes.at(4) ^= 0x01;
  write(fixture.payload, fixture.payload_bytes);
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.max_valid_values_per_window = 4;
  options.require_frozen_current135 = false;
  rejects(
      [&] {
        operator_oracle_quality135_v1::build_development_feature_variants(
            fixture.columns, metadata, fixture.root / "features", options);
      },
      "selected source values SHA256 mismatch");
}

void test_formal_scope_gate_rejects_fixture() {
  auto fixture = make_fixture();
  rejects([&] {
    operator_oracle_quality135_v1::build_features(
        fixture.columns, fixture.root / "features.jsonl");
  }, "column count differs from 3519");
}

void test_development_variants_publish_prefix24_and_multi3_contracts() {
  auto fixture = make_fixture();
  const auto metadata = write_development_metadata(fixture);
  operator_oracle_quality135_v1::FeatureBuildOptions old_options;
  old_options.max_valid_values = 4;
  old_options.require_frozen_current135 = false;
  operator_oracle_quality135_v1::build_features(
      fixture.columns, fixture.root / "old-prefix.jsonl", old_options);
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.max_valid_values_per_window = 4;
  options.require_frozen_current135 = false;
  const auto receipt =
      operator_oracle_quality135_v1::build_development_feature_variants(
          fixture.columns, metadata, fixture.root / "features", options);
  check(receipt.at("development_column_count") == 1 &&
            receipt.at("confirm_column_count_processed") == 0,
        "variant builder crossed the development-only boundary");
  const auto prefix = read_first_jsonl(fixture.root / "features" / "prefix24" /
                                       "FEATURES.jsonl");
  const auto multi = read_first_jsonl(fixture.root / "features" / "multi3" /
                                      "FEATURES.jsonl");
  check(prefix.at("feature_variant") == "prefix24" &&
            prefix.at("features").size() == 24,
        "prefix24 output contract differs");
  check(prefix.at("features") ==
            read_first_jsonl(fixture.root / "old-prefix.jsonl").at("features"),
        "new prefix24 values differ from the frozen extractor path");
  check(multi.at("feature_variant") == "multi3" &&
            multi.at("features").size() == 98,
        "multi3 output contract differs");
  const auto prefix_receipt = d::read_json(
      fixture.root / "features" / "prefix24" / "FEATURES_RECEIPT.json");
  const auto multi_receipt = d::read_json(
      fixture.root / "features" / "multi3" / "FEATURES_RECEIPT.json");
  check(prefix_receipt.at("schema") ==
                "operator-oracle-risk135-features-receipt-v1" &&
            prefix_receipt.at("feature_names") ==
                Json(operator_oracle_quality135_v1::prefix24_feature_names()) &&
            multi_receipt.at("feature_names") ==
                Json(operator_oracle_quality135_v1::multi3_feature_names()) &&
            multi_receipt.at("feature_count") == 98,
        "variant receipt names/schema differ");
  for (const auto* phase : {"source_validation_and_read",
                            "feature_extraction", "jsonl_serialization",
                            "output_commit", "total"})
    check(multi_receipt.at("timings").contains(phase) &&
              multi_receipt.at("timings").at(phase).contains("cpu_ns") &&
              multi_receipt.at("timings").at(phase).contains("wall_ns"),
          std::string("multi3 receipt lacks timing phase ") + phase);
  check(prefix.at("oracle_labels_read") == false &&
            multi.at("oracle_labels_read") == false,
        "feature variants crossed the oracle-label boundary");
}

void test_confirm_metadata_is_bound_but_confirm_manifest_is_not_opened() {
  auto fixture = make_fixture();
  auto columns = d::read_json(fixture.columns);
  auto confirm = columns.at("columns").at(0);
  confirm["dataset_id"] = "dataset-confirm";
  confirm["column_id"] = d::sha("confirm-column");
  confirm["column"] = "confirm-metric";
  confirm["manifest_path"] =
      (fixture.root / "must-not-be-opened-confirm-manifest.json").string();
  confirm["manifest_sha256"] = d::sha("unopened-confirm-manifest");
  columns["columns"].push_back(confirm);
  write(fixture.columns, columns.dump());

  const auto metadata_path = write_development_metadata(fixture);
  auto metadata = d::read_json(metadata_path);
  metadata["columns"].push_back(
      {{"column_id", confirm.at("column_id")},
       {"sample_id", confirm.at("column_id")},
       {"dataset_id", confirm.at("dataset_id")},
       {"dtype", confirm.at("dtype")},
       {"manifest_path", confirm.at("manifest_path")},
       {"manifest_sha256", confirm.at("manifest_sha256")},
       {"original_numeric_bytes", confirm.at("original_numeric_bytes")},
       {"valid_value_count", confirm.at("valid_value_count")},
       {"source_group", d::sha("confirm-source-group")},
       {"legacy_group", "source:confirm"},
       {"split", "confirm"}});
  write(metadata_path, metadata.dump());
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.max_valid_values_per_window = 4;
  options.require_frozen_current135 = false;
  const auto receipt =
      operator_oracle_quality135_v1::build_development_feature_variants(
          fixture.columns, metadata_path, fixture.root / "features", options);
  check(receipt.at("development_column_count") == 1 &&
            receipt.at("confirm_column_count_available") == 1 &&
            receipt.at("confirm_column_count_processed") == 0,
        "confirm scope counters differ");
  check(jsonl_line_count(fixture.root / "features" / "prefix24" /
                         "FEATURES.jsonl") == 1 &&
            jsonl_line_count(fixture.root / "features" / "multi3" /
                             "FEATURES.jsonl") == 1,
        "confirm column leaked into development features");
}

Fixture make_sparse_multiblock_fixture() {
  Fixture fixture;
  const std::string dataset_id = "sparse-multiblock";
  const auto column_id = d::sha(dataset_id + ":metric:int64");
  struct Part {
    std::string id;
    std::uint64_t rows;
    Bytes values;
    Bytes missing;
  };
  std::vector<Part> parts;
  parts.push_back({"left", 7,
                   words({0x8000000000000000ULL, 0,
                          0x7fffffffffffffffULL}),
                   bitmap(7, {1, 4, 5, 6})});
  parts.push_back({"empty-gap", 3, {}, bitmap(3, {0, 1, 2})});
  parts.push_back({"right", 11,
                   words({42, 42, 43, 44, 0xffffffffffffffffULL}),
                   bitmap(11, {0, 2, 5, 8, 9, 10})});
  Json blocks = Json::array();
  std::uint64_t original_bytes = 0;
  std::uint64_t total_valid = 0;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    const auto value_offset = fixture.payload_bytes.size();
    fixture.payload_bytes.insert(fixture.payload_bytes.end(),
                                 parts[index].values.begin(),
                                 parts[index].values.end());
    const auto missing_offset = fixture.payload_bytes.size();
    fixture.payload_bytes.insert(fixture.payload_bytes.end(),
                                 parts[index].missing.begin(),
                                 parts[index].missing.end());
    const auto valid = parts[index].values.size() / 8;
    original_bytes += parts[index].values.size();
    total_valid += valid;
    blocks.push_back(
        {{"block_id", parts[index].id},
         {"block_index", index},
         {"dataset_id", dataset_id},
         {"column", "metric"},
         {"dtype", "int64"},
         {"row_count", parts[index].rows},
         {"valid_count", valid},
         {"missing_count", parts[index].rows - valid},
         {"original_numeric_bytes", parts[index].values.size()},
         {"values_path", (fixture.root / "payload.bin").string()},
         {"values_offset", value_offset},
         {"values_length", parts[index].values.size()},
         {"values_sha256", d::sha(parts[index].values)},
         {"missing_bitmap_path", (fixture.root / "payload.bin").string()},
         {"missing_bitmap_offset", missing_offset},
         {"missing_bitmap_length", parts[index].missing.size()},
         {"missing_bitmap_sha256", d::sha(parts[index].missing)}});
  }
  fixture.payload = fixture.root / "payload.bin";
  write(fixture.payload, fixture.payload_bytes);
  const Json manifest{
      {"schema", "native-baseline86-scaled-column-segment-manifest-v1"},
      {"dataset_id", dataset_id},
      {"column_identity",
       {{"dataset_id", dataset_id}, {"column", "metric"}, {"dtype", "int64"}}},
      {"blocks", blocks}};
  const auto manifest_path = fixture.root / (column_id + ".json");
  const auto manifest_text = manifest.dump();
  write(manifest_path, manifest_text);
  const Json columns{
      {"schema", "operator-prepared135-source-columns-v1"},
      {"scope", "fixture"},
      {"oracle_reference_jsonl_sha256", d::sha("fixture-oracle")},
      {"columns",
       Json::array({{{"dataset_id", dataset_id},
                     {"column_id", column_id},
                     {"column", "metric"},
                     {"dtype", "int64"},
                     {"manifest_path", manifest_path.string()},
                     {"manifest_sha256", d::sha(manifest_text)},
                     {"original_numeric_bytes", original_bytes},
                     {"valid_value_count", total_valid}}})}};
  fixture.columns = fixture.root / "COLUMNS.json";
  write(fixture.columns, columns.dump());
  return fixture;
}

void test_multi3_spans_unequal_sparse_blocks_and_int64_edges() {
  auto fixture = make_sparse_multiblock_fixture();
  const auto metadata = write_development_metadata(fixture);
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.max_valid_values_per_window = 4;
  options.require_frozen_current135 = false;
  operator_oracle_quality135_v1::build_development_feature_variants(
      fixture.columns, metadata, fixture.root / "features", options);
  const auto multi = read_first_jsonl(fixture.root / "features" / "multi3" /
                                      "FEATURES.jsonl");
  const auto& windows = multi.at("multi3_sampling").at("windows");
  check(windows.size() == 3 && windows.at(0).at("segments").size() == 3,
        "head window did not preserve the empty inter-block gap");
  for (const auto value : multi.at("features").get<std::vector<double>>())
    check(std::isfinite(value), "sparse int64 edge feature is non-finite");
}

void test_short_float64_nan_window_is_deduplicated_with_bit_proof() {
  const auto values = words({0x7ff8000000000001ULL,
                             0x7ff8000000000002ULL,
                             0x7ff0000000000000ULL,
                             0xfff0000000000000ULL,
                             0x3ff0000000000000ULL});
  auto fixture = make_single_block_fixture("float-bits", "float64", values,
                                           7, bitmap(7, {1, 5}));
  const auto metadata = write_development_metadata(fixture);
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.require_frozen_current135 = false;
  operator_oracle_quality135_v1::build_development_feature_variants(
      fixture.columns, metadata, fixture.root / "features", options);
  const auto multi = read_first_jsonl(fixture.root / "features" / "multi3" /
                                      "FEATURES.jsonl");
  const auto& sampling = multi.at("multi3_sampling");
  check(sampling.at("unique_window_count") == 1 &&
            sampling.at("windows").at(0).at("roles").size() == 3,
        "short column head/middle/tail windows were not deduplicated");
  check(sampling.at("windows").at(0).at("values_sha256") == d::sha(values),
        "float64 NaN payload bits lost their sample proof");
  const auto expected = operator_recommend135_v1::extract_features(
      values, "float64", bitmap(7, {1, 5}), 7);
  const auto actual = multi.at("features").get<std::vector<double>>();
  check(actual.at(72 + 13) == expected.at(13),
        "tail non-finite fraction changed for NaN/Inf payloads");
}

void test_development_empty_column_is_rejected_explicitly() {
  auto fixture = make_single_block_fixture("empty-column", "int64", {}, 3,
                                           bitmap(3, {0, 1, 2}));
  const auto metadata = write_development_metadata(fixture);
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.require_frozen_current135 = false;
  rejects(
      [&] {
        operator_oracle_quality135_v1::build_development_feature_variants(
            fixture.columns, metadata, fixture.root / "features", options);
      },
      "empty or inconsistent numeric values");
}

void test_multi3_samples_distinct_regions_inside_one_huge_block() {
  constexpr std::size_t window = 32768;
  auto values = repeated_word(10, window);
  const auto middle_values = repeated_word(20, window);
  const auto tail_values = repeated_word(30, window);
  values.insert(values.end(), middle_values.begin(), middle_values.end());
  values.insert(values.end(), tail_values.begin(), tail_values.end());
  auto fixture = make_single_block_fixture(
      "long-drift", "int64", values, 3 * window, bitmap(3 * window, {}));
  const auto metadata = write_development_metadata(fixture);
  operator_oracle_quality135_v1::FeatureVariantBuildOptions options;
  options.max_valid_values_per_window = window;
  options.require_frozen_current135 = false;
  operator_oracle_quality135_v1::build_development_feature_variants(
      fixture.columns, metadata, fixture.root / "features", options);

  const auto prefix = read_first_jsonl(fixture.root / "features" / "prefix24" /
                                       "FEATURES.jsonl");
  const auto multi = read_first_jsonl(fixture.root / "features" / "multi3" /
                                      "FEATURES.jsonl");
  const auto actual = multi.at("features").get<std::vector<double>>();
  const auto prefix_values = prefix.at("features").get<std::vector<double>>();
  check(std::vector<double>(actual.begin(), actual.begin() + 24) ==
            prefix_values,
        "multi3 does not preserve the exact prefix24 vector");

  const auto head = operator_recommend135_v1::extract_features(
      repeated_word(10, window), "int64", bitmap(window, {}), window);
  const auto middle = operator_recommend135_v1::extract_features(
      repeated_word(20, window), "int64", bitmap(window, {}), window);
  const auto tail = operator_recommend135_v1::extract_features(
      repeated_word(30, window), "int64", bitmap(window, {}), window);
  for (std::size_t index = 0; index < 24; ++index) {
    const auto expected_mean = (head[index] + middle[index] + tail[index]) / 3.0;
    const auto expected_range =
        std::max({head[index], middle[index], tail[index]}) -
        std::min({head[index], middle[index], tail[index]});
    check(actual.at(24 + index) == expected_mean,
          "multi3 segment mean differs");
    check(actual.at(48 + index) == expected_range,
          "multi3 segment range differs");
    check(actual.at(72 + index) == tail[index],
          "multi3 tail vector differs");
  }
  check(actual.at(96) == std::log1p(static_cast<double>(3 * window)) &&
            actual.at(97) == std::log1p(static_cast<double>(3 * window)),
        "multi3 whole-column count features differ");
  const auto& windows = multi.at("multi3_sampling").at("windows");
  check(windows.size() == 3 && windows.at(0).at("valid_start") == 0 &&
            windows.at(1).at("valid_start") == window &&
            windows.at(2).at("valid_start") == 2 * window,
        "multi3 did not select head/middle/tail inside the huge block");
}

}  // namespace

int main() {
  try {
    test_first_nonempty_prefix_preserves_null_layout();
    test_selected_region_sha_is_mandatory();
    test_variant_builder_requires_development_source_sha();
    test_formal_scope_gate_rejects_fixture();
    test_development_variants_publish_prefix24_and_multi3_contracts();
    test_confirm_metadata_is_bound_but_confirm_manifest_is_not_opened();
    test_multi3_spans_unequal_sparse_blocks_and_int64_edges();
    test_short_float64_nan_window_is_deduplicated_with_bit_proof();
    test_development_empty_column_is_rejected_explicitly();
    test_multi3_samples_distinct_regions_inside_one_huge_block();
    std::cout << "quality_data_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "quality_data_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
