#include "quality_data.hpp"

#include "../operator_recommend135_v1/common.hpp"
#include "../operator_recommend135_v1/model.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace operator_oracle_quality135_v1 {
namespace {

using Json = nlohmann::json;
using Bytes = std::vector<std::uint8_t>;
namespace fs = std::filesystem;
namespace d = operator_recommend135_v1::detail;

constexpr std::string_view kFrozenOracleSha256 =
    "6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba";
constexpr std::string_view kFrozenColumnsSha256 =
    "6184f8bc0cba374c2e0e05725681d85ddee20e7f5335e9bc751c55009aaabdc8";
constexpr std::size_t kFrozenColumnCount = 3519;
constexpr std::size_t kFrozenDatasetCount = 135;
constexpr std::uint64_t kMaximumRegionBytes = 256ULL << 20;

std::string text(const Json& value, const char* key,
                 const std::string& context) {
  d::require(value.contains(key) && value.at(key).is_string(),
             context + " missing string " + key);
  const auto result = value.at(key).get<std::string>();
  d::require(!result.empty() && result.find('\0') == std::string::npos,
             context + " invalid string " + key);
  return result;
}

std::uint64_t integer(const Json& value, const char* key,
                      const std::string& context) {
  d::require(value.contains(key) &&
                 (value.at(key).is_number_unsigned() ||
                  value.at(key).is_number_integer()),
             context + " missing integer " + key);
  if (value.at(key).is_number_integer()) {
    const auto result = value.at(key).get<std::int64_t>();
    d::require(result >= 0, context + " negative integer " + key);
    return static_cast<std::uint64_t>(result);
  }
  return value.at(key).get<std::uint64_t>();
}

bool valid_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

std::uint64_t add(std::uint64_t left, std::uint64_t right,
                  const std::string& context) {
  d::require(left <= std::numeric_limits<std::uint64_t>::max() - right,
             context + " overflow");
  return left + right;
}

std::uint64_t multiply(std::uint64_t left, std::uint64_t right,
                       const std::string& context) {
  d::require(left == 0 ||
                 right <= std::numeric_limits<std::uint64_t>::max() / left,
             context + " overflow");
  return left * right;
}

std::size_t width(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "float64" || dtype == "int64") return 8;
  throw std::runtime_error("unsupported frozen column dtype " + dtype);
}

fs::path source_path(const fs::path& manifest, const std::string& raw) {
  const fs::path path(raw);
  return (path.is_absolute() ? path : manifest.parent_path() / path)
      .lexically_normal();
}

Bytes read_region(const fs::path& path, std::uint64_t offset,
                  std::uint64_t length) {
  d::require(length <= kMaximumRegionBytes,
             "source region exceeds 256 MiB bound: " + path.string());
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  d::require(descriptor >= 0, "cannot open source region " + path.string());
  try {
    struct stat status {};
    d::require(::fstat(descriptor, &status) == 0 && status.st_size >= 0,
               "cannot stat source region " + path.string());
    d::require(offset <= static_cast<std::uint64_t>(status.st_size) &&
                   length <= static_cast<std::uint64_t>(status.st_size) - offset,
               "source region lies outside file " + path.string());
    d::require(length <= std::numeric_limits<std::size_t>::max(),
               "source region exceeds size_t");
    Bytes result(static_cast<std::size_t>(length));
    std::size_t position = 0;
    while (position < result.size()) {
      const auto count = ::pread(descriptor, result.data() + position,
                                 result.size() - position,
                                 static_cast<off_t>(offset + position));
      if (count < 0 && errno == EINTR) continue;
      d::require(count > 0, "short source region read " + path.string());
      position += static_cast<std::size_t>(count);
    }
    d::require(::close(descriptor) == 0,
               "cannot close source region " + path.string());
    return result;
  } catch (...) {
    ::close(descriptor);
    throw;
  }
}

bool missing_at(const Bytes& bitmap, std::uint64_t row) {
  return ((bitmap.at(static_cast<std::size_t>(row / 8)) >> (row % 8)) & 1U) !=
         0;
}

std::uint64_t count_missing(const Bytes& bitmap, std::uint64_t rows) {
  std::uint64_t count = 0;
  for (std::uint64_t row = 0; row < rows; ++row)
    count += missing_at(bitmap, row) ? 1 : 0;
  return count;
}

void check_padding(const Bytes& bitmap, std::uint64_t rows,
                   const std::string& context) {
  if (rows % 8 == 0 || bitmap.empty()) return;
  const auto used = static_cast<unsigned>(rows % 8);
  const auto mask = static_cast<std::uint8_t>(0xffU << used);
  d::require((bitmap.back() & mask) == 0,
             context + " has set missing-bitmap padding bits");
}

struct BlockSource {
  std::size_t manifest_array_index = 0;
  std::string block_id;
  std::uint64_t block_index = 0;
  std::uint64_t row_count = 0;
  std::uint64_t valid_count = 0;
  std::uint64_t missing_count = 0;
  fs::path values_path;
  std::uint64_t values_offset = 0;
  std::uint64_t values_length = 0;
  std::string values_sha256;
  fs::path missing_path;
  std::uint64_t missing_offset = 0;
  std::uint64_t missing_length = 0;
  std::string missing_sha256;
};

BlockSource parse_block(const Json& item, std::size_t index,
                        const fs::path& manifest_path,
                        const std::string& dataset_id,
                        const std::string& column,
                        const std::string& dtype, std::size_t dtype_width) {
  const auto context = "manifest block " + std::to_string(index) + " in " +
                       manifest_path.string();
  d::require(item.is_object(), context + " is not an object");
  d::require(text(item, "dataset_id", context) == dataset_id,
             context + " dataset_id mismatch");
  d::require(text(item, "column", context) == column,
             context + " column mismatch");
  d::require(text(item, "dtype", context) == dtype,
             context + " dtype mismatch");
  BlockSource result;
  result.manifest_array_index = index;
  result.block_id = text(item, "block_id", context);
  result.block_index = integer(item, "block_index", context);
  result.row_count = integer(item, "row_count", context);
  result.valid_count = integer(item, "valid_count", context);
  result.missing_count = integer(item, "missing_count", context);
  d::require(result.valid_count <= result.row_count &&
                 result.missing_count == result.row_count - result.valid_count,
             context + " row/valid/missing counts differ");
  result.values_path =
      source_path(manifest_path, text(item, "values_path", context));
  result.values_offset = integer(item, "values_offset", context);
  result.values_length = integer(item, "values_length", context);
  result.values_sha256 = text(item, "values_sha256", context);
  result.missing_path = source_path(
      manifest_path, text(item, "missing_bitmap_path", context));
  result.missing_offset = integer(item, "missing_bitmap_offset", context);
  result.missing_length = integer(item, "missing_bitmap_length", context);
  result.missing_sha256 = text(item, "missing_bitmap_sha256", context);
  d::require(valid_sha256(result.values_sha256) &&
                 valid_sha256(result.missing_sha256),
             context + " invalid source SHA256");
  d::require(result.values_length ==
                 multiply(result.valid_count, dtype_width,
                          context + " numeric byte count"),
             context + " values_length mismatch");
  d::require(result.missing_length == (result.row_count + 7) / 8,
             context + " missing_bitmap_length mismatch");
  return result;
}

struct Sample {
  std::string dtype;
  std::uint64_t row_count = 0;
  Bytes values;
  Bytes missing;
  std::uint64_t row_end = 0;
  bool truncated = false;
};

struct TimeTotal {
  std::uint64_t cpu_ns = 0;
  std::uint64_t wall_ns = 0;
};

TimeTotal now() { return {d::cpu_ns(), d::wall_ns()}; }

TimeTotal elapsed(const TimeTotal& start) {
  return {d::cpu_ns() - start.cpu_ns, d::wall_ns() - start.wall_ns};
}

void accumulate(TimeTotal& total, const TimeTotal& value,
                const std::string& context) {
  total.cpu_ns = add(total.cpu_ns, value.cpu_ns, context + " cpu");
  total.wall_ns = add(total.wall_ns, value.wall_ns, context + " wall");
}

Json timing(const TimeTotal& value) {
  return {{"cpu_ns", value.cpu_ns}, {"wall_ns", value.wall_ns}};
}

Sample prefix_sample(const BlockSource& source, const std::string& dtype,
                     std::size_t dtype_width, std::size_t maximum_values,
                     std::uint64_t& verified_bytes) {
  auto values = read_region(source.values_path, source.values_offset,
                            source.values_length);
  auto missing = read_region(source.missing_path, source.missing_offset,
                             source.missing_length);
  verified_bytes = add(verified_bytes, values.size(), "verified source bytes");
  verified_bytes = add(verified_bytes, missing.size(), "verified source bytes");
  d::require(d::sha(values) == source.values_sha256,
             "selected source values SHA256 mismatch");
  d::require(d::sha(missing) == source.missing_sha256,
             "selected source missing bitmap SHA256 mismatch");
  check_padding(missing, source.row_count, "selected source block");
  d::require(count_missing(missing, source.row_count) == source.missing_count,
             "selected source missing count mismatch");

  const auto take = std::min<std::uint64_t>(source.valid_count, maximum_values);
  d::require(take > 0, "selected feature block is empty");
  std::uint64_t row_end = source.row_count;
  if (take < source.valid_count) {
    std::uint64_t seen = 0;
    row_end = 0;
    for (std::uint64_t row = 0; row < source.row_count; ++row) {
      if (!missing_at(missing, row)) ++seen;
      if (seen == take) {
        row_end = row + 1;
        break;
      }
    }
    d::require(row_end > 0, "cannot locate feature sample row boundary");
  }

  values.resize(static_cast<std::size_t>(take * dtype_width));
  missing.resize(static_cast<std::size_t>((row_end + 7) / 8));
  if (row_end % 8 != 0) {
    const auto used = static_cast<unsigned>(row_end % 8);
    missing.back() &= static_cast<std::uint8_t>((1U << used) - 1U);
  }
  d::require(count_missing(missing, row_end) == row_end - take,
             "feature sample bitmap/value count mismatch");
  return {dtype, row_end, std::move(values), std::move(missing), row_end,
          take < source.valid_count};
}

void write_immutable(const fs::path& output, const std::string& bytes) {
  if (fs::exists(output)) {
    const auto current = d::read_bytes(output);
    const std::string current_text(current.begin(), current.end());
    d::require(current_text == bytes,
               "refusing to replace changed feature output " + output.string());
    return;
  }
  d::write_text(output, bytes);
}

}  // namespace

Json build_features(const fs::path& columns_json, const fs::path& output_jsonl,
                    const FeatureBuildOptions& options) {
  const auto total_start = now();
  TimeTotal source_time;
  TimeTotal feature_time;
  TimeTotal serialization_time;
  TimeTotal output_commit_time;
  auto source_start = now();
  d::require(options.max_valid_values > 0 &&
                 options.max_valid_values <= 32768,
             "max_valid_values must be in [1,32768]");
  const auto columns_path = fs::absolute(columns_json).lexically_normal();
  const auto output_path = fs::absolute(output_jsonl).lexically_normal();
  const auto columns_sha256 = d::file_sha(columns_path);
  const auto root = d::read_json(columns_path);
  d::require(root.is_object() &&
                 root.value("schema", "") ==
                     "operator-prepared135-source-columns-v1",
             "unexpected COLUMNS schema");
  d::require(root.contains("columns") && root.at("columns").is_array(),
             "COLUMNS lacks columns array");
  const auto oracle_sha256 =
      text(root, "oracle_reference_jsonl_sha256", "COLUMNS");
  d::require(valid_sha256(oracle_sha256), "COLUMNS oracle SHA256 is invalid");
  if (options.require_frozen_current135) {
    d::require(root.at("columns").size() == kFrozenColumnCount,
               "frozen current135 column count differs from 3519");
    d::require(columns_sha256 == kFrozenColumnsSha256,
               "COLUMNS bytes differ from the frozen current135 source list");
    d::require(oracle_sha256 == kFrozenOracleSha256,
               "COLUMNS is not bound to the frozen oracle");
  }

  const auto names = operator_recommend135_v1::feature_names();
  const auto feature_names_sha256 = d::sha(Json(names).dump());
  accumulate(source_time, elapsed(source_start), "initial source validation");
  std::set<std::string> keys;
  std::set<std::string> datasets;
  std::string output;
  std::uint64_t verified_source_bytes = 0;
  std::uint64_t verified_manifests = 0;
  std::uint64_t truncated_samples = 0;

  for (std::size_t ci = 0; ci < root.at("columns").size(); ++ci) {
    source_start = now();
    const auto& column_row = root.at("columns").at(ci);
    const auto context = "COLUMNS row " + std::to_string(ci);
    d::require(column_row.is_object(), context + " is not an object");
    const auto dataset_id = text(column_row, "dataset_id", context);
    const auto column_id = text(column_row, "column_id", context);
    const auto column = text(column_row, "column", context);
    const auto dtype = text(column_row, "dtype", context);
    d::require(valid_sha256(column_id), context + " column_id is not SHA256");
    const auto dtype_width = width(dtype);
    d::require(keys.insert(dataset_id + ":" + column_id).second,
               context + " duplicate dataset/column identity");
    datasets.insert(dataset_id);
    const auto original_numeric_bytes =
        integer(column_row, "original_numeric_bytes", context);
    const auto valid_value_count =
        integer(column_row, "valid_value_count", context);
    d::require(original_numeric_bytes ==
                   multiply(valid_value_count, dtype_width,
                            context + " column numeric bytes"),
               context + " original_numeric_bytes mismatch");
    const auto manifest_path = fs::path(
        text(column_row, "manifest_path", context)).lexically_normal();
    const auto manifest_sha256 =
        text(column_row, "manifest_sha256", context);
    d::require(valid_sha256(manifest_sha256),
               context + " manifest SHA256 is invalid");
    d::require(d::file_sha(manifest_path) == manifest_sha256,
               context + " manifest SHA256 mismatch");
    ++verified_manifests;
    const auto manifest = d::read_json(manifest_path);
    d::require(manifest.is_object() &&
                   manifest.value("schema", "") ==
                       "native-baseline86-scaled-column-segment-manifest-v1",
               context + " unexpected manifest schema");
    d::require(text(manifest, "dataset_id", context + " manifest") == dataset_id,
               context + " manifest dataset mismatch");
    d::require(manifest_path.stem().string() == column_id,
               context + " manifest filename/column_id mismatch");
    d::require(manifest.contains("column_identity") &&
                   manifest.at("column_identity").is_object(),
               context + " manifest lacks column_identity");
    const auto& identity = manifest.at("column_identity");
    d::require(text(identity, "dataset_id", context + " identity") == dataset_id &&
                   text(identity, "column", context + " identity") == column &&
                   text(identity, "dtype", context + " identity") == dtype,
               context + " manifest column identity mismatch");
    d::require(manifest.contains("blocks") && manifest.at("blocks").is_array() &&
                   !manifest.at("blocks").empty(),
               context + " manifest has no blocks");

    std::vector<BlockSource> blocks;
    blocks.reserve(manifest.at("blocks").size());
    std::uint64_t manifest_values = 0;
    std::uint64_t manifest_valid = 0;
    const BlockSource* selected = nullptr;
    for (std::size_t bi = 0; bi < manifest.at("blocks").size(); ++bi) {
      blocks.push_back(parse_block(manifest.at("blocks").at(bi), bi,
                                   manifest_path, dataset_id, column, dtype,
                                   dtype_width));
      const auto& block = blocks.back();
      manifest_values = add(manifest_values, block.values_length,
                            context + " manifest values");
      manifest_valid = add(manifest_valid, block.valid_count,
                           context + " manifest valid count");
      if (selected == nullptr && block.valid_count > 0) selected = &blocks.back();
    }
    // Vector growth can invalidate the pointer retained above. Resolve the
    // first nonempty block again after the complete metadata pass.
    selected = nullptr;
    for (const auto& block : blocks)
      if (block.valid_count > 0) {
        selected = &block;
        break;
      }
    d::require(manifest_values == original_numeric_bytes &&
                   manifest_valid == valid_value_count,
               context + " whole-column manifest totals mismatch");
    d::require(selected != nullptr, context + " column has no nonempty block");

    auto sample = prefix_sample(*selected, dtype, dtype_width,
                                options.max_valid_values,
                                verified_source_bytes);
    accumulate(source_time, elapsed(source_start), "column source validation");
    const auto feature_start = now();
    const auto features = operator_recommend135_v1::extract_features(
        sample.values, sample.dtype, sample.missing, sample.row_count);
    accumulate(feature_time, elapsed(feature_start), "feature extraction");
    d::require(features.size() == names.size(),
               context + " feature shape mismatch");
    if (sample.truncated) ++truncated_samples;
    const auto sample_valid = sample.values.size() / dtype_width;
    const auto sample_missing = sample.row_count - sample_valid;

    const auto serialization_start = now();
    Json row{
        {"schema", "operator-oracle-quality135-feature-row-v1"},
        {"dataset_id", dataset_id},
        {"column_id", column_id},
        {"column", column},
        {"dtype", dtype},
        {"input_dtype", dtype},
        {"columns_sha256", columns_sha256},
        {"oracle_reference_jsonl_sha256", oracle_sha256},
        {"feature_contract",
         "operator_recommend135_v1::extract_features/24/first-nonempty-prefix-v1"},
        {"feature_names_sha256", feature_names_sha256},
        {"features", features},
        {"column_source",
         {{"manifest_path", manifest_path.string()},
          {"manifest_sha256", manifest_sha256},
          {"original_numeric_bytes", original_numeric_bytes},
          {"valid_value_count", valid_value_count},
          {"manifest_block_count", blocks.size()}}},
        {"sample_source",
         {{"selection", "first_nonempty_original_block_prefix"},
          {"manifest_array_index", selected->manifest_array_index},
          {"block_id", selected->block_id},
          {"block_index", selected->block_index},
          {"source_row_start", 0},
          {"source_row_end_exclusive", sample.row_end},
          {"source_block_row_count", selected->row_count},
          {"source_block_valid_count", selected->valid_count},
          {"values_path", selected->values_path.string()},
          {"values_offset", selected->values_offset},
          {"values_length", selected->values_length},
          {"values_sha256", selected->values_sha256},
          {"missing_bitmap_path", selected->missing_path.string()},
          {"missing_bitmap_offset", selected->missing_offset},
          {"missing_bitmap_length", selected->missing_length},
          {"missing_bitmap_sha256", selected->missing_sha256},
          {"full_selected_source_regions_sha256_verified", true}}},
        {"sample",
         {{"row_count", sample.row_count},
          {"valid_count", sample_valid},
          {"missing_count", sample_missing},
          {"values_bytes", sample.values.size()},
          {"missing_bytes", sample.missing.size()},
          {"values_sha256", d::sha(sample.values)},
          {"missing_sha256", d::sha(sample.missing)},
          {"max_valid_values", options.max_valid_values},
          {"truncated", sample.truncated}}},
        {"oracle_labels_read", false}};
    output += row.dump();
    output.push_back('\n');
    accumulate(serialization_time, elapsed(serialization_start),
               "feature row serialization");
  }

  if (options.require_frozen_current135)
    d::require(datasets.size() == kFrozenDatasetCount,
               "frozen current135 dataset count differs from 135");
  const auto commit_start = now();
  write_immutable(output_path, output);
  accumulate(output_commit_time, elapsed(commit_start), "feature output commit");
  const auto total_time = elapsed(total_start);
  return {{"schema", "operator-oracle-quality135-features-receipt-v1"},
          {"status", "complete"},
          {"columns_path", columns_path.string()},
          {"columns_sha256", columns_sha256},
          {"oracle_reference_jsonl_sha256", oracle_sha256},
          {"output_path", output_path.string()},
          {"output_sha256", d::sha(output)},
          {"feature_contract",
           "operator_recommend135_v1::extract_features/24/first-nonempty-prefix-v1"},
          {"feature_names", names},
          {"feature_names_sha256", feature_names_sha256},
          {"max_valid_values", options.max_valid_values},
          {"column_count", root.at("columns").size()},
          {"dataset_count", datasets.size()},
          {"verified_manifest_count", verified_manifests},
          {"verified_selected_source_region_bytes", verified_source_bytes},
          {"truncated_sample_count", truncated_samples},
          {"timing_contract",
           {{"clocks", Json::array({"CLOCK_THREAD_CPUTIME_ID", "steady_clock"})},
            {"source_validation_and_read",
             "COLUMNS read/hash/schema gates, manifest read/hash/metadata gates, selected full source-region read/hash, bitmap validation and prefix slicing"},
            {"feature_extraction",
             "operator_recommend135_v1::extract_features only"},
            {"jsonl_serialization", "feature-row JSON construction and canonical compact dump"},
            {"output_commit", "immutable output comparison or write, fsync and rename"},
            {"source_file_io_included", true},
            {"oracle_label_io_included", false}}},
          {"timings",
           {{"source_validation_and_read", timing(source_time)},
            {"feature_extraction", timing(feature_time)},
            {"jsonl_serialization", timing(serialization_time)},
            {"output_commit", timing(output_commit_time)},
            {"total", timing(total_time)}}},
          {"oracle_labels_read", false}};
}

}  // namespace operator_oracle_quality135_v1
