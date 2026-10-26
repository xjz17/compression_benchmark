#include "quality_data.hpp"

#include "../operator_recommend135_v1/common.hpp"
#include "../operator_recommend135_v1/model.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
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
constexpr std::size_t kFrozenDevelopmentColumnCount = 3433;
constexpr std::size_t kFrozenDevelopmentDatasetCount = 127;
constexpr std::size_t kFrozenConfirmColumnCount = 86;
constexpr std::string_view kFrozenInputMetadataSha256 =
    "55d94e4db2f36ce1d2a31d10dad1f5c93c55464445e8a0a0386bddc45768d03b";
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

struct LoadedBlock {
  Bytes values;
  Bytes missing;
};

using BlockCache = std::vector<std::optional<LoadedBlock>>;

const LoadedBlock& load_block(const std::vector<BlockSource>& blocks,
                              std::size_t index, BlockCache& cache,
                              std::uint64_t& verified_bytes) {
  d::require(index < blocks.size() && index < cache.size(),
             "block cache index is outside the manifest");
  if (!cache[index].has_value()) {
    const auto& source = blocks[index];
    auto values = read_region(source.values_path, source.values_offset,
                              source.values_length);
    auto missing = read_region(source.missing_path, source.missing_offset,
                               source.missing_length);
    verified_bytes = add(verified_bytes, values.size(),
                         "verified unique source bytes");
    verified_bytes = add(verified_bytes, missing.size(),
                         "verified unique source bytes");
    d::require(d::sha(values) == source.values_sha256,
               "selected source values SHA256 mismatch");
    d::require(d::sha(missing) == source.missing_sha256,
               "selected source missing bitmap SHA256 mismatch");
    check_padding(missing, source.row_count, "selected source block");
    d::require(count_missing(missing, source.row_count) ==
                   source.missing_count,
               "selected source missing count mismatch");
    cache[index] = LoadedBlock{std::move(values), std::move(missing)};
  }
  return cache[index].value();
}

std::uint64_t row_of_valid_ordinal(const Bytes& missing,
                                   std::uint64_t row_count,
                                   std::uint64_t ordinal) {
  std::uint64_t valid_before = 0;
  for (std::uint64_t byte_index = 0; byte_index < missing.size(); ++byte_index) {
    const auto first_row = byte_index * 8;
    const auto rows_in_byte =
        std::min<std::uint64_t>(8, row_count - first_row);
    const auto used_mask = rows_in_byte == 8
                               ? std::uint8_t{0xff}
                               : static_cast<std::uint8_t>((1U << rows_in_byte) - 1U);
    const auto missing_count =
        static_cast<std::uint64_t>(std::popcount(
            static_cast<unsigned>(missing[byte_index] & used_mask)));
    const auto valid_in_byte = rows_in_byte - missing_count;
    if (ordinal < valid_before + valid_in_byte) {
      for (std::uint64_t bit = 0; bit < rows_in_byte; ++bit) {
        if (((missing[byte_index] >> bit) & 1U) == 0) {
          if (valid_before == ordinal) return first_row + bit;
          ++valid_before;
        }
      }
    }
    valid_before += valid_in_byte;
  }
  throw std::runtime_error("valid-value ordinal is outside source block");
}

void append_missing_range(Bytes& output, std::uint64_t& output_rows,
                          const Bytes& source, std::uint64_t row_start,
                          std::uint64_t row_end) {
  d::require(row_start <= row_end, "invalid sampled row range");
  const auto new_rows = add(output_rows, row_end - row_start,
                            "sampled row count");
  output.resize(static_cast<std::size_t>((new_rows + 7) / 8), 0);
  for (std::uint64_t row = row_start; row < row_end; ++row) {
    if (missing_at(source, row)) {
      const auto target = output_rows + (row - row_start);
      output.at(static_cast<std::size_t>(target / 8)) |=
          static_cast<std::uint8_t>(1U << (target % 8));
    }
  }
  output_rows = new_rows;
}

struct SampledWindow {
  Sample sample;
  Json proof;
};

SampledWindow sample_valid_window(const std::vector<BlockSource>& blocks,
                                  BlockCache& cache,
                                  const std::string& dtype,
                                  std::size_t dtype_width,
                                  std::uint64_t valid_start,
                                  std::uint64_t valid_end,
                                  const std::vector<std::string>& roles,
                                  std::uint64_t whole_valid_count,
                                  std::uint64_t& verified_bytes) {
  d::require(valid_start < valid_end && valid_end <= whole_valid_count,
             "invalid whole-column valid-value window");
  std::uint64_t cumulative = 0;
  std::size_t first = blocks.size();
  std::size_t last = blocks.size();
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    const auto next = add(cumulative, blocks[index].valid_count,
                          "whole-column valid coordinates");
    if (blocks[index].valid_count > 0 && valid_start < next &&
        valid_end > cumulative) {
      if (first == blocks.size()) first = index;
      last = index;
    }
    cumulative = next;
  }
  d::require(first < blocks.size() && last < blocks.size(),
             "cannot locate whole-column valid-value window");

  Bytes sampled_values;
  Bytes sampled_missing;
  std::uint64_t sampled_rows = 0;
  std::uint64_t before_block = 0;
  for (std::size_t index = 0; index < first; ++index)
    before_block = add(before_block, blocks[index].valid_count,
                       "window block valid offset");
  Json segments = Json::array();
  for (std::size_t index = first; index <= last; ++index) {
    const auto& block = blocks[index];
    const auto block_valid_start = before_block;
    const auto block_valid_end = add(block_valid_start, block.valid_count,
                                     "window block valid end");
    const auto local_valid_start =
        index == first ? valid_start - block_valid_start : 0;
    const auto local_valid_end =
        index == last ? valid_end - block_valid_start : block.valid_count;
    d::require(local_valid_start <= local_valid_end &&
                   local_valid_end <= block.valid_count,
               "invalid block-local valid-value window");
    const auto& loaded = load_block(blocks, index, cache, verified_bytes);
    const std::uint64_t row_start =
        local_valid_start == 0
            ? 0
            : row_of_valid_ordinal(loaded.missing, block.row_count,
                                   local_valid_start);
    const std::uint64_t row_end =
        local_valid_end == block.valid_count
            ? block.row_count
            : row_of_valid_ordinal(loaded.missing, block.row_count,
                                   local_valid_end - 1) +
                  1;
    const auto byte_start = multiply(local_valid_start, dtype_width,
                                     "sample values byte start");
    const auto byte_end = multiply(local_valid_end, dtype_width,
                                   "sample values byte end");
    sampled_values.insert(
        sampled_values.end(), loaded.values.begin() + byte_start,
        loaded.values.begin() + byte_end);
    append_missing_range(sampled_missing, sampled_rows, loaded.missing,
                         row_start, row_end);
    segments.push_back(
        {{"manifest_array_index", block.manifest_array_index},
         {"block_id", block.block_id},
         {"block_index", block.block_index},
         {"block_valid_start", local_valid_start},
         {"block_valid_end_exclusive", local_valid_end},
         {"source_row_start", row_start},
         {"source_row_end_exclusive", row_end},
         {"values_path", block.values_path.string()},
         {"values_region_offset", block.values_offset},
         {"values_region_length", block.values_length},
         {"values_region_sha256", block.values_sha256},
         {"missing_bitmap_path", block.missing_path.string()},
         {"missing_region_offset", block.missing_offset},
         {"missing_region_length", block.missing_length},
         {"missing_region_sha256", block.missing_sha256},
         {"full_source_regions_sha256_verified", true}});
    before_block = block_valid_end;
  }
  const auto expected_valid = valid_end - valid_start;
  d::require(sampled_values.size() ==
                 multiply(expected_valid, dtype_width,
                          "sampled values byte count") &&
                 count_missing(sampled_missing, sampled_rows) ==
                     sampled_rows - expected_valid,
             "sampled window value/null shape differs");
  Sample sample{dtype,
                sampled_rows,
                std::move(sampled_values),
                std::move(sampled_missing),
                sampled_rows,
                expected_valid < whole_valid_count};
  Json proof{{"roles", roles},
             {"valid_start", valid_start},
             {"valid_end_exclusive", valid_end},
             {"valid_count", expected_valid},
             {"row_count", sampled_rows},
             {"missing_count", sampled_rows - expected_valid},
             {"values_sha256", d::sha(sample.values)},
             {"missing_sha256", d::sha(sample.missing)},
             {"segments", std::move(segments)}};
  return {std::move(sample), std::move(proof)};
}

Sample prefix_sample_cached(const std::vector<BlockSource>& blocks,
                            std::size_t selected_index, BlockCache& cache,
                            const std::string& dtype,
                            std::size_t dtype_width,
                            std::size_t maximum_values,
                            std::uint64_t& verified_bytes) {
  const auto& source = blocks.at(selected_index);
  const auto& loaded = load_block(blocks, selected_index, cache, verified_bytes);
  const auto take = std::min<std::uint64_t>(source.valid_count, maximum_values);
  d::require(take > 0, "selected feature block is empty");
  std::uint64_t row_end = source.row_count;
  if (take < source.valid_count)
    row_end = row_of_valid_ordinal(loaded.missing, source.row_count, take - 1) +
              1;
  Bytes values(loaded.values.begin(),
               loaded.values.begin() + multiply(take, dtype_width,
                                                 "prefix values bytes"));
  Bytes missing(loaded.missing.begin(),
                loaded.missing.begin() + (row_end + 7) / 8);
  if (row_end % 8 != 0)
    missing.back() &=
        static_cast<std::uint8_t>((1U << (row_end % 8)) - 1U);
  d::require(count_missing(missing, row_end) == row_end - take,
             "prefix sample bitmap/value count mismatch");
  return {dtype, row_end, std::move(values), std::move(missing), row_end,
          take < source.valid_count};
}

struct WindowPlan {
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  std::vector<std::string> roles;
};

std::vector<WindowPlan> plan_windows(std::uint64_t whole_valid_count,
                                     std::size_t maximum_values) {
  d::require(whole_valid_count > 0, "cannot sample an empty column");
  const auto take = std::min<std::uint64_t>(whole_valid_count, maximum_values);
  const auto span = whole_valid_count - take;
  const std::array<std::pair<std::string_view, std::uint64_t>, 3> requests{{
      {"head", 0}, {"middle", span / 2}, {"tail", span}}};
  std::vector<WindowPlan> result;
  std::map<std::uint64_t, std::size_t> by_start;
  for (const auto& [role, start] : requests) {
    const auto found = by_start.find(start);
    if (found == by_start.end()) {
      by_start.emplace(start, result.size());
      result.push_back({start, start + take, {std::string(role)}});
    } else {
      result[found->second].roles.push_back(std::string(role));
    }
  }
  return result;
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
        {"schema", "operator-oracle-risk135-feature-row-v1"},
        {"dataset_id", dataset_id},
        {"column_id", column_id},
        {"column", column},
        {"dtype", dtype},
        {"input_dtype", dtype},
        {"columns_sha256", columns_sha256},
        {"oracle_reference_jsonl_sha256", oracle_sha256},
        {"feature_contract",
         "operator_recommend135_v1::extract_features/24/first-nonempty-prefix-v1"},
        {"feature_variant", "prefix24"},
        {"feature_names", names},
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

std::vector<std::string> prefix24_feature_names() {
  return operator_recommend135_v1::feature_names();
}

std::vector<std::string> multi3_feature_names() {
  const auto base = prefix24_feature_names();
  std::vector<std::string> result;
  result.reserve(base.size() * 4 + 2);
  for (const auto& name : base) result.push_back("prefix." + name);
  for (const auto& name : base) result.push_back("segment_mean." + name);
  for (const auto& name : base) result.push_back("segment_range." + name);
  for (const auto& name : base) result.push_back("segment_tail." + name);
  result.push_back("log1p_column_valid_count");
  result.push_back("log1p_column_row_count");
  return result;
}

Json build_development_feature_variants(
    const fs::path& columns_json, const fs::path& input_metadata_json,
    const fs::path& output_directory,
    const FeatureVariantBuildOptions& options) {
  const auto total_start = now();
  TimeTotal source_time;
  TimeTotal feature_time;
  TimeTotal serialization_time;
  TimeTotal output_commit_time;
  d::require(options.max_valid_values_per_window > 0 &&
                 options.max_valid_values_per_window <= 32768,
             "max_valid_values_per_window must be in [1,32768]");
  const auto columns_path = fs::absolute(columns_json).lexically_normal();
  const auto metadata_path =
      fs::absolute(input_metadata_json).lexically_normal();
  const auto root_path = fs::absolute(output_directory).lexically_normal();
  const auto prefix_dir = root_path / "prefix24";
  const auto multi_dir = root_path / "multi3";
  const auto prefix_output = prefix_dir / "FEATURES.jsonl";
  const auto prefix_receipt_path = prefix_dir / "FEATURES_RECEIPT.json";
  const auto multi_output_path = multi_dir / "FEATURES.jsonl";
  const auto multi_receipt_path = multi_dir / "FEATURES_RECEIPT.json";
  for (const auto& output : {prefix_output, prefix_receipt_path,
                             multi_output_path, multi_receipt_path})
    d::require(!fs::exists(output),
               "refusing to overwrite feature artifact " + output.string());

  auto source_start = now();
  const auto columns_sha256 = d::file_sha(columns_path);
  const auto input_metadata_sha256 = d::file_sha(metadata_path);
  const auto columns_root = d::read_json(columns_path);
  d::require(columns_root.is_object() &&
                 columns_root.value("schema", "") ==
                     "operator-prepared135-source-columns-v1" &&
                 columns_root.contains("columns") &&
                 columns_root.at("columns").is_array(),
             "unexpected COLUMNS schema");
  const auto oracle_sha256 = text(
      columns_root, "oracle_reference_jsonl_sha256", "COLUMNS");
  d::require(valid_sha256(oracle_sha256), "COLUMNS oracle SHA256 is invalid");
  const auto metadata = d::read_json(metadata_path);
  d::require(metadata.is_object() &&
                 metadata.value("schema", "") ==
                     "operator-oracle-quality135-input-metadata-v1" &&
                 metadata.contains("columns") &&
                 metadata.at("columns").is_array(),
             "unexpected INPUT_METADATA schema");
  d::require(metadata.value("oracle_sha256", "") == oracle_sha256,
             "INPUT_METADATA oracle SHA256 differs from COLUMNS");
  if (options.require_frozen_current135) {
    d::require(columns_sha256 == kFrozenColumnsSha256,
               "COLUMNS bytes differ from the frozen current135 source list");
    d::require(input_metadata_sha256 == kFrozenInputMetadataSha256,
               "INPUT_METADATA bytes differ from the frozen current135 split metadata");
    d::require(oracle_sha256 == kFrozenOracleSha256,
               "COLUMNS is not bound to the frozen oracle");
    d::require(columns_root.at("columns").size() == kFrozenColumnCount &&
                   metadata.at("columns").size() == kFrozenColumnCount,
               "frozen current135 metadata count differs from 3519");
  }

  std::map<std::string, const Json*> metadata_by_key;
  std::size_t development_count = 0;
  std::size_t confirm_count = 0;
  std::set<std::string> development_datasets;
  for (std::size_t index = 0; index < metadata.at("columns").size(); ++index) {
    const auto& row = metadata.at("columns").at(index);
    const auto context = "INPUT_METADATA row " + std::to_string(index);
    d::require(row.is_object(), context + " is not an object");
    const auto dataset_id = text(row, "dataset_id", context);
    const auto column_id = text(row, "column_id", context);
    d::require(text(row, "sample_id", context) == column_id,
               context + " sample_id/column_id mismatch");
    const auto split = text(row, "split", context);
    d::require(split == "development" || split == "confirm",
               context + " has unsupported split");
    d::require(metadata_by_key.emplace(dataset_id + ":" + column_id, &row)
                   .second,
               context + " duplicates a dataset/column key");
    if (split == "development") {
      ++development_count;
      development_datasets.insert(dataset_id);
      d::require(row.contains("dev_fold") &&
                     row.at("dev_fold").is_number_integer() &&
                     row.at("dev_fold").get<int>() >= 0 &&
                     row.at("dev_fold").get<int>() < 5,
                 context + " has invalid development fold");
    } else {
      ++confirm_count;
    }
  }
  if (options.require_frozen_current135) {
    d::require(development_count == kFrozenDevelopmentColumnCount &&
                   confirm_count == kFrozenConfirmColumnCount &&
                   development_datasets.size() ==
                       kFrozenDevelopmentDatasetCount,
               "frozen development/confirm scope differs from 3433/86 and 127 datasets");
  }

  std::vector<const Json*> development_columns;
  development_columns.reserve(development_count);
  std::set<std::string> column_keys;
  for (std::size_t index = 0; index < columns_root.at("columns").size();
       ++index) {
    const auto& row = columns_root.at("columns").at(index);
    const auto context = "COLUMNS row " + std::to_string(index);
    d::require(row.is_object(), context + " is not an object");
    const auto dataset_id = text(row, "dataset_id", context);
    const auto column_id = text(row, "column_id", context);
    const auto key = dataset_id + ":" + column_id;
    d::require(column_keys.insert(key).second,
               context + " duplicates a dataset/column key");
    const auto found = metadata_by_key.find(key);
    d::require(found != metadata_by_key.end(),
               context + " is absent from INPUT_METADATA");
    const auto& metadata_row = *found->second;
    for (const auto* field : {"dtype", "manifest_path", "manifest_sha256",
                              "original_numeric_bytes", "valid_value_count"})
      d::require(row.contains(field) && metadata_row.contains(field) &&
                     row.at(field) == metadata_row.at(field),
                 context + " differs from INPUT_METADATA field " + field);
    if (metadata_row.at("split") == "development")
      development_columns.push_back(&row);
  }
  d::require(column_keys.size() == metadata_by_key.size() &&
                 development_columns.size() == development_count,
             "COLUMNS and INPUT_METADATA key sets differ");
  accumulate(source_time, elapsed(source_start), "input scope validation");

  fs::create_directories(prefix_dir);
  fs::create_directories(multi_dir);
  const auto prefix_names = prefix24_feature_names();
  const auto multi_names = multi3_feature_names();
  const auto prefix_names_sha256 = d::sha(Json(prefix_names).dump());
  const auto multi_names_sha256 = d::sha(Json(multi_names).dump());
  std::string prefix_output_bytes;
  std::string multi_output;
  std::uint64_t verified_source_bytes = 0;
  std::uint64_t verified_manifests = 0;
  std::uint64_t prefix_truncated_count = 0;
  std::uint64_t multi_unique_window_count = 0;

  for (std::size_t ci = 0; ci < development_columns.size(); ++ci) {
    source_start = now();
    const auto& column_row = *development_columns[ci];
    const auto context = "development COLUMNS row " + std::to_string(ci);
    const auto dataset_id = text(column_row, "dataset_id", context);
    const auto column_id = text(column_row, "column_id", context);
    const auto column = text(column_row, "column", context);
    const auto dtype = text(column_row, "dtype", context);
    const auto dtype_width = width(dtype);
    const auto original_numeric_bytes =
        integer(column_row, "original_numeric_bytes", context);
    const auto valid_value_count =
        integer(column_row, "valid_value_count", context);
    d::require(valid_value_count > 0 &&
                   original_numeric_bytes ==
                       multiply(valid_value_count, dtype_width,
                                context + " column numeric bytes"),
               context + " has empty or inconsistent numeric values");
    const auto manifest_path = fs::path(
        text(column_row, "manifest_path", context)).lexically_normal();
    const auto manifest_sha256 =
        text(column_row, "manifest_sha256", context);
    d::require(valid_sha256(manifest_sha256) &&
                   d::file_sha(manifest_path) == manifest_sha256,
               context + " manifest SHA256 mismatch");
    ++verified_manifests;
    const auto manifest = d::read_json(manifest_path);
    d::require(manifest.is_object() &&
                   manifest.value("schema", "") ==
                       "native-baseline86-scaled-column-segment-manifest-v1",
               context + " unexpected manifest schema");
    d::require(text(manifest, "dataset_id", context + " manifest") ==
                   dataset_id &&
                   manifest_path.stem().string() == column_id,
               context + " manifest identity mismatch");
    d::require(manifest.contains("column_identity") &&
                   manifest.at("column_identity").is_object(),
               context + " manifest lacks column_identity");
    const auto& identity = manifest.at("column_identity");
    d::require(text(identity, "dataset_id", context + " identity") ==
                       dataset_id &&
                   text(identity, "column", context + " identity") == column &&
                   text(identity, "dtype", context + " identity") == dtype,
               context + " manifest column identity mismatch");
    d::require(manifest.contains("blocks") &&
                   manifest.at("blocks").is_array() &&
                   !manifest.at("blocks").empty(),
               context + " manifest has no blocks");

    std::vector<BlockSource> blocks;
    blocks.reserve(manifest.at("blocks").size());
    std::uint64_t manifest_values = 0;
    std::uint64_t manifest_valid = 0;
    std::uint64_t whole_row_count = 0;
    std::size_t prefix_block = manifest.at("blocks").size();
    for (std::size_t bi = 0; bi < manifest.at("blocks").size(); ++bi) {
      blocks.push_back(parse_block(manifest.at("blocks").at(bi), bi,
                                   manifest_path, dataset_id, column, dtype,
                                   dtype_width));
      const auto& block = blocks.back();
      manifest_values = add(manifest_values, block.values_length,
                            context + " manifest values");
      manifest_valid = add(manifest_valid, block.valid_count,
                           context + " manifest valid count");
      whole_row_count = add(whole_row_count, block.row_count,
                            context + " manifest row count");
      if (prefix_block == manifest.at("blocks").size() &&
          block.valid_count > 0)
        prefix_block = bi;
    }
    d::require(manifest_values == original_numeric_bytes &&
                   manifest_valid == valid_value_count,
               context + " whole-column manifest totals mismatch");
    d::require(prefix_block < blocks.size(),
               context + " column has no nonempty block");
    BlockCache cache(blocks.size());
    auto prefix_sample = prefix_sample_cached(
        blocks, prefix_block, cache, dtype, dtype_width,
        options.max_valid_values_per_window, verified_source_bytes);
    const auto plans =
        plan_windows(valid_value_count, options.max_valid_values_per_window);
    std::vector<SampledWindow> windows;
    windows.reserve(plans.size());
    for (const auto& plan : plans)
      windows.push_back(sample_valid_window(
          blocks, cache, dtype, dtype_width, plan.start, plan.end, plan.roles,
          valid_value_count, verified_source_bytes));
    accumulate(source_time, elapsed(source_start),
               "development source validation and sampling");

    const auto feature_start = now();
    const auto prefix_features = operator_recommend135_v1::extract_features(
        prefix_sample.values, dtype, prefix_sample.missing,
        prefix_sample.row_count);
    std::vector<std::vector<double>> window_features;
    window_features.reserve(windows.size());
    std::size_t tail_window_index = windows.size();
    for (std::size_t index = 0; index < windows.size(); ++index) {
      const auto& sample = windows[index].sample;
      window_features.push_back(operator_recommend135_v1::extract_features(
          sample.values, dtype, sample.missing, sample.row_count));
      const auto& roles = plans[index].roles;
      if (std::find(roles.begin(), roles.end(), "tail") != roles.end())
        tail_window_index = index;
    }
    d::require(prefix_features.size() == 24 && !window_features.empty() &&
                   tail_window_index < window_features.size(),
               context + " feature/window shape mismatch");
    std::vector<double> means(24, 0.0);
    std::vector<double> ranges(24, 0.0);
    for (std::size_t feature = 0; feature < 24; ++feature) {
      double minimum = window_features.front()[feature];
      double maximum = minimum;
      for (const auto& vector : window_features) {
        means[feature] += vector[feature];
        minimum = std::min(minimum, vector[feature]);
        maximum = std::max(maximum, vector[feature]);
      }
      means[feature] /= static_cast<double>(window_features.size());
      ranges[feature] = maximum - minimum;
    }
    std::vector<double> multi_features;
    multi_features.reserve(98);
    multi_features.insert(multi_features.end(), prefix_features.begin(),
                          prefix_features.end());
    multi_features.insert(multi_features.end(), means.begin(), means.end());
    multi_features.insert(multi_features.end(), ranges.begin(), ranges.end());
    multi_features.insert(multi_features.end(),
                          window_features[tail_window_index].begin(),
                          window_features[tail_window_index].end());
    multi_features.push_back(
        std::log1p(static_cast<double>(valid_value_count)));
    multi_features.push_back(std::log1p(static_cast<double>(whole_row_count)));
    d::require(multi_features.size() == multi_names.size(),
               context + " multi3 feature shape mismatch");
    accumulate(feature_time, elapsed(feature_start), "feature extraction");

    if (prefix_sample.truncated) ++prefix_truncated_count;
    multi_unique_window_count =
        add(multi_unique_window_count, windows.size(),
            "multi3 unique window count");
    const auto serialization_start = now();
    const auto& selected = blocks[prefix_block];
    const auto prefix_valid = prefix_sample.values.size() / dtype_width;
    Json common{{"schema", "operator-oracle-risk135-feature-row-v1"},
                {"dataset_id", dataset_id},
                {"column_id", column_id},
                {"column", column},
                {"dtype", dtype},
                {"input_dtype", dtype},
                {"columns_sha256", columns_sha256},
                {"input_metadata_sha256", input_metadata_sha256},
                {"oracle_reference_jsonl_sha256", oracle_sha256},
                {"column_source",
                 {{"manifest_path", manifest_path.string()},
                  {"manifest_sha256", manifest_sha256},
                  {"original_numeric_bytes", original_numeric_bytes},
                  {"valid_value_count", valid_value_count},
                  {"row_count", whole_row_count},
                  {"manifest_block_count", blocks.size()}}},
                {"oracle_labels_read", false}};
    Json prefix_row = common;
    prefix_row["feature_variant"] = "prefix24";
    prefix_row["feature_contract"] =
        "operator_recommend135_v1::extract_features/24/first-nonempty-prefix-v1";
    prefix_row["feature_names"] = prefix_names;
    prefix_row["feature_names_sha256"] = prefix_names_sha256;
    prefix_row["features"] = prefix_features;
    prefix_row["sample_source"] =
        {{"selection", "first_nonempty_original_block_prefix"},
         {"manifest_array_index", selected.manifest_array_index},
         {"block_id", selected.block_id},
         {"block_index", selected.block_index},
         {"source_row_start", 0},
         {"source_row_end_exclusive", prefix_sample.row_end},
         {"source_block_row_count", selected.row_count},
         {"source_block_valid_count", selected.valid_count},
         {"values_path", selected.values_path.string()},
         {"values_offset", selected.values_offset},
         {"values_length", selected.values_length},
         {"values_sha256", selected.values_sha256},
         {"missing_bitmap_path", selected.missing_path.string()},
         {"missing_bitmap_offset", selected.missing_offset},
         {"missing_bitmap_length", selected.missing_length},
         {"missing_bitmap_sha256", selected.missing_sha256},
         {"full_selected_source_regions_sha256_verified", true}};
    prefix_row["sample"] =
        {{"row_count", prefix_sample.row_count},
         {"valid_count", prefix_valid},
         {"missing_count", prefix_sample.row_count - prefix_valid},
         {"values_bytes", prefix_sample.values.size()},
         {"missing_bytes", prefix_sample.missing.size()},
         {"values_sha256", d::sha(prefix_sample.values)},
         {"missing_sha256", d::sha(prefix_sample.missing)},
         {"max_valid_values", options.max_valid_values_per_window},
         {"truncated", prefix_sample.truncated}};
    prefix_output_bytes += prefix_row.dump();
    prefix_output_bytes.push_back('\n');

    Json window_proofs = Json::array();
    for (auto& window : windows) window_proofs.push_back(window.proof);
    Json multi_row = common;
    multi_row["feature_variant"] = "multi3";
    multi_row["feature_contract"] =
        "prefix24+unique-head-middle-tail-valid-windows-mean-range-tail+whole-column-counts/98/v1";
    multi_row["feature_names"] = multi_names;
    multi_row["feature_names_sha256"] = multi_names_sha256;
    multi_row["features"] = multi_features;
    multi_row["multi3_sampling"] =
        {{"coordinate_system",
          "packed_valid_value_index_across_manifest_blocks"},
         {"requested_positions",
          Json::array({{{"role", "head"}, {"relative_position", 0.0}},
                       {{"role", "middle"}, {"relative_position", 0.5}},
                       {{"role", "tail"}, {"relative_position", 1.0}}})},
         {"maximum_valid_values_per_window",
          options.max_valid_values_per_window},
         {"requested_window_count", 3},
         {"unique_window_count", windows.size()},
         {"deduplicated_by_valid_interval", true},
         {"whole_column_valid_count", valid_value_count},
         {"whole_column_row_count", whole_row_count},
         {"windows", std::move(window_proofs)}};
    multi_output += multi_row.dump();
    multi_output.push_back('\n');
    accumulate(serialization_time, elapsed(serialization_start),
               "feature row serialization");
  }

  const auto commit_start = now();
  write_immutable(prefix_output, prefix_output_bytes);
  write_immutable(multi_output_path, multi_output);
  accumulate(output_commit_time, elapsed(commit_start),
             "feature output commit");
  const auto total_time = elapsed(total_start);
  const Json timing_contract{
      {"clocks", Json::array({"CLOCK_THREAD_CPUTIME_ID", "steady_clock"})},
      {"source_validation_and_read",
       "COLUMNS and label-free INPUT_METADATA read/hash/schema/split gates; development manifests only; touched full source regions read/hash; bitmap validation and deterministic slicing"},
      {"feature_extraction",
       "operator_recommend135_v1::extract_features and multi3 aggregation"},
      {"jsonl_serialization",
       "prefix24 and multi3 row construction and canonical compact dump"},
      {"output_commit", "new immutable output writes"},
      {"source_file_io_included", true},
      {"oracle_label_io_included", false}};
  const Json timings{{"source_validation_and_read", timing(source_time)},
                     {"feature_extraction", timing(feature_time)},
                     {"jsonl_serialization", timing(serialization_time)},
                     {"output_commit", timing(output_commit_time)},
                     {"total", timing(total_time)}};
  const auto make_receipt =
      [&](const std::string& variant, const std::string& contract,
          const std::vector<std::string>& names,
          const std::string& names_sha256, const fs::path& output,
          const std::string& bytes) {
        return Json{
            {"schema", "operator-oracle-risk135-features-receipt-v1"},
            {"status", "complete"},
            {"feature_variant", variant},
            {"feature_contract", contract},
            {"columns_path", columns_path.string()},
            {"columns_sha256", columns_sha256},
            {"input_metadata_path", metadata_path.string()},
            {"input_metadata_sha256", input_metadata_sha256},
            {"oracle_reference_jsonl_sha256", oracle_sha256},
            {"output_path", output.string()},
            {"output_sha256", d::sha(bytes)},
            {"feature_names", names},
            {"feature_names_sha256", names_sha256},
            {"feature_count", names.size()},
            {"column_count", development_count},
            {"dataset_count", development_datasets.size()},
            {"input_development_column_count", development_count},
            {"input_confirm_column_count", confirm_count},
            {"confirm_column_count_processed", 0},
            {"confirm_source_manifests_opened", 0},
            {"verified_development_manifest_count", verified_manifests},
            {"verified_unique_source_region_bytes", verified_source_bytes},
            {"max_valid_values_per_window",
             options.max_valid_values_per_window},
            {"timing_contract", timing_contract},
            {"timings", timings},
            {"oracle_labels_read", false}};
      };
  auto prefix_receipt = make_receipt(
      "prefix24",
      "operator_recommend135_v1::extract_features/24/first-nonempty-prefix-v1",
      prefix_names, prefix_names_sha256, prefix_output, prefix_output_bytes);
  prefix_receipt["truncated_prefix_sample_count"] = prefix_truncated_count;
  auto multi_receipt = make_receipt(
      "multi3",
      "prefix24+unique-head-middle-tail-valid-windows-mean-range-tail+whole-column-counts/98/v1",
      multi_names, multi_names_sha256, multi_output_path, multi_output);
  multi_receipt["total_unique_window_count"] = multi_unique_window_count;
  const auto receipt_start = now();
  d::write_json(prefix_receipt_path, prefix_receipt);
  d::write_json(multi_receipt_path, multi_receipt);
  accumulate(output_commit_time, elapsed(receipt_start),
             "feature receipt commit");
  return {{"schema", "operator-oracle-risk135-feature-variants-receipt-v1"},
          {"status", "complete"},
          {"development_column_count", development_count},
          {"development_dataset_count", development_datasets.size()},
          {"confirm_column_count_available", confirm_count},
          {"confirm_column_count_processed", 0},
          {"oracle_labels_read", false},
          {"variants",
           {{"prefix24", prefix_receipt}, {"multi3", multi_receipt}}}};
}

}  // namespace operator_oracle_quality135_v1
