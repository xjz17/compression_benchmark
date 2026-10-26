#include "column_scales.hpp"

#include "baseline.hpp"
#include "decimal_scaling.hpp"
#include "typed_segments.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace native_baseline86_v1::column_scales {
namespace {

using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t maximum_file_bytes = 256ULL << 20;
constexpr std::uint64_t maximum_source_io_bytes = 64ULL << 30;
constexpr unsigned maximum_timing_iterations = 1000;
constexpr std::string_view receipt_schema = "column-scales-exact-decimal-v1";
constexpr std::string_view receipt_status =
    "measured_from_frozen_typed_values";
constexpr std::string_view precision_semantics =
    "typed_shortest_roundtrip_decimal_v1";
constexpr std::string_view timing_scope =
    "sum of per-block resident required_decimal_places plus the whole-column max update; source I/O, SHA-256, bitmap and Block validation excluded";
constexpr std::string_view timing_clock = "std::chrono::steady_clock";

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char* message) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::length_error(message);
  return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right,
                          const char* message) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::length_error(message);
  return left * right;
}

bool is_lower_sha256(std::string_view value) {
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
  });
}

std::uint64_t regular_file_size(const fs::path& path) {
  std::error_code error;
  const auto status = fs::status(path, error);
  require(!error && fs::is_regular_file(status),
          "source is not a regular file: " + path.string());
  const auto size = fs::file_size(path, error);
  require(!error, "cannot size source file: " + path.string());
  require(size <= maximum_file_bytes,
          "source file exceeds 256 MiB: " + path.string());
  return size;
}

Bytes read_file(const fs::path& path, std::uint64_t* source_io_bytes) {
  const auto size = regular_file_size(path);
  if (source_io_bytes) {
    const auto next = checked_add(*source_io_bytes, size,
                                  "source I/O byte count overflow");
    require(next <= maximum_source_io_bytes,
            "source I/O exceeds 64 GiB receipt bound");
    *source_io_bytes = next;
  }
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open source file: " + path.string());
  Bytes output(static_cast<std::size_t>(size));
  if (!output.empty())
    input.read(reinterpret_cast<char*>(output.data()),
               static_cast<std::streamsize>(output.size()));
  require(static_cast<bool>(input), "cannot read source file: " + path.string());
  require(input.peek() == std::char_traits<char>::eof(),
          "source file changed while reading: " + path.string());
  return output;
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  auto* context = EVP_MD_CTX_new();
  if (!context) throw std::runtime_error("cannot allocate SHA-256 context");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned digest_size = 0;
  const auto* data = bytes.empty()
      ? reinterpret_cast<const unsigned char*>("") : bytes.data();
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, data, bytes.size()) == 1 &&
      EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || digest_size != 32) throw std::runtime_error("SHA-256 failed");
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (unsigned i = 0; i < digest_size; ++i)
    result << std::setw(2) << static_cast<unsigned>(digest[i]);
  return result.str();
}

std::string required_string(const J& object, const char* field,
                            const std::string& context) {
  require(object.contains(field) && object.at(field).is_string(),
          context + " lacks string " + field);
  const auto value = object.at(field).get<std::string>();
  require(!value.empty(), context + " has empty " + field);
  return value;
}

std::string inherited_string(const J& manifest, const J& row,
                             const char* field, bool required) {
  const J* source = nullptr;
  if (row.contains(field)) source = &row;
  else if (manifest.contains(field)) source = &manifest;
  if (!source) {
    if (required) fail(std::string("missing column identity field ") + field);
    return {};
  }
  require(source->at(field).is_string(),
          std::string("column identity field is not a string: ") + field);
  const auto value = source->at(field).get<std::string>();
  if (required)
    require(!value.empty(), std::string("empty column identity field: ") + field);
  return value;
}

std::uint64_t required_u64(const J& object, const char* field,
                           const std::string& context) {
  require(object.contains(field) && object.at(field).is_number_unsigned(),
          context + " lacks unsigned integer " + field);
  return object.at(field).get<std::uint64_t>();
}

fs::path source_path(const fs::path& manifest_path, const J& row,
                     const char* field, const std::string& context) {
  const auto raw = required_string(row, field, context);
  const fs::path supplied(raw);
  return supplied.is_absolute() ? supplied : manifest_path.parent_path() / supplied;
}

struct BlockDescriptor {
  std::string block_id;
  std::string dtype;
  fs::path values_path;
  fs::path missing_path;
  std::string values_sha256;
  std::string missing_sha256;
  std::uint64_t row_count = 0;
  std::uint64_t original_numeric_bytes = 0;
  std::uint64_t source_bytes = 0;
  typed_segments::Range values_range{},missing_range{};
};

struct Column {
  std::string key;
  J identity;
  std::vector<BlockDescriptor> blocks;
  std::uint64_t original_numeric_bytes = 0;
};

struct Loaded {
  J manifest;
  std::string manifest_sha256;
  std::uint64_t source_io_bytes = 0;
  std::map<std::string, Column> columns;
  std::vector<BlockDescriptor> excluded_non_float_descriptors;
  J excluded_non_float_blocks = J::array();
  std::uint64_t float_source_bytes_per_precision_pass = 0;
  std::uint64_t non_float_validation_source_bytes = 0;
};

J identity_from_key(const std::string& key) {
  auto identity = J::parse(key);
  require(identity.is_object(), "column key is not an object");
  return identity;
}

Loaded load(const fs::path& manifest_path) {
  Loaded result;
  auto manifest_bytes = read_file(manifest_path, &result.source_io_bytes);
  result.manifest_sha256 = sha256(manifest_bytes);
  result.manifest = J::parse(manifest_bytes.begin(), manifest_bytes.end());
  require(result.manifest.is_object(), "manifest is not a JSON object");
  require(result.manifest.contains("blocks") &&
              result.manifest.at("blocks").is_array(),
          "manifest lacks blocks array");
  require(!result.manifest.at("blocks").empty(), "manifest blocks is empty");
  const auto dataset_id = inherited_string(result.manifest, J::object(),
                                             "dataset_id", true);
  std::set<std::string> seen_ids;
  for (const auto& row : result.manifest.at("blocks")) {
    require(row.is_object(), "manifest block is not an object");
    const auto block_id = required_string(row, "block_id", "manifest block");
    require(seen_ids.insert(block_id).second,
            "duplicate manifest block_id: " + block_id);
    if (row.contains("dataset_id"))
      require(required_string(row, "dataset_id", "manifest block") == dataset_id,
              "block dataset_id differs from manifest dataset_id");
    const auto dtype = required_string(row, "dtype", "manifest block " + block_id);
    require(dtype == "float32" || dtype == "float64" || dtype == "int32" ||
                dtype == "int64",
            "unsupported manifest dtype in block " + block_id);
    const auto row_count = required_u64(row, "row_count", "manifest block " + block_id);
    const auto declared_bytes = required_u64(
        row, "original_numeric_bytes", "manifest block " + block_id);
    const auto values_path = source_path(manifest_path, row, "values_path",
                                         "manifest block " + block_id);
    const auto missing_path = source_path(manifest_path, row,
        "missing_bitmap_path", "manifest block " + block_id);
    const auto declared_values_sha = required_string(
        row, "values_sha256", "manifest block " + block_id);
    const auto declared_missing_sha = required_string(
        row, "missing_bitmap_sha256", "manifest block " + block_id);
    require(is_lower_sha256(declared_values_sha) &&
                is_lower_sha256(declared_missing_sha),
            "manifest block has invalid lowercase SHA-256: " + block_id);
    const auto values_range = typed_segments::range(values_path,row,"values");
    const auto missing_range = typed_segments::range(missing_path,row,"missing_bitmap");
    const auto values_bytes = values_range.length;
    const auto missing_bytes = missing_range.length;
    require(values_bytes == declared_bytes,
            "original_numeric_bytes mismatch: " + block_id);
    const auto descriptor_source_bytes = checked_add(
        values_bytes, missing_bytes, "block source byte count overflow");
    BlockDescriptor descriptor{block_id, dtype, values_path, missing_path,
        declared_values_sha, declared_missing_sha, row_count, declared_bytes,
        descriptor_source_bytes,values_range,missing_range};
    if (dtype != "float32" && dtype != "float64") {
      result.non_float_validation_source_bytes = checked_add(
          result.non_float_validation_source_bytes, descriptor_source_bytes,
          "non-float source byte count overflow");
      result.excluded_non_float_descriptors.push_back(descriptor);
      result.excluded_non_float_blocks.push_back(
          J{{"block_id", block_id}, {"dtype", dtype},
            {"values_sha256", declared_values_sha},
            {"missing_bitmap_sha256", declared_missing_sha},
            {"original_numeric_bytes", declared_bytes}});
      continue;
    }
    result.float_source_bytes_per_precision_pass = checked_add(
        result.float_source_bytes_per_precision_pass, descriptor_source_bytes,
        "float source byte count overflow");
    const auto key = column_key(result.manifest, row);
    auto [it, inserted] = result.columns.try_emplace(key);
    if (inserted) {
      it->second.key = key;
      it->second.identity = identity_from_key(key);
    }
    it->second.original_numeric_bytes = checked_add(
        it->second.original_numeric_bytes, declared_bytes,
        "column original_numeric_bytes overflow");
    it->second.blocks.push_back(std::move(descriptor));
  }
  require(!result.columns.empty(), "manifest has no float32/float64 blocks");
  for (auto& [key, column] : result.columns) {
    (void)key;
    std::sort(column.blocks.begin(), column.blocks.end(),
              [](const auto& left, const auto& right) {
                return left.block_id < right.block_id;
              });
  }
  std::sort(result.excluded_non_float_blocks.begin(),
            result.excluded_non_float_blocks.end(),
            [](const J& left, const J& right) {
              return left.at("block_id").get<std::string>() <
                     right.at("block_id").get<std::string>();
            });
  return result;
}

native_baseline86_v1::Block read_validated_block(
    const BlockDescriptor& descriptor, std::uint64_t* source_io_bytes) {
  auto values = typed_segments::read(descriptor.values_path,descriptor.values_range);
  auto missing = typed_segments::read(descriptor.missing_path,descriptor.missing_range);
  if(source_io_bytes){
    *source_io_bytes=checked_add(*source_io_bytes,descriptor.source_bytes,"segment source I/O overflow");
    require(*source_io_bytes<=maximum_source_io_bytes,"source I/O exceeds 64 GiB receipt bound");
  }
  require(values.size() == descriptor.original_numeric_bytes,
          "values size changed: " + descriptor.block_id);
  require(sha256(values) == descriptor.values_sha256,
          "values SHA-256 mismatch: " + descriptor.block_id);
  require(sha256(missing) == descriptor.missing_sha256,
          "missing bitmap SHA-256 mismatch: " + descriptor.block_id);
  native_baseline86_v1::Block block{
      descriptor.dtype, descriptor.row_count, std::move(values),
      std::move(missing)};
  native_baseline86_v1::validate(block);
  return block;
}

struct ColumnPass {
  std::uint16_t decimal_places = 0;
  std::uint64_t elapsed_ns = 0;
};

ColumnPass scan_streamed_column(const Column& column,
                                std::uint64_t* source_io_bytes,
                                bool measure) {
  ColumnPass result;
  for (const auto& descriptor : column.blocks) {
    auto block = read_validated_block(descriptor, source_io_bytes);
    if (!measure) {
      result.decimal_places = std::max(result.decimal_places,
          decimal_scaling::required_decimal_places(block.values,
                                                     block.dtype));
      continue;
    }
    const auto begin = Clock::now();
    result.decimal_places = std::max(result.decimal_places,
        decimal_scaling::required_decimal_places(block.values, block.dtype));
    const auto end = Clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - begin).count();
    require(elapsed >= 0, "steady clock produced a negative duration");
    result.elapsed_ns = checked_add(result.elapsed_ns,
        static_cast<std::uint64_t>(elapsed),
        "whole-column precision time overflow");
  }
  return result;
}

void validate_excluded_blocks(const Loaded& loaded,
                              std::uint64_t* source_io_bytes) {
  for (const auto& descriptor : loaded.excluded_non_float_descriptors)
    (void)read_validated_block(descriptor, source_io_bytes);
}

J receipt_column(const Column& column, unsigned warmups, unsigned repeats,
                 std::uint64_t* source_io_bytes) {
  std::optional<std::uint16_t> places;
  for (unsigned i = 0; i < warmups; ++i) {
    const auto observed = scan_streamed_column(column, source_io_bytes, false);
    if (!places) places = observed.decimal_places;
    require(observed.decimal_places == *places,
            "precision warmup changed column scale");
  }
  J samples = J::array();
  for (unsigned i = 0; i < repeats; ++i) {
    const auto observed = scan_streamed_column(column, source_io_bytes, true);
    if (!places) places = observed.decimal_places;
    require(observed.decimal_places == *places,
            "precision repeat changed column scale");
    samples.push_back(observed.elapsed_ns);
  }
  require(places.has_value(), "precision scan produced no observation");
  J block_ids = J::array();
  J values_sha256s = J::array();
  J missing_sha256s = J::array();
  for (const auto& block : column.blocks) {
    block_ids.push_back(block.block_id);
    values_sha256s.push_back(block.values_sha256);
    missing_sha256s.push_back(block.missing_sha256);
  }
  return {{"column_key", column.key},
          {"column_identity", column.identity},
          {"decimal_places", *places},
          {"block_ids", std::move(block_ids)},
          {"values_sha256s", std::move(values_sha256s)},
          {"missing_bitmap_sha256s", std::move(missing_sha256s)},
          {"original_numeric_bytes", column.original_numeric_bytes},
          {"precision_derivation_ns_samples", std::move(samples)}};
}

void validate_iterations(unsigned warmups, unsigned repeats) {
  require(warmups <= maximum_timing_iterations,
          "precision warmups exceeds 1000");
  require(repeats > 0 && repeats <= maximum_timing_iterations,
          "precision repeats must be in [1,1000]");
}

std::uint64_t json_u64(const J& object, const char* field,
                       const std::string& context) {
  return required_u64(object, field, context);
}

void require_equal(const J& actual, const J& expected,
                   const std::string& field) {
  require(actual == expected, "precision receipt mismatch: " + field);
}

void require_exact_keys(const J& object,
                        std::initializer_list<std::string_view> expected,
                        const std::string& context) {
  require(object.is_object(), context + " is not an object");
  std::set<std::string> keys;
  for (const auto field : expected) keys.emplace(field);
  require(object.size() == keys.size(), context + " field count mismatch");
  for (const auto& [key, value] : object.items()) {
    (void)value;
    require(keys.contains(key), context + " has unexpected field: " + key);
  }
}

}  // namespace

std::string column_key(const J& manifest, const J& row) {
  const auto dataset_id = inherited_string(manifest, row, "dataset_id", true);
  const auto table_name = inherited_string(manifest, row, "table_name", false);
  const auto role = inherited_string(manifest, row, "role", true);
  const auto source_column = inherited_string(
      manifest, row, "source_column", false);
  const auto column = inherited_string(manifest, row, "column", false);
  require(!source_column.empty() || !column.empty(),
          "column identity requires source_column or column");
  const auto dtype = inherited_string(manifest, row, "dtype", true);
  return J{{"dataset_id", dataset_id}, {"table_name", table_name},
           {"role", role}, {"source_column", source_column},
           {"column", column}, {"dtype", dtype}}.dump();
}

J scan(const fs::path& manifest_path, unsigned warmups, unsigned repeats,
       const std::string& cli_sha256) {
  validate_iterations(warmups, repeats);
  require(is_lower_sha256(cli_sha256),
          "scanner_cli_sha256 must be lowercase SHA-256");
  auto loaded = load(manifest_path);
  const auto precision_passes = checked_add(warmups, repeats,
      "precision pass count overflow");
  const auto expected_source_io = checked_add(
      loaded.source_io_bytes,
      checked_add(loaded.non_float_validation_source_bytes,
          checked_mul(loaded.float_source_bytes_per_precision_pass,
                      precision_passes,
                      "precision source I/O byte count overflow"),
          "precision source I/O byte count overflow"),
      "precision source I/O byte count overflow");
  require(expected_source_io <= maximum_source_io_bytes,
          "source I/O exceeds 64 GiB receipt bound");
  validate_excluded_blocks(loaded, &loaded.source_io_bytes);
  J columns = J::array();
  for (const auto& [key, column] : loaded.columns) {
    (void)key;
    columns.push_back(receipt_column(column, warmups, repeats,
                                     &loaded.source_io_bytes));
  }
  require(loaded.source_io_bytes == expected_source_io,
          "precision source I/O accounting mismatch");
  return {{"schema", receipt_schema},
          {"status", receipt_status},
          {"manifest_sha256", loaded.manifest_sha256},
          {"scanner_cli_sha256", cli_sha256},
          {"precision_semantics", precision_semantics},
          {"precision_timing_scope", timing_scope},
          {"precision_timing_clock", timing_clock},
          {"precision_timing_unit", "ns"},
          {"warmups", warmups},
          {"repeats", repeats},
          {"source_io_bytes", loaded.source_io_bytes},
          {"source_float_io_bytes_per_precision_pass",
           loaded.float_source_bytes_per_precision_pass},
          {"source_non_float_validation_io_bytes",
           loaded.non_float_validation_source_bytes},
          {"source_precision_passes", precision_passes},
          {"timed_source_io_bytes", std::uint64_t{0}},
          {"columns", std::move(columns)},
          {"excluded_non_float_blocks", loaded.excluded_non_float_blocks}};
}

J validate(const fs::path& manifest_path, const J& receipt,
           const std::string& cli_sha256) {
  require_exact_keys(receipt,
      {"schema", "status", "manifest_sha256", "scanner_cli_sha256",
       "precision_semantics", "precision_timing_scope",
       "precision_timing_clock", "precision_timing_unit", "warmups",
       "repeats", "source_io_bytes",
       "source_float_io_bytes_per_precision_pass",
       "source_non_float_validation_io_bytes", "source_precision_passes",
       "timed_source_io_bytes", "columns", "excluded_non_float_blocks"},
      "precision receipt");
  require(is_lower_sha256(cli_sha256),
          "scanner_cli_sha256 must be lowercase SHA-256");
  require(required_string(receipt, "schema", "precision receipt") ==
              receipt_schema,
          "precision receipt schema mismatch");
  require(required_string(receipt, "status", "precision receipt") ==
              receipt_status,
          "precision receipt status mismatch");
  require(required_string(receipt, "scanner_cli_sha256", "precision receipt") ==
              cli_sha256,
          "precision receipt scanner CLI SHA-256 mismatch");
  require(required_string(receipt, "precision_semantics", "precision receipt") ==
              precision_semantics,
          "precision receipt semantics mismatch");
  require(required_string(receipt, "precision_timing_scope", "precision receipt") ==
              timing_scope,
          "precision receipt timing scope mismatch");
  require(required_string(receipt, "precision_timing_clock", "precision receipt") ==
              timing_clock,
          "precision receipt timing clock mismatch");
  require(required_string(receipt, "precision_timing_unit", "precision receipt") ==
              "ns",
          "precision receipt timing unit mismatch");
  const auto warmups64 = json_u64(receipt, "warmups", "precision receipt");
  const auto repeats64 = json_u64(receipt, "repeats", "precision receipt");
  require(warmups64 <= std::numeric_limits<unsigned>::max() &&
              repeats64 <= std::numeric_limits<unsigned>::max(),
          "precision timing iteration count overflows unsigned");
  validate_iterations(static_cast<unsigned>(warmups64),
                      static_cast<unsigned>(repeats64));
  require(json_u64(receipt, "timed_source_io_bytes", "precision receipt") == 0,
          "precision receipt includes timed source I/O");
  auto loaded = load(manifest_path);
  require(required_string(receipt, "manifest_sha256", "precision receipt") ==
              loaded.manifest_sha256,
          "precision receipt manifest SHA-256 mismatch");
  const auto precision_passes = checked_add(warmups64, repeats64,
      "precision pass count overflow");
  const auto expected_source_io = checked_add(
      loaded.source_io_bytes,
      checked_add(loaded.non_float_validation_source_bytes,
          checked_mul(loaded.float_source_bytes_per_precision_pass,
                      precision_passes,
                      "precision source I/O byte count overflow"),
          "precision source I/O byte count overflow"),
      "precision source I/O byte count overflow");
  require(expected_source_io <= maximum_source_io_bytes,
          "source I/O exceeds 64 GiB receipt bound");
  require(json_u64(receipt, "source_io_bytes", "precision receipt") ==
              expected_source_io,
          "precision receipt source I/O byte count mismatch");
  require(json_u64(receipt, "source_float_io_bytes_per_precision_pass",
                   "precision receipt") ==
              loaded.float_source_bytes_per_precision_pass,
          "precision receipt float source pass byte count mismatch");
  require(json_u64(receipt, "source_non_float_validation_io_bytes",
                   "precision receipt") ==
              loaded.non_float_validation_source_bytes,
          "precision receipt non-float validation byte count mismatch");
  require(json_u64(receipt, "source_precision_passes", "precision receipt") ==
              precision_passes,
          "precision receipt source pass count mismatch");
  require(receipt.contains("excluded_non_float_blocks"),
          "precision receipt lacks excluded_non_float_blocks");
  require_equal(receipt.at("excluded_non_float_blocks"),
                loaded.excluded_non_float_blocks,
                "excluded_non_float_blocks");
  require(receipt.contains("columns") && receipt.at("columns").is_array(),
          "precision receipt lacks columns array");
  require(receipt.at("columns").size() == loaded.columns.size(),
          "precision receipt column count mismatch");
  std::uint64_t validation_io = loaded.source_io_bytes;
  validate_excluded_blocks(loaded, &validation_io);
  std::set<std::string> seen_columns;
  for (const auto& item : receipt.at("columns")) {
    require_exact_keys(item,
        {"column_key", "column_identity", "decimal_places", "block_ids",
         "values_sha256s", "missing_bitmap_sha256s",
         "original_numeric_bytes", "precision_derivation_ns_samples"},
        "precision receipt column");
    const auto key = required_string(item, "column_key", "precision column");
    require(seen_columns.insert(key).second,
            "duplicate precision receipt column_key");
    const auto found = loaded.columns.find(key);
    require(found != loaded.columns.end(), "extra precision receipt column_key");
    const auto& expected = found->second;
    const auto expected_pass = scan_streamed_column(expected, &validation_io,
                                                    false);
    const auto expected_places = expected_pass.decimal_places;
    require_equal(item.at("column_identity"), expected.identity,
                  "column_identity");
    require(json_u64(item, "decimal_places", "precision column") ==
                expected_places,
            "precision receipt decimal_places mismatch");
    require(json_u64(item, "original_numeric_bytes", "precision column") ==
                expected.original_numeric_bytes,
            "precision receipt original_numeric_bytes mismatch");
    J block_ids = J::array();
    J values_sha256s = J::array();
    J missing_sha256s = J::array();
    for (const auto& block : expected.blocks) {
      block_ids.push_back(block.block_id);
      values_sha256s.push_back(block.values_sha256);
      missing_sha256s.push_back(block.missing_sha256);
    }
    require_equal(item.at("block_ids"), block_ids, "block_ids");
    require_equal(item.at("values_sha256s"), values_sha256s,
                  "values_sha256s");
    require_equal(item.at("missing_bitmap_sha256s"), missing_sha256s,
                  "missing_bitmap_sha256s");
    require(item.contains("precision_derivation_ns_samples") &&
                item.at("precision_derivation_ns_samples").is_array(),
            "precision column lacks timing samples");
    const auto& samples = item.at("precision_derivation_ns_samples");
    require(samples.size() == repeats64,
            "precision column timing sample count mismatch");
    for (const auto& sample : samples)
      require(sample.is_number_unsigned(),
              "precision timing sample is not a nonnegative integer");
  }
  require(seen_columns.size() == loaded.columns.size(),
          "precision receipt omitted a column");
  require(validation_io == checked_add(
              loaded.source_io_bytes,
              checked_add(loaded.non_float_validation_source_bytes,
                          loaded.float_source_bytes_per_precision_pass,
                          "validation source I/O byte count overflow"),
              "validation source I/O byte count overflow"),
          "validation source I/O accounting mismatch");
  return receipt;
}

}  // namespace native_baseline86_v1::column_scales
