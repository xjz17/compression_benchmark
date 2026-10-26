#include "input_builder.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace operator_recommend135_v1 {
namespace {
using J = nlohmann::json;
namespace fs = std::filesystem;

constexpr std::uint64_t kMaximumJsonBytes = 64ULL << 20;
constexpr std::uint64_t kMaximumSourceRegionBytes = 64ULL << 20;
constexpr std::size_t kMinimumValues = 64;
constexpr std::size_t kMaximumValues = 32768;
constexpr std::string_view kFrozenPlanSha256 =
    "ff1404b37e9e9bee19550ea5b7c8ce54480b3c05d6ec73c9721655fdd2c58a47";

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const std::string& label) {
  require(left <= std::numeric_limits<std::uint64_t>::max() - right,
          label + " overflow");
  return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right,
                          const std::string& label) {
  require(left == 0 || right <= std::numeric_limits<std::uint64_t>::max() / left,
          label + " overflow");
  return left * right;
}

std::size_t checked_size(std::uint64_t value, const std::string& label) {
  require(value <= std::numeric_limits<std::size_t>::max(), label + " exceeds size_t");
  return static_cast<std::size_t>(value);
}

std::string read_file(const fs::path& path, std::uint64_t maximum) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(bool(input), "cannot read " + path.string());
  const auto end = input.tellg();
  require(end >= 0, "cannot size " + path.string());
  const auto size = static_cast<std::uint64_t>(end);
  require(size <= maximum, "oversized file " + path.string());
  input.seekg(0);
  std::string result(checked_size(size, "file size"), '\0');
  if (!result.empty())
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
  require(bool(input) || result.empty(), "short read " + path.string());
  return result;
}

std::string read_region(const fs::path& path, std::uint64_t offset,
                        std::uint64_t length) {
  require(length <= kMaximumSourceRegionBytes,
          "source region exceeds memory bound: " + path.string());
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  require(descriptor >= 0, "cannot open source region " + path.string());
  try {
    struct stat status {};
    require(::fstat(descriptor, &status) == 0 && status.st_size >= 0,
            "cannot stat source region " + path.string());
    require(checked_add(offset, length, "source region") <=
                static_cast<std::uint64_t>(status.st_size),
            "source region outside file " + path.string());
    std::string result(checked_size(length, "source region length"), '\0');
    std::size_t position = 0;
    while (position < result.size()) {
      const auto count = ::pread(descriptor, result.data() + position,
                                 result.size() - position,
                                 static_cast<off_t>(offset + position));
      if (count < 0 && errno == EINTR) continue;
      require(count > 0, "short source region read " + path.string());
      position += static_cast<std::size_t>(count);
    }
    require(::close(descriptor) == 0, "cannot close source region " + path.string());
    return result;
  } catch (...) {
    ::close(descriptor);
    throw;
  }
}

std::string sha256(const void* data, std::size_t size) {
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  require(context != nullptr, "cannot allocate SHA256 context");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned digest_size = 0;
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, data, size) == 1 &&
                  EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
  EVP_MD_CTX_free(context);
  require(ok && digest_size == 32, "SHA256 failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned index = 0; index < digest_size; ++index)
    output << std::setw(2) << static_cast<unsigned>(digest[index]);
  return output.str();
}

std::string sha256(const std::string& bytes) {
  return sha256(bytes.data(), bytes.size());
}

bool valid_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::string required_text(const J& value, const char* key,
                          const std::string& context) {
  require(value.contains(key) && value.at(key).is_string(),
          context + " missing string " + key);
  const auto result = value.at(key).get<std::string>();
  require(!result.empty() && result.find('\0') == std::string::npos,
          context + " invalid string " + key);
  return result;
}

std::uint64_t required_u64(const J& value, const char* key,
                           const std::string& context) {
  require(value.contains(key) &&
              (value.at(key).is_number_unsigned() || value.at(key).is_number_integer()),
          context + " missing integer " + key);
  if (value.at(key).is_number_integer()) {
    const auto signed_value = value.at(key).get<std::int64_t>();
    require(signed_value >= 0, context + " negative integer " + key);
    return static_cast<std::uint64_t>(signed_value);
  }
  return value.at(key).get<std::uint64_t>();
}

fs::path source_path(const fs::path& manifest_path, const std::string& raw) {
  const fs::path candidate(raw);
  return (candidate.is_absolute() ? candidate : manifest_path.parent_path() / candidate)
      .lexically_normal();
}

std::size_t dtype_width(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "float64" || dtype == "int64") return 8;
  fail("unsupported source dtype " + dtype);
}

bool missing_at(const std::string& bitmap, std::uint64_t row) {
  return (static_cast<unsigned char>(bitmap[checked_size(row / 8, "bitmap index")]) >>
          (row % 8)) &
         1U;
}

std::uint64_t count_missing(const std::string& bitmap, std::uint64_t row_count) {
  std::uint64_t count = 0;
  for (std::uint64_t row = 0; row < row_count; ++row)
    count += missing_at(bitmap, row) ? 1 : 0;
  return count;
}

void append_bitmap_bit(std::string& bitmap, std::uint64_t output_row, bool missing) {
  const auto byte = checked_size(output_row / 8, "output bitmap index");
  if (byte == bitmap.size()) bitmap.push_back('\0');
  require(byte < bitmap.size(), "output bitmap append order violation");
  if (missing)
    bitmap[byte] = static_cast<char>(static_cast<unsigned char>(bitmap[byte]) |
                                     (1U << (output_row % 8)));
}

void write_all(int descriptor, const char* data, std::size_t size) {
  std::size_t position = 0;
  while (position < size) {
    const auto count = ::write(descriptor, data + position, size - position);
    if (count < 0 && errno == EINTR) continue;
    require(count > 0, "output write failed");
    position += static_cast<std::size_t>(count);
  }
}

void sync_directory(const fs::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
  require(descriptor >= 0, "cannot open output directory for fsync " + path.string());
  const int status = ::fsync(descriptor);
  ::close(descriptor);
  require(status == 0, "output directory fsync failed " + path.string());
}

void atomic_write(const fs::path& path, const std::string& bytes, mode_t mode) {
  fs::create_directories(path.parent_path());
  if (fs::exists(path)) {
    require(read_file(path, std::max<std::uint64_t>(kMaximumJsonBytes, bytes.size())) ==
                bytes,
            "refusing to replace changed output " + path.string());
    return;
  }
  const auto temporary = path.string() + ".tmp-" + std::to_string(::getpid());
  const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(descriptor >= 0, "cannot create temporary output " + temporary);
  try {
    write_all(descriptor, bytes.data(), bytes.size());
    require(::fsync(descriptor) == 0, "output fsync failed " + temporary);
    require(::fchmod(descriptor, mode) == 0, "output chmod failed " + temporary);
    require(::close(descriptor) == 0, "output close failed " + temporary);
    require(::rename(temporary.c_str(), path.c_str()) == 0,
            "output rename failed " + path.string());
    sync_directory(path.parent_path());
  } catch (...) {
    ::close(descriptor);
    ::unlink(temporary.c_str());
    throw;
  }
}

struct BlockReference {
  std::size_t manifest_array_index = 0;
  std::string block_id;
  std::uint64_t block_index = 0;
  std::uint64_t series_row_start = 0;
  std::uint64_t series_valid_start = 0;
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
  std::string expected_sequence_sha256;
};

struct Series {
  std::string dataset_id;
  std::string column_id;
  std::string dtype;
  std::size_t width = 0;
  J column_identity;
  std::string entity_id;
  std::uint64_t entity_ordinal = 0;
  fs::path manifest_path;
  std::string manifest_sha256;
  std::uint64_t row_count = 0;
  std::uint64_t valid_count = 0;
  std::vector<BlockReference> blocks;
};

using SelectionKey = std::pair<std::string, std::string>;

bool preferred_series(const Series& candidate, const Series& current) {
  if (candidate.valid_count != current.valid_count)
    return candidate.valid_count > current.valid_count;
  if (candidate.row_count != current.row_count) return candidate.row_count > current.row_count;
  return std::tie(candidate.column_id, candidate.entity_ordinal, candidate.entity_id,
                  candidate.manifest_path) <
         std::tie(current.column_id, current.entity_ordinal, current.entity_id,
                  current.manifest_path);
}

BlockReference parse_block(const J& block, std::size_t manifest_array_index,
                           const fs::path& manifest_path, const J& identity,
                           const std::string& dataset_id, const std::string& dtype,
                           std::size_t width) {
  const std::string context =
      "manifest block " + std::to_string(manifest_array_index) + " in " +
      manifest_path.string();
  require(block.is_object(), context + " is not an object");
  BlockReference result;
  result.manifest_array_index = manifest_array_index;
  result.block_id = required_text(block, "block_id", context);
  result.block_index = required_u64(block, "block_index", context);
  result.row_count = required_u64(block, "row_count", context);
  result.valid_count = required_u64(block, "valid_count", context);
  result.missing_count = required_u64(block, "missing_count", context);
  require(result.valid_count <= result.row_count &&
              result.missing_count == result.row_count - result.valid_count,
          context + " inconsistent row/valid/missing counts");
  require(required_text(block, "dataset_id", context) == dataset_id,
          context + " dataset_id differs from manifest");
  require(required_text(block, "dtype", context) == dtype,
          context + " dtype differs from column identity");
  for (const auto* key : {"column", "source_column", "table_name", "role"}) {
    if (identity.contains(key)) {
      require(block.contains(key) && block.at(key) == identity.at(key),
              context + " identity mismatch for " + key);
    }
  }
  result.values_path =
      source_path(manifest_path, required_text(block, "values_path", context));
  result.values_offset = required_u64(block, "values_offset", context);
  result.values_length = required_u64(block, "values_length", context);
  result.values_sha256 = required_text(block, "values_sha256", context);
  result.missing_path = source_path(
      manifest_path, required_text(block, "missing_bitmap_path", context));
  result.missing_offset = required_u64(block, "missing_bitmap_offset", context);
  result.missing_length = required_u64(block, "missing_bitmap_length", context);
  result.missing_sha256 = required_text(block, "missing_bitmap_sha256", context);
  require(valid_sha256(result.values_sha256) && valid_sha256(result.missing_sha256),
          context + " contains invalid source SHA256");
  require(result.values_length ==
              checked_mul(result.valid_count, width, context + " values length"),
          context + " values_length != valid_count*dtype_width");
  require(result.missing_length == (result.row_count + 7) / 8,
          context + " missing_bitmap_length != ceil(row_count/8)");
  (void)checked_add(result.values_offset, result.values_length,
                    context + " values region");
  (void)checked_add(result.missing_offset, result.missing_length,
                    context + " missing region");
  if (block.contains("expected_sequence_sha256")) {
    result.expected_sequence_sha256 =
        required_text(block, "expected_sequence_sha256", context);
    require(valid_sha256(result.expected_sequence_sha256),
            context + " invalid expected_sequence_sha256");
  }
  return result;
}

std::vector<Series> series_from_manifest(const J& manifest,
                                         const fs::path& manifest_path,
                                         const std::string& manifest_sha256,
                                         const J& job) {
  require(manifest.is_object() &&
              manifest.value("schema", "") ==
                  "native-baseline86-scaled-column-segment-manifest-v1",
          "unexpected source manifest schema " + manifest_path.string());
  require(manifest.contains("column_identity") &&
              manifest.at("column_identity").is_object(),
          "source manifest lacks column_identity " + manifest_path.string());
  const auto identity = manifest.at("column_identity");
  const auto dataset_id = required_text(identity, "dataset_id", "column_identity");
  const auto dtype = required_text(identity, "dtype", "column_identity");
  const auto width = dtype_width(dtype);
  require(dataset_id == required_text(job, "dataset_id", "plan job"),
          "plan/manifest dataset_id mismatch " + manifest_path.string());
  require(manifest.contains("dataset_id") && manifest.at("dataset_id") == dataset_id,
          "manifest dataset_id differs from column_identity " + manifest_path.string());
  require(manifest.contains("blocks") && manifest.at("blocks").is_array() &&
              !manifest.at("blocks").empty(),
          "source manifest has no blocks " + manifest_path.string());
  const auto column_id = required_text(job, "column_id", "plan job");
  std::map<std::pair<std::uint64_t, std::string>, std::vector<BlockReference>> grouped;
  for (std::size_t index = 0; index < manifest.at("blocks").size(); ++index) {
    const auto& block = manifest.at("blocks").at(index);
    const auto context = "manifest block " + std::to_string(index);
    const auto entity_id = required_text(block, "entity_id", context);
    const auto entity_ordinal = required_u64(block, "entity_ordinal", context);
    grouped[{entity_ordinal, entity_id}].push_back(
        parse_block(block, index, manifest_path, identity, dataset_id, dtype, width));
  }
  std::vector<Series> result;
  result.reserve(grouped.size());
  for (auto& [entity, blocks] : grouped) {
    std::sort(blocks.begin(), blocks.end(), [](const auto& left, const auto& right) {
      return std::tie(left.block_index, left.manifest_array_index) <
             std::tie(right.block_index, right.manifest_array_index);
    });
    require(!blocks.empty(), "empty entity group in " + manifest_path.string());
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      require(blocks[index].block_index == index,
              "non-consecutive block_index within entity " + entity.second + " in " +
                  manifest_path.string());
    }
    Series series;
    series.dataset_id = dataset_id;
    series.column_id = column_id;
    series.dtype = dtype;
    series.width = width;
    series.column_identity = identity;
    series.entity_id = entity.second;
    series.entity_ordinal = entity.first;
    series.manifest_path = manifest_path;
    series.manifest_sha256 = manifest_sha256;
    if (manifest.contains("source_dataset_manifest_sha256")) {
      const auto source_dataset_manifest_sha256 = required_text(
          manifest, "source_dataset_manifest_sha256", "source manifest");
      require(valid_sha256(source_dataset_manifest_sha256),
              "invalid source_dataset_manifest_sha256 " + manifest_path.string());
    }
    for (auto& block : blocks) {
      block.series_row_start = series.row_count;
      block.series_valid_start = series.valid_count;
      series.row_count = checked_add(series.row_count, block.row_count,
                                     "series row_count");
      series.valid_count = checked_add(series.valid_count, block.valid_count,
                                       "series valid_count");
    }
    series.blocks = std::move(blocks);
    result.push_back(std::move(series));
  }
  return result;
}

struct WindowSpec {
  std::string name;
  std::string side;
  std::uint64_t valid_start = 0;
  std::uint64_t valid_end = 0;
  std::size_t first_block = 0;
  std::size_t last_block = 0;
};

struct WindowOutput {
  WindowSpec spec;
  std::string values;
  std::string missing;
  std::uint64_t row_count = 0;
  std::uint64_t valid_count = 0;
  std::uint64_t row_start = 0;
  std::uint64_t row_end = 0;
  bool has_rows = false;
  J source_blocks = J::array();
};

std::size_t block_for_valid(const Series& series, std::uint64_t valid_position) {
  require(valid_position < series.valid_count, "valid position outside series");
  for (std::size_t index = 0; index < series.blocks.size(); ++index) {
    const auto& block = series.blocks[index];
    if (valid_position >= block.series_valid_start &&
        valid_position < block.series_valid_start + block.valid_count)
      return index;
  }
  fail("valid position is not represented by a source block");
}

std::uint64_t row_for_local_valid(const std::string& bitmap, std::uint64_t row_count,
                                  std::uint64_t local_valid_position) {
  std::uint64_t valid = 0;
  for (std::uint64_t row = 0; row < row_count; ++row) {
    if (missing_at(bitmap, row)) continue;
    if (valid == local_valid_position) return row;
    ++valid;
  }
  fail("local valid position is not represented by bitmap");
}

std::uint64_t valid_before_row(const std::string& bitmap, std::uint64_t row) {
  std::uint64_t valid = 0;
  for (std::uint64_t position = 0; position < row; ++position)
    valid += missing_at(bitmap, position) ? 0 : 1;
  return valid;
}

J build_sample_part(const Series& series, const std::string& sample_id,
                    const fs::path& output, const WindowOutput& window) {
  require(window.valid_count == window.spec.valid_end - window.spec.valid_start,
          "window valid count mismatch");
  require(window.has_rows && window.row_count == window.row_end - window.row_start,
          "window row range mismatch");
  require(window.values.size() == window.valid_count * series.width,
          "window values byte count mismatch");
  require(window.missing.size() == (window.row_count + 7) / 8,
          "window bitmap byte count mismatch");
  require(count_missing(window.missing, window.row_count) ==
              window.row_count - window.valid_count,
          "window bitmap valid count mismatch");
  const auto sample_root = output / "samples" / sample_id;
  const auto values_path = fs::absolute(sample_root /
      (window.spec.side + ".values.bin")).lexically_normal();
  const auto missing_path = fs::absolute(sample_root /
      (window.spec.side + ".missing.bitmap")).lexically_normal();
  atomic_write(values_path, window.values, 0444);
  atomic_write(missing_path, window.missing, 0444);
  J result{{"values_path", values_path.string()},
           {"missing_path", missing_path.string()},
           {"row_count", window.row_count},
           {"valid_count", window.valid_count},
           {"missing_count", window.row_count - window.valid_count},
           {"values_bytes", window.values.size()},
           {"missing_bytes", window.missing.size()},
           {"values_sha256", sha256(window.values)},
           {"missing_sha256", sha256(window.missing)},
           {"row_start", window.row_start},
           {"row_end_exclusive", window.row_end},
           {"valid_start", window.spec.valid_start},
           {"valid_end_exclusive", window.spec.valid_end},
           {"source_manifest_path", series.manifest_path.string()},
           {"source_manifest_sha256", series.manifest_sha256},
           {"entity_id", series.entity_id},
           {"entity_ordinal", series.entity_ordinal},
           {"source_blocks", window.source_blocks},
           {"source_block_count", window.source_blocks.size()},
           {"missing_bitmap_semantics", "one_bit_means_missing_lsb_first"},
           {"values_byte_order", "little"}};
  return result;
}

std::vector<J> materialize_series(const Series& series, const fs::path& output,
                                  std::size_t max_values,
                                  std::uint64_t& verified_source_bytes,
                                  std::uint64_t& verified_source_blocks) {
  require(series.valid_count >= 2 * kMinimumValues,
          "cannot sample a series with fewer than 128 valid values");
  const auto primary_values = std::min<std::uint64_t>(max_values,
                                                       series.valid_count / 2);
  struct Scale {
    std::string name;
    std::uint64_t count;
    J equivalent_scales = J::array();
  };
  std::vector<Scale> scales;
  if (primary_values > kMinimumValues)
    scales.push_back({"u64", kMinimumValues, J::array()});
  scales.push_back({"primary", primary_values,
                    primary_values == kMinimumValues ? J::array({"u64"})
                                                      : J::array()});

  J series_identity{{"column_identity", series.column_identity},
                    {"entity_id", series.entity_id},
                    {"entity_ordinal", series.entity_ordinal}};
  const auto series_id = sha256(series_identity.dump());
  std::vector<WindowOutput> windows;
  windows.reserve(scales.size() * 2);
  for (const auto& scale : scales) {
    const auto head_start = UINT64_C(0);
    const auto head_end = scale.count;
    const auto tail_start = series.valid_count - scale.count;
    const auto tail_end = series.valid_count;
    require(head_end <= tail_start, "head/tail valid ranges overlap");
    WindowSpec head{scale.name + ":head", "head", head_start, head_end,
                    block_for_valid(series, head_start),
                    block_for_valid(series, head_end - 1)};
    WindowSpec tail{scale.name + ":tail", "tail", tail_start, tail_end,
                    block_for_valid(series, tail_start),
                    block_for_valid(series, tail_end - 1)};
    WindowOutput head_output;
    head_output.spec = std::move(head);
    windows.push_back(std::move(head_output));
    WindowOutput tail_output;
    tail_output.spec = std::move(tail);
    windows.push_back(std::move(tail_output));
  }

  for (std::size_t block_position = 0; block_position < series.blocks.size();
       ++block_position) {
    const bool needed = std::any_of(windows.begin(), windows.end(),
                                    [&](const auto& window) {
      return block_position >= window.spec.first_block &&
             block_position <= window.spec.last_block;
    });
    if (!needed) continue;
    const auto& block = series.blocks[block_position];
    const auto values = read_region(block.values_path, block.values_offset,
                                    block.values_length);
    require(sha256(values) == block.values_sha256,
            "values SHA256 mismatch for source block " + block.block_id);
    const auto missing = read_region(block.missing_path, block.missing_offset,
                                     block.missing_length);
    require(sha256(missing) == block.missing_sha256,
            "missing bitmap SHA256 mismatch for source block " + block.block_id);
    require(count_missing(missing, block.row_count) == block.missing_count,
            "missing bitmap count mismatch for source block " + block.block_id);
    if (block.row_count % 8 != 0 && !missing.empty()) {
      const auto used = static_cast<unsigned>(block.row_count % 8);
      const auto high_mask = static_cast<unsigned char>(0xffU << used);
      require((static_cast<unsigned char>(missing.back()) & high_mask) == 0,
              "missing bitmap high padding bits set for source block " +
                  block.block_id);
    }
    verified_source_bytes = checked_add(
        verified_source_bytes,
        checked_add(block.values_length, block.missing_length,
                    "verified source block bytes"),
        "verified source bytes");
    ++verified_source_blocks;

    for (auto& window : windows) {
      if (block_position < window.spec.first_block ||
          block_position > window.spec.last_block)
        continue;
      std::uint64_t local_row_start = 0;
      std::uint64_t local_row_end = block.row_count;
      if (block_position == window.spec.first_block) {
        const auto local_valid_start =
            window.spec.valid_start - block.series_valid_start;
        local_row_start =
            row_for_local_valid(missing, block.row_count, local_valid_start);
      }
      if (block_position == window.spec.last_block) {
        const auto local_valid_last =
            window.spec.valid_end - 1 - block.series_valid_start;
        local_row_end = checked_add(
            row_for_local_valid(missing, block.row_count, local_valid_last), 1,
            "local row end");
      }
      require(local_row_start < local_row_end && local_row_end <= block.row_count,
              "invalid source row slice for " + block.block_id);
      const auto local_valid_start = valid_before_row(missing, local_row_start);
      const auto local_valid_end = valid_before_row(missing, local_row_end);
      const auto local_valid_count = local_valid_end - local_valid_start;
      const auto values_slice_offset = checked_mul(
          local_valid_start, series.width, "source values slice offset");
      const auto values_slice_length = checked_mul(
          local_valid_count, series.width, "source values slice length");
      require(checked_add(values_slice_offset, values_slice_length,
                          "source values slice") <= values.size(),
              "source values slice exceeds verified region " + block.block_id);
      const auto output_row_start = window.row_count;
      const auto output_valid_start = window.valid_count;
      window.values.append(values, checked_size(values_slice_offset,
                                                "source values slice offset"),
                           checked_size(values_slice_length,
                                        "source values slice length"));
      for (std::uint64_t row = local_row_start; row < local_row_end; ++row) {
        append_bitmap_bit(window.missing, window.row_count, missing_at(missing, row));
        ++window.row_count;
      }
      window.valid_count = checked_add(window.valid_count, local_valid_count,
                                       "window valid count");
      const auto global_row_start = block.series_row_start + local_row_start;
      const auto global_row_end = block.series_row_start + local_row_end;
      if (!window.has_rows) {
        window.row_start = global_row_start;
        window.has_rows = true;
      }
      window.row_end = global_row_end;
      J evidence{{"manifest_array_index", block.manifest_array_index},
                 {"block_id", block.block_id},
                 {"block_index", block.block_index},
                 {"values_path", block.values_path.string()},
                 {"values_offset", block.values_offset},
                 {"values_length", block.values_length},
                 {"values_sha256", block.values_sha256},
                 {"missing_bitmap_path", block.missing_path.string()},
                 {"missing_bitmap_offset", block.missing_offset},
                 {"missing_bitmap_length", block.missing_length},
                 {"missing_bitmap_sha256", block.missing_sha256},
                 {"source_row_start", local_row_start},
                 {"source_row_end_exclusive", local_row_end},
                 {"series_row_start", global_row_start},
                 {"series_row_end_exclusive", global_row_end},
                 {"source_valid_start", local_valid_start},
                 {"source_valid_end_exclusive", local_valid_end},
                 {"source_values_slice_offset", values_slice_offset},
                 {"source_values_slice_length", values_slice_length},
                 {"output_row_start", output_row_start},
                 {"output_row_end_exclusive", window.row_count},
                 {"output_valid_start", output_valid_start},
                 {"output_valid_end_exclusive", window.valid_count},
                 {"full_values_region_verified", true},
                 {"full_missing_region_verified", true}};
      if (!block.expected_sequence_sha256.empty())
        evidence["source_expected_sequence_sha256"] =
            block.expected_sequence_sha256;
      window.source_blocks.push_back(std::move(evidence));
    }
  }

  std::vector<J> samples;
  samples.reserve(scales.size());
  for (std::size_t scale_index = 0; scale_index < scales.size(); ++scale_index) {
    const auto& scale = scales[scale_index];
    const auto& head = windows[scale_index * 2];
    const auto& tail = windows[scale_index * 2 + 1];
    require(head.row_end <= tail.row_start, "head/tail row ranges overlap");
    const auto sample_id = "ors135-" + series_id.substr(0, 20) + "-" + scale.name;
    J sample{{"sample_id", sample_id},
             {"dataset_id", series.dataset_id},
             {"column_id", series.column_id},
             {"series_id", series_id},
             {"dtype", series.dtype},
             {"scale", scale.name},
             {"equivalent_scales", scale.equivalent_scales},
             {"element_count", scale.count},
             {"series_row_count", series.row_count},
             {"series_valid_count", series.valid_count},
             {"column_identity", series.column_identity},
             {"entity_identity",
              {{"entity_id", series.entity_id},
               {"entity_ordinal", series.entity_ordinal}}},
             {"selection", "longest_same_column_entity_per_dataset_dtype"},
             {"head", build_sample_part(series, sample_id, output, head)},
             {"tail", build_sample_part(series, sample_id, output, tail)}};
    sample["unselected_valid_ranges"] =
        J::array({{{"start", scale.count},
                   {"end_exclusive", series.valid_count - scale.count}}});
    samples.push_back(std::move(sample));
  }
  return samples;
}

}  // namespace

nlohmann::json build_inputs(const std::filesystem::path& plan_columns,
                            const std::filesystem::path& output,
                            std::size_t max_values) {
  require(max_values >= kMinimumValues && max_values <= kMaximumValues,
          "max_values must be in [64,32768]");
  const auto plan_path = fs::absolute(plan_columns).lexically_normal();
  const auto output_path = fs::absolute(output).lexically_normal();
  require(fs::is_regular_file(plan_path),
          "PLAN_COLUMNS is not a regular file " + plan_path.string());
  if (fs::exists(output_path))
    require(fs::is_directory(output_path),
            "output path is not a directory " + output_path.string());
  else
    fs::create_directories(output_path);

  const auto plan_bytes = read_file(plan_path, kMaximumJsonBytes);
  const auto plan_sha256 = sha256(plan_bytes);
  const auto plan = J::parse(plan_bytes);
  require(plan.is_object() && plan.value("schema", "") ==
                                  "cpp135-valid48-column-plan-v1",
          "unexpected PLAN_COLUMNS schema");
  require(plan.contains("jobs") && plan.at("jobs").is_array() &&
              !plan.at("jobs").empty(),
          "PLAN_COLUMNS jobs must be a nonempty array");
  require(plan.contains("manifest_sha256") &&
              plan.at("manifest_sha256").is_object(),
          "PLAN_COLUMNS manifest_sha256 must be an object");
  if (plan.contains("counts") && plan.at("counts").is_object() &&
      plan.at("counts").value("datasets", 0) == 135 &&
      plan.at("counts").value("value_columns", 0) == 3519)
    require(plan_sha256 == kFrozenPlanSha256,
            "full current135 PLAN_COLUMNS SHA256 is not the frozen reference");

  std::set<std::string> datasets;
  std::set<std::string> manifest_paths;
  std::map<SelectionKey, Series> longest;
  std::uint64_t verified_manifest_bytes = 0;
  std::uint64_t manifest_count = 0;
  for (const auto& job : plan.at("jobs")) {
    require(job.is_object(), "PLAN_COLUMNS job is not an object");
    const auto dataset_id = required_text(job, "dataset_id", "plan job");
    datasets.insert(dataset_id);
    require(job.contains("manifests") && job.at("manifests").is_array() &&
                !job.at("manifests").empty(),
            "plan job manifests must be a nonempty array");
    for (const auto& manifest_item : job.at("manifests")) {
      require(manifest_item.is_string(), "plan job manifest path is not a string");
      const auto declared_path = manifest_item.get<std::string>();
      require(!declared_path.empty(), "plan job manifest path is empty");
      require(manifest_paths.insert(declared_path).second,
              "duplicate plan manifest path " + declared_path);
      require(plan.at("manifest_sha256").contains(declared_path),
              "manifest_sha256 map lacks " + declared_path);
      const auto expected_sha = required_text(plan.at("manifest_sha256"),
                                              declared_path.c_str(),
                                              "manifest_sha256 map");
      require(valid_sha256(expected_sha),
              "invalid manifest SHA256 for " + declared_path);
      const auto manifest_path = source_path(plan_path, declared_path);
      const auto manifest_bytes = read_file(manifest_path, kMaximumJsonBytes);
      require(sha256(manifest_bytes) == expected_sha,
              "manifest SHA256 mismatch for " + manifest_path.string());
      verified_manifest_bytes = checked_add(verified_manifest_bytes,
                                            manifest_bytes.size(),
                                            "verified manifest bytes");
      ++manifest_count;
      const auto manifest = J::parse(manifest_bytes);
      for (auto& series : series_from_manifest(manifest, manifest_path,
                                               expected_sha, job)) {
        const SelectionKey key{series.dataset_id, series.dtype};
        const auto found = longest.find(key);
        if (found == longest.end())
          longest.emplace(key, std::move(series));
        else if (preferred_series(series, found->second))
          found->second = std::move(series);
      }
    }
  }
  require(plan.at("manifest_sha256").size() == manifest_paths.size(),
          "manifest_sha256 map contains paths outside jobs[].manifests");
  if (plan.contains("counts") && plan.at("counts").is_object()) {
    if (plan.at("counts").contains("datasets"))
      require(required_u64(plan.at("counts"), "datasets", "plan counts") ==
                  datasets.size(),
              "PLAN_COLUMNS dataset count mismatch");
    if (plan.at("counts").contains("value_columns"))
      require(required_u64(plan.at("counts"), "value_columns", "plan counts") ==
                  manifest_paths.size(),
              "PLAN_COLUMNS value_columns count mismatch");
  }

  static constexpr std::array<std::string_view, 3> dtypes = {
      "float32", "float64", "int64"};
  J gaps = J::array();
  std::set<std::string> covered_datasets;
  for (const auto& dataset_id : datasets) {
    for (const auto dtype_view : dtypes) {
      const std::string dtype(dtype_view);
      const auto found = longest.find({dataset_id, dtype});
      if (found == longest.end()) {
        gaps.push_back({{"dataset_id", dataset_id},
                        {"dtype", dtype},
                        {"reason", "dtype_absent"},
                        {"longest_same_entity_valid_count", 0}});
      } else if (found->second.valid_count < 2 * kMinimumValues) {
        gaps.push_back(
            {{"dataset_id", dataset_id},
             {"dtype", dtype},
             {"reason", "no_same_entity_series_with_128_valid_values"},
             {"longest_same_entity_valid_count", found->second.valid_count},
             {"column_id", found->second.column_id},
             {"entity_id", found->second.entity_id},
             {"entity_ordinal", found->second.entity_ordinal}});
      } else {
        covered_datasets.insert(dataset_id);
      }
    }
  }
  std::vector<std::string> missing_datasets;
  std::set_difference(datasets.begin(), datasets.end(), covered_datasets.begin(),
                      covered_datasets.end(), std::back_inserter(missing_datasets));
  require(missing_datasets.empty(),
          "at least one dataset has no same-entity series with 128 valid values: " +
              (missing_datasets.empty() ? std::string() : missing_datasets.front()));

  J samples = J::array();
  std::uint64_t verified_source_bytes = 0;
  std::uint64_t verified_source_blocks = 0;
  for (const auto& [key, series] : longest) {
    (void)key;
    if (series.valid_count < 2 * kMinimumValues) continue;
    for (auto& sample : materialize_series(series, output_path, max_values,
                                           verified_source_bytes,
                                           verified_source_blocks))
      samples.push_back(std::move(sample));
  }

  J dataset_ids = J::array();
  for (const auto& dataset_id : datasets) dataset_ids.push_back(dataset_id);
  J result{{"schema", "operator-recommend135-inputs-v1"},
           {"builder", "operator_recommend135_v1::build_inputs"},
           {"plan_columns_path", plan_path.string()},
           {"plan_columns_sha256", plan_sha256},
           {"max_values", max_values},
           {"minimum_values_per_window", kMinimumValues},
           {"selection_scope",
            "one_longest_same_column_entity_series_per_dataset_dtype; representative_not_all_columns"},
           {"selection_length_metric", "valid_count"},
           {"head_tail_contract",
            "valid-position windows; head=[0,N), tail=[L-N,L); exact enclosing row ranges; non-overlapping"},
           {"source_verification_contract",
            "PLAN manifest SHA256 and full bytes of every source values/missing region used by a window"},
           {"dataset_ids", dataset_ids},
           {"samples", samples},
           {"gaps", gaps},
           {"coverage",
            {{"plan_dataset_count", datasets.size()},
             {"covered_dataset_count", covered_datasets.size()},
             {"missing_dataset_ids", missing_datasets},
             {"plan_manifest_count", manifest_count},
             {"sample_count", samples.size()},
             {"gap_count", gaps.size()}}},
           {"verification",
            {{"verified_manifest_count", manifest_count},
             {"verified_manifest_bytes", verified_manifest_bytes},
             {"verified_source_block_count", verified_source_blocks},
             {"verified_source_region_bytes", verified_source_bytes}}}};
  result["manifest_sha256_scope"] =
      "canonical compact JSON before manifest_sha256 is added";
  result["manifest_sha256"] = sha256(result.dump());
  atomic_write(output_path / "INPUTS.json", result.dump(2) + "\n", 0444);
  return result;
}

}  // namespace operator_recommend135_v1
