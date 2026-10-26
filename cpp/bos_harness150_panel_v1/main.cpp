#include "api.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace uc = unified_compression_v1;
using Json = nlohmann::json;
using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::string_view kSegmentManifestSchema =
    "native-baseline86-scaled-column-segment-manifest-v1";
constexpr std::string_view kNew15ManifestSchema =
    "project150-new15-algorithm85-entity-aware-column-manifest-v1";
constexpr std::string_view kFeatureRowSchema =
    "solid-dtype-correct-column-features-v1";
constexpr std::string_view kResultSchema = "bos-harness150-panel-result-v1";
constexpr std::string_view kRunSchema = "bos-harness150-panel-run-v1";
constexpr std::string_view kSummarySchema = "bos-harness150-panel-summary-input-v1";
constexpr std::string_view kBosImplementation =
    "versioned_bos2_bfe_dsc3_hybrid";
constexpr std::string_view kBosProfileVersion = "bos-bfe-ngf-hybrid-v2";
constexpr std::string_view kBosFloatRoute = "bos-bfe-ngf-hybrid-v2";
constexpr std::string_view kBosIntegerRoute = "project_bos2_partitioned";
constexpr std::string_view kBosIntegerConfiguration =
    "delta_zigzag_split_partition_for_bitpack_v1";
constexpr std::string_view kBosFloatConfiguration =
    "min_complete_NGF(DSC3v3+BOS2,BFE1v3+BOS2)";
constexpr std::uint32_t kWarmups = 1;
constexpr std::uint32_t kRepeats = 3;
const std::array<std::string, 3> kAllowedMethods = {
    "algorithm:BOS", "algorithm:LZ4", "algorithm:Zstd"};
constexpr std::uint64_t kDefaultAddressSpaceLimit = UINT64_C(2147483648);
constexpr std::uint64_t kMaximumBlockValues = UINT64_C(10000);

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

void require_exact_bos_contract(const Json& contract,
                                std::string_view context) {
  require(contract.is_object(), std::string(context) + ": BOS contract missing");
  require(contract.value("implementation", "") == kBosImplementation &&
              contract.value("profile_version", "") == kBosProfileVersion &&
              contract.value("float_route_id", "") == kBosFloatRoute &&
              contract.value("integer_route_id", "") == kBosIntegerRoute &&
              contract.value("integer_configuration", "") ==
                  kBosIntegerConfiguration &&
              contract.value("float_configuration", "") ==
                  kBosFloatConfiguration,
          std::string(context) + ": exact new-BOS runtime identity differs");
}

std::string now_utc() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm value{};
#ifdef _WIN32
  gmtime_s(&value, &seconds);
#else
  gmtime_r(&seconds, &value);
#endif
  char output[32]{};
  std::strftime(output, sizeof(output), "%Y-%m-%dT%H:%M:%SZ", &value);
  return output;
}

std::string sha256(std::span<const std::uint8_t> input) {
  std::array<std::uint8_t, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0;
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  require(context != nullptr, "cannot allocate SHA-256 context");
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, input.data(), input.size()) == 1 &&
                  EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
  EVP_MD_CTX_free(context);
  require(ok && size == 32, "SHA-256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (unsigned index = 0; index < size; ++index) {
    result.push_back(hex[digest[index] >> 4]);
    result.push_back(hex[digest[index] & 15]);
  }
  return result;
}

std::string sha256(std::string_view input) {
  return sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(input.data()), input.size()));
}

class Digest {
 public:
  Digest() : context_(EVP_MD_CTX_new()) {
    require(context_ != nullptr &&
                EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) == 1,
            "cannot initialize SHA-256");
  }
  Digest(const Digest&) = delete;
  Digest& operator=(const Digest&) = delete;
  ~Digest() { EVP_MD_CTX_free(context_); }
  void update(std::span<const std::uint8_t> bytes) {
    require(EVP_DigestUpdate(context_, bytes.data(), bytes.size()) == 1,
            "SHA-256 update failed");
  }
  std::string finish() {
    require(!finished_, "SHA-256 already finalized");
    finished_ = true;
    std::array<std::uint8_t, EVP_MAX_MD_SIZE> output{};
    unsigned size = 0;
    require(EVP_DigestFinal_ex(context_, output.data(), &size) == 1 && size == 32,
            "SHA-256 finalization failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (unsigned i = 0; i < size; ++i) {
      result.push_back(hex[output[i] >> 4]);
      result.push_back(hex[output[i] & 15]);
    }
    return result;
  }
 private:
  EVP_MD_CTX* context_ = nullptr;
  bool finished_ = false;
};

Bytes read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot read " + path.string());
  Bytes bytes{std::istreambuf_iterator<char>(input), {}};
  require(!input.bad(), "failed while reading " + path.string());
  return bytes;
}

Json read_json(const fs::path& path) {
  std::ifstream input(path);
  require(static_cast<bool>(input), "cannot read " + path.string());
  try {
    return Json::parse(input);
  } catch (const std::exception& error) {
    fail("invalid JSON " + path.string() + ": " + error.what());
  }
}

void atomic_write(const fs::path& path, std::string_view contents) {
  fs::create_directories(path.parent_path());
  static std::uint64_t sequence = 0;
  const fs::path temporary = path.string() + ".tmp." +
      std::to_string(::getpid()) + "." + std::to_string(++sequence);
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot create " + temporary.string());
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.flush();
    require(static_cast<bool>(output), "cannot write " + temporary.string());
  }
  std::error_code error;
  fs::rename(temporary, path, error);
  if (error) {
    fs::remove(temporary);
    fail("cannot commit " + path.string() + ": " + error.message());
  }
}

void atomic_write_json(const fs::path& path, const Json& value) {
  atomic_write(path, value.dump(2) + "\n");
}

void append_json_line(const fs::path& path, const Json& value) {
  const std::string line = value.dump() + "\n";
  const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  require(descriptor >= 0, "cannot append " + path.string() + ": " +
                               std::strerror(errno));
  std::size_t position = 0;
  while (position < line.size()) {
    const ssize_t count = ::write(descriptor, line.data() + position,
                                  line.size() - position);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      const std::string reason = std::strerror(errno);
      ::close(descriptor);
      fail("cannot append " + path.string() + ": " + reason);
    }
    position += static_cast<std::size_t>(count);
  }
  const int sync_status = ::fsync(descriptor);
  const int close_status = ::close(descriptor);
  require(sync_status == 0 && close_status == 0,
          "cannot durably append " + path.string());
}

std::uint64_t natural(const Json& value, std::string_view field) {
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    require(signed_value >= 0, std::string(field) + " is negative");
    return static_cast<std::uint64_t>(signed_value);
  }
  fail(std::string(field) + " is not an integer");
}

void add_checked(std::uint64_t& total, std::uint64_t value,
                 std::string_view field) {
  require(total <= std::numeric_limits<std::uint64_t>::max() - value,
          std::string(field) + " overflows");
  total += value;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
  require(!values.empty(), "median of empty sequence");
  auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

std::string safe_component(std::string_view input) {
  std::string result;
  result.reserve(std::min<std::size_t>(input.size(), 80));
  for (const unsigned char byte : input) {
    if (result.size() >= 64) break;
    if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
        (byte >= '0' && byte <= '9') || byte == '-' || byte == '_')
      result.push_back(static_cast<char>(byte));
    else
      result.push_back('_');
  }
  if (result.empty()) result = "column";
  return result + "_" + sha256(input).substr(0, 12);
}

std::string method_leaf(std::string_view method) {
  const auto colon = method.find(':');
  return safe_component(colon == method.npos ? method : method.substr(colon + 1));
}

Bytes read_extent(const fs::path& path, std::uint64_t offset,
                  std::uint64_t length) {
  require(length <= std::numeric_limits<std::size_t>::max(),
          "extent exceeds addressable memory");
  std::error_code error;
  const auto file_bytes = fs::file_size(path, error);
  require(!error && offset <= file_bytes && length <= file_bytes - offset,
          "extent exceeds file " + path.string());
  require(offset <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamoff>::max()) &&
              length <= static_cast<std::uint64_t>(
                          std::numeric_limits<std::streamsize>::max()),
          "extent exceeds stream limits");
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open " + path.string());
  input.seekg(static_cast<std::streamoff>(offset));
  require(static_cast<bool>(input), "cannot seek " + path.string());
  Bytes output(static_cast<std::size_t>(length));
  if (length != 0)
    input.read(reinterpret_cast<char*>(output.data()),
               static_cast<std::streamsize>(length));
  require(static_cast<bool>(input), "cannot read extent " + path.string());
  return output;
}

struct BlockInput {
  std::uint64_t index = 0;
  std::uint64_t count = 0;
  Bytes values;
};

struct ColumnInput {
  fs::path manifest_path;
  std::string manifest_sha256;
  std::string dataset_id;
  std::string column_id;
  std::string stable_id;
  std::string source_column;
  std::string dtype;
  std::string source_dtype;
  std::string adapter;
  std::string input_schema;
  std::string input_profile;
  std::string source_stream_sha256;
  std::uint64_t valid_values = 0;
  std::uint64_t original_numeric_bytes = 0;
  std::uint64_t source_native_numeric_bytes = 0;
  std::vector<BlockInput> blocks;
};

unsigned dtype_width(std::string_view dtype) {
  if (dtype == "int64" || dtype == "float64") return 8;
  if (dtype == "float32") return 4;
  fail("panel accepts only int64, float32, and float64; got " +
       std::string(dtype));
}

unsigned source_dtype_width(std::string_view dtype) {
  if (dtype == "int8" || dtype == "uint8") return 1;
  if (dtype == "int16" || dtype == "uint16") return 2;
  if (dtype == "int32" || dtype == "uint32" || dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "uint64" || dtype == "float64") return 8;
  fail("unsupported source physical dtype: " + std::string(dtype));
}

ColumnInput load_column(const fs::path& manifest_path) {
  const Bytes manifest_bytes = read_file(manifest_path);
  Json manifest;
  try {
    manifest = Json::parse(manifest_bytes);
  } catch (const std::exception& error) {
    fail("invalid manifest " + manifest_path.string() + ": " + error.what());
  }
  const std::string schema = manifest.value("schema", "");
  const bool segment_manifest = schema == kSegmentManifestSchema;
  const bool new15_manifest = schema == kNew15ManifestSchema;
  require(segment_manifest || new15_manifest,
          "unexpected manifest schema: " + manifest_path.string());
  if (segment_manifest)
    require(manifest.contains("column_identity") &&
                manifest.at("column_identity").is_object(),
            "missing column_identity: " + manifest_path.string());
  require(manifest.contains("blocks") && manifest.at("blocks").is_array() &&
              !manifest.at("blocks").empty(),
          "manifest has no blocks: " + manifest_path.string());

  ColumnInput result;
  result.manifest_path = fs::absolute(manifest_path).lexically_normal();
  result.manifest_sha256 = sha256(manifest_bytes);
  result.input_schema = schema;
  if (segment_manifest) {
    const auto& identity = manifest.at("column_identity");
    result.dataset_id =
        identity.value("dataset_id", manifest.value("dataset_id", ""));
    result.column_id =
        identity.value("column_id", manifest_path.stem().string());
    result.stable_id =
        identity.value("stable_id", identity.value("column", result.column_id));
    result.source_column = identity.value("source_column", result.stable_id);
    result.dtype = identity.value("dtype", "");
    result.source_dtype = result.dtype;
    result.adapter = "identity";
    result.input_profile = manifest.value("input_profile", "");
    require(result.input_profile == "bos-nonnull-concat-reblock10000-v1" ||
                result.input_profile ==
                    "bos-float150-clean-concat-reblock10000-v1",
            "segment manifest is outside the frozen clean150 profiles");
  } else {
    result.dataset_id = manifest.value("dataset_id", "");
    result.column_id = manifest.value("column_id", "");
    result.stable_id = manifest.value("source_stable_id", "");
    result.source_column = manifest.value("source_column", result.stable_id);
    result.dtype = manifest.value("dtype", manifest.value("canonical_dtype", ""));
    result.source_dtype = manifest.value("source_dtype", "");
    result.adapter = manifest.value("adapter", "");
    result.input_profile =
        "project150-new15-algorithm85-entity-aware-v1";
    require(manifest.value("status", "") == "ready" &&
                manifest.value("one_entity_per_block", false),
            "new15 manifest is not a ready one-entity-per-block input");
    require(manifest.value("canonical_dtype", "") == result.dtype,
            "new15 canonical dtype differs from its public dtype");
    require(!result.source_dtype.empty() && !result.adapter.empty(),
            "new15 source dtype/adapter binding is incomplete");
    if (result.source_dtype == result.dtype)
      require(result.adapter == "identity",
              "new15 identity dtype has a non-identity adapter");
    else
      require(result.dtype == "int64" &&
                  (result.adapter == "exact_int8_to_int64" ||
                   result.adapter == "exact_int16_to_int64" ||
                   result.adapter == "exact_int32_to_int64" ||
                   result.adapter == "exact_uint8_to_int64" ||
                   result.adapter == "exact_uint16_to_int64" ||
                   result.adapter == "exact_uint32_to_int64"),
              "new15 manifest is outside the frozen exact integer adapter set");
  }
  require(!result.dataset_id.empty() && !result.column_id.empty(),
          "manifest column identity is incomplete");
  require(!result.stable_id.empty() && !result.source_column.empty(),
          "manifest stable/source column identity is incomplete");
  const unsigned width = dtype_width(result.dtype);

  Digest stream_digest;
  std::uint64_t expected_index = 0;
  std::optional<fs::path> contiguous_values_path;
  std::uint64_t expected_values_offset = 0;
  std::size_t manifest_array_index = 0;
  for (const auto& block : manifest.at("blocks")) {
    require(block.is_object(), "manifest block is not an object");
    const auto index = natural(block.at("block_index"), "block_index");
    if (segment_manifest) {
      require(index == expected_index++,
              "segment manifest block indexes are not contiguous");
    } else {
      const auto entity = block.value("entity_id", "");
      require(!entity.empty(), "new15 block lacks entity_id");
      require(index == expected_index++,
              "new15 block indexes are not globally contiguous");
      require(block.contains("entity_ordinal"),
              "new15 block lacks entity_ordinal");
      (void)natural(block.at("entity_ordinal"), "entity_ordinal");
    }
    require(block.value("dtype", result.dtype) == result.dtype,
            "block dtype differs from column dtype");
    const auto valid = natural(block.at("valid_count"), "valid_count");
    const auto rows = natural(block.value("row_count", Json(valid)), "row_count");
    const auto missing = natural(block.value("missing_count", Json(0)),
                                 "missing_count");
    require(valid > 0 && valid <= kMaximumBlockValues && valid == rows &&
                missing == 0,
            "panel requires a dense nonnull block of at most 10000 values");
    const auto offset = natural(block.at("values_offset"), "values_offset");
    const auto length = natural(block.at("values_length"), "values_length");
    require(valid <= std::numeric_limits<std::uint64_t>::max() / width &&
                length == valid * width,
            "block byte length differs from dtype/count");
    if (new15_manifest)
      require(natural(block.at("canonical_input_bytes"),
                      "canonical_input_bytes") == length,
              "new15 block canonical_input_bytes differs from values_length");
    if (new15_manifest) {
      require(block.value("source_dtype", "") == result.source_dtype &&
                  block.value("adapter", "") == result.adapter,
              "new15 block source dtype/adapter differs from its manifest");
      const auto source_width = source_dtype_width(result.source_dtype);
      require(valid <= std::numeric_limits<std::uint64_t>::max() /
                           source_width &&
                  natural(block.at("source_native_numeric_bytes"),
                          "source_native_numeric_bytes") ==
                      valid * source_width,
              "new15 block source-native bytes differ from physical dtype/count");
    }
    fs::path values_path = block.at("values_path").get<std::string>();
    if (values_path.is_relative()) values_path = manifest_path.parent_path() / values_path;
    values_path = fs::absolute(values_path).lexically_normal();
    if (!contiguous_values_path) contiguous_values_path = values_path;
    require(values_path == *contiguous_values_path &&
                offset == expected_values_offset,
            "manifest blocks are not one contiguous canonical value stream");
    require(expected_values_offset <=
                std::numeric_limits<std::uint64_t>::max() - length,
            "canonical value-stream byte count overflows");
    expected_values_offset += length;
    auto values = read_extent(values_path, offset, length);
    if (block.contains("values_sha256"))
      require(sha256(values) == block.at("values_sha256").get<std::string>(),
              "block values SHA-256 mismatch");
    if (block.contains("expected_sequence_sha256"))
      require(sha256(values) ==
                  block.at("expected_sequence_sha256").get<std::string>(),
              "block expected-sequence SHA-256 mismatch");
    stream_digest.update(values);
    add_checked(result.valid_values, valid, "column valid count");
    add_checked(result.original_numeric_bytes, length,
                "column canonical byte count");
    add_checked(result.source_native_numeric_bytes, natural(
        block.value("source_native_numeric_bytes", Json(length)),
        "source_native_numeric_bytes"), "column source-native byte count");
    result.blocks.push_back(
        {static_cast<std::uint64_t>(manifest_array_index++), valid,
         std::move(values)});
  }
  result.source_stream_sha256 = stream_digest.finish();
  if (segment_manifest)
    require(result.source_stream_sha256 ==
                manifest.at("source_valid_stream_sha256").get<std::string>(),
            "column source stream SHA-256 mismatch");
  else
    require(result.source_stream_sha256 ==
                manifest.at("grouped_values_sha256").get<std::string>(),
            "new15 grouped stream SHA-256 mismatch");
  if (segment_manifest) {
    const auto& totals = manifest.at("source_totals");
    require(result.valid_values == natural(totals.at("valid_count"), "valid_count"),
            "column valid count differs from source_totals");
  } else {
    fs::path grouped_path = manifest.at("grouped_values_path").get<std::string>();
    if (grouped_path.is_relative())
      grouped_path = manifest_path.parent_path() / grouped_path;
    require(fs::absolute(grouped_path).lexically_normal() ==
                *contiguous_values_path,
            "new15 grouped_values_path differs from its blocks");
    require(result.valid_values == natural(manifest.at("valid_count"), "valid_count"),
            "new15 valid count differs from its blocks");
    require(result.original_numeric_bytes ==
                natural(manifest.at("canonical_input_bytes"),
                        "canonical_input_bytes"),
            "new15 canonical byte count differs from its blocks");
    require(result.source_native_numeric_bytes ==
                natural(manifest.at("source_native_numeric_bytes"),
                        "source_native_numeric_bytes"),
            "new15 source-native byte count differs from its blocks");
  }
  return result;
}

uc::Column make_block(const ColumnInput& column, const BlockInput& block) {
  uc::Column result;
  result.dtype = column.dtype;
  result.row_count = block.count;
  result.values = block.values;
  result.missing.assign(static_cast<std::size_t>((block.count + 7) / 8), 0);
  uc::validate_column(result);
  return result;
}

struct AuditAggregate {
  std::uint64_t public_frames = 0;
  std::uint64_t strict_frames = 0;
  std::uint64_t fallback_frames = 0;
  std::uint64_t unknown_fallback_frames = 0;
  std::uint64_t native_raw_blocks = 0;
  bool native_raw_is_harness_bypass = false;
  std::map<std::string, std::uint64_t> strict_status_counts;
  std::map<std::string, std::uint64_t> frontend_counts;
  std::map<std::string, std::uint64_t> fallback_semantics_counts;

  void add(const Json& audit) {
    ++public_frames;
    if (audit.value("strict", false)) ++strict_frames;
    if (!audit.contains("fallback_used") || audit.at("fallback_used").is_null())
      ++unknown_fallback_frames;
    else if (audit.at("fallback_used").get<bool>())
      ++fallback_frames;
    native_raw_blocks += audit.value("native_raw_block_count", std::uint64_t(0));
    native_raw_is_harness_bypass = native_raw_is_harness_bypass ||
        audit.value("native_raw_is_harness_bypass", false);
    if (audit.contains("strict_status") && audit.at("strict_status").is_string())
      ++strict_status_counts[audit.at("strict_status").get<std::string>()];
    if (audit.contains("selected_frontend") &&
        audit.at("selected_frontend").is_string())
      ++frontend_counts[audit.at("selected_frontend").get<std::string>()];
    if (audit.contains("fallback_semantics") &&
        audit.at("fallback_semantics").is_string())
      ++fallback_semantics_counts[
          audit.at("fallback_semantics").get<std::string>()];
  }

  Json json() const {
    return {{"public_frame_count", public_frames},
            {"strict_frame_count", strict_frames},
            {"fallback_frame_count", fallback_frames},
            {"unknown_fallback_frame_count", unknown_fallback_frames},
            {"native_raw_block_count", native_raw_blocks},
            {"native_raw_is_harness_bypass", native_raw_is_harness_bypass},
            {"strict_status_counts", strict_status_counts},
            {"selected_frontend_counts", frontend_counts},
            {"fallback_semantics_counts", fallback_semantics_counts}};
  }
};

Json base_result(const ColumnInput& column, std::string_view method,
                 const uc::MethodInfo& info) {
  const bool catalog_dtype_eligible =
      std::find(info.dtypes.begin(), info.dtypes.end(), column.dtype) !=
      info.dtypes.end();
  return {{"schema", kResultSchema},
          {"result_key", sha256(column.manifest_sha256 + "\n" +
                                std::string(method))},
          {"created_at_utc", now_utc()},
          {"dataset_id", column.dataset_id},
          {"column_id", column.column_id},
          {"stable_id", column.stable_id},
          {"source_column", column.source_column},
          {"dtype", column.dtype},
          {"codec_input_dtype", column.dtype},
          {"source_physical_dtype", column.source_dtype},
          {"input_adapter_provenance", column.adapter},
          {"canonical_typed_plane_executed", true},
          {"implicit_cast_performed_by_panel", false},
          {"input_manifest_schema", column.input_schema},
          {"input_profile", column.input_profile},
          {"method", method},
          {"method_display_name", info.display_name},
          {"method_admitted_dtypes", info.dtypes},
          {"method_contract", info.contract},
          {"manifest_path", column.manifest_path.string()},
          {"manifest_sha256", column.manifest_sha256},
          {"source_valid_stream_sha256", column.source_stream_sha256},
          {"valid_values", column.valid_values},
          {"original_numeric_bytes", column.original_numeric_bytes},
          {"canonical_input_bytes", column.original_numeric_bytes},
          {"source_native_numeric_bytes", column.source_native_numeric_bytes},
          {"frame_count", column.blocks.size()},
          {"catalog_dtype_eligible", catalog_dtype_eligible},
          {"explicit_candidate_requested", true},
          {"execution_path", "not_executed"},
          {"strict_path", false},
          {"eligible_for_strict_selector", false},
          {"eligible_for_lossless_selector", false},
          {"native_raw_lossless_path", false},
          {"warmups", kWarmups},
          {"repeats", kRepeats},
          {"threads", 1},
          {"frame_contract", "one complete self-contained UCC1 frame per manifest block"},
          {"timing_contract", "W1/R3/T1; sample-wise sum over blocks; median of three; source I/O and independent verification excluded"},
          {"ratio_contract", "sum complete UCC1 frame bytes / exact source numeric bytes"}};
}

Json measure_method(const ColumnInput& column, std::string_view method,
                    const uc::Registry& registry) {
  const auto& info = registry.resolve(method);
  Json output = base_result(column, method, info);
  std::vector<std::uint64_t> compression_samples(kRepeats, 0);
  std::vector<std::uint64_t> decompression_samples(kRepeats, 0);
  std::uint64_t compressed_bytes = 0;
  AuditAggregate audit;
  Digest frames_digest;
  try {
    for (const auto& block : column.blocks) {
      const auto source = make_block(column, block);
      const auto measured = registry.compress(source, method, {kWarmups, kRepeats});
      require(measured.bit_exact, "Registry did not certify bit-exact recovery");
      require(measured.compression_samples_ns.size() == kRepeats &&
                  measured.decompression_samples_ns.size() == kRepeats,
              "Registry returned an incomplete timing sample set");
      const auto independently_restored = registry.decompress(measured.compressed_data);
      require(independently_restored == source,
              "independent public decode differs from source bits/schema");
      require(measured.compressed_bytes == measured.compressed_data.size(),
              "Registry compressed byte accounting differs from UCC1 frame");
      compressed_bytes += measured.compressed_bytes;
      for (std::size_t sample = 0; sample < kRepeats; ++sample) {
        compression_samples[sample] += measured.compression_samples_ns[sample];
        decompression_samples[sample] += measured.decompression_samples_ns[sample];
      }
      frames_digest.update(measured.compressed_data);
      audit.add(measured.execution_audit);
    }
    const bool strict_path = audit.strict_frames == audit.public_frames;
    const bool native_raw_path = !strict_path && method == "algorithm:BOS" &&
        audit.native_raw_blocks != 0 && !audit.native_raw_is_harness_bypass;
    const std::string path_class = strict_path ? "strict" :
        native_raw_path ? "lossless_native_raw" : "lossless_non_strict_other";
    output.update({{"status", "ok"},
                   {"lossless_bit_exact", true},
                   {"independent_decode_verified", true},
                   {"compressed_bytes", compressed_bytes},
                   {"compressed_to_original_ratio",
                    static_cast<double>(compressed_bytes) /
                        static_cast<double>(column.original_numeric_bytes)},
                   {"compression_factor",
                    static_cast<double>(column.original_numeric_bytes) /
                        static_cast<double>(compressed_bytes)},
                   {"compressed_to_source_native_reference_ratio",
                    static_cast<double>(compressed_bytes) /
                        static_cast<double>(
                            column.source_native_numeric_bytes)},
                   {"source_native_reference_ratio_contract",
                    "accounting reference only; execution used canonical typed-plane bytes and performed no cast"},
                   {"compression_samples_ns", compression_samples},
                   {"decompression_samples_ns", decompression_samples},
                   {"compression_time_ns", median(compression_samples)},
                   {"decompression_time_ns", median(decompression_samples)},
                   {"complete_frames_sha256", frames_digest.finish()},
                   {"execution_path", path_class},
                   {"strict_path", strict_path},
                   {"eligible_for_strict_selector", strict_path},
                   {"eligible_for_lossless_selector", true},
                   {"native_raw_lossless_path", native_raw_path},
                   {"explicit_measurement_retained_if_non_strict", true},
                   {"execution_audit", audit.json()}});
  } catch (const uc::Error& error) {
    output.update({{"status", "failed"},
                   {"lossless_bit_exact", false},
                   {"independent_decode_verified", false},
                   {"error_code", error.code()},
                   {"error", error.what()}});
  } catch (const std::exception& error) {
    output.update({{"status", "failed"},
                   {"lossless_bit_exact", false},
                   {"independent_decode_verified", false},
                   {"error_code", "runner_error"},
                   {"error", error.what()}});
  }
  return output;
}

struct Options {
  std::vector<fs::path> manifests;
  std::vector<fs::path> manifest_roots;
  std::optional<fs::path> manifest_list;
  fs::path output_dir;
  std::vector<std::string> methods =
      std::vector<std::string>(kAllowedMethods.begin(), kAllowedMethods.end());
  std::optional<std::size_t> canary;
  std::optional<std::size_t> expected_columns;
  std::uint64_t address_space_limit_bytes = kDefaultAddressSpaceLimit;
  bool resume = false;
  bool selftest = false;
};

struct FeatureExpectation {
  std::string dataset_id;
  std::string dtype;
  std::uint64_t valid_count = 0;
  unsigned width = 0;
  std::string manifest_sha256;
};

struct ManifestSelection {
  std::vector<fs::path> paths;
  std::map<std::string, FeatureExpectation> feature_expectations;
  std::optional<Json> list_source_binding;
};

void usage(std::ostream& output) {
  output <<
R"(Usage: bos_harness150_panel [OPTIONS]

Required campaign options:
  --output-dir PATH          Isolated result directory.
  --manifest-root PATH       Recursively discover *.json segment manifests.
  --manifest PATH            Add one segment manifest; may be repeated.
  --manifest-list PATH       Frozen FEATURES.jsonl, a plain path list, or a
                             JSON plan with jobs[].manifests.

Measurement options:
  --methods CSV              Closed subset/order of algorithm:BOS,
                             algorithm:LZ4,algorithm:Zstd.
  --expected-columns N       Refuse a non-canary scope with another count.
  --canary N                 Run the first N manifests in sorted order.
  --address-space-limit-bytes N
                             Enforce an address-space ceiling (default 2 GiB).
  --resume                   Rebuild JSONL from atomic result files and resume.

Other:
  --selftest                 Build a small int64/float32/float64 fixture and run it.
  -h, --help                 Show this help.

The timing contract is fixed at W1/R3/T1 and one process/native thread.
)";
}

std::size_t parse_size(std::string_view text, std::string_view name) {
  std::size_t value = 0;
  require(!text.empty(), std::string(name) + " requires an integer");
  for (const char c : text) {
    require(c >= '0' && c <= '9', std::string(name) + " requires an integer");
    require(value <= (std::numeric_limits<std::size_t>::max() - (c - '0')) / 10,
            std::string(name) + " is too large");
    value = value * 10 + static_cast<unsigned>(c - '0');
  }
  return value;
}

std::vector<std::string> split_csv(std::string_view input) {
  std::vector<std::string> output;
  std::size_t position = 0;
  while (position <= input.size()) {
    const auto comma = input.find(',', position);
    const auto end = comma == input.npos ? input.size() : comma;
    auto token = input.substr(position, end - position);
    while (!token.empty() && token.front() == ' ') token.remove_prefix(1);
    while (!token.empty() && token.back() == ' ') token.remove_suffix(1);
    require(!token.empty(), "empty method token");
    output.emplace_back(token);
    if (comma == input.npos) break;
    position = comma + 1;
  }
  return output;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](std::string_view name) -> std::string {
      require(index + 1 < argc, std::string(name) + " requires a value");
      return argv[++index];
    };
    if (argument == "-h" || argument == "--help") {
      usage(std::cout);
      std::exit(0);
    } else if (argument == "--manifest") {
      options.manifests.emplace_back(next(argument));
    } else if (argument == "--manifest-root") {
      options.manifest_roots.emplace_back(next(argument));
    } else if (argument == "--manifest-list") {
      require(!options.manifest_list, "--manifest-list specified twice");
      options.manifest_list = next(argument);
    } else if (argument == "--output-dir") {
      options.output_dir = next(argument);
    } else if (argument == "--methods") {
      options.methods = split_csv(next(argument));
    } else if (argument == "--canary") {
      options.canary = parse_size(next(argument), argument);
      require(*options.canary > 0, "--canary must be positive");
    } else if (argument == "--expected-columns") {
      options.expected_columns = parse_size(next(argument), argument);
      require(*options.expected_columns > 0,
              "--expected-columns must be positive");
    } else if (argument == "--address-space-limit-bytes") {
      options.address_space_limit_bytes = parse_size(next(argument), argument);
      require(options.address_space_limit_bytes > 0,
              "--address-space-limit-bytes must be positive");
    } else if (argument == "--resume") {
      options.resume = true;
    } else if (argument == "--selftest") {
      options.selftest = true;
    } else {
      fail("unknown argument: " + argument);
    }
  }
  return options;
}

void collect_json_manifest_paths(const Json& value, const fs::path& base,
                                 std::vector<fs::path>& output) {
  if (value.is_string()) {
    fs::path path = value.get<std::string>();
    if (path.is_relative()) path = base / path;
    output.push_back(path);
  } else if (value.is_array()) {
    for (const auto& item : value) collect_json_manifest_paths(item, base, output);
  } else if (value.is_object()) {
    if (value.contains("jobs")) {
      for (const auto& job : value.at("jobs")) {
        if (job.contains("manifests"))
          collect_json_manifest_paths(job.at("manifests"), base, output);
        else if (job.contains("manifest"))
          collect_json_manifest_paths(job.at("manifest"), base, output);
      }
    } else if (value.contains("manifests")) {
      collect_json_manifest_paths(value.at("manifests"), base, output);
    } else if (value.contains("manifest")) {
      collect_json_manifest_paths(value.at("manifest"), base, output);
    } else if (value.value("schema", "") == kSegmentManifestSchema ||
               value.value("schema", "") == kNew15ManifestSchema) {
      fail("--manifest-list points to a column manifest; use --manifest");
    } else {
      fail("JSON manifest list has no jobs/manifests field");
    }
  } else {
    fail("unsupported JSON manifest list structure");
  }
}

bool is_sha256_hex(std::string_view value) {
  return value.size() == 64 &&
      std::all_of(value.begin(), value.end(), [](const char byte) {
        return (byte >= '0' && byte <= '9') ||
               (byte >= 'a' && byte <= 'f');
      });
}

void collect_feature_row(const Json& row, const fs::path& base,
                         ManifestSelection& selection) {
  require(row.is_object() && row.value("schema", "") == kFeatureRowSchema,
          "FEATURES JSONL row has an unexpected schema");
  require(row.contains("dataset_id") && row.at("dataset_id").is_string() &&
              !row.at("dataset_id").get<std::string>().empty(),
          "FEATURES JSONL row has no dataset_id");
  require(row.contains("dtype") && row.at("dtype").is_string(),
          "FEATURES JSONL row has no dtype");
  const auto dtype = row.at("dtype").get<std::string>();
  require(dtype == "int64" || dtype == "float32" || dtype == "float64",
          "FEATURES JSONL row has a non-canonical dtype");
  const auto valid_count = natural(row.at("valid_count"), "valid_count");
  require(valid_count > 0, "FEATURES JSONL row has no valid values");
  const auto width = natural(row.at("width"), "width");
  require(width == dtype_width(dtype),
          "FEATURES JSONL row width differs from dtype");
  require(row.contains("manifest_path") &&
              row.at("manifest_path").is_string() &&
              !row.at("manifest_path").get<std::string>().empty(),
          "FEATURES JSONL row has no manifest_path");
  require(row.contains("manifest_sha256") &&
              row.at("manifest_sha256").is_string() &&
              is_sha256_hex(row.at("manifest_sha256").get<std::string>()),
          "FEATURES JSONL row has no valid manifest_sha256");
  fs::path path = row.at("manifest_path").get<std::string>();
  if (path.is_relative()) path = base / path;
  path = fs::absolute(path).lexically_normal();
  FeatureExpectation expectation{
      row.at("dataset_id").get<std::string>(), dtype, valid_count,
      static_cast<unsigned>(width),
      row.at("manifest_sha256").get<std::string>()};
  require(selection.feature_expectations.emplace(path.string(), expectation)
              .second,
          "FEATURES JSONL repeats manifest_path: " + path.string());
  selection.paths.push_back(path);
}

ManifestSelection discover_manifests(const Options& options) {
  ManifestSelection selection;
  auto& paths = selection.paths;
  paths = options.manifests;
  for (const auto& root : options.manifest_roots) {
    require(fs::is_directory(root), "manifest root is not a directory: " +
                                      root.string());
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json") {
        try {
          const auto candidate = read_json(entry.path());
          if (candidate.value("schema", "") == kSegmentManifestSchema ||
              candidate.value("schema", "") == kNew15ManifestSchema)
            paths.push_back(entry.path());
        } catch (const std::exception&) {
          // A root may also hold receipts. Only valid segment manifests enter.
        }
      }
    }
  }
  if (options.manifest_list) {
    const auto bytes = read_file(*options.manifest_list);
    const std::string text(bytes.begin(), bytes.end());
    const auto paths_before_list = paths.size();
    const auto list_path =
        fs::absolute(*options.manifest_list).lexically_normal();
    std::string list_format = "json_plan";
    std::size_t list_record_count = 0;
    try {
      collect_json_manifest_paths(Json::parse(text),
                                  options.manifest_list->parent_path(), paths);
    } catch (const Json::parse_error&) {
      enum class LineFormat { unknown, plain_paths, feature_jsonl };
      LineFormat format = LineFormat::unknown;
      std::size_t position = 0;
      while (position < text.size()) {
        const auto newline = text.find('\n', position);
        auto line = std::string_view(text).substr(
            position, (newline == text.npos ? text.size() : newline) - position);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
          line.remove_suffix(1);
        while (!line.empty() && line.front() == ' ') line.remove_prefix(1);
        if (!line.empty() && line.front() != '#') {
          if (line.front() == '{') {
            require(format != LineFormat::plain_paths,
                    "--manifest-list mixes paths and FEATURES JSON rows");
            format = LineFormat::feature_jsonl;
            Json row;
            try {
              row = Json::parse(line);
            } catch (const Json::parse_error& error) {
              fail("invalid FEATURES JSONL row: " +
                   std::string(error.what()));
            }
            collect_feature_row(row, options.manifest_list->parent_path(),
                                selection);
          } else {
            require(format != LineFormat::feature_jsonl,
                    "--manifest-list mixes FEATURES JSON rows and paths");
            format = LineFormat::plain_paths;
            fs::path path{std::string(line)};
            if (path.is_relative())
              path = options.manifest_list->parent_path() / path;
            paths.push_back(path);
          }
          ++list_record_count;
        }
        if (newline == text.npos) break;
        position = newline + 1;
      }
      require(format != LineFormat::unknown,
              "--manifest-list contains no records");
      list_format = format == LineFormat::feature_jsonl
          ? "features_jsonl" : "plain_path_list";
    }
    if (list_format == "json_plan")
      list_record_count = paths.size() - paths_before_list;
    if (list_format == "features_jsonl") {
      require(paths_before_list == 0,
              "FEATURES JSONL must be the sole manifest selection source");
      require(options.expected_columns.has_value(),
              "FEATURES JSONL requires --expected-columns");
      require(list_record_count == *options.expected_columns,
              "FEATURES JSONL record count differs from --expected-columns");
    }
    selection.list_source_binding = {
        {"path", list_path.string()}, {"sha256", sha256(bytes)},
        {"format", list_format}, {"record_count", list_record_count},
        {"row_schema", list_format == "features_jsonl"
             ? Json(std::string(kFeatureRowSchema)) : Json(nullptr)}};
  }
  for (auto& path : paths) path = fs::absolute(path).lexically_normal();
  std::sort(paths.begin(), paths.end());
  const auto unique_end = std::unique(paths.begin(), paths.end());
  if (!selection.feature_expectations.empty())
    require(unique_end == paths.end(),
            "FEATURES selection overlaps another manifest source");
  paths.erase(unique_end, paths.end());
  require(!paths.empty(), "no segment manifests were selected");
  for (const auto& path : paths)
    require(fs::is_regular_file(path), "manifest is not a file: " + path.string());
  if (options.canary && paths.size() > *options.canary)
    paths.resize(*options.canary);
  return selection;
}

std::vector<std::string> canonical_methods(const Options& options,
                                           const uc::Registry& registry) {
  std::vector<std::string> methods;
  const std::set<std::string> allowed(kAllowedMethods.begin(), kAllowedMethods.end());
  for (const auto& name : options.methods) {
    const auto& method = registry.resolve(name);
    require(allowed.contains(method.name),
            "candidate is outside the closed BOS/LZ4/Zstd panel: " + method.name);
    require(std::find(methods.begin(), methods.end(), method.name) == methods.end(),
            "duplicate candidate: " + method.name);
    methods.push_back(method.name);
  }
  require(!methods.empty(), "candidate list is empty");
  return methods;
}

Json verify_runtime_contract(const uc::Registry& registry) {
  std::size_t bos_routes = 0;
  for (const auto& method : registry.methods())
    if (method.name == "algorithm:BOS") ++bos_routes;
  require(bos_routes == 1, "Registry must expose exactly one algorithm:BOS route");
  const auto& bos = registry.resolve("algorithm:BOS");
  require(bos.enabled && bos.available && bos.kind == "algorithm",
          "algorithm:BOS is not an enabled Registry algorithm");
  require_exact_bos_contract(bos.contract, "algorithm:BOS Registry entry");
  require(std::set<std::string>(bos.dtypes.begin(), bos.dtypes.end()) ==
              std::set<std::string>({"int64", "float32", "float64"}),
          "new BOS does not expose the exact frozen dtype set");
  for (const auto* name : {"algorithm:LZ4", "algorithm:Zstd"}) {
    const auto& method = registry.resolve(name);
    require(method.enabled && method.available && method.kind == "algorithm",
            std::string(name) + " is not an enabled Registry algorithm");
  }
  Json excluded = Json::object();
  for (const auto* name : {"algorithm:NNCP", "algorithm:PPMD8",
                           "algorithm:DMC", "algorithm:PAQ8",
                           "algorithm:CMIX", "algorithm:LSTM"}) {
    try {
      const auto& method = registry.resolve(name);
      require(!method.enabled,
              std::string(name) + " is unexpectedly enabled in Registry");
      excluded[name] = "catalog_disabled";
    } catch (const uc::Error& error) {
      require(error.code() == "method_disabled" ||
                  error.code() == "unknown_method",
              std::string(name) + " exclusion check failed: " + error.what());
      excluded[name] = error.code();
    }
  }
  return {{"bos_public_method", bos.name},
          {"bos_implementation", bos.contract.at("implementation")},
          {"bos_profile_version", bos.contract.at("profile_version")},
          {"bos_float_route_id", bos.contract.at("float_route_id")},
          {"bos_integer_route_id", bos.contract.at("integer_route_id")},
          {"bos_integer_configuration",
           bos.contract.at("integer_configuration")},
          {"bos_float_configuration", bos.contract.at("float_configuration")},
          {"bos_legacy_integer_encode_selectable", false},
          {"bos_legacy_integer_decode_compatibility",
           bos.contract.value("legacy_integer_decode", "")},
          {"closed_candidate_allowlist", kAllowedMethods},
          {"excluded_algorithm_status", excluded}};
}

Json enforce_address_space_limit(std::uint64_t requested) {
#ifdef __linux__
  require(requested <= static_cast<std::uint64_t>(
                           std::numeric_limits<rlim_t>::max()),
          "address-space limit exceeds rlim_t");
  struct rlimit limit {};
  require(::getrlimit(RLIMIT_AS, &limit) == 0,
          "getrlimit(RLIMIT_AS) failed: " + std::string(std::strerror(errno)));
  const rlim_t requested_limit = static_cast<rlim_t>(requested);
  const rlim_t target = limit.rlim_cur == RLIM_INFINITY
      ? requested_limit
      : std::min(limit.rlim_cur, requested_limit);
  limit.rlim_cur = target;
  require(::setrlimit(RLIMIT_AS, &limit) == 0,
          "setrlimit(RLIMIT_AS) failed: " + std::string(std::strerror(errno)));
  return {{"requested_bytes", requested},
          {"effective_bytes", static_cast<std::uint64_t>(target)},
          {"enforced", true}, {"mechanism", "RLIMIT_AS"}};
#else
  return {{"requested_bytes", requested}, {"effective_bytes", nullptr},
          {"enforced", false},
          {"mechanism", "unsupported_non_linux_local_validation"}};
#endif
}

Json enforce_single_thread_environment() {
  Json values = Json::object();
  for (const auto* name : {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}) {
    require(::setenv(name, "1", 1) == 0,
            std::string("cannot set ") + name + "=1");
    values[name] = "1";
  }
  return values;
}

std::vector<Json> independent_results(const fs::path& root) {
  std::vector<Json> records;
  if (!fs::is_directory(root)) return records;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json")
      records.push_back(read_json(entry.path()));
  }
  std::sort(records.begin(), records.end(), [](const Json& left, const Json& right) {
    return left.at("result_key").get<std::string>() <
           right.at("result_key").get<std::string>();
  });
  return records;
}

using ExpectedResults = std::map<std::string, std::pair<std::string, std::string>>;

std::vector<Json> validated_independent_results(
    const fs::path& root, const ExpectedResults& expected,
    std::string_view run_config_sha256) {
  auto records = independent_results(root);
  std::set<std::string> seen;
  for (const auto& record : records) {
    require(record.value("schema", "") == kResultSchema,
            "unexpected independent result schema");
    const auto key = record.at("result_key").get<std::string>();
    require(seen.insert(key).second, "duplicate independent result key");
    const auto binding = expected.find(key);
    require(binding != expected.end(), "independent result is outside RUN_CONFIG");
    require(record.value("manifest_sha256", "") == binding->second.first &&
                record.value("method", "") == binding->second.second,
            "independent result binding differs from RUN_CONFIG");
    require(record.value("run_config_sha256", "") == run_config_sha256,
            "independent result has another RUN_CONFIG binding");
    if (record.value("method", "") == "algorithm:BOS") {
      require(record.contains("method_contract"),
              "BOS result has no runtime method contract");
      require_exact_bos_contract(record.at("method_contract"),
                                 "independent BOS result");
    }
  }
  return records;
}

void rebuild_ledger(const fs::path& output_dir,
                    const std::vector<Json>& records) {
  std::string contents;
  for (const auto& record : records) contents += record.dump() + "\n";
  atomic_write(output_dir / "results.jsonl", contents);
}

Json build_summary(const std::vector<Json>& records, const Json& run_config) {
  Json summary = {{"schema", kSummarySchema},
                  {"created_at_utc", now_utc()},
                  {"status", "completed"},
                  {"run_config_sha256", sha256(run_config.dump())},
                  {"candidate_scope", run_config.at("methods")},
                  {"input_columns", run_config.at("manifest_count")},
                  {"timing_contract", run_config.at("timing_contract")},
                  {"methods", Json::object()}};
  std::set<std::string> result_keys;
  for (const auto& record : records) {
    require(result_keys.insert(record.at("result_key")).second,
            "duplicate independent result key");
    const auto method = record.at("method").get<std::string>();
    auto& aggregate = summary["methods"][method];
    if (aggregate.is_null())
      aggregate = {{"outcomes", 0}, {"ok", 0}, {"failed", 0},
                   {"strict_columns", 0}, {"lossless_native_raw_columns", 0},
                   {"source_numeric_bytes", 0}, {"compressed_bytes", 0},
                   {"source_native_numeric_bytes", 0},
                   {"compression_time_ns_sum_of_column_medians", 0},
                   {"decompression_time_ns_sum_of_column_medians", 0}};
    aggregate["outcomes"] = aggregate.at("outcomes").get<std::uint64_t>() + 1;
    if (record.at("status") == "ok") {
      aggregate["ok"] = aggregate.at("ok").get<std::uint64_t>() + 1;
      if (record.value("strict_path", false))
        aggregate["strict_columns"] =
            aggregate.at("strict_columns").get<std::uint64_t>() + 1;
      if (record.value("native_raw_lossless_path", false))
        aggregate["lossless_native_raw_columns"] =
            aggregate.at("lossless_native_raw_columns").get<std::uint64_t>() + 1;
      for (const auto* field : {"original_numeric_bytes",
                               "source_native_numeric_bytes", "compressed_bytes",
                               "compression_time_ns", "decompression_time_ns"}) {
        const std::string target = std::string(field) == "original_numeric_bytes"
            ? "source_numeric_bytes"
            : std::string(field) == "compression_time_ns"
                ? "compression_time_ns_sum_of_column_medians"
                : std::string(field) == "decompression_time_ns"
                    ? "decompression_time_ns_sum_of_column_medians"
                    : field;
        aggregate[target] = aggregate.at(target).get<std::uint64_t>() +
                            record.at(field).get<std::uint64_t>();
      }
    } else {
      aggregate["failed"] = aggregate.at("failed").get<std::uint64_t>() + 1;
    }
  }
  for (auto& [method, aggregate] : summary["methods"].items()) {
    (void)method;
    const auto source = aggregate.at("source_numeric_bytes").get<std::uint64_t>();
    const auto source_native =
        aggregate.at("source_native_numeric_bytes").get<std::uint64_t>();
    const auto compressed = aggregate.at("compressed_bytes").get<std::uint64_t>();
    if (source && compressed) {
      aggregate["compressed_to_original_ratio_of_sums"] =
          static_cast<double>(compressed) / static_cast<double>(source);
      aggregate["compression_factor_of_sums"] =
          static_cast<double>(source) / static_cast<double>(compressed);
    }
    if (source_native && compressed) {
      aggregate["compressed_to_source_native_reference_ratio_of_sums"] =
          static_cast<double>(compressed) /
          static_cast<double>(source_native);
      aggregate["source_native_reference_ratio_contract"] =
          "accounting reference only; execution used canonical typed-plane bytes and performed no cast";
    }
  }
  summary["result_outcomes"] = result_keys.size();
  summary["expected_outcomes"] =
      run_config.at("manifest_count").get<std::uint64_t>() *
      run_config.at("methods").size();
  summary["complete"] = summary.at("result_outcomes") == summary.at("expected_outcomes");
  std::uint64_t failed = 0;
  for (const auto& [method, aggregate] : summary.at("methods").items()) {
    (void)method;
    failed += aggregate.at("failed").get<std::uint64_t>();
  }
  summary["failed_outcomes"] = failed;
  summary["all_ok"] = failed == 0;
  summary["status"] = !summary.at("complete").get<bool>() ? "partial" :
      failed == 0 ? "completed_all_lossless" : "completed_with_failures";
  return summary;
}

int run_campaign(const Options& options) {
  require(!options.output_dir.empty(), "--output-dir is required");
  const auto thread_environment = enforce_single_thread_environment();
  const auto address_space_limit =
      enforce_address_space_limit(options.address_space_limit_bytes);
  uc::Registry registry;
  const auto runtime_contract = verify_runtime_contract(registry);
  const auto methods = canonical_methods(options, registry);
  const auto manifest_selection = discover_manifests(options);
  const auto& manifests = manifest_selection.paths;
  if (options.expected_columns && !options.canary)
    require(manifests.size() == *options.expected_columns,
            "manifest count differs from --expected-columns");

  Json manifest_bindings = Json::array();
  Digest manifest_set_digest;
  for (const auto& path : manifests) {
    const auto bytes = read_file(path);
    const auto digest = sha256(bytes);
    Json binding = {{"path", path.string()}, {"sha256", digest}};
    if (const auto feature =
            manifest_selection.feature_expectations.find(path.string());
        feature != manifest_selection.feature_expectations.end()) {
      require(digest == feature->second.manifest_sha256,
              "manifest SHA-256 differs from FEATURES binding: " +
                  path.string());
      binding["features_row_binding"] = {
          {"schema", kFeatureRowSchema},
          {"dataset_id", feature->second.dataset_id},
          {"dtype", feature->second.dtype},
          {"valid_count", feature->second.valid_count},
          {"width", feature->second.width},
          {"manifest_sha256", feature->second.manifest_sha256}};
    }
    const std::string binding_line = path.string() + "\t" + digest + "\n";
    manifest_set_digest.update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(binding_line.data()),
        binding_line.size()));
    manifest_bindings.push_back(std::move(binding));
  }
  const Json run_config = {
      {"schema", kRunSchema}, {"methods", methods},
      {"manifest_count", manifests.size()},
      {"manifest_set_sha256", manifest_set_digest.finish()},
      {"manifests", manifest_bindings},
      {"manifest_list_source",
       manifest_selection.list_source_binding
           ? *manifest_selection.list_source_binding : Json(nullptr)},
      {"warmups", kWarmups}, {"repeats", kRepeats}, {"threads", 1},
      {"native_thread_environment", thread_environment},
      {"address_space_limit", address_space_limit},
      {"canary", options.canary ? Json(*options.canary) : Json(nullptr)},
      {"timing_contract", "W1/R3/T1; one public complete UCC1 frame per manifest block; source I/O and independent verification excluded"},
      {"candidate_contract", "closed explicit public Registry routes; no selector/model; no implicit cast"},
      {"runtime_contract", runtime_contract},
      {"accepted_input_manifest_schemas",
       {kSegmentManifestSchema, kNew15ManifestSchema}},
      {"excluded_families", {"NNCP", "PPMd/PPMD8", "DMC", "PAQ8", "CMIX", "LSTM"}}};
  const auto run_config_sha256 = sha256(run_config.dump());
  ExpectedResults expected_results;
  for (const auto& binding : manifest_bindings) {
    const auto manifest_sha = binding.at("sha256").get<std::string>();
    for (const auto& method : methods) {
      const auto key = sha256(manifest_sha + "\n" + method);
      require(expected_results.emplace(key,
                  std::make_pair(manifest_sha, method)).second,
              "duplicate expected manifest/method result identity");
    }
  }

  std::vector<Json> existing_results;
  if (fs::exists(options.output_dir)) {
    require(options.resume, "output directory exists; pass --resume to continue");
    require(fs::is_regular_file(options.output_dir / "RUN_CONFIG.json"),
            "resume output has no RUN_CONFIG.json");
    require(read_json(options.output_dir / "RUN_CONFIG.json") == run_config,
            "resume run configuration differs");
    existing_results = validated_independent_results(
        options.output_dir / "columns", expected_results,
        run_config_sha256);
    rebuild_ledger(options.output_dir, existing_results);
  } else {
    fs::create_directories(options.output_dir / "columns");
    atomic_write_json(options.output_dir / "RUN_CONFIG.json", run_config);
    atomic_write(options.output_dir / "results.jsonl", "");
  }

  std::set<std::string> completed;
  for (const auto& record : existing_results)
    completed.insert(record.at("result_key").get<std::string>());

  for (std::size_t manifest_index = 0; manifest_index < manifests.size();
       ++manifest_index) {
    const auto column = load_column(manifests[manifest_index]);
    if (const auto feature = manifest_selection.feature_expectations.find(
            manifests[manifest_index].string());
        feature != manifest_selection.feature_expectations.end()) {
      require(column.dataset_id == feature->second.dataset_id &&
                  column.dtype == feature->second.dtype &&
                  column.valid_values == feature->second.valid_count &&
                  dtype_width(column.dtype) == feature->second.width,
              "loaded column differs from FEATURES row binding: " +
                  manifests[manifest_index].string());
    }
    const auto column_dir = options.output_dir / "columns" /
        (safe_component(column.dataset_id + "__" + column.column_id) + "__" +
         column.manifest_sha256.substr(0, 16));
    for (const auto& method : methods) {
      const auto result_key = sha256(column.manifest_sha256 + "\n" + method);
      if (completed.contains(result_key)) continue;
      auto result = measure_method(column, method, registry);
      require(result.at("result_key") == result_key,
              "internal result identity mismatch");
      result["run_config_sha256"] = run_config_sha256;
      const auto result_path = column_dir / (method_leaf(method) + ".json");
      require(!fs::exists(result_path), "independent result path collision");
      atomic_write_json(result_path, result);
      append_json_line(options.output_dir / "results.jsonl", result);
      completed.insert(result_key);
    }
    std::cerr << "completed " << (manifest_index + 1) << "/" << manifests.size()
              << " " << column.dataset_id << "/" << column.column_id << "\n";
  }
  const auto committed_results = validated_independent_results(
      options.output_dir / "columns", expected_results, run_config_sha256);
  const auto summary = build_summary(committed_results, run_config);
  atomic_write_json(options.output_dir / "SUMMARY_INPUT.json", summary);
  std::cout << summary.dump(2) << "\n";
  return summary.at("complete").get<bool>() && summary.at("all_ok").get<bool>()
      ? 0 : 2;
}

template <class T>
void append_le(Bytes& output, T value) {
  auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
  if constexpr (std::endian::native == std::endian::big)
    std::reverse(bytes.begin(), bytes.end());
  output.insert(output.end(), bytes.begin(), bytes.end());
}

void make_fixture_manifest(const fs::path& root, std::string dtype,
                           const Bytes& values, std::uint64_t count,
                           std::string name = {}) {
  if (name.empty()) name = dtype;
  const auto values_path = root / (name + ".bin");
  {
    std::ofstream output(values_path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size()));
  }
  const unsigned width = dtype_width(dtype);
  const auto first_count = count / 2;
  Json blocks = Json::array();
  std::uint64_t offset = 0;
  for (std::uint64_t block_index = 0; block_index < 2; ++block_index) {
    const auto block_count = block_index == 0 ? first_count : count - first_count;
    const auto length = block_count * width;
    const auto block_bytes = std::span<const std::uint8_t>(values).subspan(
        static_cast<std::size_t>(offset), static_cast<std::size_t>(length));
    blocks.push_back({{"block_index", block_index}, {"dtype", dtype},
                      {"row_count", block_count}, {"valid_count", block_count},
                      {"missing_count", 0}, {"values_path", values_path.string()},
                      {"values_offset", offset}, {"values_length", length},
                      {"values_sha256", sha256(block_bytes)}});
    offset += length;
  }
  const Json manifest = {
      {"schema", kSegmentManifestSchema},
      {"input_profile", "bos-float150-clean-concat-reblock10000-v1"},
      {"dataset_id", "selftest"},
      {"column_identity", {{"dataset_id", "selftest"},
                           {"column_id", name}, {"stable_id", name},
                           {"source_column", name}, {"dtype", dtype}}},
      {"source_valid_stream_sha256", sha256(values)},
      {"source_totals", {{"valid_count", count}, {"row_count", count},
                          {"missing_count", 0}}},
      {"blocks", blocks}};
  atomic_write_json(root / (name + ".json"), manifest);
}

void make_new15_fixture_manifest(const fs::path& root, const Bytes& values,
                                 std::uint64_t count) {
  const auto values_path = root / "new15-int64.bin";
  atomic_write(values_path,
      std::string_view(reinterpret_cast<const char*>(values.data()), values.size()));
  Json blocks = Json::array();
  const auto first_count = count / 2;
  std::uint64_t offset = 0;
  for (std::uint64_t entity = 0; entity < 2; ++entity) {
    const auto block_count = entity == 0 ? first_count : count - first_count;
    const auto length = block_count * 8;
    const auto block_bytes = std::span<const std::uint8_t>(values).subspan(
        static_cast<std::size_t>(offset), static_cast<std::size_t>(length));
    blocks.push_back({{"block_id", "new15:e" + std::to_string(entity) + ":b0"},
                      {"block_index", entity}, {"entity_id", std::to_string(entity)},
                      {"entity_ordinal", entity}, {"dtype", "int64"},
                      {"source_dtype", "int16"},
                      {"adapter", "exact_int16_to_int64"},
                      {"row_count", block_count}, {"valid_count", block_count},
                      {"missing_count", 0}, {"values_path", values_path.string()},
                      {"values_offset", offset}, {"values_length", length},
                      {"canonical_input_bytes", length},
                      {"values_sha256", sha256(block_bytes)},
                      {"expected_sequence_sha256", sha256(block_bytes)},
                      {"source_native_numeric_bytes", block_count * 2}});
    offset += length;
  }
  const Json manifest = {
      {"schema", kNew15ManifestSchema}, {"status", "ready"},
      {"dataset_id", "selftest"}, {"column_id", "new15-int64"},
      {"source_stable_id", "new15-int64"},
      {"source_column", "new15-int64"}, {"source_dtype", "int16"},
      {"canonical_dtype", "int64"}, {"dtype", "int64"},
      {"adapter", "exact_int16_to_int64"},
      {"valid_count", count}, {"canonical_input_bytes", values.size()},
      {"source_native_numeric_bytes", count * 2},
      {"grouped_values_path", values_path.string()},
      {"grouped_values_sha256", sha256(values)},
      {"one_entity_per_block", true}, {"blocks", blocks}};
  atomic_write_json(root / "new15-int64.json", manifest);
}

fs::path make_fixture_features(const fs::path& root) {
  std::vector<fs::path> manifests;
  for (const auto& entry : fs::directory_iterator(root / "manifests"))
    if (entry.is_regular_file() && entry.path().extension() == ".json")
      manifests.push_back(fs::absolute(entry.path()).lexically_normal());
  std::sort(manifests.begin(), manifests.end());
  std::string lines;
  for (const auto& manifest : manifests) {
    const auto column = load_column(manifest);
    lines += Json({{"schema", kFeatureRowSchema},
                   {"dataset_id", column.dataset_id},
                   {"dtype", column.dtype},
                   {"valid_count", column.valid_values},
                   {"width", dtype_width(column.dtype)},
                   {"manifest_path", manifest.string()},
                   {"manifest_sha256", column.manifest_sha256}}).dump() + "\n";
  }
  const auto path = root / "FEATURES.jsonl";
  atomic_write(path, lines);
  return path;
}

int selftest() {
  const fs::path root = fs::temp_directory_path() /
      ("bos_harness150_panel_selftest_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "manifests");
  Bytes int64_values, new15_widened_values, float32_values, float64_values;
  const std::array<std::int64_t, 8> integer_fixture = {
      std::numeric_limits<std::int64_t>::min(), std::int64_t(-1), std::int64_t(0),
      std::int64_t(1), std::int64_t(7), std::int64_t(7), std::int64_t(1024),
      std::numeric_limits<std::int64_t>::max()};
  for (const std::int64_t value : integer_fixture)
    append_le(int64_values, value);
  for (const std::int64_t value : {-32768, -1, 0, 1, 7, 7, 1024, 32767})
    append_le(new15_widened_values, value);
  for (float value : {-0.0F, 0.0F, 1.25F, -7.5F, 1000.0F, 1000.0F})
    append_le(float32_values, value);
  for (double value : {-0.0, 0.0, 1.25, -7.5, 1000.0, 1000.0})
    append_le(float64_values, value);
  Bytes raw_float64_values;
  std::mt19937_64 random(UINT64_C(0x424f530000000001));
  for (std::size_t index = 0; index < 257; ++index)
    append_le(raw_float64_values, random());
  make_fixture_manifest(root / "manifests", "int64", int64_values, 8);
  make_fixture_manifest(root / "manifests", "float32", float32_values, 6);
  make_fixture_manifest(root / "manifests", "float64", float64_values, 6);
  make_fixture_manifest(root / "manifests", "float64", raw_float64_values,
                        257, "float64-raw");
  make_new15_fixture_manifest(root / "manifests", new15_widened_values, 8);
  Options options;
  options.manifest_list = make_fixture_features(root);
  options.output_dir = root / "output";
  options.expected_columns = 5;
  try {
    const int status = run_campaign(options);
    const auto summary = read_json(options.output_dir / "SUMMARY_INPUT.json");
    require(status == 0 && summary.at("complete") == true &&
                summary.at("result_outcomes") == 15,
            "self-test campaign did not produce fifteen complete outcomes");
    const auto run_config = read_json(options.output_dir / "RUN_CONFIG.json");
    require(run_config.at("manifest_list_source").at("format") ==
                    "features_jsonl" &&
                run_config.at("manifest_list_source").at("row_schema") ==
                    kFeatureRowSchema &&
                run_config.at("manifest_list_source").at("record_count") == 5 &&
                is_sha256_hex(run_config.at("manifest_list_source")
                                  .at("sha256").get<std::string>()),
            "self-test FEATURES JSONL source binding is incomplete");
    const auto& runtime_contract = run_config.at("runtime_contract");
    require(runtime_contract.value("bos_profile_version", "") ==
                    kBosProfileVersion &&
                runtime_contract.value("bos_float_route_id", "") ==
                    kBosFloatRoute &&
                runtime_contract.value("bos_integer_route_id", "") ==
                    kBosIntegerRoute,
            "self-test RUN_CONFIG lacks exact new-BOS runtime routes");
    ExpectedResults expected;
    for (const auto& binding : run_config.at("manifests"))
      for (const auto& method : run_config.at("methods")) {
        const auto manifest_sha = binding.at("sha256").get<std::string>();
        const auto method_name = method.get<std::string>();
        expected.emplace(sha256(manifest_sha + "\n" + method_name),
                         std::make_pair(manifest_sha, method_name));
      }
    const auto results = validated_independent_results(
        options.output_dir / "columns", expected, sha256(run_config.dump()));
    bool saw_widened_new15 = false;
    bool saw_exact_bos_contract = false;
    for (const auto& result : results) {
      require(result.at("status") == "ok" &&
                  result.at("lossless_bit_exact") == true &&
                  result.at("independent_decode_verified") == true &&
                  result.contains("execution_path") &&
                  result.contains("eligible_for_strict_selector") &&
                  result.at("eligible_for_lossless_selector") == true,
              "self-test outcome failed lossless verification");
      if (result.at("method") == "algorithm:BOS") {
        require_exact_bos_contract(result.at("method_contract"),
                                   "self-test BOS result");
        saw_exact_bos_contract = true;
      }
      if (result.at("stable_id") == "new15-int64") {
        saw_widened_new15 = true;
        require(result.at("canonical_input_bytes") == 64 &&
                    result.at("source_native_numeric_bytes") == 16 &&
                    result.at("codec_input_dtype") == "int64" &&
                    result.at("source_physical_dtype") == "int16" &&
                    result.at("input_adapter_provenance") ==
                        "exact_int16_to_int64" &&
                    result.at("canonical_typed_plane_executed") == true &&
                    result.at("implicit_cast_performed_by_panel") == false &&
                    result.at("compressed_to_source_native_reference_ratio")
                            .get<double>() >
                        result.at("compressed_to_original_ratio").get<double>(),
                "widened new15 accounting/execution boundary changed");
      }
    }
    require(saw_widened_new15,
            "self-test did not exercise widened new15 canonical input");
    require(saw_exact_bos_contract,
            "self-test did not exercise exact BOS runtime identity");
    {
      const auto bos_result = std::find_if(
          results.begin(), results.end(), [](const Json& result) {
            return result.at("method") == "algorithm:BOS";
          });
      require(bos_result != results.end(),
              "self-test has no BOS result for resume rejection");
      auto tampered = *bos_result;
      tampered["method_contract"]["float_route_id"] =
          "bos-bfe-ngf-hybrid-wrong";
      const auto tampered_root = root / "tampered-route-result";
      fs::create_directories(tampered_root);
      atomic_write_json(tampered_root / "result.json", tampered);
      bool rejected = false;
      try {
        (void)validated_independent_results(
            tampered_root, expected, sha256(run_config.dump()));
      } catch (const std::exception&) {
        rejected = true;
      }
      require(rejected,
              "resume accepted a BOS result with another float route");
      fs::remove_all(tampered_root);
    }
    require(summary.at("methods").at("algorithm:BOS").at(
                "lossless_native_raw_columns").get<std::uint64_t>() != 0,
            "self-test did not exercise BOS native-raw classification");
    {
      uc::Registry registry;
      Options forbidden;
      forbidden.methods = {"algorithm:PPMD8"};
      bool rejected = false;
      try {
        (void)canonical_methods(forbidden, registry);
      } catch (const std::exception&) {
        rejected = true;
      }
      require(rejected, "excluded PPMD8 entered the closed public panel");
    }
    options.resume = true;
    require(run_campaign(options) == 0, "self-test resume failed");
    std::ifstream ledger(options.output_dir / "results.jsonl");
    std::size_t ledger_lines = 0;
    std::string line;
    while (std::getline(ledger, line)) if (!line.empty()) ++ledger_lines;
    require(ledger_lines == 15, "resume duplicated or lost JSONL outcomes");
    fs::remove_all(root);
    std::cout << "bos_harness150_panel self-test passed: 5 manifests x 3 methods; strict/native-raw and resume stable\n";
    return 0;
  } catch (...) {
    std::cerr << "self-test artifacts retained at " << root << "\n";
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    if (options.selftest) {
      require(argc == 2, "--selftest cannot be combined with campaign options");
      return selftest();
    }
    require(!options.manifests.empty() || !options.manifest_roots.empty() ||
                options.manifest_list,
            "provide --manifest, --manifest-root, or --manifest-list");
    return run_campaign(options);
  } catch (const std::exception& error) {
    std::cerr << "bos_harness150_panel: " << error.what() << "\n";
    return 1;
  }
}
