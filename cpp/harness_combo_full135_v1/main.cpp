#include "../harness_combo_pilot_v1/runtime.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vector>

// Full-column, streaming driver for the isolated 30-recipe pilot runtime.
// This file deliberately does not extend the pilot catalog or claim coverage
// of the frozen 85/97/182 recommendation oracles.
namespace p = harness_combo_pilot_v1;
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Bytes = native_graph_v1::Bytes;
using Path = fs::path;

namespace {

constexpr std::string_view kContainerMagic = "HCF1";
constexpr std::string_view kBlockMagic = "BLK1";
constexpr std::string_view kFooterMagic = "END1";
constexpr std::uint64_t kMaxJsonBytes = 256ULL << 20;
constexpr std::uint64_t kMaxSourceRegionBytes = 256ULL << 20;
constexpr std::uint64_t kMaxFrameBytes = 1ULL << 30;
constexpr std::uint64_t kMaxBlockCount = 10'000'000ULL;

Path executable_path;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

struct TimePoint {
  std::uint64_t cpu_ns;
  std::uint64_t wall_ns;
};

TimePoint now() {
  timespec cpu{};
  require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu) == 0,
          "clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
  return {std::uint64_t(cpu.tv_sec) * 1'000'000'000ULL +
              std::uint64_t(cpu.tv_nsec),
          std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count())};
}

TimePoint elapsed(TimePoint start) {
  const auto finish = now();
  return {finish.cpu_ns - start.cpu_ns, finish.wall_ns - start.wall_ns};
}

Json timing_json(TimePoint value) {
  return {{"cpu_ns", value.cpu_ns}, {"wall_ns", value.wall_ns}};
}

std::uint64_t child_cpu_ns() {
  rusage usage{};
  require(getrusage(RUSAGE_CHILDREN, &usage) == 0, "getrusage failed");
  return (std::uint64_t(usage.ru_utime.tv_sec) +
          std::uint64_t(usage.ru_stime.tv_sec)) *
             1'000'000'000ULL +
         (std::uint64_t(usage.ru_utime.tv_usec) +
          std::uint64_t(usage.ru_stime.tv_usec)) *
             1'000ULL;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
  require(!values.empty(), "cannot take median of zero samples");
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

class Sha256 {
 public:
  Sha256() : context_(EVP_MD_CTX_new()) {
    require(context_ != nullptr &&
                EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) == 1,
            "SHA-256 initialization failed");
  }
  Sha256(const Sha256&) = delete;
  Sha256& operator=(const Sha256&) = delete;
  Sha256(Sha256&& other) noexcept : context_(other.context_) {
    other.context_ = nullptr;
  }
  Sha256& operator=(Sha256&& other) noexcept {
    if (this != &other) {
      EVP_MD_CTX_free(context_);
      context_ = other.context_;
      other.context_ = nullptr;
    }
    return *this;
  }
  ~Sha256() { EVP_MD_CTX_free(context_); }

  void update(std::span<const std::uint8_t> bytes) {
    if (!bytes.empty())
      require(EVP_DigestUpdate(context_, bytes.data(), bytes.size()) == 1,
              "SHA-256 update failed");
  }
  void update(std::string_view text) {
    update(std::span(reinterpret_cast<const std::uint8_t*>(text.data()),
                     text.size()));
  }
  std::string finish() {
    std::array<unsigned char, 32> result{};
    unsigned length = 0;
    require(EVP_DigestFinal_ex(context_, result.data(), &length) == 1 &&
                length == result.size(),
            "SHA-256 finalization failed");
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : result) output << std::setw(2) << unsigned(byte);
    return output.str();
  }

 private:
  EVP_MD_CTX* context_;
};

std::string sha256(std::span<const std::uint8_t> bytes) {
  Sha256 hash;
  hash.update(bytes);
  return hash.finish();
}

std::string sha256(std::string_view text) {
  return sha256(std::span(reinterpret_cast<const std::uint8_t*>(text.data()),
                          text.size()));
}

std::string file_sha256(const Path& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot open file for SHA-256: " + path.string());
  Sha256 hash;
  std::array<std::uint8_t, 1 << 20> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const auto count = input.gcount();
    if (count > 0) hash.update(std::span(buffer.data(), std::size_t(count)));
  }
  require(input.eof(), "failed while hashing file: " + path.string());
  return hash.finish();
}

Json read_json(const Path& path) {
  require(fs::exists(path), "JSON file does not exist: " + path.string());
  require(fs::file_size(path) <= kMaxJsonBytes,
          "JSON exceeds safety limit: " + path.string());
  std::ifstream input(path);
  require(bool(input), "cannot open JSON: " + path.string());
  Json result;
  input >> result;
  require(bool(input), "cannot parse JSON: " + path.string());
  return result;
}

void fsync_file(const Path& path) {
  const auto fd = ::open(path.c_str(), O_RDONLY);
  require(fd >= 0, "cannot open file for fsync: " + path.string());
  const auto rc = ::fsync(fd);
  const auto saved = errno;
  ::close(fd);
  if (rc != 0) {
    errno = saved;
    fail("fsync failed: " + path.string());
  }
}

void atomic_text(const Path& path, std::string_view content) {
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  auto temporary = path;
  temporary += ".tmp." + std::to_string(getpid());
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(bool(output), "cannot create temporary output: " +
                              temporary.string());
    output.write(content.data(), std::streamsize(content.size()));
    output.flush();
    require(bool(output), "failed to write temporary output: " +
                              temporary.string());
  }
  fsync_file(temporary);
  fs::rename(temporary, path);
}

void atomic_json(const Path& path, const Json& value) {
  atomic_text(path, value.dump(2) + "\n");
}

void atomic_jsonl(const Path& path, const Json& rows) {
  std::ostringstream text;
  for (const auto& row : rows) text << row.dump() << '\n';
  atomic_text(path, text.str());
}

unsigned dtype_width(const std::string& dtype) {
  if (dtype == "int32" || dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "float64") return 8;
  fail("unsupported source dtype: " + dtype);
}

std::uint64_t validate_block(const p::Block& block) {
  const auto expected_mask = (block.row_count + 7) / 8;
  require(block.missing.size() == expected_mask,
          "missing bitmap length does not match row count");
  std::uint64_t missing = 0;
  for (const auto byte : block.missing) missing += std::popcount(byte);
  if (block.row_count % 8 != 0 && !block.missing.empty())
    require((block.missing.back() >> (block.row_count % 8)) == 0,
            "missing bitmap has nonzero padding bits");
  require(missing <= block.row_count, "invalid missing count");
  const auto valid = block.row_count - missing;
  require(valid <= std::numeric_limits<std::uint64_t>::max() /
                       dtype_width(block.dtype),
          "valid byte count overflow");
  require(valid * dtype_width(block.dtype) == block.values.size(),
          "compact value bytes do not match bitmap");
  return valid;
}

Bytes read_region(const Path& manifest_path, const Json& block,
                  const char* path_key, const char* offset_key,
                  const char* length_key, const char* sha_key) {
  Path path = block.at(path_key).get<std::string>();
  if (path.is_relative()) path = manifest_path.parent_path() / path;
  const auto offset = block.at(offset_key).get<std::uint64_t>();
  const auto length = block.at(length_key).get<std::uint64_t>();
  require(length <= kMaxSourceRegionBytes,
          "one source region exceeds the 256 MiB runtime stream bound");
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(bool(input), "cannot open source payload: " + path.string());
  const auto size = input.tellg();
  require(size >= 0 && offset <= std::uint64_t(size) &&
              length <= std::uint64_t(size) - offset,
          "source payload region is outside file: " + path.string());
  input.seekg(std::streamoff(offset));
    Bytes result(std::size_t(length), std::uint8_t{0});
  if (length)
    input.read(reinterpret_cast<char*>(result.data()),
               std::streamsize(length));
  require(bool(input), "source payload region is truncated: " + path.string());
  require(sha256(result) == block.at(sha_key).get<std::string>(),
          "source payload region SHA-256 mismatch: " + path.string());
  return result;
}

p::Block load_block(const Path& manifest_path, const Json& metadata,
                    const std::string& dtype) {
  require(metadata.at("dtype") == dtype, "mixed dtype in column manifest");
  p::Block block;
  block.dtype = dtype;
  block.row_count = metadata.at("row_count").get<std::uint64_t>();
  block.values = read_region(manifest_path, metadata, "values_path",
                             "values_offset", "values_length",
                             "values_sha256");
  block.missing = read_region(manifest_path, metadata,
                              "missing_bitmap_path",
                              "missing_bitmap_offset",
                              "missing_bitmap_length",
                              "missing_bitmap_sha256");
  const auto valid = validate_block(block);
  require(valid == metadata.at("valid_count").get<std::uint64_t>(),
          "manifest valid_count mismatch");
  require(block.row_count - valid ==
              metadata.at("missing_count").get<std::uint64_t>(),
          "manifest missing_count mismatch");
  return block;
}

void put_u64(std::ostream& output, std::uint64_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (unsigned i = 0; i < 8; ++i) bytes[i] = std::uint8_t(value >> (8 * i));
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  require(bool(output), "failed to write container integer");
}

std::uint64_t get_u64(std::istream& input) {
  std::array<std::uint8_t, 8> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  require(bool(input), "truncated container integer");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
  return value;
}

void put_json(std::ostream& output, const Json& value) {
  const auto text = value.dump();
  put_u64(output, text.size());
  output.write(text.data(), std::streamsize(text.size()));
  require(bool(output), "failed to write container JSON");
}

Json get_json(std::istream& input) {
  const auto length = get_u64(input);
  require(length <= kMaxJsonBytes, "container JSON exceeds safety limit");
  std::string text(std::size_t(length), '\0');
  if (length) input.read(text.data(), std::streamsize(length));
  require(bool(input), "truncated container JSON");
  return Json::parse(text);
}

void put_magic(std::ostream& output, std::string_view magic) {
  output.write(magic.data(), std::streamsize(magic.size()));
  require(bool(output), "failed to write container magic");
}

bool consume_magic(std::istream& input, std::string_view expected) {
  std::string actual(expected.size(), '\0');
  input.read(actual.data(), std::streamsize(actual.size()));
  if (!input) return false;
  return actual == expected;
}

void digest_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (unsigned i = 0; i < 8; ++i) bytes[i] = std::uint8_t(value >> (8 * i));
  digest.update(bytes);
}

void digest_text(Sha256& digest, std::string_view value) {
  digest_u64(digest, value.size());
  digest.update(value);
}

void ordered_block_digest(Sha256& digest, std::uint64_t manifest_array_index,
                          const std::string& block_id,
                          const std::string& entity_id,
                          std::uint64_t block_index,
                          std::uint64_t row_count,
                          std::uint64_t valid_count,
                          const std::string& values_sha,
                          const std::string& missing_sha) {
  digest_u64(digest, manifest_array_index);
  digest_text(digest, block_id);
  digest_text(digest, entity_id);
  digest_u64(digest, block_index);
  digest_u64(digest, row_count);
  digest_u64(digest, valid_count);
  digest_text(digest, values_sha);
  digest_text(digest, missing_sha);
}

std::string safe_name(std::string value) {
  for (auto& character : value)
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '-' && character != '_')
      character = '_';
  return value;
}

Json full_catalog() {
  const auto pilot = p::catalog();
  require(pilot.at("candidates").size() == 30,
          "pilot catalog is no longer the frozen 30-candidate scope");
  return {{"schema", "harness-combo-full135-catalog-v1"},
          {"phase", 1},
          {"candidate_scope", "harness_combo_pilot_v1_fixed30"},
          {"candidate_count", 30},
          {"measurement_unit", "complete self-contained HCF1 column container"},
          {"no_cartesian_product_claim", true},
          {"comparison_boundary",
           {{"fixed_baselines", Json::array({"zstd3", "lz4"})},
            {"control18", "new_combination=false within this catalog"},
            {"new_combinations", "new_combination=true within this catalog"},
            {"frozen_algorithm85_included", false},
            {"frozen_operator97_included", false},
            {"frozen_observed_pipeline182_included", false}}},
          {"pilot_catalog", pilot},
          {"candidates", pilot.at("candidates")}};
}

std::string classify_exception(const std::string& message,
                               const std::string& phase) {
  auto lower = message;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  if (lower.find("eligible") != std::string::npos ||
      lower.find("domain") != std::string::npos ||
      lower.find("requires") != std::string::npos ||
      lower.find("representable") != std::string::npos)
    return "domain_rejected";
  if (lower.find("bit mismatch") != std::string::npos ||
      lower.find("nondeterministic") != std::string::npos)
    return "roundtrip_mismatch";
  return phase == "decode" ? "decode_error" : "encode_error";
}

struct CandidateState {
  Json catalog;
  std::size_t ordinal = 0;
  std::string id;
  Path final_container;
  Path temporary_container;
  Path receipt_path;
  Path checkpoint_path;
  std::ofstream output;
  bool resumed = false;
  bool active = true;
  bool completed = false;
  std::string status = "pending";
  std::string error;
  Path failure_artifact;
  std::uint64_t frame_bytes = 0;
  std::uint64_t container_io_wall_ns = 0;
  std::vector<std::uint64_t> encode_wall;
  std::vector<std::uint64_t> encode_cpu;
  std::vector<std::uint64_t> decode_wall;
  std::vector<std::uint64_t> decode_cpu;
  Sha256 ordered_frames;
  Json audit_summary;
  Json result;
};

Json candidate_checkpoint(const CandidateState& state,
                          const std::string& task_sha,
                          const std::string& catalog_sha,
                          const std::string& binary_sha,
                          const std::string& expected_ordered_sha) {
  Json value = state.result;
  value["schema"] = "harness-combo-full135-candidate-checkpoint-v1";
  value["task_sha256"] = task_sha;
  value["catalog_json_sha256"] = catalog_sha;
  value["binary_sha256"] = binary_sha;
  value["expected_ordered_source_blocks_sha256"] = expected_ordered_sha;
  return value;
}

bool valid_success_checkpoint(const Path& path, const std::string& task_sha,
                              const std::string& catalog_sha,
                              const std::string& binary_sha, Json& result) {
  if (!fs::exists(path)) return false;
  try {
    const auto checkpoint = read_json(path);
    if (checkpoint.value("schema", "") !=
            "harness-combo-full135-candidate-checkpoint-v1" ||
        checkpoint.value("task_sha256", "") != task_sha ||
        checkpoint.value("catalog_json_sha256", "") != catalog_sha ||
        checkpoint.value("binary_sha256", "") != binary_sha ||
        checkpoint.value("status", "") != "success")
      return false;
    const Path container = checkpoint.at("container_path").get<std::string>();
    const Path receipt = checkpoint.at("receipt_path").get<std::string>();
    if (!fs::exists(container) || !fs::exists(receipt) ||
        fs::file_size(container) != checkpoint.at("container_bytes") ||
        file_sha256(container) !=
            checkpoint.at("container_sha256").get<std::string>() ||
        file_sha256(receipt) !=
            checkpoint.at("receipt_sha256").get<std::string>())
      return false;
    result = checkpoint;
    return true;
  } catch (...) {
    return false;
  }
}

bool valid_completed_result(const Path& path, const std::string& task_sha,
                            const std::string& catalog_sha,
                            const std::string& binary_sha) {
  if (!fs::exists(path)) return false;
  try {
    const auto result = read_json(path);
    if (result.value("schema", "") != "harness-combo-full135-result-v1" ||
        result.value("task_sha256", "") != task_sha ||
        result.value("catalog_json_sha256", "") != catalog_sha ||
        result.value("binary_sha256", "") != binary_sha ||
        result.value("status", "") != "complete" ||
        result.at("candidates").size() != 30)
      return false;
    for (const auto& candidate : result.at("candidates")) {
      if (candidate.value("status", "") != "success") continue;
      const Path container = candidate.at("container_path").get<std::string>();
      const Path receipt = candidate.at("receipt_path").get<std::string>();
      if (!fs::exists(container) || !fs::exists(receipt) ||
          fs::file_size(container) != candidate.at("container_bytes") ||
          file_sha256(container) !=
              candidate.at("container_sha256").get<std::string>() ||
          file_sha256(receipt) !=
              candidate.at("receipt_sha256").get<std::string>())
        return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

class AttemptArchive {
 public:
  explicit AttemptArchive(Path root) : root_(fs::absolute(std::move(root))) {}

  void preserve(const Path& source, const std::string& reason) {
    if (!fs::exists(source) || !fs::is_regular_file(source)) return;
    ensure_directory();
    std::error_code error;
    auto relative = fs::relative(fs::absolute(source), root_, error);
    if (error || relative.empty() || relative.string().starts_with(".."))
      relative = Path("external") / source.filename();
    auto destination = directory_ / relative;
    fs::create_directories(destination.parent_path());
    if (fs::exists(destination)) {
      auto suffix = std::size_t{1};
      const auto original = destination;
      do {
        destination = original;
        destination += "." + std::to_string(suffix++);
      } while (fs::exists(destination));
    }
    const auto bytes = fs::file_size(source);
    const auto digest = file_sha256(source);
    fs::rename(source, destination);
    entries_.push_back({{"original_path", fs::absolute(source).string()},
                        {"archived_path", destination.string()},
                        {"bytes", bytes},
                        {"sha256", digest},
                        {"reason", reason}});
    flush_manifest();
  }

  bool used() const { return !entries_.empty(); }
  const Path& directory() const { return directory_; }

 private:
  void ensure_directory() {
    if (!directory_.empty()) return;
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now()
                                 .time_since_epoch())
                             .count();
    directory_ = root_ / "attempt_history" /
                 (std::to_string(seconds) + "_" + std::to_string(getpid()));
    fs::create_directories(directory_);
  }
  void flush_manifest() {
    atomic_json(directory_ / "MANIFEST.json",
                {{"schema", "harness-combo-full135-attempt-history-v1"},
                 {"status", "preserved"},
                 {"entry_count", entries_.size()},
                 {"entries", entries_}});
  }

  Path root_;
  Path directory_;
  Json entries_ = Json::array();
};

struct TaskSpec {
  Json raw;
  Json column;
  Json split;
  Json bindings;
  Json manifest;
  Path task_path;
  Path manifest_path;
  std::string task_sha;
  std::string manifest_sha;
  std::string dtype;
  std::string dataset_id;
  std::string column_id;
  std::string column_name;
  std::uint64_t original_numeric_bytes = 0;
  std::uint64_t valid_value_count = 0;
};

TaskSpec load_task(const Path& path, bool allow_synthetic = false) {
  TaskSpec task;
  task.task_path = fs::absolute(path);
  task.task_sha = file_sha256(task.task_path);
  task.raw = read_json(task.task_path);
  require(task.raw.value("schema", "") ==
              "harness-combo-full135-column-task-v1",
          "unexpected column task schema");
  const auto scope = task.raw.value("scope", "");
  require(scope == "legacy_current135" ||
              (allow_synthetic && scope == "synthetic_selftest"),
          "task scope must be legacy_current135");
  require(task.raw.value("phase", 1) == 1,
          "only the phase-1 fixed 30-candidate scope is supported");
  task.column = task.raw.contains("column") ? task.raw.at("column") : task.raw;
  task.split = task.raw.value("split", Json::object());
  task.bindings = task.raw.value("bindings", Json::object());
  for (const auto* field : {"dataset_id", "column_id", "column", "dtype",
                            "manifest_path", "manifest_sha256",
                            "original_numeric_bytes", "valid_value_count"})
    require(task.column.contains(field),
            std::string("task column is missing ") + field);
  task.dataset_id = task.column.at("dataset_id").get<std::string>();
  task.column_id = task.column.at("column_id").get<std::string>();
  task.column_name = task.column.at("column").get<std::string>();
  task.dtype = task.column.at("dtype").get<std::string>();
  (void)dtype_width(task.dtype);
  task.manifest_path = task.column.at("manifest_path").get<std::string>();
  if (task.manifest_path.is_relative())
    task.manifest_path = task.task_path.parent_path() / task.manifest_path;
  task.manifest_sha = task.column.at("manifest_sha256").get<std::string>();
  require(file_sha256(task.manifest_path) == task.manifest_sha,
          "column manifest SHA-256 does not match task");
  task.original_numeric_bytes =
      task.column.at("original_numeric_bytes").get<std::uint64_t>();
  task.valid_value_count =
      task.column.at("valid_value_count").get<std::uint64_t>();
  require(task.valid_value_count <=
              std::numeric_limits<std::uint64_t>::max() /
                  dtype_width(task.dtype) &&
              task.valid_value_count * dtype_width(task.dtype) ==
                  task.original_numeric_bytes,
          "task numeric byte/count contract is inconsistent");
  task.manifest = read_json(task.manifest_path);
  require(task.manifest.value("schema", "") ==
              "native-baseline86-scaled-column-segment-manifest-v1",
          "unexpected source manifest schema");
  require(task.manifest.at("dataset_id") == task.dataset_id,
          "source manifest dataset identity mismatch");
  require(task.manifest.contains("column_identity") &&
              task.manifest.at("column_identity").at("column") ==
                  task.column_name &&
              task.manifest.at("column_identity").at("dtype") == task.dtype,
          "source manifest column identity mismatch");
  require(task.manifest.contains("blocks") &&
              task.manifest.at("blocks").is_array() &&
              task.manifest.at("blocks").size() <= kMaxBlockCount,
          "invalid or excessive source block count");
  return task;
}

Json container_header(const TaskSpec& task, const CandidateState& state,
                      const std::string& catalog_sha,
                      const std::string& binary_sha, unsigned warmups,
                      unsigned repeats) {
  return {{"schema", "harness-combo-full135-container-header-v1"},
          {"container_format", "HCF1"},
          {"phase", 1},
          {"candidate_scope", "harness_combo_pilot_v1_fixed30"},
          {"candidate_id", state.id},
          {"candidate_ordinal", state.ordinal},
          {"catalog_json_sha256", catalog_sha},
          {"binary_sha256", binary_sha},
          {"task_sha256", task.task_sha},
          {"dataset_id", task.dataset_id},
          {"column_id", task.column_id},
          {"column", task.column_name},
          {"dtype", task.dtype},
          {"source_manifest_sha256", task.manifest_sha},
          {"source_block_count", task.manifest.at("blocks").size()},
          {"original_numeric_bytes", task.original_numeric_bytes},
          {"valid_value_count", task.valid_value_count},
          {"warmups", warmups},
          {"repeats", repeats},
          {"lossless_required", true},
          {"strict_required", true}};
}

void open_container(const TaskSpec& task, CandidateState& state,
                    const std::string& catalog_sha,
                    const std::string& binary_sha, unsigned warmups,
                    unsigned repeats) {
  fs::create_directories(state.temporary_container.parent_path());
  state.output.open(state.temporary_container,
                    std::ios::binary | std::ios::trunc);
  require(bool(state.output), "cannot create candidate container: " +
                                  state.temporary_container.string());
  put_magic(state.output, kContainerMagic);
  put_json(state.output,
           container_header(task, state, catalog_sha, binary_sha, warmups,
                            repeats));
}

void abandon_container(CandidateState& state, const std::string& status,
                       const std::string& reason) {
  state.active = false;
  state.status = status;
  state.error = reason;
  if (state.output.is_open()) state.output.close();
  if (fs::exists(state.temporary_container)) {
    state.failure_artifact = state.temporary_container;
    state.failure_artifact += "." + safe_name(status);
    fs::rename(state.temporary_container, state.failure_artifact);
  }
}

void add_frame_digest(Sha256& digest, std::uint64_t index,
                      std::uint64_t bytes, const std::string& frame_sha) {
  digest_u64(digest, index);
  digest_u64(digest, bytes);
  digest_text(digest, frame_sha);
}

void measure_and_write_block(CandidateState& state, const p::Block& block,
                             const Json& metadata,
                             std::uint64_t manifest_array_index,
                             unsigned warmups, unsigned repeats) {
  Bytes selected_frame;
  std::string deterministic_frame_sha;
  Json selected_audit;
  const auto iterations = warmups + repeats;
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    p::Encoded encoded;
    TimePoint encode_elapsed{};
    try {
      const auto start = now();
      encoded = p::encode(state.id, block);
      encode_elapsed = elapsed(start);
    } catch (const std::exception& exception) {
      abandon_container(state, classify_exception(exception.what(), "encode"),
                        exception.what());
      return;
    }
    require(encoded.audit.value("strict", false) &&
                !encoded.audit.value("fallback_used", true) &&
                encoded.audit.value("raw_fallback_stage_count", 1) == 0,
            "pilot runtime returned a non-strict or fallback encode audit");
    p::Block decoded;
    TimePoint decode_elapsed{};
    try {
      const auto start = now();
      decoded = p::decode(encoded.frame);
      decode_elapsed = elapsed(start);
    } catch (const std::exception& exception) {
      abandon_container(state, classify_exception(exception.what(), "decode"),
                        exception.what());
      return;
    }
    if (!(decoded == block)) {
      abandon_container(state, "roundtrip_mismatch",
                        "same-process restored block differs bit-for-bit");
      return;
    }
    const auto iteration_frame_sha = sha256(encoded.frame);
    if (!deterministic_frame_sha.empty() &&
        deterministic_frame_sha != iteration_frame_sha) {
      abandon_container(state, "roundtrip_mismatch",
                        "candidate emitted nondeterministic HCB1 frames");
      return;
    }
    deterministic_frame_sha = iteration_frame_sha;
    if (iteration + 1 == iterations)
      selected_frame = std::move(encoded.frame);
    selected_audit = std::move(encoded.audit);
    if (iteration >= warmups) {
      const auto sample = iteration - warmups;
      state.encode_wall[sample] += encode_elapsed.wall_ns;
      state.encode_cpu[sample] += encode_elapsed.cpu_ns;
      state.decode_wall[sample] += decode_elapsed.wall_ns;
      state.decode_cpu[sample] += decode_elapsed.cpu_ns;
    }
  }
  require(!selected_frame.empty(), "candidate emitted an empty HCB1 frame");
  const auto& frame_sha = deterministic_frame_sha;
  const auto block_id = metadata.at("block_id").get<std::string>();
  const auto entity_id = metadata.at("entity_id").get<std::string>();
  const auto block_index = metadata.at("block_index").get<std::uint64_t>();
  const auto valid_count = metadata.at("valid_count").get<std::uint64_t>();
  Json record{{"schema", "harness-combo-full135-container-block-v1"},
              {"manifest_array_index", manifest_array_index},
              {"block_id", block_id},
              {"entity_id", entity_id},
              {"block_index", block_index},
              {"dtype", block.dtype},
              {"row_count", block.row_count},
              {"valid_count", valid_count},
              {"missing_count", block.row_count - valid_count},
              {"values_sha256", sha256(block.values)},
              {"missing_sha256", sha256(block.missing)},
              {"frame_bytes", selected_frame.size()},
              {"frame_sha256", frame_sha},
              {"strict", true}};
  const auto io_start = now();
  put_magic(state.output, kBlockMagic);
  put_json(state.output, record);
  put_u64(state.output, selected_frame.size());
  state.output.write(reinterpret_cast<const char*>(selected_frame.data()),
                     std::streamsize(selected_frame.size()));
  require(bool(state.output), "failed to write candidate block frame");
  state.container_io_wall_ns += elapsed(io_start).wall_ns;
  state.frame_bytes += selected_frame.size();
  add_frame_digest(state.ordered_frames, manifest_array_index,
                   selected_frame.size(), frame_sha);
  if (state.audit_summary.is_null()) {
    state.audit_summary =
        {{"schema", selected_audit.value("schema", "")},
         {"strict", selected_audit.value("strict", false)},
         {"fallback_used", selected_audit.value("fallback_used", true)},
         {"raw_fallback_stage_count",
          selected_audit.value("raw_fallback_stage_count", 1)}};
  }
}

void finish_container(CandidateState& state, std::uint64_t block_count) {
  const auto frame_order_sha = state.ordered_frames.finish();
  put_magic(state.output, kFooterMagic);
  put_json(state.output,
           {{"schema", "harness-combo-full135-container-footer-v1"},
            {"block_count", block_count},
            {"hcb1_frame_bytes", state.frame_bytes},
            {"ordered_frame_records_sha256", frame_order_sha}});
  state.output.flush();
  require(bool(state.output), "failed to flush candidate container");
  state.output.close();
  fsync_file(state.temporary_container);
  fs::rename(state.temporary_container, state.final_container);
  state.status = "encoded";
}

Json verify_container(const Path& input_path, const Path& output_path) {
  const auto overall = now();
  const auto input_sha = file_sha256(input_path);
  const auto input_bytes = fs::file_size(input_path);
  std::ifstream input(input_path, std::ios::binary);
  require(bool(input), "cannot open HCF1 container: " + input_path.string());
  require(consume_magic(input, kContainerMagic), "invalid HCF1 container magic");
  const auto header = get_json(input);
  require(header.value("schema", "") ==
              "harness-combo-full135-container-header-v1" &&
              header.value("container_format", "") == "HCF1" &&
              header.value("strict_required", false),
          "invalid HCF1 container header");
  const auto verifier_binary_sha = file_sha256(executable_path);
  require(header.at("binary_sha256").get<std::string>() ==
              verifier_binary_sha,
          "HCF1 was produced by a different binary");
  const auto block_count =
      header.at("source_block_count").get<std::uint64_t>();
  require(block_count <= kMaxBlockCount,
          "container declares excessive block count");
  Sha256 restored_order;
  Sha256 frame_order;
  std::uint64_t decoded_frame_bytes = 0;
  std::uint64_t decoded_valid_values = 0;
  std::uint64_t decoded_numeric_bytes = 0;
  std::uint64_t decode_wall_ns = 0;
  std::uint64_t decode_cpu_ns = 0;
  std::set<std::pair<std::string, std::uint64_t>> entity_blocks;
  for (std::uint64_t index = 0; index < block_count; ++index) {
    require(consume_magic(input, kBlockMagic),
            "missing HCF1 block record magic");
    const auto record = get_json(input);
    require(record.value("schema", "") ==
                "harness-combo-full135-container-block-v1" &&
                record.at("manifest_array_index") == index &&
                record.at("dtype") == header.at("dtype") &&
                record.value("strict", false),
            "invalid or out-of-order HCF1 block record");
    const auto frame_bytes = get_u64(input);
    require(frame_bytes == record.at("frame_bytes").get<std::uint64_t>() &&
                frame_bytes <= kMaxFrameBytes,
            "invalid HCB1 frame length in HCF1 container");
    Bytes frame(std::size_t(frame_bytes), std::uint8_t{0});
    if (frame_bytes)
      input.read(reinterpret_cast<char*>(frame.data()),
                 std::streamsize(frame_bytes));
    require(bool(input), "truncated HCB1 frame in HCF1 container");
    const auto frame_sha = sha256(frame);
    require(frame_sha == record.at("frame_sha256").get<std::string>(),
            "HCB1 frame SHA-256 mismatch");
    const auto decode_start = now();
    const auto restored = p::decode(frame);
    const auto decode_elapsed = elapsed(decode_start);
    decode_wall_ns += decode_elapsed.wall_ns;
    decode_cpu_ns += decode_elapsed.cpu_ns;
    const auto valid = validate_block(restored);
    const auto values_sha = sha256(restored.values);
    const auto missing_sha = sha256(restored.missing);
    require(restored.dtype == header.at("dtype").get<std::string>() &&
                restored.row_count ==
                    record.at("row_count").get<std::uint64_t>() &&
                valid == record.at("valid_count").get<std::uint64_t>() &&
                values_sha == record.at("values_sha256").get<std::string>() &&
                missing_sha == record.at("missing_sha256").get<std::string>(),
            "fresh process restoration differs from container source binding");
    const auto block_id = record.at("block_id").get<std::string>();
    const auto entity_id = record.at("entity_id").get<std::string>();
    const auto block_index = record.at("block_index").get<std::uint64_t>();
    require(entity_blocks.emplace(entity_id, block_index).second,
            "duplicate entity/block_index in HCF1 container");
    ordered_block_digest(restored_order, index, block_id, entity_id,
                         block_index, restored.row_count, valid, values_sha,
                         missing_sha);
    add_frame_digest(frame_order, index, frame_bytes, frame_sha);
    decoded_frame_bytes += frame_bytes;
    decoded_valid_values += valid;
    decoded_numeric_bytes += restored.values.size();
  }
  require(consume_magic(input, kFooterMagic), "missing HCF1 footer magic");
  const auto footer = get_json(input);
  require(footer.value("schema", "") ==
              "harness-combo-full135-container-footer-v1" &&
              footer.at("block_count") == block_count &&
              footer.at("hcb1_frame_bytes") == decoded_frame_bytes &&
              footer.at("ordered_frame_records_sha256") == frame_order.finish(),
          "HCF1 footer binding mismatch");
  require(input.peek() == std::char_traits<char>::eof(),
          "trailing bytes after HCF1 footer");
  require(decoded_valid_values ==
              header.at("valid_value_count").get<std::uint64_t>() &&
              decoded_numeric_bytes ==
                  header.at("original_numeric_bytes").get<std::uint64_t>(),
          "restored column totals differ from HCF1 header");
  Json receipt{{"schema", "harness-combo-full135-verify-receipt-v1"},
               {"verified", true},
               {"frame_only_decode", true},
               {"decode_pid", getpid()},
               {"input_path", fs::absolute(input_path).string()},
               {"input_bytes", input_bytes},
               {"input_sha256", input_sha},
               {"candidate_id", header.at("candidate_id")},
               {"candidate_ordinal", header.at("candidate_ordinal")},
               {"task_sha256", header.at("task_sha256")},
               {"catalog_json_sha256", header.at("catalog_json_sha256")},
               {"binary_sha256", verifier_binary_sha},
               {"dataset_id", header.at("dataset_id")},
               {"column_id", header.at("column_id")},
               {"dtype", header.at("dtype")},
               {"source_manifest_sha256",
                header.at("source_manifest_sha256")},
               {"block_count", block_count},
               {"restored_valid_value_count", decoded_valid_values},
               {"restored_numeric_bytes", decoded_numeric_bytes},
               {"ordered_restored_blocks_sha256", restored_order.finish()},
               {"decode", {{"cpu_ns", decode_cpu_ns},
                            {"wall_ns", decode_wall_ns}}},
               {"verify_total", timing_json(elapsed(overall))}};
  atomic_json(output_path, receipt);
  return receipt;
}

Json fresh_verify(const CandidateState& state) {
  const auto start = now();
  const auto child_start = child_cpu_ns();
  const auto pid = fork();
  require(pid >= 0, "fork for fresh verifier failed");
  if (pid == 0) {
#if defined(__linux__)
    rlimit memory{2ULL << 30, 2ULL << 30};
    rlimit core{0, 0};
    if (setrlimit(RLIMIT_AS, &memory) || setrlimit(RLIMIT_CORE, &core))
      _exit(125);
#endif
    execl(executable_path.c_str(), executable_path.c_str(), "verify", "--input",
          state.final_container.c_str(), "--output", state.receipt_path.c_str(),
          nullptr);
    _exit(127);
  }
  int status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(pid, &status, 0);
  } while (waited < 0 && errno == EINTR);
  const auto total = elapsed(start);
  require(waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "fresh verifier process failed for " + state.id +
              " (wait_status=" + std::to_string(status) + ")");
  auto receipt = read_json(state.receipt_path);
  require(receipt.at("verified") == true && receipt.at("decode_pid") == pid &&
              receipt.at("decode_pid") != getpid(),
          "fresh verifier receipt identity mismatch");
  receipt["parent_observed_cold_verify_wall_ns"] = total.wall_ns;
  receipt["parent_observed_child_cpu_ns"] = child_cpu_ns() - child_start;
  return receipt;
}

Json summarize_scope(const Json& candidates) {
  std::uint64_t success = 0, ineligible = 0, domain = 0, failed = 0;
  for (const auto& candidate : candidates) {
    const auto status = candidate.at("status").get<std::string>();
    if (status == "success") ++success;
    else if (status == "ineligible") ++ineligible;
    else if (status == "domain_rejected") ++domain;
    else ++failed;
  }
  return {{"success", success},
          {"ineligible", ineligible},
          {"domain_rejected", domain},
          {"failed", failed}};
}

void mark_pareto(Json& candidates) {
  for (auto& candidate : candidates) candidate["pareto_bytes_encode"] = false;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].at("status") != "success") continue;
    const auto bytes_i = candidates[i].at("container_bytes").get<std::uint64_t>();
    const auto time_i = candidates[i].at("encode").at("wall_median_ns")
                            .get<std::uint64_t>();
    bool dominated = false;
    for (std::size_t j = 0; j < candidates.size(); ++j) {
      if (i == j || candidates[j].at("status") != "success") continue;
      const auto bytes_j =
          candidates[j].at("container_bytes").get<std::uint64_t>();
      const auto time_j = candidates[j].at("encode").at("wall_median_ns")
                              .get<std::uint64_t>();
      if (bytes_j <= bytes_i && time_j <= time_i &&
          (bytes_j < bytes_i || time_j < time_i)) {
        dominated = true;
        break;
      }
    }
    candidates[i]["pareto_bytes_encode"] = !dominated;
  }
}

Json best_success(const Json& candidates, const std::string& metric,
                  bool new_only, bool control_only) {
  const Json* best = nullptr;
  for (const auto& candidate : candidates) {
    if (candidate.at("status") != "success") continue;
    const bool is_new = candidate.value("new_combination", false);
    if (new_only && !is_new) continue;
    if (control_only && is_new) continue;
    const auto value = metric == "bytes"
                           ? candidate.at("container_bytes").get<std::uint64_t>()
                           : candidate.at("encode").at("wall_median_ns")
                                 .get<std::uint64_t>();
    if (best == nullptr) {
      best = &candidate;
    } else {
      const auto prior = metric == "bytes"
                             ? best->at("container_bytes").get<std::uint64_t>()
                             : best->at("encode").at("wall_median_ns")
                                   .get<std::uint64_t>();
      if (value < prior ||
          (value == prior && candidate.at("id").get<std::string>() <
                                 best->at("id").get<std::string>()))
        best = &candidate;
    }
  }
  if (best == nullptr) return nullptr;
  return {{"id", best->at("id")},
          {"container_bytes", best->at("container_bytes")},
          {"encode_wall_median_ns",
           best->at("encode").at("wall_median_ns")}};
}

Json find_candidate(const Json& candidates, const std::string& id) {
  for (const auto& candidate : candidates)
    if (candidate.at("id") == id) return candidate;
  return nullptr;
}

Json run_column(const Path& task_path, const Path& output_directory,
                unsigned warmups, unsigned repeats,
                bool allow_synthetic = false) {
  require(repeats >= 1 && repeats <= 20 && warmups <= 20,
          "repeats must be 1..20 and warmups 0..20");
  const auto run_start = now();
  const auto child_cpu_start = child_cpu_ns();
  const auto task = load_task(task_path, allow_synthetic);
  const auto catalog = full_catalog();
  const auto binary_sha = file_sha256(executable_path);
  // This binds the exact pretty JSON plus newline emitted by `catalog`; the
  // preparation/report scripts bind that file byte-for-byte.
  const auto catalog_sha = sha256(catalog.dump(2) + "\n");
  if (task.bindings.contains("catalog_sha256"))
    require(task.bindings.at("catalog_sha256").get<std::string>() ==
                catalog_sha,
            "task catalog file SHA-256 differs from this fixed HCB30 catalog");
  fs::create_directories(output_directory);
  if (valid_completed_result(output_directory / "RESULT.json", task.task_sha,
                             catalog_sha, binary_sha)) {
    auto prior = read_json(output_directory / "RESULT.json");
    std::cout << Json({{"event", "column_checkpoint_skipped"},
                       {"dataset_id", task.dataset_id},
                       {"column_id", task.column_id},
                       {"result", fs::absolute(output_directory / "RESULT.json")}})
                     .dump()
              << '\n';
    return prior;
  }
  AttemptArchive archive(output_directory);
  archive.preserve(output_directory / "RESULT.json",
                   "terminal result failed task/catalog/binary/file validation");
  archive.preserve(output_directory / "RESULTS.jsonl",
                   "aggregate JSONL belongs to an incomplete attempt");
  const auto catalog_path = output_directory / "CATALOG.json";
  if (fs::exists(catalog_path) && file_sha256(catalog_path) != catalog_sha)
    archive.preserve(catalog_path, "catalog bytes differ from current fixed HCB30 catalog");
  if (!fs::exists(catalog_path)) atomic_json(catalog_path, catalog);
  std::vector<Path> stale_temporary_files;
  for (const auto& item : fs::recursive_directory_iterator(output_directory)) {
    if (!item.is_regular_file()) continue;
    const auto relative = fs::relative(item.path(), output_directory);
    if (!relative.empty() && *relative.begin() == "attempt_history") continue;
    if (item.path().filename().string().find(".tmp.") != std::string::npos)
      stale_temporary_files.push_back(item.path());
  }
  for (const auto& path : stale_temporary_files)
    archive.preserve(path, "temporary file left by an interrupted attempt");
  fs::create_directories(output_directory / "containers");
  fs::create_directories(output_directory / "receipts");
  fs::create_directories(output_directory / "checkpoints");

  std::vector<CandidateState> states;
  states.reserve(30);
  for (const auto& entry : catalog.at("candidates")) {
    states.emplace_back();
    auto& state = states.back();
    state.catalog = entry;
    state.ordinal = entry.at("candidate_ordinal").get<std::size_t>();
    state.id = entry.at("id").get<std::string>();
    std::ostringstream prefix;
    prefix << std::setw(2) << std::setfill('0') << state.ordinal << '-'
           << safe_name(state.id);
    state.final_container = fs::absolute(
        output_directory / "containers" / (prefix.str() + ".hcf"));
    state.temporary_container = state.final_container;
    state.temporary_container += ".tmp." + std::to_string(getpid());
    state.receipt_path = fs::absolute(
        output_directory / "receipts" / (prefix.str() + ".json"));
    state.checkpoint_path = fs::absolute(
        output_directory / "checkpoints" / (prefix.str() + ".json"));
    state.encode_wall.assign(repeats, 0);
    state.encode_cpu.assign(repeats, 0);
    state.decode_wall.assign(repeats, 0);
    state.decode_cpu.assign(repeats, 0);
    Json resumed;
    if (valid_success_checkpoint(state.checkpoint_path, task.task_sha,
                                 catalog_sha, binary_sha, resumed)) {
      state.resumed = true;
      state.active = false;
      state.completed = true;
      state.status = "success";
      state.result = std::move(resumed);
      state.result["resumed_from_checkpoint"] = true;
      continue;
    }
    archive.preserve(state.checkpoint_path,
                     "candidate checkpoint failed task/catalog/binary/file validation");
    archive.preserve(state.final_container,
                     "candidate container has no valid success checkpoint");
    archive.preserve(state.receipt_path,
                     "candidate receipt has no valid success checkpoint");
    open_container(task, state, catalog_sha, binary_sha, warmups, repeats);
  }

  Sha256 expected_order;
  TimePoint source_io_sum{};
  std::uint64_t actual_numeric_bytes = 0;
  std::uint64_t actual_valid_values = 0;
  std::uint64_t actual_rows = 0;
  std::set<std::string> block_ids;
  std::map<std::string, std::uint64_t> next_entity_index;
  const auto& blocks = task.manifest.at("blocks");
  for (std::size_t manifest_array_index = 0;
       manifest_array_index < blocks.size(); ++manifest_array_index) {
    const auto& metadata = blocks.at(manifest_array_index);
    for (const auto* field : {"block_id", "entity_id", "block_index", "dtype",
                              "row_count", "valid_count", "missing_count",
                              "values_path", "values_offset", "values_length",
                              "values_sha256", "missing_bitmap_path",
                              "missing_bitmap_offset", "missing_bitmap_length",
                              "missing_bitmap_sha256"})
      require(metadata.contains(field),
              std::string("manifest block is missing ") + field);
    const auto block_id = metadata.at("block_id").get<std::string>();
    const auto entity_id = metadata.at("entity_id").get<std::string>();
    const auto block_index = metadata.at("block_index").get<std::uint64_t>();
    require(block_ids.insert(block_id).second,
            "duplicate block_id in source manifest");
    require(block_index == next_entity_index[entity_id],
            "manifest order has a non-consecutive block_index within entity " +
                entity_id);
    ++next_entity_index[entity_id];
    const auto io_start = now();
    const auto block = load_block(task.manifest_path, metadata, task.dtype);
    const auto io_elapsed = elapsed(io_start);
    source_io_sum.cpu_ns += io_elapsed.cpu_ns;
    source_io_sum.wall_ns += io_elapsed.wall_ns;
    const auto valid = validate_block(block);
    const auto values_sha = sha256(block.values);
    const auto missing_sha = sha256(block.missing);
    ordered_block_digest(expected_order, manifest_array_index, block_id,
                         entity_id, block_index, block.row_count, valid,
                         values_sha, missing_sha);
    actual_numeric_bytes += block.values.size();
    actual_valid_values += valid;
    actual_rows += block.row_count;

    const auto eligible_list = p::eligible(block);
    const std::set<std::string> eligible(eligible_list.begin(),
                                        eligible_list.end());
    for (auto& state : states) {
      if (!state.active) continue;
      if (!eligible.contains(state.id)) {
        abandon_container(
            state, "ineligible",
            "candidate is ineligible for at least one complete source block");
        continue;
      }
      try {
        measure_and_write_block(state, block, metadata, manifest_array_index,
                                warmups, repeats);
      } catch (const std::exception& exception) {
        abandon_container(state,
                          classify_exception(exception.what(), "encode"),
                          exception.what());
      }
    }
    if ((manifest_array_index + 1) % 1000 == 0 ||
        manifest_array_index + 1 == blocks.size())
      std::cout << Json({{"event", "column_progress"},
                         {"dataset_id", task.dataset_id},
                         {"column_id", task.column_id},
                         {"blocks_complete", manifest_array_index + 1},
                         {"blocks_total", blocks.size()}})
                       .dump()
                << '\n'
                << std::flush;
  }
  require(actual_numeric_bytes == task.original_numeric_bytes &&
              actual_valid_values == task.valid_value_count,
          "streamed source totals differ from task column contract");
  const auto expected_ordered_sha = expected_order.finish();

  for (auto& state : states) {
    if (state.resumed) {
      require(state.result.at("expected_ordered_source_blocks_sha256") ==
                  expected_ordered_sha,
              "candidate checkpoint source ordering differs from current manifest");
      continue;
    }
    if (!state.active) {
      state.result =
          {{"ordinal", state.ordinal},
           {"id", state.id},
           {"kind", state.catalog.at("kind")},
           {"new_combination", state.catalog.at("new_combination")},
           {"status", state.status},
           {"failure_reason", state.error},
           {"rank_eligible", false},
           {"strict", false},
           {"complete_column", false},
           {"warmups", warmups},
           {"repeats", repeats}};
      if (!state.failure_artifact.empty() &&
          fs::exists(state.failure_artifact)) {
        state.result["failure_artifact_path"] =
            state.failure_artifact.string();
        state.result["failure_artifact_bytes"] =
            fs::file_size(state.failure_artifact);
        state.result["failure_artifact_sha256"] =
            file_sha256(state.failure_artifact);
      }
      continue;
    }
    try {
      finish_container(state, blocks.size());
      const auto container_bytes = fs::file_size(state.final_container);
      const auto container_sha = file_sha256(state.final_container);
      const auto receipt = fresh_verify(state);
      require(receipt.at("task_sha256") == task.task_sha &&
                  receipt.at("catalog_json_sha256") == catalog_sha &&
                  receipt.at("binary_sha256") == binary_sha &&
                  receipt.at("candidate_id") == state.id &&
                  receipt.at("source_manifest_sha256") == task.manifest_sha &&
                  receipt.at("input_bytes") == container_bytes &&
                  receipt.at("input_sha256") == container_sha &&
                  receipt.at("ordered_restored_blocks_sha256") ==
                      expected_ordered_sha,
              "fresh verifier receipt differs from parent source binding");
      state.result =
          {{"ordinal", state.ordinal},
           {"id", state.id},
           {"kind", state.catalog.at("kind")},
           {"new_combination", state.catalog.at("new_combination")},
           {"status", "success"},
           {"rank_eligible", true},
           {"strict", true},
           {"complete_column", true},
           {"bit_exact", true},
           {"bit_exact_same_process", true},
           {"bit_exact_fresh_exec", true},
           {"warmups", warmups},
           {"repeats", repeats},
           {"hcb1_frame_bytes", state.frame_bytes},
           {"container_path", state.final_container.string()},
           {"container_bytes", container_bytes},
           {"compressed_bytes", container_bytes},
           {"container_sha256", container_sha},
           {"original_numeric_bytes", task.original_numeric_bytes},
           {"compression_ratio",
            task.original_numeric_bytes == 0
                ? Json(nullptr)
                : Json(double(container_bytes) /
                       double(task.original_numeric_bytes))},
           {"encode", {{"wall_samples_ns", state.encode_wall},
                        {"cpu_samples_ns", state.encode_cpu},
                        {"wall_median_ns", median(state.encode_wall)},
                        {"cpu_median_ns", median(state.encode_cpu)}}},
           {"decode", {{"wall_samples_ns", state.decode_wall},
                        {"cpu_samples_ns", state.decode_cpu},
                        {"wall_median_ns", median(state.decode_wall)},
                        {"cpu_median_ns", median(state.decode_cpu)}}},
           {"container_write_wall_ns", state.container_io_wall_ns},
           {"encode_audit_summary", state.audit_summary},
           {"receipt_path", state.receipt_path.string()},
           {"receipt_sha256", file_sha256(state.receipt_path)},
           {"independent_decode",
            {{"verified", true},
             {"decode_pid", receipt.at("decode_pid")},
             {"frame_only_decode", true},
             {"ordered_restored_blocks_sha256",
              receipt.at("ordered_restored_blocks_sha256")}}},
           {"cold_verify_wall_ns",
            receipt.at("parent_observed_cold_verify_wall_ns")},
           {"cold_verify_child_cpu_ns",
            receipt.at("parent_observed_child_cpu_ns")},
           {"fresh_verify_decode", receipt.at("decode")},
           {"expected_ordered_source_blocks_sha256", expected_ordered_sha},
           {"resumed_from_checkpoint", false}};
      state.status = "success";
      state.completed = true;
      atomic_json(state.checkpoint_path,
                  candidate_checkpoint(state, task.task_sha, catalog_sha,
                                       binary_sha, expected_ordered_sha));
    } catch (const std::exception& exception) {
      if (fs::exists(state.temporary_container))
        abandon_container(state, "verify_failed", exception.what());
      state.status = "verify_failed";
      state.error = exception.what();
      state.result =
          {{"ordinal", state.ordinal},
           {"id", state.id},
           {"kind", state.catalog.at("kind")},
           {"new_combination", state.catalog.at("new_combination")},
           {"status", "verify_failed"},
           {"failure_reason", state.error},
           {"rank_eligible", false},
           {"strict", false},
           {"complete_column", false},
           {"container_path", state.final_container.string()},
           {"container_retained", fs::exists(state.final_container)},
           {"warmups", warmups},
           {"repeats", repeats}};
      if (!state.failure_artifact.empty() &&
          fs::exists(state.failure_artifact)) {
        state.result["failure_artifact_path"] =
            state.failure_artifact.string();
        state.result["failure_artifact_bytes"] =
            fs::file_size(state.failure_artifact);
        state.result["failure_artifact_sha256"] =
            file_sha256(state.failure_artifact);
      }
    }
  }

  Json candidate_results = Json::array();
  std::sort(states.begin(), states.end(),
            [](const CandidateState& left, const CandidateState& right) {
              return left.ordinal < right.ordinal;
            });
  for (auto& state : states) {
    state.result["ordinal"] = state.ordinal;
    state.result["id"] = state.id;
    state.result["candidate_id"] = state.id;
    state.result["kind"] = state.catalog.at("kind");
    state.result["new_combination"] = state.catalog.at("new_combination");
    candidate_results.push_back(state.result);
  }
  mark_pareto(candidate_results);
  const auto zstd3 = find_candidate(candidate_results, "zstd3");
  const auto lz4 = find_candidate(candidate_results, "lz4");
  const auto best_all_bytes = best_success(candidate_results, "bytes", false, false);
  const auto best_all_time = best_success(candidate_results, "time", false, false);
  const auto best_new_bytes = best_success(candidate_results, "bytes", true, false);
  const auto best_new_time = best_success(candidate_results, "time", true, false);
  const auto best_control_bytes = best_success(candidate_results, "bytes", false, true);
  const auto best_control_time = best_success(candidate_results, "time", false, true);
  std::string outcome = "no_rankable_new_combination";
  bool improve_bytes = false, improve_time = false;
  if (!best_new_bytes.is_null() && !best_control_bytes.is_null())
    improve_bytes = best_new_bytes.at("container_bytes").get<std::uint64_t>() <
                    best_control_bytes.at("container_bytes").get<std::uint64_t>();
  if (!best_new_time.is_null() && !best_control_time.is_null())
    improve_time =
        best_new_time.at("encode_wall_median_ns").get<std::uint64_t>() <
        best_control_time.at("encode_wall_median_ns").get<std::uint64_t>();
  if (!best_new_bytes.is_null()) {
    if (improve_bytes && improve_time) outcome = "new_combo_improves_bytes_and_encode_time";
    else if (improve_bytes) outcome = "new_combo_improves_bytes";
    else if (improve_time) outcome = "new_combo_improves_encode_time";
    else outcome = "new_combo_no_improvement_over_control18";
  }
  const auto run_elapsed = elapsed(run_start);
  Json result{{"schema", "harness-combo-full135-result-v1"},
              {"status", "complete"},
              {"execution_status",
               summarize_scope(candidate_results).at("failed") == 0 &&
                       summarize_scope(candidate_results).at("ineligible") == 0 &&
                       summarize_scope(candidate_results).at("domain_rejected") == 0
                   ? "complete_all_candidates"
                   : "complete_with_rejections"},
              {"phase", 1},
              {"candidate_scope", "harness_combo_pilot_v1_fixed30"},
              {"candidate_count", 30},
              {"task_id", task.raw.value("task_id", task.column_id)},
              {"task_ordinal", task.raw.value("ordinal", -1)},
              {"task_path", task.task_path.string()},
              {"task_sha256", task.task_sha},
              {"catalog_json_sha256", catalog_sha},
              {"binary_sha256", binary_sha},
              {"column", task.column},
              {"split", task.split},
              {"bindings", task.bindings},
              {"source_manifest_path", task.manifest_path.string()},
              {"source_manifest_sha256", task.manifest_sha},
              {"source_block_count", blocks.size()},
              {"source_entity_count", next_entity_index.size()},
              {"source_row_count", actual_rows},
              {"source_valid_value_count", actual_valid_values},
              {"source_numeric_bytes", actual_numeric_bytes},
              {"ordered_source_blocks_sha256", expected_ordered_sha},
              {"source_io_and_sha", timing_json(source_io_sum)},
              {"warmups", warmups},
              {"repeats", repeats},
              {"timing_contract",
               {{"samples", "per-repeat wall/cpu nanoseconds summed across every source block; median is taken after summation"},
                {"encode_includes", "pilot validation, strict graph execution, HCB1 framing and audit"},
                {"encode_excludes", "source IO, HCF1 container write and fresh verifier"},
                {"decode_includes", "HCB1 parse, strict inverse graph and binding checks"},
                {"cold_verify", "fresh exec reads the HCF1 container and decodes all frames"},
                {"end_to_end", "task/manifest checks, source IO/SHA, all candidate trials, container IO, cold verification and result commit except final RESULT write"}}},
              {"candidates", candidate_results},
              {"failure_summary", summarize_scope(candidate_results)},
              {"comparison",
               {{"byte_metric", "actual complete HCF1 container file bytes"},
                {"time_metric", "median total encode wall ns across all blocks"},
                {"best_all_by_bytes", best_all_bytes},
                {"best_all_by_encode_time", best_all_time},
                {"best_new_combination_by_bytes", best_new_bytes},
                {"best_new_combination_by_encode_time", best_new_time},
                {"best_control18_by_bytes", best_control_bytes},
                {"best_control18_by_encode_time", best_control_time},
                {"fixed_zstd3", zstd3},
                {"fixed_lz4", lz4},
                {"new_improves_control18_bytes", improve_bytes},
                {"new_improves_control18_encode_time", improve_time},
                {"outcome_class", outcome},
                {"independent_optima_are_not_joint_optimum", true}}},
              {"oracle_comparability",
               {{"frozen_algorithm85_measured_here", false},
                {"frozen_operator97_measured_here", false},
                {"frozen_observed_pipeline182_measured_here", false},
                {"full_operator_by_algorithm_cartesian_product", false},
                {"valid_claim", "within-column comparison over the fixed HCB30 catalog only"}}},
              {"end_to_end_parent", timing_json(run_elapsed)},
              {"fresh_verifier_children_cpu_ns", child_cpu_ns() - child_cpu_start},
              {"memory_contract", "one source block and one candidate frame at a time; fresh verifier RLIMIT_AS=2GiB"},
              {"fallback_in_strict_rank", false}};
  if (archive.used())
    result["preserved_attempt_history"] = archive.directory().string();
  atomic_jsonl(output_directory / "RESULTS.jsonl", candidate_results);
  atomic_json(output_directory / "RESULT.json", result);
  std::cout << Json({{"event", "column_complete"},
                     {"dataset_id", task.dataset_id},
                     {"column_id", task.column_id},
                     {"result", fs::absolute(output_directory / "RESULT.json")},
                     {"outcome_class", outcome}})
                   .dump()
            << '\n';
  return result;
}

void write_binary(const Path& path, std::span<const std::uint8_t> bytes) {
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  auto temporary = path;
  temporary += ".tmp." + std::to_string(getpid());
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(bool(output), "cannot create binary fixture");
    if (!bytes.empty())
      output.write(reinterpret_cast<const char*>(bytes.data()),
                   std::streamsize(bytes.size()));
    require(bool(output), "cannot write binary fixture");
  }
  fs::rename(temporary, path);
}

void append_u64(Bytes& output, std::uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    output.push_back(std::uint8_t(value >> (8 * byte)));
}

int selftest(const Path& output_directory) {
  const auto fixture = output_directory / "fixture";
  fs::create_directories(fixture);
  const std::array<std::vector<std::uint64_t>, 2> values{{
      {0x0000000000000000ULL, 0x8000000000000000ULL,
       0x7ff8000000001234ULL, 0x7ff0000000000000ULL,
       0xfff0000000000000ULL, 0x0000000000000001ULL,
       0x7fefffffffffffffULL},
      {0x3ff0000000000000ULL, 0xbff0000000000000ULL,
       0x000fffffffffffffULL, 0x7ff0000000000001ULL}}};
  const std::array<std::uint64_t, 2> rows{9, 5};
  const std::array<Bytes, 2> masks{Bytes{0x84, 0x00}, Bytes{0x04}};
  Json blocks = Json::array();
  std::uint64_t total_values = 0;
  std::uint64_t total_valid = 0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    Bytes value_region{0xaa, 0xbb, 0xcc};
    for (const auto bits : values[index]) append_u64(value_region, bits);
    value_region.push_back(0xdd);
    Bytes mask_region{0x55};
    mask_region.insert(mask_region.end(), masks[index].begin(),
                       masks[index].end());
    mask_region.push_back(0xee);
    const auto value_path = fixture / ("values-" + std::to_string(index) + ".bin");
    const auto mask_path = fixture / ("missing-" + std::to_string(index) + ".bin");
    write_binary(value_path, value_region);
    write_binary(mask_path, mask_region);
    Bytes compact(value_region.begin() + 3, value_region.end() - 1);
    Bytes bitmap = masks[index];
    const auto valid = values[index].size();
    total_values += compact.size();
    total_valid += valid;
    blocks.push_back(
        {{"block_id", "fixture:block:" + std::to_string(index)},
         {"entity_id", "fixture-entity"},
         {"block_index", index},
         {"dtype", "float64"},
         {"row_count", rows[index]},
         {"valid_count", valid},
         {"missing_count", rows[index] - valid},
         {"values_path", value_path.filename().string()},
         {"values_offset", 3},
         {"values_length", compact.size()},
         {"values_sha256", sha256(compact)},
         {"missing_bitmap_path", mask_path.filename().string()},
         {"missing_bitmap_offset", 1},
         {"missing_bitmap_length", bitmap.size()},
         {"missing_bitmap_sha256", sha256(bitmap)}});
  }
  const auto manifest_path = fixture / "manifest.json";
  atomic_json(manifest_path,
              {{"schema", "native-baseline86-scaled-column-segment-manifest-v1"},
               {"dataset_id", "synthetic_full135_fixture"},
               {"column_identity", {{"column", "special_bits"},
                                    {"dtype", "float64"}}},
               {"blocks", blocks}});
  const auto task_path = fixture / "TASK.json";
  atomic_json(task_path,
              {{"schema", "harness-combo-full135-column-task-v1"},
               {"scope", "synthetic_selftest"},
               {"phase", 1},
               {"column",
                {{"dataset_id", "synthetic_full135_fixture"},
                 {"column_id", "fixture_special_bits_float64"},
                 {"column", "special_bits"},
                 {"dtype", "float64"},
                 {"manifest_path", manifest_path.string()},
                 {"manifest_sha256", file_sha256(manifest_path)},
                 {"original_numeric_bytes", total_values},
                 {"valid_value_count", total_valid}}},
               {"split", {{"split", "development"},
                          {"source_group", "synthetic"}}},
               {"bindings", {{"fixture", true}}}});
  const auto result = run_column(task_path, output_directory / "run", 0, 1,
                                 true);
  const auto zstd3 = find_candidate(result.at("candidates"), "zstd3");
  const auto lz4 = find_candidate(result.at("candidates"), "lz4");
  require(zstd3.at("status") == "success" && lz4.at("status") == "success",
          "selftest fixed baselines did not pass");
  const auto summary = result.at("failure_summary");
  require(summary.at("failed").get<std::uint64_t>() == 0 &&
              summary.at("domain_rejected").get<std::uint64_t>() == 0 &&
              summary.at("success").get<std::uint64_t>() +
                      summary.at("ineligible").get<std::uint64_t>() ==
                  30,
          "selftest observed an unexpected eligible-candidate failure");
  Json receipt{{"schema", "harness-combo-full135-selftest-v1"},
               {"status", "passed"},
               {"failures", 0},
               {"fixture", "two blocks, non-byte-aligned missing bitmaps, nonzero file offsets, signed zeros, NaN payloads, infinities and subnormals"},
               {"candidate_count", 30},
               {"failure_summary", summary},
               {"result_path", fs::absolute(output_directory / "run" / "RESULT.json").string()},
               {"result_sha256", file_sha256(output_directory / "run" / "RESULT.json")}};
  atomic_json(output_directory / "SELFTEST.json", receipt);
  std::cout << receipt.dump() << '\n';
  return 0;
}

std::map<std::string, std::string> parse_options(int argc, char** argv,
                                                 int start) {
  std::map<std::string, std::string> result;
  for (int index = start; index < argc; index += 2) {
    require(index + 1 < argc, "option has no value: " + std::string(argv[index]));
    const std::string key = argv[index];
    require(key.starts_with("--"), "unexpected positional argument: " + key);
    require(result.emplace(key, argv[index + 1]).second,
            "duplicate option: " + key);
  }
  return result;
}

std::string required_option(const std::map<std::string, std::string>& options,
                            const std::string& key) {
  const auto found = options.find(key);
  require(found != options.end(), "missing required option " + key);
  return found->second;
}

unsigned unsigned_option(const std::map<std::string, std::string>& options,
                         const std::string& key, unsigned fallback) {
  const auto found = options.find(key);
  if (found == options.end()) return fallback;
  std::size_t parsed = 0;
  const auto value = std::stoul(found->second, &parsed);
  require(parsed == found->second.size(), "invalid integer option " + key);
  return unsigned(value);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 2, "usage: catalog|run-column|verify|selftest [options]");
    try {
      executable_path = fs::canonical("/proc/self/exe");
    } catch (...) {
      executable_path = fs::canonical(argv[0]);
    }
    const std::string command = argv[1];
    const auto options = parse_options(argc, argv, 2);
    if (command == "catalog") {
      require(options.size() == 1 && options.contains("--output"),
              "catalog usage: catalog --output PATH");
      atomic_json(required_option(options, "--output"), full_catalog());
      return 0;
    }
    if (command == "run-column") {
      for (const auto& [key, value] : options) {
        (void)value;
        require(key == "--task" || key == "--out" || key == "--warmups" ||
                    key == "--repeats",
                "unknown run-column option: " + key);
      }
      run_column(required_option(options, "--task"),
                 required_option(options, "--out"),
                 unsigned_option(options, "--warmups", 1),
                 unsigned_option(options, "--repeats", 3));
      return 0;
    }
    if (command == "verify") {
      require(options.size() == 2 && options.contains("--input") &&
                  options.contains("--output"),
              "verify usage: verify --input CONTAINER --output RECEIPT");
      verify_container(required_option(options, "--input"),
                       required_option(options, "--output"));
      return 0;
    }
    if (command == "selftest") {
      require(options.size() == 1 && options.contains("--out"),
              "selftest usage: selftest --out DIR");
      return selftest(required_option(options, "--out"));
    }
    fail("unknown command: " + command);
  } catch (const std::exception& exception) {
    std::cerr << Json({{"event", "error"}, {"error", exception.what()}}).dump()
              << '\n';
    return 1;
  }
}
