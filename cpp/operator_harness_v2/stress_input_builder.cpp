// Build byte-exact real-data slices and deterministic boundary fixtures for
// operator-tree stress validation.  This utility never executes an operator.
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {
using J = nlohmann::json;
namespace fs = std::filesystem;

constexpr std::uint64_t kMaxSpecBytes = 64ULL << 20;
constexpr std::uint64_t kMaxBlockBytes = 64ULL << 20;
constexpr std::uint64_t kMaxSampleBytes = 16ULL << 20;
constexpr std::uint64_t kMaxVerifiedSourceBytesPerSample = 64ULL << 20;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b,
                          const std::string& label) {
  require(a <= std::numeric_limits<std::uint64_t>::max() - b,
          label + " overflow");
  return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b,
                          const std::string& label) {
  require(a == 0 || b <= std::numeric_limits<std::uint64_t>::max() / a,
          label + " overflow");
  return a * b;
}

std::string read_file(const fs::path& path, std::uint64_t maximum) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(bool(input), "cannot read " + path.string());
  const auto end = input.tellg();
  require(end >= 0 && static_cast<std::uint64_t>(end) <= maximum,
          "oversized file " + path.string());
  input.seekg(0);
  std::string result(static_cast<std::size_t>(end), '\0');
  if (!result.empty()) input.read(result.data(), static_cast<std::streamsize>(result.size()));
  require(bool(input) || result.empty(), "short read " + path.string());
  return result;
}

std::string sha256(const void* data, std::size_t size) {
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  require(context != nullptr, "cannot allocate SHA256 context");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned length = 0;
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, data, size) == 1 &&
                  EVP_DigestFinal_ex(context, digest.data(), &length) == 1;
  EVP_MD_CTX_free(context);
  require(ok && length == 32, "SHA256 failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned i = 0; i < length; ++i)
    output << std::setw(2) << static_cast<unsigned>(digest[i]);
  return output.str();
}

std::string sha256(const std::string& bytes) {
  return sha256(bytes.data(), bytes.size());
}

bool digest_string(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

std::string text(const J& value, const char* key) {
  require(value.contains(key) && value.at(key).is_string(),
          std::string("missing string ") + key);
  const auto result = value.at(key).get<std::string>();
  require(!result.empty() && result.find('\0') == std::string::npos,
          std::string("invalid string ") + key);
  return result;
}

std::uint64_t number(const J& value, const char* key,
                     std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max()) {
  require(value.contains(key) && value.at(key).is_number_unsigned(),
          std::string("missing unsigned integer ") + key);
  const auto result = value.at(key).get<std::uint64_t>();
  require(result <= maximum, std::string(key) + " exceeds limit");
  return result;
}

void valid_sample_id(const std::string& id) {
  static const std::regex expression("^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$");
  require(std::regex_match(id, expression), "invalid sample_id " + id);
}

void write_all(int descriptor, const char* data, std::size_t size) {
  std::size_t position = 0;
  while (position < size) {
    const auto count = ::write(descriptor, data + position, size - position);
    if (count < 0 && errno == EINTR) continue;
    require(count > 0, "write failed");
    position += static_cast<std::size_t>(count);
  }
}

void sync_directory(const fs::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
  require(descriptor >= 0, "cannot open output directory for fsync");
  const int status = ::fsync(descriptor);
  ::close(descriptor);
  require(status == 0, "output directory fsync failed");
}

void atomic_write(const fs::path& path, const std::string& bytes, mode_t mode) {
  fs::create_directories(path.parent_path());
  if (fs::exists(path)) {
    require(read_file(path, kMaxSpecBytes) == bytes,
            "refusing to replace changed output " + path.string());
    return;
  }
  const auto temporary = path.string() + ".tmp-" + std::to_string(::getpid());
  const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(descriptor >= 0, "cannot create temporary output " + temporary);
  try {
    write_all(descriptor, bytes.data(), bytes.size());
    require(::fsync(descriptor) == 0, "output fsync failed");
    require(::fchmod(descriptor, mode) == 0, "output chmod failed");
    require(::close(descriptor) == 0, "output close failed");
    require(::rename(temporary.c_str(), path.c_str()) == 0, "output rename failed");
    sync_directory(path.parent_path());
  } catch (...) {
    ::close(descriptor);
    ::unlink(temporary.c_str());
    throw;
  }
}

std::string read_region(const fs::path& path, std::uint64_t offset,
                        std::uint64_t length) {
  require(length <= kMaxBlockBytes, "source block exceeds limit");
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  require(descriptor >= 0, "cannot open source values " + path.string());
  struct stat status {};
  require(::fstat(descriptor, &status) == 0 && status.st_size >= 0,
          "cannot stat source values");
  require(checked_add(offset, length, "source region") <=
              static_cast<std::uint64_t>(status.st_size),
          "source block is outside values file");
  std::string result(static_cast<std::size_t>(length), '\0');
  std::size_t position = 0;
  while (position < result.size()) {
    const auto count = ::pread(descriptor, result.data() + position,
                               result.size() - position,
                               static_cast<off_t>(offset + position));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) break;
    position += static_cast<std::size_t>(count);
  }
  ::close(descriptor);
  require(position == result.size(), "short source block read");
  return result;
}

void append_le(std::string& output, std::uint64_t value, unsigned width) {
  for (unsigned index = 0; index < width; ++index)
    output.push_back(static_cast<char>((value >> (8U * index)) & 0xffU));
}

std::uint64_t xorshift64(std::uint64_t value) {
  value ^= value << 13U;
  value ^= value >> 7U;
  value ^= value << 17U;
  return value;
}

std::string boundary_bytes(const J& fixture) {
  const auto name = text(fixture, "fixture");
  const auto width = static_cast<unsigned>(number(fixture, "source_byte_width", 8));
  const auto count = number(fixture, "element_count", kMaxSampleBytes);
  const auto byte_count = checked_mul(width, count, "boundary byte count");
  require(width > 0 && byte_count <= kMaxSampleBytes, "invalid boundary size");
  std::string output;
  output.reserve(static_cast<std::size_t>(byte_count));
  if (name == "empty") {
    require(count == 0, "empty fixture requires element_count=0");
    return output;
  }
  if (name == "bytes_pattern_cycle" || name == "bytes_tail_cycle") {
    require(width == 1, "byte pattern fixture requires width 1");
    constexpr std::array<std::uint8_t, 16> pattern = {
        0x00, 0xff, 0x01, 0xfe, 0x55, 0xaa, 0x7f, 0x80,
        0x10, 0x20, 0x40, 0x08, 0xf0, 0x0f, 0x33, 0xcc};
    for (std::uint64_t i = 0; i < count; ++i)
      output.push_back(static_cast<char>(pattern[static_cast<std::size_t>(i % pattern.size())]));
    return output;
  }
  if (name == "bytes_singleton_ff") {
    require(width == 1 && count == 1, "bytes_singleton_ff requires one byte");
    output.push_back(static_cast<char>(0xff));
    return output;
  }

  if (name == "int64_singleton_min" || name == "int64_singleton_max") {
    require(width == 8 && count == 1, "int64 singleton requires width 8/count 1");
    append_le(output, name == "int64_singleton_min" ? UINT64_C(0x8000000000000000)
                                                     : UINT64_C(0x7fffffffffffffff), 8);
    return output;
  }
  if (name == "int64_extrema_cycle" || name == "int64_tail_cycle") {
    require(width == 8, "int64 fixture requires width 8");
    constexpr std::array<std::uint64_t, 16> values = {
        UINT64_C(0x8000000000000000), UINT64_C(0x7fffffffffffffff), 0, 1,
        UINT64_C(0xffffffffffffffff), 2, UINT64_C(0xfffffffffffffffe),
        UINT64_C(0x000000007fffffff), UINT64_C(0xffffffff80000000),
        UINT64_C(0x5555555555555555), UINT64_C(0xaaaaaaaaaaaaaaaa),
        UINT64_C(0x0000000100000000), UINT64_C(0xffffffff00000000),
        UINT64_C(0x0101010101010101), UINT64_C(0x8080808080808080), 17};
    for (std::uint64_t i = 0; i < count; ++i)
      append_le(output, values[static_cast<std::size_t>(i % values.size())], 8);
    return output;
  }
  if (name == "uint64_singleton_max") {
    require(width == 8 && count == 1, "uint64 singleton requires width 8/count 1");
    append_le(output, std::numeric_limits<std::uint64_t>::max(), 8);
    return output;
  }
  if (name == "uint64_bit_patterns_cycle") {
    require(width == 8, "uint64 fixture requires width 8");
    constexpr std::array<std::uint64_t, 12> values = {
        0, 1, UINT64_MAX, UINT64_C(0x8000000000000000),
        UINT64_C(0x7fffffffffffffff), UINT64_C(0x5555555555555555),
        UINT64_C(0xaaaaaaaaaaaaaaaa), UINT64_C(0x0101010101010101),
        UINT64_C(0x8080808080808080), UINT64_C(0x00000000ffffffff),
        UINT64_C(0xffffffff00000000), UINT64_C(0x0123456789abcdef)};
    for (std::uint64_t i = 0; i < count; ++i)
      append_le(output, values[static_cast<std::size_t>(i % values.size())], 8);
    return output;
  }
  if (name == "uint64_bounded_cycle") {
    require(width == 8, "bounded uint64 fixture requires width 8");
    const auto bits = number(fixture, "bit_width", 64);
    const auto mask = bits == 64 ? UINT64_MAX : bits == 0 ? 0 : (UINT64_C(1) << bits) - 1;
    std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15) ^ bits;
    for (std::uint64_t i = 0; i < count; ++i) {
      state = xorshift64(state + i + 1);
      append_le(output, state & mask, 8);
    }
    return output;
  }

  constexpr std::array<std::uint64_t, 12> finite64 = {
      UINT64_C(0x0000000000000000), UINT64_C(0x8000000000000000),
      UINT64_C(0x0000000000000001), UINT64_C(0x8000000000000001),
      UINT64_C(0x0010000000000000), UINT64_C(0x8010000000000000),
      UINT64_C(0x7fefffffffffffff), UINT64_C(0xffefffffffffffff),
      UINT64_C(0x3ff0000000000000), UINT64_C(0xbff0000000000000),
      UINT64_C(0x3fd5555555555555), UINT64_C(0xc005bf0a8b145769)};
  constexpr std::array<std::uint64_t, 18> special64 = {
      0, UINT64_C(0x8000000000000000), 1, UINT64_C(0x8000000000000001),
      UINT64_C(0x0010000000000000), UINT64_C(0x8010000000000000),
      UINT64_C(0x7fefffffffffffff), UINT64_C(0xffefffffffffffff),
      UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
      UINT64_C(0x7ff8000000000000), UINT64_C(0x7ff8000000000001),
      UINT64_C(0xfff8000000000001), UINT64_C(0x7ff0000000000001),
      UINT64_C(0x3ff0000000000000), UINT64_C(0xbff0000000000000),
      UINT64_C(0x3fd5555555555555), UINT64_C(0xc005bf0a8b145769)};
  if (name == "float64_singleton_negative_zero") {
    require(width == 8 && count == 1, "float64 singleton requires width 8/count 1");
    append_le(output, UINT64_C(0x8000000000000000), 8);
    return output;
  }
  if (name == "float64_finite_extrema_cycle" || name == "float64_tail_cycle" ||
      name == "float64_special_cycle") {
    require(width == 8, "float64 fixture requires width 8");
    for (std::uint64_t i = 0; i < count; ++i) {
      const auto value = name == "float64_special_cycle"
                             ? special64[static_cast<std::size_t>(i % special64.size())]
                             : finite64[static_cast<std::size_t>(i % finite64.size())];
      append_le(output, value, 8);
    }
    return output;
  }

  constexpr std::array<std::uint32_t, 12> finite32 = {
      UINT32_C(0x00000000), UINT32_C(0x80000000), UINT32_C(0x00000001),
      UINT32_C(0x80000001), UINT32_C(0x00800000), UINT32_C(0x80800000),
      UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff), UINT32_C(0x3f800000),
      UINT32_C(0xbf800000), UINT32_C(0x3eaaaaab), UINT32_C(0xc02df854)};
  constexpr std::array<std::uint32_t, 18> special32 = {
      0, UINT32_C(0x80000000), 1, UINT32_C(0x80000001), UINT32_C(0x00800000),
      UINT32_C(0x80800000), UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff),
      UINT32_C(0x7f800000), UINT32_C(0xff800000), UINT32_C(0x7fc00000),
      UINT32_C(0x7fc00001), UINT32_C(0xffc00001), UINT32_C(0x7f800001),
      UINT32_C(0x3f800000), UINT32_C(0xbf800000), UINT32_C(0x3eaaaaab),
      UINT32_C(0xc02df854)};
  if (name == "float32_singleton_negative_zero") {
    require(width == 4 && count == 1, "float32 singleton requires width 4/count 1");
    append_le(output, UINT32_C(0x80000000), 4);
    return output;
  }
  if (name == "float32_finite_extrema_cycle" || name == "float32_tail_cycle" ||
      name == "float32_special_cycle" || name == "uint32_bit_patterns_cycle") {
    require(width == 4, "float32/uint32 fixture requires width 4");
    for (std::uint64_t i = 0; i < count; ++i) {
      std::uint32_t value = 0;
      if (name == "float32_special_cycle")
        value = special32[static_cast<std::size_t>(i % special32.size())];
      else if (name == "uint32_bit_patterns_cycle")
        value = static_cast<std::uint32_t>(UINT32_C(0x9e3779b9) * static_cast<std::uint32_t>(i + 1));
      else
        value = finite32[static_cast<std::size_t>(i % finite32.size())];
      append_le(output, value, 4);
    }
    return output;
  }
  throw std::runtime_error("unknown boundary fixture " + name);
}

J real_sample(const J& item) {
  const auto id = text(item, "sample_id");
  valid_sample_id(id);
  const auto width = number(item, "source_byte_width", 8);
  require(width > 0, "source_byte_width must be positive");
  const auto count = number(item, "element_count");
  const auto sample_bytes = checked_mul(width, count, "real sample byte count");
  require(sample_bytes <= kMaxSampleBytes, "real sample exceeds limit");
  const auto manifest_path = fs::path(text(item, "source_manifest"));
  const auto manifest_digest = text(item, "source_manifest_sha256");
  require(manifest_path.is_absolute() && digest_string(manifest_digest),
          "invalid source manifest provenance");
  const auto manifest_bytes = read_file(manifest_path, kMaxSpecBytes);
  require(sha256(manifest_bytes) == manifest_digest,
          "source manifest SHA256 mismatch");
  const auto manifest = J::parse(manifest_bytes);
  require(manifest.is_object() &&
              manifest.value("schema", "") ==
                  "native-baseline86-scaled-column-segment-manifest-v1",
          "unexpected source manifest schema");
  require(item.contains("column_identity") && item.at("column_identity").is_object() &&
              manifest.contains("column_identity") &&
              manifest.at("column_identity") == item.at("column_identity"),
          "source manifest column identity mismatch");
  require(manifest.contains("blocks") && manifest.at("blocks").is_array() &&
              !manifest.at("blocks").empty(),
          "source manifest has no accepted blocks");
  std::string bytes;
  bytes.reserve(static_cast<std::size_t>(sample_bytes));
  J boundaries = J::array();
  std::uint64_t verified_source_bytes = 0;
  std::uint64_t previous_manifest_array_index = 0;
  bool first_segment = true;
  auto append_segment = [&](const J& segment) {
    const auto manifest_array_index = number(segment, "source_block_index",
                                             manifest.at("blocks").size() - 1);
    if (!first_segment)
      require(manifest_array_index == previous_manifest_array_index + 1,
              "real source segments are not consecutive in manifest order");
    first_segment = false;
    previous_manifest_array_index = manifest_array_index;
    const auto& manifest_block =
        manifest.at("blocks").at(static_cast<std::size_t>(manifest_array_index));
    require(manifest_block.is_object(), "source manifest block is not an object");
    require(text(segment, "source_block_id") == text(manifest_block, "block_id"),
            "source segment block ID is not the selected manifest member");
    const auto block_offset = number(segment, "source_block_offset");
    const auto block_bytes = number(segment, "source_block_bytes", kMaxBlockBytes);
    const auto slice_offset = number(segment, "slice_offset_in_source_block_bytes", block_bytes);
    const auto segment_count = number(segment, "element_count");
    const auto segment_bytes = checked_mul(width, segment_count, "real segment byte count");
    require(slice_offset % width == 0, "real segment is not element aligned");
    require(checked_add(slice_offset, segment_bytes, "real segment") <= block_bytes,
            "real segment exceeds accepted block");
    const auto expected = text(segment, "source_block_sha256");
    require(digest_string(expected), "invalid source block SHA256");
    const auto values_path = fs::path(text(segment, "source_values_path"));
    require(values_path.is_absolute(), "source_values_path must be absolute");
    require(text(manifest_block, "values_path") == values_path.string() &&
                number(manifest_block, "values_offset") == block_offset &&
                number(manifest_block, "values_length", kMaxBlockBytes) == block_bytes &&
                text(manifest_block, "values_sha256") == expected,
            "source segment path/offset/length/SHA is not the selected manifest member");
    require(text(manifest_block, "dtype") == text(item, "dtype"),
            "source segment dtype differs from manifest block");
    const auto manifest_valid_count = number(manifest_block, "valid_count");
    require(number(segment, "source_block_valid_count") == manifest_valid_count &&
                segment_count <= manifest_valid_count,
            "source segment valid_count differs from manifest block");
    if (segment.contains("source_block_missing_count") &&
        !segment.at("source_block_missing_count").is_null())
      require(number(segment, "source_block_missing_count") ==
                  number(manifest_block, "missing_count"),
              "source segment missing_count differs from manifest block");
    const auto& identity = item.at("column_identity");
    for (const auto* key : {"dataset_id", "column", "dtype", "role",
                            "source_column", "table_name"}) {
      if (identity.contains(key) && manifest_block.contains(key))
        require(identity.at(key) == manifest_block.at(key),
                std::string("source manifest block identity mismatch: ") + key);
    }
    verified_source_bytes = checked_add(verified_source_bytes, block_bytes,
                                        "verified source bytes");
    require(verified_source_bytes <= kMaxVerifiedSourceBytesPerSample,
            "cumulative verified source blocks exceed per-sample limit");
    const auto block = read_region(values_path, block_offset, block_bytes);
    require(sha256(block) == expected, "full accepted source block SHA256 mismatch");
    boundaries.push_back(bytes.size());
    bytes.append(block, static_cast<std::size_t>(slice_offset),
                 static_cast<std::size_t>(segment_bytes));
  };
  if (item.contains("source_segments")) {
    require(item.at("source_segments").is_array() &&
                !item.at("source_segments").empty() &&
                item.at("source_segments").size() <= 64,
            "invalid real source_segments");
    for (const auto& segment : item.at("source_segments")) append_segment(segment);
  } else {
    J segment{{"source_block_offset", item.at("source_block_offset")},
              {"source_block_bytes", item.at("source_block_bytes")},
              {"slice_offset_in_source_block_bytes",
               item.at("slice_offset_in_source_block_bytes")},
              {"source_block_sha256", item.at("source_block_sha256")},
              {"source_values_path", item.at("source_values_path")},
              {"source_block_id", item.at("source_block_id")},
              {"source_block_index", item.at("source_block_index")},
              {"source_block_valid_count", item.at("source_block_valid_count")},
              {"element_count", count}};
    if (item.contains("source_block_missing_count"))
      segment["source_block_missing_count"] = item.at("source_block_missing_count");
    append_segment(segment);
  }
  require(bytes.size() == sample_bytes, "real source segment accounting mismatch");
  J record = item;
  record["category"] = "large_real";
  record["case_class"] = "large_real";
  record["sample_bytes"] = sample_bytes;
  record["sample_sha256"] = sha256(bytes);
  record["original_order_preserved"] = true;
  record["full_source_block_verified"] = true;
  record["verified_source_bytes"] = verified_source_bytes;
  record["verified_source_byte_limit"] = kMaxVerifiedSourceBytesPerSample;
  record["source_segment_output_offsets"] = boundaries;
  record["payload"] = bytes;
  return record;
}

J boundary_sample(const J& item) {
  const auto id = text(item, "sample_id");
  valid_sample_id(id);
  auto bytes = boundary_bytes(item);
  J record = item;
  record["category"] = "boundary";
  record["case_class"] = "boundary";
  record["boundary_family"] = item.value("boundary_family", text(item, "fixture"));
  record["sample_bytes"] = bytes.size();
  record["sample_sha256"] = sha256(bytes);
  record["synthetic"] = true;
  record["generator"] = "operator_harness_v2_stress_input_builder_cpp_v3";
  record["payload"] = std::move(bytes);
  return record;
}

J publish(J record, const fs::path& root, const std::string& spec_sha256) {
  const auto id = record.at("sample_id").get<std::string>();
  const auto payload = record.at("payload").get<std::string>();
  record.erase("payload");
  const auto input = fs::absolute(root / "inputs" / (id + ".bin")).lexically_normal();
  require(input.parent_path() == fs::absolute(root / "inputs").lexically_normal(),
          "sample output escapes inputs directory");
  atomic_write(input, payload, 0444);
  record["input_path"] = input.string();
  record["builder_spec_sha256"] = spec_sha256;
  const auto receipt_path = root / "receipts" / (id + ".json");
  J receipt{{"schema", "operator-tree-stress-input-receipt-v3"},
            {"sample", record}};
  receipt["receipt_sha256"] = sha256(receipt.dump());
  atomic_write(receipt_path, receipt.dump(2) + "\n", 0444);
  record["receipt_path"] = fs::absolute(receipt_path).lexically_normal().string();
  record["receipt_sha256"] = receipt.at("receipt_sha256");
  return record;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: operator_harness_v2_stress_input_builder SPEC_JSON OUTPUT_DIR");
    const auto spec_path = fs::canonical(argv[1]);
    const auto output = fs::absolute(argv[2]).lexically_normal();
    const auto raw_spec = read_file(spec_path, kMaxSpecBytes);
    const auto spec = J::parse(raw_spec);
    require(spec.is_object() && spec.value("schema", "") ==
                                    "operator-tree-stress-input-spec-v3",
            "wrong stress input spec schema");
    require(spec.contains("real_slices") && spec.at("real_slices").is_array() &&
                spec.contains("boundary_fixtures") &&
                spec.at("boundary_fixtures").is_array(),
            "stress input spec arrays are missing");
    const auto total = checked_add(spec.at("real_slices").size(),
                                   spec.at("boundary_fixtures").size(),
                                   "input count");
    require(total > 0 && total <= 256, "invalid stress input count");
    if (fs::exists(output) && !fs::is_directory(output))
      throw std::runtime_error("output path is not a directory");
    fs::create_directories(output);
    const auto spec_digest = sha256(raw_spec);
    std::vector<J> samples;
    samples.reserve(static_cast<std::size_t>(total));
    for (const auto& item : spec.at("real_slices"))
      samples.push_back(publish(real_sample(item), output, spec_digest));
    for (const auto& item : spec.at("boundary_fixtures"))
      samples.push_back(publish(boundary_sample(item), output, spec_digest));
    std::sort(samples.begin(), samples.end(), [](const J& a, const J& b) {
      return a.at("sample_id").get_ref<const std::string&>() <
             b.at("sample_id").get_ref<const std::string&>();
    });
    for (std::size_t index = 1; index < samples.size(); ++index)
      require(samples[index - 1].at("sample_id") != samples[index].at("sample_id"),
              "duplicate sample_id");
    J manifest{{"schema", "operator-tree-stress-inputs-v3"},
               {"suite_id", spec.value("suite_id", "operator_tree_stress_v3")},
               {"terminology", "operator_tree"},
               {"builder", "operator_harness_v2_stress_input_builder_cpp_v3"},
               {"builder_spec_path", spec_path.string()},
               {"builder_spec_sha256", spec_digest},
               {"excluded_ids", {"A106", "A107", "A108", "A109", "A110"}},
               {"samples", samples}};
    manifest["sample_count"] = samples.size();
    manifest["manifest_sha256"] = sha256(manifest.dump());
    atomic_write(output / "INPUTS.json", manifest.dump(2) + "\n", 0444);
    std::cout << J{{"status", "ok"},
                   {"sample_count", samples.size()},
                   {"inputs_manifest", fs::absolute(output / "INPUTS.json").string()},
                   {"manifest_sha256", manifest.at("manifest_sha256")}}
                     .dump()
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "STRESS_INPUT_BUILDER_FATAL: " << error.what() << '\n';
    return 1;
  }
}
