#include "baseline.hpp"
#include "column_scales.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
using J = nlohmann::json;
using native_baseline86_v1::Bytes;
using native_baseline86_v1::column_scales::column_key;
using native_baseline86_v1::column_scales::scan;

namespace native_baseline86_v1 {

// The production helper links baseline.cpp.  This focused unit test supplies
// only the same public Block validator contract so it does not link every
// optional compression backend.
void validate(const Block& block) {
  std::uint64_t width = 0;
  if (block.dtype == "float32" || block.dtype == "int32") width = 4;
  else if (block.dtype == "float64" || block.dtype == "int64") width = 8;
  else throw std::invalid_argument("unsupported fixture dtype");
  if (block.missing.size() != (block.row_count + 7) / 8)
    throw std::invalid_argument("fixture bitmap length mismatch");
  if (block.row_count % 8 && !block.missing.empty() &&
      (block.missing.back() & (0xffU << (block.row_count % 8))))
    throw std::invalid_argument("fixture bitmap padding mismatch");
  std::uint64_t missing = 0;
  for (const auto byte : block.missing) missing += std::popcount(byte);
  if (missing > block.row_count ||
      block.values.size() != (block.row_count - missing) * width)
    throw std::invalid_argument("fixture value length mismatch");
}

}  // namespace native_baseline86_v1

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void throws(Function&& function, const char* message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  auto* context = EVP_MD_CTX_new();
  if (!context) throw std::runtime_error("EVP allocation failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned size = 0;
  const auto* data = bytes.empty()
      ? reinterpret_cast<const unsigned char*>("") : bytes.data();
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, data, bytes.size()) == 1 &&
      EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || size != 32) throw std::runtime_error("EVP SHA-256 failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned i = 0; i < size; ++i)
    output << std::setw(2) << static_cast<unsigned>(digest[i]);
  return output.str();
}

void write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("fixture write failed");
}

template <class Float>
Bytes bytes(std::initializer_list<Float> values) {
  using UInt = std::conditional_t<sizeof(Float) == 4, std::uint32_t,
                                  std::uint64_t>;
  Bytes output;
  for (const auto value : values) {
    const auto raw = std::bit_cast<UInt>(value);
    for (std::size_t byte = 0; byte < sizeof(Float); ++byte)
      output.push_back(static_cast<std::uint8_t>(raw >> (byte * 8)));
  }
  return output;
}

J block(const fs::path& root, std::string id, std::string dtype,
        std::string table, std::string source_column, std::string column,
        Bytes values, std::uint64_t rows) {
  const Bytes missing((rows + 7) / 8, 0);
  const auto values_name = id + ".values";
  const auto missing_name = id + ".missing";
  write(root / values_name, values);
  write(root / missing_name, missing);
  return {{"block_id", std::move(id)}, {"dtype", std::move(dtype)},
          {"table_name", std::move(table)}, {"role", "value"},
          {"source_column", std::move(source_column)},
          {"column", std::move(column)}, {"row_count", rows},
          {"original_numeric_bytes", values.size()},
          {"values_path", values_name}, {"missing_bitmap_path", missing_name},
          {"values_sha256", sha256(values)},
          {"missing_bitmap_sha256", sha256(missing)}};
}

}  // namespace

int main() {
  std::uint64_t assertions = 0;
  const auto check = [&](bool condition, const char* message) {
    ++assertions;
    require(condition, message);
  };
  const auto root = fs::temp_directory_path() /
      ("column-scales-test-" + std::to_string(
          static_cast<unsigned long long>(
              std::chrono::steady_clock::now().time_since_epoch().count())));
  fs::create_directories(root);
  try {
    J manifest{{"dataset_id", "fixture"}, {"blocks", J::array()}};
    manifest["blocks"].push_back(block(root, "float64-b2", "float64", "t",
        "temperature", "c0001", bytes<double>({0.1, 1.25}), 2));
    manifest["blocks"].push_back(block(root, "float64-b1", "float64", "t",
        "temperature", "c0001",
        bytes<double>({std::nextafter(1.0, 2.0)}), 1));
    manifest["blocks"].push_back(block(root, "float32-b1", "float32", "t",
        "temperature", "c0001", bytes<float>({0.1F}), 1));
    manifest["blocks"].push_back(block(root, "int64-b1", "int64", "t",
        "count", "c0002", bytes<std::int64_t>({7, 9}), 2));
    const auto manifest_path = root / "manifest.json";
    const auto encoded = manifest.dump(2);
    write(manifest_path, std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(encoded.data()), encoded.size()));

    const std::string cli_sha(64, 'a');
    const auto receipt = scan(manifest_path, 1, 3, cli_sha);
    check(receipt.at("schema") == "column-scales-exact-decimal-v1",
          "receipt schema");
    check(receipt.at("columns").size() == 2,
          "float32 and float64 must be distinct columns");
    check(receipt.at("excluded_non_float_blocks").size() == 1 &&
              receipt.at("excluded_non_float_blocks")[0].at("block_id") ==
                  "int64-b1",
          "integer block exclusion");
    check(receipt.at("source_float_io_bytes_per_precision_pass") == 31 &&
              receipt.at("source_non_float_validation_io_bytes") == 17 &&
              receipt.at("source_precision_passes") == 4 &&
              receipt.at("source_io_bytes") == encoded.size() + 141 &&
              receipt.at("timed_source_io_bytes") == 0,
          "streamed source I/O accounting");
    check(native_baseline86_v1::column_scales::validate(
              manifest_path, receipt, cli_sha) == receipt,
          "validated receipt must be returned unchanged");
    bool saw64 = false;
    bool saw32 = false;
    for (const auto& column : receipt.at("columns")) {
      check(column.at("precision_derivation_ns_samples").size() == 3,
            "timing sample count");
      if (column.at("column_identity").at("dtype") == "float64") {
        saw64 = true;
        check(column.at("decimal_places") == 16, "float64 column scale");
        check(column.at("block_ids") == J::array({"float64-b1", "float64-b2"}),
              "float64 sorted exact block union");
      } else if (column.at("column_identity").at("dtype") == "float32") {
        saw32 = true;
        check(column.at("decimal_places") == 1, "float32 column scale");
      }
    }
    check(saw64 && saw32, "dtype-specific columns missing");

    auto larger_p = receipt;
    for (auto& column : larger_p["columns"])
      if (column["column_identity"]["dtype"] == "float64")
        column["decimal_places"] = 17;
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, larger_p, cli_sha); },
           "validate accepted inflated decimal_places");
    ++assertions;

    auto missing_column = receipt;
    missing_column["columns"].erase(missing_column["columns"].begin());
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, missing_column, cli_sha); },
           "validate accepted omitted column");
    auto duplicate_column = receipt;
    duplicate_column["columns"].push_back(duplicate_column["columns"][0]);
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, duplicate_column, cli_sha); },
           "validate accepted duplicate column");
    auto reordered_blocks = receipt;
    for (auto& column : reordered_blocks["columns"])
      if (column["block_ids"].size() == 2)
        std::swap(column["block_ids"][0], column["block_ids"][1]);
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, reordered_blocks, cli_sha); },
           "validate accepted reordered block union");
    auto negative_sample = receipt;
    negative_sample["columns"][0]["precision_derivation_ns_samples"][0] = -1;
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, negative_sample, cli_sha); },
           "validate accepted negative timing sample");
    auto bad_manifest_sha = receipt;
    bad_manifest_sha["manifest_sha256"] = std::string(64, '0');
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, bad_manifest_sha, cli_sha); },
           "validate accepted changed manifest SHA");
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, receipt, std::string(64, 'b')); },
           "validate accepted changed CLI SHA");
    auto bad_source_io = receipt;
    bad_source_io["source_io_bytes"] =
        bad_source_io["source_io_bytes"].get<std::uint64_t>() + 1;
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, bad_source_io, cli_sha); },
           "validate accepted changed source I/O accounting");
    auto extra_field = receipt;
    extra_field["unbound_note"] = "not permitted";
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, extra_field, cli_sha); },
           "validate accepted unexpected receipt field");
    assertions += 8;

    const auto original_values = bytes<float>({0.1F});
    auto changed_values = original_values;
    changed_values[0] ^= 1;
    write(root / "float32-b1.values", changed_values);
    throws([&] { (void)native_baseline86_v1::column_scales::validate(
                    manifest_path, receipt, cli_sha); },
           "validate accepted changed frozen values");
    write(root / "float32-b1.values", original_values);
    ++assertions;

    auto duplicate_manifest = manifest;
    duplicate_manifest["blocks"][1]["block_id"] = "float64-b2";
    const auto duplicate_path = root / "duplicate.json";
    const auto duplicate_text = duplicate_manifest.dump();
    write(duplicate_path, std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(duplicate_text.data()),
        duplicate_text.size()));
    throws([&] { (void)scan(duplicate_path, 1, 3, cli_sha); },
           "scan accepted duplicate block_id");
    throws([&] { (void)scan(manifest_path, 1, 0, cli_sha); },
           "scan accepted zero repeats");
    assertions += 2;

    J key_manifest{{"dataset_id", "fixture"}, {"role", "value"}};
    J key_row{{"dtype", "float64"}, {"column", "c"}};
    check(J::parse(column_key(key_manifest, key_row)).at("table_name") == "",
          "missing table_name must canonicalize to empty string");
    throws([&] {
      J bad{{"dtype", "float64"}};
      (void)column_key(key_manifest, bad);
    }, "column_key accepted no source_column/column");
    ++assertions;

    fs::remove_all(root);
    std::cout << "{\"schema\":\"column-scales-test-v1\","
              << "\"status\":\"PASS\",\"assertions\":" << assertions
              << ",\"columns\":2,\"float_blocks\":3,"
              << "\"excluded_integer_blocks\":1}\n";
    return 0;
  } catch (...) {
    std::error_code error;
    fs::remove_all(root, error);
    throw;
  }
}
