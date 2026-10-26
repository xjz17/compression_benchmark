#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "../native_baseline86_v1/baseline.hpp"

namespace unified_compression_v1 {

using Bytes = std::vector<std::uint8_t>;

// Original little-endian numeric bytes, with null values omitted. A set bit in
// missing (LSB first) means NULL. The bitmap always has ceil(row_count/8) bytes.
// Supported source dtypes: int32, int64, float32 and float64. No numeric casts
// are performed by compression or decompression.
using Column = native_baseline86_v1::Block;

struct MethodInfo {
  std::string name;  // algorithm:LABEL, operator:Axxx or combo_v1:RECIPE.
  std::string display_name;
  std::string kind;
  std::vector<std::string> aliases;
  std::vector<std::string> dtypes;
  bool enabled = false;
  bool available = false;
  std::string reason;
  nlohmann::json contract = nlohmann::json::object();
};

class Error : public std::runtime_error {
 public:
  Error(std::string code, std::string message)
      : std::runtime_error(std::move(message)), code_(std::move(code)) {}
  const std::string& code() const noexcept { return code_; }
 private:
  std::string code_;
};

struct MeasurementOptions {
  std::uint32_t warmups = 0;
  std::uint32_t repeats = 1;
};

struct CompressionResult {
  std::string method;
  Bytes compressed_data;  // Complete UCC1 frame; sufficient for cold decode.
  std::uint64_t original_numeric_bytes = 0;
  std::uint64_t compressed_bytes = 0;
  // Complete frame bytes / valid source numeric bytes; lower is better.
  // Empty/all-null input has no numeric denominator and therefore nullopt.
  std::optional<double> compression_ratio;
  std::uint64_t compression_time_ns = 0;
  std::uint64_t decompression_time_ns = 0;
  std::vector<std::uint64_t> compression_samples_ns;
  std::vector<std::uint64_t> decompression_samples_ns;
  bool bit_exact = false;
  // Bit-exact recovery does not imply strict execution: catalog operators may
  // explicitly permit a raw bypass. Selectors must inspect this audit.
  nlohmann::json execution_audit = nlohmann::json::object();
};

struct RegistryOptions {
  // Empty uses the source-tree catalog selected at build time. Only operators
  // need this file; algorithm inventory and decoding are self-contained.
  std::string operator_catalog_path;
};

class Registry {
 public:
  explicit Registry(RegistryOptions options = {});
  const std::vector<MethodInfo>& methods() const noexcept;
  const MethodInfo& resolve(std::string_view name) const;
  CompressionResult compress(const Column&, std::string_view method,
                             MeasurementOptions options = {}) const;
  CompressionResult compress_tree(const Column&, const nlohmann::json& tree,
                                  MeasurementOptions options = {}) const;
  nlohmann::json tree_contract() const;
  Column decompress(std::span<const std::uint8_t> frame) const;
 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

std::uint32_t dtype_width(std::string_view dtype);
void validate_column(const Column&);
Column make_column(std::span<const std::int32_t> values);
Column make_column(std::span<const std::int64_t> values);
Column make_column(std::span<const float> values);
Column make_column(std::span<const double> values);

// Convenience interface: one numeric column + one string -> bytes and costs.
CompressionResult compress_column(const Column&, std::string_view method,
                                 MeasurementOptions options = {});
CompressionResult compress_tree(const Column&, const nlohmann::json& tree,
                                MeasurementOptions options = {});
CompressionResult compress_column(std::span<const double>, std::string_view method,
                                 MeasurementOptions options = {});
CompressionResult compress_column(std::span<const float>, std::string_view method,
                                 MeasurementOptions options = {});
CompressionResult compress_column(std::span<const std::int64_t>, std::string_view method,
                                 MeasurementOptions options = {});
CompressionResult compress_column(std::span<const std::int32_t>, std::string_view method,
                                 MeasurementOptions options = {});
Column decompress_column(std::span<const std::uint8_t> frame);

}  // namespace unified_compression_v1
