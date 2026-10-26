#include "operator102_measure.hpp"
#include "operator102_dispatcher.hpp"
#if defined(OPERATOR102_ENABLE_SPECIAL_BRIDGES)
#include "operator102_special_bridges.hpp"
#endif
#include "../native_baseline86_v1/typed_segments.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <zstd.h>

#include <filesystem>
#include <fstream>
#include <bit>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

using J = nlohmann::json;
using namespace native_graph_v1;
using namespace native_graph_v1::operator102_measure;

namespace {
namespace fs = std::filesystem;

void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument(message);
}

std::string lower(std::string value) {
  for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return value;
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  auto* context = EVP_MD_CTX_new();
  if (!context) throw std::runtime_error("SHA256 context allocation failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned length = 0;
  const auto ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1 &&
                  EVP_DigestFinal_ex(context, digest.data(), &length) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || length != 32) throw std::runtime_error("SHA256 calculation failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned index = 0; index < length; ++index) output << std::setw(2) << unsigned(digest[index]);
  return output.str();
}

Bytes read_file(const fs::path& path, std::uint64_t offset = 0,
                std::uint64_t requested = std::numeric_limits<std::uint64_t>::max()) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(bool(input), "cannot read file: " + path.string());
  const auto end = input.tellg();
  require(end >= 0, "cannot size file: " + path.string());
  const auto size = static_cast<std::uint64_t>(end);
  require(offset <= size, "file offset exceeds size: " + path.string());
  const auto available = size - offset;
  const auto bytes = requested == std::numeric_limits<std::uint64_t>::max() ? available : requested;
  require(bytes <= available && bytes <= (1ULL << 31), "invalid/oversized file slice: " + path.string());
  input.seekg(static_cast<std::streamoff>(offset));
  Bytes output(static_cast<std::size_t>(bytes));
  if (bytes) input.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(bytes));
  require(bool(input) || bytes == 0, "short read: " + path.string());
  return output;
}

fs::path resolve(const fs::path& base, const std::string& value) {
  fs::path path(value);
  return path.is_absolute() ? path : base / path;
}

std::uint32_t dtype_width(const std::string& name) {
  const auto value = lower(name);
  if (value == "float64" || value == "double" || value == "int64" || value == "uint64") return 8;
  if (value == "float32" || value == "float" || value == "int32" || value == "uint32") return 4;
  if (value == "int16" || value == "uint16") return 2;
  if (value == "int8" || value == "uint8" || value == "byte" || value == "bytes") return 1;
  throw std::invalid_argument("unsupported manifest dtype: " + name);
}

Kind decode_kind(const std::string& name) {
  if (name == "serial") return Kind::Serial;
  if (name == "numeric") return Kind::Numeric;
  if (name == "struct") return Kind::Struct;
  throw std::invalid_argument("unknown input kind: " + name);
}

struct Block {
  std::string block_id;
  std::string dtype;
  std::uint32_t width = 0;
  std::uint64_t valid_count = 0;
  Bytes values;
  std::vector<Bytes> auxiliaries;
  EntropySummary value_entropy;
  std::vector<EntropySummary> auxiliary_entropy;
};

void verify_digest(const Bytes& bytes, const J& row, const char* key, const std::string& label) {
  if (!row.contains(key) || row.at(key).is_null() || row.at(key).get<std::string>().empty()) return;
  const auto expected = lower(row.at(key).get<std::string>());
  require(sha256(bytes) == expected, label + " SHA256 mismatch");
}

Block load_block(const J& row, const fs::path& base) {
  Block block;
  block.block_id = row.value("block_id", std::string{});
  block.dtype = row.at("dtype").get<std::string>();
  block.width = row.value("width", dtype_width(block.dtype));
  block.valid_count = row.at("valid_count").get<std::uint64_t>();
  require(row.value("byte_order", std::string("little")) == "little",
          "only little-endian typed payloads are accepted");
  require(row.value("values_are_valid_only", true),
          "values payload must contain valid values only");
  const auto values_path = resolve(base, row.at("values_path").get<std::string>());
  if (row.contains("values_offset") || row.contains("values_length"))
    block.values = native_baseline86_v1::typed_segments::read(values_path, row, "values");
  else
    block.values = read_file(values_path, row.value("values_offset_bytes", std::uint64_t{0}),
                             row.value("values_size_bytes", std::numeric_limits<std::uint64_t>::max()));
  verify_digest(block.values, row, "values_sha256", "values");
  require(block.width != 0 && block.values.size() % block.width == 0,
          "values payload is not dtype aligned");
  require(block.values.size() == block.valid_count * block.width,
          "values payload is not exactly valid_count*dtype_width");

  if (row.contains("missing_bitmap_path") && !row.at("missing_bitmap_path").is_null() &&
      !row.at("missing_bitmap_path").get<std::string>().empty()) {
    require(row.value("missing_bitmap_semantics", std::string{}) ==
                "one_bit_means_missing_lsb_first",
            "unknown missing bitmap semantics");
    const auto bitmap_path = resolve(base, row.at("missing_bitmap_path").get<std::string>());
    auto bitmap = row.contains("missing_bitmap_offset") || row.contains("missing_bitmap_length")
        ? native_baseline86_v1::typed_segments::read(bitmap_path, row, "missing_bitmap")
        : read_file(bitmap_path);
    verify_digest(bitmap, row, "missing_bitmap_sha256", "missing bitmap");
    const auto row_count = row.at("row_count").get<std::uint64_t>();
    const auto missing_count = row.at("missing_count").get<std::uint64_t>();
    require(row_count >= block.valid_count && row_count - block.valid_count == missing_count,
            "row/valid/missing counts disagree");
    require(bitmap.size() == (row_count + 7U) / 8U,
            "missing bitmap length disagrees with row_count");
    std::uint64_t counted = 0;
    for (const auto byte : bitmap) counted += static_cast<unsigned>(std::popcount(byte));
    require(counted == missing_count, "missing bitmap popcount disagrees with missing_count");
    if (!bitmap.empty() && row_count % 8U) {
      const auto used = static_cast<unsigned>(row_count % 8U);
      require((bitmap.back() & static_cast<std::uint8_t>(0xffU << used)) == 0,
              "missing bitmap unused high bits are nonzero");
    }
    block.auxiliaries.push_back(std::move(bitmap));
  } else if (row.contains("missing_count")) {
    require(row.at("missing_count").get<std::uint64_t>() == 0,
            "missing rows require a missing bitmap");
  }
  if (row.contains("auxiliary_paths")) {
    for (const auto& item : row.at("auxiliary_paths")) {
      const auto path = item.is_string() ? item.get<std::string>() : item.at("path").get<std::string>();
      auto auxiliary = read_file(resolve(base, path));
      if (item.is_object()) verify_digest(auxiliary, item, "sha256", "auxiliary");
      block.auxiliaries.push_back(std::move(auxiliary));
    }
  }
#if !defined(OPERATOR102_TIMING_MODE)
  block.value_entropy = entropy_one(block.values);
  block.auxiliary_entropy.reserve(block.auxiliaries.size());
  for (const auto& auxiliary : block.auxiliaries)
    block.auxiliary_entropy.push_back(entropy_one(auxiliary));
#endif
  return block;
}

std::vector<Block> load_blocks(const J& request, J& manifest_identity) {
  J rows;
  fs::path base = fs::current_path();
  if (request.contains("manifest_path")) {
    const auto manifest_path = fs::absolute(request.at("manifest_path").get<std::string>());
    const auto bytes = read_file(manifest_path);
    if (request.contains("expected_manifest_sha256"))
      require(sha256(bytes) == lower(request.at("expected_manifest_sha256").get<std::string>()),
              "manifest SHA256 mismatch");
    auto manifest = J::parse(bytes.begin(), bytes.end());
    require(manifest.contains("blocks") && manifest.at("blocks").is_array(),
            "manifest requires blocks array");
    rows = manifest.at("blocks");
    for (auto& row : rows) {
      for (const auto key : {"byte_order", "values_are_valid_only", "missing_bitmap_semantics"})
        if (!row.contains(key) && manifest.contains(key)) row[key] = manifest.at(key);
    }
    base = manifest_path.parent_path();
    manifest_identity = {{"path", manifest_path.string()}, {"sha256", sha256(bytes)}};
    for (const auto key : {"dataset_id", "column_id", "schema"})
      if (manifest.contains(key)) manifest_identity[key] = manifest.at(key);
  } else {
    require(request.contains("blocks") && request.at("blocks").is_array(),
            "request requires manifest_path or blocks");
    rows = request.at("blocks");
    if (request.contains("base_path")) base = fs::absolute(request.at("base_path").get<std::string>());
    manifest_identity = {{"inline", true}};
  }
  if (request.contains("input_contract")) {
    const auto& contract = request.at("input_contract");
    require(contract.is_object(), "input_contract must be an object");
    for (auto& row : rows)
      for (const auto key : {"byte_order", "values_are_valid_only", "missing_bitmap_semantics"})
        if (!row.contains(key) && contract.contains(key)) row[key] = contract.at(key);
    manifest_identity["input_contract"] = contract;
  }
  require(!rows.empty(), "column contains no blocks");
  std::vector<Block> blocks;
  blocks.reserve(rows.size());
  for (const auto& row : rows) blocks.push_back(load_block(row, base));
  return blocks;
}

std::string parameter_string(const J& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_number_integer()) return std::to_string(value.get<std::int64_t>());
  if (value.is_number_unsigned()) return std::to_string(value.get<std::uint64_t>());
  if (value.is_number_float()) return value.dump();
  throw std::invalid_argument("stage parameter must be scalar");
}

PipelineStep decode_step(const J& row) {
  PipelineStep step;
  step.request.action = row.at("action").get<std::string>();
  step.request.profile = row.value("profile", std::string{});
  if (row.contains("parameters"))
    for (auto item = row.at("parameters").begin(); item != row.at("parameters").end(); ++item)
      step.request.parameters[item.key()] = parameter_string(item.value());
  return step;
}

std::vector<PipelineStep> decode_pipeline(const J& implementation, const char* key) {
  std::vector<PipelineStep> steps;
  if (implementation.contains(key)) {
    require(implementation.at(key).is_array(), std::string(key) + " must be an array");
    for (const auto& row : implementation.at(key)) steps.push_back(decode_step(row));
    return steps;
  }
  if (std::string(key) == "present_pipeline") {
    if (implementation.contains("stage")) return {decode_step(implementation.at("stage"))};
    if (implementation.contains("configuration")) {
      const auto& config = implementation.at("configuration");
      J row = {{"action", config.at("runtime_action")},
               {"profile", config.value("profile", config.value("input_view", std::string{}))},
               {"parameters", J::object()}};
      if (config.contains("parameter_items"))
        for (const auto& item : config.at("parameter_items"))
          row["parameters"][item.at(0).get<std::string>()] = item.at(1);
      return {decode_step(row)};
    }
  }
  return steps;
}

J entropy_json(const EntropySummary& entropy, std::uint64_t valid_count) {
  const auto per = [](double value, std::uint64_t denominator) -> J {
    return denominator ? J(value / static_cast<double>(denominator)) : J(nullptr);
  };
  return {{"bytes", entropy.bytes},
          {"streams", entropy.streams},
          {"h0_bits", entropy.h0_bits},
          {"h0_bits_per_byte", per(entropy.h0_bits, entropy.bytes)},
          {"h0_bits_per_valid_value", per(entropy.h0_bits, valid_count)},
          {"markov3_conditioned_bytes", entropy.markov3_conditioned_bytes},
          {"markov3_conditional_bits", entropy.markov3_conditional_bits},
          {"markov3_conditional_bits_per_conditioned_byte",
           per(entropy.markov3_conditional_bits, entropy.markov3_conditioned_bytes)},
          {"markov3_prefix_h0_bits", entropy.markov3_prefix_h0_bits},
          {"markov3_modeled_bits", entropy.markov3_modeled_bits},
          {"markov3_modeled_bits_per_byte", per(entropy.markov3_modeled_bits, entropy.bytes)},
          {"markov3_modeled_bits_per_valid_value", per(entropy.markov3_modeled_bits, valid_count)}};
}

J entropy_delta(const EntropySummary& after, const EntropySummary& before,
                std::uint64_t valid_count) {
  const auto per = [valid_count](double value) -> J {
    return valid_count ? J(value / static_cast<double>(valid_count)) : J(nullptr);
  };
  return {{"bytes", static_cast<std::int64_t>(after.bytes) - static_cast<std::int64_t>(before.bytes)},
          {"h0_bits", after.h0_bits - before.h0_bits},
          {"h0_bits_per_valid_value", per(after.h0_bits - before.h0_bits)},
          {"markov3_conditional_bits", after.markov3_conditional_bits - before.markov3_conditional_bits},
          {"markov3_modeled_bits", after.markov3_modeled_bits - before.markov3_modeled_bits},
          {"markov3_modeled_bits_per_valid_value",
           per(after.markov3_modeled_bits - before.markov3_modeled_bits)}};
}

void append_u32(Bytes& output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(Bytes& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

Bytes bundle_record_descriptor(std::uint8_t role, Kind kind,
                               std::uint32_t width, std::uint32_t ordinal,
                               std::uint64_t length) {
  Bytes output;
  output.reserve(18);
  output.push_back(role);
  output.push_back(static_cast<std::uint8_t>(kind));
  append_u32(output, width);
  append_u32(output, ordinal);
  append_u64(output, length);
  return output;
}

void append_bundle_record(Bytes& output, std::uint8_t role, Kind kind,
                          std::uint32_t width, std::uint32_t ordinal,
                          std::span<const std::uint8_t> payload) {
  const auto descriptor = bundle_record_descriptor(role, kind, width, ordinal,
                                                   payload.size());
  output.insert(output.end(), descriptor.begin(), descriptor.end());
  output.insert(output.end(), payload.begin(), payload.end());
}

// O2R1 is a deterministic self-describing physical representation.  Empty
// inverse-metadata records are retained because the stage count/ordinal is
// part of what the decoder needs to interpret the bundle.  Every block is
// independently bundled and compressed, so neither the operator entropy nor
// the Zstd sink gets an artificial context across entity/block boundaries.
Bytes representation_bundle(const PipelineMeasurement& measurement,
                            const std::vector<Bytes>& auxiliaries,
                            const Bytes* adapter) {
  const auto record_count = measurement.final_outputs.size() +
                            measurement.inverse_metadata.size() +
                            auxiliaries.size() + (adapter ? 1U : 0U);
  require(record_count <= std::numeric_limits<std::uint32_t>::max(),
          "O2R1 record count overflow");
  Bytes output{'O', '2', 'R', '1'};
  append_u32(output, static_cast<std::uint32_t>(record_count));
  for (std::size_t index = 0; index < measurement.final_outputs.size(); ++index) {
    const auto& stream = measurement.final_outputs[index];
    append_bundle_record(output, 1, stream.kind, stream.width,
                         static_cast<std::uint32_t>(index), stream.data);
  }
  for (std::size_t index = 0; index < measurement.inverse_metadata.size(); ++index)
    append_bundle_record(output, 2, Kind::Serial, 1, static_cast<std::uint32_t>(index),
                         measurement.inverse_metadata[index]);
  for (std::size_t index = 0; index < auxiliaries.size(); ++index)
    append_bundle_record(output, 3, Kind::Serial, 1, static_cast<std::uint32_t>(index),
                         auxiliaries[index]);
  if (adapter)
    append_bundle_record(output, 4, Kind::Struct, 1, 0, *adapter);
  return output;
}

std::uint64_t zstd3_roundtrip_size(std::span<const std::uint8_t> input) {
  Bytes compressed(ZSTD_compressBound(input.size()));
  const auto compressed_size = ZSTD_compress(compressed.data(), compressed.size(),
                                             input.data(), input.size(), 3);
  if (ZSTD_isError(compressed_size))
    throw std::runtime_error(std::string("Zstd-3 sink compression failed: ") +
                             ZSTD_getErrorName(compressed_size));
  compressed.resize(compressed_size);
  Bytes restored(input.size());
  const auto restored_size = ZSTD_decompress(restored.data(), restored.size(),
                                             compressed.data(), compressed.size());
  if (ZSTD_isError(restored_size) || restored_size != input.size() ||
      !std::equal(restored.begin(), restored.end(), input.begin(), input.end()))
    throw std::runtime_error("Zstd-3 sink round-trip mismatch");
  return compressed.size();
}

struct Aggregate {
  std::uint64_t valid_count = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t adapted_input_bytes = 0;
  std::uint64_t unchanged_auxiliary_bytes = 0;
  std::uint64_t adapter_metadata_bytes = 0;
  std::uint64_t present_payload_bytes = 0;
  std::uint64_t present_operator_metadata_bytes = 0;
  std::uint64_t omitted_payload_bytes = 0;
  std::uint64_t omitted_operator_metadata_bytes = 0;
  std::uint64_t present_sink_uncompressed_bytes = 0;
  std::uint64_t present_sink_compressed_bytes = 0;
  std::uint64_t omitted_sink_uncompressed_bytes = 0;
  std::uint64_t omitted_sink_compressed_bytes = 0;
  EntropySummary baseline, focal_before, focal_after, present_final, omitted_final;
  std::string bridge_contract;
  std::uint64_t target_block_count = 0;
  std::uint64_t raw_fallback_block_count = 0;
  std::uint64_t raw_fallback_bytes = 0;
  std::map<std::string, std::uint64_t> raw_fallback_reason_counts;
};

#if defined(OPERATOR102_ENABLE_SPECIAL_BRIDGES)
std::pair<PipelineMeasurement, PipelineMeasurement> measure_special(
    const std::string& implementation_id, std::span<const std::uint8_t> adapted,
    std::string& contract) {
  const auto bridged = native_graph_v1::operator102::run_special_bridge(
      implementation_id, adapted);
  require(bridged.target_after_ports.size() == bridged.output_kinds.size() &&
              bridged.target_after_ports.size() == bridged.output_widths.size(),
          "special bridge output descriptors disagree with ports");
  PipelineMeasurement present;
  PipelineMeasurement omitted;
  const Stream before{decode_kind(bridged.target_before_kind),
                      bridged.target_before_width, bridged.target_before};
  omitted.final_outputs = {before};
  omitted.inverse_metadata = bridged.prefix_inverse_metadata;
  omitted.output_payload_bytes = before.data.size();
  for (const auto& packet : omitted.inverse_metadata)
    omitted.metadata_bytes += packet.size();
  omitted.focal_before = omitted.focal_after = omitted.final_representation =
      representation_entropy(omitted.final_outputs, omitted.inverse_metadata);
  omitted.bitexact = true;

  present.final_outputs.reserve(bridged.target_after_ports.size());
  for (std::size_t index = 0; index < bridged.target_after_ports.size(); ++index) {
    const auto kind = decode_kind(bridged.output_kinds[index]);
    const auto width = bridged.output_widths[index];
    require(width != 0 && bridged.target_after_ports[index].size() % width == 0,
            "special bridge output descriptor is not width aligned");
    present.output_payload_bytes += bridged.target_after_ports[index].size();
    present.final_outputs.push_back({kind, width, bridged.target_after_ports[index]});
  }
  present.inverse_metadata = bridged.prefix_inverse_metadata;
  present.inverse_metadata.push_back(bridged.inverse_metadata);
  for (const auto& packet : present.inverse_metadata)
    present.metadata_bytes += packet.size();
  present.focal_before = omitted.final_representation;
  present.focal_after = present.final_representation =
      representation_entropy(present.final_outputs, present.inverse_metadata);
  present.bitexact = bridged.target_inverse == bridged.target_before &&
                     std::equal(bridged.fully_restored.begin(), bridged.fully_restored.end(),
                                adapted.begin(), adapted.end());
  require(present.bitexact, "special bridge was not bit-exact");
  contract = bridged.contract;
  return {std::move(present), std::move(omitted)};
}
#endif

Bytes adapt_values(const Block& block, std::uint32_t target_width,
                   std::string& operation) {
  require(target_width != 0, "adapter target width is zero");
  if (target_width <= block.width) {
    require(block.width % target_width == 0,
            "raw-bit lane split requires target width to divide source width");
    operation = target_width == block.width ? "identity" : "raw_bit_lane_split";
    return block.values;
  }
  require(target_width % block.width == 0,
          "zero-extension requires target width to be a multiple of source width");
  operation = "per_value_little_endian_zero_extend";
  Bytes adapted(block.valid_count * target_width, 0);
  for (std::size_t index = 0; index < block.valid_count; ++index)
    std::copy_n(block.values.begin() + index * block.width, block.width,
                adapted.begin() + index * target_width);
  // This is also the adapter inverse check: every high byte must be zero and
  // every original-width low lane must reproduce the canonical typed bytes.
  for (std::size_t index = 0; index < block.valid_count; ++index) {
    require(std::equal(block.values.begin() + index * block.width,
                       block.values.begin() + (index + 1) * block.width,
                       adapted.begin() + index * target_width),
            "zero-extension adapter changed original low bytes");
    require(std::all_of(adapted.begin() + index * target_width + block.width,
                        adapted.begin() + (index + 1) * target_width,
                        [](std::uint8_t byte) { return byte == 0; }),
            "zero-extension adapter high bytes are not zero");
  }
  return adapted;
}

void charge_common_representation(Aggregate& total, std::span<const std::uint8_t> bytes,
                                  bool adapter, std::uint32_t ordinal,
                                  const EntropySummary* cached = nullptr) {
  EntropySummary entropy = cached ? *cached : entropy_one(bytes);
  entropy += entropy_one(bundle_record_descriptor(
      adapter ? 4 : 3, adapter ? Kind::Struct : Kind::Serial, 1, ordinal,
      bytes.size()));
  if (adapter) total.adapter_metadata_bytes += bytes.size();
  else total.unchanged_auxiliary_bytes += bytes.size();
  total.baseline += entropy;
  total.focal_before += entropy;
  total.focal_after += entropy;
  total.present_final += entropy;
  total.omitted_final += entropy;
}

void charge_auxiliary(Aggregate& total, const Block& block) {
  for (std::size_t index = 0; index < block.auxiliaries.size(); ++index) {
    charge_common_representation(total, block.auxiliaries[index], false,
                                 static_cast<std::uint32_t>(index),
                                 &block.auxiliary_entropy.at(index));
  }
}

Bytes adapter_metadata(const Block& block, const J& implementation,
                       std::uint32_t target_width, const std::string& operation) {
  require(block.dtype.size() <= 255, "dtype name is too long for adapter metadata");
  const auto adapter = implementation.at("input_adapter_kind").get<std::string>();
  require(adapter.size() <= 65535, "adapter kind is too long for adapter metadata");
  Bytes bytes{'A', 'D', 'P', '2', static_cast<std::uint8_t>(block.dtype.size())};
  bytes.insert(bytes.end(), block.dtype.begin(), block.dtype.end());
  bytes.push_back(static_cast<std::uint8_t>(block.width));
  bytes.push_back(static_cast<std::uint8_t>(target_width));
  for (unsigned shift = 0; shift < 64; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(block.valid_count >> shift));
  bytes.push_back(static_cast<std::uint8_t>(adapter.size()));
  bytes.push_back(static_cast<std::uint8_t>(adapter.size() >> 8U));
  bytes.insert(bytes.end(), adapter.begin(), adapter.end());
  require(operation.size() <= 255, "adapter operation is too long");
  bytes.push_back(static_cast<std::uint8_t>(operation.size()));
  bytes.insert(bytes.end(), operation.begin(), operation.end());
  return bytes;
}

J measure_implementation(const StageDispatcher& dispatcher,
                         const J& implementation, const std::vector<Block>& blocks) {
  J result = {{"implementation_id", implementation.at("implementation_id")},
              {"map_key", implementation.value("map_key", std::string{})}};
  if (!implementation.value("applicable", true)) {
    result["status"] = "not_applicable";
    result["reason"] = implementation.value("reason", "mapper-declared not applicable");
    return result;
  }
  try {
    const auto present = decode_pipeline(implementation, "present_pipeline");
    const auto omitted = decode_pipeline(implementation, "omitted_pipeline");
    require(!present.empty(), "present pipeline is empty");
    const auto focal = implementation.value("focal_step_index", present.size() - 1);
    require(focal < present.size(), "focal step index exceeds present pipeline");
    Aggregate total;
    for (const auto& block : blocks) {
      const auto kind = decode_kind(implementation.value("initial_kind",
          block.dtype == "bytes" ? std::string("serial") : std::string("numeric")));
      const auto width = implementation.value("initial_width", kind == Kind::Serial ? 1U : block.width);
      std::string adapter_operation;
      auto adapted_values = adapt_values(block, width, adapter_operation);
      require(adapted_values.size() % width == 0, "adapted input is not width aligned");
      Stream input{kind, width, std::move(adapted_values)};
      total.valid_count += block.valid_count;
      total.input_bytes += block.values.size();
      total.adapted_input_bytes += input.data.size();
      const auto adapted_entropy = adapter_operation != "per_value_little_endian_zero_extend"
          ? block.value_entropy : entropy_one(input.data);
      total.baseline += adapted_entropy;
      total.baseline += entropy_one(bundle_record_descriptor(
          1, input.kind, input.width, 0, input.data.size()));

      PipelineMeasurement measured;
      PipelineMeasurement control;
#if defined(OPERATOR102_ENABLE_SPECIAL_BRIDGES)
      const auto implementation_id = implementation.at("implementation_id").get<std::string>();
      if (native_graph_v1::operator102::supports_special_bridge(implementation_id)) {
        auto pair = measure_special(implementation_id, input.data, total.bridge_contract);
        measured = std::move(pair.first);
        control = std::move(pair.second);
      } else
#endif
      {
        measured = measure_pipeline(dispatcher, input, present, focal, &adapted_entropy);
        control = measure_pipeline(dispatcher, input, omitted,
                                   omitted.empty() ? 0 : omitted.size() - 1,
                                   &adapted_entropy);
      }
      require(measured.bitexact && control.bitexact, "bit-exact check failed");
      if (implementation.contains("domain_fallback_contract")) {
        ++total.target_block_count;
        require(focal < measured.inverse_metadata.size(),
                "lossless-domain wrapper focal metadata is absent");
        const auto fallback = inspect_exact_raw_fallback_metadata(
            measured.inverse_metadata[focal]);
        require(fallback.wrapper,
                "lossless-domain contract row did not emit the charged LFW1 selector");
        if (fallback.used_raw_fallback) {
          ++total.raw_fallback_block_count;
          total.raw_fallback_bytes += fallback.original_bytes;
          ++total.raw_fallback_reason_counts[fallback.reason];
        }
      }
      total.present_payload_bytes += measured.output_payload_bytes;
      total.present_operator_metadata_bytes += measured.metadata_bytes;
      total.omitted_payload_bytes += control.output_payload_bytes;
      total.omitted_operator_metadata_bytes += control.metadata_bytes;
      total.focal_before += measured.focal_before;
      total.focal_after += measured.focal_after;
      total.present_final += measured.final_representation;
      total.omitted_final += control.final_representation;

      Bytes adapter_packet;
      const Bytes* adapter_packet_ptr = nullptr;
      if (implementation.value("charge_input_adapter_metadata", false)) {
        adapter_packet = adapter_metadata(block, implementation, width, adapter_operation);
        adapter_packet_ptr = &adapter_packet;
      }
      const auto present_bundle = representation_bundle(measured, block.auxiliaries,
                                                        adapter_packet_ptr);
      const auto omitted_bundle = representation_bundle(control, block.auxiliaries,
                                                        adapter_packet_ptr);
      total.present_sink_uncompressed_bytes += present_bundle.size();
      total.present_sink_compressed_bytes += zstd3_roundtrip_size(present_bundle);
      total.omitted_sink_uncompressed_bytes += omitted_bundle.size();
      total.omitted_sink_compressed_bytes += zstd3_roundtrip_size(omitted_bundle);
      charge_auxiliary(total, block);
      if (adapter_packet_ptr)
        charge_common_representation(total, adapter_packet, true, 0);
    }
    const auto ratio = [](std::uint64_t numerator, std::uint64_t denominator) -> J {
      return denominator ? J(static_cast<double>(numerator) / denominator) : J(nullptr);
    };
    result.update({{"status", "ok"}, {"bitexact", true}, {"blocks", blocks.size()},
                   {"present_pipeline_steps", present.size()},
                   {"omitted_pipeline_steps", omitted.size()},
                   {"focal_step_index", focal},
                   {"valid_count", total.valid_count}, {"input_numeric_bytes", total.input_bytes},
                   {"adapted_input_bytes", total.adapted_input_bytes},
                   {"unchanged_auxiliary_bytes", total.unchanged_auxiliary_bytes},
                   {"input_adapter_metadata_bytes", total.adapter_metadata_bytes},
                   {"input_adapter_metadata_contract", implementation.value(
                        "charge_input_adapter_metadata", false)
                        ? "ADP2 + uint8 dtype-name length + dtype UTF-8 + uint8 original width + uint8 target width + uint64-le original valid_count + uint16-le adapter-kind length + adapter-kind UTF-8 + uint8 operation length + operation UTF-8, one independent packet per block"
                        : "none"},
                   {"baseline", entropy_json(total.baseline, total.valid_count)},
                   {"focal_before", entropy_json(total.focal_before, total.valid_count)},
                   {"focal_after", entropy_json(total.focal_after, total.valid_count)},
                   {"focal_delta", entropy_delta(total.focal_after, total.focal_before,
                                                  total.valid_count)},
                   {"operator_present", entropy_json(total.present_final, total.valid_count)},
                   {"operator_omitted", entropy_json(total.omitted_final, total.valid_count)},
                   {"ablation_delta", entropy_delta(total.present_final, total.omitted_final,
                                                     total.valid_count)},
                   {"present_payload_bytes", total.present_payload_bytes},
                   {"present_operator_metadata_bytes", total.present_operator_metadata_bytes},
                   {"omitted_payload_bytes", total.omitted_payload_bytes},
                   {"omitted_operator_metadata_bytes", total.omitted_operator_metadata_bytes},
                   {"present_to_omitted_representation_ratio",
                    ratio(total.present_final.bytes, total.omitted_final.bytes)},
                   {"ablation_sink",
                    {{"codec", "zstd"}, {"level", 3}, {"threads", 1},
                     {"container_contract",
                      "O2R1 per-block deterministic bundle: header/count plus role, kind, width, ordinal, length and bytes for every physical output, every inverse-metadata packet (including empty descriptors), unchanged auxiliary, and charged adapter packet; no cross-block Zstd context"},
                     {"operator_present",
                      {{"uncompressed_bytes", total.present_sink_uncompressed_bytes},
                       {"compressed_bytes", total.present_sink_compressed_bytes},
                       {"compressed_bytes_per_valid_value",
                        ratio(total.present_sink_compressed_bytes, total.valid_count)},
                       {"bitexact", true}}},
                     {"operator_omitted",
                      {{"uncompressed_bytes", total.omitted_sink_uncompressed_bytes},
                       {"compressed_bytes", total.omitted_sink_compressed_bytes},
                       {"compressed_bytes_per_valid_value",
                        ratio(total.omitted_sink_compressed_bytes, total.valid_count)},
                       {"bitexact", true}}},
                     {"compressed_byte_delta",
                      static_cast<std::int64_t>(total.present_sink_compressed_bytes) -
                          static_cast<std::int64_t>(total.omitted_sink_compressed_bytes)},
                     {"compressed_bits_per_valid_value_delta",
                      total.valid_count
                          ? J(8.0 * (static_cast<double>(total.present_sink_compressed_bytes) -
                                     static_cast<double>(total.omitted_sink_compressed_bytes)) /
                              static_cast<double>(total.valid_count))
                          : J(nullptr)},
                     {"bitexact", true}}},
                   {"entropy_contract",
                    "each entity block, physical output port, unchanged auxiliary, inverse-metadata packet, and its role/kind/width/ordinal/length descriptor is an independent stream; no cross-boundary contexts"},
                   {"markov3_contract",
                    "empirical H(X_t|X_{t-3:t-1}); first min(3,n) bytes per stream are separately charged by prefix H0 in markov3_modeled_bits"}});
    if (!total.bridge_contract.empty())
      result["special_bridge_contract"] = total.bridge_contract;
    if (implementation.contains("domain_fallback_contract")) {
      result["domain_fallback_contract"] = implementation.at("domain_fallback_contract");
      result["target_block_count"] = total.target_block_count;
      result["raw_fallback_block_count"] = total.raw_fallback_block_count;
      result["strict_target_block_count"] =
          total.target_block_count - total.raw_fallback_block_count;
      result["raw_fallback_bytes"] = total.raw_fallback_bytes;
      result["raw_fallback_reason_counts"] = total.raw_fallback_reason_counts;
    }
  } catch (const std::exception& error) {
    result["status"] = "failed";
    result["bitexact"] = false;
    result["error"] = {{"class", "native_measurement_error"}, {"message", error.what()}};
  }
  return result;
}

#if defined(OPERATOR102_TIMING_MODE)
constexpr std::size_t kTimingWarmups = 1;
constexpr std::size_t kTimingRepeats = 3;

std::uint64_t median3(std::array<std::uint64_t, kTimingRepeats> values) {
  std::sort(values.begin(), values.end());
  return values[1];
}

J time_implementation(const StageDispatcher& dispatcher,
                      const J& implementation,
                      const std::vector<Block>& blocks) {
  J result = {{"implementation_id", implementation.at("implementation_id")},
              {"map_key", implementation.value("map_key", std::string{})}};
  if (!implementation.value("applicable", true)) {
    result["status"] = "not_applicable";
    result["reason"] = implementation.value(
        "reason", "mapper-declared not applicable");
    return result;
  }
  try {
    const auto present = decode_pipeline(implementation, "present_pipeline");
    require(!present.empty(), "present pipeline is empty");
    std::array<std::uint64_t, kTimingRepeats> encode_ns{};
    std::array<std::uint64_t, kTimingRepeats> decode_ns{};
    std::array<std::uint64_t, kTimingRepeats> full_encode_ns{};
    std::array<std::uint64_t, kTimingRepeats> full_decode_ns{};
    std::uint64_t valid_count = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t adapted_input_bytes = 0;
    std::uint64_t bitexact_checks = 0;
    const auto implementation_id =
        implementation.at("implementation_id").get<std::string>();
    bool special_bridge = false;
    for (const auto& block : blocks) {
      const auto kind = decode_kind(implementation.value(
          "initial_kind", block.dtype == "bytes" ? std::string("serial")
                                                   : std::string("numeric")));
      const auto width = implementation.value(
          "initial_width", kind == Kind::Serial ? 1U : block.width);
      std::string adapter_operation;
      auto adapted_values = adapt_values(block, width, adapter_operation);
      require(adapted_values.size() % width == 0,
              "adapted input is not width aligned");
      Stream input{kind, width, std::move(adapted_values)};
      valid_count += block.valid_count;
      input_bytes += block.values.size();
      adapted_input_bytes += input.data.size();
      Bytes adapter_packet;
      const Bytes* adapter_packet_ptr = nullptr;
      if (implementation.value("charge_input_adapter_metadata", false)) {
        adapter_packet = adapter_metadata(block, implementation, width,
                                          adapter_operation);
        adapter_packet_ptr = &adapter_packet;
      }

#if defined(OPERATOR102_ENABLE_SPECIAL_BRIDGES)
      if (native_graph_v1::operator102::supports_special_bridge(
              implementation_id)) {
        special_bridge = true;
        const auto sample = native_graph_v1::operator102::benchmark_special_bridge(
            implementation_id, input.data, block.auxiliaries, adapter_packet_ptr,
            kTimingWarmups, kTimingRepeats);
        require(sample.bitexact && sample.encode_ns.size() == kTimingRepeats &&
                    sample.decode_ns.size() == kTimingRepeats &&
                    sample.full_encode_ns.size() == kTimingRepeats &&
                    sample.full_decode_ns.size() == kTimingRepeats,
                "special bridge timing did not return all bit-exact repeats");
        for (std::size_t repeat = 0; repeat < kTimingRepeats; ++repeat) {
          encode_ns[repeat] += sample.encode_ns[repeat];
          decode_ns[repeat] += sample.decode_ns[repeat];
          full_encode_ns[repeat] += sample.full_encode_ns[repeat];
          full_decode_ns[repeat] += sample.full_decode_ns[repeat];
        }
      } else
#endif
      {
        const auto sample = benchmark_pipeline(
            dispatcher, input, present, block.auxiliaries, adapter_packet_ptr,
            kTimingWarmups, kTimingRepeats);
        require(sample.bitexact && sample.encode_ns.size() == kTimingRepeats &&
                    sample.decode_ns.size() == kTimingRepeats &&
                    sample.full_encode_ns.size() == kTimingRepeats &&
                    sample.full_decode_ns.size() == kTimingRepeats,
                "pipeline timing did not return all bit-exact repeats");
        for (std::size_t repeat = 0; repeat < kTimingRepeats; ++repeat) {
          encode_ns[repeat] += sample.encode_ns[repeat];
          decode_ns[repeat] += sample.decode_ns[repeat];
          full_encode_ns[repeat] += sample.full_encode_ns[repeat];
          full_decode_ns[repeat] += sample.full_decode_ns[repeat];
        }
      }
      bitexact_checks += kTimingWarmups + kTimingRepeats;
    }
    const auto encode_median = median3(encode_ns);
    const auto decode_median = median3(decode_ns);
    const auto full_encode_median = median3(full_encode_ns);
    const auto full_decode_median = median3(full_decode_ns);
    const auto per_valid = [valid_count](std::uint64_t ns) -> J {
      return valid_count
          ? J(static_cast<double>(ns) / static_cast<double>(valid_count))
          : J(nullptr);
    };
    result.update({
        {"status", "ok"},
        {"bitexact", true},
        {"blocks", blocks.size()},
        {"valid_count", valid_count},
        {"input_numeric_bytes", input_bytes},
        {"adapted_input_bytes", adapted_input_bytes},
        {"present_pipeline_steps", present.size()},
        {"special_bridge", special_bridge},
        {"timing",
         {{"clock", "std::chrono::steady_clock"},
          {"warmups_per_block", kTimingWarmups},
          {"repeats", kTimingRepeats},
          {"threads", 1},
          {"ranking_view", "full_o2r1"},
          {"kernel_encode_ns", encode_ns},
          {"kernel_decode_ns", decode_ns},
          {"kernel_encode_median_ns", encode_median},
          {"kernel_decode_median_ns", decode_median},
          {"kernel_encode_median_ns_per_valid_value", per_valid(encode_median)},
          {"kernel_decode_median_ns_per_valid_value", per_valid(decode_median)},
          {"full_encode_ns", full_encode_ns},
          {"full_decode_ns", full_decode_ns},
          {"full_encode_median_ns", full_encode_median},
          {"full_decode_median_ns", full_decode_median},
          {"full_encode_median_ns_per_valid_value", per_valid(full_encode_median)},
          {"full_decode_median_ns_per_valid_value", per_valid(full_decode_median)},
          // Backward-compatible aliases are explicitly kernel-only and are
          // never used by the timing finalizer for formal ranking.
          {"encode_ns", encode_ns},
          {"decode_ns", decode_ns},
          {"encode_median_ns", encode_median},
          {"decode_median_ns", decode_median},
          {"encode_median_ns_per_valid_value", per_valid(encode_median)},
          {"decode_median_ns_per_valid_value", per_valid(decode_median)},
          {"bitexact_checks", bitexact_checks},
          {"bitexact_each_warmup_and_repeat", true},
          {"sample_aggregation",
           "for each repeat index, sum entity-block nanoseconds; report median of the three column totals"},
          {"boundary",
           "full_encode includes every present-pipeline forward stage plus exact O2R1 packing; full_decode includes exact O2R1 parsing plus inverse stages; kernel_* excludes only O2R1 pack/parse; special bridges use the same full boundary on their real prefix+focal and focal+prefix paths"},
          {"excluded",
           "manifest/file I/O, JSON, process startup, catalog lookup, input adapter construction, input copy before clock, and post-clock bit-exact comparison"}}}});
  } catch (const std::exception& error) {
    // This is the sole active97 support-boundary exception.  A004's frozen
    // int64 Camel implementation first converts each integer to float64 and
    // its public lossless contract rejects values that cannot round-trip
    // exactly.  Preserve the native rejection verbatim, but do not turn any
    // other A004 error (or any other implementation error) into N/A.
    const auto implementation_id =
        implementation.at("implementation_id").get<std::string>();
    if (implementation_id == "A004" &&
        std::string_view(error.what()) ==
            "Camel integer is not exactly representable as float64") {
      result["status"] = "not_applicable";
      result["bitexact"] = false;
      result["rankable"] = false;
      result["reason"] =
          "unsupported_input_domain: Camel int64-to-float64 exactness precondition is false";
      result["native_domain_rejection"] = {
          {"class", "native_timing_error"}, {"message", error.what()}};
      return result;
    }
    result["status"] = "failed";
    result["bitexact"] = false;
    result["error"] = {{"class", "native_timing_error"},
                       {"message", error.what()}};
  }
  return result;
}
#endif

J process(const J& request) {
#if defined(OPERATOR102_TIMING_MODE)
  require(request.value("schema", std::string{}) ==
              "operator102-timing-column-request-v1",
          "wrong timing request schema");
#else
  require(request.value("schema", std::string{}) == "operator102-measure-column-request-v1",
          "wrong request schema");
#endif
  require(request.contains("implementations") && request.at("implementations").is_array(),
          "request implementations must be an array");
  J manifest_identity;
  const auto blocks = load_blocks(request, manifest_identity);
  Operator102Dispatcher dispatcher;
  J response = {{"schema",
#if defined(OPERATOR102_TIMING_MODE)
                 "operator102-timing-column-response-v1"},
#else
                 "operator102-measure-column-response-v1"},
#endif
                {"request_id", request.value("request_id", std::string{})},
                {"dataset_id", request.at("dataset_id")},
                {"column_id", request.at("column_id")},
                {"manifest", std::move(manifest_identity)},
                {"dispatcher", {{"identity", dispatcher.identity()}}},
                {"results", J::array()}};
  if (request.contains("provenance")) response["provenance"] = request.at("provenance");
  for (const auto& implementation : request.at("implementations"))
#if defined(OPERATOR102_TIMING_MODE)
    response["results"].push_back(time_implementation(dispatcher, implementation, blocks));
#else
    response["results"].push_back(measure_implementation(dispatcher, implementation, blocks));
#endif
  return response;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 1 && !(argc == 2 && std::string(argv[1]) == "--jsonl")) {
#if defined(OPERATOR102_TIMING_MODE)
    std::cerr << "usage: operator102_timing_cli [--jsonl] < requests.jsonl > responses.jsonl\n";
#else
    std::cerr << "usage: operator102_measure_cli [--jsonl] < requests.jsonl > responses.jsonl\n";
#endif
    return 64;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    try {
      std::cout << process(J::parse(line)).dump() << '\n' << std::flush;
    } catch (const std::exception& error) {
      J response = {{"schema",
#if defined(OPERATOR102_TIMING_MODE)
                     "operator102-timing-column-response-v1"},
#else
                     "operator102-measure-column-response-v1"},
#endif
                    {"status", "request_failed"},
                    {"error", {{"class", "request_error"}, {"message", error.what()}}}};
      try {
        const auto request = J::parse(line);
        if (request.contains("request_id")) response["request_id"] = request.at("request_id");
        if (request.contains("dataset_id")) response["dataset_id"] = request.at("dataset_id");
        if (request.contains("column_id")) response["column_id"] = request.at("column_id");
      } catch (...) {}
      std::cout << response.dump() << '\n' << std::flush;
    }
  }
  return 0;
}
