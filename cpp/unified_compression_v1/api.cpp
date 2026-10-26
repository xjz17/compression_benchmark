#include "api.hpp"

#include "algorithm_adapter.hpp"
#include "operator_adapter.hpp"
#include "bos_bfe_v2/ucc_bos_bfe_bridge.hpp"
#include "../harness_combo_pilot_v1/runtime.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <utility>

#ifndef UNIFIED_COMPRESSION_OPERATOR_CATALOG
#define UNIFIED_COMPRESSION_OPERATOR_CATALOG ""
#endif

namespace unified_compression_v1 {
namespace {

constexpr std::uint64_t kMaxFrameBytes = 1ULL << 30;
constexpr std::size_t kHeaderBytes = 20;
constexpr std::size_t kDigestBytes = 32;
using Clock = std::chrono::steady_clock;
namespace combo = harness_combo_pilot_v1;

std::string combo_id(std::string_view name) {
  if (name == "combo_v1:zstd3") return "zstd3";
  if (name == "combo_v1:lz4") return "lz4";
  return std::string(name);
}

[[noreturn]] void fail(const char* code, const std::string& message) {
  throw Error(code, message);
}

std::array<std::uint8_t, kDigestBytes> digest(std::span<const std::uint8_t> data) {
  std::array<std::uint8_t, kDigestBytes> result{};
  unsigned size = 0;
  if (EVP_Digest(data.data(), data.size(), result.data(), &size,
                 EVP_sha256(), nullptr) != 1 || size != result.size())
    fail("checksum_error", "SHA-256 failed");
  return result;
}

void put(Bytes& out, std::uint64_t value, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i) {
    out.push_back(static_cast<std::uint8_t>(value & 255));
    value >>= 8;
  }
}

std::uint64_t get(std::span<const std::uint8_t> data, std::size_t start,
                  unsigned bytes) {
  std::uint64_t result = 0;
  for (unsigned i = 0; i < bytes; ++i)
    result |= std::uint64_t(data[start + i]) << (8 * i);
  return result;
}

std::string name_key(std::string_view input) {
  while (!input.empty() && (input.front() == ' ' || input.front() == '\t'))
    input.remove_prefix(1);
  while (!input.empty() && (input.back() == ' ' || input.back() == '\t'))
    input.remove_suffix(1);
  if (input.empty() || input.size() > 512 || input.find('\0') != input.npos)
    fail("unknown_method", "method name is empty or invalid");
  std::string result(input);
  for (auto& c : result) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  return result;
}

Bytes wrap(const MethodInfo& method, Bytes payload) {
  if (method.name.size() > 512 || payload.size() > kMaxFrameBytes)
    fail("size_limit", "compressed frame exceeds the supported limit");
  Bytes result;
  result.reserve(kHeaderBytes + method.name.size() + payload.size() + kDigestBytes);
  result.insert(result.end(), {'U', 'C', 'C', '1'});
  result.push_back(1);
  result.push_back(method.kind == "algorithm" ? 1 :
                   method.kind == "operator" ? 2 :
                   method.kind == "combo" ? 3 : 4);
  put(result, 0, 2);
  put(result, method.name.size(), 4);
  put(result, payload.size(), 8);
  result.insert(result.end(), method.name.begin(), method.name.end());
  result.insert(result.end(), payload.begin(), payload.end());
  const auto checksum = digest(result);
  result.insert(result.end(), checksum.begin(), checksum.end());
  return result;
}

struct Envelope {
  std::string method;
  std::uint8_t kind;
  std::span<const std::uint8_t> payload;
};

Envelope unwrap(std::span<const std::uint8_t> frame) {
  if (frame.size() < kHeaderBytes + kDigestBytes ||
      frame.size() > kMaxFrameBytes + 1024)
    fail("invalid_frame", "invalid UCC1 frame size");
  if (!std::equal(frame.begin(), frame.begin() + 4, "UCC1") || frame[4] != 1 ||
      (frame[5] < 1 || frame[5] > 4) || frame[6] != 0 || frame[7] != 0)
    fail("invalid_frame", "unsupported UCC1 header");
  const auto name_size = get(frame, 8, 4);
  const auto payload_size = get(frame, 12, 8);
  if (name_size == 0 || name_size > 512 || payload_size > kMaxFrameBytes ||
      kHeaderBytes + name_size + payload_size + kDigestBytes != frame.size())
    fail("invalid_frame", "invalid UCC1 frame lengths");
  const auto actual = digest(frame.first(frame.size() - kDigestBytes));
  if (!std::equal(actual.begin(), actual.end(), frame.end() - kDigestBytes))
    fail("invalid_frame", "UCC1 checksum mismatch");
  std::string method(reinterpret_cast<const char*>(frame.data() + kHeaderBytes),
                     static_cast<std::size_t>(name_size));
  if (method.find('\0') != method.npos)
    fail("invalid_frame", "UCC1 method contains NUL");
  return {std::move(method), frame[5],
          frame.subspan(kHeaderBytes + static_cast<std::size_t>(name_size),
                        static_cast<std::size_t>(payload_size))};
}

void admit(const MethodInfo& method) {
  if (method.contract.value("policy_disabled", false))
    fail("method_disabled", method.name + ": " + method.reason);
  if (!method.available)
    fail("method_unavailable", method.name + ": " + method.reason);
  if (!method.enabled) fail("method_disabled", method.name + ": " + method.reason);
}

Column decode_any(std::span<const std::uint8_t> frame,
                  const detail::AlgorithmAdapter* prepared = nullptr) {
  const auto envelope = unwrap(frame);
  Column result;
  if (envelope.kind == 1) {
    const auto& algorithms = [&]() -> const detail::AlgorithmAdapter& {
      if (prepared) return *prepared;
      static const detail::AlgorithmAdapter shared;
      return shared;
    }();
    const auto& methods = algorithms.methods();
    const auto method = std::find_if(methods.begin(), methods.end(),
        [&](const auto& v) { return v.name == envelope.method; });
    if (method == methods.end()) fail("invalid_frame", "unknown algorithm in UCC1 frame");
    admit(*method);
    result = algorithms.decode(envelope.method, envelope.payload);
  } else if (envelope.kind == 2) {
    if (!envelope.method.starts_with("operator:"))
      fail("invalid_frame", "UCC1 route/method mismatch");
    result = detail::OperatorAdapter::decode(envelope.method, envelope.payload);
  } else {
    try {
      combo::Block block;
      if (envelope.kind == 3) {
        if (!envelope.method.starts_with("combo_v1:"))
          fail("invalid_frame", "UCC1 combo route/method mismatch");
        block = combo::decode(envelope.payload, combo_id(envelope.method));
      } else {
        if (!envelope.method.starts_with("tree_v1:"))
          fail("invalid_frame", "UCC1 tree route/method mismatch");
        block = combo::decode_tree(envelope.payload, envelope.method);
      }
      result = {std::move(block.dtype), block.row_count,
                std::move(block.values), std::move(block.missing)};
    } catch (const Error&) { throw; }
      catch (const std::exception& error) { fail("invalid_frame", error.what()); }
  }
  validate_column(result);
  return result;
}

std::uint64_t elapsed(Clock::time_point start, Clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

std::uint64_t median(std::vector<std::uint64_t> times) {
  const auto middle = times.begin() + static_cast<std::ptrdiff_t>(times.size() / 2);
  std::nth_element(times.begin(), middle, times.end());
  return *middle;
}

template <class Encode>
CompressionResult measure(const Column& column, const MethodInfo& method,
                          MeasurementOptions options, Encode encode,
                          const detail::AlgorithmAdapter* algorithms) {
  if (options.warmups > 31 || options.repeats == 0 || options.repeats > 1001)
    fail("invalid_options", "warmups must be 0..31 and repeats must be 1..1001");
  CompressionResult result;
  result.method = method.name;
  result.original_numeric_bytes = column.values.size();
  result.compression_samples_ns.reserve(options.repeats);
  result.decompression_samples_ns.reserve(options.repeats);
  bool all_strict = true, any_fallback = false, any_unknown_fallback = false;
  nlohmann::json audit_samples = nlohmann::json::array();
  std::optional<std::size_t> first_frame_bytes;
  for (std::uint32_t i = 0; i < options.warmups + options.repeats; ++i) {
    const auto encode_start = Clock::now();
    auto inner = encode();
    auto frame = wrap(method, std::move(inner.frame));
    const auto encode_end = Clock::now();
    const auto decode_start = Clock::now();
    auto restored = decode_any(frame, algorithms);
    const auto decode_end = Clock::now();
    if (!(restored == column))
      fail("roundtrip_mismatch", "decoded column differs from original bits/schema");
    if (first_frame_bytes && *first_frame_bytes != frame.size())
      fail("measurement_inconsistent", "repeated encode changed complete frame byte count");
    first_frame_bytes = frame.size();
    const nlohmann::json& iteration_audit = inner.audit;
    all_strict = all_strict && iteration_audit.value("strict", false);
    const auto fallback = iteration_audit.find("fallback_used");
    any_unknown_fallback = any_unknown_fallback || fallback == iteration_audit.end() ||
                           !fallback->is_boolean();
    any_fallback = any_fallback || (fallback != iteration_audit.end() &&
        fallback->is_boolean() && fallback->get<bool>());
    audit_samples.push_back(inner.audit);
    if (i >= options.warmups) {
      result.compression_samples_ns.push_back(elapsed(encode_start, encode_end));
      result.decompression_samples_ns.push_back(elapsed(decode_start, decode_end));
    }
    result.compressed_data = std::move(frame);
    result.execution_audit = std::move(inner.audit);
  }
  result.compressed_bytes = result.compressed_data.size();
  if (result.original_numeric_bytes)
    result.compression_ratio = static_cast<double>(result.compressed_bytes) /
                               static_cast<double>(result.original_numeric_bytes);
  result.compression_time_ns = median(result.compression_samples_ns);
  result.decompression_time_ns = median(result.decompression_samples_ns);
  result.bit_exact = true;
  result.execution_audit["strict"] = all_strict;
  result.execution_audit["fallback_used"] = any_fallback ? nlohmann::json(true) :
      any_unknown_fallback ? nlohmann::json(nullptr) : nlohmann::json(false);
  result.execution_audit["measurement_audits"] = std::move(audit_samples);
  result.execution_audit["all_iterations_strict"] = all_strict;
  return result;
}

template <class T> Column typed_column(std::span<const T> values, const char* dtype) {
  Column result;
  result.dtype = dtype;
  result.row_count = values.size();
  if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
    fail("invalid_input", "numeric column byte count overflows");
  result.values.resize(values.size_bytes());
  result.missing.resize(values.size() / 8 + (values.size() % 8 != 0), 0);
  if constexpr (std::endian::native == std::endian::little) {
    if (!values.empty()) std::memcpy(result.values.data(), values.data(), values.size_bytes());
  } else {
    for (std::size_t i = 0; i < values.size(); ++i) {
      auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(values[i]);
      std::reverse(bytes.begin(), bytes.end());
      std::copy(bytes.begin(), bytes.end(), result.values.begin() + i * sizeof(T));
    }
  }
  return result;
}

}  // namespace

std::uint32_t dtype_width(std::string_view dtype) {
  if (dtype == "int32" || dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "float64") return 8;
  fail("unsupported_dtype", "dtype must be int32, int64, float32 or float64");
}

void validate_column(const Column& column) {
  const auto width = dtype_width(column.dtype);
  const auto bitmap_size = column.row_count / 8 + (column.row_count % 8 != 0);
  if (bitmap_size != column.missing.size())
    fail("invalid_input", "missing bitmap size must be ceil(row_count/8)");
  if (column.row_count % 8 &&
      (column.missing.back() >> (column.row_count % 8)) != 0)
    fail("invalid_input", "missing bitmap padding bits must be zero");
  std::uint64_t nulls = 0;
  for (const auto byte : column.missing) nulls += std::popcount(byte);
  const auto valid = column.row_count - nulls;
  if (valid > std::numeric_limits<std::uint64_t>::max() / width ||
      valid * width != column.values.size())
    fail("invalid_input", "numeric payload does not match dtype, rows and null bitmap");
}

struct Registry::Impl {
  detail::AlgorithmAdapter algorithms;
  detail::OperatorAdapter operators;
  std::vector<MethodInfo> methods;
  std::map<std::string, std::set<std::size_t>> lookup;

  explicit Impl(const RegistryOptions& options)
      : operators(options.operator_catalog_path.empty()
                      ? UNIFIED_COMPRESSION_OPERATOR_CATALOG
                      : options.operator_catalog_path) {
    methods = algorithms.methods();
    methods.insert(methods.end(), operators.methods().begin(), operators.methods().end());
    const auto combo_catalog = combo::catalog();
    for (const auto& entry : combo_catalog.at("candidates")) {
      MethodInfo method;
      const auto id = entry.at("id").get<std::string>();
      method.name = id.starts_with("combo_v1:") ? id : "combo_v1:" + id;
      method.display_name = method.name;
      method.kind = "combo";
      method.dtypes = entry.at("dtypes").get<std::vector<std::string>>();
      method.enabled = method.available = true;
      method.contract = entry;
      method.contract["pilot_candidate_id"] = id;
      method.contract["scope"] = "fixed30; separate from frozen oracle candidates";
      methods.push_back(std::move(method));
    }
    for (std::size_t i = 0; i < methods.size(); ++i) {
      methods[i].contract["unified_input"] = {
          {"column", "original little-endian numeric values with compact nulls"},
          {"dtypes", methods[i].dtypes},
          {"missing", "ceil(row_count/8) bytes, LSB first, set means NULL"},
          {"selection", "one string method name; no implicit search or tuning"}};
      methods[i].contract["unified_output"] = {
          {"binary", "complete UCC1 frame containing the native NGF1 frame"},
          {"reconstruction", "original dtype, row count, numeric bits and null bitmap"},
          {"metrics", {"compression_ratio", "compression_time_ns", "decompression_time_ns"}}};
      lookup[name_key(methods[i].name)].insert(i);
      lookup[name_key(methods[i].display_name)].insert(i);
      for (const auto& alias : methods[i].aliases) {
        lookup[name_key(alias)].insert(i);
        lookup[name_key(methods[i].kind + ":" + alias)].insert(i);
      }
    }
  }
};

Registry::Registry(RegistryOptions options)
    : impl_(std::make_shared<const Impl>(options)) {}

const std::vector<MethodInfo>& Registry::methods() const noexcept { return impl_->methods; }

const MethodInfo& Registry::resolve(std::string_view name) const {
  const auto key = name_key(name);
  const auto found = impl_->lookup.find(key);
  if (found == impl_->lookup.end()) {
    auto token = key;
    if (token.starts_with("algorithm:")) token.erase(0, 10);
    if (token == "nncp" || token == "ppmd" || token == "dmc" ||
        token == "paq8" || token == "cmix" || token == "lstm")
      fail("method_disabled", "explicitly disabled compression family: " + std::string(name));
    fail("unknown_method", "unknown compression method: " + std::string(name));
  }
  if (found->second.size() != 1)
    fail("ambiguous_method", "ambiguous name; use algorithm:LABEL or operator:Axxx: " + std::string(name));
  return impl_->methods[*found->second.begin()];
}

CompressionResult Registry::compress(const Column& column, std::string_view name,
                                     MeasurementOptions options) const {
  if (options.warmups > 31 || options.repeats == 0 || options.repeats > 1001)
    fail("invalid_options", "warmups must be 0..31 and repeats must be 1..1001");
  const auto& method = resolve(name);
  admit(method);
  if (std::find(method.dtypes.begin(), method.dtypes.end(), column.dtype) == method.dtypes.end())
    fail("domain_rejected", method.name + " does not accept source dtype " + column.dtype);
  return measure(column, method, options, [&]() -> combo::Encoded {
    try {
      if (method.kind == "combo")
        return combo::encode(combo_id(method.name),
            {column.dtype, column.row_count, column.values, column.missing});
      nlohmann::json audit;
      if (method.kind == "algorithm") {
        auto frame = impl_->algorithms.encode(method.name, column);
        if (method.name == "algorithm:BOS") {
          const auto info = bos_bfe_bridge::inspect(frame);
          const bool native_raw = info.raw_block_count != 0;
          audit = {{"candidate_id", method.name},
              {"implementation_route", column.dtype == "int64"
                  ? bos_bfe_bridge::kIntegerRouteId : bos_bfe_bridge::kRouteId},
              {"strict", !native_raw}, {"fallback_used", native_raw},
              {"raw_fallback_stage_count", info.raw_block_count},
              {"harness_raw_fallback_stage_count", 0},
              {"native_raw_block_count", info.raw_block_count},
              {"fallback_semantics", native_raw ? "native_bfe_lossless_raw_candidate" : "none"},
              {"native_raw_is_harness_bypass", false},
              {"strict_status", native_raw ? "native_raw_selected" : "strict"},
              {"execution_status", native_raw ? "native_raw_selected" : "strict_ready"},
              {"selected_frontend", info.selected_frontend},
              {"block_count", info.block_count},
              {"bfe_block_count", info.bfe_block_count},
              {"dsc3_block_count", info.dsc3_block_count},
              {"bos2_block_count", info.bos2_block_count},
              {"whole_column_decimal_places", info.whole_column_decimal_places},
              {"full_search_and_framing_in_encode_time", true}};
          return {std::move(frame), std::move(audit)};
        }
        auto adapter_contract = method.contract.value("configuration", std::string()) + " " +
                                method.contract.value("limitations", std::string());
        for (auto& ch : adapter_contract)
          if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
        const bool adapter_may_bypass = adapter_contract.find("fallback") != std::string::npos ||
                                       adapter_contract.find("tagged_raw") != std::string::npos;
        // The legacy codec API does not export a per-chunk adapter-bypass
        // audit. Do not turn mere recovery into proof of strict execution.
        audit = {{"strict", !adapter_may_bypass},
                 {"fallback_used", adapter_may_bypass ? nlohmann::json(nullptr) : nlohmann::json(false)},
                 {"raw_fallback_stage_count", adapter_may_bypass ? nlohmann::json(nullptr) : nlohmann::json(0)},
                 {"candidate_id", method.name},
                 {"strict_status", adapter_may_bypass ? "adapter_bypass_unverified" : "strict"},
                 {"adapter_bypass_may_be_present", adapter_may_bypass}};
        return {std::move(frame), std::move(audit)};
      }
      auto frame = impl_->operators.encode(method.name, column, &audit);
      return {std::move(frame), std::move(audit)};
    } catch (const Error&) { throw; }
      catch (const std::exception& error) { fail("candidate_execution_failed", error.what()); }
  }, &impl_->algorithms);
}

CompressionResult Registry::compress_tree(const Column& column,
    const nlohmann::json& tree, MeasurementOptions options) const {
  MethodInfo method;
  method.kind = "tree";
  try { method.name = combo::tree_identifier(tree); }
  catch (const std::exception& error) { fail("invalid_tree", error.what()); }
  return measure(column, method, options, [&]() -> combo::Encoded {
    try {
      return combo::encode_tree(tree,
          {column.dtype, column.row_count, column.values, column.missing});
    } catch (const Error&) { throw; }
      catch (const std::exception& error) { fail("tree_execution_failed", error.what()); }
  }, &impl_->algorithms);
}

nlohmann::json Registry::tree_contract() const { return combo::tree_contract(); }

Column Registry::decompress(std::span<const std::uint8_t> frame) const {
  return decode_any(frame, &impl_->algorithms);
}

Column make_column(std::span<const std::int32_t> values) { return typed_column(values, "int32"); }
Column make_column(std::span<const std::int64_t> values) { return typed_column(values, "int64"); }
Column make_column(std::span<const float> values) { return typed_column(values, "float32"); }
Column make_column(std::span<const double> values) { return typed_column(values, "float64"); }

CompressionResult compress_column(const Column& column, std::string_view name,
                                  MeasurementOptions options) {
  static const Registry registry;
  return registry.compress(column, name, options);
}
#define UCC_TYPED_OVERLOAD(T) \
  CompressionResult compress_column(std::span<const T> values, std::string_view name, \
                                     MeasurementOptions options) { \
    return compress_column(make_column(values), name, options); \
  }
UCC_TYPED_OVERLOAD(std::int32_t)
UCC_TYPED_OVERLOAD(std::int64_t)
UCC_TYPED_OVERLOAD(float)
UCC_TYPED_OVERLOAD(double)
#undef UCC_TYPED_OVERLOAD

CompressionResult compress_tree(const Column& column, const nlohmann::json& tree,
                                MeasurementOptions options) {
  static const Registry registry;
  return registry.compress_tree(column, tree, options);
}

Column decompress_column(std::span<const std::uint8_t> frame) { return decode_any(frame); }

}  // namespace unified_compression_v1
