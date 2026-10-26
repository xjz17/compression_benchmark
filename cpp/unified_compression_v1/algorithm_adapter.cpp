#include "algorithm_adapter.hpp"

#include "../native_baseline86_v1/baseline.hpp"
#include "../native_graph_v1/graph_internal.hpp"
#include "bos_bfe_v2/ucc_bos_bfe_bridge.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <utility>

namespace unified_compression_v1::detail {
namespace {

using BaselineBlock = native_baseline86_v1::Block;

constexpr std::string_view kCanonicalPrefix = "algorithm:";
constexpr std::string_view kInnerActionPrefix = "baseline86/";

std::string normalized_token(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  for (const unsigned char byte : text) {
    if (std::isalnum(byte))
      result.push_back(static_cast<char>(std::tolower(byte)));
  }
  return result;
}

std::optional<std::string> forbidden_reason(std::string_view name) {
  if (name.starts_with(kCanonicalPrefix)) name.remove_prefix(kCanonicalPrefix.size());
  const auto token = normalized_token(name);
  if (token == "glza")
    return "GLZA is retired by the active-method policy";
  if (token == "ppmd" || token == "ppmd8")
    return "PPMd/PPMD8 is disabled by the unified compression policy";
  if (token == "nncp" || token == "dmc" || token == "paq8" ||
      token == "cmix" || token == "lstm")
    return std::string(name) + " belongs to an explicitly disabled family";
  return std::nullopt;
}

bool public_dtype(std::string_view dtype) {
  return dtype == "int32" || dtype == "int64" || dtype == "float32" ||
         dtype == "float64";
}

std::vector<std::string> public_dtypes(const std::vector<std::string>& source) {
  std::vector<std::string> result;
  for (const auto& dtype : source)
    if (public_dtype(dtype)) result.push_back(dtype);
  return result;
}

bool admitted_implementation(const native_baseline86_v1::Record& record) {
  return record.implementation_class == "direct_current" ||
         record.implementation_class == "direct_generic" ||
         record.implementation_class == "direct_exact_subset";
}

std::string disabled_reason(const native_baseline86_v1::Record& record) {
  if (const auto forbidden = forbidden_reason(record.label)) return *forbidden;
  if (!record.linked)
    return record.limitations.empty() ? "in-process backend is unavailable in this build"
                                      : record.limitations;
  if (!record.primary_comparison)
    return "the linked implementation is not admitted as a same-label primary method";
  if (!admitted_implementation(record))
    return "implementation class " + record.implementation_class +
           " cannot be presented as the named algorithm";
  return {};
}

MethodInfo adapt(const native_baseline86_v1::Record& record) {
  MethodInfo result;
  result.name = std::string(kCanonicalPrefix) + record.label;
  result.display_name = record.label;
  result.kind = "algorithm";
  result.aliases = {record.label};
  if (record.label == "Zstd") {
    result.aliases.insert(result.aliases.end(), {"zstd3", "zstd_level_3", "Zstandard"});
  }
  if (record.label == "PPMD8") result.aliases.push_back("PPMd");
  result.dtypes = public_dtypes(record.dtypes);
  if (record.label == "BOS") {
    for (const auto* dtype : {"float32", "float64"})
      if (std::find(result.dtypes.begin(), result.dtypes.end(), dtype) ==
          result.dtypes.end())
        result.dtypes.emplace_back(dtype);
  }
  result.available = record.linked;
  result.reason = disabled_reason(record);
  result.enabled = result.reason.empty();
  result.contract = {
      {"inner_frame", "NGF1"},
      {"profile_version", native_baseline86_v1::profile_version},
      {"implementation", record.implementation},
      {"implementation_class", record.implementation_class},
      {"configuration", record.configuration},
      {"limitations", record.limitations},
      {"linked", record.linked},
      {"policy_disabled", forbidden_reason(record.label).has_value()},
      {"primary_comparison", record.primary_comparison},
      {"shared_or_project", record.shared_or_project},
      {"lossless_contract", "bit_exact_for_admitted_inputs"},
      {"ratio_denominator", "valid_original_numeric_bytes"},
  };
  if (record.label == "BOS") {
    result.contract["legacy_catalog_implementation"] = record.implementation;
    result.contract["legacy_catalog_configuration"] = record.configuration;
    result.contract["implementation"] = "versioned_bos2_bfe_dsc3_hybrid";
    result.contract["profile_version"] = bos_bfe_bridge::kRouteId;
    result.contract["configuration"] = bos_bfe_bridge::kRouteId;
    result.contract["integer_configuration"] =
        "delta_zigzag_split_partition_for_bitpack_v1";
    result.contract["float_configuration"] =
        "min_complete_NGF(DSC3v3+BOS2,BFE1v3+BOS2)";
    result.contract["dtype_configurations"] = {
        {"int64", "delta_zigzag_split_partition_for_bitpack_v1"},
        {"float32", "min_complete_NGF(DSC3v3+BOS2,BFE1v3+BOS2)"},
        {"float64", "min_complete_NGF(DSC3v3+BOS2,BFE1v3+BOS2)"}};
    result.contract["dtype_implementations"] = {
        {"int64", "project_bos2_partitioned"},
        {"float32", "complete-NGF min(DSC3+BOS2,BFE1+BOS2)"},
        {"float64", "complete-NGF min(DSC3+BOS2,BFE1+BOS2)"}};
    result.contract["integer_route_id"] = bos_bfe_bridge::kIntegerRouteId;
    result.contract["float_route_id"] = bos_bfe_bridge::kRouteId;
    result.contract["legacy_integer_decode"] = "project_bos64_adapter";
    result.contract["float_ngf_block_valid_values"] =
        bos_bfe_bridge::kMaximumBlockValues;
    result.contract["float_column_precision"] =
        "exact maximum from bounded scans; shared by every block";
    result.contract["float_selection_boundary"] =
        "complete NGF bytes; strict smaller; tie selects DSC3";
    result.contract["timed_float_boundary"] =
        "whole-column precision scan + both complete NGF candidates + "
        "BHC1/UCC1 construction";
  }
  return result;
}

const MethodInfo* find_method(const std::vector<MethodInfo>& methods,
                              std::string_view canonical_name) {
  const auto it = std::find_if(methods.begin(), methods.end(), [&](const auto& method) {
    return method.name == canonical_name;
  });
  return it == methods.end() ? nullptr : &*it;
}

void ensure_enabled(const MethodInfo& method) {
  if (forbidden_reason(method.display_name))
    throw Error("method_disabled", method.name + ": " + method.reason);
  if (!method.available)
    throw Error("method_unavailable", method.name + ": " + method.reason);
  if (!method.enabled)
    throw Error("method_disabled", method.name + ": " + method.reason);
}

void ensure_dtype(const MethodInfo& method, std::string_view dtype) {
  if (!public_dtype(dtype))
    throw Error("invalid_dtype", "unsupported unified numeric dtype: " +
                                     std::string(dtype));
  if (std::find(method.dtypes.begin(), method.dtypes.end(), dtype) ==
      method.dtypes.end())
    throw Error("input_domain", method.name + " does not admit dtype " +
                                    std::string(dtype));
}

const BaselineBlock& to_baseline(const Column& column) { return column; }

Column from_baseline(BaselineBlock block) {
  return block;
}

struct InnerBinding {
  std::string label;
  std::string dtype;
};

InnerBinding inspect_inner(std::span<const std::uint8_t> inner_frame,
                           bool admit_versioned_bos_route) {
  if (admit_versioned_bos_route && bos_bfe_bridge::is_frame(inner_frame)) {
    try {
      const auto info = bos_bfe_bridge::inspect(inner_frame);
      return {"BOS", info.dtype};
    } catch (const std::exception& error) {
      throw Error("invalid_frame",
                  std::string("invalid versioned BOS frame: ") + error.what());
    }
  }
  native_graph_v1::detail::Frame frame;
  try {
    frame = native_graph_v1::detail::parse_frame(inner_frame, {});
  } catch (const std::exception& error) {
    throw Error("invalid_frame", std::string("cannot parse inner NGF1 frame: ") +
                                     error.what());
  }
  if (frame.policy_registry != native_baseline86_v1::profile_version)
    throw Error("invalid_frame", "inner frame is not an unscaled baseline86 profile");
  if (frame.nodes.size() != 1 ||
      frame.nodes.front().step.kind != native_graph_v1::StepKind::Stage)
    throw Error("invalid_frame", "baseline86 frame must contain exactly one stage");
  const auto& request = frame.nodes.front().step.stage;
  if (!request.action.starts_with(kInnerActionPrefix))
    throw Error("invalid_frame", "inner stage is outside the baseline86 namespace");
  auto label = request.action.substr(kInnerActionPrefix.size());
  if (label.empty() || label.find('/') != std::string::npos)
    throw Error("invalid_frame", "inner baseline86 method label is malformed");

  // Check the serialized fixed profile before native_baseline86_v1::decode()
  // dispatches any inverse kernel.  Its decoder repeats these checks.
  const native_baseline86_v1::Record* record = nullptr;
  try {
    record = &native_baseline86_v1::method(label);
  } catch (const std::exception& error) {
    throw Error("invalid_frame", std::string("unknown inner algorithm: ") +
                                     error.what());
  }
  if (const auto forbidden = forbidden_reason(record->label))
    throw Error("method_disabled", std::string(kCanonicalPrefix) + record->label +
                                       ": " + *forbidden);
  if (!record->linked || !record->primary_comparison ||
      !admitted_implementation(*record))
    throw Error("method_disabled",
                std::string(kCanonicalPrefix) + record->label +
                    ": inner implementation is not an admitted same-label method");
  if (request.profile.empty() ||
      std::find(record->dtypes.begin(), record->dtypes.end(), request.profile) ==
          record->dtypes.end())
    throw Error("invalid_frame", "inner dtype is outside the method contract");
  const auto configuration = request.parameters.find("configuration");
  if (configuration == request.parameters.end() ||
      configuration->second != record->configuration)
    throw Error("invalid_frame", "inner method configuration differs from the catalog");
  return {std::move(label), request.profile};
}

}  // namespace

AlgorithmAdapter::AlgorithmAdapter() {
  const auto& inventory = native_baseline86_v1::inventory();
  if (inventory.size() != 86)
    throw Error("catalog_error", "baseline86 inventory no longer contains 86 labels");
  methods_.reserve(inventory.size());
  for (const auto& record : inventory) methods_.push_back(adapt(record));
}

const std::vector<MethodInfo>& AlgorithmAdapter::methods() const noexcept {
  return methods_;
}

const MethodInfo& AlgorithmAdapter::require_method(
    std::string_view canonical_name) const {
  if (const auto* method = find_method(methods_, canonical_name)) return *method;
  if (const auto forbidden = forbidden_reason(canonical_name))
    throw Error("method_disabled", std::string(canonical_name) + ": " + *forbidden);
  throw Error("unknown_method", "unknown algorithm method: " +
                                    std::string(canonical_name));
}

Bytes AlgorithmAdapter::encode(std::string_view canonical_name,
                               const Column& column) const {
  const auto& method = require_method(canonical_name);
  ensure_enabled(method);
  ensure_dtype(method, column.dtype);
  try {
    auto frame = method.display_name == "BOS"
                     ? bos_bfe_bridge::encode(to_baseline(column))
                     : native_baseline86_v1::encode(method.display_name,
                                                    to_baseline(column));
    const auto inner =
        inspect_inner(frame, method.display_name == "BOS");
    if (inner.label != method.display_name || inner.dtype != column.dtype)
      throw Error("inner_binding_mismatch",
                  "encoded NGF1 method or dtype differs from the requested method");
    return frame;
  } catch (const Error&) {
    throw;
  } catch (const native_baseline86_v1::Unsupported& error) {
    throw Error("input_domain", method.name + ": " + error.what());
  } catch (const std::invalid_argument& error) {
    throw Error("invalid_input", method.name + ": " + error.what());
  } catch (const std::exception& error) {
    throw Error("encode_failed", method.name + ": " + error.what());
  }
}

Column AlgorithmAdapter::decode(
    std::string_view expected_name,
    std::span<const std::uint8_t> inner_frame) const {
  const auto& expected = require_method(expected_name);
  ensure_enabled(expected);
  const auto inner = inspect_inner(
      inner_frame,
      expected.display_name == "BOS");  // Preflight before inverse dispatch.
  if (inner.label != expected.display_name)
    throw Error("inner_binding_mismatch",
                "outer method " + expected.name + " does not match inner algorithm:" +
                    inner.label);
  ensure_dtype(expected, inner.dtype);
  try {
    auto result = from_baseline(
        expected.display_name == "BOS" && bos_bfe_bridge::is_frame(inner_frame)
            ? bos_bfe_bridge::decode(inner_frame)
            : native_baseline86_v1::decode(inner_frame));
    if (result.dtype != inner.dtype)
      throw Error("inner_binding_mismatch",
                  "decoded dtype differs from the preflighted inner frame");
    return result;
  } catch (const Error&) {
    throw;
  } catch (const native_baseline86_v1::Unsupported& error) {
    throw Error("invalid_frame", expected.name + ": " + error.what());
  } catch (const std::exception& error) {
    throw Error("decode_failed", expected.name + ": " + error.what());
  }
}

}  // namespace unified_compression_v1::detail
