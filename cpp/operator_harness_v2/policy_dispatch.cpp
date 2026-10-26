#include "policy_dispatch.hpp"

#include "a048_typed_planes.hpp"
#include "special_stages.hpp"

#include <algorithm>
#include <array>
#include <cctype>

namespace operator_harness_v2 {
namespace {

std::string lower(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return result;
}

std::optional<std::string_view> strict_input_domain_code(
    const native_graph_v1::StageRequest& request, std::string_view message) {
  if (request.action == "op.eraser.camel.cpp_v1") {
    if (message == "Camel integer is not exactly representable as float64")
      return "CAMEL_INT64_NOT_EXACTLY_REPRESENTABLE";
    if (message == "Camel decimal path requires finite input")
      return "CAMEL_NONFINITE_INPUT";
    if (message == "Camel decimal scale did not converge")
      return "CAMEL_DECIMAL_SCALE_UNREPRESENTABLE";
  }
  if (request.action == "op.xor.fpc.cpp_v1" &&
      message == "FPC requires non-empty 64-bit-word bytes")
    return "FPC_EMPTY_INPUT";
  if (request.action == "op.xor.gorilla.cpp_v1" &&
      message == "Gorilla input contains reserved end-marker bit pattern")
    return "GORILLA_RESERVED_MARKER_INPUT";
  return std::nullopt;
}

}  // namespace

bool excluded_implementation_id(std::string_view implementation_id) {
  static constexpr std::array<std::string_view, 5> denied{
      "A106", "A107", "A108", "A109", "A110"};
  return std::find(denied.begin(), denied.end(), implementation_id) != denied.end();
}

std::optional<std::string> excluded_action_reason(std::string_view action) {
  const auto name = lower(action);
  if (name.find("nncp") != std::string::npos ||
      name.find("lstm") != std::string::npos)
    return "NNCP_LSTM_DISABLED_BY_HARNESS";
  if (name.find(".ppmd.") != std::string::npos ||
      name.find(".ppm.") != std::string::npos)
    return "PPMD_DISABLED_BY_HARNESS";
  if (name.find(".dmc.") != std::string::npos)
    return "DMC_DISABLED_BY_HARNESS";
  if (name.find(".paq.") != std::string::npos ||
      name.find("paq8") != std::string::npos)
    return "PAQ8_DISABLED_BY_HARNESS";
  if (name.find(".cmix.") != std::string::npos)
    return "CMIX_DISABLED_BY_HARNESS";
  return std::nullopt;
}

PolicyDispatcher::PolicyDispatcher(bool audit_fallbacks)
    : audit_fallbacks_(audit_fallbacks) {}

std::string PolicyDispatcher::identity() const {
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  constexpr std::string_view closure = "support-deps-on";
#else
  constexpr std::string_view closure = "support-deps-off";
#endif
  // The closure is part of the on-wire dispatcher identity.  A frame produced
  // by the residual build must fail closed if decoded by the ordinary build.
  return "operator_harness_v2.policy_dispatch.no-a106-a110.v2.special-v3.a048-pure-v1." +
         std::string(closure) + "+" + inner_.identity();
}

void PolicyDispatcher::enforce(const native_graph_v1::StageRequest& request) const {
  if (const auto reason = excluded_action_reason(request.action))
    throw ExcludedPolicyError(*reason + ": " + request.action);
}

void PolicyDispatcher::audit(std::span<const std::uint8_t> metadata) const {
  if (!audit_fallbacks_) return;
  const auto info =
      native_graph_v1::operator102_measure::inspect_exact_raw_fallback_metadata(metadata);
  if (!info.wrapper) return;
  ++fallback_audit_.wrapped_stage_count;
  if (!info.used_raw_fallback) return;
  ++fallback_audit_.raw_fallback_stage_count;
  if (!info.reason.empty()) ++fallback_audit_.reason_counts[info.reason];
  if (!info.reason.empty() &&
      std::find(fallback_audit_.reasons.begin(), fallback_audit_.reasons.end(),
                info.reason) == fallback_audit_.reasons.end())
    fallback_audit_.reasons.push_back(info.reason);
}

native_graph_v1::StageResult PolicyDispatcher::forward(
    const native_graph_v1::StageRequest& request,
    const std::vector<native_graph_v1::Stream>& inputs) const {
  enforce(request);
  try {
    native_graph_v1::StageResult result;
    if (handles_a048_typed_planes(request))
      result = forward_a048_typed_planes(request, inputs);
    else if (handles_special(request))
      result = forward_special(request, inputs);
    else
      result = inner_.forward(request, inputs);
    audit(result.metadata);
    return result;
  } catch (const std::exception& error) {
    if (const auto code = strict_input_domain_code(request, error.what()))
      throw StrictInputDomainError(std::string(*code), error.what());
    throw;
  }
}

std::vector<native_graph_v1::Stream> PolicyDispatcher::inverse(
    const native_graph_v1::StageRequest& request,
    const std::vector<native_graph_v1::Stream>& outputs,
    std::span<const std::uint8_t> metadata) const {
  enforce(request);
  if (handles_a048_typed_planes(request)) {
    auto result = inverse_a048_typed_planes(request, outputs, metadata);
    audit(metadata);
    return result;
  }
  if (handles_special(request)) {
    auto result = inverse_special(request, outputs, metadata);
    audit(metadata);
    return result;
  }
  auto result = inner_.inverse(request, outputs, metadata);
  audit(metadata);
  return result;
}

}  // namespace operator_harness_v2
