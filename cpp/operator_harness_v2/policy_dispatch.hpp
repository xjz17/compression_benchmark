#pragma once

#include "operator102_dispatcher.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace operator_harness_v2 {

struct FallbackAudit {
  std::uint64_t wrapped_stage_count = 0;
  std::uint64_t raw_fallback_stage_count = 0;
  std::vector<std::string> reasons;
  std::map<std::string, std::uint64_t> reason_counts;
};

class ExcludedPolicyError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A catalog-bound strict kernel rejected the source values in its documented
// forward domain. This type is emitted only at the forward policy boundary;
// inverse, frame, metadata and bit-exactness failures retain their original
// exception types and remain execution failures.
class StrictInputDomainError final : public std::runtime_error {
 public:
  StrictInputDomainError(std::string code, std::string message)
      : std::runtime_error(std::move(message)), code_(std::move(code)) {}

  const std::string& code() const noexcept { return code_; }

 private:
  std::string code_;
};

// Returns a stable policy reason for the five permanently excluded
// probability/model implementations, otherwise nullopt.
std::optional<std::string> excluded_action_reason(std::string_view action);
bool excluded_implementation_id(std::string_view implementation_id);

// Policy wrapper over the source-exact Operator102Dispatcher.  The wrapper is
// used for both forward and inverse, so an old frame cannot re-enable an
// excluded action during cold decode.
class PolicyDispatcher final : public native_graph_v1::StageDispatcher {
 public:
  explicit PolicyDispatcher(bool audit_fallbacks = false);

  std::string identity() const override;
  native_graph_v1::StageResult forward(
      const native_graph_v1::StageRequest&,
      const std::vector<native_graph_v1::Stream>&) const override;
  std::vector<native_graph_v1::Stream> inverse(
      const native_graph_v1::StageRequest&,
      const std::vector<native_graph_v1::Stream>&,
      std::span<const std::uint8_t>) const override;

  const FallbackAudit& fallback_audit() const { return fallback_audit_; }

 private:
  void enforce(const native_graph_v1::StageRequest&) const;
  void audit(std::span<const std::uint8_t>) const;

  bool audit_fallbacks_ = false;
  mutable FallbackAudit fallback_audit_;
  native_graph_v1::operator102_measure::Operator102Dispatcher inner_;
};

}  // namespace operator_harness_v2

