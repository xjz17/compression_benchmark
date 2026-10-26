#pragma once

#include "api.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace unified_compression_v1 {

// A proposal is either a registered method name or a complete explicit
// operator-tree specification. Exactly one of method and tree must be set.
struct Candidate {
  std::string id;
  std::optional<std::string> method;
  std::optional<nlohmann::json> tree;
  nlohmann::json metadata = nlohmann::json::object();

  static Candidate method_candidate(std::string id, std::string method);
  static Candidate tree_candidate(std::string id, nlohmann::json tree);
};

enum class SelectionObjective {
  compressed_bytes,
  compression_time,
  decompression_time,
  balanced,
};

struct SelectionConstraints {
  std::optional<std::uint64_t> max_compressed_bytes;
  std::optional<std::uint64_t> max_compression_time_ns;
  std::optional<std::uint64_t> max_decompression_time_ns;
};

struct SelectionBudget {
  // Hard limit. Every returned proposal consumes one trial, including an
  // invalid, duplicate, unsupported or failed proposal.
  std::size_t max_trials = 8;
  // Soft admission deadline. It is checked before a proposal and immediately
  // before a trial. An in-flight codec call is never interrupted.
  std::optional<std::uint64_t> soft_wall_time_ns;
};

struct BalancedWeights {
  double compressed_bytes = 1.0;
  double compression_time = 1.0;
  double decompression_time = 1.0;
};

// Stable status values are: "ok", "unsupported", "non_strict", "error"
// and "duplicate". Feasibility is separate: an "ok" trial can be
// infeasible because it violates a caller-provided cost constraint.
struct TrialFeedback {
  Candidate candidate;
  std::string status;
  std::string error_code;
  std::string error_message;
  std::optional<std::uint64_t> compressed_bytes;
  std::optional<double> compression_ratio;
  std::optional<std::uint64_t> compression_time_ns;
  std::optional<std::uint64_t> decompression_time_ns;
  bool bit_exact = false;
  bool strict = false;
  bool feasible = false;
  std::optional<double> score;
  std::uint64_t trial_wall_time_ns = 0;
  nlohmann::json execution_audit = nlohmann::json::object();
};

// Called once with empty feedback, then again after every charged trial. A
// nullopt return ends proposal generation. The callback must not retain a
// reference to the feedback vector.
using Proposer = std::function<std::optional<Candidate>(
    const std::vector<TrialFeedback>& feedback)>;

struct SelectionOptions {
  SelectionObjective objective = SelectionObjective::compressed_bytes;
  SelectionConstraints constraints;
  SelectionBudget budget;
  BalancedWeights balanced_weights;
  MeasurementOptions measurement;

  // Proposal precedence: proposer, then candidates, then the adaptive fixed30
  // strategy documented by selection_report().
  std::vector<Candidate> candidates;
  Proposer proposer;

  // Empty is filled with "fixed30" for the built-in list and "custom" for
  // caller-provided candidates/proposers. It is descriptive, not trusted.
  std::string candidate_scope;
};

struct SelectionCosts {
  // Time outside Registry::compress/compress_tree and final result movement.
  std::uint64_t selection_overhead_ns = 0;
  // Wall time around all charged harness trials, including failed trials.
  std::uint64_t probe_wall_time_ns = 0;
  // Medians reported by successful harness trials, summed for accounting.
  std::uint64_t probe_compression_time_ns = 0;
  std::uint64_t probe_decompression_time_ns = 0;
  // Moving the already measured winning frame into the result. No codec runs.
  std::uint64_t finalization_wall_time_ns = 0;
  std::uint64_t total_wall_time_ns = 0;
  bool final_recompression_performed = false;
};

struct SelectionResult {
  std::string candidate_scope;
  std::string stop_reason;
  std::vector<TrialFeedback> feedback;
  std::optional<Candidate> selected_candidate;
  std::optional<CompressionResult> selected_result;
  SelectionCosts costs;
};

// The exact isolated 30-candidate scope used by harness_combo_pilot_v1. The
// two baselines use their unified canonical names and are charged like every
// other proposal.
std::vector<Candidate> fixed30_candidates();

// Stateless feedback policy used by the built-in selector and reusable by a
// prepared recommender. Optional seed_order is rank transport only: it is
// restricted to the fixed30 scope and does not claim a model fitted to it.
std::optional<Candidate> propose_fixed30(
    const std::vector<TrialFeedback>& feedback,
    std::span<const std::string> seed_order = {},
    std::span<const std::string> eligible_methods = {});

// Runs complete-column harness trials and returns the already measured winning
// UCC1 frame. This function neither fits nor updates a recommendation model.
SelectionResult select(const Registry& registry, const Column& column,
                       const SelectionOptions& options = {});

// Strictly parses one nested "selection" object. Unknown keys and malformed
// candidates are rejected. A callback cannot be represented in JSON.
SelectionOptions selection_options_from_json(const nlohmann::json& selection);

// Machine-readable receipt containing proposals, outcomes, costs and the
// selected metrics/audit. The selected compressed bytes are intentionally
// omitted.
nlohmann::json selection_report(const SelectionResult& result);

}  // namespace unified_compression_v1
