#include "selection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace unified_compression_v1 {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point begin,
                         Clock::time_point end = Clock::now()) {
  const auto value =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  return value <= 0 ? 0 : static_cast<std::uint64_t>(value);
}

std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return std::numeric_limits<std::uint64_t>::max();
  return left + right;
}

bool soft_deadline_reached(const SelectionOptions& options,
                           Clock::time_point selection_begin) {
  return options.budget.soft_wall_time_ns.has_value() &&
         elapsed_ns(selection_begin) >= *options.budget.soft_wall_time_ns;
}

bool is_unsupported_code(std::string_view code) {
  static constexpr std::array<std::string_view, 8> tokens{
      "input_domain", "domain_rejected", "unsupported", "unavailable", "disabled",
      "excluded", "forbidden", "not_available"};
  return std::any_of(tokens.begin(), tokens.end(), [&](std::string_view token) {
    return code.find(token) != std::string_view::npos;
  });
}

bool audit_has_fallback(const nlohmann::json& audit) {
  if (!audit.is_object()) return true;
  bool has_explicit_count = false;
  for (const auto* key : {"raw_fallback_stage_count", "fallback_count"}) {
    const auto it = audit.find(key);
    if (it == audit.end()) continue;
    has_explicit_count = true;
    if (it->is_number_unsigned()) {
      if (it->get<std::uint64_t>() != 0) return true;
    } else if (it->is_number_integer()) {
      if (it->get<std::int64_t>() != 0) return true;
    } else {
      // Null, strings and other malformed values cannot prove that no raw
      // fallback occurred.
      return true;
    }
  }
  bool has_explicit_flag = false;
  for (const auto* key : {"raw_fallback", "fallback_used", "used_fallback"}) {
    const auto it = audit.find(key);
    if (it == audit.end()) continue;
    has_explicit_flag = true;
    if (!it->is_boolean() || it->get<bool>()) return true;
  }
  const auto path = audit.find("execution_path");
  if (path != audit.end() &&
      (!path->is_string() || path->get<std::string>() != "strict"))
    return true;
  // A strict=true assertion alone is insufficient. Ranking requires explicit
  // zero-count and false-flag evidence for the measured execution.
  return !has_explicit_count || !has_explicit_flag;
}

bool audit_is_strict(const nlohmann::json& audit) {
  if (!audit.is_object()) return false;
  const auto strict = audit.find("strict");
  return strict != audit.end() && strict->is_boolean() &&
         strict->get<bool>() && !audit_has_fallback(audit);
}

void validate_options(const SelectionOptions& options) {
  if (options.measurement.repeats == 0)
    throw std::invalid_argument("selection measurement repeats must be positive");
  for (const auto value : {options.balanced_weights.compressed_bytes,
                           options.balanced_weights.compression_time,
                           options.balanced_weights.decompression_time}) {
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument(
          "balanced selection weights must be finite and non-negative");
  }
}

double objective_score(const CompressionResult& result,
                       const SelectionOptions& options) {
  switch (options.objective) {
    case SelectionObjective::compressed_bytes:
      return static_cast<double>(result.compressed_bytes);
    case SelectionObjective::compression_time:
      return static_cast<double>(result.compression_time_ns);
    case SelectionObjective::decompression_time:
      return static_cast<double>(result.decompression_time_ns);
    case SelectionObjective::balanced: {
      // All terms are dimensionless after normalization by source bytes. The
      // logarithm prevents one large time term from overwhelming both others.
      const double source =
          static_cast<double>(std::max<std::uint64_t>(
              1, result.original_numeric_bytes));
      const double bytes_per_byte =
          static_cast<double>(result.compressed_bytes) / source;
      const double encode_ns_per_byte =
          static_cast<double>(result.compression_time_ns) / source;
      const double decode_ns_per_byte =
          static_cast<double>(result.decompression_time_ns) / source;
      return options.balanced_weights.compressed_bytes *
                 std::log1p(bytes_per_byte) +
             options.balanced_weights.compression_time *
                 std::log1p(encode_ns_per_byte) +
             options.balanced_weights.decompression_time *
                 std::log1p(decode_ns_per_byte);
    }
  }
  throw std::logic_error("unknown selection objective");
}

struct BestKey {
  double score = std::numeric_limits<double>::infinity();
  std::uint64_t compressed_bytes = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t compression_time_ns =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t decompression_time_ns =
      std::numeric_limits<std::uint64_t>::max();
};

bool better_than(const CompressionResult& candidate, double score,
                 const BestKey& best, SelectionObjective objective) {
  if (score != best.score) return score < best.score;
  switch (objective) {
    case SelectionObjective::compressed_bytes:
      if (candidate.compression_time_ns != best.compression_time_ns)
        return candidate.compression_time_ns < best.compression_time_ns;
      if (candidate.decompression_time_ns != best.decompression_time_ns)
        return candidate.decompression_time_ns < best.decompression_time_ns;
      return false;
    case SelectionObjective::compression_time:
      if (candidate.compressed_bytes != best.compressed_bytes)
        return candidate.compressed_bytes < best.compressed_bytes;
      if (candidate.decompression_time_ns != best.decompression_time_ns)
        return candidate.decompression_time_ns < best.decompression_time_ns;
      return false;
    case SelectionObjective::decompression_time:
      if (candidate.compressed_bytes != best.compressed_bytes)
        return candidate.compressed_bytes < best.compressed_bytes;
      if (candidate.compression_time_ns != best.compression_time_ns)
        return candidate.compression_time_ns < best.compression_time_ns;
      return false;
    case SelectionObjective::balanced:
      if (candidate.compressed_bytes != best.compressed_bytes)
        return candidate.compressed_bytes < best.compressed_bytes;
      if (candidate.compression_time_ns != best.compression_time_ns)
        return candidate.compression_time_ns < best.compression_time_ns;
      if (candidate.decompression_time_ns != best.decompression_time_ns)
        return candidate.decompression_time_ns < best.decompression_time_ns;
      return false;
  }
  return false;
}

bool satisfies_constraints(const CompressionResult& result,
                           const SelectionConstraints& constraints) {
  if (constraints.max_compressed_bytes &&
      result.compressed_bytes > *constraints.max_compressed_bytes)
    return false;
  if (constraints.max_compression_time_ns &&
      result.compression_time_ns > *constraints.max_compression_time_ns)
    return false;
  if (constraints.max_decompression_time_ns &&
      result.decompression_time_ns > *constraints.max_decompression_time_ns)
    return false;
  return true;
}

std::vector<std::string> fixed30_ids() {
  return {
      "combo_v1:zstd3",
      "combo_v1:lz4",
      "combo_v1:columnize_zstd3",
      "combo_v1:columnize_lz4",
      "combo_v1:xor_zstd3",
      "combo_v1:xor_lz4",
      "combo_v1:for_bitpack_zstd3",
      "combo_v1:for_bitpack_lz4",
      "combo_v1:a048_planes_zstd3",
      "combo_v1:a048_planes_lz4",
      "combo_v1:float_deconstruct_zstd3",
      "combo_v1:float_deconstruct_lz4",
      "combo_v1:xor_semantic_raw_zstd3",
      "combo_v1:xor_semantic_raw_lz4",
      "combo_v1:xor_semantic_bitpack_zstd3",
      "combo_v1:xor_semantic_bitpack_lz4",
      "combo_v1:xor_semantic_a048_zstd3",
      "combo_v1:xor_semantic_a048_lz4",
      "combo_v1:dictionary_bitpack_zstd3",
      "combo_v1:dictionary_bitpack_lz4",
      "combo_v1:dictionary_raw_dictionary_zstd3",
      "combo_v1:dictionary_raw_dictionary_lz4",
      "combo_v1:for_bitpack_raw_base_zstd3",
      "combo_v1:for_bitpack_raw_base_lz4",
      "combo_v1:columnize_zstd3_lz4",
      "combo_v1:columnize_lz4_zstd3",
      "combo_v1:a048_planes_zstd3_lz4",
      "combo_v1:a048_planes_lz4_zstd3",
      "combo_v1:float_deconstruct_zstd3_lz4",
      "combo_v1:float_deconstruct_lz4_zstd3",
  };
}

const TrialFeedback* feedback_for(
    const std::vector<TrialFeedback>& feedback, std::string_view id) {
  const auto found = std::find_if(
      feedback.begin(), feedback.end(), [&](const TrialFeedback& item) {
        return item.candidate.id == id;
      });
  return found == feedback.end() ? nullptr : &*found;
}

std::optional<double> observed_score(const TrialFeedback* feedback) {
  if (feedback == nullptr || feedback->status != "ok" ||
      !feedback->feasible || !feedback->score)
    return std::nullopt;
  return feedback->score;
}

Candidate adaptive_candidate(std::string id, std::string stage,
                             std::string preferred_backend) {
  Candidate result = Candidate::method_candidate(id, id);
  result.metadata = {
      {"proposal_strategy", "feedback_adaptive_fixed30_v1"},
      {"proposal_stage", std::move(stage)},
      {"preferred_backend", std::move(preferred_backend)},
      {"available_candidate_scope", "fixed30"},
      {"uses_ground_truth_labels", false},
  };
  return result;
}

bool method_is_eligible(std::span<const std::string> eligible_methods,
                        std::string_view id) {
  return eligible_methods.empty() ||
         std::find(eligible_methods.begin(), eligible_methods.end(), id) !=
             eligible_methods.end();
}

std::optional<Candidate> adaptive_fixed30_proposal(
    const std::vector<TrialFeedback>& feedback,
    std::span<const std::string> eligible_methods = {}) {
  if (feedback_for(feedback, "combo_v1:zstd3") == nullptr)
    return adaptive_candidate("combo_v1:zstd3", "mandatory_baseline", "unknown");
  if (feedback_for(feedback, "combo_v1:lz4") == nullptr)
    return adaptive_candidate("combo_v1:lz4", "mandatory_baseline", "unknown");

  const auto zstd_score =
      observed_score(feedback_for(feedback, "combo_v1:zstd3"));
  const auto lz4_score =
      observed_score(feedback_for(feedback, "combo_v1:lz4"));
  const bool prefer_lz4 =
      lz4_score && (!zstd_score || *lz4_score < *zstd_score);
  const std::string preferred_backend = prefer_lz4 ? "lz4" : "zstd3";

  // Each row is the same transform with the Zstd-first and LZ4-first terminal
  // assignments. The first arm follows the observed baseline winner.
  static const std::array<std::array<std::string_view, 2>, 14> groups{{
      {"combo_v1:columnize_zstd3", "combo_v1:columnize_lz4"},
      {"combo_v1:xor_zstd3", "combo_v1:xor_lz4"},
      {"combo_v1:for_bitpack_zstd3", "combo_v1:for_bitpack_lz4"},
      {"combo_v1:a048_planes_zstd3", "combo_v1:a048_planes_lz4"},
      {"combo_v1:float_deconstruct_zstd3",
       "combo_v1:float_deconstruct_lz4"},
      {"combo_v1:xor_semantic_raw_zstd3",
       "combo_v1:xor_semantic_raw_lz4"},
      {"combo_v1:xor_semantic_bitpack_zstd3",
       "combo_v1:xor_semantic_bitpack_lz4"},
      {"combo_v1:xor_semantic_a048_zstd3",
       "combo_v1:xor_semantic_a048_lz4"},
      {"combo_v1:dictionary_bitpack_zstd3",
       "combo_v1:dictionary_bitpack_lz4"},
      {"combo_v1:dictionary_raw_dictionary_zstd3",
       "combo_v1:dictionary_raw_dictionary_lz4"},
      {"combo_v1:for_bitpack_raw_base_zstd3",
       "combo_v1:for_bitpack_raw_base_lz4"},
      {"combo_v1:columnize_zstd3_lz4",
       "combo_v1:columnize_lz4_zstd3"},
      {"combo_v1:a048_planes_zstd3_lz4",
       "combo_v1:a048_planes_lz4_zstd3"},
      {"combo_v1:float_deconstruct_zstd3_lz4",
       "combo_v1:float_deconstruct_lz4_zstd3"},
  }};

  std::optional<double> current_best;
  for (const auto& item : feedback) {
    const auto score = observed_score(&item);
    if (score && (!current_best || *score < *current_best))
      current_best = score;
  }

  for (const auto& group : groups) {
    const std::string preferred(
        group[prefer_lz4 ? 1U : 0U]);
    const std::string counterpart(
        group[prefer_lz4 ? 0U : 1U]);
    if (!method_is_eligible(eligible_methods, preferred)) continue;
    const auto* preferred_feedback = feedback_for(feedback, preferred);
    if (preferred_feedback == nullptr)
      return adaptive_candidate(preferred, "preferred_backend_probe",
                                preferred_backend);

    // Spend a second trial on the same transform only when its preferred
    // backend was feasible and remains competitive with the best observation.
    // This is feedback-directed exploration; the 10% guard is fixed and does
    // not use corpus labels or the frozen recommendation oracle.
    const auto preferred_score = observed_score(preferred_feedback);
    const bool competitive =
        preferred_score && current_best &&
        (*preferred_score <= *current_best * 1.10 ||
         (*current_best == 0.0 && *preferred_score == 0.0));
    if (competitive && method_is_eligible(eligible_methods, counterpart) &&
        feedback_for(feedback, counterpart) == nullptr)
      return adaptive_candidate(counterpart, "counterpart_validation",
                                preferred_backend);
  }
  return std::nullopt;
}

void reject_unknown_keys(const nlohmann::json& value,
                         std::initializer_list<std::string_view> allowed,
                         std::string_view context) {
  if (!value.is_object())
    throw std::invalid_argument(std::string(context) + " must be an object");
  for (auto it = value.begin(); it != value.end(); ++it) {
    const bool known =
        std::find(allowed.begin(), allowed.end(), it.key()) != allowed.end();
    if (!known)
      throw std::invalid_argument(std::string(context) +
                                  " contains unknown key: " + it.key());
  }
}

std::optional<std::uint64_t> optional_u64(const nlohmann::json& object,
                                          const char* key) {
  const auto it = object.find(key);
  if (it == object.end()) return std::nullopt;
  if (it->is_number_unsigned()) return it->get<std::uint64_t>();
  if (it->is_number_integer()) {
    const auto value = it->get<std::int64_t>();
    if (value >= 0) return static_cast<std::uint64_t>(value);
  }
  throw std::invalid_argument(std::string(key) +
                              " must be a non-negative integer");
}

nlohmann::json candidate_json(const Candidate& candidate) {
  nlohmann::json result = {
      {"id", candidate.id},
      {"metadata", candidate.metadata},
  };
  if (candidate.method) result["method"] = *candidate.method;
  if (candidate.tree) result["tree"] = *candidate.tree;
  return result;
}

}  // namespace

Candidate Candidate::method_candidate(std::string id, std::string method) {
  Candidate result;
  result.id = std::move(id);
  result.method = std::move(method);
  return result;
}

Candidate Candidate::tree_candidate(std::string id, nlohmann::json tree) {
  Candidate result;
  result.id = std::move(id);
  result.tree = std::move(tree);
  return result;
}

std::vector<Candidate> fixed30_candidates() {
  std::vector<Candidate> result;
  const auto ids = fixed30_ids();
  result.reserve(ids.size());
  for (const auto& id : ids)
    result.push_back(Candidate::method_candidate(id, id));
  return result;
}

std::optional<Candidate> propose_fixed30(
    const std::vector<TrialFeedback>& feedback,
    std::span<const std::string> seed_order,
    std::span<const std::string> eligible_methods) {
  // Baselines establish an observed backend preference before transported
  // ranks or transform exploration consume the remaining budget.
  if (feedback_for(feedback, "combo_v1:zstd3") == nullptr ||
      feedback_for(feedback, "combo_v1:lz4") == nullptr)
    return adaptive_fixed30_proposal(feedback, eligible_methods);

  const auto ids = fixed30_ids();
  for (const auto& seeded : seed_order) {
    const bool in_scope =
        std::find(ids.begin(), ids.end(), seeded) != ids.end();
    if (!in_scope || !method_is_eligible(eligible_methods, seeded) ||
        seeded == "combo_v1:zstd3" ||
        seeded == "combo_v1:lz4" || feedback_for(feedback, seeded) != nullptr)
      continue;
    Candidate candidate =
        adaptive_candidate(seeded, "seeded_prior", "observed_after_baselines");
    candidate.metadata["seed_order_kind"] = "caller_rank_transport";
    candidate.metadata["model_fitted_to_fixed30"] = false;
    return candidate;
  }
  return adaptive_fixed30_proposal(feedback, eligible_methods);
}

SelectionResult select(const Registry& registry, const Column& column,
                       const SelectionOptions& options) {
  const auto selection_begin = Clock::now();
  validate_options(options);
  validate_column(column);

  SelectionResult output;
  const bool uses_default_scope =
      !options.proposer && options.candidates.empty();
  output.candidate_scope =
      options.candidate_scope.empty()
          ? (uses_default_scope ? "fixed30" : "custom")
          : options.candidate_scope;

  std::vector<Candidate> static_candidates;
  if (!options.proposer && !uses_default_scope)
    static_candidates = options.candidates;

  // One metadata-only eligibility pass protects the bounded default budget
  // from known dtype/nonempty rejections. Custom proposals remain charged and
  // are intentionally not prefiltered.
  std::vector<std::string> default_eligible_methods;
  if (uses_default_scope) {
    for (const auto& candidate : fixed30_candidates()) {
      const auto& method = registry.resolve(*candidate.method);
      const bool dtype_ok =
          std::find(method.dtypes.begin(), method.dtypes.end(), column.dtype) !=
          method.dtypes.end();
      const bool requires_nonempty =
          method.contract.value("requires_nonempty_values", false);
      if (method.enabled && method.available && dtype_ok &&
          (!requires_nonempty || !column.values.empty()))
        default_eligible_methods.push_back(*candidate.method);
    }
  }

  std::size_t next_static = 0;
  std::unordered_set<std::string> seen_ids;
  BestKey best;
  std::optional<CompressionResult> winning_result;
  std::optional<Candidate> winning_candidate;

  while (output.feedback.size() < options.budget.max_trials) {
    if (soft_deadline_reached(options, selection_begin)) {
      output.stop_reason = "soft_wall_budget_reached";
      break;
    }

    std::optional<Candidate> proposed;
    if (options.proposer) {
      proposed = options.proposer(output.feedback);
      if (!proposed) {
        output.stop_reason = "proposer_exhausted";
        break;
      }
    } else if (uses_default_scope) {
      proposed = propose_fixed30(output.feedback, {},
                                 default_eligible_methods);
      if (!proposed) {
        output.stop_reason = "proposer_exhausted";
        break;
      }
    } else {
      if (next_static >= static_candidates.size()) {
        output.stop_reason = "candidate_list_exhausted";
        break;
      }
      proposed = static_candidates[next_static++];
    }

    // A slow proposer cannot cause a new codec call to be admitted after the
    // soft deadline. The unexecuted proposal is not a charged harness trial.
    if (soft_deadline_reached(options, selection_begin)) {
      output.stop_reason = "soft_wall_budget_reached";
      break;
    }

    Candidate candidate = std::move(*proposed);
    TrialFeedback feedback;
    feedback.candidate = candidate;

    const bool shape_valid =
        !candidate.id.empty() &&
        (candidate.method.has_value() != candidate.tree.has_value()) &&
        (!candidate.method || !candidate.method->empty()) &&
        (!candidate.tree || !candidate.tree->is_null());
    if (!shape_valid) {
      feedback.status = "error";
      feedback.error_code = "invalid_candidate";
      feedback.error_message =
          "candidate requires a non-empty id and exactly one method or tree";
      output.feedback.push_back(std::move(feedback));
      continue;
    }
    if (!seen_ids.insert(candidate.id).second) {
      feedback.status = "duplicate";
      feedback.error_code = "duplicate_candidate";
      feedback.error_message = "candidate id was already measured";
      output.feedback.push_back(std::move(feedback));
      continue;
    }

    const auto trial_begin = Clock::now();
    try {
      CompressionResult measured =
          candidate.method
              ? registry.compress(column, *candidate.method,
                                  options.measurement)
              : registry.compress_tree(column, *candidate.tree,
                                       options.measurement);
      feedback.trial_wall_time_ns = elapsed_ns(trial_begin);
      output.costs.probe_wall_time_ns = saturating_add(
          output.costs.probe_wall_time_ns, feedback.trial_wall_time_ns);
      output.costs.probe_compression_time_ns = saturating_add(
          output.costs.probe_compression_time_ns,
          measured.compression_time_ns);
      output.costs.probe_decompression_time_ns = saturating_add(
          output.costs.probe_decompression_time_ns,
          measured.decompression_time_ns);

      feedback.compressed_bytes = measured.compressed_bytes;
      feedback.compression_ratio = measured.compression_ratio;
      feedback.compression_time_ns = measured.compression_time_ns;
      feedback.decompression_time_ns = measured.decompression_time_ns;
      feedback.bit_exact = measured.bit_exact;
      feedback.execution_audit = measured.execution_audit;
      feedback.strict = measured.bit_exact &&
                        audit_is_strict(measured.execution_audit);

      // Bit-exact, explicitly strict and fallback-free are unconditional
      // ranking invariants rather than caller-relaxable cost constraints.
      if (!feedback.strict) {
        feedback.status = "non_strict";
      } else {
        feedback.status = "ok";
        feedback.feasible =
            satisfies_constraints(measured, options.constraints);
        if (feedback.feasible) {
          const double score = objective_score(measured, options);
          feedback.score = score;
          if (better_than(measured, score, best, options.objective)) {
            best = {score, measured.compressed_bytes,
                    measured.compression_time_ns,
                    measured.decompression_time_ns};
            winning_candidate = candidate;
            winning_result = std::move(measured);
          }
        }
      }
    } catch (const Error& error) {
      feedback.trial_wall_time_ns = elapsed_ns(trial_begin);
      output.costs.probe_wall_time_ns = saturating_add(
          output.costs.probe_wall_time_ns, feedback.trial_wall_time_ns);
      feedback.error_code = error.code();
      feedback.error_message = error.what();
      feedback.status =
          is_unsupported_code(error.code()) ? "unsupported" : "error";
    } catch (const std::exception& error) {
      feedback.trial_wall_time_ns = elapsed_ns(trial_begin);
      output.costs.probe_wall_time_ns = saturating_add(
          output.costs.probe_wall_time_ns, feedback.trial_wall_time_ns);
      feedback.status = "error";
      feedback.error_code = "execution_error";
      feedback.error_message = error.what();
    } catch (...) {
      feedback.trial_wall_time_ns = elapsed_ns(trial_begin);
      output.costs.probe_wall_time_ns = saturating_add(
          output.costs.probe_wall_time_ns, feedback.trial_wall_time_ns);
      feedback.status = "error";
      feedback.error_code = "unknown_execution_error";
      feedback.error_message = "candidate execution threw a non-standard exception";
    }
    output.feedback.push_back(std::move(feedback));
  }

  if (output.stop_reason.empty()) {
    if (output.feedback.size() >= options.budget.max_trials)
      output.stop_reason = "trial_budget_exhausted";
    else
      output.stop_reason = "candidate_list_exhausted";
  }

  const auto finalization_begin = Clock::now();
  output.selected_candidate = std::move(winning_candidate);
  output.selected_result = std::move(winning_result);
  output.costs.finalization_wall_time_ns = elapsed_ns(finalization_begin);
  output.costs.total_wall_time_ns = elapsed_ns(selection_begin);
  const auto accounted = saturating_add(output.costs.probe_wall_time_ns,
                                        output.costs.finalization_wall_time_ns);
  output.costs.selection_overhead_ns =
      output.costs.total_wall_time_ns > accounted
          ? output.costs.total_wall_time_ns - accounted
          : 0;
  output.costs.final_recompression_performed = false;
  return output;
}

SelectionOptions selection_options_from_json(
    const nlohmann::json& selection) {
  reject_unknown_keys(
      selection,
      {"objective", "constraints", "budget", "measurement",
       "balanced_weights", "candidates", "candidate_scope"},
      "selection");
  SelectionOptions result;

  if (const auto it = selection.find("objective"); it != selection.end()) {
    if (!it->is_string())
      throw std::invalid_argument("selection.objective must be a string");
    const auto value = it->get<std::string>();
    if (value == "compressed_bytes")
      result.objective = SelectionObjective::compressed_bytes;
    else if (value == "compression_time")
      result.objective = SelectionObjective::compression_time;
    else if (value == "decompression_time")
      result.objective = SelectionObjective::decompression_time;
    else if (value == "balanced")
      result.objective = SelectionObjective::balanced;
    else
      throw std::invalid_argument("unknown selection objective: " + value);
  }

  if (const auto it = selection.find("constraints");
      it != selection.end()) {
    reject_unknown_keys(
        *it,
        {"max_compressed_bytes", "max_compression_time_ns",
         "max_decompression_time_ns"},
        "selection.constraints");
    result.constraints.max_compressed_bytes =
        optional_u64(*it, "max_compressed_bytes");
    result.constraints.max_compression_time_ns =
        optional_u64(*it, "max_compression_time_ns");
    result.constraints.max_decompression_time_ns =
        optional_u64(*it, "max_decompression_time_ns");
  }

  if (const auto it = selection.find("budget"); it != selection.end()) {
    reject_unknown_keys(*it, {"max_trials", "soft_wall_time_ns"},
                        "selection.budget");
    if (const auto trials = it->find("max_trials"); trials != it->end()) {
      nlohmann::json temporary = {{"value", *trials}};
      const auto value = *optional_u64(temporary, "value");
      if (value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(
            "selection.budget.max_trials exceeds size_t");
      result.budget.max_trials = static_cast<std::size_t>(value);
    }
    result.budget.soft_wall_time_ns =
        optional_u64(*it, "soft_wall_time_ns");
  }

  if (const auto it = selection.find("measurement");
      it != selection.end()) {
    reject_unknown_keys(*it, {"warmups", "repeats"},
                        "selection.measurement");
    if (const auto warmups = optional_u64(*it, "warmups")) {
      if (*warmups > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("selection.measurement.warmups too large");
      result.measurement.warmups = static_cast<std::uint32_t>(*warmups);
    }
    if (const auto repeats = optional_u64(*it, "repeats")) {
      if (*repeats > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("selection.measurement.repeats too large");
      result.measurement.repeats = static_cast<std::uint32_t>(*repeats);
    }
  }

  if (const auto it = selection.find("balanced_weights");
      it != selection.end()) {
    reject_unknown_keys(
        *it,
        {"compressed_bytes", "compression_time", "decompression_time"},
        "selection.balanced_weights");
    const auto read_weight = [&](const char* key, double& destination) {
      const auto weight = it->find(key);
      if (weight == it->end()) return;
      if (!weight->is_number())
        throw std::invalid_argument(
            std::string("selection.balanced_weights.") + key +
            " must be numeric");
      destination = weight->get<double>();
    };
    read_weight("compressed_bytes",
                result.balanced_weights.compressed_bytes);
    read_weight("compression_time",
                result.balanced_weights.compression_time);
    read_weight("decompression_time",
                result.balanced_weights.decompression_time);
  }

  if (const auto it = selection.find("candidate_scope");
      it != selection.end()) {
    if (!it->is_string())
      throw std::invalid_argument(
          "selection.candidate_scope must be a string");
    result.candidate_scope = it->get<std::string>();
  }

  if (const auto it = selection.find("candidates");
      it != selection.end()) {
    if (!it->is_array())
      throw std::invalid_argument("selection.candidates must be an array");
    for (const auto& value : *it) {
      reject_unknown_keys(value, {"id", "method", "tree", "metadata"},
                          "selection candidate");
      if (!value.contains("id") || !value.at("id").is_string())
        throw std::invalid_argument(
            "selection candidate id must be a string");
      const bool has_method = value.contains("method");
      const bool has_tree = value.contains("tree");
      if (has_method == has_tree)
        throw std::invalid_argument(
            "selection candidate requires exactly one method or tree");
      Candidate candidate;
      candidate.id = value.at("id").get<std::string>();
      if (has_method) {
        if (!value.at("method").is_string())
          throw std::invalid_argument(
              "selection candidate method must be a string");
        candidate.method = value.at("method").get<std::string>();
      } else {
        if (!value.at("tree").is_object())
          throw std::invalid_argument(
              "selection candidate tree must be an object");
        candidate.tree = value.at("tree");
      }
      if (value.contains("metadata")) {
        if (!value.at("metadata").is_object())
          throw std::invalid_argument(
              "selection candidate metadata must be an object");
        candidate.metadata = value.at("metadata");
      }
      result.candidates.push_back(std::move(candidate));
    }
  }
  validate_options(result);
  return result;
}

nlohmann::json selection_report(const SelectionResult& result) {
  nlohmann::json trials = nlohmann::json::array();
  for (const auto& item : result.feedback) {
    trials.push_back({
        {"candidate", candidate_json(item.candidate)},
        {"status", item.status},
        {"error_code", item.error_code.empty()
                           ? nlohmann::json(nullptr)
                           : nlohmann::json(item.error_code)},
        {"error_message", item.error_message.empty()
                              ? nlohmann::json(nullptr)
                              : nlohmann::json(item.error_message)},
        {"compressed_bytes", item.compressed_bytes
                                 ? nlohmann::json(*item.compressed_bytes)
                                 : nlohmann::json(nullptr)},
        {"compression_ratio", item.compression_ratio
                                  ? nlohmann::json(*item.compression_ratio)
                                  : nlohmann::json(nullptr)},
        {"compression_time_ns", item.compression_time_ns
                                    ? nlohmann::json(*item.compression_time_ns)
                                    : nlohmann::json(nullptr)},
        {"decompression_time_ns",
         item.decompression_time_ns
             ? nlohmann::json(*item.decompression_time_ns)
             : nlohmann::json(nullptr)},
        {"bit_exact", item.bit_exact},
        {"strict", item.strict},
        {"feasible", item.feasible},
        {"score", item.score ? nlohmann::json(*item.score)
                              : nlohmann::json(nullptr)},
        {"trial_wall_time_ns", item.trial_wall_time_ns},
        {"execution_audit", item.execution_audit},
    });
  }

  nlohmann::json selected = nullptr;
  if (result.selected_candidate && result.selected_result) {
    const auto& measured = *result.selected_result;
    selected = {
        {"candidate", candidate_json(*result.selected_candidate)},
        {"method", measured.method},
        {"original_numeric_bytes", measured.original_numeric_bytes},
        {"compressed_bytes", measured.compressed_bytes},
        {"compression_ratio", measured.compression_ratio
                                  ? nlohmann::json(*measured.compression_ratio)
                                  : nlohmann::json(nullptr)},
        {"compression_time_ns", measured.compression_time_ns},
        {"decompression_time_ns", measured.decompression_time_ns},
        {"bit_exact", measured.bit_exact},
        {"execution_audit", measured.execution_audit},
        {"compressed_data_omitted", true},
    };
  }

  return {
      {"schema", "unified-compression-selection-report-v1"},
      {"candidate_scope", result.candidate_scope},
      {"proposal_strategy",
       result.candidate_scope == "fixed30"
           ? "feedback_adaptive_fixed30_v1"
           : "caller_provided"},
      {"stop_reason", result.stop_reason},
      {"trial_count", result.feedback.size()},
      {"trials", std::move(trials)},
      {"selected", std::move(selected)},
      {"costs",
       {
           {"selection_overhead_ns",
            result.costs.selection_overhead_ns},
           {"probe_wall_time_ns", result.costs.probe_wall_time_ns},
           {"probe_compression_time_ns",
            result.costs.probe_compression_time_ns},
           {"probe_decompression_time_ns",
            result.costs.probe_decompression_time_ns},
           {"finalization_wall_time_ns",
            result.costs.finalization_wall_time_ns},
           {"total_wall_time_ns", result.costs.total_wall_time_ns},
           {"final_recompression_performed",
            result.costs.final_recompression_performed},
       }},
      {"compressed_binary_included", false},
  };
}

}  // namespace unified_compression_v1
