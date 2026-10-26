#pragma once

#include "catalog.hpp"
#include "features.hpp"
#include "graph.hpp"
#include "policy.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wc::pipeline_v1 {

inline constexpr const char* online_runtime_schema =
    "native-online-sequential-policy-v3";
inline constexpr const char* online_semantic_mask_contract =
    "native-online-semantic-action-mask-v1";
inline constexpr const char* online_semantic_mask_contract_sha256 =
    "53a5162e32be591e05fce9ba31ee758bc343d733a133d865b4e2ddef43947cd0";
inline constexpr const char* online_control_schema =
    "native-online-flow-stop-controls-v2";
inline constexpr const char* online_training_session_schema =
    "native-online-training-session-v1";
inline constexpr const char* online_matched_credential_schema =
    "native-online-matched-training-credential-v1";
inline constexpr const char* online_matched_training_contract =
    "native-online-v13-raw-root-per-flow-stop-grpo-v1";
inline constexpr std::uint32_t online_action_count = 6373;
inline constexpr std::uint32_t flow_stop_begin = 6341;
inline constexpr std::uint32_t flow_stop_count = 32;
inline constexpr std::uint32_t online_matched_max_policy_steps = 10;
inline constexpr std::uint32_t online_matched_max_nodes = 64;
inline constexpr std::uint32_t online_matched_max_active_streams = 32;
inline constexpr std::uint32_t online_matched_max_ordinary_depth = 10;
inline constexpr const char* online_registry_fingerprint =
    "cbd7756de4ebaf544a4eac138dc92c9eac85c701d09e8855a1b624e456170542";
inline constexpr const char* online_control_descriptor_canonical_json =
    R"({"action_count":6373,"base_action_count":6341,"base_registry_fingerprint":"50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd","disabled_global_stop_action":0,"flow_stop_begin":6341,"flow_stop_count":32,"schema":"native-online-flow-stop-controls-v2","slot_semantics":"ascending_open_stream_id"})";

// This request describes one already-loaded in-memory block. Model and catalog
// loading are deliberately outside compress(). The roots are copied into the
// persistent GraphSession after the outer online clock starts.
struct OnlineRequest {
  std::vector<native_graph_v1::RootInput> roots;
  std::array<double, 3> weights{1.0, 0.0, 0.0};
  native_graph_v1::Limits graph_limits;
  // Default deployment bound from the experiment contract. Values through 64
  // are accepted for explicit ablations; reaching the bound explicitly closes
  // every remaining open flow.
  std::uint32_t max_policy_steps = 10;
};

struct OnlineTimings {
  // Exact outer-clock offset at which the terminal recipe is ready and the
  // sole GraphSession::stop() framing call is about to begin.
  std::uint64_t recipe_ready_ns = 0;
  std::uint64_t feature_ns = 0;
  std::uint64_t mask_ns = 0;
  std::uint64_t policy_ns = 0;
  std::uint64_t stage_ns = 0;
  std::uint64_t framing_ns = 0;
  std::uint64_t total_ns = 0;
};

struct OnlineCounters {
  std::uint32_t policy_calls = 0;
  // Actions returned by the policy. Budget-forced flow closes are counted
  // separately, although their explicit IDs remain in recipe_action_ids.
  std::uint32_t selected_policy_actions = 0;
  std::uint32_t forward_stage_calls = 0;      // Includes internal adapters.
  std::uint32_t candidate_evaluations = 0;
  std::uint32_t replayed_prefix_stages = 0;
  std::uint32_t closed_flow_count = 0;
  std::uint32_t forced_flow_stops = 0;
  // A flow scan computes all 19 DBP max_point decisions at once. The cache is
  // session-local and keyed by immutable GraphSession flow IDs.
  std::uint32_t semantic_mask_flow_scans = 0;
  std::uint64_t semantic_mask_value_visits = 0;
};

struct OnlineResult {
  native_graph_v1::Bytes frame;
  // Policy-visible graph/flow-close IDs. The legacy global STOP ID 0 never
  // appears. Budget-forced flow closes are explicit in this trace. Internal
  // carrier adapters appear only in native_steps with policy_action_id == -1.
  std::vector<std::uint32_t> recipe_action_ids;
  std::vector<native_graph_v1::Step> native_steps;
  OnlineCounters counters;
  OnlineTimings timing;
  native_graph_v1::Limits effective_graph_limits;
  std::uint64_t original_numeric_bytes = 0;
  std::string termination;
  // True only when one uninterrupted C++ outer clock includes feature/mask,
  // policy inference, selected transitions, graph scheduling and framing.
  // Interactive sessions and model-free forced replays deliberately leave
  // this false so their timings cannot be relabelled as deployment E_total.
  bool complete_online_timing = false;
  bool matched_training = false;
  bool diagnostic_only = true;
  std::uint32_t matched_max_policy_steps = 0;
  std::string model_sha256;
  std::string source_checkpoint_sha256;
  std::string matched_credential_sha256;

  std::optional<double> compression_ratio() const;
  std::optional<double> compression_ns_per_byte() const;
};

struct OnlineObservation {
  std::array<double, wc::features_v1::observation_width> observation{};
  std::vector<std::uint32_t> legal_action_ids;
  std::uint32_t policy_calls = 0;
  std::uint32_t policy_graph_nodes = 0;
  std::uint32_t open_flow_count = 0;
  std::uint32_t semantic_mask_flow_scans = 0;
  std::uint64_t semantic_mask_value_visits = 0;
};

struct OnlineTransition {
  OnlineObservation state;
  std::uint32_t action_id = 0;
  // The model action is populated by OnlinePipeline::replay_forced(). The
  // forced action remains authoritative for the state transition.
  std::optional<std::uint32_t> greedy_action_id;
};

struct OnlineReplayResult {
  std::vector<OnlineTransition> transitions;
  OnlineResult result;
  // False for OnlineRuntime's model-free teacher helper. Only a replay through
  // OnlinePipeline evaluates the current model on every step and may be used
  // as the compression-time term of a training reward.
  bool reward_timing_eligible = false;
};

enum class OnlineTimingMode {
  Interactive,
  ContinuousOuterClock,
};

class OnlineRuntime;

// Stateful raw-root environment shared by deployment, teacher replay and RL.
// It is non-copyable and borrows its OnlineRuntime and StageDispatcher, which
// must outlive the session. observe(), step() and finish() are the only state
// transition path; callers cannot supply a second mask implementation.
class OnlineSession {
 public:
  ~OnlineSession();
  OnlineSession(OnlineSession&&) noexcept;
  OnlineSession& operator=(OnlineSession&&) noexcept;
  OnlineSession(const OnlineSession&) = delete;
  OnlineSession& operator=(const OnlineSession&) = delete;

  bool terminal() const noexcept;
  const OnlineObservation& observe();
  // Applies exactly one legal policy-visible action. Reaching the horizon
  // closes every remaining flow from the highest open slot downward.
  void step(std::uint32_t action_id);
  // Explicitly close remaining flows and produce a complete cold-decodable
  // frame. A terminal session is unchanged.
  void finish(std::string termination = "forced_actions_finished");
  const OnlineResult& result() const;
  OnlineResult take_result();

 private:
  struct Impl;
  explicit OnlineSession(std::unique_ptr<Impl>);
  void add_policy_time(std::uint64_t);
  std::unique_ptr<Impl> impl_;
  friend class OnlineRuntime;
  friend class OnlinePipeline;
};

class OnlineRuntime {
 public:
  static OnlineRuntime load(const std::filesystem::path& catalog,
                            const std::filesystem::path& experiment_profile);

  OnlineSession start(const OnlineRequest&,
                      const native_graph_v1::StageDispatcher&,
                      OnlineTimingMode = OnlineTimingMode::Interactive) const;
  // Model-free helper for generating teacher states and checking forced
  // trajectories. Its timing is never reward/deployment eligible.
  OnlineReplayResult replay_forced(
      const OnlineRequest&, std::span<const std::uint32_t> forced_actions,
      const native_graph_v1::StageDispatcher&,
      bool capture_transitions = true) const;

  const ActionCatalog& catalog() const noexcept { return catalog_; }

 private:
  OnlineRuntime(ActionCatalog catalog,
                std::array<std::vector<std::uint32_t>, 32> actions_by_slot,
                std::vector<std::uint32_t> concat_actions)
      : catalog_(std::move(catalog)),
        actions_by_slot_(std::move(actions_by_slot)),
        concat_actions_(std::move(concat_actions)) {}

  ActionCatalog catalog_;
  std::array<std::vector<std::uint32_t>, 32> actions_by_slot_;
  std::vector<std::uint32_t> concat_actions_;
  friend class OnlineSession;
  friend class OnlinePipeline;
};

struct OnlineModelAdmission {
  bool matched_training = false;
  bool diagnostic_only = true;
  std::uint32_t matched_max_policy_steps = 0;
  std::string model_sha256;
  std::string source_checkpoint_sha256;
  std::string credential_sha256;
};

class OnlinePipeline {
 public:
  // Loading and all identity/profile validation happen once, outside the
  // per-block online timing boundary.
  static OnlinePipeline load(const std::filesystem::path& model,
                             const std::filesystem::path& catalog,
                             const std::filesystem::path& experiment_profile,
                             std::optional<std::filesystem::path>
                                 matched_credential = std::nullopt);

  OnlineResult compress(const OnlineRequest&,
                        const native_graph_v1::StageDispatcher&) const;
  // Uses the current model for the same per-step inference cost and legality
  // checks as compress(), while the supplied legal actions drive transitions.
  // This is the matched offline path for teacher/GRPO reward measurement.
  OnlineReplayResult replay_forced(
      const OnlineRequest&, std::span<const std::uint32_t> forced_actions,
      const native_graph_v1::StageDispatcher&,
      bool capture_transitions = true) const;

  const ActionCatalog& catalog() const noexcept { return runtime_.catalog(); }
  const weighted_grpo::native_policy_v1::Identity& model_identity() const noexcept {
    return policy_.identity();
  }
  const OnlineModelAdmission& admission() const noexcept { return admission_; }

 private:
  OnlinePipeline(weighted_grpo::native_policy_v1::Policy policy,
                 OnlineRuntime runtime, OnlineModelAdmission admission)
      : policy_(std::move(policy)),
        runtime_(std::move(runtime)), admission_(std::move(admission)) {}

  weighted_grpo::native_policy_v1::Policy policy_;
  OnlineRuntime runtime_;
  OnlineModelAdmission admission_;
};

}  // namespace wc::pipeline_v1
