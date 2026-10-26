#include "online.hpp"
#include "stage_dispatch.hpp"
#include "test_fixture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;
namespace np = weighted_grpo::native_policy_v1;
using pipeline_test::Bytes;
using pipeline_test::put;
using pipeline_test::text;
using pipeline_test::write;

namespace {

void check(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

std::string action_ids(const std::vector<std::uint32_t>& ids) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) out << ',';
    out << ids[i];
  }
  out << ']';
  return out.str();
}

std::string digest_hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    out.push_back(digits[byte >> 4]);
    out.push_back(digits[byte & 15]);
  }
  return out;
}

void model_fixture(const std::filesystem::path& path,
                   const std::map<std::uint32_t, double>& biases,
                   bool legacy_identity = false,
                   bool migrated_identity = false,
                   bool semantic_mask_identity = true) {
  const auto model_action_count = legacy_identity
      ? ngpipe::action_count : ngpipe::online_action_count;
  const auto model_stop_action = legacy_identity
      ? ngpipe::stop_action : ngpipe::flow_stop_begin;
  Bytes body;
  text(body, np::checkpoint_schema);
  text(body, std::string(64, 'a'));
  text(body, std::string(64, 'b'));
  text(body, legacy_identity ? ngpipe::registry_fingerprint
                             : ngpipe::online_registry_fingerprint);
  text(body, np::feature_schema);
  for (auto n : {model_action_count, model_stop_action, 18U, 28U, 128U, 128U})
    put(body, n, 4);
  for (unsigned i = 0; i < 2; ++i) {
    put(body, 1 + unsigned(migrated_identity) +
                  unsigned(!legacy_identity && semantic_mask_identity &&
                           i == 0), 4);
    text(body, "SYNTHETIC_ONLINE_CONTRACT_ONLY");
    text(body, "not_a_trained_model");
    if (migrated_identity) {
      text(body, "online_control_migration");
      text(body, ngpipe::online_registry_fingerprint);
    }
    if (!legacy_identity && semantic_mask_identity && i == 0) {
      text(body, "semantic_mask_contract");
      text(body, ngpipe::online_semantic_mask_contract_sha256);
    }
  }
  const std::array<std::vector<std::uint32_t>, 6> shapes{{
      {49, 128}, {128}, {128, 128}, {128},
      {128, model_action_count}, {model_action_count}}};
  put(body, 6, 4);
  for (unsigned parameter = 0; parameter < shapes.size(); ++parameter) {
    const auto& shape = shapes[parameter];
    put(body, shape.size(), 4);
    std::uint64_t count = 1;
    for (auto dimension : shape) {
      put(body, dimension, 4);
      count *= dimension;
    }
    put(body, count, 8);
    for (std::uint64_t i = 0; i < count; ++i) {
      double value = 0;
      if (parameter == 5) {
        const auto found = biases.find(static_cast<std::uint32_t>(i));
        if (found != biases.end()) value = found->second;
      }
      put(body, std::bit_cast<std::uint64_t>(value), 8);
    }
  }
  Bytes all{'W', 'G', 'P', 'N', 'A', 'T', '1', 0};
  put(all, body.size(), 8);
  all.insert(all.end(), body.begin(), body.end());
  const auto digest = np::sha256(all);
  all.insert(all.end(), digest.begin(), digest.end());
  write(path, all);
}

void credential_fixture(const std::filesystem::path& model,
                        const std::filesystem::path& path,
                        const std::filesystem::path& catalog,
                        const std::filesystem::path& profile) {
  const auto model_bytes = pipeline_test::read(model);
  const auto model_sha = digest_hex(np::sha256(model_bytes));
  const auto catalog_sha = digest_hex(np::sha256(pipeline_test::read(catalog)));
  const auto profile_sha = digest_hex(np::sha256(pipeline_test::read(profile)));
  const auto quoted = [](const std::string& value) {
    return "\"" + value + "\"";
  };
  const std::string json =
      "{" +
      quoted("action_count") + ":6373," +
      quoted("candidate_evaluations") + ":0," +
      quoted("catalog_sha256") + ":" + quoted(catalog_sha) + "," +
      quoted("checkpoint_export_receipt_sha256") + ":" +
          quoted(std::string(64, 'd')) + "," +
      quoted("control_schema") + ":" +
          quoted(ngpipe::online_control_schema) + "," +
      quoted("feature_schema") + ":" +
          quoted(np::feature_schema) + "," +
      quoted("flow_stop_begin") + ":6341," +
      quoted("flow_stop_count") + ":32," +
      quoted("experiment_profile_sha256") + ":" + quoted(profile_sha) + "," +
      quoted("max_active_streams") + ":32," +
      quoted("max_nodes") + ":64," +
      quoted("max_ordinary_depth") + ":10," +
      quoted("max_policy_steps") + ":10," +
      quoted("native_model_sha256") + ":" + quoted(model_sha) + "," +
      quoted("online_control_migration") + ":false," +
      quoted("per_flow_stop") + ":true," +
      quoted("registry_fingerprint") + ":" +
          quoted(ngpipe::online_registry_fingerprint) + "," +
      quoted("replayed_prefix_stages") + ":0," +
      quoted("runtime_schema") + ":" +
          quoted(ngpipe::online_runtime_schema) + "," +
      quoted("semantic_mask_contract") + ":" +
          quoted(ngpipe::online_semantic_mask_contract) + "," +
      quoted("schema") + ":" +
          quoted(ngpipe::online_matched_credential_schema) + "," +
      quoted("source_checkpoint_sha256") + ":" +
          quoted(std::string(64, 'b')) + "," +
      quoted("starts_from_raw_roots") + ":true," +
      quoted("training_contract") + ":" +
          quoted(ngpipe::online_matched_training_contract) + "," +
      quoted("training_run_receipt_sha256") + ":" +
          quoted(std::string(64, 'c')) + "," +
      quoted("training_session_schema") + ":" +
          quoted(ngpipe::online_training_session_schema) + "}";
  write(path, Bytes(json.begin(), json.end()));
}

template <class T>
Bytes raw(std::initializer_list<T> values) {
  Bytes out;
  for (const auto value : values) {
    const auto bits = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
    out.insert(out.end(), bits.begin(), bits.end());
  }
  return out;
}

class CountingDispatcher final : public graph::StageDispatcher {
 public:
  CountingDispatcher()
      : delegate_(graph::make_default_dispatcher()) {}

  std::string identity() const override { return delegate_->identity(); }

  graph::StageResult forward(
      const graph::StageRequest& request,
      const std::vector<graph::Stream>& inputs) const override {
    ++forward_calls;
    forward_actions.push_back(request.action);
    if (fail_action && request.action == *fail_action)
      throw std::runtime_error("injected selected-stage failure");
    return delegate_->forward(request, inputs);
  }

  std::vector<graph::Stream> inverse(
      const graph::StageRequest& request,
      const std::vector<graph::Stream>& outputs,
      std::span<const std::uint8_t> metadata) const override {
    ++inverse_calls;
    return delegate_->inverse(request, outputs, metadata);
  }

  mutable std::uint32_t forward_calls = 0;
  mutable std::uint32_t inverse_calls = 0;
  mutable std::vector<std::string> forward_actions;
  std::optional<std::string> fail_action;

 private:
  std::unique_ptr<graph::StageDispatcher> delegate_;
};

ngpipe::OnlineRequest single_int64_request() {
  ngpipe::OnlineRequest request;
  request.roots = {{{graph::Kind::Numeric, 8,
                     raw<std::int64_t>({0, 1, 3, 7, 15, 31, 63, 127,
                                        255, 511, 1023, 2047})},
                    true, "int64"}};
  request.max_policy_steps = 1;
  return request;
}

void verify_roots(const graph::Decoded& decoded,
                  const ngpipe::OnlineRequest& request) {
  check(decoded.roots.size() == request.roots.size(),
        "cold decode root count changed");
  for (std::size_t i = 0; i < request.roots.size(); ++i) {
    const auto& expected = request.roots[i];
    const auto& actual = decoded.roots[i];
    check(expected.stream == actual.stream &&
              expected.logical_view == actual.logical_view &&
              expected.original_numeric_payload ==
                  actual.original_numeric_payload,
          "cold decode changed root bytes or metadata");
  }
}

void verify_success(const ngpipe::OnlinePipeline& pipeline,
                    const ngpipe::OnlineRequest& request,
                    std::uint32_t expected_action,
                    std::uint32_t expected_forward_calls,
                    std::uint32_t expected_open_outputs = 1) {
  CountingDispatcher dispatcher;
  const auto result = pipeline.compress(request, dispatcher);
  std::vector<std::uint32_t> expected_ids{expected_action};
  for (auto remaining = expected_open_outputs; remaining > 0; --remaining)
    expected_ids.push_back(ngpipe::flow_stop_begin + remaining - 1);
  check(result.recipe_action_ids == expected_ids,
        "online policy did not select expected action " +
            std::to_string(expected_action) + " then close its flow: " +
            action_ids(result.recipe_action_ids));
  check(result.counters.policy_calls == 1 &&
            result.counters.selected_policy_actions == 1 &&
            result.counters.closed_flow_count == expected_open_outputs &&
            result.counters.forced_flow_stops == expected_open_outputs,
        "online policy/action counters are wrong");
  check(result.counters.forward_stage_calls == expected_forward_calls &&
            dispatcher.forward_calls == expected_forward_calls,
        "selected native stages did not execute exactly once");
  check(result.counters.candidate_evaluations == 0 &&
            result.counters.replayed_prefix_stages == 0,
        "online path performed candidate evaluation or prefix replay");
  check(dispatcher.inverse_calls == 0,
        "online compression invoked an inverse stage");
  check(result.termination == "budget_closed",
        "max-step termination was not explicit");
  check(result.diagnostic_only,
        "old-checkpoint online run lost its diagnostic-only marker");
  const auto subtiming = result.timing.feature_ns + result.timing.mask_ns +
      result.timing.policy_ns + result.timing.stage_ns +
      result.timing.framing_ns;
  check(result.timing.total_ns >= subtiming,
        "nonoverlapping subtimers exceed the outer online clock");
  check(result.timing.total_ns >= result.timing.recipe_ready_ns &&
            result.timing.recipe_ready_ns >=
                result.timing.feature_ns + result.timing.mask_ns +
                    result.timing.policy_ns + result.timing.stage_ns,
        "recipe-ready boundary does not enclose online decision work");
  check(result.compression_ratio() && result.compression_ns_per_byte(),
        "nonempty numeric input produced N/A metrics");

  const auto decoded = graph::decode(result.frame, dispatcher,
                                     result.effective_graph_limits);
  verify_roots(decoded, request);
  graph::Recipe recipe;
  recipe.policy_registry = ngpipe::online_registry_fingerprint;
  recipe.steps = result.native_steps;
  const auto independently_encoded = graph::encode(
      request.roots, recipe, dispatcher, result.effective_graph_limits);
  check(independently_encoded == result.frame,
        "incremental frame differs from fixed-recipe native encoding");
}

template <class F>
void rejected(F&& fn, const std::string& expected) {
  bool thrown = false;
  try {
    fn();
  } catch (const std::exception& error) {
    thrown = true;
    check(std::string(error.what()).find(expected) != std::string::npos,
          "rejection had unexpected reason: " + std::string(error.what()));
  }
  check(thrown, "invalid online request was accepted; expected rejection "
                    "containing: " + expected);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 4,
          "usage: native_pipeline_v1_online_contract CATALOG PROFILE FIXTURE_DIR");
    const std::filesystem::path catalog_path(argv[1]);
    const std::filesystem::path profile_path(argv[2]);
    const std::filesystem::path fixture_dir(argv[3]);
    std::filesystem::create_directories(fixture_dir);

    auto descriptor_bytes = pipeline_test::read(
        catalog_path.parent_path() / "online_flow_stop_controls_v2.json");
    if (!descriptor_bytes.empty() && descriptor_bytes.back() == '\n')
      descriptor_bytes.pop_back();
    check(std::string(descriptor_bytes.begin(), descriptor_bytes.end()) ==
              ngpipe::online_control_descriptor_canonical_json,
          "online flow-stop descriptor is not canonical");
    check(digest_hex(np::sha256(descriptor_bytes)) ==
              ngpipe::online_registry_fingerprint,
          "online flow-stop descriptor fingerprint changed");

    const auto load = [&](const std::string& name,
                          const std::map<std::uint32_t, double>& biases) {
      const auto path = fixture_dir / (name + ".wgpn");
      model_fixture(path, biases);
      return ngpipe::OnlinePipeline::load(path, catalog_path, profile_path);
    };

    const auto direct = load("direct", {{101, 10.0}});
    verify_success(direct, single_int64_request(), 101, 1);

    // A013 creates multiple active data flows; no candidate run is needed to
    // discover its output ports, and the complete frame cold-decodes exactly.
    const auto branch = load("branch", {{91, 10.0}});
    verify_success(branch, single_int64_request(), 91, 1, 2);

    const auto split = load("split", {{194, 10.0}});
    verify_success(split, single_int64_request(), 194, 0, 2);

    auto concat_request = single_int64_request();
    concat_request.roots.push_back(
        {{graph::Kind::Numeric, 8,
          raw<std::int64_t>({5, 8, 13, 21, 34, 55, 89, 144})},
         true, "int64"});
    const auto concat = load("concat", {{6241, 10.0}});
    verify_success(concat, concat_request, 6241, 0);

    // Flow-close IDs address the current ascending OPEN-flow slots. Close the
    // second root, transform the remaining slot zero once, then close it. The
    // closed root remains in the complete frame and cold decode.
    const auto per_flow = load("per_flow", {
        {ngpipe::flow_stop_begin + 1, 12.0}, {101, 11.0},
        {ngpipe::flow_stop_begin, 10.0}});
    auto per_flow_request = concat_request;
    per_flow_request.max_policy_steps = 3;
    CountingDispatcher per_flow_dispatcher;
    const auto per_flow_result =
        per_flow.compress(per_flow_request, per_flow_dispatcher);
    check(per_flow_result.recipe_action_ids ==
              std::vector<std::uint32_t>{ngpipe::flow_stop_begin + 1, 101,
                                         ngpipe::flow_stop_begin} &&
              per_flow_result.counters.policy_calls == 3 &&
              per_flow_result.counters.selected_policy_actions == 3 &&
              per_flow_result.counters.closed_flow_count == 2 &&
              per_flow_result.counters.forced_flow_stops == 0 &&
              per_flow_result.counters.forward_stage_calls == 1 &&
              per_flow_dispatcher.forward_calls == 1 &&
              per_flow_result.termination == "all_flows_closed",
          "per-flow STOP did not preserve current-open-slot semantics");
    verify_roots(graph::decode(per_flow_result.frame, per_flow_dispatcher,
                               per_flow_result.effective_graph_limits),
                 per_flow_request);

    // Deployment, interactive RL, trace replay and timed forced replay share
    // one OnlineSession transition/mask implementation. A per-flow STOP must
    // close only its addressed slot and leave the other branch observable.
    const auto online_runtime =
        ngpipe::OnlineRuntime::load(catalog_path, profile_path);
    CountingDispatcher session_dispatcher;
    auto session = online_runtime.start(per_flow_request, session_dispatcher);
    const auto initial_state = session.observe();
    check(initial_state.observation.size() == 49 &&
              !std::binary_search(initial_state.legal_action_ids.begin(),
                                  initial_state.legal_action_ids.end(),
                                  ngpipe::stop_action) &&
              std::binary_search(initial_state.legal_action_ids.begin(),
                                 initial_state.legal_action_ids.end(),
                                 ngpipe::flow_stop_begin + 1),
          "shared online session exposed the wrong initial feature/mask");
    session.step(ngpipe::flow_stop_begin + 1);
    check(!session.terminal() && session.observe().open_flow_count == 1 &&
              std::binary_search(session.observe().legal_action_ids.begin(),
                                 session.observe().legal_action_ids.end(), 101),
          "per-flow STOP terminated the whole interactive session");
    session.step(101);
    check(!session.terminal(),
          "interactive graph action unexpectedly terminated the session");
    session.step(ngpipe::flow_stop_begin);
    check(session.terminal(),
          "interactive per-flow STOP did not reach terminal state");
    const auto session_result = session.take_result();
    check(session_result.frame == per_flow_result.frame &&
              session_result.recipe_action_ids ==
                  per_flow_result.recipe_action_ids &&
              session_result.counters.candidate_evaluations == 0 &&
              session_result.counters.replayed_prefix_stages == 0 &&
              !session_result.complete_online_timing,
          "interactive and deployment sessions diverged");

    const std::vector<std::uint32_t> forced_actions{
        ngpipe::flow_stop_begin + 1, 101, ngpipe::flow_stop_begin};
    CountingDispatcher trace_dispatcher;
    const auto trace_replay = per_flow.replay_forced(
        per_flow_request, forced_actions, trace_dispatcher, true);
    check(!trace_replay.reward_timing_eligible &&
              !trace_replay.result.complete_online_timing &&
              trace_replay.transitions.size() == forced_actions.size() &&
              trace_replay.transitions.front().state.observation ==
                  initial_state.observation &&
              trace_replay.transitions.front().state.legal_action_ids ==
                  initial_state.legal_action_ids &&
              trace_replay.result.frame == per_flow_result.frame &&
              trace_replay.result.recipe_action_ids ==
                  per_flow_result.recipe_action_ids,
          "trace replay changed shared features, mask, recipe or frame");

    CountingDispatcher timed_replay_dispatcher;
    const auto timed_replay = per_flow.replay_forced(
        per_flow_request, forced_actions, timed_replay_dispatcher, false);
    check(timed_replay.reward_timing_eligible &&
              timed_replay.transitions.empty() &&
              timed_replay.result.complete_online_timing &&
              timed_replay.result.timing.policy_ns > 0 &&
              timed_replay.result.frame == per_flow_result.frame &&
              timed_replay.result.recipe_action_ids ==
                  per_flow_result.recipe_action_ids &&
              timed_replay.result.counters.candidate_evaluations == 0 &&
              timed_replay.result.counters.replayed_prefix_stages == 0,
          "timed forced replay omitted model time or changed the frame");
    const auto partial_replay = per_flow.replay_forced(
        per_flow_request,
        std::vector<std::uint32_t>{ngpipe::flow_stop_begin + 1},
        timed_replay_dispatcher, false);
    check(!partial_replay.reward_timing_eligible &&
              !partial_replay.result.complete_online_timing &&
              partial_replay.result.termination == "forced_actions_finished",
          "external forced finish was incorrectly admitted as online E_total");
    rejected([&] {
      (void)per_flow.replay_forced(
          per_flow_request,
          std::vector<std::uint32_t>{ngpipe::flow_stop_begin + 1, 101,
                                     ngpipe::flow_stop_begin,
                                     ngpipe::flow_stop_begin},
          timed_replay_dispatcher, false);
    }, "after terminal state");
    rejected([&] {
      (void)online_runtime.replay_forced(
          per_flow_request, std::vector<std::uint32_t>{ngpipe::stop_action},
          timed_replay_dispatcher, true);
    }, "outside the shared online legal set");

    // Closing a lower slot reindexes the remaining open flows. Action 101 is
    // initially masked because slot zero is serial; after closing that flow,
    // the int64 root becomes slot zero, is transformed exactly once, and its
    // serial output is then closed. Both original roots remain recoverable.
    const auto reindex = load("lower_slot_reindex", {
        {101, 20.0}, {ngpipe::flow_stop_begin, 10.0}});
    auto reindex_request = single_int64_request();
    reindex_request.roots.insert(
        reindex_request.roots.begin(),
        {{graph::Kind::Serial, 1, Bytes{4, 3, 2, 1}}, false, "bytes"});
    reindex_request.max_policy_steps = 3;
    CountingDispatcher reindex_dispatcher;
    const auto reindexed =
        reindex.compress(reindex_request, reindex_dispatcher);
    check(reindexed.recipe_action_ids ==
              std::vector<std::uint32_t>{ngpipe::flow_stop_begin, 101,
                                         ngpipe::flow_stop_begin} &&
              reindexed.counters.policy_calls == 3 &&
              reindexed.counters.forced_flow_stops == 0 &&
              reindexed.counters.forward_stage_calls == 1 &&
              reindex_dispatcher.forward_calls == 1,
          "closing a lower flow slot did not reindex the remaining target");
    verify_roots(graph::decode(reindexed.frame, reindex_dispatcher,
                               reindexed.effective_graph_limits),
                 reindex_request);

    // Closed leaves are retained in the wire frame but no longer consume the
    // policy's open-flow budget. With max_open=2, close one of two roots and
    // split the other into two open outputs; the three physical leaves must
    // still serialize and cold-decode under the published effective limits.
    const auto retained_split = load("retained_leaf_split", {
        {ngpipe::flow_stop_begin + 1, 12.0}, {194, 11.0}});
    auto retained_split_request = concat_request;
    retained_split_request.graph_limits.max_active_streams = 2;
    retained_split_request.max_policy_steps = 2;
    CountingDispatcher retained_split_dispatcher;
    const auto retained_split_result = retained_split.compress(
        retained_split_request, retained_split_dispatcher);
    check(retained_split_result.recipe_action_ids ==
              std::vector<std::uint32_t>{ngpipe::flow_stop_begin + 1, 194,
                                         ngpipe::flow_stop_begin + 1,
                                         ngpipe::flow_stop_begin} &&
              retained_split_result.counters.policy_calls == 2 &&
              retained_split_result.counters.closed_flow_count == 3 &&
              retained_split_result.counters.forced_flow_stops == 2 &&
              retained_split_result.effective_graph_limits.max_active_streams >=
                  3,
          "retained closed leaves incorrectly consumed the open-flow budget");
    verify_roots(graph::decode(retained_split_result.frame,
                               retained_split_dispatcher,
                               retained_split_result.effective_graph_limits),
                 retained_split_request);

    // The first policy action makes the second action legal. The latter uses
    // one internal carrier adapter. Delta, adapter and entropy stages must each
    // execute exactly once, with no replay of the Delta prefix.
    const auto adapter = load("adapter", {{171, 10.0}, {107, 9.0}});
    auto adapter_request = single_int64_request();
    adapter_request.max_policy_steps = 2;
    CountingDispatcher adapter_dispatcher;
    const auto adapted = adapter.compress(adapter_request, adapter_dispatcher);
    check(adapted.recipe_action_ids ==
              std::vector<std::uint32_t>{107, 171,
                                         ngpipe::flow_stop_begin},
          "adapter scenario selected the wrong policy sequence");
    check(adapted.counters.policy_calls == 2 &&
              adapted.counters.selected_policy_actions == 2 &&
              adapted.counters.forced_flow_stops == 1 &&
              adapted.counters.forward_stage_calls == 3 &&
              adapter_dispatcher.forward_calls == 3,
          "incremental adapter path repeated or skipped a stage");
    check(adapted.counters.candidate_evaluations == 0 &&
              adapted.counters.replayed_prefix_stages == 0 &&
              adapter_dispatcher.inverse_calls == 0,
          "adapter path performed forbidden search/replay/inverse work");
    verify_roots(graph::decode(adapted.frame, adapter_dispatcher,
                               adapted.effective_graph_limits),
                 adapter_request);

    // A dispatcher rejection for the selected action is surfaced immediately;
    // the policy does not fall through to a lower-ranked legal action.
    CountingDispatcher failed_dispatcher;
    failed_dispatcher.fail_action =
        "op.bitpack_int.max_value_width.cpp_v1";
    rejected([&] { (void)direct.compress(single_int64_request(),
                                           failed_dispatcher); },
             "ONLINE_SELECTED_ACTION_FAILED action_id=101");
    check(failed_dispatcher.forward_calls == 1 &&
              failed_dispatcher.inverse_calls == 0,
          "selected-stage failure triggered a retry or inverse operation");

    // Reference A018 has data-dependent port metadata and emits three real
    // outputs. With max_open=2 it is masked while both roots are open. After
    // the policy closes one root, the conservative two-port precheck admits
    // it; the authoritative three-port result must then fail explicitly
    // before another feature/mask step, without retrying another action.
    const auto reference_overflow = load("reference_output_overflow", {
        {ngpipe::flow_stop_begin + 1, 12.0}, {96, 11.0}});
    auto reference_overflow_request = concat_request;
    reference_overflow_request.graph_limits.max_active_streams = 2;
    reference_overflow_request.max_policy_steps = 2;
    CountingDispatcher reference_overflow_dispatcher;
    rejected([&] {
      (void)reference_overflow.compress(reference_overflow_request,
                                        reference_overflow_dispatcher);
    }, "ONLINE_SELECTED_ACTION_FAILED action_id=96");
    check(reference_overflow_dispatcher.forward_calls == 1 &&
              reference_overflow_dispatcher.inverse_calls == 0,
          "data-dependent output overflow triggered a retry or inverse");

    // Continuation edit tokens remain in the 6,341-ID checkpoint identity but
    // are inapplicable to this append-only online runtime.
    const auto edit = load("edit", {{6272, 10.0},
                                     {ngpipe::flow_stop_begin, 5.0}});
    CountingDispatcher edit_dispatcher;
    const auto edit_result = edit.compress(single_int64_request(), edit_dispatcher);
    check(edit_result.recipe_action_ids ==
              std::vector<std::uint32_t>{ngpipe::flow_stop_begin},
          "append-only online runtime exposed a continuation edit token");

    // A111 has exactly three output ports. With 31 active roots, replacing one
    // root would create 33 active flows, so the cheap structural mask must
    // reject it without a forward probe.
    ngpipe::OnlineRequest saturated;
    saturated.max_policy_steps = 1;
    for (std::uint32_t i = 0; i < 31; ++i)
      saturated.roots.push_back(
          {{graph::Kind::Numeric, 4, raw<std::uint32_t>({i, i + 1})},
           true, "uint32"});
    const auto outlier = load("outlier_capacity", {{185, 10.0},
        {ngpipe::flow_stop_begin, 5.0}});
    CountingDispatcher saturated_dispatcher;
    const auto saturated_result =
        outlier.compress(saturated, saturated_dispatcher);
    check(!saturated_result.recipe_action_ids.empty() &&
              saturated_result.recipe_action_ids.front() ==
                  ngpipe::flow_stop_begin &&
              saturated_result.counters.closed_flow_count == 31 &&
              saturated_dispatcher.forward_calls == 0,
          "A111 capacity mask executed or admitted a three-port overflow");

    // An arbitrarily high legacy global STOP logit is ignored because action
    // zero is absent from the online-v2 legal set.
    const auto legacy_stop_disabled = load(
        "legacy_stop_disabled",
        {{ngpipe::stop_action, 100.0}, {ngpipe::flow_stop_begin, 10.0}});
    CountingDispatcher legacy_stop_dispatcher;
    const auto legacy_stop_result = legacy_stop_disabled.compress(
        single_int64_request(), legacy_stop_dispatcher);
    check(legacy_stop_result.recipe_action_ids ==
              std::vector<std::uint32_t>{ngpipe::flow_stop_begin} &&
              legacy_stop_result.termination == "all_flows_closed",
          "legacy global STOP action leaked into online v2");

    const auto old_model_path = fixture_dir / "old_6341_identity.wgpn";
    model_fixture(old_model_path, {{ngpipe::stop_action, 1.0}}, true);
    rejected([&] {
      (void)ngpipe::OnlinePipeline::load(
          old_model_path, catalog_path, profile_path);
    }, "identity mismatch");

    // A matched credential binds the actual WGPN digest, source checkpoint,
    // v13 training contract and fixed horizon. Missing credentials remain
    // diagnostic, and a horizon mismatch fails before inference.
    const auto matched_credential_path = fixture_dir / "matched.json";
    const auto matched_model_path = fixture_dir / "matched_stop.wgpn";
    model_fixture(matched_model_path, {{ngpipe::flow_stop_begin, 10.0}});
    credential_fixture(matched_model_path, matched_credential_path,
                       catalog_path, profile_path);
    const auto matched = ngpipe::OnlinePipeline::load(
        matched_model_path, catalog_path, profile_path,
        matched_credential_path);
    auto matched_request = single_int64_request();
    matched_request.max_policy_steps =
        ngpipe::online_matched_max_policy_steps;
    CountingDispatcher matched_dispatcher;
    const auto matched_result = matched.compress(
        matched_request, matched_dispatcher);
    check(matched_result.matched_training &&
              !matched_result.diagnostic_only &&
              matched_result.matched_max_policy_steps ==
                  ngpipe::online_matched_max_policy_steps &&
              !matched_result.model_sha256.empty() &&
              matched_result.source_checkpoint_sha256 ==
                  std::string(64, 'b') &&
              !matched_result.matched_credential_sha256.empty(),
          "valid matched-training credential was not propagated");
    rejected([&] {
      (void)matched.compress(single_int64_request(), matched_dispatcher);
    }, "max_policy_steps differs");
    auto mismatched_limits = matched_request;
    mismatched_limits.graph_limits.max_nodes = 63;
    rejected([&] {
      (void)matched.compress(mismatched_limits, matched_dispatcher);
    }, "graph limits differ");

    const auto old_mask_model_path = fixture_dir / "old_mask_contract.wgpn";
    const auto old_mask_credential_path =
        fixture_dir / "old_mask_contract.json";
    model_fixture(old_mask_model_path, {{ngpipe::flow_stop_begin, 10.0}},
                  false, false, false);
    credential_fixture(old_mask_model_path, old_mask_credential_path,
                       catalog_path, profile_path);
    rejected([&] {
      (void)ngpipe::OnlinePipeline::load(
          old_mask_model_path, catalog_path, profile_path,
          old_mask_credential_path);
    }, "data_fingerprints must bind semantic_mask_contract");

    // The diagnostic migration exporter stamps this identity key into both
    // fingerprint sets. A separately written credential cannot relabel it.
    const auto migrated_path = fixture_dir / "migrated_stop_clone.wgpn";
    const auto migrated_credential_path = fixture_dir / "migrated.json";
    model_fixture(migrated_path, {{ngpipe::flow_stop_begin, 1.0}}, false,
                  true);
    credential_fixture(migrated_path, migrated_credential_path,
                       catalog_path, profile_path);
    rejected([&] {
      (void)ngpipe::OnlinePipeline::load(
          migrated_path, catalog_path, profile_path,
          migrated_credential_path);
    }, "diagnostic online-control migration cannot be admitted");

    auto empty_request = single_int64_request();
    empty_request.roots = {
        {{graph::Kind::Serial, 1, Bytes{0, 1, 0, 1}}, false, "bytes"}};
    CountingDispatcher empty_dispatcher;
    const auto empty = direct.compress(empty_request, empty_dispatcher);
    check(empty.original_numeric_bytes == 0 &&
              !empty.compression_ratio() &&
              !empty.compression_ns_per_byte() &&
              empty.counters.policy_calls == 0 &&
              empty.recipe_action_ids ==
                  std::vector<std::uint32_t>{ngpipe::flow_stop_begin} &&
              empty.counters.forced_flow_stops == 1 &&
              empty.termination == "no_original_numeric_payload_n_a",
          "B_raw=0 did not produce explicit N/A/STOP behavior");

    auto invalid_weights = single_int64_request();
    invalid_weights.weights = {1.0, 1.0, 0.0};
    CountingDispatcher invalid_dispatcher;
    rejected([&] { (void)direct.compress(invalid_weights,
                                           invalid_dispatcher); },
             "weights must sum to one");

    auto overfull_roots = concat_request;
    overfull_roots.graph_limits.max_active_streams = 1;
    rejected([&] { (void)direct.compress(overfull_roots,
                                           invalid_dispatcher); },
             "root count exceeds policy open-flow limit");

    for (const auto& row : direct.catalog().actions()) {
      if (!row.implementation_id ||
          (*row.implementation_id != "A109" &&
           *row.implementation_id != "A110"))
        continue;
      check(!direct.catalog().enabled(row.action_id),
            "CMIX/NNCP action remained enabled in the online scope");
    }

    std::cout << "native online sequential-policy contract passed; "
              << "candidate_evaluations=0; replayed_prefix_stages=0; "
              << "default_max_policy_steps=10\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
