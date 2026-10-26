// Persistent JSONL driver for the exact v13 raw-root environment. Interactive
// step latency is never a compression metric because caller/IPC idle time sits
// between steps. Timed forced replay runs uninterrupted inside this process.
#include "online.hpp"
#include "stage_dispatch.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace p = wc::pipeline_v1;
namespace g = native_graph_v1;
namespace pol = weighted_grpo::native_policy_v1;
using J = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

void check(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument(message);
}

std::uint64_t elapsed(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
      .count();
}

void check_semantic_model_identity(const pol::Identity& identity) {
  const auto found = std::find_if(
      identity.data_fingerprints.begin(), identity.data_fingerprints.end(),
      [](const auto& row) { return row.first == "semantic_mask_contract"; });
  check(found != identity.data_fingerprints.end() &&
            found->second == p::online_semantic_mask_contract_sha256,
        "training server model data_fingerprints do not bind the current "
        "semantic_mask_contract");
}

g::Bytes unhex(const std::string& text) {
  check(text.size() % 2 == 0 && text.size() <= 512ULL * 1024 * 1024,
        "invalid hex size");
  const auto digit = [](char value) -> unsigned {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::invalid_argument("invalid lowercase hex");
  };
  g::Bytes output;
  output.reserve(text.size() / 2);
  for (std::size_t i = 0; i < text.size(); i += 2)
    output.push_back((digit(text[i]) << 4) | digit(text[i + 1]));
  return output;
}

std::string hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    output.push_back(digits[byte >> 4]);
    output.push_back(digits[byte & 15]);
  }
  return output;
}

p::OnlineRequest parse_request(const J& doc) {
  p::OnlineRequest request;
  request.weights = doc.at("weights").get<std::array<double, 3>>();
  request.max_policy_steps = doc.value("max_policy_steps", 10U);
  if (doc.contains("graph_limits")) {
    const auto& limits = doc.at("graph_limits");
    request.graph_limits.max_nodes =
        limits.value("max_nodes", request.graph_limits.max_nodes);
    request.graph_limits.max_active_streams = limits.value(
        "max_active_streams", request.graph_limits.max_active_streams);
    request.graph_limits.max_ordinary_depth = limits.value(
        "max_ordinary_depth", request.graph_limits.max_ordinary_depth);
    request.graph_limits.max_stream_bytes = limits.value(
        "max_stream_bytes", request.graph_limits.max_stream_bytes);
    request.graph_limits.max_working_bytes = limits.value(
        "max_working_bytes", request.graph_limits.max_working_bytes);
    request.graph_limits.max_frame_bytes = limits.value(
        "max_frame_bytes", request.graph_limits.max_frame_bytes);
    request.graph_limits.max_metadata_bytes = limits.value(
        "max_metadata_bytes", request.graph_limits.max_metadata_bytes);
  }
  check(doc.at("roots").is_array() && !doc.at("roots").empty() &&
            doc.at("roots").size() <= 32,
        "invalid roots");
  for (const auto& item : doc.at("roots")) {
    const auto kind = item.at("kind").get<std::string>();
    check(kind == "numeric" || kind == "serial" || kind == "struct",
          "invalid root kind");
    request.roots.push_back(
        {{kind == "numeric" ? g::Kind::Numeric
                            : kind == "serial" ? g::Kind::Serial
                                                 : g::Kind::Struct,
          item.at("width").get<std::uint32_t>(),
          unhex(item.at("data_hex").get<std::string>())},
         item.at("original_numeric_payload").get<bool>(),
         item.at("logical_view").get<std::string>()});
  }
  return request;
}

void verify(const g::Decoded& decoded, const p::OnlineRequest& request) {
  check(decoded.roots.size() == request.roots.size(),
        "decoded root count changed");
  std::uint64_t raw = 0;
  for (std::size_t i = 0; i < request.roots.size(); ++i) {
    const auto& expected = request.roots[i];
    const auto& actual = decoded.roots[i];
    check(expected.stream == actual.stream &&
              expected.logical_view == actual.logical_view &&
              expected.original_numeric_payload ==
                  actual.original_numeric_payload,
          "bit-exact root roundtrip failed");
    if (expected.original_numeric_payload) raw += expected.stream.data.size();
  }
  check(decoded.original_numeric_bytes == raw,
        "decoded original numeric denominator changed");
}

J timing(const p::OnlineResult& result) {
  const auto& value = result.timing;
  return {{"recipe_ready_ns", value.recipe_ready_ns},
          {"feature_ns", value.feature_ns},
          {"mask_ns", value.mask_ns},
          {"policy_ns", value.policy_ns},
          {"stage_ns", value.stage_ns},
          {"framing_ns", value.framing_ns},
          {"total_ns", value.total_ns}};
}

J admission(const p::OnlineModelAdmission& value) {
  return {{"matched_training", value.matched_training},
          {"diagnostic_only", value.diagnostic_only},
          {"matched_max_policy_steps", value.matched_max_policy_steps},
          {"model_sha256", value.model_sha256},
          {"source_checkpoint_sha256", value.source_checkpoint_sha256},
          {"matched_credential_sha256", value.credential_sha256}};
}

J observation(const p::OnlineObservation& state, std::uint64_t epoch) {
  return {{"schema", p::online_training_session_schema},
          {"session_epoch", epoch},
          {"terminal", false},
          {"observation", state.observation},
          {"legal_action_ids", state.legal_action_ids},
          {"policy_calls", state.policy_calls},
          {"policy_graph_nodes", state.policy_graph_nodes},
          {"open_flow_count", state.open_flow_count},
          {"action_count", p::online_action_count},
          {"flow_stop_begin", p::flow_stop_begin},
          {"registry_fingerprint", p::online_registry_fingerprint},
          {"candidate_evaluations", 0},
          {"replayed_prefix_stages", 0}};
}

std::string recipe_key(const p::OnlineRequest& request,
                       const p::OnlineResult& result) {
  const auto& limits = request.graph_limits;
  const J canonical{{"registry_fingerprint", p::online_registry_fingerprint},
                    {"max_policy_steps", request.max_policy_steps},
                    {"graph_limits",
                     {{"max_nodes", limits.max_nodes},
                      {"max_active_streams", limits.max_active_streams},
                      {"max_ordinary_depth", limits.max_ordinary_depth},
                      {"max_stream_bytes", limits.max_stream_bytes},
                      {"max_working_bytes", limits.max_working_bytes},
                      {"max_frame_bytes", limits.max_frame_bytes},
                      {"max_metadata_bytes", limits.max_metadata_bytes}}},
                    {"recipe_action_ids", result.recipe_action_ids}};
  const auto encoded = canonical.dump();
  return hex(pol::sha256({
      reinterpret_cast<const std::uint8_t*>(encoded.data()),
      encoded.size()}));
}

J terminal(const p::OnlineRequest& request, const p::OnlineResult& result,
           std::uint64_t epoch,
           const std::vector<std::uint64_t>& encode_samples = {},
           const std::vector<std::uint64_t>& decode_samples = {}) {
  const auto& counters = result.counters;
  return {{"schema", p::online_training_session_schema},
          {"session_epoch", epoch},
          {"terminal", true},
          {"terminal_status",
           result.original_numeric_bytes ? "ok" : "n_a_no_numeric_payload"},
          {"termination", result.termination},
          {"recipe_action_ids", result.recipe_action_ids},
          {"canonical_recipe_key", recipe_key(request, result)},
          {"frame_bytes", result.frame.size()},
          {"frame_hex", hex(result.frame)},
          {"frame_sha256", hex(pol::sha256(result.frame))},
          {"original_numeric_bytes", result.original_numeric_bytes},
          {"encode_samples_ns", encode_samples},
          {"decode_samples_ns", decode_samples},
          {"timing", timing(result)},
          {"complete_online_timing", result.complete_online_timing},
          {"matched_training", result.matched_training},
          {"diagnostic_only", result.diagnostic_only},
          {"matched_max_policy_steps", result.matched_max_policy_steps},
          {"model_sha256", result.model_sha256},
          {"source_checkpoint_sha256", result.source_checkpoint_sha256},
          {"matched_credential_sha256",
           result.matched_credential_sha256},
          {"policy_calls", counters.policy_calls},
          {"selected_policy_actions", counters.selected_policy_actions},
          {"forward_stage_calls", counters.forward_stage_calls},
          {"semantic_mask_flow_scans", counters.semantic_mask_flow_scans},
          {"semantic_mask_value_visits", counters.semantic_mask_value_visits},
          {"candidate_evaluations", counters.candidate_evaluations},
          {"replayed_prefix_stages", counters.replayed_prefix_stages},
          {"closed_flow_count", counters.closed_flow_count},
          {"forced_flow_stops", counters.forced_flow_stops}};
}

J transitions(const std::vector<p::OnlineTransition>& rows) {
  J output = J::array();
  for (const auto& row : rows) {
    J item{{"observation", row.state.observation},
           {"legal_action_ids", row.state.legal_action_ids},
           {"action_id", row.action_id}};
    item["greedy_action_id"] = row.greedy_action_id
        ? J(*row.greedy_action_id) : J(nullptr);
    output.push_back(std::move(item));
  }
  return output;
}

void check_matched_request(const std::optional<p::OnlinePipeline>& pipeline,
                           const p::OnlineRequest& request) {
  if (!pipeline || !pipeline->admission().matched_training) return;
  check(request.max_policy_steps ==
            pipeline->admission().matched_max_policy_steps,
        "matched model max_policy_steps differs from its training credential");
  check(request.graph_limits.max_nodes == p::online_matched_max_nodes &&
            request.graph_limits.max_active_streams ==
                p::online_matched_max_active_streams &&
            request.graph_limits.max_ordinary_depth ==
                p::online_matched_max_ordinary_depth,
        "matched model graph limits differ from its training credential");
}

}  // namespace

int main(int argc, char** argv) try {
  check(argc == 3,
        "usage: native_online_training_server CATALOG EXPERIMENT_PROFILE");
  const std::filesystem::path catalog_path(argv[1]);
  const std::filesystem::path profile_path(argv[2]);
  const auto runtime = p::OnlineRuntime::load(catalog_path, profile_path);
  const auto dispatcher = g::make_default_dispatcher();
  std::optional<p::OnlinePipeline> pipeline;
  std::optional<p::OnlineSession> active;
  std::optional<p::OnlineRequest> active_request;
  std::uint64_t epoch = 0;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    J request_id = nullptr;
    try {
      const auto doc = J::parse(line);
      request_id = doc.value("request_id", J(nullptr));
      const auto operation = doc.at("op").get<std::string>();
      J reply;
      if (operation == "load_model") {
        const auto model_path =
            std::filesystem::path(doc.at("model_path").get<std::string>());
        const auto credential = doc.contains("matched_credential_path")
            ? std::optional<std::filesystem::path>(doc.at(
                  "matched_credential_path").get<std::string>())
            : std::nullopt;
        auto loaded = p::OnlinePipeline::load(
            model_path, catalog_path, profile_path, credential);
        check_semantic_model_identity(loaded.model_identity());
        pipeline.emplace(std::move(loaded));
        reply = admission(pipeline->admission());
        reply["schema"] = p::online_training_session_schema;
        reply["model_loaded"] = true;
      } else if (operation == "reset") {
        active_request = parse_request(doc);
        check_matched_request(pipeline, *active_request);
        active.emplace(runtime.start(*active_request, *dispatcher));
        ++epoch;
        reply = active->terminal()
            ? terminal(*active_request, active->result(), epoch)
            : observation(active->observe(), epoch);
      } else if (operation == "observe") {
        check(active.has_value(), "no active online session");
        reply = active->terminal()
            ? terminal(*active_request, active->result(), epoch)
            : observation(active->observe(), epoch);
      } else if (operation == "step") {
        check(active.has_value(), "no active online session");
        active->step(doc.at("action_id").get<std::uint32_t>());
        reply = active->terminal()
            ? terminal(*active_request, active->result(), epoch)
            : observation(active->observe(), epoch);
      } else if (operation == "finish") {
        check(active.has_value(), "no active online session");
        active->finish();
        reply = terminal(*active_request, active->result(), epoch);
      } else if (operation == "close") {
        active.reset();
        active_request.reset();
        reply = {{"schema", p::online_training_session_schema},
                 {"closed", true}, {"session_epoch", epoch}};
      } else if (operation == "replay") {
        const auto replay_request = parse_request(doc);
        check_matched_request(pipeline, replay_request);
        const auto actions =
            doc.at("forced_action_ids").get<std::vector<std::uint32_t>>();
        const auto warmups = doc.value("timing_warmups", 0U);
        const auto repetitions = doc.value("timing_repetitions", 1U);
        const auto capture = doc.value("capture_transitions", true);
        check(warmups <= 31 && repetitions >= 1 && repetitions <= 1001,
              "invalid timing repetition count");

        // Trace collection is deliberately separate from every measured run.
        // Copying 49D observations and legal sets into JSON is never E_total.
        std::optional<p::OnlineReplayResult> trace;
        if (capture) {
          trace = pipeline
              ? pipeline->replay_forced(replay_request, actions, *dispatcher,
                                        true)
              : runtime.replay_forced(replay_request, actions, *dispatcher,
                                      true);
        }
        for (std::uint32_t i = 0; i < warmups; ++i) {
          auto sample = pipeline
              ? pipeline->replay_forced(replay_request, actions, *dispatcher,
                                        false)
              : runtime.replay_forced(replay_request, actions, *dispatcher,
                                      false);
          verify(g::decode(sample.result.frame, *dispatcher,
                           sample.result.effective_graph_limits),
                 replay_request);
        }
        std::vector<std::uint64_t> encode_samples;
        encode_samples.reserve(repetitions);
        std::optional<p::OnlineResult> selected;
        for (std::uint32_t i = 0; i < repetitions; ++i) {
          auto sample = pipeline
              ? pipeline->replay_forced(replay_request, actions, *dispatcher,
                                        false)
              : runtime.replay_forced(replay_request, actions, *dispatcher,
                                      false);
          check(sample.transitions.empty(),
                "timed forced replay unexpectedly captured transitions");
          encode_samples.push_back(sample.result.timing.total_ns);
          if (!selected) selected = std::move(sample.result);
          else check(selected->frame == sample.result.frame &&
                         selected->recipe_action_ids ==
                             sample.result.recipe_action_ids,
                     "forced replay is nondeterministic");
        }
        if (trace)
          check(trace->result.frame == selected->frame &&
                    trace->result.recipe_action_ids ==
                        selected->recipe_action_ids,
                "trace and timed replay produced different recipes/frames");
        std::vector<std::uint64_t> decode_samples;
        decode_samples.reserve(repetitions);
        for (std::uint32_t i = 0; i < repetitions; ++i) {
          const auto begin = Clock::now();
          const auto decoded = g::decode(selected->frame, *dispatcher,
                                         selected->effective_graph_limits);
          decode_samples.push_back(elapsed(begin, Clock::now()));
          verify(decoded, replay_request);
        }
        reply = terminal(replay_request, *selected, ++epoch,
                         encode_samples, decode_samples);
        reply["transitions"] = trace
            ? transitions(trace->transitions) : J::array();
        reply["reward_timing_eligible"] =
            pipeline.has_value() && selected->complete_online_timing;
        reply["timing_scope"] = !pipeline
            ? "model_free_forced_runtime_not_reward_eligible"
            : selected->complete_online_timing
                ? "matched_policy_forced_action_total"
                : "loaded_policy_external_finish_not_reward_eligible";
        reply["timing_warmups"] = warmups;
        reply["timing_repetitions"] = repetitions;
      } else {
        throw std::invalid_argument("unknown operation: " + operation);
      }
      reply["runtime_schema"] = p::online_runtime_schema;
      reply["semantic_mask_contract"] = p::online_semantic_mask_contract;
      reply["ok"] = true;
      reply["request_id"] = request_id;
      std::cout << reply.dump() << '\n' << std::flush;
    } catch (const std::exception& error) {
      std::cout << J{{"ok", false}, {"request_id", request_id},
                     {"error", error.what()},
                     {"runtime_schema", p::online_runtime_schema},
                     {"semantic_mask_contract", p::online_semantic_mask_contract},
                     {"schema", p::online_training_session_schema}}
                       .dump()
                << '\n' << std::flush;
    }
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
