#include "catalog.hpp"
#include "stage_dispatch.hpp"
#include "task.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;

namespace {
using graph::Bytes;

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

template<class T> Bytes raw(std::initializer_list<T> values) {
  Bytes out;
  for (const auto value : values) {
    const auto bits = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
    out.insert(out.end(), bits.begin(), bits.end());
  }
  return out;
}

ngpipe::TaskRequest request(std::string id, graph::RootInput root) {
  ngpipe::TaskRequest out;
  out.task_id = std::move(id);
  out.roots = {std::move(root)};
  out.block_index = 0;
  out.config.timing_warmups = 0;
  out.config.timing_repetitions = 1;
  return out;
}

void verify_roundtrip(const ngpipe::GraphTask& task, const ngpipe::TaskState& state,
                      const graph::StageDispatcher& dispatcher) {
  const auto frame = graph::encode(task.request().roots, task.compile_recipe(state.recipe),
                                   dispatcher, task.request().config.graph_limits);
  require(task.certified_frame_byte_lower_bound(state) <= frame.size(),
          "certified NGF1 byte lower bound exceeded an actual complete frame");
  const auto restored = graph::decode(frame, dispatcher, task.request().config.graph_limits);
  require(restored.roots.size() == task.request().roots.size(), "root count changed");
  for (std::size_t i = 0; i < restored.roots.size(); ++i)
    require(restored.roots[i].stream == task.request().roots[i].stream &&
            restored.roots[i].logical_view == task.request().roots[i].logical_view &&
            restored.roots[i].original_numeric_payload ==
                task.request().roots[i].original_numeric_payload,
            "roundtrip changed root " + std::to_string(i));
}

std::size_t verify_all_masked_transitions(const ngpipe::GraphTask& task,
                                          const ngpipe::TaskState& state,
                                          const graph::StageDispatcher& dispatcher,
                                          bool roundtrip) {
  std::size_t checked = 0;
  for (const auto action : task.legal_actions(state)) {
    if (action == ngpipe::stop_action) continue;
    try {
      const auto next = task.transition(state, action);
      require(next.has_value(), "transition returned no state");
      if (roundtrip && task.is_terminal_evaluable(*next))
        verify_roundtrip(task, *next, dispatcher);
    } catch (const std::exception& error) {
      throw std::runtime_error("masked action failed transition/roundtrip: " +
          std::to_string(action) + " " + task.action_record(action).key + ": " +
          error.what());
    }
    ++checked;
  }
  return checked;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: native_pipeline_v1_action_safety CATALOG");
    const auto catalog = ngpipe::ActionCatalog::load(std::filesystem::path(argv[1]));
    auto dispatcher = graph::make_default_dispatcher();
    std::size_t checked = 0;

    const std::vector<ngpipe::TaskRequest> requests{
        request("int64", {{graph::Kind::Numeric, 8,
            raw<std::int64_t>({0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047})},
            true, "int64"}),
        request("int64-negative", {{graph::Kind::Numeric, 8,
            raw<std::int64_t>({INT64_MIN, -4096, -17, -1, 0, 1, 17, 4096, INT64_MAX})},
            true, "int64"}),
        request("float64", {{graph::Kind::Numeric, 8,
            raw<std::uint64_t>({0, 0x8000000000000000ULL, 0x3ff0000000000000ULL,
              0xbff0000000000000ULL, 0x7ff8000000000042ULL, 0x7ff0000000000000ULL,
              0xfff0000000000000ULL, 1})}, true, "float64"}),
        request("float32", {{graph::Kind::Numeric, 4,
            raw<std::uint32_t>({0, 0x80000000U, 0x3f800000U, 0xbf800000U,
              0x7fc00042U, 0x7f800000U, 0xff800000U, 1})}, true, "float32"}),
        request("uint32", {{graph::Kind::Numeric, 4,
            raw<std::uint32_t>({0, 1, 3, 7, 15, 31, 63, 127, 255, 1023})}, true, "uint32"})};

    for (const auto& req : requests) {
      ngpipe::GraphTask task(catalog, *dispatcher, req);
      checked += verify_all_masked_transitions(task, task.initial_state(), *dispatcher, true);
    }

    // Camel's int64 adapter is defined only when every integer survives the
    // policy-visible int64 -> binary64 conversion exactly.  Check this in
    // integer space: INT64_MAX converts to 2^63 and must never reach an
    // implementation-defined/undefined double-to-int64 round trip.
    {
      ngpipe::GraphTask small(catalog, *dispatcher, requests.front());
      const auto legal = small.legal_actions(small.initial_state());
      require(std::find(legal.begin(), legal.end(), 5) != legal.end(),
              "Camel int64 was masked for exactly representable input");
      const auto encoded = small.transition(small.initial_state(), 5);
      require(encoded.has_value(), "Camel int64 exact transition failed");
      verify_roundtrip(small, *encoded, *dispatcher);

      auto exact_boundary_request = request("camel-int64-exact-boundary",
          {{graph::Kind::Numeric, 8,
            raw<std::int64_t>({INT64_MIN, INT64_MAX - 1023})}, true, "int64"});
      ngpipe::GraphTask exact_boundary(
          catalog, *dispatcher, exact_boundary_request);
      const auto exact_boundary_legal =
          exact_boundary.legal_actions(exact_boundary.initial_state());
      require(std::find(exact_boundary_legal.begin(), exact_boundary_legal.end(), 5) !=
                  exact_boundary_legal.end(),
              "Camel int64 masked an exactly representable upper boundary");
      const auto boundary_encoded =
          exact_boundary.transition(exact_boundary.initial_state(), 5);
      require(boundary_encoded.has_value(),
              "Camel int64 exact boundary transition failed");
      verify_roundtrip(exact_boundary, *boundary_encoded, *dispatcher);

      ngpipe::GraphTask boundary(catalog, *dispatcher, requests[1]);
      const auto boundary_legal = boundary.legal_actions(boundary.initial_state());
      require(std::find(boundary_legal.begin(), boundary_legal.end(), 5) ==
                  boundary_legal.end(),
              "Camel int64 exposed an inexact binary64 boundary value");
    }

    // Audit deeper append masks after a multi-output node, a continuous split,
    // and a concat.  These states exercise target-slot expansion, active-flow
    // bounds, physical/logical view changes, and non-root stage preconditions.
    {
      ngpipe::GraphTask task(catalog, *dispatcher, requests.front());
      auto state = task.transition(task.initial_state(), 91);
      require(state.has_value(), "A013 multi-output setup failed");
      checked += verify_all_masked_transitions(task, *state, *dispatcher, true);
      state = task.transition(*state, 389);  // half split of target slot 1
      require(state.has_value(), "continuous split setup failed");
      checked += verify_all_masked_transitions(task, *state, *dispatcher, true);
      state = task.transition(*state, 6241);  // concat active slots 0 and 1
      require(state.has_value(), "concat setup failed");
      checked += verify_all_masked_transitions(task, *state, *dispatcher, true);
    }

    // Exercise target-slot selection with a serial auxiliary stream present.
    auto mixed = requests.front();
    mixed.task_id = "numeric-plus-mask";
    mixed.roots.push_back({{graph::Kind::Serial, 1, Bytes{0, 1, 0, 1, 1, 0, 0, 1}},
                           false, "bytes"});
    {
      ngpipe::GraphTask task(catalog, *dispatcher, mixed);
      checked += verify_all_masked_transitions(task, task.initial_state(), *dispatcher, true);
    }


    // At the 32-flow boundary, every action that would add a port is masked,
    // while Concat remains legal and can free capacity for a subsequent split.
    auto saturated = requests.front();
    saturated.task_id = "active-flow-boundary";
    saturated.roots.clear();
    for (unsigned i = 0; i < 32; ++i)
      saturated.roots.push_back({{graph::Kind::Numeric, 8,
          raw<std::int64_t>({static_cast<std::int64_t>(i), static_cast<std::int64_t>(i + 1)})},
          true, "int64"});
    {
      ngpipe::GraphTask task(catalog, *dispatcher, saturated);
      const auto legal = task.legal_actions(task.initial_state());
      require(std::find(legal.begin(), legal.end(), 91) == legal.end(),
              "multi-output action exceeded active-flow bound");
      require(std::find(legal.begin(), legal.end(), 194) == legal.end(),
              "split exceeded active-flow bound");
      require(std::find(legal.begin(), legal.end(), 6241) != legal.end(),
              "concat was masked at active-flow bound");
      const auto joined = task.transition(task.initial_state(), 6241);
      require(joined.has_value(), "concat failed to free one active-flow slot");
      const auto after = task.legal_actions(*joined);
      require(std::find(after.begin(), after.end(), 194) != after.end(),
              "split did not reuse capacity freed by concat");
    }

    // Enumerate every token exposed by continuation edit masks, including all
    // legal operators after an insert/replace position.  A masked transition
    // must succeed and every completed edit must form an independently
    // decodable frame.
    auto continuation = requests.front();
    continuation.task_id = "continuation";
    continuation.block_index = 1;
    continuation.continuation_edit_mode = true;
    continuation.previous_recipe = {91};
    ngpipe::GraphTask edit_task(catalog, *dispatcher, continuation);
    for (const auto edit_type : edit_task.legal_actions(edit_task.initial_state())) {
      if (edit_type == ngpipe::stop_action) continue;
      const auto typed = edit_task.transition(edit_task.initial_state(), edit_type);
      require(typed.has_value(), "masked edit type failed transition");
      ++checked;
      for (const auto position : edit_task.legal_actions(*typed)) {
      const auto positioned = edit_task.transition(*typed, position);
      require(positioned.has_value(), "masked edit position failed transition");
      ++checked;
        if (edit_task.is_terminal_evaluable(*positioned)) {
          verify_roundtrip(edit_task, *positioned, *dispatcher);
          continue;
        }
        try {
          checked += verify_all_masked_transitions(edit_task, *positioned, *dispatcher, true);
        } catch (const std::exception& error) {
          throw std::runtime_error("continuation operator mask failed after edit_type=" +
              std::to_string(edit_type) + " position=" + std::to_string(position) + ": " +
              error.what());
        }
      }
    }

    auto parameter_edit = request("parameter-change", {{graph::Kind::Numeric, 8,
        raw<double>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})}, true, "float64"});
    parameter_edit.block_index = 1;
    parameter_edit.continuation_edit_mode = true;
    parameter_edit.previous_recipe = {6};
    ngpipe::GraphTask parameter_task(catalog, *dispatcher, parameter_edit);
    const auto parameter_type = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) { return row.edit_kind == "parameter_change"; })->action_id;
    const auto parameter_position = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) {
          return row.category == ngpipe::ActionCategory::EditPosition && row.parameter_value == 0;
        })->action_id;
    auto parameter_prefix = parameter_task.transition(parameter_task.initial_state(), parameter_type);
    require(parameter_prefix.has_value(), "parameter-change edit type was masked");
    parameter_prefix = parameter_task.transition(*parameter_prefix, parameter_position);
    require(parameter_prefix.has_value(), "parameter-change position was masked");
    checked += verify_all_masked_transitions(
        parameter_task, *parameter_prefix, *dispatcher, true);

    // A recipe that was valid on the preceding block may be data-inapplicable
    // on the current block.  Its failed evaluation consumes one of the 16
    // complete-candidate slots, and two completed local edits may pass through
    // an invalid intermediate recipe before repairing it.
    auto invalid = requests[1];
    invalid.task_id = "invalid-cross-block";
    invalid.block_index = 1;
    invalid.continuation_edit_mode = true;
    invalid.previous_recipe = {3, 4};  // Elf and Camel reject the special values fixture.
    ngpipe::GraphTask invalid_task(catalog, *dispatcher, invalid);
    require(!invalid_task.initial_state().recipe_valid, "invalid previous recipe was not retained");
    const auto invalid_cache = invalid_task.replay_cache_stats();
    require(invalid_cache.misses == 1 && invalid_cache.entries == 0 &&
                invalid_cache.retained_bytes == 0,
            "failed recipe polluted the immutable replay cache");
    const auto failed = invalid_task.evaluate_terminal(invalid_task.initial_state());
    require(!failed.success && failed.candidate_ordinal == 1 && failed.frame.empty() &&
            failed.compression_samples_ns.empty() && failed.decompression_samples_ns.empty(),
            "invalid previous recipe fabricated a measured terminal");
    const auto delete_type = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) { return row.edit_kind == "delete"; })->action_id;
    const auto position_zero = std::find_if(catalog.actions().begin(), catalog.actions().end(),
        [](const auto& row) {
          return row.category == ngpipe::ActionCategory::EditPosition && row.parameter_value == 0;
        })->action_id;
    auto edit = invalid_task.transition(invalid_task.initial_state(), delete_type);
    require(edit.has_value(), "invalid recipe could not begin first repair");
    auto first_repair = invalid_task.transition(*edit, position_zero);
    require(first_repair.has_value() && !first_repair->recipe_valid && first_repair->edit_count == 1,
            "first repair could not retain an invalid intermediate recipe");
    edit = invalid_task.transition(*first_repair, delete_type);
    require(edit.has_value(), "invalid intermediate could not begin second repair");
    auto second_repair = invalid_task.transition(*edit, position_zero);
    require(second_repair.has_value() && second_repair->recipe_valid && second_repair->edit_count == 2,
            "second repair did not restore a valid recipe");
    const auto repaired = invalid_task.evaluate_terminal(*second_repair);
    require(repaired.success && repaired.candidate_ordinal == 2,
            "repaired recipe did not produce the second measured candidate");
    verify_roundtrip(invalid_task, *second_repair, *dispatcher);

    // A native benchmark rejection after a recipe replay succeeds is a
    // charged, structured candidate failure. It must not terminate the whole
    // search or fabricate a partial frame/timing sample.
    auto frame_limited = requests.front();
    frame_limited.task_id = "benchmark-frame-limit-failure";
    frame_limited.config.graph_limits.max_frame_bytes = 128;
    frame_limited.config.graph_limits.max_metadata_bytes = 128;
    ngpipe::GraphTask frame_limited_task(catalog, *dispatcher, frame_limited);
    const auto frame_failure = frame_limited_task.evaluate_terminal(
        frame_limited_task.initial_state());
    require(!frame_failure.success && frame_failure.candidate_ordinal == 1 &&
                frame_failure.failure_error.starts_with(
                    "NATIVE_CANDIDATE_BENCHMARK_FAILED: ") &&
                frame_failure.frame.empty() &&
                frame_failure.compression_samples_ns.empty() &&
                frame_failure.decompression_samples_ns.empty(),
            "native benchmark exception was not a structured failed candidate");

    std::cout << "native_pipeline_v1 masked-action safety passed; transitions=" << checked << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
