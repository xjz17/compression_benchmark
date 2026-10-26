#include "online.hpp"
#include "stage_dispatch.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;

namespace {

using Bytes = graph::Bytes;

void check(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

template <class T>
Bytes raw(const std::vector<T>& values) {
  Bytes out;
  out.reserve(values.size() * sizeof(T));
  for (const auto value : values) {
    const auto bytes =
        std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
    out.insert(out.end(), bytes.begin(), bytes.end());
  }
  return out;
}

ngpipe::OnlineRequest request(graph::Kind kind, std::uint32_t width, Bytes data,
                            std::string view) {
  ngpipe::OnlineRequest out;
  out.roots = {{{kind, width, std::move(data)}, true, std::move(view)}};
  return out;
}

bool legal(const ngpipe::OnlineObservation& observation, std::uint32_t action) {
  return std::binary_search(observation.legal_action_ids.begin(),
                            observation.legal_action_ids.end(), action);
}

std::uint32_t action(const ngpipe::ActionCatalog& catalog,
                     const std::string& runtime_action,
                     const std::string& input_view,
                     std::uint32_t target_slot = 0,
                     std::optional<std::int64_t> max_point = std::nullopt) {
  for (const auto& spec : catalog.actions()) {
    if (spec.runtime_action == runtime_action &&
        spec.input_view == input_view &&
        spec.target_slot == target_slot && spec.max_point == max_point)
      return spec.action_id;
  }
  throw std::runtime_error("test could not find required catalog action");
}

std::uint32_t split_action(const ngpipe::ActionCatalog& catalog,
                           std::uint32_t target_slot = 0) {
  for (const auto& spec : catalog.actions())
    if (spec.category == ngpipe::ActionCategory::SplitSegments &&
        spec.target_slot == target_slot && spec.parameter_mode == "half")
      return spec.action_id;
  throw std::runtime_error("test could not find half split action");
}

void verify_raw_shape_masks(const ngpipe::OnlineRuntime& runtime,
                            const graph::StageDispatcher& dispatcher,
                            const std::array<std::uint32_t, 4>& consumers,
                            std::uint32_t max_value_action) {
  const std::vector<std::int64_t> i64{0, 1, -1, 7};
  const std::vector<double> f64{0.0, 1.25, -2.5, 17.75};
  const std::vector<float> f32{0.0F, 1.25F, -2.5F, 17.75F};
  const Bytes bitmap{0x55, 0xaa, 0x00, 0xff, 0x11, 0x22, 0x33, 0x44};
  std::vector<ngpipe::OnlineRequest> requests;
  requests.push_back(request(graph::Kind::Numeric, 8, raw(i64), "int64"));
  requests.push_back(request(graph::Kind::Numeric, 8, raw(f64), "float64"));
  requests.push_back(request(graph::Kind::Numeric, 4, raw(f32), "float32"));
  requests.push_back(request(graph::Kind::Struct, 1, bitmap, "bytes"));
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto session = runtime.start(requests[i], dispatcher);
    const auto& observation = session.observe();
    check(observation.semantic_mask_flow_scans == (i == 1 ? 1U : 0U),
          "semantic mask scanned a non-float64 flow");
    for (const auto consumer : consumers)
      check(!legal(observation, consumer),
            "DBS1 consumer was advertised on a raw root");
    check(legal(observation, max_value_action) == (i < 2),
          "64-bit lane mask disagrees with dispatcher shape contract");
  }
}

void verify_scaling_and_packet_masks(
    const ngpipe::OnlineRuntime& runtime,
    const graph::StageDispatcher& dispatcher,
    const std::array<std::uint32_t, 4>& consumers,
    std::uint32_t scaling0, std::uint32_t dbp_scaling2,
    std::uint32_t simple8b_scaling2, std::uint32_t ts2diff_scaling2,
    std::uint32_t split) {
  const auto exact_request = request(
      graph::Kind::Numeric, 8,
      raw(std::vector<double>{0.0, 1.25, -2.5, 17.75}), "float64");
  {
    auto session = runtime.start(exact_request, dispatcher);
    const auto& before = session.observe();
    check(!legal(before, scaling0) && legal(before, dbp_scaling2),
          "DBP max_point feasibility mask is wrong for exact decimals");
  }
  const std::array<std::uint32_t, 4> producers{
      dbp_scaling2, simple8b_scaling2, dbp_scaling2, ts2diff_scaling2};
  for (std::size_t i = 0; i < consumers.size(); ++i) {
    auto session = runtime.start(exact_request, dispatcher);
    session.step(producers[i]);
    const auto& packet_observation = session.observe();
    check(legal(packet_observation, consumers[i]),
          "scaled-packet consumer missing after its direct producer");
    for (std::size_t j = 0; j < consumers.size(); ++j) {
      const bool same_packet_kind =
          (i == j) || ((i == 0 || i == 2) && (j == 0 || j == 2));
      check(legal(packet_observation, consumers[j]) == same_packet_kind,
            "scaled-packet consumer accepted the wrong packet kind");
    }
    session.step(consumers[i]);
    session.finish();
    const auto result = session.take_result();
    const auto decoded = graph::decode(result.frame, dispatcher,
                                       result.effective_graph_limits);
    check(decoded.roots.size() == exact_request.roots.size(),
          "DBS1 consumer transition changed root count");
    for (std::size_t i = 0; i < decoded.roots.size(); ++i)
      check(decoded.roots[i].stream == exact_request.roots[i].stream &&
                decoded.roots[i].original_numeric_payload ==
                    exact_request.roots[i].original_numeric_payload &&
                decoded.roots[i].logical_view ==
                    exact_request.roots[i].logical_view,
            "DBS1 consumer transition did not cold round-trip");
  }

  // Splitting any scaling packet preserves bytes, but each fragment is no
  // longer a complete packet and must lose its typed provenance.
  for (std::size_t i = 0; i < consumers.size(); ++i) {
    auto session = runtime.start(exact_request, dispatcher);
    session.step(producers[i]);
    session.step(split);
    const auto& after_split = session.observe();
    check(!legal(after_split, consumers[i]),
          "scaled-packet consumer accepted a split packet fragment");
  }

  for (const auto& values : std::vector<std::vector<double>>{
           {-0.0},
           {std::numeric_limits<double>::quiet_NaN()},
           {std::numeric_limits<double>::infinity()},
           {1.0e20},
       }) {
    auto bad = request(graph::Kind::Numeric, 8, raw(values), "float64");
    auto session = runtime.start(bad, dispatcher);
    const auto& observation = session.observe();
    for (const auto& spec : runtime.catalog().actions())
      if (spec.runtime_action == "op.scaling.delta_binary_packed.cpp_v1" &&
          spec.target_slot == 0)
        check(!legal(observation, spec.action_id),
              "DBP scaling advertised a value rejected by its encoder");
  }
}

void verify_cache_invalidation(const ngpipe::OnlineRuntime& runtime,
                               const graph::StageDispatcher& dispatcher,
                               std::uint32_t split) {
  const std::vector<double> values{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  auto two = request(graph::Kind::Numeric, 8, raw(values), "float64");
  two.roots.push_back(two.roots.front());
  auto session = runtime.start(two, dispatcher);
  check(session.observe().semantic_mask_flow_scans == 2,
        "initial semantic mask did not scan each float flow exactly once");
  session.step(ngpipe::flow_stop_begin + 1);
  check(session.observe().semantic_mask_flow_scans == 2,
        "unchanged flow was rescanned after another flow closed");
  session.step(split);
  const auto& after_split = session.observe();
  check(after_split.semantic_mask_flow_scans == 4,
        "new split flow IDs reused stale semantic-mask cache entries");
  check(after_split.semantic_mask_value_visits <= values.size() * 6,
        "semantic feasibility exceeded two value visits per scanned flow");
}

void report_mask_cost(const ngpipe::OnlineRuntime& runtime,
                      const graph::StageDispatcher& dispatcher,
                      std::size_t bytes) {
  std::vector<double> values(bytes / 8);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<double>(i % 1000);
  auto session = runtime.start(
      request(graph::Kind::Numeric, 8, raw(values), "float64"), dispatcher);
  session.observe();
  session.finish();
  const auto result = session.take_result();
  std::cout << "semantic_mask_input_bytes=" << bytes
            << " mask_ns=" << result.timing.mask_ns
            << " flow_scans=" << result.counters.semantic_mask_flow_scans
            << " value_visits="
            << result.counters.semantic_mask_value_visits << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 3,
          "usage: online_semantic_mask_contract CATALOG PROFILE");
    const auto runtime = ngpipe::OnlineRuntime::load(argv[1], argv[2]);
    const auto dispatcher = graph::make_default_dispatcher();
    const auto& catalog = runtime.catalog();
    const std::array<std::uint32_t, 4> consumers{
        action(catalog, "op.bitpack_int.fastpfor_pfor.cpp_v1", "bytes"),
        action(catalog, "op.bitpack_int.simple8b.cpp_v1", "bytes"),
        action(catalog, "op.delta.stream_vbyte.cpp_v1", "bytes"),
        action(catalog, "op.delta.ts_2diff.cpp_v1", "bytes"),
    };
    const auto max_value =
        action(catalog, "op.bitpack_int.max_value_width.cpp_v1", "bytes");
    const auto scaling0 = action(
        catalog, "op.scaling.delta_binary_packed.cpp_v1", "float64", 0, 0);
    const auto scaling2 = action(
        catalog, "op.scaling.delta_binary_packed.cpp_v1", "float64", 0, 2);
    const auto simple8b_scaling2 = action(
        catalog, "op.scaling.exact_decimal.cpp_v2", "float64", 0, 2);
    std::uint32_t ts2diff_scaling2 = 0;
    for (const auto& spec : catalog.actions())
      if (spec.runtime_action == "op.scaling.exact_decimal.cpp_v2" &&
          spec.input_view == "float64" && spec.target_slot == 0 &&
          spec.max_point == 2 && spec.io_variant ==
              "op.scaling.ts_2diff.cpp_v1")
        ts2diff_scaling2 = spec.action_id;
    check(ts2diff_scaling2 != 0,
          "test could not find TS_2DIFF scaling producer");
    const auto split = split_action(catalog);
    verify_raw_shape_masks(runtime, *dispatcher, consumers, max_value);
    verify_scaling_and_packet_masks(runtime, *dispatcher, consumers,
                                    scaling0, scaling2, simple8b_scaling2,
                                    ts2diff_scaling2, split);
    verify_cache_invalidation(runtime, *dispatcher, split);
    report_mask_cost(runtime, *dispatcher, 64U << 10);
    report_mask_cost(runtime, *dispatcher, 256U << 10);
    std::cout << "online_semantic_mask_contract=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
