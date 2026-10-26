#include "protocol.hpp"
#include "harness_dispatch.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  using nlohmann::json;
  using operator_harness_v1::handle_line;

  auto malformed = handle_line("{");
  expect(malformed.at("status") == "failed", "malformed request must fail");

  auto capabilities = handle_line(json{{"schema", "operator-harness-request-v1"},
                                       {"command", "capabilities"},
                                       {"case_id", "caps"}}
                                      .dump());
  expect(capabilities.at("status") == "ok", "capabilities request failed");
  expect(capabilities.at("nncp") == "disabled", "NNCP was not disabled");
  bool has_a048 = false;
  for (const auto& action : capabilities.at("actions")) {
    const auto name = action.at("action").get<std::string>();
    expect(name.find("nncp") == std::string::npos, "NNCP leaked into capabilities");
    if (name == "harness.a048.byte_planes.v1") has_a048 = true;
  }
  expect(has_a048, "A048 bridge is missing");

  // Four uint64 offsets, width 9 -> two independently represented planes.
  const json run{{"schema", "operator-harness-request-v1"},
                 {"command", "run"},
                 {"case_id", "a048-roundtrip"},
                 {"warmups", 0},
                 {"repeats", 1},
                 {"roots", {{{"kind", "numeric"},
                              {"width", 8},
                              {"data_hex", "00000000000000000100000000000000ff01000000000000aa00000000000000"},
                              {"original_numeric_payload", true},
                              {"logical_view", "uint64"}}}},
                 {"recipe", {{"policy_registry", "operator-harness-test-v1"},
                              {"steps", {{{"kind", "stage"},
                                          {"inputs", {0}},
                                          {"stage", {{"action", "harness.a048.byte_planes.v1"},
                                                     {"profile", "uint64"},
                                                     {"parameters", {{"bit_width", "9"}}}}}},
                                         {{"kind", "stop"}}}}}}};
  auto encoded = handle_line(run.dump());
  expect(encoded.at("status") == "ok", "A048 run failed");
  expect(encoded.at("bitexact") == true, "A048 run was not bit-exact");

  auto decoded = handle_line(json{{"schema", "operator-harness-request-v1"},
                                  {"command", "decode"},
                                  {"case_id", "a048-cold"},
                                  {"frame_hex", encoded.at("frame_hex")}}
                                 .dump());
  expect(decoded.at("status") == "ok", "A048 cold decode failed");
  expect(decoded.at("roots").at(0).at("data_hex") ==
             run.at("roots").at(0).at("data_hex"),
         "A048 cold decode differed");

  auto zero_width = run;
  zero_width["case_id"] = "a048-zero-width";
  zero_width["roots"][0]["data_hex"] =
      "0000000000000000000000000000000000000000000000000000000000000000";
  zero_width["recipe"]["steps"][0]["stage"]["parameters"]["bit_width"] = "0";
  auto zero_encoded = handle_line(zero_width.dump());
  expect(zero_encoded.at("status") == "ok", "A048 zero-width run failed");
  auto zero_decoded = handle_line(json{{"schema", "operator-harness-request-v1"},
                                       {"command", "decode"},
                                       {"case_id", "a048-zero-width-cold"},
                                       {"frame_hex", zero_encoded.at("frame_hex")}}
                                      .dump());
  expect(zero_decoded.at("status") == "ok" &&
             zero_decoded.at("roots").at(0).at("data_hex") ==
                 zero_width.at("roots").at(0).at("data_hex"),
         "A048 zero-width carrier did not cold-decode bit-exactly");

  const json split_concat{{"schema", "operator-harness-request-v1"},
                          {"command", "run"},
                          {"case_id", "for-split-concat-pack"},
                          {"warmups", 0},
                          {"repeats", 1},
                          {"roots", run.at("roots")},
                          {"recipe", {{"policy_registry", "operator-harness-test-v1"},
                                      {"steps", {
                                          {{"kind", "stage"}, {"inputs", {0}},
                                           {"stage", {{"action", "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1"},
                                                      {"profile", "int64"}, {"parameters", json::object()}}}},
                                          {{"kind", "split_segments"}, {"inputs", {2}},
                                           {"segment_sizes", {2, 2}}},
                                          {{"kind", "concat"}, {"inputs", {3, 4}}},
                                          {{"kind", "stage"}, {"inputs", {5}},
                                           {"stage", {{"action", "op.bitpack_int.max_value_width.cpp_v1"},
                                                      {"profile", "uint64"}, {"parameters", json::object()}}}},
                                          {{"kind", "stop"}}
                                      }}}}};
  auto chained = handle_line(split_concat.dump());
  expect(chained.at("status") == "ok" && chained.at("bitexact") == true,
         "FOR/Split/Concat/BitPack typed chain failed");

  auto denied = run;
  denied["case_id"] = "nncp-denied";
  denied["recipe"]["steps"][0]["stage"]["action"] =
      "op.prediction_probability.nncp.provider_v1";
  denied["recipe"]["steps"][0]["stage"]["profile"] = "";
  denied["recipe"]["steps"][0]["stage"]["parameters"] = json::object();
  auto denied_response = handle_line(denied.dump());
  expect(denied_response.at("status") == "failed", "NNCP request did not fail");
  expect(denied_response.at("error").get<std::string>().find("NNCP_DISABLED_BY_HARNESS") !=
             std::string::npos,
         "NNCP request failed for the wrong reason");

  // The failed NNCP request did not poison the JSONL service.
  auto after_failure = handle_line(json{{"schema", "operator-harness-request-v1"},
                                        {"command", "capabilities"},
                                        {"case_id", "after-failure"}}
                                       .dump());
  expect(after_failure.at("status") == "ok", "failure poisoned later request");

  // Inverse metadata is untrusted frame content. Reject negative and huge
  // counts before the byte-plane join can allocate a reconstructed vector.
  operator_harness_v1::HarnessDispatcher dispatcher;
  const native_graph_v1::StageRequest a048{
      operator_harness_v1::kBytePlaneAction, "uint64", {{"bit_width", "0"}}};
  const std::vector<native_graph_v1::Stream> empty_carrier{
      {native_graph_v1::Kind::Serial, 1, {}}};
  auto expect_bad_metadata = [&](const json& metadata) {
    const auto text = metadata.dump();
    try {
      (void)dispatcher.inverse(
          a048, empty_carrier,
          {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
    } catch (const std::exception&) {
      return;
    }
    throw std::runtime_error("malformed A048 metadata was accepted");
  };
  expect_bad_metadata({{"schema", "harness-a048-metadata-v1"},
                       {"count", -1}, {"bit_width", 0},
                       {"semantic_plane_count", 0}, {"zero_plane_carrier", true}});
  const native_graph_v1::Stream one_word{native_graph_v1::Kind::Numeric, 8,
                                          std::vector<std::uint8_t>(8, 0)};
  const auto for_ports = dispatcher.forward(
      {"op.frame_of_reference.minimum.int64_uint64_le.cpp_v1", "int64", {}},
      {one_word});
  expect(for_ports.output_logical_views == std::vector<std::string>({"int64", "uint64"}),
         "FOR semantic output views were not annotated");
  const auto zigzag_port = dispatcher.forward(
      {"op.zigzag.signed64.raw.cpp_v2", "int64", {}}, {one_word});
  expect(zigzag_port.output_logical_views == std::vector<std::string>({"uint64"}),
         "ZigZag semantic output view was not annotated");
  const auto xor_ports = dispatcher.forward(
      {"op.xor.previous_word.semantic_ports.cpp_v1", "uint64", {}}, {one_word});
  expect(xor_ports.output_logical_views ==
             std::vector<std::string>({"uint64", "uint64"}),
         "semantic XOR output views were not annotated");
  expect_bad_metadata({{"schema", "harness-a048-metadata-v1"},
                       {"count", 33554433ULL}, {"bit_width", 0},
                       {"semantic_plane_count", 0}, {"zero_plane_carrier", true}});
  try {
    (void)dispatcher.inverse(
        {"op.prediction_probability.nncp.provider_v1", "", {}}, empty_carrier, {});
    throw std::runtime_error("NNCP inverse was accepted");
  } catch (const std::invalid_argument& error) {
    expect(std::string(error.what()).find("NNCP_DISABLED_BY_HARNESS") != std::string::npos,
           "NNCP inverse failed for the wrong reason");
  }
  std::cout << "operator_harness_v1 protocol test passed\n";
}
