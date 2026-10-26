#include "protocol.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create test input");
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!output) throw std::runtime_error("cannot write test input");
}

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest, &length,
                 EVP_sha256(), nullptr) != 1 || length != 32)
    throw std::runtime_error("test SHA-256 failed");
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  for (unsigned index = 0; index < length; ++index) {
    result.push_back(alphabet[digest[index] >> 4]);
    result.push_back(alphabet[digest[index] & 15]);
  }
  return result;
}

Json stop_recipe() {
  return {{"policy_registry", "operator-harness-v2-test"},
          {"steps", Json::array({{{"kind", "stop"}}})}};
}

Json for_recipe() {
  return {{"policy_registry", "operator-harness-v2-test"},
          {"steps",
           Json::array(
               {{{"kind", "stage"},
                 {"inputs", Json::array({0})},
                 {"stage",
                  {{"action",
                    "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1"},
                   {"profile", "int64"},
                   {"parameters", Json::object()}}}},
                {{"kind", "stop"}}})}};
}

Json a048_recipe(std::string bit_width) {
  return {{"policy_registry", "operator-harness-tree-v3-test"},
          {"steps",
           Json::array(
               {{{"kind", "stage"},
                 {"inputs", Json::array({0})},
                 {"stage",
                  {{"action", "harness.a048.byte_planes.v1"},
                   {"profile", "uint64"},
                   {"parameters", {{"bit_width", std::move(bit_width)}}}}}},
                {{"kind", "stop"}}})}};
}

Json stage_recipe(std::string action, std::string profile) {
  return {{"policy_registry", "operator-harness-v2"},
          {"steps",
           Json::array(
               {{{"kind", "stage"},
                 {"inputs", Json::array({0})},
                 {"stage", {{"action", std::move(action)},
                             {"profile", std::move(profile)},
                             {"parameters", Json::object()}}}},
                {{"kind", "stop"}}})}};
}

Json xor_pack_recipe(std::string xor_action, std::string pack_action) {
  return {{"policy_registry", "operator-harness-v2"},
          {"steps",
           Json::array(
               {{{"kind", "stage"},
                 {"inputs", Json::array({0})},
                 {"stage", {{"action", std::move(xor_action)},
                             {"profile", "float64"},
                             {"parameters", Json::object()}}}},
                {{"kind", "stage"},
                 {"inputs", Json::array({2})},
                 {"stage", {{"action", std::move(pack_action)},
                             {"profile", "bytes"},
                             {"parameters", Json::object()}}}},
                {{"kind", "stop"}}})}};
}

std::vector<std::uint8_t> words_le(
    std::initializer_list<std::uint64_t> words) {
  std::vector<std::uint8_t> result;
  result.reserve(words.size() * 8);
  for (const auto word : words)
    for (unsigned shift = 0; shift < 64; shift += 8)
      result.push_back(static_cast<std::uint8_t>(word >> shift));
  return result;
}

}  // namespace

int main() {
  using operator_harness_v2::handle_line;
#if defined(__unix__) || defined(__APPLE__)
  const auto pid = std::to_string(::getpid());
#else
  const auto pid = std::string("0");
#endif
  const auto directory =
      fs::temp_directory_path() / ("operator_harness_v2_test_" + pid);
  fs::create_directories(directory);
  const auto input_path = directory / "sample.bin";
  const auto frame_path = directory / "sample.ngf";
  const auto output_path = directory / "restored.bin";
  const fs::path catalog_path(OPERATOR_HARNESS_V2_TEST_CATALOG);
  const auto catalog_sha256 = sha256(read_bytes(catalog_path));
  const fs::path tree_catalog_path(OPERATOR_HARNESS_V3_TEST_CATALOG);
  const auto tree_catalog_sha256 = sha256(read_bytes(tree_catalog_path));

  const std::vector<std::uint8_t> input{
      0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  write_bytes(input_path, input);

  const Json run{{"schema", "operator-harness-request-v2"},
                 {"command", "run"},
                 {"request_id", "a013-real-file"},
                 {"implementation_ids", Json::array({"A013"})},
                 {"catalog_path", catalog_path.string()},
                 {"catalog_sha256", catalog_sha256},
                 {"input_path", input_path.string()},
                 {"initial_kind", "numeric"},
                 {"initial_width", 8},
                 {"input_view", "int64"},
                 {"dtype", "int64"},
                 {"element_count", 5},
                 {"source_byte_width", 8},
                 {"original_numeric_payload", true},
                 {"recipe", for_recipe()},
                 {"warmups", 0},
                 {"repeats", 1},
                 {"frame_path", frame_path.string()}};
  const auto encoded = handle_line(run.dump());
  expect(encoded.at("status") == "ok", "file-backed A013 run failed");
  expect(encoded.at("phase") == "run", "run phase was not reported");
  expect(encoded.at("in_process_bitexact") == true,
         "run was not verified bit-exactly");
  expect(encoded.at("source_schema").at("source_dtype") == "int64",
         "source schema was not charged and reported");
  expect(fs::is_regular_file(frame_path), "frame was not committed");

  const Json decode{{"schema", "operator-harness-request-v2"},
                    {"command", "decode"},
                    {"request_id", "a013-cold"},
                    {"implementation_ids", Json::array({"A013"})},
                    {"catalog_sha256", catalog_sha256},
                    {"frame_path", frame_path.string()},
                    {"output_path", output_path.string()},
                    {"warmups", 0},
                    {"repeats", 1}};
  const auto decoded = handle_line(decode.dump());
  expect(decoded.at("status") == "ok", "file-backed A013 decode failed");
  expect(decoded.at("frame_verified_bitexact") == true,
         "cold decode did not verify the embedded root digest");
  expect(decoded.at("source_schema_present") == true,
         "cold decode lost the source schema root");
  expect(read_bytes(output_path) == input, "cold output differs bit-for-bit");
  expect(decoded.at("output_sha256") == encoded.at("input_sha256"),
         "cold output digest differs from run input digest");

  const auto a048_input_path = directory / "a048.bin";
  const auto a048_frame_path = directory / "a048.ngf";
  const auto a048_output_path = directory / "a048-restored.bin";
  const std::vector<std::uint8_t> a048_input{
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01};
  write_bytes(a048_input_path, a048_input);
  Json a048_run{{"schema", "operator-harness-request-v2"},
                 {"command", "run"},
                 {"request_id", "a048-pure-typed-plane"},
                 {"implementation_ids", Json::array({"A048"})},
                 {"catalog_path", tree_catalog_path.string()},
                 {"catalog_sha256", tree_catalog_sha256},
                 {"input_path", a048_input_path.string()},
                 {"initial_kind", "numeric"},
                 {"initial_width", 8},
                 {"input_view", "uint64"},
                 {"dtype", "uint64"},
                 {"element_count", 4},
                 {"source_byte_width", 8},
                 {"original_numeric_payload", true},
                 {"recipe", a048_recipe("64")},
                 {"warmups", 0},
                 {"repeats", 1},
                 {"frame_path", a048_frame_path.string()}};
  const auto a048_encoded = handle_line(a048_run.dump());
  expect(a048_encoded.at("status") == "ok" &&
             a048_encoded.at("execution_binding").at("execution_variant") ==
                 "pure_typed_plane" &&
             a048_encoded.at("typed_plane_contract").at(
                 "semantic_plane_count") == 8 &&
             a048_encoded.at("typed_plane_contract").at(
                 "total_plane_payload_bytes") == a048_input.size() &&
             a048_encoded.at("typed_plane_contract").at(
                 "inverse_consumes_all_planes") == true &&
             a048_encoded.at("a048_typed_plane_proof").at("per_plane").size() ==
                 8 &&
             a048_encoded.at("a048_typed_plane_proof").at(
                 "expected_shift_bytes_match") == true &&
             a048_encoded.at("a048_typed_plane_proof").at(
                 "inverse_from_outputs_and_shape_metadata_bitexact") == true,
         "A048 pure typed-plane run contract failed");
  const Json a048_decode{{"schema", "operator-harness-request-v2"},
                         {"command", "decode"},
                         {"request_id", "a048-pure-cold"},
                         {"implementation_ids", Json::array({"A048"})},
                         {"catalog_sha256", tree_catalog_sha256},
                         {"recipe_sha256",
                          a048_encoded.at("execution_binding").at(
                              "recipe_sha256")},
                         {"frame_path", a048_frame_path.string()},
                         {"output_path", a048_output_path.string()},
                         {"warmups", 0},
                         {"repeats", 1}};
  const auto a048_decoded = handle_line(a048_decode.dump());
  expect(a048_decoded.at("status") == "ok" &&
             a048_decoded.at("frame_verified_bitexact") == true &&
             a048_decoded.at("typed_plane_contract") ==
                 a048_encoded.at("typed_plane_contract") &&
             read_bytes(a048_output_path) == a048_input,
         "A048 frame-only cold decode failed");

  auto a048_domain = a048_run;
  a048_domain["request_id"] = "a048-domain-rejected";
  a048_domain["recipe"] = a048_recipe("8");
  a048_domain["frame_path"] = (directory / "a048-domain.ngf").string();
  const auto a048_domain_response = handle_line(a048_domain.dump());
  expect(a048_domain_response.at("status") == "domain_rejected" &&
             a048_domain_response.at("reason_category") ==
                 "STRICT_INPUT_DOMAIN_REJECTED" &&
             a048_domain_response.at("reason_code") ==
                 "A048_VALUE_EXCEEDS_BIT_WIDTH",
         "A048 strict input-domain rejection was not classified narrowly");

  auto a048_bad_parameter = a048_run;
  a048_bad_parameter["request_id"] = "a048-invalid-width";
  a048_bad_parameter["recipe"] = a048_recipe("65");
  a048_bad_parameter["frame_path"] =
      (directory / "a048-invalid-width.ngf").string();
  const auto a048_bad_parameter_response =
      handle_line(a048_bad_parameter.dump());
  expect(a048_bad_parameter_response.at("status") == "failed",
         "out-of-domain A048 harness parameter was accepted");

  auto strict_domain_run = run;
  strict_domain_run["catalog_path"] = tree_catalog_path.string();
  strict_domain_run["catalog_sha256"] = tree_catalog_sha256;
  strict_domain_run["original_numeric_payload"] = false;
  strict_domain_run["warmups"] = 0;
  strict_domain_run["repeats"] = 1;

  const auto camel_path = directory / "camel-int64-max.bin";
  write_bytes(camel_path, words_le({0x7fffffffffffffffull}));
  auto camel_domain = strict_domain_run;
  camel_domain["request_id"] = "a004-domain-rejected";
  camel_domain["implementation_ids"] = Json::array({"A004"});
  camel_domain["input_path"] = camel_path.string();
  camel_domain["input_view"] = "int64";
  camel_domain["dtype"] = "int64";
  camel_domain["element_count"] = 1;
  camel_domain["frame_path"] = (directory / "camel.ngf").string();
  camel_domain["recipe"] =
      stage_recipe("op.eraser.camel.cpp_v1", "int64");
  const auto camel_response = handle_line(camel_domain.dump());
  expect(camel_response.at("status") == "domain_rejected" &&
             camel_response.at("reason_category") ==
                 "STRICT_INPUT_DOMAIN_REJECTED" &&
             camel_response.at("reason_code") ==
                 "CAMEL_INT64_NOT_EXACTLY_REPRESENTABLE" &&
             camel_response.at("error") ==
                 "Camel integer is not exactly representable as float64",
         "Camel forward domain rejection classification failed");

  const auto empty_path = directory / "empty.bin";
  write_bytes(empty_path, {});
  auto fpc_empty_run = strict_domain_run;
  fpc_empty_run["request_id"] = "a033-native-empty";
  fpc_empty_run["implementation_ids"] = Json::array({"A033"});
  fpc_empty_run["input_path"] = empty_path.string();
  fpc_empty_run["input_view"] = "float64";
  fpc_empty_run["dtype"] = "float64";
  fpc_empty_run["element_count"] = 0;
  fpc_empty_run["frame_path"] = (directory / "fpc-empty.ngf").string();
  fpc_empty_run["recipe"] = stage_recipe("op.xor.fpc.cpp_v1", "bytes");
  const auto fpc_response = handle_line(fpc_empty_run.dump());
  expect(fpc_response.at("status") == "ok" &&
             fpc_response.at("in_process_bitexact") == true &&
             fpc_response.at("fallback").at("raw_fallback_stage_count") == 0,
         "FPC native empty run failed or used a wrapper");
  auto fpc_empty_decode = decode;
  fpc_empty_decode["request_id"] = "a033-native-empty-decode";
  fpc_empty_decode["implementation_ids"] = Json::array({"A033"});
  fpc_empty_decode["catalog_path"] = tree_catalog_path.string();
  fpc_empty_decode["catalog_sha256"] = tree_catalog_sha256;
  fpc_empty_decode["frame_path"] = fpc_empty_run.at("frame_path");
  const auto fpc_restored = directory / "fpc-empty-restored.bin";
  fpc_empty_decode["output_path"] = fpc_restored.string();
  const auto fpc_decoded = handle_line(fpc_empty_decode.dump());
  expect(fpc_decoded.at("status") == "ok" &&
             fpc_decoded.at("output_bytes") == 0 &&
             fs::is_regular_file(fpc_restored) && read_bytes(fpc_restored).empty(),
         "FPC native empty frame-only decode failed");

  const auto marker_path = directory / "canonical-nan.bin";
  write_bytes(marker_path, words_le({0x7ff8000000000000ull}));
  auto gorilla_domain = strict_domain_run;
  gorilla_domain["request_id"] = "a038-domain-rejected";
  gorilla_domain["implementation_ids"] = Json::array({"A038"});
  gorilla_domain["input_path"] = marker_path.string();
  gorilla_domain["input_view"] = "float64";
  gorilla_domain["dtype"] = "float64";
  gorilla_domain["element_count"] = 1;
  gorilla_domain["frame_path"] = (directory / "gorilla-marker.ngf").string();
  gorilla_domain["recipe"] = xor_pack_recipe(
      "op.xor.gorilla.cpp_v1", "op.trailing_zero_pack.gorilla.cpp_v1");
  const auto gorilla_response = handle_line(gorilla_domain.dump());
  expect(gorilla_response.at("status") == "domain_rejected" &&
             gorilla_response.at("reason_code") ==
                 "GORILLA_RESERVED_MARKER_INPUT" &&
             gorilla_response.at("error") ==
                 "Gorilla input contains reserved end-marker bit pattern",
         "Gorilla forward domain rejection classification failed");

  const auto chimp_input_path = directory / "chimp-marker-sequence.bin";
  const auto chimp_input = words_le(
      {0x7ff8000000000000ull, 0x3ff0000000000000ull,
       0x7ff8000000000000ull, 0x8000000000000000ull,
       0x7ff8000000000000ull});
  write_bytes(chimp_input_path, chimp_input);
  auto chimp_run = strict_domain_run;
  chimp_run["request_id"] = "a039-marker-escape";
  chimp_run["implementation_ids"] = Json::array({"A039"});
  chimp_run["input_path"] = chimp_input_path.string();
  chimp_run["input_view"] = "float64";
  chimp_run["dtype"] = "float64";
  chimp_run["element_count"] = 5;
  chimp_run["frame_path"] = (directory / "chimp-marker.ngf").string();
  chimp_run["recipe"] = xor_pack_recipe(
      "op.xor.chimp128.cpp_v1", "op.trailing_zero_pack.chimp128.cpp_v1");
  const auto chimp_response = handle_line(chimp_run.dump());
  expect(chimp_response.at("status") == "ok" &&
             chimp_response.at("in_process_bitexact") == true,
         "Chimp128 marker escape run failed");
  const auto chimp_output_path = directory / "chimp-marker-restored.bin";
  const Json chimp_decode{{"schema", "operator-harness-request-v2"},
                          {"command", "decode"},
                          {"request_id", "a039-marker-cold"},
                          {"implementation_ids", Json::array({"A039"})},
                          {"catalog_sha256", tree_catalog_sha256},
                          {"recipe_sha256",
                           chimp_response.at("execution_binding").at(
                               "recipe_sha256")},
                          {"frame_path", chimp_run.at("frame_path")},
                          {"output_path", chimp_output_path.string()},
                          {"warmups", 0},
                          {"repeats", 1}};
  const auto chimp_decoded = handle_line(chimp_decode.dump());
  expect(chimp_decoded.at("status") == "ok" &&
             chimp_decoded.at("frame_verified_bitexact") == true &&
             read_bytes(chimp_output_path) == chimp_input,
         "Chimp128 marker escape cold decode failed");

  auto lane_run = run;
  lane_run["request_id"] = "raw-lane-reinterpret";
  lane_run["implementation_ids"] = Json::array({"A103"});
  lane_run["initial_width"] = 4;
  lane_run["input_view"] = "float32";
  lane_run["recipe"] =
      Json{{"policy_registry", "operator-harness-v2-test"},
           {"steps",
            Json::array(
                {{{"kind", "stage"},
                  {"inputs", Json::array({0})},
                  {"stage", {{"action", "float_deconstruct"},
                              {"profile", "float32"},
                              {"parameters", Json::object()}}}},
                 {{"kind", "stop"}}})}};
  lane_run["frame_path"] = (directory / "lanes.ngf").string();
  const auto lane_response = handle_line(lane_run.dump());
  expect(lane_response.at("status") == "ok" &&
             lane_response.at("bit_adapter") == "raw_bit_lane_split" &&
             lane_response.at("graph_element_count") == 10,
         "bit-preserving raw lane reinterpretation failed");

  const auto fallback_input_path = directory / "float64.bin";
  const std::vector<std::uint8_t> fallback_input{
      0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xb9, 0x3f,
      0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xc9, 0x3f,
      0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xd3, 0x3f,
      0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xd9, 0x3f,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x3f};
  write_bytes(fallback_input_path, fallback_input);
  auto fallback_run = run;
  fallback_run["request_id"] = "a003-json-bool-fallback";
  fallback_run["implementation_ids"] = Json::array({"A003"});
  fallback_run["input_path"] = fallback_input_path.string();
  fallback_run["input_view"] = "float64";
  fallback_run["dtype"] = "float64";
  fallback_run["frame_path"] = (directory / "fallback.ngf").string();
  fallback_run["recipe"] =
      Json{{"policy_registry", "operator-harness-v2-test"},
           {"steps",
            Json::array(
                {{{"kind", "stage"},
                  {"inputs", Json::array({0})},
                  {"stage", {{"action", "op.eraser.elf.cpp_v1"},
                              {"profile", "float64"},
                              {"parameters", {{"exact_raw_fallback", "true"}}}}}},
                 {{"kind", "stop"}}})}};
  const auto fallback_response = handle_line(fallback_run.dump());
  expect(fallback_response.at("status") == "ok" &&
             fallback_response.at("fallback").at("wrapped_stage_count") == 1 &&
             fallback_response.at("fallback").at("target_block_count") == 1 &&
             fallback_response.at("fallback").at("raw_fallback_stage_count") == 1 &&
             fallback_response.at("fallback").at("raw_fallback_block_count") == 1 &&
             fallback_response.at("fallback").at("strict_target_block_count") == 0 &&
             fallback_response.at("fallback")
                     .at("raw_fallback_reason_counts")
                     .at("Elf32 exact action requires a bitwise lossless float32 cast") == 1,
         "normalized parameter or fallback audit failed");

  auto wider = run;
  wider["request_id"] = "wider-is-gap";
  wider["source_byte_width"] = 4;
  wider["element_count"] = 10;
  const auto wider_response = handle_line(wider.dump());
  expect(wider_response.at("status") == "gap" &&
             wider_response.at("reason_code") ==
                 "INPUT_WIDTH_ADAPTATION_NOT_BIT_PRESERVING",
         "wider target did not fail closed as a gap");

  auto fraudulent_stop = run;
  fraudulent_stop["request_id"] = "a001-stop-only-is-rejected";
  fraudulent_stop["implementation_ids"] = Json::array({"A001"});
  fraudulent_stop["recipe"] = stop_recipe();
  fraudulent_stop["frame_path"] = (directory / "fraudulent.ngf").string();
  const auto fraudulent_response = handle_line(fraudulent_stop.dump());
  expect(fraudulent_response.at("status") == "failed" &&
             fraudulent_response.at("error").get<std::string>().find(
                 "catalog binding") != std::string::npos,
         "stop-only recipe was allowed to claim A001");

  auto denied_id = decode;
  denied_id["request_id"] = "a110-denied-before-decode";
  denied_id["implementation_ids"] = Json::array({"A110"});
  const auto denied_id_response = handle_line(denied_id.dump());
  expect(denied_id_response.at("status") == "excluded",
         "A110 decode was not excluded");

  auto wrong_decode_binding = decode;
  wrong_decode_binding["request_id"] = "wrong-cold-binding";
  wrong_decode_binding["implementation_ids"] = Json::array({"A001"});
  wrong_decode_binding["output_path"] = (directory / "wrong.bin").string();
  const auto wrong_decode_response = handle_line(wrong_decode_binding.dump());
  expect(wrong_decode_response.at("status") == "failed",
         "cold decode accepted an implementation ID differing from the frame");

  auto denied_action = run;
  denied_action["request_id"] = "paq-denied-by-action";
  denied_action["implementation_ids"] = Json::array({"A001"});
  denied_action["recipe"]["steps"][0]["stage"]["action"] =
      "op.prediction_probability.paq.provider_v1";
  const auto denied_action_response = handle_line(denied_action.dump());
  expect(denied_action_response.at("status") == "excluded",
         "PAQ action was not excluded independently of trace ID");

  const auto capabilities = handle_line(
      Json{{"schema", "operator-harness-request-v2"},
           {"command", "capabilities"},
           {"request_id", "capabilities"}}
          .dump());
  expect(capabilities.at("status") == "ok", "capabilities failed");
  expect(capabilities.at("excluded_implementation_ids").size() == 5,
         "excluded implementation set changed");

  fs::remove_all(directory);
  std::cout << "operator_harness_v2 protocol test passed\n";
}

