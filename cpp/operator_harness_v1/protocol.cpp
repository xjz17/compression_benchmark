#include "protocol.hpp"

#include "graph.hpp"
#include "harness_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace operator_harness_v1 {
namespace {
using Json = nlohmann::json;
using namespace native_graph_v1;

constexpr std::size_t kMaxCaseIdBytes = 1024;
constexpr std::size_t kMaxHexCharacters = 512ULL << 20;

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) invalid(message);
}

std::string checked_text(const Json& value, const std::string& field,
                         std::size_t max_bytes) {
  require(value.contains(field) && value.at(field).is_string(), field + " must be a string");
  auto result = value.at(field).get<std::string>();
  require(result.size() <= max_bytes && result.find('\0') == std::string::npos,
          field + " is invalid or too long");
  return result;
}

bool optional_bool(const Json& value, const std::string& field, bool fallback) {
  if (!value.contains(field)) return fallback;
  require(value.at(field).is_boolean(), field + " must be boolean");
  return value.at(field).get<bool>();
}

std::uint64_t unsigned_value(const Json& value, const std::string& field,
                             std::uint64_t fallback, std::uint64_t maximum) {
  if (!value.contains(field)) return fallback;
  const auto& item = value.at(field);
  require(item.is_number_integer() || item.is_number_unsigned(), field + " must be an integer");
  std::uint64_t result = 0;
  if (item.is_number_unsigned()) result = item.get<std::uint64_t>();
  else {
    const auto signed_value = item.get<std::int64_t>();
    require(signed_value >= 0, field + " must be nonnegative");
    result = static_cast<std::uint64_t>(signed_value);
  }
  require(result <= maximum, field + " exceeds the supported bound");
  return result;
}

std::int64_t signed_value(const Json& value, const std::string& field,
                          std::int64_t fallback) {
  if (!value.contains(field)) return fallback;
  require(value.at(field).is_number_integer(), field + " must be a signed integer");
  return value.at(field).get<std::int64_t>();
}

Bytes unhex(const std::string& text) {
  require(text.size() % 2 == 0 && text.size() <= kMaxHexCharacters,
          "invalid or oversized hex payload");
  auto digit = [](char c) -> unsigned {
    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
    invalid("hex payload must be canonical lowercase");
  };
  Bytes result;
  result.reserve(text.size() / 2);
  for (std::size_t i = 0; i < text.size(); i += 2)
    result.push_back(static_cast<std::uint8_t>((digit(text[i]) << 4) | digit(text[i + 1])));
  return result;
}

std::string hex(std::span<const std::uint8_t> bytes) {
  static constexpr char alphabet[] = "0123456789abcdef";
  require(bytes.size() <= std::numeric_limits<std::size_t>::max() / 2,
          "hex response size overflow");
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    result.push_back(alphabet[byte >> 4]);
    result.push_back(alphabet[byte & 15]);
  }
  return result;
}

Kind parse_kind(const std::string& value) {
  if (value == "serial") return Kind::Serial;
  if (value == "numeric") return Kind::Numeric;
  if (value == "struct") return Kind::Struct;
  invalid("unknown flow kind");
}

const char* kind_name(Kind kind) {
  if (kind == Kind::Serial) return "serial";
  if (kind == Kind::Numeric) return "numeric";
  return "struct";
}

Limits parse_limits(const Json& request) {
  Limits result;
  if (!request.contains("limits")) return result;
  const auto& limits = request.at("limits");
  require(limits.is_object(), "limits must be an object");
  result.max_nodes = static_cast<std::uint32_t>(
      unsigned_value(limits, "max_nodes", result.max_nodes, 4096));
  result.max_active_streams = static_cast<std::uint32_t>(
      unsigned_value(limits, "max_active_streams", result.max_active_streams, 1024));
  result.max_ordinary_depth = static_cast<std::uint32_t>(
      unsigned_value(limits, "max_ordinary_depth", result.max_ordinary_depth, 4096));
  result.max_stream_bytes = unsigned_value(
      limits, "max_stream_bytes", result.max_stream_bytes, 1ULL << 40);
  result.max_working_bytes = unsigned_value(
      limits, "max_working_bytes", result.max_working_bytes, 1ULL << 40);
  result.max_frame_bytes = unsigned_value(
      limits, "max_frame_bytes", result.max_frame_bytes, 1ULL << 40);
  result.max_metadata_bytes = unsigned_value(
      limits, "max_metadata_bytes", result.max_metadata_bytes, 1ULL << 40);
  require(result.max_nodes > 0 && result.max_active_streams > 0 &&
              result.max_ordinary_depth > 0 && result.max_stream_bytes <= result.max_working_bytes &&
              result.max_metadata_bytes <= result.max_frame_bytes && result.max_frame_bytes >= 48,
          "inconsistent limits");
  return result;
}

std::vector<RootInput> parse_roots(const Json& value) {
  require(value.is_array() && !value.empty() && value.size() <= 32, "invalid roots");
  std::vector<RootInput> result;
  result.reserve(value.size());
  for (const auto& row : value) {
    require(row.is_object(), "root must be an object");
    const auto root_kind = parse_kind(checked_text(row, "kind", 16));
    const auto width = static_cast<std::uint32_t>(unsigned_value(row, "width", 0, 65536));
    const auto data = unhex(checked_text(row, "data_hex", kMaxHexCharacters));
    require(row.contains("original_numeric_payload") &&
                row.at("original_numeric_payload").is_boolean(),
            "original_numeric_payload must be boolean");
    result.push_back({{root_kind, width, data},
                      row.at("original_numeric_payload").get<bool>(),
                      checked_text(row, "logical_view", 128)});
  }
  return result;
}

std::vector<StreamId> parse_inputs(const Json& value) {
  require(value.is_array() && value.size() <= 1024, "inputs must be a bounded array");
  std::vector<StreamId> result;
  result.reserve(value.size());
  for (const auto& item : value) {
    Json holder{{"value", item}};
    result.push_back(static_cast<StreamId>(unsigned_value(holder, "value", 0, UINT32_MAX)));
  }
  return result;
}

std::vector<std::uint64_t> parse_segments(const Json& value) {
  require(value.is_array() && value.size() <= 1024, "segment_sizes must be a bounded array");
  std::vector<std::uint64_t> result;
  result.reserve(value.size());
  for (const auto& item : value) {
    Json holder{{"value", item}};
    result.push_back(unsigned_value(holder, "value", 0, UINT64_MAX));
  }
  return result;
}

Recipe parse_recipe(const Json& value) {
  require(value.is_object(), "recipe must be an object");
  Recipe result;
  result.policy_registry = checked_text(value, "policy_registry", 128);
  require(value.contains("steps") && value.at("steps").is_array() &&
              !value.at("steps").empty() && value.at("steps").size() <= 65,
          "invalid recipe size");
  for (const auto& row : value.at("steps")) {
    require(row.is_object(), "recipe step must be an object");
    Step step;
    const auto operation = checked_text(row, "kind", 32);
    if (operation == "stage") step.kind = StepKind::Stage;
    else if (operation == "split_segments") step.kind = StepKind::SplitSegments;
    else if (operation == "concat") step.kind = StepKind::Concat;
    else if (operation == "stop") step.kind = StepKind::Stop;
    else invalid("unknown step kind");
    if (row.contains("inputs")) step.inputs = parse_inputs(row.at("inputs"));
    if (row.contains("segment_sizes")) step.segment_sizes = parse_segments(row.at("segment_sizes"));
    step.policy_action_id = signed_value(row, "policy_action_id", -1);
    if (row.contains("stage")) {
      const auto& stage = row.at("stage");
      require(stage.is_object(), "stage must be an object");
      step.stage.action = checked_text(stage, "action", 512);
      if (stage.contains("profile")) step.stage.profile = checked_text(stage, "profile", 512);
      if (stage.contains("parameters")) {
        require(stage.at("parameters").is_object() && stage.at("parameters").size() <= 64,
                "stage parameters must be a bounded object");
        for (const auto& [key, item] : stage.at("parameters").items()) {
          require(!key.empty() && key.size() <= 512 && item.is_string(),
                  "invalid stage parameter");
          const auto parameter = item.get<std::string>();
          require(parameter.size() <= 4096 && parameter.find('\0') == std::string::npos,
                  "invalid stage parameter value");
          step.stage.parameters.emplace(key, parameter);
        }
      }
    }
    result.steps.push_back(std::move(step));
  }
  return result;
}

Json roots_json(const Decoded& decoded) {
  Json result = Json::array();
  for (const auto& root : decoded.roots) {
    result.push_back({{"kind", kind_name(root.stream.kind)},
                      {"width", root.stream.width},
                      {"data_hex", hex(root.stream.data)},
                      {"original_numeric_payload", root.original_numeric_payload},
                      {"logical_view", root.logical_view}});
  }
  return result;
}

void verify_roots(const Decoded& decoded, const std::vector<RootInput>& expected) {
  require(decoded.roots.size() == expected.size(), "cold decode root count mismatch");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto expected_view = expected[index].logical_view.empty()
                                   ? (expected[index].stream.kind == Kind::Serial ? "bytes" :
                                      expected[index].stream.kind == Kind::Numeric ? "physical_word" : "struct")
                                   : expected[index].logical_view;
    require(decoded.roots[index].stream == expected[index].stream &&
                decoded.roots[index].original_numeric_payload ==
                    expected[index].original_numeric_payload &&
                decoded.roots[index].logical_view == expected_view,
            "cold decode bitwise root mismatch");
  }
}

std::uint64_t elapsed(std::chrono::steady_clock::time_point begin,
                      std::chrono::steady_clock::time_point end) {
  const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  require(value >= 0, "nonmonotonic steady clock");
  return static_cast<std::uint64_t>(value);
}

Json base_response(const std::string& command, const std::string& case_id) {
  return {{"schema", "operator-harness-response-v1"},
          {"status", "ok"},
          {"command", command},
          {"case_id", case_id}};
}
}  // namespace

Json handle_request(const Json& request) {
  require(request.is_object(), "request must be an object");
  require(checked_text(request, "schema", 64) == "operator-harness-request-v1",
          "wrong request schema");
  const auto command = checked_text(request, "command", 32);
  const auto case_id = checked_text(request, "case_id", kMaxCaseIdBytes);
  require(command == "run" || command == "decode" || command == "capabilities",
          "unknown command");
  auto response = base_response(command, case_id);

  const bool allow_context_heavy = optional_bool(request, "allow_context_heavy", false);
  HarnessDispatcher dispatcher(allow_context_heavy);
  response["dispatcher"] = dispatcher.identity();

  if (command == "capabilities") {
    response["nncp"] = "disabled";
    response["actions"] = Json::array();
    for (const auto& capability : harness_capabilities()) {
      response["actions"].push_back({{"action", capability.stage.action},
                                     {"source", capability.stage.source},
                                     {"semantics", capability.stage.semantics},
                                     {"multiple_outputs", capability.stage.multiple_outputs},
                                     {"enabled_by_default", capability.enabled_by_default},
                                     {"context_heavy", capability.context_heavy}});
    }
    response["action_count"] = response["actions"].size();
    return response;
  }

  const auto limits = parse_limits(request);
  if (command == "decode") {
    const auto frame = unhex(checked_text(request, "frame_hex", kMaxHexCharacters));
    const bool measure = optional_bool(request, "measure_decode", false);
    const auto warmups = static_cast<std::uint32_t>(unsigned_value(request, "warmups", 0, 31));
    const auto repeats = static_cast<std::uint32_t>(unsigned_value(request, "repeats", 1, 1001));
    require(repeats > 0, "repeats must be positive");
    Decoded decoded;
    Json timings = Json::array();
    if (measure) {
      for (std::uint32_t iteration = 0; iteration < warmups + repeats; ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        auto candidate = decode(frame, dispatcher, limits);
        const auto end = std::chrono::steady_clock::now();
        if (iteration >= warmups) timings.push_back(elapsed(begin, end));
        decoded = std::move(candidate);
      }
      response["decode_ns"] = std::move(timings);
      response["timing_boundary"] =
          "native NGF1 parse + frame checksum/validation + inverse scheduling + all inverse kernels; excludes process startup, JSON/hex conversion, and driver I/O";
    } else {
      require(warmups == 0 && repeats == 1,
              "warmups/repeats require measure_decode=true for decode");
      decoded = decode(frame, dispatcher, limits);
    }
    response["original_numeric_bytes"] = decoded.original_numeric_bytes;
    response["frame_bytes"] = frame.size();
    response["roots"] = roots_json(decoded);
    response["bitexact"] = true;
    return response;
  }

  const auto roots = parse_roots(request.at("roots"));
  const auto recipe = parse_recipe(request.at("recipe"));
  const auto warmups = static_cast<std::uint32_t>(unsigned_value(request, "warmups", 1, 31));
  const auto repeats = static_cast<std::uint32_t>(unsigned_value(request, "repeats", 3, 1001));
  require(repeats > 0, "repeats must be positive");
  auto measured = benchmark(roots, recipe, dispatcher, warmups, repeats, limits);

  // Use a distinct dispatcher instance and the serialized frame. This catches
  // hidden in-memory state. The decode command supports a separate-process
  // cold check using exactly the returned frame_hex.
  HarnessDispatcher cold_dispatcher(allow_context_heavy);
  auto cold = decode(measured.frame, cold_dispatcher, limits);
  verify_roots(cold, roots);

  response["original_numeric_bytes"] = measured.original_numeric_bytes;
  response["frame_bytes"] = measured.frame.size();
  response["frame_hex"] = hex(measured.frame);
  response["encode_ns"] = measured.encode_nanoseconds;
  response["decode_ns"] = measured.decode_nanoseconds;
  response["timing_boundary"] =
      "native fixed-recipe graph scheduling + kernels + complete NGF1 framing/checksum for encode; NGF1 parse/checksum + inverse scheduling + kernels for decode; excludes JSON/hex I/O and recipe selection";
  response["verification_boundary"] =
      "bitwise root verification after decode with a distinct dispatcher instance; use command=decode in a fresh process for process-cold verification";
  response["bitexact"] = true;
  return response;
}

Json handle_line(const std::string& line) {
  std::string command;
  std::string case_id;
  try {
    require(line.size() <= (kMaxHexCharacters + (16ULL << 20)), "request line is too large");
    const auto request = Json::parse(line);
    if (request.is_object()) {
      if (request.contains("command") && request.at("command").is_string())
        command = request.at("command").get<std::string>();
      if (request.contains("case_id") && request.at("case_id").is_string())
        case_id = request.at("case_id").get<std::string>();
    }
    return handle_request(request);
  } catch (const std::exception& error) {
    return {{"schema", "operator-harness-response-v1"},
            {"status", "failed"},
            {"command", command},
            {"case_id", case_id},
            {"error", error.what()}};
  }
}

}  // namespace operator_harness_v1
