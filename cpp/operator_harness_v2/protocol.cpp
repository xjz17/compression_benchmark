#include "protocol.hpp"

#include "a048_typed_planes.hpp"
#include "binding_contract.hpp"
#include "graph.hpp"
#include "policy_dispatch.hpp"
#include "stage_dispatch.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace operator_harness_v2 {
namespace {

using Json = nlohmann::json;
using namespace native_graph_v1;
namespace fs = std::filesystem;

constexpr std::size_t kMaxRequestBytes = 16ULL << 20;
constexpr std::size_t kMaxTextBytes = 4096;

class GapError final : public std::runtime_error {
 public:
  GapError(std::string code, std::string message)
      : std::runtime_error(std::move(message)), code_(std::move(code)) {}
  const std::string& code() const { return code_; }

 private:
  std::string code_;
};

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) invalid(message);
}

std::string checked_text(const Json& value, const std::string& field,
                         std::size_t maximum = kMaxTextBytes) {
  require(value.contains(field) && value.at(field).is_string(),
          field + " must be a string");
  auto result = value.at(field).get<std::string>();
  require(!result.empty() && result.size() <= maximum &&
              result.find('\0') == std::string::npos,
          field + " is empty, invalid, or too long");
  return result;
}

bool checked_bool(const Json& value, const std::string& field, bool fallback) {
  if (!value.contains(field)) return fallback;
  require(value.at(field).is_boolean(), field + " must be boolean");
  return value.at(field).get<bool>();
}

std::uint64_t unsigned_value(const Json& value, const std::string& field,
                             std::uint64_t fallback, std::uint64_t maximum) {
  if (!value.contains(field)) return fallback;
  const auto& item = value.at(field);
  require(item.is_number_integer() || item.is_number_unsigned(),
          field + " must be an integer");
  std::uint64_t result = 0;
  if (item.is_number_unsigned()) {
    result = item.get<std::uint64_t>();
  } else {
    const auto signed_result = item.get<std::int64_t>();
    require(signed_result >= 0, field + " must be nonnegative");
    result = static_cast<std::uint64_t>(signed_result);
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

Kind parse_kind(std::string_view kind) {
  if (kind == "serial") return Kind::Serial;
  if (kind == "numeric") return Kind::Numeric;
  if (kind == "struct") return Kind::Struct;
  invalid("initial_kind must be serial, numeric, or struct");
}

const char* kind_name(Kind kind) {
  if (kind == Kind::Serial) return "serial";
  if (kind == Kind::Numeric) return "numeric";
  return "struct";
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const std::string& label) {
  require(right == 0 || left <= std::numeric_limits<std::uint64_t>::max() / right,
          label + " overflows uint64");
  return left * right;
}

fs::path checked_absolute_path(const Json& value, const std::string& field) {
  const auto text = checked_text(value, field, 16384);
  fs::path path(text);
  require(path.is_absolute(), field + " must be an absolute path");
  return path.lexically_normal();
}

Bytes read_file(const fs::path& path, std::uint64_t maximum,
                const std::string& label) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  require(!error, label + " file size failed: " + error.message());
  require(size <= maximum && size <= std::numeric_limits<std::size_t>::max(),
          label + " exceeds the configured size limit");
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open " + label + ": " + path.string());
  Bytes bytes(static_cast<std::size_t>(size));
  if (!bytes.empty())
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  require(input.good() || input.eof(), "failed while reading " + label);
  require(static_cast<std::uint64_t>(input.gcount()) == size || size == 0,
          "short read from " + label);
  return bytes;
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int length = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), digest, &length,
                     EVP_sha256(), nullptr) == 1 && length == 32,
          "OpenSSL SHA-256 failed");
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (unsigned index = 0; index < length; ++index) {
    result.push_back(alphabet[digest[index] >> 4]);
    result.push_back(alphabet[digest[index] & 15]);
  }
  return result;
}

void write_file_atomic(const fs::path& destination,
                       std::span<const std::uint8_t> bytes) {
  require(destination.has_parent_path(), "output path requires a parent directory");
  require(fs::is_directory(destination.parent_path()),
          "output parent directory does not exist");
  static std::atomic<std::uint64_t> sequence{0};
  const auto nonce = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
#if defined(__unix__) || defined(__APPLE__)
  const auto process = static_cast<unsigned long long>(::getpid());
#else
  const auto process = 0ULL;
#endif
  auto temporary = destination;
  temporary += ".tmp." + std::to_string(process) + "." +
               std::to_string(nonce) + "." +
               std::to_string(sequence.fetch_add(1));
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(output.good(), "cannot create temporary output: " + temporary.string());
    if (!bytes.empty())
      output.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    output.flush();
    require(output.good(), "failed while writing temporary output");
  }
  std::error_code error;
  fs::rename(temporary, destination, error);
  if (error) {
    std::error_code cleanup_error;
    fs::remove(temporary, cleanup_error);
    invalid("atomic output commit failed: " + error.message());
  }
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
  result.max_stream_bytes =
      unsigned_value(limits, "max_stream_bytes", result.max_stream_bytes, 1ULL << 40);
  result.max_working_bytes =
      unsigned_value(limits, "max_working_bytes", result.max_working_bytes, 1ULL << 40);
  result.max_frame_bytes =
      unsigned_value(limits, "max_frame_bytes", result.max_frame_bytes, 1ULL << 40);
  result.max_metadata_bytes = unsigned_value(
      limits, "max_metadata_bytes", result.max_metadata_bytes, 1ULL << 40);
  require(result.max_nodes > 0 && result.max_active_streams > 0 &&
              result.max_ordinary_depth > 0 &&
              result.max_stream_bytes <= result.max_working_bytes &&
              result.max_metadata_bytes <= result.max_frame_bytes &&
              result.max_frame_bytes >= 48,
          "inconsistent limits");
  return result;
}

std::vector<StreamId> parse_inputs(const Json& value) {
  require(value.is_array() && value.size() <= 1024,
          "inputs must be a bounded array");
  std::vector<StreamId> result;
  for (const auto& item : value) {
    Json holder{{"value", item}};
    result.push_back(static_cast<StreamId>(
        unsigned_value(holder, "value", 0, UINT32_MAX)));
  }
  return result;
}

std::vector<std::uint64_t> parse_segments(const Json& value) {
  require(value.is_array() && value.size() <= 1024,
          "segment_sizes must be a bounded array");
  std::vector<std::uint64_t> result;
  for (const auto& item : value) {
    Json holder{{"value", item}};
    result.push_back(unsigned_value(holder, "value", 0, UINT64_MAX));
  }
  return result;
}

Recipe parse_recipe(const Json& value) {
  require(value.is_object(), "recipe must be an object");
  Recipe result;
  result.policy_registry = checked_text(value, "policy_registry", 512);
  require(value.contains("steps") && value.at("steps").is_array() &&
              !value.at("steps").empty() && value.at("steps").size() <= 4097,
          "invalid recipe size");
  for (const auto& row : value.at("steps")) {
    require(row.is_object(), "recipe step must be an object");
    Step step;
    const auto operation = checked_text(row, "kind", 32);
    if (operation == "stage") step.kind = StepKind::Stage;
    else if (operation == "split_segments") step.kind = StepKind::SplitSegments;
    else if (operation == "concat") step.kind = StepKind::Concat;
    else if (operation == "stop") step.kind = StepKind::Stop;
    else invalid("unknown recipe step kind");
    if (row.contains("inputs")) step.inputs = parse_inputs(row.at("inputs"));
    if (row.contains("segment_sizes"))
      step.segment_sizes = parse_segments(row.at("segment_sizes"));
    step.policy_action_id = signed_value(row, "policy_action_id", -1);
    if (row.contains("stage")) {
      const auto& stage = row.at("stage");
      require(stage.is_object(), "stage must be an object");
      step.stage.action = checked_text(stage, "action", 512);
      if (const auto reason = excluded_action_reason(step.stage.action))
        throw ExcludedPolicyError(*reason + ": " + step.stage.action);
      if (stage.contains("profile"))
        step.stage.profile = checked_text(stage, "profile", 512);
      if (stage.contains("parameters")) {
        require(stage.at("parameters").is_object() &&
                    stage.at("parameters").size() <= 64,
                "stage parameters must be a bounded object");
        for (const auto& [key, item] : stage.at("parameters").items()) {
          require(!key.empty() && key.size() <= 512 && item.is_primitive() &&
                      !item.is_null(),
                  "stage parameter must be a JSON string, number, or boolean");
          const auto parameter = item.is_string()
                                     ? item.get<std::string>()
                                     : item.is_boolean()
                                           ? (item.get<bool>() ? "true" : "false")
                                           : item.dump();
          require(parameter.size() <= 4096 &&
                      parameter.find('\0') == std::string::npos,
                  "invalid stage parameter value");
          step.stage.parameters.emplace(key, parameter);
        }
      }
    }
    result.steps.push_back(std::move(step));
  }
  return result;
}

std::vector<std::string> parse_implementation_ids(const Json& request) {
  require(request.contains("implementation_ids") &&
              request.at("implementation_ids").is_array() &&
              !request.at("implementation_ids").empty() &&
              request.at("implementation_ids").size() <= 102,
          "implementation_ids must be a nonempty bounded array");
  std::vector<std::string> result;
  for (const auto& value : request.at("implementation_ids")) {
    require(value.is_string(), "implementation ID must be a string");
    const auto id = value.get<std::string>();
    require(id.size() == 4 && id[0] == 'A' &&
                std::all_of(id.begin() + 1, id.end(), [](unsigned char c) {
                  return c >= '0' && c <= '9';
                }),
            "invalid implementation ID");
    if (excluded_implementation_id(id))
      throw ExcludedPolicyError(
          "EXCLUDED_IMPLEMENTATION_POLICY: " + id +
          " (PPMd/DMC/PAQ8/CMIX/LSTM/NNCP remain disabled)");
    result.push_back(id);
  }
  std::sort(result.begin(), result.end());
  require(std::adjacent_find(result.begin(), result.end()) == result.end(),
          "duplicate implementation ID");
  return result;
}

std::uint64_t elapsed(std::chrono::steady_clock::time_point begin,
                      std::chrono::steady_clock::time_point end) {
  const auto value =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  require(value >= 0, "nonmonotonic steady clock");
  return static_cast<std::uint64_t>(value);
}

Json fallback_json(const FallbackAudit& audit) {
  require(audit.raw_fallback_stage_count <= audit.wrapped_stage_count,
          "fallback audit count exceeds target count");
  return {{"wrapped_stage_count", audit.wrapped_stage_count},
          {"raw_fallback_stage_count", audit.raw_fallback_stage_count},
          {"reasons", audit.reasons},
          // Catalog-contract names count the one untimed source block that
          // reached each wrapped target. Keep the legacy stage names above for
          // v2 consumers while exposing the machine-auditable v3 fields.
          {"target_block_count", audit.wrapped_stage_count},
          {"strict_target_block_count",
           audit.wrapped_stage_count - audit.raw_fallback_stage_count},
          {"raw_fallback_block_count", audit.raw_fallback_stage_count},
          {"raw_fallback_reason_counts", audit.reason_counts}};
}

std::optional<Json> typed_plane_contract(const Json& execution_binding,
                                         std::uint64_t element_count,
                                         std::uint64_t input_bytes) {
  if (execution_binding.value("execution_variant", "") !=
      "pure_typed_plane")
    return std::nullopt;
  require(execution_binding.value("schema", "") ==
                  "operator-harness-execution-binding-v3" &&
              execution_binding.contains("harness_parameters") &&
              execution_binding.at("harness_parameters").contains(
                  "bit_width") &&
              execution_binding.at("harness_parameters").at("bit_width")
                  .is_string(),
          "invalid A048 typed-plane execution binding");
  const auto bit_width_text =
      execution_binding.at("harness_parameters").at("bit_width")
          .get<std::string>();
  unsigned bit_width = 0;
  for (const auto c : bit_width_text) {
    require(c >= '0' && c <= '9', "invalid bound A048 bit_width");
    bit_width = bit_width * 10 + static_cast<unsigned>(c - '0');
  }
  require(bit_width <= 64, "bound A048 bit_width exceeds 64");
  const auto plane_count = static_cast<std::uint64_t>((bit_width + 7) / 8);
  const auto total_plane_payload_bytes =
      checked_multiply(element_count, plane_count,
                       "A048 total plane payload bytes");
  Json shapes = Json::array();
  for (std::uint64_t plane = 0; plane < plane_count; ++plane) {
    shapes.push_back({{"plane_index", plane},
                      {"kind", "serial"},
                      {"width", 1},
                      {"bytes", element_count}});
  }
  Json physical_shapes = shapes;
  if (plane_count == 0) {
    physical_shapes.push_back({{"plane_index", nullptr},
                               {"kind", "serial"},
                               {"width", 1},
                               {"bytes", 0},
                               {"carrier_only", true}});
  }
  return Json{{"schema", "operator-harness-a048-typed-plane-contract-v1"},
              {"bit_width", bit_width},
              {"bit_width_parameter", bit_width_text},
              {"source_word_width_bytes", 8},
              {"plane_width_bytes", 1},
              {"semantic_plane_count", plane_count},
              {"physical_output_count",
               std::max<std::uint64_t>(plane_count, 1)},
              {"element_count", element_count},
              {"plane_order",
               "width_aligned_most_significant_byte_to_least_significant_byte"},
              {"output_shapes", std::move(shapes)},
              {"physical_output_shapes", std::move(physical_shapes)},
              {"total_plane_payload_bytes", total_plane_payload_bytes},
              {"input_bytes", input_bytes},
              {"plane_payload_equals_input_bytes",
               total_plane_payload_bytes == input_bytes},
              {"zero_plane_physical_carrier", plane_count == 0},
              {"inverse_consumes_all_planes", true}};
}

std::optional<Json> audit_a048_typed_plane_ports(
    const Recipe& recipe, const Stream& input,
    const std::optional<Json>& contract) {
  if (!contract) return std::nullopt;
  require(recipe.steps.size() == 2 &&
              recipe.steps.front().kind == StepKind::Stage &&
              recipe.steps.front().stage.action == kA048TypedPlaneAction,
          "A048 proof requires one typed-plane stage followed by stop");
  const auto bit_width = contract->at("bit_width").get<unsigned>();
  const auto plane_count =
      contract->at("semantic_plane_count").get<std::uint64_t>();
  const auto element_count = contract->at("element_count").get<std::uint64_t>();
  require(input.kind == Kind::Numeric && input.width == 8 &&
              input.data.size() == element_count * 8,
          "A048 proof input shape differs from the contract");

  PolicyDispatcher dispatcher(false);
  const auto result =
      dispatcher.forward(recipe.steps.front().stage, {input});
  require(result.outputs.size() == std::max<std::uint64_t>(plane_count, 1),
          "A048 proof physical output count mismatch");
  std::vector<std::uint64_t> values(static_cast<std::size_t>(element_count));
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
      value |= static_cast<std::uint64_t>(input.data[index * 8 + byte])
               << (byte * 8);
    values[index] = value;
  }

  Json per_plane = Json::array();
  std::uint64_t payload_bytes = 0;
  const auto padding = static_cast<unsigned>(plane_count * 8) - bit_width;
  for (std::uint64_t plane = 0; plane < plane_count; ++plane) {
    const auto& output = result.outputs.at(static_cast<std::size_t>(plane));
    require(output.kind == Kind::Serial && output.width == 1 &&
                output.data.size() == element_count,
            "A048 proof encountered a non-byte or wrong-length plane");
    const auto shift = static_cast<unsigned>((plane_count - 1 - plane) * 8);
    for (std::size_t index = 0; index < values.size(); ++index) {
      const auto aligned =
          padding == 0 ? values[index] : values[index] << padding;
      require(output.data[index] == static_cast<std::uint8_t>(aligned >> shift),
              "A048 proof plane bytes differ from the width-aligned input bits");
    }
    payload_bytes += output.data.size();
    per_plane.push_back({{"plane_index", plane},
                         {"kind", "serial"},
                         {"width", 1},
                         {"bytes", output.data.size()},
                         {"sha256", sha256(output.data)}});
  }
  bool valid_zero_carrier = false;
  if (plane_count == 0) {
    const auto& carrier = result.outputs.front();
    valid_zero_carrier = carrier.kind == Kind::Serial && carrier.width == 1 &&
                         carrier.data.empty();
    require(valid_zero_carrier,
            "A048 proof found an invalid zero-plane physical carrier");
  }
  require(payload_bytes ==
              contract->at("total_plane_payload_bytes").get<std::uint64_t>(),
          "A048 proof semantic plane payload sum differs from the contract");

  const std::string metadata_text(result.metadata.begin(),
                                  result.metadata.end());
  const auto metadata = Json::parse(metadata_text);
  require(metadata.is_object() &&
              metadata.value("schema", "") ==
                  "operator-harness-a048-typed-plane-v2" &&
              metadata.at("count") == element_count &&
              metadata.at("bit_width") == bit_width &&
              metadata.at("semantic_plane_count") == plane_count,
          "A048 proof shape-only metadata differs from actual ports");
  const auto restored = dispatcher.inverse(recipe.steps.front().stage,
                                            result.outputs, result.metadata);
  require(restored.size() == 1 && restored.front() == input,
          "A048 proof inverse did not consume the actual ports bit-exactly");

  return Json{
      {"schema", "operator-harness-a048-typed-plane-proof-v1"},
      {"stage_result_source", "untimed_policy_dispatcher_forward"},
      {"bit_width", bit_width},
      {"element_count", element_count},
      {"semantic_plane_count", plane_count},
      {"physical_output_count", result.outputs.size()},
      {"per_plane", std::move(per_plane)},
      {"total_plane_payload_bytes", payload_bytes},
      {"zero_plane_physical_carrier_valid", valid_zero_carrier},
      {"plane_order",
       "width_aligned_most_significant_byte_to_least_significant_byte"},
      {"expected_shift_bytes_match", true},
      {"inverse_from_outputs_and_shape_metadata_bitexact", true},
      {"metadata_schema", metadata.at("schema")},
      {"metadata_bytes", result.metadata.size()},
      {"metadata_sha256", sha256(result.metadata)},
      {"metadata_is_shape_only", true}};
}

Bytes json_bytes(const Json& value) {
  const auto text = value.dump();
  return {text.begin(), text.end()};
}

Json parse_source_schema(const RootInput& root) {
  require(root.stream.kind == Kind::Serial && root.stream.width == 1 &&
              !root.original_numeric_payload &&
              root.logical_view == "operator_harness_v2_source_schema",
          "invalid source-schema auxiliary root contract");
  const std::string text(root.stream.data.begin(), root.stream.data.end());
  const auto schema = Json::parse(text);
  require(schema.is_object() &&
              schema.value("schema", "") ==
                  "operator-harness-source-schema-v2",
          "invalid source-schema auxiliary root payload");
  return schema;
}

Json base_response(std::string phase, std::string request_id) {
  return {{"schema", "operator-harness-response-v2"},
          {"status", "ok"},
          {"phase", std::move(phase)},
          {"request_id", std::move(request_id)}};
}

Json capabilities_response(const std::string& request_id) {
  auto response = base_response("capabilities", request_id);
  response["dispatcher"] = PolicyDispatcher{}.identity();
  response["excluded_implementation_ids"] =
      Json::array({"A106", "A107", "A108", "A109", "A110"});
  response["excluded_families"] =
      Json::array({"PPMd", "DMC", "PAQ8", "CMIX", "LSTM", "NNCP"});
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  response["optional_support_deps"] = "enabled";
#else
  response["optional_support_deps"] = "disabled";
#endif
  response["actions"] = Json::array();
  response["action_inventory_scope"] =
      "native_stage_capabilities base registry only; Operator102Dispatcher and "
      "operator_harness_v2 special-stage actions are additional";
  response["exact_catalog_support_inventory"] = false;
  for (const auto& capability : native_stage_capabilities()) {
    if (excluded_action_reason(capability.action)) continue;
    response["actions"].push_back(
        {{"action", capability.action},
         {"source", capability.source},
         {"semantics", capability.semantics},
         {"multiple_outputs", capability.multiple_outputs}});
  }
  response["actions"].push_back(
      {{"action", kA048TypedPlaneAction},
       {"source", "cpp/operator_harness_v2/a048_typed_planes.cpp"},
       {"semantics", "A048 pure typed uint64 to independent byte-plane ports"},
       {"multiple_outputs", true}});
  response["native_action_count"] = response["actions"].size();
  return response;
}

Json run_request(const Json& request, const std::string& request_id,
                 const std::vector<std::string>& implementation_ids) {
  // Parse first so the non-overridable action-family policy is enforced even
  // when a denied request also lies about its catalog binding.
  const auto recipe = parse_recipe(request.at("recipe"));
  const auto execution_binding = validate_catalog_binding(request);
  const auto limits = parse_limits(request);
  const auto input_path = checked_absolute_path(request, "input_path");
  const auto frame_path = checked_absolute_path(request, "frame_path");
  const auto initial_kind = parse_kind(checked_text(request, "initial_kind", 16));
  const auto initial_width = static_cast<std::uint32_t>(
      unsigned_value(request, "initial_width", 0, 65536));
  const auto source_byte_width = static_cast<std::uint32_t>(
      unsigned_value(request, "source_byte_width", 0, 65536));
  const auto source_element_count =
      unsigned_value(request, "element_count", 0, UINT64_MAX);
  const auto dtype = checked_text(request, "dtype", 128);
  const auto input_view = checked_text(request, "input_view", 128);
  const bool original_numeric_payload =
      checked_bool(request, "original_numeric_payload", true);
  require(initial_width > 0 && source_byte_width > 0,
          "initial_width and source_byte_width must be positive");

  if (initial_width > source_byte_width ||
      source_byte_width % initial_width != 0) {
    throw GapError(
        "INPUT_WIDTH_ADAPTATION_NOT_BIT_PRESERVING",
        "initial_width must divide source_byte_width and cannot be wider; "
        "the v2 harness never casts, extends, or truncates real values");
  }

  auto input = read_file(input_path, limits.max_stream_bytes, "input");
  const auto expected_bytes = checked_multiply(
      source_element_count, source_byte_width, "source byte count");
  require(input.size() == expected_bytes,
          "input file size does not equal element_count * source_byte_width");
  require(input.size() % initial_width == 0,
          "input byte count is not aligned to initial_width");

  const auto warmups = static_cast<std::uint32_t>(
      unsigned_value(request, "warmups", 0, 31));
  const auto repeats = static_cast<std::uint32_t>(
      unsigned_value(request, "repeats", 1, 1001));
  require(repeats > 0, "repeats must be positive");

  const auto adapter_operation =
      initial_width == source_byte_width ? "identity" : "raw_bit_lane_split";
  const auto graph_element_count = input.size() / initial_width;
  const auto a048_contract =
      typed_plane_contract(execution_binding, graph_element_count, input.size());
  Json source_schema{
      {"schema", "operator-harness-source-schema-v2"},
      {"source_dtype", dtype},
      {"source_byte_width", source_byte_width},
      {"source_element_count", source_element_count},
      {"adapter_operation", adapter_operation},
      {"initial_kind", kind_name(initial_kind)},
      {"initial_width", initial_width},
      {"input_view", input_view},
      {"graph_element_count", graph_element_count},
      {"input_bytes", input.size()},
      {"execution_binding", execution_binding}};
  if (a048_contract)
    source_schema["typed_plane_contract"] = *a048_contract;
  auto source_schema_bytes = json_bytes(source_schema);
  const std::vector<RootInput> roots{
      {{initial_kind, initial_width, input}, original_numeric_payload, input_view},
      {{Kind::Serial, 1, source_schema_bytes}, false,
       "operator_harness_v2_source_schema"}};
  const auto a048_proof =
      audit_a048_typed_plane_ports(recipe, roots.front().stream,
                                   a048_contract);

  // A single untimed native preflight records whether exact-domain wrappers
  // used their strict implementation or their explicitly charged raw path.
  PolicyDispatcher audit_dispatcher(true);
  (void)encode(roots, recipe, audit_dispatcher, limits);

  PolicyDispatcher measured_dispatcher(false);
  auto measured = benchmark(roots, recipe, measured_dispatcher,
                            warmups, repeats, limits);
  write_file_atomic(frame_path, measured.frame);

  auto response = base_response("run", request_id);
  response["implementation_ids"] = implementation_ids;
  response["dispatcher"] = measured_dispatcher.identity();
  response["input_path"] = input_path.string();
  response["frame_path"] = frame_path.string();
  response["dtype"] = dtype;
  response["initial_kind"] = kind_name(initial_kind);
  response["initial_width"] = initial_width;
  response["input_view"] = input_view;
  response["source_byte_width"] = source_byte_width;
  response["source_element_count"] = source_element_count;
  response["graph_element_count"] = graph_element_count;
  response["bit_adapter"] = adapter_operation;
  response["source_schema"] = source_schema;
  response["source_schema_bytes"] = source_schema_bytes.size();
  response["source_schema_sha256"] = sha256(source_schema_bytes);
  response["execution_binding"] = execution_binding;
  if (a048_contract) response["typed_plane_contract"] = *a048_contract;
  if (a048_proof) response["a048_typed_plane_proof"] = *a048_proof;
  response["input_bytes"] = input.size();
  response["input_sha256"] = sha256(input);
  response["frame_bytes"] = measured.frame.size();
  response["frame_sha256"] = sha256(measured.frame);
  response["original_numeric_bytes"] = measured.original_numeric_bytes;
  response["warmups"] = warmups;
  response["repeats"] = repeats;
  response["encode_ns"] = measured.encode_nanoseconds;
  response["decode_ns"] = measured.decode_nanoseconds;
  response["in_process_bitexact"] = true;
  response["fallback"] = fallback_json(audit_dispatcher.fallback_audit());
  response["timing_boundary"] =
      "native fixed-recipe GraphSession scheduling + kernels + complete NGF1 framing/checksum for encode; NGF1 parse/checksum + inverse scheduling + kernels for decode; excludes file I/O, JSON, process startup, recipe resolution, and untimed fallback audit";
  response["verification_boundary"] =
      "benchmark verifies every decoded root bit-for-bit in process; command=decode must be invoked in a fresh process for cold verification";
  return response;
}

Json decode_request(const Json& request, const std::string& request_id,
                    const std::vector<std::string>& implementation_ids) {
  const auto limits = parse_limits(request);
  const auto frame_path = checked_absolute_path(request, "frame_path");
  const auto output_path = checked_absolute_path(request, "output_path");
  const auto warmups = static_cast<std::uint32_t>(
      unsigned_value(request, "warmups", 0, 31));
  const auto repeats = static_cast<std::uint32_t>(
      unsigned_value(request, "repeats", 1, 1001));
  require(repeats > 0, "repeats must be positive");
  auto frame = read_file(frame_path, limits.max_frame_bytes, "frame");

  // Keep the cold timing boundary identical to graph::benchmark(): policy and
  // kernels only.  Fallback selection is already recorded by the untimed run
  // preflight; parsing it here inside inverse would add audit work to samples.
  PolicyDispatcher dispatcher(false);
  Decoded decoded;
  Json timings = Json::array();
  for (std::uint32_t iteration = 0; iteration < warmups + repeats; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    auto candidate = decode(frame, dispatcher, limits);
    const auto end = std::chrono::steady_clock::now();
    if (iteration >= warmups) timings.push_back(elapsed(begin, end));
    decoded = std::move(candidate);
  }
  require(decoded.roots.size() == 2,
          "v2 file protocol requires one data root and one source-schema root");
  const auto& root = decoded.roots.front();
  const auto source_schema = parse_source_schema(decoded.roots[1]);
  validate_decoded_binding(source_schema, request);
  write_file_atomic(output_path, root.stream.data);
  const auto output_digest = sha256(root.stream.data);

  auto response = base_response("decode", request_id);
  response["implementation_ids"] = implementation_ids;
  response["dispatcher"] = dispatcher.identity();
  response["frame_path"] = frame_path.string();
  response["output_path"] = output_path.string();
  response["frame_bytes"] = frame.size();
  response["frame_sha256"] = sha256(frame);
  response["output_bytes"] = root.stream.data.size();
  response["output_sha256"] = output_digest;
  // decode() verifies this restored root against the original digest embedded
  // in NGF1 before it returns.  Therefore this is also the verified input
  // digest without reading input_path in the decode process.
  response["input_sha256"] = output_digest;
  response["input_sha256_source"] =
      "restored root accepted against embedded NGF1 original digest";
  response["restored_kind"] = kind_name(root.stream.kind);
  response["restored_width"] = root.stream.width;
  response["restored_view"] = root.logical_view;
  response["original_numeric_payload"] = root.original_numeric_payload;
  response["original_numeric_bytes"] = decoded.original_numeric_bytes;
  response["source_schema"] = source_schema;
  if (source_schema.contains("typed_plane_contract"))
    response["typed_plane_contract"] = source_schema.at("typed_plane_contract");
  response["source_schema_present"] = true;
  response["warmups"] = warmups;
  response["repeats"] = repeats;
  response["decode_ns"] = std::move(timings);
  response["frame_verified_bitexact"] = true;
  response["fallback_audit_collected"] = false;
  response["timing_boundary"] =
      "native NGF1 parse/checksum + inverse scheduling + all inverse kernels; excludes frame/output file I/O, hashing, JSON, and process startup";
  return response;
}

}  // namespace

Json handle_request(const Json& request) {
  require(request.is_object(), "request must be an object");
  require(checked_text(request, "schema", 64) ==
              "operator-harness-request-v2",
          "wrong request schema");
  const auto phase = checked_text(request, "command", 32);
  const auto request_id = checked_text(request, "request_id", 1024);
  require(phase == "run" || phase == "decode" || phase == "capabilities",
          "unknown command");
  if (phase == "capabilities") return capabilities_response(request_id);
  const auto implementation_ids = parse_implementation_ids(request);
  if (phase == "run")
    return run_request(request, request_id, implementation_ids);
  return decode_request(request, request_id, implementation_ids);
}

Json handle_line(const std::string& line) {
  std::string phase;
  std::string request_id;
  try {
    require(line.size() <= kMaxRequestBytes, "request line is too large");
    const auto request = Json::parse(line);
    if (request.is_object()) {
      if (request.contains("command") && request.at("command").is_string())
        phase = request.at("command").get<std::string>();
      if (request.contains("request_id") && request.at("request_id").is_string())
        request_id = request.at("request_id").get<std::string>();
    }
    return handle_request(request);
  } catch (const ExcludedPolicyError& error) {
    auto code = std::string(error.what());
    if (const auto position = code.find(':'); position != std::string::npos)
      code.resize(position);
    return {{"schema", "operator-harness-response-v2"},
            {"status", "excluded"},
            {"phase", phase},
            {"request_id", request_id},
            {"reason_code", code},
            {"error", error.what()}};
  } catch (const GapError& error) {
    return {{"schema", "operator-harness-response-v2"},
            {"status", "gap"},
            {"phase", phase},
            {"request_id", request_id},
            {"reason_code", error.code()},
            {"error", error.what()}};
  } catch (const A048InputDomainError& error) {
    return {{"schema", "operator-harness-response-v2"},
            {"status", "domain_rejected"},
            {"phase", phase},
            {"request_id", request_id},
            {"reason_category", "STRICT_INPUT_DOMAIN_REJECTED"},
            {"reason_code", "A048_VALUE_EXCEEDS_BIT_WIDTH"},
            {"error", error.what()}};
  } catch (const StrictInputDomainError& error) {
    return {{"schema", "operator-harness-response-v2"},
            {"status", "domain_rejected"},
            {"phase", phase},
            {"request_id", request_id},
            {"reason_category", "STRICT_INPUT_DOMAIN_REJECTED"},
            {"reason_code", error.code()},
            {"error", error.what()}};
  } catch (const std::exception& error) {
    // The existing dispatcher uses this stable prefix when a catalog-bound
    // action is absent from the selected native build closure.  Report that
    // expected coverage limitation as a gap; all other runtime/validation
    // exceptions remain failures.
    constexpr std::string_view unsupported = "UNSUPPORTED_NATIVE_STAGE:";
    const std::string message = error.what();
    if (message.find(unsupported) != std::string::npos) {
      return {{"schema", "operator-harness-response-v2"},
              {"status", "gap"},
              {"phase", phase},
              {"request_id", request_id},
              {"reason_code", "UNSUPPORTED_NATIVE_STAGE"},
              {"error", message}};
    }
    return {{"schema", "operator-harness-response-v2"},
            {"status", "failed"},
            {"phase", phase},
            {"request_id", request_id},
            {"reason_code", "REQUEST_OR_EXECUTION_FAILED"},
            {"error", error.what()}};
  }
}

}  // namespace operator_harness_v2
