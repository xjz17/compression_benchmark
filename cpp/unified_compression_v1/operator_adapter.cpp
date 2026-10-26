#include "operator_adapter.hpp"

#include "../native_graph_v1/graph.hpp"
#include "../native_graph_v1/graph_internal.hpp"
#include "../operator_harness_v2/binding_contract.hpp"
#include "../operator_harness_v2/policy_dispatch.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace unified_compression_v1::detail {
namespace {

using Json = nlohmann::json;
using native_graph_v1::Kind;
using native_graph_v1::Recipe;
using native_graph_v1::RootInput;
using native_graph_v1::StageRequest;
using native_graph_v1::Step;
using native_graph_v1::StepKind;
namespace fs = std::filesystem;

constexpr std::string_view kSourceSchema =
    "operator-harness-source-schema-v2";
constexpr std::string_view kSourceSchemaView =
    "operator_harness_v2_source_schema";
constexpr std::array<std::string_view, 5> kDisabledIds{
    "A106", "A107", "A108", "A109", "A110"};

[[noreturn]] void fail(std::string code, std::string message) {
  throw Error(std::move(code), std::move(message));
}

void require(bool condition, std::string code, std::string message) {
  if (!condition) fail(std::move(code), std::move(message));
}

Bytes read_file(const fs::path& path) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  require(!error, "operator_catalog_read_failed",
          "cannot stat operator catalog: " + error.message());
  require(size <= (64ULL << 20) &&
              size <= std::numeric_limits<std::size_t>::max(),
          "operator_catalog_too_large",
          "operator catalog exceeds the 64 MiB bound");
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "operator_catalog_read_failed",
          "cannot open operator catalog: " + path.string());
  Bytes result(static_cast<std::size_t>(size));
  if (!result.empty()) {
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size()));
    require(static_cast<std::size_t>(input.gcount()) == result.size(),
            "operator_catalog_read_failed", "short operator catalog read");
  }
  return result;
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int length = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), digest, &length,
                     EVP_sha256(), nullptr) == 1 &&
              length == 32,
          "sha256_failed", "OpenSSL SHA-256 failed");
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (unsigned index = 0; index < length; ++index) {
    result.push_back(alphabet[digest[index] >> 4]);
    result.push_back(alphabet[digest[index] & 15]);
  }
  return result;
}

std::string sha256(std::string_view text) {
  return sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

std::string scalar_text(const Json& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_number_unsigned())
    return std::to_string(value.get<std::uint64_t>());
  if (value.is_number_integer())
    return std::to_string(value.get<std::int64_t>());
  if (value.is_number_float()) return value.dump();
  fail("operator_catalog_invalid", "operator parameter is not scalar");
}

Kind parse_kind(std::string_view value) {
  if (value == "serial") return Kind::Serial;
  if (value == "numeric") return Kind::Numeric;
  if (value == "struct") return Kind::Struct;
  fail("operator_catalog_invalid", "unknown initial stream kind");
}

std::string_view kind_name(Kind value) {
  if (value == Kind::Serial) return "serial";
  if (value == Kind::Numeric) return "numeric";
  return "struct";
}

std::vector<std::string> supported_dtypes(std::uint32_t initial_width) {
  // This is the v2 bit-preserving adapter contract: the catalog width may
  // split a wider source value into raw lanes, but it never casts, extends, or
  // truncates a numeric value.
  if (initial_width == 1 || initial_width == 4)
    return {"int32", "int64", "float32", "float64"};
  if (initial_width == 8) return {"int64", "float64"};
  return {};
}

bool disabled_id(std::string_view id) {
  return std::find(kDisabledIds.begin(), kDisabledIds.end(), id) !=
         kDisabledIds.end();
}

bool canonical_id(std::string_view id) {
  return id.size() == 4 && id[0] == 'A' &&
         std::all_of(id.begin() + 1, id.end(), [](unsigned char c) {
           return c >= '0' && c <= '9';
         });
}

std::string implementation_id(std::string_view canonical_name) {
  constexpr std::string_view prefix = "operator:";
  require(canonical_name.starts_with(prefix), "invalid_operator_name",
          "operator decoder requires canonical name operator:Axxx");
  const auto id = canonical_name.substr(prefix.size());
  require(canonical_id(id), "invalid_operator_name",
          "operator decoder requires canonical name operator:Axxx");
  require(!disabled_id(id), "operator_disabled",
          "disabled implementation " + std::string(id) +
              " (PPMd/DMC/PAQ8/CMIX/LSTM/NNCP remain disabled)");
  return std::string(id);
}

bool needs_optional_support(const Json& pipeline) {
  static const std::set<std::string> actions{
      "op.literal_entropy.zstd_huffman.cpp_reference_v1",
      "op.run_length.bzip2_uint8.cpp_reference_v1",
      "op.block_state_control.brotli_residual_core.cpp_v1",
      "op.block_state_control.zstd_residual_core.cpp_v1",
      "op.block_state_control.blosc1_residual_core_t8.cpp_v1",
      "op.block_state_control.blosc1_residual_core.cpp_v2",
      "op.block_state_control.blosc2_residual_core_t8.cpp_v1",
      "op.block_state_control.blosc2_residual_core.cpp_v2",
      "op.block_state_control.bzip2_residual_core.cpp_v1",
      "op.block_state_control.zlib_deflate_residual_core.cpp_v1",
      "op.block_state_control.deflate_residual_core.cpp_v2",
      "op.block_state_control.xz_lzma2_residual_core.cpp_v1",
      "op.block_state_control.reger_legacy_block_core.cpp_v1",
      "op.container_frame.reger3_legacy.cpp_v1",
      "op.block_state_control.gzip_deflate_residual_core.cpp_v1",
      "op.container_frame.stored_parts.cpp_v2",
      "op.block_state_control.blosc2_3_3_3_binding_macro.cpp_v1",
      "build.blosc2_physical_chunk_from_b2p1",
      "op.container_frame.blosc2_legacy_chunk.cpp_v1"};
  for (const auto& stage : pipeline) {
    if (stage.is_object() && stage.contains("action") &&
        stage.at("action").is_string() &&
        actions.contains(stage.at("action").get<std::string>()))
      return true;
  }
  return false;
}

struct CompiledRecipe {
  Json json;
  Recipe native;
};

CompiledRecipe compile_recipe(const Json& row,
                              const std::string& policy_registry) {
  CompiledRecipe result;
  result.json = {{"policy_registry", policy_registry},
                 {"steps", Json::array()}};
  result.native.policy_registry = policy_registry;
  const auto& pipeline = row.at("present_pipeline");
  require(pipeline.is_array() && !pipeline.empty(), "operator_catalog_invalid",
          "operator pipeline must be nonempty");
  for (std::size_t index = 0; index < pipeline.size(); ++index) {
    const auto& stage = pipeline.at(index);
    StageRequest request;
    request.action = stage.at("action").get<std::string>();
    request.profile = stage.at("profile").get<std::string>();
    Json parameters = Json::object();
    for (const auto& [name, value] : stage.at("parameters").items()) {
      auto text = scalar_text(value);
      request.parameters.emplace(name, text);
      parameters[name] = std::move(text);
    }
    const auto input = static_cast<std::uint32_t>(index == 0 ? 0 : index + 1);
    Step step;
    step.kind = StepKind::Stage;
    step.inputs = {input};
    step.stage = request;
    result.native.steps.push_back(std::move(step));
    result.json["steps"].push_back(
        {{"kind", "stage"},
         {"inputs", Json::array({input})},
         {"stage", {{"action", request.action},
                    {"profile", request.profile},
                    {"parameters", std::move(parameters)}}}});
  }
  Step stop;
  stop.kind = StepKind::Stop;
  result.native.steps.push_back(std::move(stop));
  result.json["steps"].push_back({{"kind", "stop"}});
  return result;
}

std::string hex_encode(std::span<const std::uint8_t> bytes) {
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const auto value : bytes) {
    result.push_back(alphabet[value >> 4]);
    result.push_back(alphabet[value & 15]);
  }
  return result;
}

unsigned hex_nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return 10U + value - 'a';
  if (value >= 'A' && value <= 'F') return 10U + value - 'A';
  fail("operator_frame_invalid", "missing bitmap is not hexadecimal");
}

Bytes hex_decode(std::string_view text) {
  require(text.size() % 2 == 0, "operator_frame_invalid",
          "missing bitmap hex length is odd");
  Bytes result(text.size() / 2);
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<std::uint8_t>(
        (hex_nibble(text[index * 2]) << 4) | hex_nibble(text[index * 2 + 1]));
  return result;
}

Json a048_contract(const Json& binding, std::uint64_t element_count,
                   std::uint64_t input_bytes) {
  const auto text = binding.at("harness_parameters").at("bit_width")
                        .get<std::string>();
  unsigned bit_width = 0;
  for (const char c : text) {
    require(c >= '0' && c <= '9', "operator_binding_invalid",
            "A048 bit_width is not decimal");
    bit_width = bit_width * 10 + static_cast<unsigned>(c - '0');
  }
  require(bit_width <= 64, "operator_binding_invalid",
          "A048 bit_width exceeds 64");
  const auto plane_count = static_cast<std::uint64_t>((bit_width + 7) / 8);
  require(element_count == 0 ||
              plane_count <=
                  std::numeric_limits<std::uint64_t>::max() / element_count,
          "operator_input_too_large", "A048 plane byte count overflows");
  const auto plane_bytes = element_count * plane_count;
  Json shapes = Json::array();
  for (std::uint64_t plane = 0; plane < plane_count; ++plane)
    shapes.push_back({{"plane_index", plane},
                      {"kind", "serial"},
                      {"width", 1},
                      {"bytes", element_count}});
  Json physical = shapes;
  if (plane_count == 0)
    physical.push_back({{"plane_index", nullptr},
                        {"kind", "serial"},
                        {"width", 1},
                        {"bytes", 0},
                        {"carrier_only", true}});
  return {{"schema", "operator-harness-a048-typed-plane-contract-v1"},
          {"bit_width", bit_width},
          {"bit_width_parameter", text},
          {"source_word_width_bytes", 8},
          {"plane_width_bytes", 1},
          {"semantic_plane_count", plane_count},
          {"physical_output_count", std::max<std::uint64_t>(plane_count, 1)},
          {"element_count", element_count},
          {"plane_order",
           "width_aligned_most_significant_byte_to_least_significant_byte"},
          {"output_shapes", std::move(shapes)},
          {"physical_output_shapes", std::move(physical)},
          {"total_plane_payload_bytes", plane_bytes},
          {"input_bytes", input_bytes},
          {"plane_payload_equals_input_bytes", plane_bytes == input_bytes},
          {"zero_plane_physical_carrier", plane_count == 0},
          {"inverse_consumes_all_planes", true}};
}

std::uint64_t unsigned_field(const Json& object, const char* name) {
  require(object.contains(name) && object.at(name).is_number_unsigned(),
          "operator_frame_invalid", std::string("invalid ") + name);
  return object.at(name).get<std::uint64_t>();
}

std::string string_field(const Json& object, const char* name) {
  require(object.contains(name) && object.at(name).is_string(),
          "operator_frame_invalid", std::string("invalid ") + name);
  return object.at(name).get<std::string>();
}

Json preflight_frame(std::span<const std::uint8_t> bytes,
                     const std::string& expected_id) {
  const auto frame = native_graph_v1::detail::parse_frame(bytes, {});
  operator_harness_v2::PolicyDispatcher dispatcher(false);
  require(frame.dispatcher == dispatcher.identity(), "operator_frame_invalid",
          "native dispatcher version differs from this build");
  require(frame.policy_registry == "operator-harness-tree-v3",
          "operator_frame_invalid", "operator policy registry is not tree v3");
  require(frame.roots.size() == 2 && frame.roots.at(0).numeric &&
              !frame.roots.at(1).numeric &&
              frame.roots.at(1).spec.kind == Kind::Serial &&
              frame.roots.at(1).spec.width == 1 &&
              frame.roots.at(1).logical_view == kSourceSchemaView,
          "operator_frame_invalid", "operator frame root table is invalid");
  const auto schema_leaf = frame.leaves.find(1);
  require(schema_leaf != frame.leaves.end() &&
              native_graph_v1::detail::spec(schema_leaf->second) ==
                  frame.roots.at(1).spec &&
              native_graph_v1::detail::sha256(schema_leaf->second.data) ==
                  frame.roots.at(1).digest,
          "operator_frame_invalid",
          "source-schema root is absent, transformed, or corrupt");
  const std::string schema_text(schema_leaf->second.data.begin(),
                                schema_leaf->second.data.end());
  const auto schema = Json::parse(schema_text);
  require(schema.is_object() && schema.value("schema", "") == kSourceSchema,
          "operator_frame_invalid", "operator source schema is invalid");
  require(schema.contains("execution_binding") &&
              schema.at("execution_binding").is_object() &&
              schema.at("execution_binding").contains("implementation_ids") &&
              schema.at("execution_binding").at("implementation_ids").is_array() &&
              schema.at("execution_binding").at("implementation_ids").size() == 1 &&
              schema.at("execution_binding").at("implementation_ids").front().is_string(),
          "operator_frame_invalid", "operator frame lacks one bound implementation");
  const auto bound_id = schema.at("execution_binding")
                            .at("implementation_ids")
                            .front()
                            .get<std::string>();
  require(canonical_id(bound_id), "operator_frame_invalid",
          "operator frame contains an invalid implementation ID");
  require(!disabled_id(bound_id), "operator_disabled",
          "disabled implementation " + bound_id +
              " (PPMd/DMC/PAQ8/CMIX/LSTM/NNCP remain disabled)");
  require(bound_id == expected_id, "inner_binding_mismatch",
          "requested operator:" + expected_id + " differs from inner operator:" +
              bound_id);
  operator_harness_v2::validate_decoded_binding(
      schema, Json{{"implementation_ids", Json::array({expected_id})}});

  const auto& binding = schema.at("execution_binding");
  require(binding.value("pipeline_stage_count", UINT64_MAX) ==
              frame.nodes.size(),
          "operator_frame_invalid",
          "operator node count differs from the immutable binding");
  require(frame.roots.at(0).spec.kind ==
                  parse_kind(string_field(schema, "initial_kind")) &&
              frame.roots.at(0).spec.width ==
                  unsigned_field(schema, "initial_width") &&
              frame.roots.at(0).logical_view ==
                  string_field(schema, "input_view"),
          "operator_frame_invalid",
          "data root table differs from the embedded source schema");

  Json recipe{{"policy_registry", frame.policy_registry},
              {"steps", Json::array()}};
  for (std::size_t index = 0; index < frame.nodes.size(); ++index) {
    const auto& node = frame.nodes.at(index);
    require(node.step.kind == StepKind::Stage &&
                node.step.policy_action_id == -1 &&
                node.step.segment_sizes.empty() &&
                node.step.inputs == std::vector<std::uint32_t>{
                                        static_cast<std::uint32_t>(
                                            index == 0 ? 0 : index + 1)},
            "operator_frame_invalid",
            "operator recipe is not the catalog single-output chain");
    if (const auto reason =
            operator_harness_v2::excluded_action_reason(node.step.stage.action))
      fail("operator_disabled", *reason + ": " + node.step.stage.action);
    Json parameters = Json::object();
    for (const auto& [name, value] : node.step.stage.parameters)
      parameters[name] = value;
    recipe["steps"].push_back(
        {{"kind", "stage"},
         {"inputs", node.step.inputs},
         {"stage", {{"action", node.step.stage.action},
                    {"profile", node.step.stage.profile},
                    {"parameters", std::move(parameters)}}}});
  }
  recipe["steps"].push_back({{"kind", "stop"}});
  require(sha256(recipe.dump()) == string_field(binding, "recipe_sha256"),
          "operator_frame_invalid",
          "operator recipe differs from the immutable catalog binding");
  return schema;
}

}  // namespace

struct OperatorAdapter::Impl {
  struct Entry {
    MethodInfo method;
    std::string id;
    Kind initial_kind = Kind::Serial;
    std::uint32_t initial_width = 1;
    std::string input_view;
    CompiledRecipe recipe;
    Json execution_binding;
  };

  std::vector<MethodInfo> methods;
  std::vector<Entry> entries;
  std::unordered_map<std::string, std::size_t> lookup;
  std::set<std::string> ambiguous_aliases;
};

OperatorAdapter::OperatorAdapter(const std::string& catalog_path) {
  try {
    auto impl = std::make_shared<Impl>();
    const auto absolute_path = fs::absolute(fs::path(catalog_path)).lexically_normal();
    const auto bytes = read_file(absolute_path);
    const auto digest = sha256(bytes);
    const auto catalog = Json::parse(bytes.begin(), bytes.end());
    require(catalog.is_object() && catalog.value("schema", "") ==
                                       "operator-harness-tree-catalog-v3",
            "operator_catalog_invalid", "operator-tree catalog must be v3");
    require(catalog.contains("operators") && catalog.at("operators").is_array() &&
                catalog.at("operators").size() == 102,
            "operator_catalog_invalid", "operator catalog must contain 102 rows");
    const auto policy_registry = catalog.value("policy_registry", "");
    require(!policy_registry.empty(), "operator_catalog_invalid",
            "operator catalog lacks policy_registry");
    const bool support_deps =
        operator_harness_v2::PolicyDispatcher{}.identity().find(
            "support-deps-on") != std::string::npos;

    std::size_t enabled_count = 0;
    std::size_t disabled_count = 0;
    for (const auto& row : catalog.at("operators")) {
      Impl::Entry entry;
      entry.id = row.at("implementation_id").get<std::string>();
      require(canonical_id(entry.id), "operator_catalog_invalid",
              "operator catalog contains an invalid implementation ID");
      const auto display_name = row.at("name").get<std::string>();
      const bool selectable =
          row.at("selection").at("selectable").get<bool>();
      require(selectable != disabled_id(entry.id), "operator_catalog_invalid",
              "catalog selectable set differs from A106-A110 policy");
      enabled_count += selectable ? 1 : 0;
      disabled_count += selectable ? 0 : 1;

      entry.initial_kind =
          parse_kind(row.at("initial_kind").get<std::string>());
      entry.initial_width = row.at("initial_width").get<std::uint32_t>();
      entry.input_view = row.at("input_view").get<std::string>();
      entry.recipe = compile_recipe(row, policy_registry);
      const auto dtypes = supported_dtypes(entry.initial_width);
      require(!dtypes.empty(), "operator_catalog_invalid",
              "catalog initial width has no bit-preserving column adapter");

      const bool dependency_available =
          support_deps || !needs_optional_support(row.at("present_pipeline"));
      entry.method.name = "operator:" + entry.id;
      entry.method.display_name = display_name;
      entry.method.kind = "operator";
      entry.method.aliases = {entry.id, display_name};
      entry.method.dtypes = dtypes;
      entry.method.enabled = selectable;
      entry.method.available = selectable && dependency_available;
      if (!selectable) {
        entry.method.reason = row.at("selection").value(
            "reason", "disabled by the frozen A106-A110 exclusion policy");
      } else if (!dependency_available) {
        entry.method.reason = "optional operator102 support dependencies are not linked";
      }

      if (selectable) {
        Json request{{"implementation_ids", Json::array({entry.id})},
                     {"catalog_path", absolute_path.string()},
                     {"catalog_sha256", digest},
                     {"initial_kind", std::string(kind_name(entry.initial_kind))},
                     {"initial_width", entry.initial_width},
                     {"input_view", entry.input_view},
                     {"recipe", entry.recipe.json}};
        entry.execution_binding =
            operator_harness_v2::validate_catalog_binding(request);
      }

      entry.method.contract =
          {{"schema", "unified-operator-method-contract-v1"},
           {"implementation_id", entry.id},
           {"policy_disabled", !selectable},
           {"catalog_schema", catalog.at("schema")},
           {"catalog_file_sha256", digest},
           {"batch_id", row.at("batch_id")},
           {"layer", row.at("layer")},
           {"initial_kind", row.at("initial_kind")},
           {"initial_width", entry.initial_width},
           {"input_view", entry.input_view},
           {"input_adapter_kind", row.at("input_adapter_kind")},
           {"recipe", entry.recipe.json},
           {"domain_execution", row.at("domain_execution")},
           {"source_representation",
            "original_little_endian_compact_valid_numeric_bytes"},
           {"adapter_rule",
            "identity_when_source_width_equals_initial_width; otherwise "
            "split_each_source_value_into_ordered_raw_bit_lanes when the "
            "catalog initial width exactly divides the source dtype width"},
           {"bit_preserving", true},
           {"numeric_conversion", false},
           {"missing_bitmap_charged_in_frame", true},
           {"automatic_backend_codec", nullptr}};
      if (selectable)
        entry.method.contract["execution_binding"] = entry.execution_binding;

      const auto index = impl->entries.size();
      auto add_lookup = [&](const std::string& name) {
        if (impl->ambiguous_aliases.contains(name)) return;
        const auto [position, inserted] = impl->lookup.emplace(name, index);
        if (!inserted && position->second != index) {
          impl->lookup.erase(position);
          impl->ambiguous_aliases.insert(name);
        }
      };
      add_lookup(entry.method.name);
      add_lookup(entry.id);
      add_lookup(display_name);
      impl->methods.push_back(entry.method);
      impl->entries.push_back(std::move(entry));
    }
    require(enabled_count == 97 && disabled_count == 5,
            "operator_catalog_invalid",
            "operator catalog must contain 97 enabled and 5 disabled rows");
    impl_ = std::move(impl);
  } catch (const Error&) {
    throw;
  } catch (const std::exception& error) {
    fail("operator_catalog_invalid", error.what());
  }
}

const std::vector<MethodInfo>& OperatorAdapter::methods() const noexcept {
  return impl_->methods;
}

Bytes OperatorAdapter::encode(std::string_view canonical_name,
                              const Column& column,
                              nlohmann::json* execution_audit) const {
  try {
    validate_column(column);
    const auto found = impl_->lookup.find(std::string(canonical_name));
    require(found != impl_->lookup.end(), "unknown_operator",
            "unknown or ambiguous operator name: " + std::string(canonical_name));
    const auto& entry = impl_->entries.at(found->second);
    require(entry.method.enabled, "operator_disabled",
            "disabled implementation " + entry.id +
                " (PPMd/DMC/PAQ8/CMIX/LSTM/NNCP remain disabled)");
    require(entry.method.available, "operator_unavailable", entry.method.reason);
    require(std::find(entry.method.dtypes.begin(), entry.method.dtypes.end(),
                      column.dtype) != entry.method.dtypes.end(),
            "operator_dtype_unsupported",
            entry.method.name + " does not accept dtype " + column.dtype);

    const auto source_width = dtype_width(column.dtype);
    require(entry.initial_width <= source_width &&
                source_width % entry.initial_width == 0 &&
                column.values.size() % entry.initial_width == 0,
            "operator_width_adapter_rejected",
            "catalog width must divide the source dtype width without a cast");
    const auto source_elements = column.values.size() / source_width;
    const auto graph_elements = column.values.size() / entry.initial_width;
    Json schema{{"schema", kSourceSchema},
                {"source_dtype", column.dtype},
                {"source_byte_width", source_width},
                {"source_element_count", source_elements},
                {"source_row_count", column.row_count},
                {"source_value_count", source_elements},
                {"source_value_bytes", column.values.size()},
                {"missing_encoding", "lsb_first_set_means_null"},
                {"missing_bytes", column.missing.size()},
                {"missing_hex", hex_encode(column.missing)},
                {"adapter_operation",
                 entry.initial_width == source_width ? "identity"
                                                     : "raw_bit_lane_split"},
                {"initial_kind", std::string(kind_name(entry.initial_kind))},
                {"initial_width", entry.initial_width},
                {"input_view", entry.input_view},
                {"graph_element_count", graph_elements},
                {"input_bytes", column.values.size()},
                {"execution_binding", entry.execution_binding}};
    if (entry.execution_binding.value("execution_variant", "") ==
        "pure_typed_plane")
      schema["typed_plane_contract"] =
          a048_contract(entry.execution_binding, graph_elements,
                        column.values.size());
    const auto schema_text = schema.dump();
    Bytes schema_bytes(schema_text.begin(), schema_text.end());
    std::vector<RootInput> roots{
        {{entry.initial_kind, entry.initial_width, column.values}, true,
         entry.input_view},
        {{Kind::Serial, 1, std::move(schema_bytes)}, false,
         std::string(kSourceSchemaView)}};
    operator_harness_v2::PolicyDispatcher dispatcher(execution_audit != nullptr);
    auto frame = native_graph_v1::encode(roots, entry.recipe.native, dispatcher);
    if (execution_audit) {
      const auto& audit = dispatcher.fallback_audit();
      *execution_audit = {{"strict", audit.raw_fallback_stage_count == 0},
          {"fallback_used", audit.raw_fallback_stage_count != 0},
          {"wrapped_stage_count", audit.wrapped_stage_count},
          {"raw_fallback_stage_count", audit.raw_fallback_stage_count},
          {"reasons", audit.reasons}, {"reason_counts", audit.reason_counts},
          {"candidate_id", entry.method.name}};
    }
    return frame;
  } catch (const Error&) {
    throw;
  } catch (const operator_harness_v2::StrictInputDomainError& error) {
    fail(error.code(), error.what());
  } catch (const operator_harness_v2::ExcludedPolicyError& error) {
    fail("operator_disabled", error.what());
  } catch (const std::exception& error) {
    fail("operator_encode_failed", error.what());
  }
}

Column OperatorAdapter::decode(std::string_view expected_name,
                               std::span<const std::uint8_t> frame) {
  const auto id = implementation_id(expected_name);
  try {
    const auto preflight_schema = preflight_frame(frame, id);
    operator_harness_v2::PolicyDispatcher dispatcher(false);
    auto decoded = native_graph_v1::decode(frame, dispatcher);
    require(decoded.roots.size() == 2, "operator_frame_invalid",
            "operator frame must contain data and source-schema roots");
    auto& data_root = decoded.roots.at(0);
    const auto& schema_root = decoded.roots.at(1);
    require(data_root.original_numeric_payload &&
                !schema_root.original_numeric_payload &&
                schema_root.stream.kind == Kind::Serial &&
                schema_root.stream.width == 1 &&
                schema_root.logical_view == kSourceSchemaView,
            "operator_frame_invalid", "operator frame root contract is invalid");
    const std::string schema_text(schema_root.stream.data.begin(),
                                  schema_root.stream.data.end());
    const auto schema = Json::parse(schema_text);
    require(schema.is_object() && schema.value("schema", "") == kSourceSchema,
            "operator_frame_invalid", "operator source schema is invalid");
    require(schema == preflight_schema, "operator_frame_invalid",
            "restored source schema differs from the preflighted frame leaf");
    operator_harness_v2::validate_decoded_binding(
        schema, Json{{"implementation_ids", Json::array({id})}});

    const auto dtype = string_field(schema, "source_dtype");
    const auto source_width = unsigned_field(schema, "source_byte_width");
    require(source_width == dtype_width(dtype), "operator_frame_invalid",
            "source dtype width differs from the embedded schema");
    const auto initial_kind = parse_kind(string_field(schema, "initial_kind"));
    const auto initial_width = unsigned_field(schema, "initial_width");
    const auto input_view = string_field(schema, "input_view");
    require(initial_width <= UINT32_MAX &&
                data_root.stream.kind == initial_kind &&
                data_root.stream.width == initial_width &&
                data_root.logical_view == input_view,
            "operator_frame_invalid",
            "restored data root differs from the embedded source schema");
    require(data_root.stream.data.size() % source_width == 0,
            "operator_frame_invalid", "restored numeric bytes are misaligned");
    const auto value_count = data_root.stream.data.size() / source_width;
    require(unsigned_field(schema, "source_element_count") == value_count &&
                unsigned_field(schema, "source_value_count") == value_count &&
                unsigned_field(schema, "source_value_bytes") ==
                    data_root.stream.data.size() &&
                unsigned_field(schema, "input_bytes") ==
                    data_root.stream.data.size() &&
                decoded.original_numeric_bytes == data_root.stream.data.size(),
            "operator_frame_invalid", "restored numeric size contract is invalid");
    require(string_field(schema, "missing_encoding") ==
                "lsb_first_set_means_null",
            "operator_frame_invalid", "unknown missing bitmap encoding");
    auto missing = hex_decode(string_field(schema, "missing_hex"));
    require(unsigned_field(schema, "missing_bytes") == missing.size(),
            "operator_frame_invalid", "missing bitmap size differs from schema");

    Column result{dtype, unsigned_field(schema, "source_row_count"),
                  std::move(data_root.stream.data), std::move(missing)};
    validate_column(result);
    return result;
  } catch (const Error&) {
    throw;
  } catch (const operator_harness_v2::ExcludedPolicyError& error) {
    fail("operator_disabled", error.what());
  } catch (const std::exception& error) {
    fail("operator_decode_failed", error.what());
  }
}

}  // namespace unified_compression_v1::detail
