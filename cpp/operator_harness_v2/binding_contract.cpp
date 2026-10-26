#include "binding_contract.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace operator_harness_v2 {
namespace {

using Json = nlohmann::json;
namespace fs = std::filesystem;

constexpr std::uint64_t kMaxCatalogBytes = 64ULL << 20;
constexpr std::array<std::string_view, 5> kDisabledIds{
    "A106", "A107", "A108", "A109", "A110"};
constexpr std::string_view kBatchCatalogSchema =
    "operator-harness-batch-catalog-v2";
constexpr std::string_view kTreeCatalogSchema =
    "operator-harness-tree-catalog-v3";
constexpr std::string_view kA048Action = "harness.a048.byte_planes.v1";

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument("catalog binding: " + message);
}

void require(bool condition, const std::string& message) {
  if (!condition) invalid(message);
}

bool lower_hex_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int length = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), digest, &length,
                     EVP_sha256(), nullptr) == 1 &&
              length == 32,
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

std::string sha256(std::string_view text) {
  return sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

std::vector<std::uint8_t> read_catalog(const fs::path& path) {
  require(path.is_absolute(), "catalog_path must be absolute");
  std::error_code error;
  const auto size = fs::file_size(path, error);
  require(!error, "cannot stat catalog_path: " + error.message());
  require(size <= kMaxCatalogBytes &&
              size <= std::numeric_limits<std::size_t>::max(),
          "catalog exceeds the size limit");
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open catalog_path");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  if (!result.empty()) {
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size()));
    require(static_cast<std::size_t>(input.gcount()) == result.size(),
            "short catalog read");
  }
  return result;
}

std::string required_string(const Json& object, const char* field) {
  require(object.is_object() && object.contains(field) &&
              object.at(field).is_string(),
          std::string(field) + " must be a string");
  const auto value = object.at(field).get<std::string>();
  require(!value.empty() && value.find('\0') == std::string::npos,
          std::string(field) + " is empty or invalid");
  return value;
}

std::string single_implementation_id(const Json& request) {
  require(request.is_object() && request.contains("implementation_ids") &&
              request.at("implementation_ids").is_array() &&
              request.at("implementation_ids").size() == 1 &&
              request.at("implementation_ids").front().is_string(),
          "implementation_ids must contain exactly one string");
  const auto id = request.at("implementation_ids").front().get<std::string>();
  require(id.size() == 4 && id.front() == 'A' &&
              std::all_of(id.begin() + 1, id.end(), [](unsigned char c) {
                return c >= '0' && c <= '9';
              }),
          "invalid implementation ID");
  require(std::find(kDisabledIds.begin(), kDisabledIds.end(), id) ==
              kDisabledIds.end(),
          "disabled implementation ID " + id);
  return id;
}

std::string normalized_parameter(const Json& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_number_unsigned())
    return std::to_string(value.get<std::uint64_t>());
  if (value.is_number_integer())
    return std::to_string(value.get<std::int64_t>());
  if (value.is_number_float()) return value.dump();
  invalid("catalog stage parameter is not a scalar");
}

Json normalized_parameters(const Json& parameters) {
  require(parameters.is_object(), "catalog stage parameters must be an object");
  Json result = Json::object();
  for (const auto& [key, value] : parameters.items())
    result[key] = normalized_parameter(value);
  return result;
}

void validate_actual_parameters(const Json& parameters) {
  require(parameters.is_object(), "request stage parameters must be an object");
  for (const auto& [key, value] : parameters.items()) {
    require(!key.empty() && value.is_string(),
            "request stage parameters must be string-valued");
    require(value.get_ref<const std::string&>().find('\0') == std::string::npos,
            "request stage parameter contains NUL");
  }
}

unsigned canonical_decimal(const Json& value, const std::string& label,
                           unsigned maximum) {
  require(value.is_string(), label + " must be a string");
  const auto text = value.get<std::string>();
  require(!text.empty() && text.size() <= 2 &&
              std::all_of(text.begin(), text.end(), [](unsigned char c) {
                return c >= '0' && c <= '9';
              }) &&
              (text == "0" || text.front() != '0'),
          label + " must be canonical decimal");
  unsigned result = 0;
  for (const auto c : text)
    result = result * 10 + static_cast<unsigned>(c - '0');
  require(result <= maximum, label + " exceeds its catalog domain");
  return result;
}

void validate_exact_keys(const Json& object,
                         std::initializer_list<std::string_view> required,
                         std::initializer_list<std::string_view> optional,
                         const char* label) {
  require(object.is_object(), std::string(label) + " must be an object");
  std::set<std::string_view> allowed(required.begin(), required.end());
  allowed.insert(optional.begin(), optional.end());
  for (const auto& [key, _] : object.items())
    require(allowed.contains(key),
            std::string(label) + " contains unsupported field " + key);
  for (const auto key : required)
    require(object.contains(std::string(key)),
            std::string(label) + " lacks field " + std::string(key));
}

void validate_catalog_shape(const Json& catalog) {
  require(catalog.is_object() && catalog.contains("schema") &&
              catalog.at("schema").is_string() &&
              (catalog.at("schema") == kBatchCatalogSchema ||
               catalog.at("schema") == kTreeCatalogSchema),
          "wrong catalog schema");
  require(catalog.contains("operators") && catalog.at("operators").is_array() &&
              catalog.at("operators").size() == 102,
          "catalog must contain exact102 rows");
  require(catalog.contains("counts") && catalog.at("counts").is_object() &&
              catalog.at("counts").value("all_trace_entries", 0) == 102 &&
              catalog.at("counts").value("eligible", 0) == 97 &&
              catalog.at("counts").value("disabled_by_policy", 0) == 5,
          "catalog count contract is not exact102/exact97/exact5");
  require(catalog.contains("policy") && catalog.at("policy").is_object() &&
              catalog.at("policy").value("runtime_override_allowed", true) ==
                  false &&
              catalog.at("policy").contains("disabled_ids") &&
              catalog.at("policy").at("disabled_ids").is_array(),
          "catalog exclusion policy is invalid");
  std::set<std::string> disabled;
  for (const auto& value : catalog.at("policy").at("disabled_ids")) {
    require(value.is_string(), "catalog disabled ID is not a string");
    disabled.insert(value.get<std::string>());
  }
  require(disabled == std::set<std::string>(kDisabledIds.begin(),
                                            kDisabledIds.end()),
          "catalog disabled set is not A106-A110");

  std::set<std::string> ids;
  std::set<std::string> nonselectable;
  std::size_t eligible = 0;
  for (const auto& row : catalog.at("operators")) {
    const auto id = required_string(row, "implementation_id");
    require(id.size() == 4 && id.front() == 'A' &&
                std::all_of(id.begin() + 1, id.end(), [](unsigned char c) {
                  return c >= '0' && c <= '9';
                }),
            "catalog contains an invalid implementation ID");
    require(ids.emplace(id).second, "duplicate catalog implementation ID " + id);
    require(row.contains("selection") && row.at("selection").is_object() &&
                row.at("selection").contains("selectable") &&
                row.at("selection").at("selectable").is_boolean() &&
                row.at("selection").contains("overridable") &&
                row.at("selection").at("overridable").is_boolean() &&
                !row.at("selection").at("overridable").get<bool>(),
            id + " lacks a selectable policy");
    if (row.at("selection").at("selectable").get<bool>()) {
      ++eligible;
      require(row.contains("batch_id") && row.at("batch_id").is_string() &&
                  !row.at("batch_id").get_ref<const std::string&>().empty(),
              id + " eligible row lacks a batch");
    } else {
      nonselectable.insert(id);
      require(row.contains("batch_id") && row.at("batch_id").is_null(),
              id + " disabled row unexpectedly belongs to a batch");
    }
  }
  require(ids.size() == 102 && eligible == 97,
          "catalog rows do not independently reproduce exact102/exact97");
  require(nonselectable == disabled,
          "catalog nonselectable rows differ from A106-A110 policy");

  if (catalog.at("schema") == kTreeCatalogSchema) {
    require(catalog.value("policy_registry", "") ==
                "operator-harness-tree-v3" &&
                catalog.contains("parameterized_operator_ids") &&
                catalog.at("parameterized_operator_ids") ==
                    Json::array({"A048"}),
            "operator-tree catalog parameterized ID contract is invalid");
    for (const auto& row : catalog.at("operators")) {
      const auto id = row.at("implementation_id").get<std::string>();
      if (id != "A048") {
        require(!row.contains("harness_parameter_domains"),
                id + " unexpectedly declares a harness parameter domain");
        continue;
      }
      require(row.value("initial_kind", "") == "numeric" &&
                  row.value("initial_width", 0) == 8 &&
                  row.value("input_view", "") == "uint64" &&
                  row.value("focal_step_index", 1) == 0 &&
                  row.at("present_pipeline").size() == 1,
              "A048 pure typed-plane input or pipeline contract is invalid");
      const auto& stage = row.at("present_pipeline").front();
      require(stage.value("action", "") == kA048Action &&
                  stage.value("profile", "") == "uint64" &&
                  stage.contains("parameters") &&
                  stage.at("parameters") == Json{{"bit_width", "64"}},
              "A048 pure typed-plane default stage is invalid");
      require(row.contains("harness_parameter_domains") &&
                  row.at("harness_parameter_domains").is_object() &&
                  row.at("harness_parameter_domains").size() == 1 &&
                  row.at("harness_parameter_domains").contains("bit_width"),
              "A048 must declare exactly the bit_width parameter domain");
      const auto& domain =
          row.at("harness_parameter_domains").at("bit_width");
      validate_exact_keys(
          domain,
          {"type", "minimum", "maximum", "default", "step_index",
           "action", "profile"},
          {}, "A048 bit_width domain");
      require(domain.at("type") == "canonical_decimal_integer_string" &&
                  domain.at("minimum") == 0 && domain.at("maximum") == 64 &&
                  domain.at("default") == "64" &&
                  domain.at("step_index") == 0 &&
                  domain.at("action") == kA048Action &&
                  domain.at("profile") == "uint64",
              "A048 bit_width domain differs from the closed [0,64] contract");
    }
  }

  const auto content_sha = required_string(catalog, "catalog_sha256");
  require(lower_hex_sha256(content_sha), "invalid catalog content SHA-256");
  auto unhashed = catalog;
  unhashed.erase("catalog_sha256");
  require(sha256(unhashed.dump()) == content_sha,
          "catalog canonical content SHA-256 mismatch");
}

const Json& selected_row(const Json& catalog, const std::string& id) {
  const Json* selected = nullptr;
  for (const auto& row : catalog.at("operators")) {
    if (row.at("implementation_id") == id) {
      require(selected == nullptr, "duplicate selected implementation row");
      selected = &row;
    }
  }
  require(selected != nullptr, "implementation ID is absent from catalog");
  const auto& selection = selected->at("selection");
  require(selection.value("selectable", false) &&
              !selection.value("overridable", true) &&
              !selected->at("batch_id").is_null(),
          id + " is not an eligible non-overridable catalog row");
  return *selected;
}

void validate_typed_input(const Json& request, const Json& row) {
  require(required_string(request, "initial_kind") ==
              required_string(row, "initial_kind"),
          "initial_kind differs from selected catalog row");
  require(request.contains("initial_width") &&
              request.at("initial_width").is_number_unsigned() &&
              row.contains("initial_width") &&
              row.at("initial_width").is_number_unsigned(),
          "initial_width must be an integer");
  require(request.at("initial_width") == row.at("initial_width"),
          "initial_width differs from selected catalog row");
  require(required_string(request, "input_view") ==
              required_string(row, "input_view"),
          "input_view differs from selected catalog row");
}

void validate_recipe(const Json& recipe, const Json& row) {
  validate_exact_keys(recipe, {"policy_registry", "steps"}, {}, "recipe");
  (void)required_string(recipe, "policy_registry");
  require(recipe.at("steps").is_array(), "recipe steps must be an array");
  const auto& expected = row.at("present_pipeline");
  require(expected.is_array() && !expected.empty(),
          "selected catalog pipeline is empty");
  require(recipe.at("steps").size() == expected.size() + 1,
          "recipe must contain exactly the selected pipeline followed by stop");

  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& actual_step = recipe.at("steps").at(index);
    const auto& expected_step = expected.at(index);
    validate_exact_keys(actual_step, {"kind", "inputs", "stage"},
                        {"policy_action_id"}, "recipe stage step");
    require(actual_step.at("kind") == "stage",
            "selected pipeline may contain stage steps only");
    if (actual_step.contains("policy_action_id"))
      require(actual_step.at("policy_action_id").is_number_integer(),
              "policy_action_id must be an integer");
    const auto expected_input = index == 0 ? std::uint64_t{0}
                                            : std::uint64_t{index + 1};
    require(actual_step.at("inputs").is_array() &&
                actual_step.at("inputs").size() == 1 &&
                actual_step.at("inputs").front().is_number_unsigned() &&
                actual_step.at("inputs").front().get<std::uint64_t>() ==
                    expected_input,
            "recipe stage input edge is not the single-output chain 0,2,3,...");

    const auto& actual = actual_step.at("stage");
    validate_exact_keys(actual, {"action", "profile", "parameters"}, {},
                        "recipe stage");
    require(required_string(actual, "action") ==
                required_string(expected_step, "action"),
            "recipe action differs from selected catalog pipeline");
    require(required_string(actual, "profile") ==
                required_string(expected_step, "profile"),
            "recipe profile differs from selected catalog pipeline");
    validate_actual_parameters(actual.at("parameters"));
    if (row.contains("harness_parameter_domains")) {
      require(row.at("implementation_id") == "A048" && index == 0 &&
                  actual.at("action") == kA048Action &&
                  actual.at("profile") == "uint64" &&
                  actual.at("parameters").is_object() &&
                  actual.at("parameters").size() == 1 &&
                  actual.at("parameters").contains("bit_width"),
              "parameter domain is not scoped to the A048 typed-plane stage");
      (void)canonical_decimal(actual.at("parameters").at("bit_width"),
                              "A048 bit_width", 64);
    } else {
      require(actual.at("parameters") ==
                  normalized_parameters(expected_step.at("parameters")),
              "recipe parameters differ from the normalized catalog pipeline");
    }
  }

  const auto& stop = recipe.at("steps").back();
  validate_exact_keys(stop, {"kind"}, {"policy_action_id"}, "recipe stop");
  require(stop.at("kind") == "stop", "recipe does not end with stop");
  if (stop.contains("policy_action_id"))
    require(stop.at("policy_action_id").is_number_integer(),
            "stop policy_action_id must be an integer");
}

}  // namespace

Json validate_catalog_binding(const Json& request) {
  require(request.is_object(), "request must be an object");
  const auto implementation_id = single_implementation_id(request);
  const fs::path catalog_path(required_string(request, "catalog_path"));
  const auto requested_file_sha = required_string(request, "catalog_sha256");
  require(lower_hex_sha256(requested_file_sha),
          "catalog_sha256 must be lowercase SHA-256 hex");
  const auto bytes = read_catalog(catalog_path);
  require(sha256(bytes) == requested_file_sha,
          "catalog file SHA-256 differs from request");

  const Json catalog = Json::parse(bytes.begin(), bytes.end());
  validate_catalog_shape(catalog);
  const auto& row = selected_row(catalog, implementation_id);
  validate_typed_input(request, row);
  require(request.contains("recipe"), "request lacks recipe");
  validate_recipe(request.at("recipe"), row);

  const auto recipe_sha = sha256(request.at("recipe").dump());
  const bool pure_a048 = catalog.at("schema") == kTreeCatalogSchema &&
                         implementation_id == "A048";
  Json binding{
      {"schema", pure_a048 ? "operator-harness-execution-binding-v3"
                            : "operator-harness-execution-binding-v2"},
      {"implementation_ids", Json::array({implementation_id})},
      {"catalog_schema", catalog.at("schema")},
      {"catalog_sha256", requested_file_sha},
      {"catalog_content_sha256", catalog.at("catalog_sha256")},
      {"recipe_sha256", recipe_sha},
      {"execution_variant", pure_a048 ? "pure_typed_plane" : "catalog_exact"},
      {"batch_id", row.at("batch_id")},
      {"layer", row.at("layer")},
      {"dispatcher", row.at("dispatcher")},
      {"initial_kind", row.at("initial_kind")},
      {"initial_width", row.at("initial_width")},
      {"input_view", row.at("input_view")},
      {"input_adapter_kind", row.at("input_adapter_kind")},
      {"pipeline_stage_count", row.at("present_pipeline").size()}};
  if (pure_a048) {
    binding["harness_parameters"] = {
        {"bit_width",
         request.at("recipe").at("steps").front().at("stage").at(
             "parameters").at("bit_width")}};
  }
  return binding;
}

void validate_decoded_binding(const Json& source_schema, const Json& request) {
  require(source_schema.is_object() &&
              source_schema.contains("execution_binding") &&
              source_schema.at("execution_binding").is_object(),
          "decoded source schema lacks execution_binding");
  const auto& binding = source_schema.at("execution_binding");
  require(binding.value("schema", "") ==
                  "operator-harness-execution-binding-v2" ||
              binding.value("schema", "") ==
                  "operator-harness-execution-binding-v3",
          "decoded execution binding has wrong schema");
  const auto request_id = single_implementation_id(request);
  require(binding.contains("implementation_ids") &&
              binding.at("implementation_ids") == Json::array({request_id}),
          "decode implementation ID differs from frame binding");
  const auto frame_catalog_sha = required_string(binding, "catalog_sha256");
  const auto frame_catalog_content_sha =
      required_string(binding, "catalog_content_sha256");
  const auto frame_recipe_sha = required_string(binding, "recipe_sha256");
  require(lower_hex_sha256(frame_catalog_sha) &&
              lower_hex_sha256(frame_catalog_content_sha) &&
              lower_hex_sha256(frame_recipe_sha),
          "decoded execution binding contains an invalid SHA-256");
  require(required_string(source_schema, "initial_kind") ==
                  required_string(binding, "initial_kind") &&
              source_schema.contains("initial_width") &&
              source_schema.at("initial_width").is_number_unsigned() &&
              binding.contains("initial_width") &&
              binding.at("initial_width").is_number_unsigned() &&
              source_schema.at("initial_width") == binding.at("initial_width") &&
              required_string(source_schema, "input_view") ==
                  required_string(binding, "input_view"),
          "decoded source schema differs from its execution binding");
  if (binding.at("schema") == "operator-harness-execution-binding-v3") {
    require(request_id == "A048" &&
                binding.value("catalog_schema", "") == kTreeCatalogSchema &&
                binding.value("execution_variant", "") ==
                    "pure_typed_plane" &&
                binding.contains("harness_parameters") &&
                binding.at("harness_parameters").is_object() &&
                binding.at("harness_parameters").size() == 1 &&
                binding.at("harness_parameters").contains("bit_width"),
            "decoded v3 binding is not the A048 pure typed-plane variant");
    const auto bit_width = canonical_decimal(
        binding.at("harness_parameters").at("bit_width"),
        "decoded A048 bit_width", 64);
    require(source_schema.contains("typed_plane_contract") &&
                source_schema.at("typed_plane_contract").is_object() &&
                source_schema.contains("graph_element_count") &&
                source_schema.at("graph_element_count").is_number_unsigned() &&
                source_schema.contains("input_bytes") &&
                source_schema.at("input_bytes").is_number_unsigned(),
            "decoded A048 source schema lacks the typed-plane contract");
    const auto element_count =
        source_schema.at("graph_element_count").get<std::uint64_t>();
    const auto input_bytes =
        source_schema.at("input_bytes").get<std::uint64_t>();
    require(element_count <= std::numeric_limits<std::uint64_t>::max() / 8 &&
                input_bytes == element_count * 8,
            "decoded A048 input shape is inconsistent");
    const auto plane_count = static_cast<std::uint64_t>((bit_width + 7) / 8);
    require(element_count == 0 ||
                plane_count <=
                    std::numeric_limits<std::uint64_t>::max() / element_count,
            "decoded A048 plane payload size overflows");
    const auto plane_bytes = element_count * plane_count;
    const auto& contract = source_schema.at("typed_plane_contract");
    validate_exact_keys(
        contract,
        {"schema", "bit_width", "bit_width_parameter",
         "source_word_width_bytes", "plane_width_bytes",
         "semantic_plane_count", "physical_output_count", "element_count",
         "plane_order", "output_shapes", "physical_output_shapes",
         "total_plane_payload_bytes", "input_bytes",
         "plane_payload_equals_input_bytes", "zero_plane_physical_carrier",
         "inverse_consumes_all_planes"},
        {}, "A048 typed-plane contract");
    require(contract.at("schema") ==
                    "operator-harness-a048-typed-plane-contract-v1" &&
                contract.at("bit_width") == bit_width &&
                contract.at("bit_width_parameter") ==
                    binding.at("harness_parameters").at("bit_width") &&
                contract.at("source_word_width_bytes") == 8 &&
                contract.at("plane_width_bytes") == 1 &&
                contract.at("semantic_plane_count") == plane_count &&
                contract.at("physical_output_count") ==
                    std::max<std::uint64_t>(plane_count, 1) &&
                contract.at("element_count") == element_count &&
                contract.at("plane_order") ==
                    "width_aligned_most_significant_byte_to_least_significant_byte" &&
                contract.at("total_plane_payload_bytes") == plane_bytes &&
                contract.at("input_bytes") == input_bytes &&
                contract.at("plane_payload_equals_input_bytes") ==
                    (plane_bytes == input_bytes) &&
                contract.at("zero_plane_physical_carrier") ==
                    (plane_count == 0) &&
                contract.at("inverse_consumes_all_planes") == true,
            "decoded A048 typed-plane scalar contract is inconsistent");
    require(contract.at("output_shapes").is_array() &&
                contract.at("output_shapes").size() == plane_count &&
                contract.at("physical_output_shapes").is_array() &&
                contract.at("physical_output_shapes").size() ==
                    std::max<std::uint64_t>(plane_count, 1),
            "decoded A048 typed-plane output shape count is inconsistent");
    for (std::uint64_t plane = 0; plane < plane_count; ++plane) {
      const auto& shape = contract.at("output_shapes").at(plane);
      require(shape == Json{{"plane_index", plane},
                            {"kind", "serial"},
                            {"width", 1},
                            {"bytes", element_count}},
              "decoded A048 semantic plane shape is inconsistent");
      require(contract.at("physical_output_shapes").at(plane) == shape,
              "decoded A048 physical plane shape is inconsistent");
    }
    if (plane_count == 0) {
      require(contract.at("physical_output_shapes").front() ==
                  Json{{"plane_index", nullptr},
                       {"kind", "serial"},
                       {"width", 1},
                       {"bytes", 0},
                       {"carrier_only", true}},
              "decoded A048 zero-plane carrier shape is inconsistent");
    }
  } else {
    require(binding.value("execution_variant", "catalog_exact") ==
                "catalog_exact",
            "decoded v2 binding has an unexpected execution variant");
  }
  if (request.contains("catalog_sha256")) {
    const auto requested = required_string(request, "catalog_sha256");
    require(requested == frame_catalog_sha,
            "decode catalog_sha256 differs from frame binding");
  }
  if (request.contains("recipe_sha256")) {
    const auto requested = required_string(request, "recipe_sha256");
    require(requested == frame_recipe_sha,
            "decode recipe_sha256 differs from frame binding");
  }
}

}  // namespace operator_harness_v2
