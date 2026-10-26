#include "catalog.hpp"
#include "policy.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>

namespace wc::pipeline_v1 {
namespace {
using J = nlohmann::json;
void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument(message);
}
std::string hex(std::span<const std::uint8_t> bytes) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const auto byte : bytes) out << std::setw(2) << unsigned(byte);
  return out.str();
}
std::string scalar(const J& value) {
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_number_integer()) return std::to_string(value.get<std::int64_t>());
  if (value.is_number_unsigned()) return std::to_string(value.get<std::uint64_t>());
  if (value.is_string()) return value.get<std::string>();
  throw std::invalid_argument("catalog parameter_items must contain scalar values");
}
ActionCategory category(const std::string& value) {
  if (value == "stop") return ActionCategory::Stop;
  if (value == "mechanism") return ActionCategory::Mechanism;
  if (value == "split_segments") return ActionCategory::SplitSegments;
  if (value == "concat") return ActionCategory::Concat;
  if (value == "edit_type") return ActionCategory::EditType;
  if (value == "edit_position") return ActionCategory::EditPosition;
  throw std::invalid_argument("unknown action category: " + value);
}
template<class T> std::optional<T> optional(const J& row, const char* key) {
  if (!row.contains(key) || row.at(key).is_null()) return std::nullopt;
  return row.at(key).get<T>();
}
std::vector<std::string> strings(const J& value) {
  require(value.is_array(), "catalog string list is not an array");
  return value.get<std::vector<std::string>>();
}
std::vector<std::uint32_t> widths(const J& value) {
  require(value.is_array(), "catalog width list is not an array");
  return value.get<std::vector<std::uint32_t>>();
}
bool is_graph(ActionCategory value) {
  return value == ActionCategory::Mechanism || value == ActionCategory::SplitSegments ||
         value == ActionCategory::Concat;
}
}  // namespace

const char* category_name(ActionCategory value) {
  switch (value) {
    case ActionCategory::Stop: return "stop";
    case ActionCategory::Mechanism: return "mechanism";
    case ActionCategory::SplitSegments: return "split_segments";
    case ActionCategory::Concat: return "concat";
    case ActionCategory::EditType: return "edit_type";
    case ActionCategory::EditPosition: return "edit_position";
  }
  return "unknown";
}

ActionCatalog ActionCatalog::load(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot open frozen complete action catalog: " + path.string());
  std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(input)), {});
  require(!raw.empty() && raw.size() <= 64ULL * 1024 * 1024,
          "empty or oversized frozen complete action catalog");
  const J document = J::parse(raw.begin(), raw.end());
  require(document.at("schema") == "native-pipeline-v1-frozen-complete-action-catalog",
          "unexpected native catalog schema");
  require(document.at("action_count") == action_count && document.at("stop_action") == stop_action,
          "native catalog action identity mismatch");
  require(document.at("registry_fingerprint") == registry_fingerprint,
          "native catalog registry fingerprint mismatch");
  const auto& payload = document.at("registry_payload");
  require(payload.at("schema") == "weight-conditioned-fixed-graph-actions-v1",
          "unexpected production action payload schema");
  // nlohmann's object storage is key ordered. Its compact dump matches the
  // Python sort_keys=True,separators=(\",\",\":\") fingerprint representation.
  const auto canonical = payload.dump();
  const auto digest = weighted_grpo::native_policy_v1::sha256(
      {reinterpret_cast<const std::uint8_t*>(canonical.data()), canonical.size()});
  require(hex(digest) == registry_fingerprint,
          "production action payload fingerprint does not match its frozen identity");

  ActionCatalog out;
  out.source_sha256_ = hex(weighted_grpo::native_policy_v1::sha256(raw));
  for (const auto& [consumer, value] : payload.at("internal_parent_adapters").items()) {
    out.adapters_.emplace(consumer, InternalAdapter{
        value.at("adapter").get<std::string>(),
        value.at("required_policy_parents").get<std::vector<std::string>>()});
  }
  // A060's frozen policy action represents the limb-batch match mechanism.
  // Its required lo/hi-limb transform is an execution-only graph node, just
  // like the other cataloged internal carrier adapters: it never receives a
  // policy action ID and therefore does not change the 6,341-action identity.
  out.adapters_.try_emplace(
      "op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1",
      InternalAdapter{"op.block_transform.int64_lo_hi32.cpp_v1", {}});
  const auto& actions = payload.at("actions");
  require(actions.is_array() && actions.size() == action_count,
          "production catalog must contain exactly 6341 rows");
  out.actions_.reserve(action_count);
  for (std::uint32_t index = 0; index < action_count; ++index) {
    const auto& row = actions.at(index);
    ActionSpec value;
    value.action_id = row.at("action_id");
    require(value.action_id == index, "production catalog action IDs are not contiguous");
    value.key = row.at("key"); value.runtime_action = row.at("runtime_action");
    value.target_slot = optional<std::uint32_t>(row, "target_slot");
    value.other_slot = optional<std::uint32_t>(row, "other_slot");
    value.input_view = row.at("input_view");
    value.parameter_mode = row.at("parameter_mode");
    value.parameter_value = optional<std::int64_t>(row, "parameter_value");
    value.max_point = optional<std::int64_t>(row, "max_point");
    value.io_variant = optional<std::string>(row, "io_variant");
    value.implementation_id = optional<std::string>(row, "implementation_id");
    value.edit_kind = optional<std::string>(row, "edit_kind");
    value.required_parent_runtime_actions = strings(row.at("required_parent_runtime_actions"));
    value.required_logical_views = strings(row.at("required_logical_views"));
    value.input_byte_multiple = row.at("input_byte_multiple");
    value.required_nonserial_width = optional<std::uint32_t>(row, "required_nonserial_width");
    value.allowed_physical_widths = widths(row.at("allowed_physical_widths"));
    value.input_nonempty = row.at("input_nonempty");
    value.ordinary_operator = row.at("ordinary_operator");
    value.category = category(row.at("category"));
    for (const auto& item : row.at("parameter_items")) {
      require(item.is_array() && item.size() == 2, "invalid parameter_items entry");
      require(value.parameter_items.emplace(item[0].get<std::string>(), scalar(item[1])).second,
              "duplicate fixed catalog parameter");
    }
    require(value.input_byte_multiple > 0, "catalog input-byte multiple must be positive");
    out.actions_.push_back(std::move(value));
  }
  require(out.actions_.front().category == ActionCategory::Stop &&
          out.actions_.front().runtime_action == "stop", "action zero must be STOP");
  out.enabled_.assign(action_count, true);

  std::set<std::string> capabilities;
  for (const auto& row : native_graph_v1::native_stage_capabilities())
    capabilities.insert(row.action);
  out.support_.resize(action_count);
  for (const auto& action : out.actions_) {
    auto& status = out.support_[action.action_id];
    if (action.category == ActionCategory::Stop || action.category == ActionCategory::EditType ||
        action.category == ActionCategory::EditPosition || action.category == ActionCategory::SplitSegments ||
        action.category == ActionCategory::Concat) {
      status = {true, {}};
      continue;
    }
    if (!capabilities.contains(action.runtime_action)) {
      status.reason = "UNSUPPORTED_NATIVE_STAGE: " + action.runtime_action;
      continue;
    }
    if (action.io_variant && *action.io_variant != action.runtime_action) {
      const bool exact_decimal_carrier = action.runtime_action == "op.scaling.exact_decimal.cpp_v2" &&
          (*action.io_variant == "op.scaling.simple8b.cpp_v1" ||
           *action.io_variant == "op.scaling.ts_2diff.cpp_v1");
      if (!exact_decimal_carrier) {
        status.reason = "UNREPRESENTABLE_IO_VARIANT: " + *action.io_variant;
        continue;
      }
    }
    const auto adapter = out.adapters_.find(action.runtime_action);
    if (adapter != out.adapters_.end() && !capabilities.contains(adapter->second.action)) {
      status.reason = "UNSUPPORTED_INTERNAL_ADAPTER: " + adapter->second.action;
      continue;
    }
    status.executable = true;
  }
  return out;
}

void ActionCatalog::load_experiment_profile(const std::filesystem::path& path) {
  require(!experiment_scope_.has_value(), "experiment action scope is already loaded");
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot open experiment action scope: " + path.string());
  std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(input)), {});
  require(!raw.empty() && raw.size() <= 4ULL * 1024 * 1024,
          "empty or oversized experiment action scope");
  const J document = J::parse(raw.begin(), raw.end());
  require(document.at("schema") == "native-experiment-action-scope-v1",
          "unexpected experiment action scope schema");
  require(document.at("registry_fingerprint") == registry_fingerprint &&
              document.at("catalog_sha256") == source_sha256_ &&
              document.at("catalog_action_count") == action_count,
          "experiment action scope catalog identity mismatch");

  ExperimentScope scope;
  scope.schema = document.at("schema");
  scope.name = document.at("name");
  scope.profile_path = std::filesystem::canonical(path).string();
  scope.profile_sha256 = hex(weighted_grpo::native_policy_v1::sha256(raw));
  scope.catalog_sha256 = document.at("catalog_sha256");
  scope.registry_fingerprint = document.at("registry_fingerprint");
  scope.excluded_implementation_ids =
      document.at("excluded_implementation_ids").get<std::vector<std::string>>();
  scope.excluded_action_ids =
      document.at("excluded_action_ids").get<std::vector<std::uint32_t>>();
  require(std::is_sorted(scope.excluded_implementation_ids.begin(),
                         scope.excluded_implementation_ids.end()) &&
              std::adjacent_find(scope.excluded_implementation_ids.begin(),
                                 scope.excluded_implementation_ids.end()) ==
                  scope.excluded_implementation_ids.end(),
          "experiment excluded implementation IDs must be sorted and unique");
  require(std::is_sorted(scope.excluded_action_ids.begin(),
                         scope.excluded_action_ids.end()) &&
              std::adjacent_find(scope.excluded_action_ids.begin(),
                                 scope.excluded_action_ids.end()) ==
                  scope.excluded_action_ids.end(),
          "experiment excluded action IDs must be sorted and unique");
  const std::set<std::string> excluded_implementations(
      scope.excluded_implementation_ids.begin(),
      scope.excluded_implementation_ids.end());
  require(excluded_implementations == std::set<std::string>{"A109", "A110"},
          "this experiment profile must exclude exactly CMIX A109 and NNCP A110");

  std::vector<std::uint32_t> derived_excluded;
  std::size_t catalog_fixed = 0, excluded_fixed = 0;
  for (const auto& action : actions_) {
    if (action.category == ActionCategory::Mechanism && action.target_slot == 0)
      ++catalog_fixed;
    if (action.implementation_id &&
        excluded_implementations.contains(*action.implementation_id)) {
      require(action.category == ActionCategory::Mechanism,
              "excluded implementation contains a non-mechanism action");
      derived_excluded.push_back(action.action_id);
      if (action.target_slot == 0) ++excluded_fixed;
    }
  }
  require(derived_excluded == scope.excluded_action_ids,
          "experiment excluded action IDs do not exactly cover A109/A110");
  require(!std::binary_search(scope.excluded_action_ids.begin(),
                              scope.excluded_action_ids.end(), stop_action),
          "experiment scope cannot exclude STOP");

  scope.excluded_action_count = scope.excluded_action_ids.size();
  scope.enabled_action_count = action_count - scope.excluded_action_count;
  scope.catalog_fixed_configuration_count = catalog_fixed;
  scope.excluded_fixed_configuration_count = excluded_fixed;
  scope.enabled_fixed_configuration_count = catalog_fixed - excluded_fixed;
  require(document.at("excluded_action_count") == scope.excluded_action_count &&
              document.at("enabled_action_count") == scope.enabled_action_count &&
              document.at("catalog_fixed_configuration_count") ==
                  scope.catalog_fixed_configuration_count &&
              document.at("excluded_fixed_configuration_count") ==
                  scope.excluded_fixed_configuration_count &&
              document.at("enabled_fixed_configuration_count") ==
                  scope.enabled_fixed_configuration_count,
          "experiment action scope declared counts do not match the catalog");
  require(scope.excluded_action_count == 64 &&
              scope.enabled_action_count == 6277 &&
              scope.catalog_fixed_configuration_count == 193 &&
              scope.excluded_fixed_configuration_count == 2 &&
              scope.enabled_fixed_configuration_count == 191,
          "experiment action scope does not match the frozen no-CMIX/no-NNCP space");

  enabled_.assign(action_count, true);
  for (const auto id : scope.excluded_action_ids) enabled_.at(id) = false;
  experiment_scope_ = std::move(scope);
}

const ActionSpec& ActionCatalog::at(std::uint32_t id) const {
  if (id >= actions_.size()) throw std::out_of_range("unknown fixed graph action");
  return actions_[id];
}
const NativeSupport& ActionCatalog::support(std::uint32_t id) const {
  if (id >= support_.size()) throw std::out_of_range("unknown fixed graph action");
  return support_[id];
}
bool ActionCatalog::enabled(std::uint32_t id) const {
  if (id >= enabled_.size()) throw std::out_of_range("unknown fixed graph action");
  return enabled_[id];
}
std::vector<std::uint32_t> ActionCatalog::executable_action_ids() const {
  std::vector<std::uint32_t> result;
  for (std::uint32_t id = 0; id < support_.size(); ++id)
    if (support_[id].executable) result.push_back(id);
  return result;
}
std::vector<std::uint32_t> ActionCatalog::unsupported_graph_action_ids() const {
  std::vector<std::uint32_t> result;
  for (const auto& row : actions_)
    if (is_graph(row.category) && !support_[row.action_id].executable) result.push_back(row.action_id);
  return result;
}
std::size_t ActionCatalog::executable_count() const {
  return std::count_if(support_.begin(), support_.end(), [](const auto& s) { return s.executable; });
}
bool ActionCatalog::complete_native_graph_space() const {
  return unsupported_graph_action_ids().empty();
}

}  // namespace wc::pipeline_v1
