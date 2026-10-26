#pragma once

#include "stage_dispatch.hpp"
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace wc::pipeline_v1 {

inline constexpr std::uint32_t action_count = 6341;
inline constexpr std::uint32_t stop_action = 0;
inline constexpr const char* registry_fingerprint =
    "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd";

enum class ActionCategory {
  Stop, Mechanism, SplitSegments, Concat, EditType, EditPosition
};

struct ActionSpec {
  std::uint32_t action_id = 0;
  std::string key;
  std::string runtime_action;
  std::optional<std::uint32_t> target_slot, other_slot;
  std::string input_view = "bytes";
  std::string parameter_mode = "none";
  std::optional<std::int64_t> parameter_value, max_point;
  std::map<std::string, std::string> parameter_items;
  std::optional<std::string> io_variant, implementation_id, edit_kind;
  std::vector<std::string> required_parent_runtime_actions;
  std::vector<std::string> required_logical_views;
  std::uint32_t input_byte_multiple = 1;
  std::optional<std::uint32_t> required_nonserial_width;
  std::vector<std::uint32_t> allowed_physical_widths;
  bool input_nonempty = false;
  bool ordinary_operator = true;
  ActionCategory category = ActionCategory::Mechanism;
};

struct NativeSupport {
  bool executable = false;
  std::string reason;
};

struct InternalAdapter {
  std::string action;
  std::vector<std::string> required_policy_parents;
};

struct ExperimentScope {
  std::string schema;
  std::string name;
  std::string profile_path;
  std::string profile_sha256;
  std::string catalog_sha256;
  std::string registry_fingerprint;
  std::vector<std::string> excluded_implementation_ids;
  std::vector<std::uint32_t> excluded_action_ids;
  std::size_t excluded_action_count = 0;
  std::size_t enabled_action_count = 0;
  std::size_t catalog_fixed_configuration_count = 0;
  std::size_t excluded_fixed_configuration_count = 0;
  std::size_t enabled_fixed_configuration_count = 0;
};

class ActionCatalog {
 public:
  static ActionCatalog load(const std::filesystem::path&);
  void load_experiment_profile(const std::filesystem::path&);
  const ActionSpec& at(std::uint32_t id) const;
  const NativeSupport& support(std::uint32_t id) const;
  bool enabled(std::uint32_t id) const;
  const std::optional<ExperimentScope>& experiment_scope() const noexcept {
    return experiment_scope_;
  }
  const std::vector<ActionSpec>& actions() const noexcept { return actions_; }
  const std::map<std::string, InternalAdapter>& internal_adapters() const noexcept {
    return adapters_;
  }
  std::vector<std::uint32_t> executable_action_ids() const;
  std::vector<std::uint32_t> unsupported_graph_action_ids() const;
  std::size_t executable_count() const;
  bool complete_native_graph_space() const;
  const std::string& source_sha256() const noexcept { return source_sha256_; }

 private:
  std::vector<ActionSpec> actions_;
  std::vector<NativeSupport> support_;
  std::vector<bool> enabled_;
  std::map<std::string, InternalAdapter> adapters_;
  std::string source_sha256_;
  std::optional<ExperimentScope> experiment_scope_;
};

const char* category_name(ActionCategory);

}  // namespace wc::pipeline_v1
