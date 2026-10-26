#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace remaining_codec_actions {

enum class Evidence {
  codec_ablation_cpp,
  exact_stage_cpp,
  reference_stage_cpp,
  atomic_cpp,
  source_hook_unwired,
  proxy_only,
  non_atomic_experimental,
  blocked_no_lossless_decoder,
};

struct PipelineSpec {
  std::string_view method;
  std::string_view canonical_method;
  std::string_view kind;
  std::string_view selected_level;
  std::string_view implementation_source;
  Evidence evidence;
  std::span<const std::string_view> operators;
};

struct ConcreteAction {
  std::string name;
  std::string operator_id;
  std::string implementation_id;
  std::string implementation_source;
  std::string method;
  std::string selected_level;
  Evidence evidence;
  bool standalone_composable;
};

std::span<const PipelineSpec> pipelines();
std::vector<ConcreteAction> concrete_actions();
std::string_view evidence_name(Evidence evidence);

}  // namespace remaining_codec_actions
