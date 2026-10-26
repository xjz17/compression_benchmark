// JSONL is an untimed training/driver protocol. Compression, candidate timing,
// frozen-policy search, final framing and frame-only decode remain in-process C++.
#include "catalog.hpp"
#include "search.hpp"
#include "stage_dispatch.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <tuple>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;
namespace policy = weighted_grpo::native_policy_v1;
using J = nlohmann::json;
namespace {
void require(bool value, const std::string& message) { if (!value) throw std::invalid_argument(message); }
graph::Bytes unhex(const std::string& value) {
  require(value.size()%2==0 && value.size()<=2ULL*(256ULL<<20),"invalid/oversized data_hex");
  auto digit=[](char c)->unsigned{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;
    throw std::invalid_argument("data_hex must use lowercase canonical hex");};
  graph::Bytes out;out.reserve(value.size()/2);for(std::size_t i=0;i<value.size();i+=2)out.push_back((digit(value[i])<<4)|digit(value[i+1]));return out;
}
std::string hex(std::span<const std::uint8_t> bytes) {
  static constexpr char h[]="0123456789abcdef";std::string out;out.reserve(bytes.size()*2);
  for(auto byte:bytes){out.push_back(h[byte>>4]);out.push_back(h[byte&15]);}return out;
}
graph::Kind kind(const std::string& value) {
  if(value=="serial")return graph::Kind::Serial;if(value=="numeric")return graph::Kind::Numeric;
  if(value=="struct")return graph::Kind::Struct;throw std::invalid_argument("unknown root kind");
}
std::vector<graph::RootInput> roots(const J& input) {
  require(input.is_array()&&!input.empty()&&input.size()<=32,"roots must contain 1..32 entries");
  std::vector<graph::RootInput> out;for(const auto& row:input)out.push_back({
    {kind(row.at("kind")),row.at("width").get<std::uint32_t>(),unhex(row.at("data_hex"))},
    row.at("original_numeric_payload").get<bool>(),row.at("logical_view").get<std::string>()});return out;
}
template<std::size_t N> std::array<double,N> array(const J& input,const char* name) {
  require(input.is_array()&&input.size()==N,std::string(name)+" has the wrong width");std::array<double,N> out{};
  for(std::size_t i=0;i<N;++i){out[i]=input[i];require(std::isfinite(out[i]),std::string(name)+" contains non-finite value");}return out;
}
ngpipe::TaskRequest request(const J& row) {
  ngpipe::TaskRequest out;out.task_id=row.at("task_id");out.roots=roots(row.at("roots"));out.weights=array<3>(row.at("weights"),"weights");
  out.compression_reference_ns_per_byte=row.at("compression_reference_ns_per_byte");
  out.decompression_reference_ns_per_byte=row.at("decompression_reference_ns_per_byte");
  out.previous_recipe=row.value("previous_recipe_action_ids",std::vector<std::uint32_t>{});
  out.starting_recipe=row.value("starting_recipe_action_ids",std::vector<std::uint32_t>{});
  out.block_index=row.value("block_index",0U);out.continuation_edit_mode=row.value("continuation_edit_mode",false);
  out.max_recipe_edits=row.value("max_recipe_edits",2U);out.require_complete_native_catalog=row.value("require_complete_native_catalog",false);
  if(row.contains("original_numeric_bytes")&&!row.at("original_numeric_bytes").is_null())out.original_numeric_bytes=row.at("original_numeric_bytes");
  out.config.timing_warmups=row.value("timing_warmups",1U);out.config.timing_repetitions=row.value("timing_repetitions",5U);
  out.config.allow_large_memory=row.value("allow_large_memory",false);
  if(row.contains("limits")){const auto& l=row.at("limits");out.config.max_ordinary_depth=l.value("max_ordinary_depth",10U);
    out.config.max_graph_nodes=l.value("max_graph_nodes",64U);out.config.max_active_streams=l.value("max_active_streams",32U);
    out.config.max_policy_edits=l.value("max_policy_edits",64U);out.config.graph_limits.max_stream_bytes=l.value("max_stream_bytes",256ULL<<20);
    out.config.graph_limits.max_working_bytes=l.value("max_working_bytes",512ULL<<20);out.config.graph_limits.max_frame_bytes=l.value("max_frame_bytes",1ULL<<30);
    out.config.graph_limits.max_metadata_bytes=l.value("max_metadata_bytes",16ULL<<20);}
  return out;
}
J state_json(const ngpipe::TaskState& state) {
  J edits=J::array();for(const auto& e:state.edit_history)edits.push_back({{"kind",e.kind},{"position",e.position},{"action_id",e.action_id}});
  return {{"recipe_action_ids",state.recipe},{"recipe_valid",state.recipe_valid},{"recipe_error",state.recipe_error},
    {"policy_edits",state.policy_edits},{"replayed_nodes",state.replayed_nodes},
    {"continuation_edit_mode",state.continuation_edit_mode},{"pending_edit_kind",state.pending_edit_kind},
    {"pending_edit_index",state.pending_edit_index},{"edit_count",state.edit_count},{"edit_history",edits}};
}
J inspection_json(const ngpipe::StateInspection& value) {
  return {{"task_features",value.task_features},{"state_features",value.state_features},{"legal_action_ids",value.legal_action_ids},
    {"terminal_evaluable",value.terminal_evaluable},{"node_count",value.node_count},{"active_data_flows",value.active_data_flows},
    {"maximum_depth",value.maximum_depth}};
}
J terminal_json(const ngpipe::Evaluation& value,bool include_frame) {
  if(!value.success)return {{"terminal_status","failed"},{"error",value.failure_error},
    {"original_numeric_bytes",value.original_numeric_bytes},{"max_frame_bytes",value.max_frame_bytes},
    {"candidate_ordinal",value.candidate_ordinal},
    {"candidate_budget",value.candidate_budget}};
  const auto digest=policy::sha256(value.frame);
  return {{"terminal_status","ok"},{"original_numeric_bytes",value.original_numeric_bytes},{"frame_bytes",value.frame.size()},
    {"frame_hex",include_frame?J(hex(value.frame)):J(nullptr)},{"frame_sha256",hex(digest)},
    {"encode_ns",value.compression_time_ns},{"decode_ns",value.decompression_time_ns},
    {"encode_samples_ns",value.compression_samples_ns},{"decode_samples_ns",value.decompression_samples_ns},
    {"compression_ratio",value.compression_ratio},{"compression_ns_per_byte",value.compression_ns_per_byte},
    {"decompression_ns_per_byte",value.decompression_ns_per_byte},{"J_selection",value.objective},
    {"candidate_ordinal",value.candidate_ordinal},{"candidate_budget",value.candidate_budget}};
}
struct Entry {std::unique_ptr<ngpipe::GraphTask> task;std::map<std::string,ngpipe::TaskState> states;std::uint64_t next_state=1;};
struct SolutionLibrary {
  struct Recipe {std::string id,dtype;std::vector<std::uint32_t> actions;};
  std::vector<Recipe> recipes;std::string file_sha256,content_sha256;bool pinned=false;
};
SolutionLibrary load_library(const std::string& path,const ngpipe::ActionCatalog& catalog,
                             const std::optional<std::string>& expected_file_sha256){std::ifstream in(path,std::ios::binary);
  require(bool(in),"cannot open training-only solution library");std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)),{});
  require(!raw.empty()&&raw.size()<=64ULL*1024*1024,"empty or oversized solution library");const auto doc=J::parse(raw.begin(),raw.end());
  require(doc.at("schema")=="weighted-grpo-training-solution-library-v1"&&doc.at("source_scope")=="training_only",
          "solution library must be weighted-grpo training_only v1");
  require(doc.at("action_registry_fingerprint")==ngpipe::registry_fingerprint,"solution-library registry mismatch");
  const auto expected=doc.at("library_content_sha256").get<std::string>();auto basis=doc;basis.erase("library_content_sha256");
  const auto encoded=basis.dump();const auto actual=hex(policy::sha256({reinterpret_cast<const std::uint8_t*>(encoded.data()),encoded.size()}));
  require(actual==expected,"solution-library content fingerprint mismatch");SolutionLibrary out;out.content_sha256=expected;
  out.file_sha256=hex(policy::sha256(raw));
  if(expected_file_sha256){require(expected_file_sha256->size()==64&&
      std::all_of(expected_file_sha256->begin(),expected_file_sha256->end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}),
      "expected solution-library SHA-256 must be lowercase hex");require(out.file_sha256==*expected_file_sha256,
      "solution-library file SHA-256 mismatch");out.pinned=true;}std::set<std::string> ids;
  std::set<std::pair<std::string,std::vector<std::uint32_t>>> unique_recipes;
  for(const auto& row:doc.at("recipes")){SolutionLibrary::Recipe recipe{row.at("recipe_id"),row.at("dtype"),row.at("recipe_action_ids")};
    require(ids.insert(recipe.id).second&&!recipe.actions.empty()&&recipe.actions.size()<=64,"invalid/duplicate solution-library recipe");
    require(unique_recipes.emplace(recipe.dtype,recipe.actions).second,"duplicate solution-library dtype/recipe pair");
    require(row.at("source_split")=="train","solution-library recipe is not training split");for(const auto action:recipe.actions){
      const auto& spec=catalog.at(action);require(action!=ngpipe::stop_action&&
        (spec.category==ngpipe::ActionCategory::Mechanism||spec.category==ngpipe::ActionCategory::SplitSegments||spec.category==ngpipe::ActionCategory::Concat),
        "solution-library recipe contains STOP/edit/non-recipe action");
      require(catalog.enabled(action),"solution-library recipe contains an action excluded by the experiment scope");
    }out.recipes.push_back(std::move(recipe));}
  require(!out.recipes.empty(),"solution library has no measured recipe");return out;}
class Server {
 public:
  Server(ngpipe::ActionCatalog catalog,std::unique_ptr<policy::Policy> policy,std::unique_ptr<graph::StageDispatcher> dispatcher,
         std::optional<SolutionLibrary> library)
      :catalog_(std::move(catalog)),policy_(std::move(policy)),dispatcher_(std::move(dispatcher)),library_(std::move(library)){}
  J handle(const J& input){const auto command=input.at("cmd").get<std::string>();
    if(command=="schema")return schema();if(command=="catalog")return catalog_detail();if(command=="reset")return reset(input);if(command=="inspect")return inspect(input);if(command=="step")return step(input);
    if(command=="evaluate_terminal")return evaluate(input);if(command=="search")return search(input);
    if(command=="select_library")return select_library(input);if(command=="reset_from_library")return reset_from_library(input);
    if(command=="release_task")return release(input);throw std::invalid_argument("unknown command: "+command);}
 private:
  J execution_mappings()const{return{{"schema","native-pipeline-v1-execution-mappings"},{"mappings",J::array({
    {{"runtime_action","op.scaling.legacy_exact_decimal.cpp_v1"},{"input_view","float64"},
     {"stage_parameters",{{"max_point","18"}}},{"legality","current-block native forward probe must pass bit-exact"}},
    {{"runtime_action","op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1"},
     {"internal_parent_adapter","op.block_transform.int64_lo_hi32.cpp_v1"},
     {"policy_action_id",nullptr},{"legality","adapter and consumer native forward probe must pass"}}
  })}};}
  J experiment_scope_json()const{if(!catalog_.experiment_scope())return nullptr;
    const auto& scope=*catalog_.experiment_scope();return{{"schema",scope.schema},
      {"profile_path",scope.profile_path},{"profile_sha256",scope.profile_sha256},
      {"catalog_sha256",scope.catalog_sha256},
      {"excluded_action_ids",scope.excluded_action_ids},{"excluded_action_count",scope.excluded_action_count},
      {"enabled_action_count",scope.enabled_action_count},
      {"excluded_fixed_configuration_count",scope.excluded_fixed_configuration_count},
      {"enabled_fixed_configuration_count",scope.enabled_fixed_configuration_count}};}
  J catalog_status()const{const auto blocked=catalog_.unsupported_graph_action_ids();return{{"action_count",ngpipe::action_count},
    {"stop_action",ngpipe::stop_action},{"registry_fingerprint",ngpipe::registry_fingerprint},{"fingerprint",ngpipe::registry_fingerprint},{"catalog_file_sha256",catalog_.source_sha256()},
    {"native_executable_action_count",catalog_.executable_count()},{"unsupported_graph_action_count",blocked.size()},
    {"complete_native_graph_space",catalog_.complete_native_graph_space()},
    {"experiment_scope",experiment_scope_json()},
    {"execution_mapping_sha256",[&]{const auto value=execution_mappings().dump();return hex(policy::sha256(
      {reinterpret_cast<const std::uint8_t*>(value.data()),value.size()}));}()}};}
  J catalog_detail()const{J unsupported=J::array();for(const auto id:catalog_.unsupported_graph_action_ids()){
    const auto& a=catalog_.at(id);unsupported.push_back({{"action_id",id},{"key",a.key},{"runtime_action",a.runtime_action},
      {"category",ngpipe::category_name(a.category)},{"reason",catalog_.support(id).reason}});}auto out=catalog_status();out["unsupported_graph_actions"]=unsupported;
    out["execution_mappings"]=execution_mappings();return out;}
  J schema()const{return{{"schema","native-pipeline-v1-jsonl"},{"commands",{"schema","catalog","reset","reset_from_library","inspect","step","evaluate_terminal","search","select_library","release_task"}},
    {"roots_encoding","lowercase data_hex"},{"state_handles","immutable and task-scoped"},{"stdout","one JSON response per input line"},
    {"search_modes",{"policy","beam30"}},{"search_defaults",{{"search_mode","policy"},
      {"beam_width",30},{"enable_certified_pruning",true},{"enable_objective_guided_ordering",true}}},
    {"catalog",catalog_status()},{"model_loaded",bool(policy_)},{"solution_library_loaded",bool(library_)},
    {"solution_library_file_sha256",library_?J(library_->file_sha256):J(nullptr)},
    {"solution_library_pinned",library_?J(library_->pinned):J(false)}};}
  std::pair<Entry&,ngpipe::TaskState&> state(const J& row){const auto sh=row.at("state_handle").get<std::string>();
    if(row.contains("task_handle")){const auto th=row.at("task_handle").get<std::string>();auto task=tasks_.find(th);
      if(task==tasks_.end())throw std::invalid_argument("unknown task_handle");auto s=task->second.states.find(sh);
      if(s==task->second.states.end())throw std::invalid_argument("unknown state_handle for task");return{task->second,s->second};}
    for(auto& [_,task]:tasks_){auto s=task.states.find(sh);if(s!=task.states.end())return{task,s->second};}
    throw std::invalid_argument("unknown state_handle");}
  J state_response(const std::string& th,const std::string& sh,Entry& entry,ngpipe::TaskState& state){J out={{"task_handle",th},{"state_handle",sh},{"state",state_json(state)},
    {"max_steps",entry.task->max_steps()},{"candidate_budget",entry.task->candidate_budget()},{"candidate_used",entry.task->candidate_used()}};
    out.update(inspection_json(entry.task->inspect(state)));out["catalog"]=catalog_status();return out;}
  J reset(const J& row){const auto th="t"+std::to_string(next_task_++);Entry entry;
    entry.task=std::make_unique<ngpipe::GraphTask>(catalog_,*dispatcher_,request(row));const auto sh="s"+std::to_string(next_state_++);
    entry.states.emplace(sh,entry.task->initial_state());auto [it,_]=tasks_.emplace(th,std::move(entry));
    return state_response(th,sh,it->second,it->second.states.at(sh));}
  J inspect(const J& row){auto [entry,current]=state(row);const auto th=row.contains("task_handle")?
    row.at("task_handle").get<std::string>():task_handle(entry);const auto sh=row.at("state_handle").get<std::string>();
    return state_response(th,sh,entry,current);}
  J step(const J& row){auto [entry,current]=state(row);const auto next=entry.task->transition(current,row.at("action_id"));
    require(next.has_value(),"action is illegal or STOP (evaluate_terminal closes a recipe)");const auto sh="s"+std::to_string(next_state_++);
    auto [it,_]=entry.states.emplace(sh,*next);const auto th=row.contains("task_handle")?row.at("task_handle").get<std::string>():task_handle(entry);
    return state_response(th,sh,entry,it->second);}
  J evaluate(const J& row){auto [entry,current]=state(row);auto value=entry.task->evaluate_terminal(current);
    J out={{"recipe_action_ids",current.recipe}};out.update(terminal_json(value,row.value("include_frame_hex",false)));return out;}
  J search(const J& row){const auto mode_text=row.value("search_mode",std::string("policy"));
    ngpipe::SearchOptions options;options.mode=mode_text=="policy"?ngpipe::SearchMode::Policy:
      mode_text=="beam30"?ngpipe::SearchMode::Beam30:throw std::invalid_argument("search_mode must be policy or beam30");
    options.beam_width=row.value("beam_width",30U);
    options.enable_certified_pruning=row.value("enable_certified_pruning",true);
    options.enable_objective_guided_ordering=row.value("enable_objective_guided_ordering",true);
    require(options.mode!=ngpipe::SearchMode::Policy||bool(policy_),"policy search requires --model");
    require(options.mode!=ngpipe::SearchMode::Beam30||options.beam_width==30,
      "beam30 search requires beam_width=30");const auto th=row.at("task_handle").get<std::string>();
    auto it=tasks_.find(th);if(it==tasks_.end())throw std::invalid_argument("unknown task_handle");
    using SearchClock=std::chrono::steady_clock;const auto total_begin=SearchClock::now();
    require(it->second.task->candidate_used()==0,"search requires an unused complete-candidate budget");
    auto fresh_request=it->second.task->fresh_request();std::vector<std::vector<std::uint32_t>> starts;
    if(row.value("use_solution_library",false)){require(bool(library_),"solution library not loaded");
      if(fresh_request.require_complete_native_catalog)require(library_->pinned,"formal search requires --solution-library-sha256");
      require(fresh_request.block_index==0&&!fresh_request.continuation_edit_mode,"library search is first-block only");
      require(fresh_request.starting_recipe.empty(),"library search owns the starting recipe");
      const auto dtype=row.value("dtype",fresh_request.roots.front().logical_view);for(const auto& recipe:library_->recipes)if(recipe.dtype==dtype)starts.push_back(recipe.actions);
      require(!starts.empty(),"solution library has no recipe for dtype");}
    const auto initialization_begin=SearchClock::now();
    auto fresh_task=std::make_unique<ngpipe::GraphTask>(catalog_,*dispatcher_,std::move(fresh_request));
    const auto initialization_end=SearchClock::now();
    require(starts.size()<=fresh_task->candidate_budget(),"library candidates exceed complete-candidate budget");
    it->second.task=std::move(fresh_task);ngpipe::PolicySearch runner(policy_.get());
    const auto initialization_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
      initialization_end-initialization_begin).count();
    auto value=runner.run(*it->second.task,*dispatcher_,options,starts,
      total_begin,static_cast<std::uint64_t>(initialization_ns));
    const auto replay_cache=it->second.task->replay_cache_stats();J candidates=J::array();for(const auto& c:value.candidates)
      {J item={{"terminal_status",c.success?"ok":"failed"},{"recipe_action_ids",c.recipe},{"search_action_ids",c.search_actions}};
       if(c.success)item.update({{"J_selection",c.objective},{"compression_ratio",c.compression_ratio},
         {"compression_ns_per_byte",c.compression_ns_per_byte},{"decompression_ns_per_byte",c.decompression_ns_per_byte},
         {"frame_bytes",c.complete_bytes}});else item["error"]=c.failure_error;candidates.push_back(std::move(item));}
    J out={{"search_status",value.success?"ok":"failed"},{"best_search_action_ids",value.best_search_actions},
      {"best_recipe_action_ids",value.best_recipe},{"recipe_action_ids",value.success?value.best_recipe:it->second.task->initial_state().recipe},
      {"evaluations",value.evaluations},{"evaluation_budget",value.evaluation_budget},{"budget_unexplored_actions",value.budget_unexplored_actions},
      {"duplicate_states_skipped",value.duplicate_states_skipped},{"candidates",candidates},
      {"search_mode",ngpipe::search_mode_name(options.mode)},
      {"enable_certified_pruning",options.enable_certified_pruning},
      {"enable_objective_guided_ordering",options.enable_objective_guided_ordering},
      {"certified_pruned_states",value.certified_pruned_states},{"pruning_enabled",options.enable_certified_pruning},
      {"replay_cache",{{"hits",replay_cache.hits},{"misses",replay_cache.misses},
        {"evictions",replay_cache.evictions},{"retained_bytes",replay_cache.retained_bytes},
        {"byte_limit",replay_cache.byte_limit},{"entries",replay_cache.entries}}},
      {"timing",{{"task_initialization_ns",value.timing.task_initialization_ns},
        {"feature_ns",value.timing.feature_ns},{"policy_ns",value.timing.policy_ns},{"action_ordering_ns",value.timing.action_ordering_ns},
        {"transition_ns",value.timing.transition_ns},{"candidate_evaluation_ns",value.timing.candidate_evaluation_ns},
        {"certified_pruning_ns",value.timing.certified_pruning_ns},
        {"total_prediction_ns",value.timing.total_prediction_ns},{"final_encode_ns",value.timing.final_encode_ns},
        {"total_compression_ns",value.timing.total_compression_ns}}}};
    if(!value.success){out["terminal_status"]="failed";out["error"]=value.failure_errors.empty()?
      "all search candidates were inapplicable":"all search candidates failed: "+value.failure_errors.front();
      out["original_numeric_bytes"]=it->second.task->original_numeric_bytes();
      out["max_frame_bytes"]=it->second.task->request().config.graph_limits.max_frame_bytes;return out;}
    auto terminal=terminal_json(value.best_evaluation,row.value("include_frame_hex",false));terminal["frame_bytes"]=value.final_frame.size();
    if(row.value("include_frame_hex",false))terminal["frame_hex"]=hex(value.final_frame);out.update(terminal);return out;}
  J select_library(const J& row){require(bool(library_),"solution library not loaded");auto req=request(row);
    if(req.require_complete_native_catalog)require(library_->pinned,"formal library selection requires --solution-library-sha256");
    require(req.block_index==0&&!req.continuation_edit_mode,"library selection is first-block only");require(req.starting_recipe.empty()&&req.previous_recipe.empty(),"library selection owns the starting recipe");
    ngpipe::GraphTask task(catalog_,*dispatcher_,std::move(req));const auto dtype=row.value("dtype",task.request().roots.front().logical_view);
    const SolutionLibrary::Recipe* best_recipe=nullptr;std::optional<ngpipe::Evaluation> best;std::uint32_t evaluated=0;
    for(const auto& recipe:library_->recipes)if(recipe.dtype==dtype){require(evaluated<task.candidate_budget(),"library candidates exceed complete-candidate budget");
      auto state=task.starting_state(recipe.actions);auto value=task.evaluate_terminal(state);++evaluated;
      if(value.success&&(!best||value.objective<best->objective)){best=std::move(value);best_recipe=&recipe;}}
    require(best.has_value(),"solution library has no recipe for dtype");auto external=best_recipe->actions;external.push_back(ngpipe::stop_action);
    J out={{"selected_recipe_id",best_recipe->id},{"starting_recipe_action_ids",external},{"recipe_action_ids",best_recipe->actions},{"evaluations",evaluated},{"evaluation_budget",task.candidate_budget()},
      {"solution_library_file_sha256",library_->file_sha256},{"solution_library_content_sha256",library_->content_sha256}};
    out.update(terminal_json(*best,row.value("include_frame_hex",false)));return out;}
  J reset_from_library(const J& row){require(bool(library_),"solution library not loaded");auto req=request(row);
    if(req.require_complete_native_catalog)require(library_->pinned,"formal library reset requires --solution-library-sha256");
    require(req.block_index==0&&!req.continuation_edit_mode,"library reset is first-block only");
    require(req.starting_recipe.empty()&&req.previous_recipe.empty(),"library reset owns the starting recipe");
    auto task=std::make_unique<ngpipe::GraphTask>(catalog_,*dispatcher_,std::move(req));
    const auto dtype=row.value("dtype",task->request().roots.front().logical_view);
    std::vector<const SolutionLibrary::Recipe*> candidates;for(const auto& recipe:library_->recipes)if(recipe.dtype==dtype)candidates.push_back(&recipe);
    require(!candidates.empty(),"solution library has no recipe for dtype");
    require(candidates.size()<=task->candidate_budget(),"library candidates exceed complete-candidate budget");
    const SolutionLibrary::Recipe* best_recipe=nullptr;std::optional<ngpipe::TaskState> best_state;
    std::optional<ngpipe::Evaluation> best;J failures=J::array();
    for(const auto* recipe:candidates){const auto before=task->candidate_used();try{auto candidate=task->starting_state(recipe->actions);
        auto value=task->evaluate_terminal(candidate);if(!value.success){failures.push_back({{"recipe_id",recipe->id},
          {"recipe_action_ids",recipe->actions},{"error",value.failure_error}});continue;}
        if(!best||value.objective<best->objective||
          (value.objective==best->objective&&std::tie(recipe->actions,recipe->id)<std::tie(best_recipe->actions,best_recipe->id))){
          best=std::move(value);best_state=std::move(candidate);best_recipe=recipe;}}
      catch(const std::exception& error){if(task->candidate_used()==before)task->consume_failed_candidate();
        failures.push_back({{"recipe_id",recipe->id},{"recipe_action_ids",recipe->actions},{"error",error.what()}});}}
    require(best&&best_state,"all solution-library candidates failed on the current block");
    const auto th="t"+std::to_string(next_task_++),sh="s"+std::to_string(next_state_++);Entry entry;entry.task=std::move(task);
    entry.states.emplace(sh,*best_state);auto [it,_]=tasks_.emplace(th,std::move(entry));auto out=state_response(th,sh,it->second,it->second.states.at(sh));
    auto external=best_recipe->actions;external.push_back(ngpipe::stop_action);out.update({{"selected_recipe_id",best_recipe->id},
      {"starting_recipe_action_ids",external},{"recipe_action_ids",best_recipe->actions},{"library_candidate_count",candidates.size()},
      {"library_failures",failures},{"solution_library_file_sha256",library_->file_sha256},{"solution_library_content_sha256",library_->content_sha256}});
    out["library_selection"]={{"selected_recipe_id",best_recipe->id},{"starting_recipe_action_ids",external},
      {"recipe_action_ids",best_recipe->actions},{"candidate_count",candidates.size()},
      {"evaluations_charged",it->second.task->candidate_used()},{"failed_count",failures.size()},{"failures",failures}};
    out.update(terminal_json(*best,row.value("include_frame_hex",false)));return out;}
  J release(const J& row){const auto th=row.at("task_handle").get<std::string>();const auto count=tasks_.erase(th);
    require(count==1,"unknown task_handle");return{{"released",true},{"task_handle",th}};}
  std::string task_handle(const Entry& target)const{for(const auto& [handle,entry]:tasks_)if(&entry==&target)return handle;
    throw std::logic_error("task entry is not registered");}
  ngpipe::ActionCatalog catalog_;std::unique_ptr<policy::Policy> policy_;std::unique_ptr<graph::StageDispatcher> dispatcher_;
  std::optional<SolutionLibrary> library_;std::map<std::string,Entry> tasks_;std::uint64_t next_task_=1,next_state_=1;
};
}  // namespace

int main(int argc,char** argv){
  try{std::optional<std::string> catalog_path,model_path,library_path,library_sha256,experiment_profile;bool serve=false;for(int i=1;i<argc;++i){const std::string a=argv[i];
    if(a=="--catalog"&&i+1<argc)catalog_path=argv[++i];else if(a=="--model"&&i+1<argc)model_path=argv[++i];
    else if(a=="--experiment-profile"&&i+1<argc)experiment_profile=argv[++i];
    else if(a=="--solution-library"&&i+1<argc)library_path=argv[++i];
    else if(a=="--solution-library-sha256"&&i+1<argc)library_sha256=argv[++i];
    else if(a=="serve")serve=true;else throw std::invalid_argument("usage: native_pipeline_v1_server --catalog FILE [--experiment-profile FILE] [--model FILE] [--solution-library FILE --solution-library-sha256 HEX] serve");}
    require(catalog_path.has_value()&&serve&&(!library_sha256||library_path.has_value()),"usage: native_pipeline_v1_server --catalog FILE [--experiment-profile FILE] [--model FILE] [--solution-library FILE --solution-library-sha256 HEX] serve");
    auto catalog=ngpipe::ActionCatalog::load(*catalog_path);
    if(experiment_profile)catalog.load_experiment_profile(*experiment_profile);
    std::unique_ptr<policy::Policy> loaded;
    if(model_path)loaded=std::make_unique<policy::Policy>(policy::Policy::load(*model_path,
      {ngpipe::registry_fingerprint,policy::feature_schema,ngpipe::action_count,ngpipe::stop_action,18,28}));
    std::optional<SolutionLibrary> library;if(library_path)library=load_library(*library_path,catalog,library_sha256);
    Server server(std::move(catalog),std::move(loaded),graph::make_default_dispatcher(),std::move(library));std::string line;
    while(std::getline(std::cin,line)){if(line.empty())continue;J id=nullptr;try{const auto input=J::parse(line);if(input.contains("request_id"))id=input.at("request_id");
        auto out=server.handle(input);out["ok"]=true;out["request_id"]=id;std::cout<<out.dump()<<'\n'<<std::flush;
      }catch(const std::exception& e){std::cout<<J{{"ok",false},{"request_id",id},{"error",e.what()}}.dump()<<'\n'<<std::flush;}}
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
