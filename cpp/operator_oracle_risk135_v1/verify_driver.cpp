// Independent regression harness for the frozen risk-selector driver.
// Include the production driver in this translation unit so its anonymous-
// namespace load_task/select path is exercised without changing frozen code.
#define main operator_oracle_risk135_frozen_main
#include "quality_experiment.cpp"
#undef main

#include <chrono>
#include <cstdlib>
#include <sstream>

namespace {

struct TemporaryDirectory {
  fs::path path;

  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path = fs::temp_directory_path() /
           ("operator-oracle-risk135-driver-" + std::to_string(nonce));
    require(fs::create_directory(path), "cannot create driver test directory");
  }

  ~TemporaryDirectory() { std::error_code ignored; fs::remove_all(path, ignored); }
};

struct ScorerCase {
  std::string name;
  double ridge_lambda;
  std::size_t knn_k;
  double ridge_weight;
  WeightMode weight_mode;
};

const std::array<ScorerCase, 4> kScorers{{
    {"ridge_l1", 1.0, 0, 1.0, WeightMode::DatasetEqual},
    {"ridge_l10", 10.0, 0, 1.0, WeightMode::SqrtBytes},
    {"blend_l1_k24", 1.0, 24, 0.5, WeightMode::MixedBytes},
    {"blend_l10_k24", 10.0, 24, 0.5, WeightMode::DatasetEqual},
}};

std::vector<std::string> synthetic_candidates(q::Domain domain) {
  if (domain == q::Domain::Algorithm)
    return {"AlgA", "AlgB", "AlgC", "AlgD", "AlgE"};
  if (domain == q::Domain::OperatorZstd3)
    return {"A001", "A016", "A048", "A089", "A090"};
  return {"algorithm:AlgA", "algorithm:AlgB", "operator_zstd3:A001",
          "operator_zstd3:A048", "operator_zstd3:A090"};
}

std::vector<double> synthetic_features(std::size_t row,
                                       std::size_t dimensions) {
  std::vector<double> result;
  result.reserve(dimensions);
  for (std::size_t feature = 0; feature < dimensions; ++feature) {
    result.push_back(
        static_cast<double>(((row + 5) * (feature + 11)) % 127) /
            static_cast<double>(feature + 3) +
        0.0001 * static_cast<double>((row + 1) * (feature % 7)));
  }
  return result;
}

std::vector<q::TrainingRow> synthetic_rows(
    q::Domain domain, const std::vector<std::string>& candidate_ids,
    std::size_t dimensions, WeightMode weight_mode) {
  constexpr std::size_t row_count = 31;
  std::vector<q::TrainingRow> rows;
  rows.reserve(row_count);
  std::array<std::size_t, 7> dataset_counts{};
  std::array<double, row_count> raw_bytes{};
  double sum_sqrt = 0.0;
  double sum_bytes = 0.0;
  for (std::size_t row = 0; row < row_count; ++row) {
    ++dataset_counts[row % dataset_counts.size()];
    raw_bytes[row] = 128.0 + static_cast<double>((row + 1) * (row + 9) * 17);
    sum_sqrt += std::sqrt(raw_bytes[row]);
    sum_bytes += raw_bytes[row];
  }
  constexpr double dataset_count = 7.0;
  for (std::size_t row = 0; row < row_count; ++row) {
    const std::size_t dataset = row % dataset_counts.size();
    double weight = 1.0 / static_cast<double>(dataset_counts[dataset]);
    if (weight_mode == WeightMode::SqrtBytes)
      weight = dataset_count * std::sqrt(raw_bytes[row]) / sum_sqrt;
    else if (weight_mode == WeightMode::MixedBytes)
      weight = dataset_count *
          (0.5 / (dataset_count * static_cast<double>(dataset_counts[dataset])) +
           0.5 * raw_bytes[row] / sum_bytes);
    std::vector<q::CandidateCosts> candidates;
    for (std::size_t candidate = 0; candidate < candidate_ids.size(); ++candidate) {
      const double structure =
          static_cast<double>(((row + 3) * (candidate + 7)) % 37);
      const double domain_offset = static_cast<double>(domain == q::Domain::Algorithm
          ? 3 : domain == q::Domain::OperatorZstd3 ? 11 : 19);
      const double cost = 80.0 + 9.0 * candidate + structure + domain_offset;
      candidates.push_back({candidate_ids[candidate], true, cost,
                            std::nullopt, std::nullopt, std::nullopt});
    }
    rows.push_back({"sample-" + std::to_string(row),
                    "dataset-" + std::to_string(dataset),
                    "source-" + std::to_string(row % 5), weight,
                    synthetic_features(row, dimensions), std::move(candidates)});
  }
  return rows;
}

Json training_binding(const PolicyConfig& policy,
                      const std::vector<q::TrainingRow>& rows) {
  Json ordered = Json::array();
  for (const auto& row : rows)
    ordered.push_back({{"sample_id", row.sample_id}, {"weight", row.weight}});
  return {{"schema", "operator-oracle-risk135-final-training-binding-v1"},
          {"row_count", rows.size()},
          {"feature_variant", policy.feature_variant},
          {"feature_names_sha256",
           d::sha(Json(policy.model.feature_names).dump())},
          {"weight_mode", weight_mode_name(policy.weight_mode)},
          {"ordered_sample_weight_json_sha256", d::sha(ordered.dump())}};
}

void verify_training_binding(const Json& binding, const PolicyConfig& policy,
                             const q::QualityModel& model,
                             const std::vector<q::TrainingRow>* rows,
                             const Json* serialized_core = nullptr) {
  require(binding.value("schema", std::string()) ==
              "operator-oracle-risk135-final-training-binding-v1",
          "training binding schema mismatch");
  require(binding.at("feature_variant") == policy.feature_variant &&
              binding.at("feature_names_sha256") ==
                  d::sha(Json(policy.model.feature_names).dump()) &&
              binding.at("weight_mode") == weight_mode_name(policy.weight_mode),
          "training binding feature/weight metadata mismatch");
  require(model.feature_count() == policy.model.feature_names.size(),
          "training binding/model feature count mismatch");
  if (rows != nullptr)
    require(binding == training_binding(policy, *rows),
            "synthetic training binding does not bind actual rows/weights");
  if (serialized_core != nullptr) {
    const auto& core = *serialized_core;
    require(core.at("feature_names") == Json(policy.model.feature_names) &&
                core.at("points").is_array() &&
                core.at("training").at("row_count") == core.at("points").size() &&
                binding.at("row_count") == core.at("points").size(),
            "production binding row/features disagree with serialized core");
    Json ordered = Json::array();
    for (const auto& point : core.at("points"))
      ordered.push_back({{"sample_id", text(point, "sample_id", "model point")},
                         {"weight", finite_number(point, "row_weight",
                                                   "model point")}});
    require(binding.at("ordered_sample_weight_json_sha256") ==
                d::sha(ordered.dump()),
            "production binding sample/weight SHA disagrees with core points");
  }
}

void require_exact_ranking(const std::vector<q::RankedCandidate>& expected,
                           const std::vector<q::RankedCandidate>& actual,
                           const std::string& context) {
  require(expected.size() == actual.size(), context + " ranking size mismatch");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    require(expected[i].candidate_id == actual[i].candidate_id &&
                expected[i].predicted_log_cost == actual[i].predicted_log_cost &&
                expected[i].ridge_log_cost == actual[i].ridge_log_cost &&
                expected[i].knn_log_cost == actual[i].knn_log_cost &&
                expected[i].support_rows == actual[i].support_rows &&
                expected[i].neighbor_count == actual[i].neighbor_count,
            context + " full ranking differs at position " + std::to_string(i));
  }
}

Json make_index_task(const fs::path& index_path, const Task& task,
                     const PolicyConfig& policy,
                     const std::vector<q::TrainingRow>& rows,
                     const std::vector<std::string>& candidate_ids,
                     const q::QualityModel& reference_model,
                     std::size_t serial) {
  const Json binding = training_binding(policy, rows);
  const Json pools{{"float64", candidate_ids}};
  const Json envelope{
      {"schema", "operator-oracle-risk135-model-envelope-v1"},
      {"task_id", task.id},
      {"core", reference_model.to_json()},
      {"driver_metadata",
       {{"selected_policy", policy_json(policy)},
        {"final_training_binding", binding},
        {"dtype_candidate_pools", pools},
        {"declared_candidate_ids", candidate_ids}}}};
  const auto filename = "model-" + std::to_string(serial) + ".json";
  const auto model_path = index_path.parent_path() / filename;
  write_new_json(model_path, envelope);
  return {{"task_id", task.id},
          {"domain", q::domain_name(task.domain)},
          {"objective", q::objective_name(task.objective)},
          {"selected_policy", policy_json(policy)},
          {"final_training_binding", binding},
          {"model_path", filename},
          {"model_sha256", d::file_sha(model_path)},
          {"dtype_candidate_pools", pools},
          {"declared_candidate_ids", candidate_ids}};
}

Json synthetic_driver_test() {
  TemporaryDirectory root;
  std::size_t case_count = 0;
  for (const auto& variant : {std::string("prefix24"), std::string("multi3")}) {
    const auto& names = variant == "prefix24" ? prefix_feature_names()
                                                : multi_feature_names();
    for (const auto& scorer : kScorers) {
      const fs::path case_root = root.path /
          (variant + "-" + scorer.name);
      require(fs::create_directory(case_root), "cannot create synthetic case");
      const fs::path index_path = case_root / "MODEL_INDEX.json";
      Json index{{"schema", "operator-oracle-risk135-model-index-v1"},
                 {"status", "synthetic_driver_regression"},
                 {"confirm_used_for_fit_or_selection", false},
                 {"NNCP", "OFF"},
                 {"joint_cross_product_measured", false},
                 {"tasks", Json::array()}};
      struct Reference {
        q::QualityModel model;
        std::vector<q::TrainingRow> rows;
        std::vector<std::string> candidates;
        FeatureRow query;
      };
      std::map<std::string, Reference> references;
      std::size_t serial = 0;
      for (const auto& task : kTasks) {
        PolicyConfig policy;
        policy.id = variant + "__" + scorer.name + "__" +
                    std::string(q::domain_name(task.domain));
        policy.feature_variant = variant;
        policy.weight_mode = scorer.weight_mode;
        policy.frozen_prefix_control = false;
        policy.model.ridge_lambda = scorer.ridge_lambda;
        policy.model.knn_k = scorer.knn_k;
        policy.model.ridge_weight = scorer.ridge_weight;
        policy.model.center_log_cost_per_row = true;
        policy.model.feature_names = names;
        auto candidates = synthetic_candidates(task.domain);
        policy.model.candidate_catalog_sha256 =
            d::sha(Json(candidates).dump());
        auto rows = synthetic_rows(task.domain, candidates, names.size(),
                                   scorer.weight_mode);
        auto model = q::fit(rows, task.domain, task.objective, policy.model);
        FeatureRow query{"query-dataset", "query-column", "float64",
                         synthetic_features(47 + serial, names.size())};
        index["tasks"].push_back(make_index_task(
            index_path, task, policy, rows, candidates, model, serial));
        references.emplace(task.id, Reference{std::move(model), std::move(rows),
                                              std::move(candidates),
                                              std::move(query)});
        ++serial;
      }
      write_new_json(index_path, index);
      const auto parsed_index = d::read_json(index_path);
      require(parsed_index.at("schema") ==
                  "operator-oracle-risk135-model-index-v1" &&
                  parsed_index.at("confirm_used_for_fit_or_selection") == false &&
                  parsed_index.at("NNCP") == "OFF" &&
                  parsed_index.at("joint_cross_product_measured") == false,
              "synthetic model index safety contract mismatch");
      for (const auto& item : parsed_index.at("tasks")) {
        const auto loaded = load_task(index_path, item);
        const auto& reference = references.at(loaded.task.id);
        verify_training_binding(item.at("final_training_binding"), loaded.policy,
                                loaded.model, &reference.rows);
        require(loaded.policy.model.ridge_lambda == scorer.ridge_lambda &&
                    loaded.policy.model.knn_k == scorer.knn_k &&
                    loaded.policy.model.ridge_weight == scorer.ridge_weight &&
                    loaded.policy.model.center_log_cost_per_row,
                "loaded core four-parameter scorer binding mismatch");
        require(loaded.policy.feature_variant == variant &&
                    loaded.policy.weight_mode == scorer.weight_mode &&
                    loaded.model.candidate_ids() == reference.candidates,
                "loaded feature/weight/catalog binding mismatch");
        const std::map<std::string, std::string> no_constants;
        const auto selected = select(loaded.policy, &loaded.model, no_constants,
                                     loaded.pools, reference.query);
        const auto direct = reference.model.rank_with_scoring(
            reference.query.features, reference.candidates,
            scorer.knn_k, scorer.ridge_weight);
        require(selected.candidate_id &&
                    *selected.candidate_id == direct.ranking.front().candidate_id &&
                    selected.predicted_log_cost ==
                        direct.ranking.front().predicted_log_cost,
                "production select differs from unserialized core rank 1");
        require_exact_ranking(direct.ranking, selected.ranking,
                              variant + "/" + scorer.name + "/" +
                                  loaded.task.id);
        ++case_count;
      }
    }
  }
  require(case_count == 24,
          "synthetic driver matrix must cover 2x4x3 cases");
  return {{"status", "pass"},
          {"case_count", case_count},
          {"feature_variants", Json::array({"prefix24", "multi3"})},
          {"feature_dimensions", Json::array({24, 98})},
          {"scorers", Json::array({"ridge_l1", "ridge_l10",
                                    "blend_l1_k24", "blend_l10_k24"})},
          {"weight_modes_covered",
           Json::array({"dataset_equal", "sqrt_bytes", "mixed_bytes"})},
          {"domains", Json::array({"algorithm", "operator_zstd3",
                                    "observed_pipeline"})},
          {"full_rank_ids_and_scores_exact", true},
          {"unserialized_core_compared", true}};
}

struct ProductionTask {
  LoadedTask loaded;
  q::QualityModel independent_core;
};

Json production_verify(const fs::path& root) {
  const auto index_path = root / "models" / "MODEL_INDEX.json";
  const auto prefix_path = root / "features" / "prefix24" / "FEATURES.jsonl";
  const auto multi_path = root / "features" / "multi3" / "FEATURES.jsonl";
  const auto predictions_path = root / "DEVELOPMENT_PREDICTIONS.jsonl";
  const auto lock_path = root / "LOCK.json";
  for (const auto& path : {index_path, prefix_path, multi_path,
                           predictions_path, lock_path})
    require(fs::is_regular_file(path), "missing production input " + path.string());
  const auto lock = d::read_json(lock_path);
  require(lock.value("schema", std::string()) ==
              "operator-oracle-risk135-lock-v1" &&
              lock.value("status", std::string()) == "locked_development_only" &&
              lock.value("confirm_evaluation_performed", true) == false &&
              lock.contains("files") && lock.at("files").is_object(),
          "unsafe/incomplete production lock");
  for (const auto& entry : lock.at("files").items()) {
    const auto path = root / entry.key();
    require(fs::is_regular_file(path) && entry.value().is_string() &&
                d::file_sha(path) == entry.value().get<std::string>(),
            "production LOCK hash mismatch for " + entry.key());
  }
  require(lock.at("files").at("DEVELOPMENT_PREDICTIONS.jsonl") ==
              d::file_sha(predictions_path) &&
              lock.at("files").at("models/MODEL_INDEX.json") ==
                  d::file_sha(index_path),
          "production LOCK omits/mismatches required prediction/model index");
  const auto index = d::read_json(index_path);
  require(index.value("schema", std::string()) ==
              "operator-oracle-risk135-model-index-v1" &&
              index.value("confirm_used_for_fit_or_selection", true) == false &&
              index.value("NNCP", std::string()) == "OFF" &&
              index.value("joint_cross_product_measured", true) == false,
          "unsafe production model index");
  std::map<std::string, ProductionTask> tasks;
  for (const auto& item : index.at("tasks")) {
    auto loaded = load_task(index_path, item);
    const auto model_path =
        (index_path.parent_path() / text(item, "model_path", "index task"))
            .lexically_normal();
    const auto envelope = d::read_json(model_path);
    auto independent = q::QualityModel::from_json(envelope.at("core"));
    verify_training_binding(item.at("final_training_binding"), loaded.policy,
                            loaded.model, nullptr, &envelope.at("core"));
    require(tasks.emplace(loaded.task.id,
                          ProductionTask{std::move(loaded),
                                         std::move(independent)}).second,
            "duplicate production task");
  }
  require(tasks.size() == kTasks.size(),
          "production index must contain exactly three tasks");

  FeatureSets features;
  features.emplace("prefix24", FeatureSet{
      "prefix24", prefix_feature_names(), prefix_path,
      prefix_path.parent_path() / "FEATURES_RECEIPT.json",
      load_features(prefix_path, prefix_feature_names(), true)});
  features.emplace("multi3", FeatureSet{
      "multi3", multi_feature_names(), multi_path,
      multi_path.parent_path() / "FEATURES_RECEIPT.json",
      load_features(multi_path, multi_feature_names(), true)});
  require(features.at("prefix24").rows.size() == kExpectedDevelopmentRows &&
              features.at("multi3").rows.size() == kExpectedDevelopmentRows,
          "production verification requires frozen development features");
  for (const auto& [identity, prefix] : features.at("prefix24").rows) {
    const auto multi = features.at("multi3").rows.find(identity);
    require(multi != features.at("multi3").rows.end() &&
                multi->second.dataset_id == prefix.dataset_id &&
                multi->second.sample_id == prefix.sample_id &&
                multi->second.dtype == prefix.dtype,
            "production feature variants have different identities/dtypes");
  }
  require(index.at("inputs").at("features").at("prefix24").at("sha256") ==
              d::file_sha(prefix_path) &&
              index.at("inputs").at("features").at("multi3").at("sha256") ==
                  d::file_sha(multi_path),
          "production feature SHA differs from locked index");

  const auto receipt_path = fs::path(predictions_path.string() + ".receipt.json");
  const auto receipt = d::read_json(receipt_path);
  require(receipt.value("schema", std::string()) ==
              "operator-oracle-risk135-prediction-receipt-v1" &&
              receipt.value("status", std::string()) == "complete" &&
              receipt.at("output").at("sha256") == d::file_sha(predictions_path) &&
              receipt.at("output").at("row_count") ==
                  kExpectedDevelopmentRows * kTasks.size() &&
              receipt.value("query_costs_read", true) == false,
          "production prediction receipt mismatch");
  require(fs::path(text(receipt.at("output"), "path", "prediction receipt"))
                  .lexically_normal() == fs::absolute(predictions_path).lexically_normal(),
          "prediction receipt output path mismatch");
  require(receipt.at("inputs").at("model_index").at("sha256") ==
              d::file_sha(index_path) &&
              receipt.at("inputs").at("features").at("prefix24").at("sha256") ==
                  d::file_sha(prefix_path) &&
              receipt.at("inputs").at("features").at("multi3").at("sha256") ==
                  d::file_sha(multi_path) &&
              receipt.at("task_count") == kTasks.size() &&
              receipt.at("feature_row_count") == kExpectedDevelopmentRows,
          "prediction receipt does not bind all model/feature inputs");
  const auto& receipt_timing = receipt.at("timing");
  require(receipt_timing.at("clock") ==
              "CLOCK_THREAD_CPUTIME_ID_and_steady_clock" &&
              receipt_timing.at("per_query_selection_includes_warmup") == false &&
              receipt_timing.at("per_query_selection_repeats") == 5 &&
              receipt_timing.at("per_query_selection_warm_repeats") == 1 &&
              receipt_timing.at("codec_runtime_included") == false,
          "prediction receipt timing contract mismatch");
  for (const auto* key : {
           "model_load_parse_prepare_cpu_ns", "model_load_parse_prepare_wall_ns",
           "features_load_parse_validate_cpu_ns", "features_load_parse_validate_wall_ns",
           "prediction_build_cpu_ns", "prediction_build_wall_ns",
           "output_write_cpu_ns", "output_write_wall_ns",
           "total_through_output_write_cpu_ns",
           "total_through_output_write_wall_ns"})
    (void)unsigned_integer(receipt_timing, key, "prediction receipt timing");
  std::set<std::tuple<std::string, std::string, std::string>> seen;
  std::size_t rows = 0;
  std::uint64_t measured_selection_cpu_ns = 0;
  for (const auto& prediction : read_jsonl(predictions_path)) {
    require(prediction.value("schema", std::string()) ==
                "operator-oracle-risk135-prediction-v1",
            "production prediction row schema mismatch");
    const auto dataset = text(prediction, "dataset_id", "prediction");
    const auto sample = text(prediction, "sample_id", "prediction");
    const auto task_id = text(prediction, "task_id", "prediction");
    require(prediction.at("column_id") == sample,
            "prediction sample/column identity mismatch");
    const auto key_value = std::make_tuple(dataset, sample, task_id);
    require(seen.insert(key_value).second, "duplicate production prediction");
    auto& task = tasks.at(task_id);
    const auto& feature =
        features.at(task.loaded.policy.feature_variant).rows.at(key(dataset, sample));
    const auto& pool = task.loaded.pools.at(feature.dtype);
    const std::map<std::string, std::string> no_constants;
    const auto selected = select(task.loaded.policy, &task.loaded.model,
                                 no_constants, task.loaded.pools, feature);
    const auto direct = task.independent_core.rank(feature.features, pool);
    require_exact_ranking(direct.ranking, selected.ranking,
                          "production " + task_id + "/" + dataset + "/" + sample);
    require(prediction.at("selected_policy") == policy_json(task.loaded.policy) &&
                prediction.at("candidate_pool") == Json(pool) &&
                prediction.at("selected_candidate_id") ==
                    selected.ranking.front().candidate_id &&
                prediction.at("dtype") == feature.dtype &&
                prediction.at("domain") == q::domain_name(task.loaded.task.domain) &&
                prediction.at("objective") ==
                    q::objective_name(task.loaded.task.objective),
            "production prediction policy/pool/rank1 mismatch");
    require(prediction.at("query_costs_read") == false &&
                prediction.at("query_feature_variant") ==
                    task.loaded.policy.feature_variant &&
                prediction.at("query_features_sha256") ==
                    d::file_sha(task.loaded.policy.feature_variant == "prefix24"
                                    ? prefix_path : multi_path) &&
                prediction.at("model_index_sha256") == d::file_sha(index_path) &&
                prediction.at("selection_cpu_repeats") == 5 &&
                prediction.at("selection_warm_repeats") == 1 &&
                prediction.at("timing_excludes_model_setup") == true &&
                prediction.at("timing_excludes_feature_extraction") == true,
            "production prediction provenance/timing contract mismatch");
    const auto& samples = prediction.at("selection_cpu_ns_samples");
    require(samples.is_array() && samples.size() == 5,
            "prediction timing sample shape mismatch");
    std::array<std::uint64_t, 5> observed_times{};
    for (std::size_t i = 0; i < observed_times.size(); ++i) {
      require(samples.at(i).is_number_unsigned(),
              "prediction timing sample must be unsigned integer");
      observed_times[i] = samples.at(i).get<std::uint64_t>();
    }
    auto sorted_times = observed_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    require(prediction.at("selection_cpu_ns_median") == sorted_times[2],
            "prediction timing median is not the middle of five samples");
    measured_selection_cpu_ns += sorted_times[2];
    const auto& ids = prediction.at("ranking_candidate_ids");
    const auto& scores = prediction.at("ranking_predicted_log_costs");
    require(ids.is_array() && scores.is_array() &&
                ids.size() == direct.ranking.size() &&
                scores.size() == direct.ranking.size(),
            "production prediction full-rank shape mismatch");
    for (std::size_t i = 0; i < direct.ranking.size(); ++i) {
      require(ids.at(i) == direct.ranking[i].candidate_id &&
                  scores.at(i).is_number() &&
                  scores.at(i).get<double>() == direct.ranking[i].predicted_log_cost,
              "production prediction serialized full rank differs at position " +
                  std::to_string(i));
    }
    ++rows;
  }
  require(rows == kExpectedDevelopmentRows * kTasks.size(),
          "production predictions must contain exactly 10,299 development rows");
  for (const auto& [identity, unused] : features.at("prefix24").rows) {
    (void)unused;
    const auto newline = identity.find('\n');
    const auto dataset = identity.substr(0, newline);
    const auto sample = identity.substr(newline + 1);
    for (const auto& task : kTasks)
      require(seen.contains({dataset, sample, task.id}),
              "production prediction coverage is incomplete");
  }
  Json model_hashes = Json::object();
  for (const auto& item : index.at("tasks")) {
    const auto relative = text(item, "model_path", "index task");
    const auto path = (index_path.parent_path() / relative).lexically_normal();
    model_hashes[relative] = d::file_sha(path);
  }
  return {{"status", "pass"},
          {"development_prediction_rows", rows},
          {"task_count", tasks.size()},
          {"full_rank_ids_and_scores_exact", true},
          {"production_load_task_and_select_exercised", true},
          {"independent_core_rank_exercised", true},
          {"prediction_timing_contract_verified", true},
          {"sum_per_query_selection_cpu_ns_medians", measured_selection_cpu_ns},
          {"input_sha256",
           {{"LOCK.json", d::file_sha(lock_path)},
            {"models/MODEL_INDEX.json", d::file_sha(index_path)},
            {"features/prefix24/FEATURES.jsonl", d::file_sha(prefix_path)},
            {"features/prefix24/FEATURES_RECEIPT.json",
             d::file_sha(prefix_path.parent_path() / "FEATURES_RECEIPT.json")},
            {"features/multi3/FEATURES.jsonl", d::file_sha(multi_path)},
            {"features/multi3/FEATURES_RECEIPT.json",
             d::file_sha(multi_path.parent_path() / "FEATURES_RECEIPT.json")},
            {"DEVELOPMENT_PREDICTIONS.jsonl", d::file_sha(predictions_path)},
            {"DEVELOPMENT_PREDICTIONS.jsonl.receipt.json",
             d::file_sha(receipt_path)}}},
          {"model_envelope_sha256", std::move(model_hashes)},
          {"confirm_costs_accessed", false}};
}

void driver_usage() {
  std::cerr << "usage: verify_driver ROOT OUTPUT_JSON\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3) {
      const fs::path root = fs::absolute(argv[1]).lexically_normal();
      const fs::path output = fs::absolute(argv[2]).lexically_normal();
      require(fs::is_directory(root), "ROOT is not a directory");
      require(!fs::exists(output), "refusing to overwrite " + output.string());
      require(fs::is_directory(output.parent_path()),
              "OUTPUT_JSON parent directory does not exist");
      const auto wall_start = d::wall_ns();
      const auto cpu_start = d::cpu_ns();
      auto synthetic = synthetic_driver_test();
      const auto synthetic_cpu_ns = d::cpu_ns() - cpu_start;
      const auto synthetic_wall_ns = d::wall_ns() - wall_start;
      const auto production_wall_start = d::wall_ns();
      const auto production_cpu_start = d::cpu_ns();
      auto production = production_verify(root);
      const auto production_cpu_ns = d::cpu_ns() - production_cpu_start;
      const auto production_wall_ns = d::wall_ns() - production_wall_start;
      const Json result{
          {"schema", "operator-oracle-risk135-driver-verification-v1"},
          {"status", "pass"},
          {"scope", "development_only_no_confirm_cost_access"},
          {"root", root.string()},
          {"synthetic", std::move(synthetic)},
          {"production", std::move(production)},
          {"verification_timing",
           {{"clock", "CLOCK_THREAD_CPUTIME_ID_and_steady_clock"},
            {"synthetic_cpu_ns", synthetic_cpu_ns},
            {"synthetic_wall_ns", synthetic_wall_ns},
            {"production_cpu_ns", production_cpu_ns},
            {"production_wall_ns", production_wall_ns},
            {"total_cpu_ns", d::cpu_ns() - cpu_start},
            {"total_wall_ns", d::wall_ns() - wall_start}}},
          {"confirm_costs_accessed", false},
          {"NNCP", "OFF"}};
      write_new_json(output, result);
      std::cout << "verify_driver: PASS output=" << output << '\n';
      return 0;
    }
    driver_usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "verify_driver: FAIL: " << error.what() << '\n';
    return 1;
  }
}
