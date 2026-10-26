#include "recommend150.hpp"

#include "../operator_recommend135_v1/common.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>
#include <zlib.h>

namespace fs = std::filesystem;
namespace d = operator_recommend135_v1::detail;
using Json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void append_jsonl(std::ofstream& output, const Json& value) {
  output << value.dump() << '\n';
  require(bool(output), "fixture JSONL write failed");
}

void gzip_copy(const fs::path& source_path, const fs::path& target_path) {
  std::ifstream source(source_path, std::ios::binary);
  require(bool(source), "cannot open gzip fixture source");
  gzFile target = gzopen(target_path.c_str(), "wb");
  require(target != nullptr, "cannot open gzip fixture target");
  std::array<char, 4096> buffer{};
  while (source) {
    source.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = source.gcount();
    if (count > 0)
      require(gzwrite(target, buffer.data(), static_cast<unsigned>(count)) == count,
              "gzip fixture write failed");
  }
  require(source.eof(), "gzip fixture read failed");
  require(gzclose(target) == Z_OK, "gzip fixture close failed");
}

Json feature(const std::string& dataset, const std::string& sample,
             std::size_t index) {
  const bool integer = index == 0;
  const std::set<std::string> integer_null = {
      "sample_decimal_min_p_max", "sample_decimal_min_p_median",
      "sample_decimal_min_p_p90", "sample_decimal_reversible_fraction_max_p0",
      "sample_decimal_reversible_fraction_max_p12",
      "sample_decimal_reversible_fraction_max_p18",
      "sample_decimal_reversible_fraction_max_p3",
      "sample_decimal_reversible_fraction_max_p6",
      "sample_decimal_reversible_fraction_max_p9",
      "sample_decimal_reversible_fraction_p0_to_p18",
      "sample_negative_zero_rate", "sample_nonfinite_rate"};
  Json map = Json::object();
  std::vector<std::string> names(integer_null.begin(), integer_null.end());
  for (std::size_t i = names.size(); i < 43; ++i) {
    char name[8];
    std::snprintf(name, sizeof(name), "f%02zu", i);
    names.emplace_back(name);
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    map[names[i]] = integer && integer_null.contains(names[i])
                        ? Json(nullptr)
                        : Json(static_cast<double>((index + 1) * (i + 1)) /
                               100.0);
  }
  const std::uint64_t valid = 100 + index;
  return {{"schema", "solid-dtype-correct-column-features-v1"},
          {"dataset_id", dataset},
          {"column_id", sample},
          {"column_key", "descriptive-key-that-is-not-" + sample},
          {"column_name", "name_" + sample},
          {"dtype", integer ? "int64" : "float64"},
          {"width", 8},
          {"row_count", valid},
          {"valid_count", valid},
          {"features", std::move(map)}};
}

void write_methods(std::ofstream& output, const std::string& dataset,
                   const std::string& sample, std::size_t index) {
  const std::vector<std::string> methods = {"BOS", "LZ4", "PPMD8", "Zstd"};
  for (std::size_t method = 0; method < methods.size(); ++method) {
    const auto valid = static_cast<std::uint64_t>(100 + index);
    // Make the best method depend on the feature row and objective.
    const auto bytes = static_cast<std::uint64_t>(
        100 + ((index + method * 3) % methods.size()) * 20 + method);
    const auto encode = static_cast<std::uint64_t>(
        1000 + ((index * 2 + method) % methods.size()) * 100 + method);
    const auto decode = static_cast<std::uint64_t>(
        800 + ((index + method * 2) % methods.size()) * 90 + method);
    Json row = {{"schema", "selftest-method-row-v1"},
                  {"dataset_id", dataset},
                  {"stable_id", sample},
                  {"method", methods[method]},
                  {"status", "ok"},
                  {"lossless_verified", true},
                  {"rank_eligible", true},
                  {"canonical_input_bytes", valid * 8},
                  {"source_native_numeric_bytes",
                   index == 0 ? valid : valid * 8},
                  {"valid_values", valid},
                  {"compressed_bytes", bytes},
                  {"compression_time_ns_per_value",
                   static_cast<double>(encode) / valid},
                  {"decompression_time_ns_per_value",
                   static_cast<double>(decode) / valid},
                  {"binary_sha256", std::string(64, char('a' + method))},
                  {"route_id", methods[method] == "BOS" ? "new-bos-selftest"
                                                          : "selftest-route"},
                  {"input_profile", "selftest-input-v1"},
                  {"profile_version", "selftest-profile-v1"},
                  {"timing_contract", "selftest-w1-r3-t1"},
                  {"configuration", "selftest"}};
    if (methods[method] == "BOS")
      row["provenance"] =
          {{"source", "selftest_bos"},
           {"source_file_sha256", std::string(64, 'f')},
           {"input_profile", "selftest-input-v1"}};
    append_jsonl(output, row);
  }
}

}  // namespace

int main() {
  fs::path root = fs::temp_directory_path() /
                  ("bos_recommend150_selftest_" + std::to_string(::getpid()));
  try {
    fs::remove_all(root);
    fs::create_directories(root);
    const auto features_path = root / "FEATURES.jsonl";
    const auto split_path = root / "split.json";
    const auto cohorts_path = root / "cohorts.json";
    const auto candidates_path = root / "candidates.json";
    const auto dev_path = root / "development.jsonl";
    const auto confirm_path = root / "confirm.jsonl";
    const auto external_path = root / "external.jsonl";
    std::ofstream features(features_path);
    std::ofstream development(dev_path);
    std::ofstream confirm(confirm_path);
    std::ofstream external(external_path);
    Json assignments = Json::object();
    Json cohort_assignments = Json::object();
    for (std::size_t i = 0; i < 12; ++i) {
      const auto dataset = "dataset_" + std::to_string(i);
      const auto sample = "column_" + std::to_string(i);
      append_jsonl(features, feature(dataset, sample, i));
      if (i < 10) {
        assignments[dataset] =
            {{"split", "development"},
             {"dev_fold", i % 5},
             {"source_identity_sha256", "group_" + std::to_string(i % 5)}};
        cohort_assignments[dataset] =
            {{"cohort", "development"},
             {"dev_fold", i % 5},
             {"source_identity_sha256", "group_" + std::to_string(i % 5)},
             {"legacy_group", "legacy_" + std::to_string(i % 5)}};
        write_methods(development, dataset, sample, i);
      } else if (i == 10) {
        assignments[dataset] =
            {{"split", "confirm"},
             {"dev_fold", nullptr},
             {"source_identity_sha256", "confirm_group"}};
        cohort_assignments[dataset] =
            {{"cohort", "confirm"},
             {"dev_fold", nullptr},
             {"source_identity_sha256", "confirm_group"},
             {"legacy_group", "legacy_0"}};
        write_methods(confirm, dataset, sample, i);
      } else {
        cohort_assignments[dataset] =
            {{"cohort", "external_final"},
             {"dev_fold", nullptr},
             {"source_identity_sha256", "external_group"},
             {"legacy_group", "external_legacy"}};
        write_methods(external, dataset, sample, i);
      }
    }
    features.close();
    development.close();
    confirm.close();
    external.close();
    const auto external_gz_path = root / "external.jsonl.gz";
    gzip_copy(external_path, external_gz_path);
    d::write_json(split_path,
                  {{"schema", "operator-selection-source-group-split-v1"},
                   {"policy", {{"dev_fold_count", 5}}},
                   {"dataset_assignments", std::move(assignments)}});
    d::write_json(cohorts_path,
                  {{"schema", "bos-recommend150-split-cohorts-v1"},
                   {"status", "frozen_label_free"},
                   {"compression_results_read", false},
                   {"confirm_or_external_used_for_tuning", false},
                   {"dataset_assignments", std::move(cohort_assignments)}});
    Json candidate_contract =
        {{"schema", "bos-recommend150-candidates-v1"},
         {"primary_selectable_ids", {"BOS", "LZ4", "Zstd"}},
         {"restricted_ablation_ids", {"BOS", "LZ4", "Zstd"}},
         {"full_reference_excluded_from_serving", {"PPMD8"}},
         {"bos_replacement",
          {{"candidate_id", "BOS"},
           {"duplicate_candidate_created", false},
           {"float_route", "test-float"},
           {"int64_route", "test-int"}}}};
    candidate_contract["bos_replacement"]["benchmark_measurement_provenance"]
                      ["selftest"] =
        {{"source", "selftest_bos"},
         {"source_result_sha256", std::string(64, 'f')},
         {"input_profile", "selftest-input-v1"},
         {"timing_contract", "selftest-w1-r3-t1"},
         {"allowed_binary_sha256", {std::string(64, 'a')}},
         {"allowed_route_ids", {"new-bos-selftest"}},
         {"allowed_profile_versions", {"selftest-profile-v1"}}};
    d::write_json(candidates_path, candidate_contract);

    bos_recommend150_v1::TuneRequest tune_request;
    tune_request.features_jsonl = features_path;
    tune_request.split_json = split_path;
    tune_request.cohorts_json = cohorts_path;
    tune_request.candidates_json = candidates_path;
    tune_request.development_feed_jsonl = dev_path;
    tune_request.output_directory = root / "models";
    tune_request.expected_samples = 10;
    tune_request.expected_methods = 4;
    const auto tuned = bos_recommend150_v1::tune(tune_request);
    require(tuned.at("status") == "locked", "tune did not lock");
    require(fs::exists(root / "models" / "LOCK.json"), "lock missing");
    const auto index = d::read_json(root / "models" / "MODEL_INDEX.json");
    require(index.at("feature_names").size() == 61 &&
                index.at("feature_transform").at("null_missing_indicators") == 12,
            "frozen null-safe feature transform is not serialized");
    const auto cv = d::read_json(root / "models" / "CV_SUMMARY.json");
    const auto& purge = cv.at("scopes").at("primary84").at("space")
                            .at("purge_audit");
    require(purge.at("final_fit").at("retained_rows") == 8,
            "confirm legacy alias was not purged from final fit");
    require(purge.at("oof_folds").at(1).at("retained_rows") == 6,
            "held-out source/legacy groups were not purged from OOF fit");
    const auto& oof = cv.at("scopes").at("primary84").at("space")
                          .at("selected_oof_evaluation");
    require(oof.at("selector").at("realized_metrics").at("row_count") == 10,
            "strict OOF realized evaluation lacks full development coverage");
    require(oof.at("selector")
                .at("realized_metrics")
                .at("source_native_input_bytes") == 7660.0 &&
                oof.at("selector")
                    .at("realized_metrics")
                    .contains("compressed_to_source_native_reference_ratio"),
            "source-native accounting does not preserve widened input bytes");
    require(oof.at("budgeted_harness_feedback").at("budgets").size() == 4,
            "frozen budgeted harness feedback curve is incomplete");
    const auto& oof_budgets =
        oof.at("budgeted_harness_feedback").at("budgets");
    const std::array<std::size_t, 4> expected_requested = {1, 3, 5, 10};
    const std::array<std::size_t, 4> expected_effective = {1, 3, 3, 3};
    const std::array<std::size_t, 4> expected_probe_executions = {10, 30, 30,
                                                                  30};
    for (std::size_t i = 0; i < expected_requested.size(); ++i) {
      const auto& budget = oof_budgets.at(i);
      require(budget.at("requested_budget") == expected_requested[i] &&
                  budget.at("effective_budget_values") ==
                      Json::array({expected_effective[i]}) &&
                  budget.at("harness_probe_cost")
                          .at("candidate_executions") ==
                      expected_probe_executions[i],
              "development budget prefix or probe execution accounting differs");
    }
    require(oof_budgets.at(0).at("realized_metrics") ==
                oof.at("selector").at("realized_metrics"),
            "budget one must reproduce the top-1 selector exactly");

    bos_recommend150_v1::FinalRequest final_request;
    final_request.features_jsonl = features_path;
    final_request.split_json = split_path;
    final_request.cohorts_json = cohorts_path;
    final_request.candidates_json = candidates_path;
    final_request.model_directory = root / "models";
    final_request.expected_samples = 1;
    final_request.expected_methods = 4;
    final_request.timing_repeats = 5;
    final_request.partition = "confirm";
    final_request.feed_jsonl = confirm_path;
    final_request.output_json = root / "CONFIRM_REPORT.json";
    const auto confirm_result = bos_recommend150_v1::final_evaluate(final_request);
    require(confirm_result.at("status") == "complete", "confirm failed");

    final_request.partition = "external-final";
    final_request.feed_jsonl = external_gz_path;
    final_request.output_json = root / "EXTERNAL_REPORT.json";
    const auto external_result = bos_recommend150_v1::final_evaluate(final_request);
    require(external_result.at("status") == "complete", "external failed");

    bool rejected = false;
    try {
      tune_request.development_feed_jsonl = confirm_path;
      tune_request.output_directory = root / "must_reject";
      tune_request.expected_samples = 1;
      (void)bos_recommend150_v1::tune(tune_request);
    } catch (const std::exception& error) {
      rejected = std::string(error.what()).find("partition rejection") !=
                 std::string::npos;
    }
    require(rejected, "tune did not reject confirm labels");

    const auto bad_identity_path = root / "bad_stable_identity.jsonl";
    {
      std::ifstream source(dev_path);
      std::ofstream target(bad_identity_path);
      std::string line;
      bool mutated = false;
      while (std::getline(source, line)) {
        auto row = Json::parse(line);
        if (!mutated) {
          // Matching descriptive fields and a matching column_id must not be
          // accepted when stable_id names the descriptive alias instead.
          row["stable_id"] = "name_column_0";
          row["column_id"] = "column_0";
          row["column_name"] = "name_column_0";
          mutated = true;
        }
        target << row.dump() << '\n';
      }
      require(mutated && bool(source) == false && source.eof() && bool(target),
              "failed to create stable identity rejection fixture");
    }
    rejected = false;
    try {
      tune_request.development_feed_jsonl = bad_identity_path;
      tune_request.output_directory = root / "must_reject_identity_fallback";
      tune_request.expected_samples = 10;
      (void)bos_recommend150_v1::tune(tune_request);
    } catch (const std::exception& error) {
      rejected = std::string(error.what()).find(
                     "stable_id cannot join clean150 feature column_id") !=
                 std::string::npos;
    }
    require(rejected, "tune accepted a descriptive identity fallback");

    const auto bad_bos_path = root / "bad_bos_provenance.jsonl";
    {
      std::ifstream source(dev_path);
      std::ofstream target(bad_bos_path);
      std::string line;
      bool mutated = false;
      while (std::getline(source, line)) {
        auto row = Json::parse(line);
        if (!mutated && row.value("method", std::string()) == "BOS") {
          row["route_id"] = "deployment-runtime-route-is-not-provenance";
          mutated = true;
        }
        target << row.dump() << '\n';
      }
      require(mutated && bool(source) == false && source.eof() && bool(target),
              "failed to create provenance rejection fixture");
    }
    rejected = false;
    try {
      tune_request.development_feed_jsonl = bad_bos_path;
      tune_request.output_directory = root / "must_reject_bos_provenance";
      tune_request.expected_samples = 10;
      (void)bos_recommend150_v1::tune(tune_request);
    } catch (const std::exception& error) {
      rejected = std::string(error.what()).find(
                     "benchmark measurement provenance") != std::string::npos;
    }
    require(rejected, "tune accepted deployment route as BOS provenance");

    const auto report = d::read_json(root / "CONFIRM_REPORT.json");
    require(report.at("objectives").contains("space") &&
                report.at("objectives").contains("encode") &&
                report.at("objectives").contains("decode"),
            "final report lacks objectives");
    require(report.at("objectives").at("space").contains("fixed_lz4") &&
                report.at("objectives").at("space").contains("fixed_zstd3") &&
                report.at("objectives").at("space").contains(
                    "experiment_scope_full85_algorithm_oracle"),
            "final report lacks baselines/oracle");
    const auto& overhead = report.at("selection_overhead");
    require(overhead.at("raw_column_feature_extraction_included") == false &&
                overhead.at("json_feature_read_or_generation_included") == false &&
                overhead.at("measurements").size() == 6,
            "selection overhead boundary is incomplete");
    for (const auto& measurement : overhead.at("measurements")) {
      const auto& phases = measurement.at("phases");
      require(phases.contains(
                  "prepared_registry_plus_feature_row_validation") &&
                  phases.contains("prepared_model_hot_full_rank") &&
                  phases.contains("validation_registry_rank_end_to_end") &&
                  measurement.at("wall") ==
                      phases.at("prepared_model_hot_full_rank").at("wall") &&
                  measurement.at("thread_cpu") ==
                      phases.at("prepared_model_hot_full_rank").at("thread_cpu"),
              "selection overhead phases or legacy rank-only boundary differ");
    }
    for (const char* objective : {"space", "encode", "decode"}) {
      const auto& block = report.at("objectives").at(objective);
      require(block.at("primary84_selector")
                  .at("budgeted_harness_feedback")
                  .at("budgets")
                  .size() == 4,
              "held-out budgeted harness feedback curve is incomplete");
      const auto& heldout_budgets =
          block.at("primary84_selector")
              .at("budgeted_harness_feedback")
              .at("budgets");
      const std::array<std::size_t, 4> expected_heldout_probes = {1, 3, 3, 3};
      for (std::size_t i = 0; i < expected_requested.size(); ++i) {
        const auto& budget = heldout_budgets.at(i);
        require(budget.at("requested_budget") == expected_requested[i] &&
                    budget.at("effective_budget_values") ==
                        Json::array({expected_effective[i]}) &&
                    budget.at("harness_probe_cost")
                            .at("candidate_executions") ==
                        expected_heldout_probes[i],
                "held-out budget prefix or probe execution accounting differs");
      }
      require(heldout_budgets.at(0).at("realized_metrics") ==
                  block.at("primary84_selector").at("realized_metrics"),
              "held-out budget one must reproduce the top-1 selector exactly");
      for (const char* policy : {"primary84_selector", "restricted3_selector",
                                 "fixed_lz4", "fixed_zstd3", "pool84_oracle",
                                 "restricted3_oracle",
                                 "experiment_scope_full85_algorithm_oracle"}) {
        const auto& realized = block.at(policy).at("realized_metrics");
        require(realized.contains("compression_ratio") &&
                    realized.contains("compression_time_ns_per_value") &&
                    realized.contains("decompression_time_ns_per_value") &&
                    realized.contains("source_native_input_bytes") &&
                    realized.contains(
                        "compressed_to_source_native_reference_ratio") &&
                    realized.contains("selected_candidate_counts"),
                std::string("realized three-metric report missing for ") +
                    objective + "/" + policy);
      }
    }
    fs::remove_all(root);
    std::cout << "bos_recommend150_v1 selftest: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "bos_recommend150_v1 selftest: FAIL: " << error.what()
              << "\nfixture: " << root << '\n';
    return EXIT_FAILURE;
  }
}
