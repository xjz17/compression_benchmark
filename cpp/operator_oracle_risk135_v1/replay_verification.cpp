// Supplementary verification only: the production source and models stay frozen.
#define main frozen_production_main
#include "quality_experiment.cpp"
#undef main

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      std::cout << "replay_verification ROOT OUTPUT_DIR\n";
      return 0;
    }
    require(argc == 3, "usage: replay_verification ROOT OUTPUT_DIR");
    const auto root = fs::absolute(argv[1]).lexically_normal();
    const auto output = fs::absolute(argv[2]).lexically_normal();
    require(!fs::exists(output), "refusing to overwrite replay output");
    require(fs::create_directories(output), "cannot create replay output");
    const auto protocol = parse_protocol(root / "PROTOCOL.json");
    const auto prefix_path = root / "features/prefix24/FEATURES.jsonl";
    const auto multi_path = root / "features/multi3/FEATURES.jsonl";
    FeatureSets features;
    features.emplace("prefix24", FeatureSet{
        "prefix24", prefix_feature_names(), prefix_path,
        prefix_path.parent_path() / "FEATURES_RECEIPT.json",
        load_features(prefix_path, prefix_feature_names(), true)});
    features.emplace("multi3", FeatureSet{
        "multi3", multi_feature_names(), multi_path,
        multi_path.parent_path() / "FEATURES_RECEIPT.json",
        load_features(multi_path, multi_feature_names(), true)});
    const auto costs_path = root / "costs/DEVELOPMENT_COSTS.jsonl";
    require(d::file_sha(costs_path) == protocol.development_costs_sha256,
            "development costs differ from frozen protocol");
    std::map<q::Domain, std::vector<std::string>> declared;
    const auto costs = load_costs(costs_path, protocol, declared);
    Json files = Json::array();
    std::size_t policy_count = 0;
    std::size_t row_count = 0;
    for (const auto& task : kTasks) {
      require(fs::create_directory(output / task.id), "task output exists");
      CVModelCache cache;
      for (const auto& policy : policy_grid(task)) {
        std::cout << "[verification] replay " << task.id << " "
                  << policy.id << std::endl;
        const auto cv = evaluate_policy(costs, features, task, protocol,
                                        declared.at(task.domain), policy,
                                        false, cache);
        const auto path = output / task.id / (policy.id + ".jsonl");
        write_new_text(path, jsonl_text(cv.rows));
        files.push_back({{"task_id", task.id}, {"policy_id", policy.id},
                         {"path", fs::relative(path, output).string()},
                         {"sha256", d::file_sha(path)},
                         {"rows", cv.rows.size()}});
        ++policy_count;
        row_count += cv.rows.size();
      }
    }
    require(policy_count == 55 && row_count == 188815,
            "replay scope differs from frozen 19/18/18 grid");
    write_new_json(output / "COMPLETE.json",
                   {{"schema", "operator-oracle-risk135-replay-v1"},
                    {"status", "complete_development_only"},
                    {"policies", policy_count}, {"oof_rows", row_count},
                    {"development_costs_sha256", d::file_sha(costs_path)},
                    {"protocol_sha256", d::file_sha(root / "PROTOCOL.json")},
                    {"confirm_cost_payload_read", false},
                    {"files", std::move(files)}});
    std::cout << "[verification] complete policies=" << policy_count
              << " rows=" << row_count << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "replay_verification failed: " << error.what() << '\n';
    return 1;
  }
}
