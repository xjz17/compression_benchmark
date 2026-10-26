#include "quality_data.hpp"

#include "../operator_recommend135_v1/common.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {
namespace fs = std::filesystem;
namespace d = operator_recommend135_v1::detail;
using Json = nlohmann::json;
}

int main(int argc, char** argv) {
  try {
    d::require(argc == 4,
               "usage: quality_features COLUMNS INPUT_METADATA OUTPUT_DIR");
    const auto columns = fs::absolute(argv[1]).lexically_normal();
    const auto input_metadata = fs::absolute(argv[2]).lexically_normal();
    const auto output_directory = fs::absolute(argv[3]).lexically_normal();
    d::require(fs::is_regular_file(columns),
               "COLUMNS is not a regular file " + columns.string());
    d::require(fs::is_regular_file(input_metadata),
               "INPUT_METADATA is not a regular file " +
                   input_metadata.string());

    auto receipt =
        operator_oracle_quality135_v1::build_development_feature_variants(
            columns, input_metadata, output_directory);
    receipt["driver"] = "quality_features";
    receipt["driver_contract"] =
        "COLUMNS INPUT_METADATA OUTPUT_DIR; development-only frozen current135 gates enabled; emits prefix24 and multi3 subdirectories";
    receipt["output_artifact_overwrite_allowed"] = false;
    std::cout << Json{{"status", "complete"},
                      {"output_directory", output_directory.string()},
                      {"development_column_count",
                       receipt.at("development_column_count")},
                      {"confirm_column_count_processed", 0},
                      {"feature_variants",
                       Json::array({"prefix24", "multi3"})},
                      {"oracle_labels_read", false}}
                         .dump()
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << Json{{"status", "failed"}, {"error", error.what()}}.dump()
              << '\n';
    return 1;
  }
}
