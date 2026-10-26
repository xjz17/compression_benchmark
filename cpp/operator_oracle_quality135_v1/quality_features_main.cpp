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
               "usage: quality_features COLUMNS OUTPUT_JSONL RECEIPT_JSON");
    const auto columns = fs::absolute(argv[1]).lexically_normal();
    const auto output = fs::absolute(argv[2]).lexically_normal();
    const auto receipt_path = fs::absolute(argv[3]).lexically_normal();
    d::require(fs::is_regular_file(columns),
               "COLUMNS is not a regular file " + columns.string());
    d::require(output != receipt_path,
               "OUTPUT_JSONL and RECEIPT_JSON must differ");
    d::require(!fs::exists(output),
               "refusing to overwrite OUTPUT_JSONL " + output.string());
    d::require(!fs::exists(receipt_path),
               "refusing to overwrite RECEIPT_JSON " + receipt_path.string());

    auto receipt = operator_oracle_quality135_v1::build_features(
        columns, output);
    receipt["driver"] = "quality_features";
    receipt["driver_contract"] =
        "COLUMNS OUTPUT_JSONL RECEIPT_JSON; frozen current135 gates enabled";
    receipt["output_and_receipt_overwrite_allowed"] = false;
    d::write_json(receipt_path, receipt);
    std::cout << Json{{"status", "complete"},
                      {"output_path", output.string()},
                      {"output_sha256", receipt.at("output_sha256")},
                      {"receipt_path", receipt_path.string()},
                      {"column_count", receipt.at("column_count")},
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
