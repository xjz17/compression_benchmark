#include "composite.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace native_composite_candidate_v1;

namespace {
Bytes read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(input), {}};
}
void write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!output) throw std::runtime_error("cannot write " + path.string());
}
std::uint64_t natural(const char* value) {
  std::size_t used = 0;
  const auto result = std::stoull(value, &used);
  if (!value[0] || value[used]) throw std::invalid_argument("invalid integer");
  return result;
}
double median(std::vector<std::uint64_t> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2) return static_cast<double>(values[middle]);
  return (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}
void json_samples(const char* key, const std::vector<std::uint64_t>& values) {
  std::cout << ",\"" << key << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}
Weights parse_weights(char** argv, int first) {
  return {std::stod(argv[first]), std::stod(argv[first + 1]),
          std::stod(argv[first + 2])};
}
std::unique_ptr<Selector> selector(const std::string& name) {
  if (name == "train-linear-v1")
    return std::make_unique<ObjectiveLinearSelector>(
        train_only_objective_linear_v1());
  constexpr std::string_view data_prefix = "data-aware-v1@";
  if (name.starts_with(data_prefix))
    return std::make_unique<DataAwareLinearSelector>(
        DataAwareLinearSelector::load(
            name.substr(data_prefix.size()),
            "8298b92bb6a014e4e80cc88120e3236d0375f53804d57ad9d4d672238e340fb7"));
  return std::make_unique<FixedSelector>(backend_from_label(name));
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2)
    throw std::invalid_argument(
        "expected inventory, features, select, encode, decode, or measure");
  const std::string command = argv[1];
  if (command == "inventory") {
    std::cout << "registry_version\t" << registry_version << '\n'
              << "registry_sha256\t" << registry_sha256() << '\n'
              << "baseline_profile\t" << native_baseline86_v1::profile_version
              << '\n';
    for (const auto& item : inventory())
      std::cout << static_cast<unsigned>(item.backend) << '\t' << item.label << '\t'
                << (item.linked ? "linked" : "unlinked") << '\t'
                << item.implementation << '\t' << item.configuration << '\t'
                << item.source << '\t' << item.thread_contract << '\n';
    return 0;
  }
  if (command == "features") {
    if (argc != 6)
      throw std::invalid_argument("features DTYPE ROWS VALUES MISSING");
    const Block block{argv[2], natural(argv[3]), read(argv[4]), read(argv[5])};
    const auto values = task_features(block);
    std::cout << "{\"schema\":\"" << feature_schema << "\",\"features\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << std::setprecision(17) << values[index];
    }
    std::cout << "]}\n";
    return 0;
  }
  if (command == "select") {
    if (argc != 10)
      throw std::invalid_argument(
          "select MODEL W_R W_E W_D DTYPE ROWS VALUES MISSING");
    auto selected = selector(argv[2]);
    const auto* data =
        dynamic_cast<const DataAwareLinearSelector*>(selected.get());
    if (!data)
      throw std::invalid_argument(
          "select diagnostics require a data-aware-v1 model");
    const Block block{argv[6], natural(argv[7]), read(argv[8]), read(argv[9])};
    const auto weights = parse_weights(argv, 3);
    const auto features = task_features(block);
    const auto predictions = data->predict(features, weights);
    const auto chosen = data->select(features, weights);
    std::cout << std::setprecision(17)
              << "{\"feature_schema\":\"" << feature_schema
              << "\",\"selector_identity\":\"" << data->identity()
              << "\",\"backend\":\"" << candidate(chosen).label
              << "\",\"features\":[";
    for (std::size_t index = 0; index < features.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << features[index];
    }
    std::cout << "],\"predictions\":[";
    for (std::size_t index = 0; index < predictions.size(); ++index) {
      if (index) std::cout << ',';
      const auto& item = predictions[index];
      std::cout << "{\"backend\":\"" << candidate(item.backend).label
                << "\",\"costs\":[" << item.costs[0] << ',' << item.costs[1]
                << ',' << item.costs[2] << "],\"weighted_score\":"
                << item.weighted_score << '}';
    }
    std::cout << "]}\n";
    return 0;
  }
  if (command == "encode") {
    if (argc != 11)
      throw std::invalid_argument(
          "encode BACKEND W_R W_E W_D DTYPE ROWS VALUES MISSING FRAME");
    auto selected = selector(argv[2]);
    const Block block{argv[6], natural(argv[7]), read(argv[8]), read(argv[9])};
    write(argv[10], encode(*selected, parse_weights(argv, 3), block));
    return 0;
  }
  if (command == "decode") {
    if (argc != 5)
      throw std::invalid_argument("decode FRAME VALUES MISSING");
    const auto restored = decode(read(argv[2]));
    write(argv[3], restored.block.values);
    write(argv[4], restored.block.missing);
    std::cout << candidate(restored.backend).label << '\t'
              << restored.block.dtype << '\t' << restored.block.row_count << '\t'
              << restored.original_numeric_bytes << '\t'
              << restored.selector_identity << '\n';
    return 0;
  }
  if (command == "decode-direct") {
    if (argc != 5)
      throw std::invalid_argument("decode-direct FRAME VALUES MISSING");
    const auto restored = native_baseline86_v1::decode(read(argv[2]));
    write(argv[3], restored.values);
    write(argv[4], restored.missing);
    std::cout << restored.dtype << '\t' << restored.row_count << '\t'
              << restored.values.size() << '\n';
    return 0;
  }
  if (command == "measure-direct") {
    if (argc != 10)
      throw std::invalid_argument(
          "measure-direct BACKEND DTYPE ROWS VALUES MISSING WARMUPS REPEATS FRAME");
    const auto backend = backend_from_label(argv[2]);
    const Block block{argv[3], natural(argv[4]), read(argv[5]), read(argv[6])};
    auto result = measure_direct_backend(backend, block, natural(argv[7]),
                                         natural(argv[8]));
    write(argv[9], result.frame);
    const auto raw = result.original_numeric_bytes;
    std::cout << std::setprecision(17)
              << "{\"backend\":\"" << candidate(result.backend).label
              << "\",\"B_raw\":" << raw
              << ",\"B_compressed\":" << result.frame.size();
    json_samples("compression_ns_samples", result.compression_ns);
    json_samples("decompression_ns_samples", result.decompression_ns);
    if (raw == 0) {
      std::cout << ",\"status\":\"N/A\"";
    } else {
      std::cout << ",\"status\":\"ok\",\"compression_ratio\":"
                << static_cast<double>(result.frame.size()) / raw
                << ",\"compression_ns_per_byte\":"
                << median(result.compression_ns) / raw
                << ",\"decompression_ns_per_byte\":"
                << median(result.decompression_ns) / raw;
    }
    std::cout << ",\"selector_identity\":\"" << result.selector_identity
              << "\",\"frame_mode\":\"NGF1\",\"registry_sha256\":\""
              << registry_sha256() << "\"}\n";
    return 0;
  }
  if (command == "measure") {
    if (argc != 13)
      throw std::invalid_argument(
          "measure BACKEND W_R W_E W_D DTYPE ROWS VALUES MISSING WARMUPS REPEATS FRAME");
    auto selected = selector(argv[2]);
    const Block block{argv[6], natural(argv[7]), read(argv[8]), read(argv[9])};
    auto result = measure(*selected, parse_weights(argv, 3), block,
                          natural(argv[10]), natural(argv[11]));
    write(argv[12], result.frame);
    const auto raw = result.original_numeric_bytes;
    std::cout << std::setprecision(17);
    std::cout << "{\"backend\":\"" << candidate(result.backend).label
              << "\",\"B_raw\":" << raw
              << ",\"B_compressed\":" << result.frame.size();
    json_samples("compression_ns_samples", result.compression_ns);
    json_samples("decompression_ns_samples", result.decompression_ns);
    if (raw == 0) {
      std::cout << ",\"status\":\"N/A\"";
    } else {
      std::cout << ",\"status\":\"ok\",\"compression_ratio\":"
                << static_cast<double>(result.frame.size()) / raw
                << ",\"compression_ns_per_byte\":"
                << median(result.compression_ns) / raw
                << ",\"decompression_ns_per_byte\":"
                << median(result.decompression_ns) / raw;
    }
    std::cout << ",\"selector_identity\":\"" << result.selector_identity
              << "\",\"frame_mode\":\"NGF1\",\"registry_sha256\":\""
              << registry_sha256()
              << "\"}\n";
    return 0;
  }
  throw std::invalid_argument("unknown command");
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
