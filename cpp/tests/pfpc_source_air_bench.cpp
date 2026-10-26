#include "pfpc.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;
using Clock = std::chrono::high_resolution_clock;

struct Dataset {
  std::vector<Bytes> columns;
  std::size_t integer_columns = 0;
  std::size_t float_columns = 0;
  std::size_t timestamp_columns = 0;
  std::size_t rows = 0;
};

struct Result {
  std::string omitted_operator;
  std::size_t original_bytes = 0;
  std::size_t compressed_bytes = 0;
  long long compression_ns = 0;
  long long decompression_ns = 0;
  bool lossless = true;
};

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (c == ',' && !quoted) {
      fields.push_back(std::move(field));
      field.clear();
    } else if (c != '\r') {
      field.push_back(c);
    }
  }
  fields.push_back(std::move(field));
  return fields;
}

bool parse_i64(std::string_view text, std::int64_t& value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_f64(std::string_view text, double& value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

template <typename T>
Bytes as_bytes(const std::vector<T>& values) {
  Bytes bytes(values.size() * sizeof(T));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

Dataset load_dataset(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dataset: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty dataset");
  const auto headers = split_csv(line);
  std::vector<std::vector<std::string>> values(headers.size());
  Dataset dataset;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    if (fields.size() != headers.size()) throw std::runtime_error("CSV width mismatch");
    ++dataset.rows;
    for (std::size_t i = 0; i < fields.size(); ++i) values[i].push_back(fields[i]);
  }
  for (std::size_t column = 0; column < headers.size(); ++column) {
    std::string name = headers[column];
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    std::vector<std::int64_t> integers;
    integers.reserve(dataset.rows);
    bool all_integer = true;
    for (const auto& text : values[column]) {
      std::int64_t value = 0;
      if (!parse_i64(text, value)) {
        all_integer = false;
        break;
      }
      integers.push_back(value);
    }
    if (all_integer) {
      dataset.columns.push_back(as_bytes(integers));
      if (name == "timestamp" || name == "time") ++dataset.timestamp_columns;
      else ++dataset.integer_columns;
      continue;
    }
    std::vector<double> doubles;
    doubles.reserve(dataset.rows);
    for (const auto& text : values[column]) {
      double value = 0.0;
      if (!parse_f64(text, value)) throw std::runtime_error("non-numeric column: " + name);
      doubles.push_back(value);
    }
    dataset.columns.push_back(as_bytes(doubles));
    ++dataset.float_columns;
  }
  return dataset;
}

Result run(const Dataset& dataset, const std::string& omitted_operator) {
  Result result;
  result.omitted_operator = omitted_operator;
  for (const auto& column : dataset.columns) {
    Bytes encoded(column.size() * 2 + 1024);
    const auto t0 = Clock::now();
    std::size_t compressed = 0;
    if (omitted_operator.empty()) {
      compressed = pfpc_compress_memory(column.data(), column.size(), encoded.data(),
                                        encoded.size(), 16, 1, 1024);
    } else {
      const auto mode = omitted_operator == "xor"
          ? PFPC_ABLATE_XOR
          : PFPC_ABLATE_TRAILING_ZERO_PACK;
      compressed = pfpc_compress_memory_ablation(
          column.data(), column.size(), encoded.data(), encoded.size(), 16, mode);
    }
    const auto t1 = Clock::now();
    if (compressed == 0) throw std::runtime_error("pFPC compression failed");
    encoded.resize(compressed);
    Bytes decoded(column.size());
    const std::size_t consumed = omitted_operator.empty()
        ? pfpc_decompress_memory(encoded.data(), encoded.size(), decoded.data(), decoded.size())
        : pfpc_decompress_memory_ablation(encoded.data(), encoded.size(), decoded.data(), decoded.size());
    const auto t2 = Clock::now();
    result.original_bytes += column.size();
    result.compressed_bytes += encoded.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && consumed == encoded.size() && decoded == column;
  }
  return result;
}

void write_results(const std::string& path, const Dataset& dataset,
                   const std::vector<Result>& results) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write output");
  output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
  const double baseline = static_cast<double>(results.front().compressed_bytes) /
      results.front().original_bytes;
  output << std::setprecision(9);
  for (const auto& result : results) {
    const double rate = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    output << "Air-Quality,pfpc,"
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << baseline << ',' << (rate - baseline) << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows
           << ",pred16_t1,pfpc_upstream_source_operator_ablation\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: pfpc_source_air_bench DATASET.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<Result> results = {
        run(dataset, ""), run(dataset, "xor"), run(dataset, "trailing_zero_pack")};
    for (const auto& result : results) {
      if (!result.lossless) throw std::runtime_error("pFPC lossless verification failed");
    }
    write_results(argv[2], dataset, results);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
