#include "lzf_operator_ablation.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using clock_ns = std::chrono::steady_clock;

struct Dataset {
  std::vector<std::vector<std::int64_t>> integers;
  std::vector<std::vector<double>> floats;
  std::vector<std::int64_t> times;
  std::size_t rows = 0;
};

std::vector<std::string> split_csv_row(const std::string& line) {
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

bool parse_int(std::string_view text, std::int64_t& value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_double(std::string_view text, double& value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
         std::isfinite(value);
}

Dataset load_dataset(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dataset: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty dataset");
  const auto headers = split_csv_row(line);
  std::vector<std::vector<std::string>> values(headers.size());
  Dataset dataset;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv_row(line);
    if (fields.size() != headers.size()) throw std::runtime_error("CSV width mismatch");
    ++dataset.rows;
    for (std::size_t i = 0; i < fields.size(); ++i) values[i].push_back(fields[i]);
  }
  for (std::size_t column = 0; column < headers.size(); ++column) {
    std::string name = headers[column];
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::vector<std::int64_t> integers;
    integers.reserve(dataset.rows);
    bool all_integer = true;
    for (const auto& text : values[column]) {
      std::int64_t value = 0;
      if (!parse_int(text, value)) {
        all_integer = false;
        break;
      }
      integers.push_back(value);
    }
    if (name == "timestamp" || name == "time") {
      if (!all_integer) throw std::runtime_error("timestamp is not int64");
      dataset.times = std::move(integers);
    } else if (all_integer) {
      dataset.integers.push_back(std::move(integers));
    } else {
      std::vector<double> floats;
      floats.reserve(dataset.rows);
      for (const auto& text : values[column]) {
        double value = 0.0;
        if (!parse_double(text, value)) throw std::runtime_error("non-numeric column");
        floats.push_back(value);
      }
      dataset.floats.push_back(std::move(floats));
    }
  }
  if (dataset.times.empty()) {
    dataset.times.resize(dataset.rows);
    for (std::size_t i = 0; i < dataset.rows; ++i) dataset.times[i] = static_cast<std::int64_t>(i);
  }
  return dataset;
}

template <typename T>
std::vector<std::uint8_t> as_bytes(const std::vector<T>& values) {
  std::vector<std::uint8_t> output(values.size() * sizeof(T));
  if (!output.empty()) std::memcpy(output.data(), values.data(), output.size());
  return output;
}

std::vector<std::vector<std::uint8_t>> raw_columns(const Dataset& dataset) {
  std::vector<std::vector<std::uint8_t>> columns;
  columns.push_back(as_bytes(dataset.times));
  for (const auto& column : dataset.integers) columns.push_back(as_bytes(column));
  for (const auto& column : dataset.floats) columns.push_back(as_bytes(column));
  return columns;
}

struct Result {
  std::string omitted;
  std::size_t original = 0;
  std::size_t compressed = 0;
  std::int64_t compression_ns = 0;
  std::int64_t decompression_ns = 0;
};

Result run(const Dataset& dataset, std::string omitted) {
  Result result{std::move(omitted)};
  for (const auto& column : raw_columns(dataset)) {
    const auto t0 = clock_ns::now();
    const auto payload = lzf_operator_ablation::encode(column, 1, result.omitted);
    const auto t1 = clock_ns::now();
    const auto restored = lzf_operator_ablation::decode(
        payload, column.size(), 1, result.omitted);
    const auto t2 = clock_ns::now();
    if (restored != column) throw std::runtime_error("LZF lossless check failed");
    result.original += column.size();
    result.compressed += payload.size();
    result.compression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: lzf_source_air_bench DATASET.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<std::string> omissions = {
        "", "lz_match", "match_window_model", "match_token_format"};
    std::vector<Result> results;
    for (const auto& omission : omissions) results.push_back(run(dataset, omission));
    const auto baseline = results.front();
    std::ofstream output(argv[2]);
    output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,"
               "baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,"
               "integer_columns,float_columns,timestamp_columns,columns,rows,configuration,"
               "implementation\n";
    for (const auto& result : results) {
      const double rate = static_cast<double>(result.compressed) / result.original;
      const double baseline_rate = static_cast<double>(baseline.compressed) / baseline.original;
      output << "Air-Quality,LZF," << (result.omitted.empty() ? "none" : result.omitted) << ','
             << result.original << ',' << result.compressed << ',' << std::fixed
             << std::setprecision(9) << rate << ',' << baseline_rate << ','
             << rate - baseline_rate << ',' << result.compression_ns << ','
             << result.decompression_ns << ",1," << dataset.integers.size() << ','
             << dataset.floats.size() << ",1,"
             << dataset.integers.size() + dataset.floats.size() + 1 << ',' << dataset.rows
             << ",lzf_1,lzf_upstream_source_operator_ablation\n";
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
