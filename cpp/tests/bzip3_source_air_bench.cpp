// Source-level Air-Quality validation for Bzip3 BWT and arithmetic-coding
// ablations. Every stream is decoded by the paired upstream decoder path.

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

extern "C" {
#include "bwt/bzip3/include/libbz3.h"
}

namespace {

using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;
constexpr std::int32_t kBlockSize = 1 << 20;

struct Dataset {
  std::vector<Bytes> columns;
  std::size_t rows = 0;
  std::size_t integer_columns = 0;
  std::size_t float_columns = 0;
  std::size_t timestamp_columns = 0;
};

struct Result {
  std::string omitted_operator;
  std::uint64_t original_bytes = 0;
  std::uint64_t compressed_bytes = 0;
  std::uint64_t compression_ns = 0;
  std::uint64_t decompression_ns = 0;
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

Bytes encode_column(const Bytes& input, int mode) {
  bz3_state* state = bz3_new(kBlockSize);
  if (state == nullptr) throw std::runtime_error("Bzip3 state allocation failed");
  bz3_set_ablation_mode(state, mode);
  Bytes output(bz3_bound(input.size()) + 64);
  std::copy(input.begin(), input.end(), output.begin());
  const std::int32_t size = bz3_encode_block(
      state, output.data(), static_cast<std::int32_t>(input.size()));
  if (size < 0) {
    const std::string message = bz3_strerror(state);
    bz3_free(state);
    throw std::runtime_error("Bzip3 encode failed: " + message);
  }
  bz3_free(state);
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes decode_column(const Bytes& input, std::size_t expected_size, int mode) {
  bz3_state* state = bz3_new(kBlockSize);
  if (state == nullptr) throw std::runtime_error("Bzip3 decode state allocation failed");
  bz3_set_ablation_mode(state, mode);
  Bytes output(std::max<std::size_t>(bz3_bound(expected_size) + 64, input.size()));
  std::copy(input.begin(), input.end(), output.begin());
  const std::int32_t size = bz3_decode_block(
      state, output.data(), output.size(), static_cast<std::int32_t>(input.size()),
      static_cast<std::int32_t>(expected_size));
  if (size != static_cast<std::int32_t>(expected_size)) {
    const std::string message = bz3_strerror(state);
    bz3_free(state);
    throw std::runtime_error("Bzip3 decode failed: " + message);
  }
  bz3_free(state);
  output.resize(expected_size);
  return output;
}

int ablation_mode(const std::string& omitted_operator) {
  if (omitted_operator == "block transform") return 1;
  if (omitted_operator == "special code") return 2;
  if (omitted_operator == "run length") return 3;
  if (omitted_operator == "lz_match") return 4;
  return 0;
}

Result run(const Dataset& dataset, const std::string& omitted_operator) {
  Result result;
  result.omitted_operator = omitted_operator;
  const int mode = ablation_mode(omitted_operator);
  for (const auto& column : dataset.columns) {
    const auto t0 = Clock::now();
    const Bytes encoded = encode_column(column, mode);
    const auto t1 = Clock::now();
    const Bytes decoded = decode_column(encoded, column.size(), mode);
    const auto t2 = Clock::now();
    result.original_bytes += column.size();
    result.compressed_bytes += encoded.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  return result;
}

void write_results(const std::string& path, const Dataset& dataset,
                   const std::vector<Result>& results) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write output: " + path);
  output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
  const double baseline = static_cast<double>(results.front().compressed_bytes) /
                          results.front().original_bytes;
  output << std::setprecision(9);
  for (const auto& result : results) {
    const double rate = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    output << "Air-Quality,Bzip3,"
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << baseline << ',' << rate - baseline << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows
           << ",block_1m,bzip3_upstream_source_operator_ablation\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: bzip3_source_air_bench DATASET OUTPUT\n";
      return 2;
    }
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<Result> results = {
        run(dataset, ""),
        run(dataset, "run length"),
        run(dataset, "lz_match"),
        run(dataset, "block transform"),
        run(dataset, "special code"),
    };
    write_results(argv[2], dataset, results);
    for (const auto& result : results) {
      std::cout << (result.omitted_operator.empty() ? "baseline" : result.omitted_operator)
                << " lossless=" << result.lossless << " compressed="
                << result.compressed_bytes << '\n';
    }
    return std::all_of(results.begin(), results.end(),
                       [](const Result& result) { return result.lossless; })
               ? 0
               : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
