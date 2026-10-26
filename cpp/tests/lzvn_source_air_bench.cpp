// Source-level Air-Quality validation for LZVN. Each ablation uses the
// upstream match finder or native token writer and a matching decoder.

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
#include "lz/lzfse/lzvn.h"
}

namespace {

using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;

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

std::uint32_t read_u32(const Bytes& input, std::size_t& position) {
  if (position + 4 > input.size()) throw std::runtime_error("LZVN token truncated");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(input[position++]) << shift;
  }
  return value;
}

Bytes encode_native(const Bytes& input) {
  Bytes output(input.size() * 2 + 1024);
  Bytes work(lzvn_encode_scratch_size());
  const std::size_t size = lzvn_encode_buffer(output.data(), output.size(), input.data(),
                                              input.size(), work.data());
  if (size == 0) throw std::runtime_error("LZVN baseline encode failed");
  output.resize(size);
  return output;
}

Bytes decode_native(const Bytes& input, std::size_t expected_size) {
  Bytes output(expected_size);
  Bytes work(lzvn_decode_scratch_size());
  const std::size_t size = lzvn_decode_buffer_scratch(
      output.data(), output.size(), input.data(), input.size(), work.data());
  if (size != expected_size) throw std::runtime_error("LZVN native decode size mismatch");
  return output;
}

Bytes encode_without_matches(const Bytes& input) {
  Bytes output(input.size() + input.size() / 128 + 64);
  const std::size_t size = lzvn_encode_buffer_without_matches(
      output.data(), output.size(), input.data(), input.size());
  if (size == 0) throw std::runtime_error("LZVN no-match encode failed");
  output.resize(size);
  return output;
}

Bytes encode_without_token_format(const Bytes& input) {
  Bytes output(input.size() * 6 + 1024);
  Bytes work(lzvn_encode_scratch_size());
  const std::size_t size = lzvn_encode_buffer_token_ablation(
      output.data(), output.size(), input.data(), input.size(), work.data());
  if (size == 0) throw std::runtime_error("LZVN token-ablation encode failed");
  output.resize(size);
  return output;
}

Bytes decode_without_token_format(const Bytes& input, std::size_t expected_size) {
  if (input.size() < 5 || input[0] != 'L' || input[1] != 'Z' || input[2] != 'T' ||
      input[3] != 'A') {
    throw std::runtime_error("LZVN token-ablation magic mismatch");
  }
  Bytes output;
  output.reserve(expected_size);
  std::size_t position = 4;
  bool ended = false;
  while (position < input.size()) {
    const std::uint8_t tag = input[position++];
    if (tag == 2) {
      ended = true;
      break;
    }
    if (tag == 0) {
      const std::size_t length = read_u32(input, position);
      if (position + length > input.size() || output.size() + length > expected_size) {
        throw std::runtime_error("LZVN token-ablation literal invalid");
      }
      output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(position),
                    input.begin() + static_cast<std::ptrdiff_t>(position + length));
      position += length;
      continue;
    }
    if (tag != 1) throw std::runtime_error("LZVN token-ablation tag invalid");
    const std::size_t literal_length = read_u32(input, position);
    const std::size_t match_length = read_u32(input, position);
    const std::size_t distance = read_u32(input, position);
    if (position + literal_length > input.size() ||
        output.size() + literal_length + match_length > expected_size) {
      throw std::runtime_error("LZVN token-ablation match size invalid");
    }
    output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(position),
                  input.begin() + static_cast<std::ptrdiff_t>(position + literal_length));
    position += literal_length;
    if (distance == 0 || distance > output.size()) {
      throw std::runtime_error("LZVN token-ablation distance invalid");
    }
    const std::size_t start = output.size() - distance;
    for (std::size_t i = 0; i < match_length; ++i) output.push_back(output[start + i]);
  }
  if (!ended || position != input.size() || output.size() != expected_size) {
    throw std::runtime_error("LZVN token-ablation stream incomplete");
  }
  return output;
}

Result run(const Dataset& dataset, const std::string& omitted_operator) {
  Result result;
  result.omitted_operator = omitted_operator;
  if (omitted_operator == "block/state") {
    Bytes joined;
    for (const auto& column : dataset.columns)
      joined.insert(joined.end(), column.begin(), column.end());
    const auto t0 = Clock::now();
    const Bytes encoded = encode_native(joined);
    const auto t1 = Clock::now();
    const Bytes decoded = decode_native(encoded, joined.size());
    const auto t2 = Clock::now();
    result.original_bytes = joined.size();
    result.compressed_bytes = encoded.size();
    result.compression_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = decoded == joined;
    return result;
  }
  for (const auto& column : dataset.columns) {
    const auto t0 = Clock::now();
    const Bytes encoded = omitted_operator.empty()
                              ? encode_native(column)
                              : omitted_operator == "lz_match"
                                    ? encode_without_matches(column)
                                    : encode_without_token_format(column);
    const auto t1 = Clock::now();
    const Bytes decoded = omitted_operator == "match_token_format"
                              ? decode_without_token_format(encoded, column.size())
                              : decode_native(encoded, column.size());
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
    output << "Air-Quality,LZVN,"
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << baseline << ',' << rate - baseline << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows
           << ",lzvn,lzvn_upstream_source_operator_ablation\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: lzvn_source_air_bench DATASET OUTPUT\n";
      return 2;
    }
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<Result> results = {
        run(dataset, ""),
        run(dataset, "lz_match"),
        run(dataset, "match_token_format"),
        run(dataset, "block/state"),
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
