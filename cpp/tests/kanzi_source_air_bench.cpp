#include "misc/kanzi-cpp/src/InputStream.hpp"
#include "misc/kanzi-cpp/src/OutputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedInputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedOutputStream.hpp"
#include "misc/kanzi-cpp/src/util/fixedbuf.hpp"

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
constexpr int kBlockSize = 4 * 1024 * 1024;

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

Bytes encode(const Bytes& input, const std::string& transform,
             const std::string& entropy) {
  Bytes output(input.size() * 2 + 65536);
  ofixedbuf buffer(reinterpret_cast<char*>(output.data()), output.size());
  std::iostream stream(&buffer);
  kanzi::CompressedOutputStream compressed(stream, 1, entropy, transform,
                                           kBlockSize, 0, input.size());
  compressed.write(reinterpret_cast<const char*>(input.data()),
                   static_cast<std::streamsize>(input.size()));
  compressed.close();
  const std::size_t size = static_cast<std::size_t>(compressed.getWritten());
  if (size == 0 || size > output.size()) throw std::runtime_error("Kanzi encode failed");
  output.resize(size);
  return output;
}

Bytes decode(const Bytes& input, std::size_t expected_size) {
  ifixedbuf buffer(reinterpret_cast<char*>(const_cast<unsigned char*>(input.data())),
                   input.size());
  std::iostream stream(&buffer);
  kanzi::CompressedInputStream compressed(stream, 1);
  Bytes output(expected_size);
  compressed.read(reinterpret_cast<char*>(output.data()),
                  static_cast<std::streamsize>(output.size()));
  const std::size_t decoded = static_cast<std::size_t>(compressed.gcount());
  compressed.close();
  if (decoded != expected_size) throw std::runtime_error("Kanzi decode failed");
  return output;
}

Result run(const Dataset& dataset, const std::string& omitted_operator) {
  Result result;
  result.omitted_operator = omitted_operator;
  const std::string transform = omitted_operator == "block transform"
      ? "NONE"
      : "TEXT+UTF+BWT+RANK+ZRLT";
  const std::string entropy = omitted_operator == "special code" ? "NONE" : "ANS0";
  if (omitted_operator == "block/state") {
    Bytes joined;
    for (const auto& column : dataset.columns)
      joined.insert(joined.end(), column.begin(), column.end());
    const auto t0 = Clock::now();
    const Bytes encoded = encode(joined, transform, entropy);
    const auto t1 = Clock::now();
    const Bytes decoded = decode(encoded, joined.size());
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
    const Bytes encoded = encode(column, transform, entropy);
    const auto t1 = Clock::now();
    const Bytes decoded = decode(encoded, column.size());
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
  if (!output) throw std::runtime_error("cannot write output");
  output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
  const double baseline = static_cast<double>(results.front().compressed_bytes) /
      results.front().original_bytes;
  output << std::setprecision(9);
  for (const auto& result : results) {
    const double rate = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    output << "Air-Quality,Kanzi,"
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << baseline << ',' << (rate - baseline) << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows
           << ",level5_bwt_ans0,kanzi_upstream_source_operator_ablation\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: kanzi_source_air_bench DATASET.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<Result> results = {
        run(dataset, ""), run(dataset, "block transform"), run(dataset, "special code"),
        run(dataset, "block/state")};
    for (const auto& result : results) {
      if (!result.lossless) throw std::runtime_error("Kanzi lossless verification failed");
    }
    write_results(argv[2], dataset, results);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
