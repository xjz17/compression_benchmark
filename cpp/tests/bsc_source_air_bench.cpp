#include <cstddef>

#include "bwt/libbsc/libbsc/bwt/bwt.h"
#include "bwt/libbsc/libbsc/coder/coder.h"
#include "bwt/libbsc/libbsc/libbsc.h"

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
constexpr int kFeatures = LIBBSC_FEATURE_NONE;
constexpr int kCoder = LIBBSC_CODER_QLFC_STATIC;

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

void append_u32(Bytes& output, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    output.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
  }
}

std::uint32_t read_u32(const Bytes& input, std::size_t& position) {
  if (position + 4 > input.size()) throw std::runtime_error("BSC ablation header truncated");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(input[position++]) << shift;
  }
  return value;
}

Bytes encode_baseline(const Bytes& input) {
  Bytes output(input.size() + LIBBSC_HEADER_SIZE);
  const int size = bsc_compress(input.data(), output.data(), static_cast<int>(input.size()),
                                0, 0, LIBBSC_BLOCKSORTER_BWT, kCoder, kFeatures);
  if (size < 0) throw std::runtime_error("BSC baseline compression failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes decode_baseline(const Bytes& input, std::size_t expected_size) {
  Bytes output(expected_size);
  const int status = bsc_decompress(input.data(), static_cast<int>(input.size()), output.data(),
                                    static_cast<int>(output.size()), kFeatures);
  if (status != LIBBSC_NO_ERROR) throw std::runtime_error("BSC baseline decompression failed");
  return output;
}

Bytes encode_without_block_transform(const Bytes& input) {
  Bytes output{'B', 'S', 'C', 'C'};
  append_u32(output, static_cast<std::uint32_t>(input.size()));
  output.resize(output.size() + input.size());
  const int size = bsc_coder_compress(input.data(), output.data() + 8,
                                      static_cast<int>(input.size()), kCoder, kFeatures);
  if (size >= 0) {
    output[3] = 'Q';
    output.resize(8 + static_cast<std::size_t>(size));
  } else {
    output[3] = 'R';
    std::copy(input.begin(), input.end(), output.begin() + 8);
  }
  return output;
}

Bytes decode_without_block_transform(const Bytes& input) {
  if (input.size() < 8 || input[0] != 'B' || input[1] != 'S' || input[2] != 'C') {
    throw std::runtime_error("BSC coder-only magic mismatch");
  }
  std::size_t position = 4;
  const std::size_t size = read_u32(input, position);
  Bytes output(size);
  if (input[3] == 'R') {
    if (input.size() != 8 + size) throw std::runtime_error("BSC raw fallback size mismatch");
    std::copy(input.begin() + 8, input.end(), output.begin());
  } else if (input[3] == 'Q') {
    const int decoded = bsc_coder_decompress(input.data() + 8, output.data(), kCoder, kFeatures);
    if (decoded != static_cast<int>(size)) throw std::runtime_error("BSC coder-only decode failed");
  } else {
    throw std::runtime_error("BSC coder-only mode mismatch");
  }
  return output;
}

Bytes encode_without_special_code(const Bytes& input) {
  Bytes transformed = input;
  const int index = bsc_bwt_encode(transformed.data(), static_cast<int>(transformed.size()),
                                   nullptr, nullptr, kFeatures);
  if (index < 0) throw std::runtime_error("BSC BWT encode failed");
  Bytes output{'B', 'S', 'C', 'B'};
  append_u32(output, static_cast<std::uint32_t>(input.size()));
  append_u32(output, static_cast<std::uint32_t>(index));
  output.insert(output.end(), transformed.begin(), transformed.end());
  return output;
}

Bytes decode_without_special_code(const Bytes& input) {
  if (input.size() < 12 || input[0] != 'B' || input[1] != 'S' || input[2] != 'C' ||
      input[3] != 'B') {
    throw std::runtime_error("BSC BWT-only magic mismatch");
  }
  std::size_t position = 4;
  const std::size_t size = read_u32(input, position);
  const int index = static_cast<int>(read_u32(input, position));
  if (input.size() != 12 + size) throw std::runtime_error("BSC BWT-only size mismatch");
  Bytes output(input.begin() + 12, input.end());
  const int status = bsc_bwt_decode(output.data(), static_cast<int>(output.size()), index,
                                    0, nullptr, kFeatures);
  if (status != LIBBSC_NO_ERROR) throw std::runtime_error("BSC BWT decode failed");
  return output;
}

Result run(const Dataset& dataset, const std::string& omitted_operator) {
  Result result;
  result.omitted_operator = omitted_operator;
  for (const auto& column : dataset.columns) {
    const auto t0 = Clock::now();
    const Bytes encoded = omitted_operator.empty()
        ? encode_baseline(column)
        : omitted_operator == "block transform"
            ? encode_without_block_transform(column)
            : encode_without_special_code(column);
    const auto t1 = Clock::now();
    const Bytes decoded = omitted_operator.empty()
        ? decode_baseline(encoded, column.size())
        : omitted_operator == "block transform"
            ? decode_without_block_transform(encoded)
            : decode_without_special_code(encoded);
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
    output << "Air-Quality,BSC,"
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << baseline << ',' << (rate - baseline) << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows
           << ",bwt_qlfc_static,bsc_upstream_source_operator_ablation\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: bsc_source_air_bench DATASET.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    if (bsc_init(kFeatures) != LIBBSC_NO_ERROR) throw std::runtime_error("BSC init failed");
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<Result> results = {
        run(dataset, ""), run(dataset, "block transform"), run(dataset, "special code")};
    for (const auto& result : results) {
      if (!result.lossless) throw std::runtime_error("BSC lossless verification failed");
    }
    write_results(argv[2], dataset, results);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
