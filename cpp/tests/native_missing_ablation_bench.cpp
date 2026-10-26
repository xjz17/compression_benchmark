#include "native_lz_family.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<char>;

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  for (char c : line) {
    if (c == ',') {
      fields.push_back(std::move(field));
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  fields.push_back(std::move(field));
  return fields;
}

std::vector<Bytes> load_columns(const std::string& path, std::size_t& rows) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dataset");
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty dataset");
  const auto headers = split(line);
  std::vector<std::vector<std::string>> values(headers.size());
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() != headers.size()) throw std::runtime_error("CSV width mismatch");
    ++rows;
    for (std::size_t i = 0; i < fields.size(); ++i) values[i].push_back(fields[i]);
  }
  std::vector<Bytes> columns;
  columns.reserve(values.size());
  for (const auto& column : values) {
    std::vector<std::int64_t> integers;
    bool all_integer = true;
    for (const auto& text : column) {
      std::int64_t value = 0;
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        all_integer = false;
        break;
      }
      integers.push_back(value);
    }
    if (all_integer) {
      Bytes bytes(integers.size() * sizeof(std::int64_t));
      if (!bytes.empty()) std::memcpy(bytes.data(), integers.data(), bytes.size());
      columns.push_back(std::move(bytes));
      continue;
    }
    std::vector<double> doubles;
    doubles.reserve(column.size());
    for (const auto& text : column) doubles.push_back(std::stod(text));
    Bytes bytes(doubles.size() * sizeof(double));
    if (!bytes.empty()) std::memcpy(bytes.data(), doubles.data(), bytes.size());
    columns.push_back(std::move(bytes));
  }
  return columns;
}

struct Row {
  std::string method;
  std::string omitted;
  std::size_t original = 0;
  std::size_t compressed = 0;
  long long compression_ns = 0;
  long long decompression_ns = 0;
};

Row run(const std::vector<Bytes>& columns, std::string method, std::string omitted) {
  Row row{std::move(method), std::move(omitted)};
  for (const auto& column : columns) {
    Bytes encoded;
    Bytes decoded;
    const auto t0 = std::chrono::steady_clock::now();
    if (row.method == "LZSSE") {
      encoded = compress_native_lz_family_ablation(column, "lzsse8",
          row.omitted == "block_state_control" ? "match_token_format" : row.omitted);
    } else {
      const auto payload = compress_native_lz_family(column, "tamp_15");
      if (row.omitted.empty()) {
        encoded.reserve(payload.size() + 1);
        encoded.push_back(static_cast<char>(0x54));
        encoded.insert(encoded.end(), payload.begin(), payload.end());
      } else {
        encoded = payload;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    row.compression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    if (row.method == "LZSSE") {
      decoded = decompress_native_lz_family_ablation(encoded, "lzsse8", column.size(),
          row.omitted == "block_state_control" ? "match_token_format" : row.omitted);
    } else if (row.omitted.empty()) {
      decoded = decompress_native_lz_family(
          Bytes(encoded.begin() + 1, encoded.end()), "tamp_15", column.size());
    } else {
      decoded = decompress_native_lz_family(encoded, "tamp_15", column.size());
    }
    const auto t2 = std::chrono::steady_clock::now();
    row.decompression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    row.original += column.size();
    row.compressed += encoded.size();
    if (decoded != column) throw std::runtime_error(row.method + " bypass is not lossless");
  }
  return row;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    std::size_t rows = 0;
    const auto columns = load_columns(argv[1], rows);
    const auto lz_base = run(columns, "LZSSE", "");
    const auto lz_abl = run(columns, "LZSSE", "block_state_control");
    const auto tamp_base = run(columns, "TAMP", "");
    const auto tamp_abl = run(columns, "TAMP", "container_frame");
    std::ofstream output(argv[2]);
    output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
    for (const auto& pair : {std::pair{lz_base, lz_base}, std::pair{lz_base, lz_abl},
                             std::pair{tamp_base, tamp_base}, std::pair{tamp_base, tamp_abl}}) {
      const auto& row = pair.second;
      const double base = static_cast<double>(pair.first.compressed) / pair.first.original;
      const double rate = static_cast<double>(row.compressed) / row.original;
      output << "Bird-migration," << row.method << ','
             << (row.omitted.empty() ? "none" : row.omitted) << ','
             << row.original << ',' << row.compressed << ',' << std::setprecision(9)
             << rate << ',' << base << ',' << rate - base
             << ',' << row.compression_ns << ',' << row.decompression_ns
             << ",1,2,4,1,7," << rows
             << ",native_operator_alias,native_operator_alias_validation\n";
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
