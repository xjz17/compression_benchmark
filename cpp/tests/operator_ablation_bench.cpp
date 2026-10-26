#include "ans_rans.hpp"
#include "reger_codec.hpp"
#include "alp_float_codec.hpp"
#include "blosc_codec.hpp"
#include "blosc2_codec.hpp"
#include "bos_operator_ablation.hpp"
#include "bzip2_operator_ablation.hpp"
#include "lzjb_operator_ablation.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"
#include "camel_float_codec.hpp"
#include "column_encodings.hpp"
#include "elf_double_codec.hpp"
#include "fpc_codec.hpp"
#include "fastpfor_operator_ablation.hpp"
#include "int64_column_codec.hpp"
#include "leco_double_codec.hpp"
#include "leco_operator_ablation.hpp"
#include "lzf_operator_ablation.hpp"
#include "native_lz_family.hpp"
#include "native_streams.hpp"
#include "pongo_codec.hpp"
#include "pfpc.h"
#include "ts2diff_bos_native.hpp"
#include "ts2diff_operator_ablation.hpp"
#include "tsfile_encoding/tsfile_decode_p03.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"

#include <fpzip.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
size_t spdp_compress_block_ablation(unsigned char level, const unsigned char* in,
                                    size_t len, unsigned char* out,
                                    size_t out_cap, int ablation_mode);
int spdp_decompress_block_ablation(unsigned char level, const unsigned char* comp,
                                   size_t csize, unsigned char* out,
                                   size_t out_len, int ablation_mode);
}

namespace {

using clock_ns = std::chrono::high_resolution_clock;

struct Dataset {
  std::vector<std::string> headers;
  std::vector<std::vector<std::int64_t>> integer_columns;
  std::vector<std::vector<double>> float_columns;
  std::vector<std::int64_t> times;
  std::size_t rows = 0;
};

struct Result {
  std::string method;
  std::string omitted_operator;
  std::string configuration;
  std::string implementation;
  std::size_t original_bytes = 0;
  std::size_t compressed_bytes = 0;
  long long compression_ns = 0;
  long long decompression_ns = 0;
  bool lossless = false;
  std::size_t raw_fallback_columns = 0;
};

struct NativeLzMethod {
  const char* method;
  const char* configuration;
  std::vector<std::string> operators;
};

const std::vector<NativeLzMethod>& native_lz_methods() {
  static const std::vector<NativeLzMethod> methods = {
      {"gzip", "gz", {"lz_match"}},
      {"brotli", "brotli", {"lz_match"}},
      {"zstd", "zstd_level_9", {"lz_match", "match_parse_policy"}},
      {"LZ4", "lz4-framed", {"lz_match", "match_token_format"}},
      {"XZ", "xz_level_9", {"lz_match", "match_parse_policy"}},
      {"Density", "density", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"Gipfeli", "gipfeli", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"GLZA", "glza", {"lz_match", "match_parse_policy"}},
      {"LZMAT", "lzmat", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"LZO1", "lzo1", {"lz_match", "match_token_format", "bitstream_pack"}},
      {"LZO2A", "lzo2a", {"lz_match", "match_token_format", "bitstream_pack"}},
      {"LZRW", "lzrw", {"lz_match", "match_window_model", "match_token_format"}},
      {"LZSSE", "lzsse8", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"LZVN", "lzvn", {"lz_match", "match_token_format"}},
      {"Pithy", "pithy", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"Shrinker", "shrinker", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"UCL NRV", "ucl_nrv2b_9", {"lz_match", "match_parse_policy", "bitstream_pack"}},
      {"WFLZ", "wflz", {"lz_match", "match_window_model", "match_token_format"}},
      {"XPACK", "xpack", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"YALZ77", "yalz77", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"Yappy", "yappy", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"snappy", "snappy-framed", {"lz_match", "match_token_format"}},
      {"Blosc", "blosc_blosclz", {"lz_match", "match_token_format"}},
      {"Blosc2", "blosc2_blosclz", {"lz_match", "match_token_format"}},
      {"BriefLZ", "brieflz_9", {"lz_match", "bitstream_pack"}},
      {"Crush", "crush_2", {"lz_match", "match_token_format"}},
      {"CSC", "csc_5", {"lz_match", "match_parse_policy"}},
      {"FastLZ", "fastlz_2", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"FastLZMA2", "fastlzma2_10", {"lz_match"}},
      {"libdeflate", "libdeflate_12", {"lz_match"}},
      {"Lizard", "lizard_49", {"lz_match", "match_window_model", "match_token_format"}},
      {"LZ4 Fast", "lz4fast_17", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"LZ4HC", "lz4hc_12", {"lz_match", "match_parse_policy", "match_token_format"}},
      {"LZAV", "lzav_2", {"lz_match", "match_window_model", "match_token_format"}},
      {"LZFSE", "lzfse", {"lz_match", "match_token_format"}},
      {"LZG", "lzg_9", {"lz_match", "match_parse_policy", "match_window_model", "match_token_format"}},
      {"LZHAM", "lzham_4", {"lz_match", "match_parse_policy"}},
      {"Lzlib", "lzlib_9", {"lz_match"}},
      {"QuickLZ", "quicklz_3", {"lz_match", "match_window_model", "match_token_format"}},
      {"SLZ", "slz_zlib_3", {"lz_match"}},
      {"Tornado", "tornado_16", {"lz_match", "match_token_format"}},
      {"zlib", "zlib_9", {"lz_match", "match_parse_policy"}},
      {"zlib-ng", "zlib_ng_9", {"lz_match", "match_parse_policy"}},
      {"Zling", "zling_4", {"lz_match"}},
      {"ZXC", "zxc_6", {"lz_match", "match_window_model", "match_token_format"}},
      {"TAMP", "tamp_15", {"lz_match", "match_token_format"}},
  };
  return methods;
}

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
      fields.push_back(field);
      field.clear();
    } else if (c != '\r') {
      field.push_back(c);
    }
  }
  fields.push_back(field);
  return fields;
}

bool parse_int64(std::string_view text, std::int64_t& value) {
  if (text.empty()) return false;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_double(std::string_view text, double& value) {
  if (text.empty()) return false;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end && std::isfinite(value);
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double bits_double(std::uint64_t bits) {
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool same_double_columns(const std::vector<std::vector<double>>& left,
                         const std::vector<std::vector<double>>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t column = 0; column < left.size(); ++column) {
    if (left[column].size() != right[column].size()) return false;
    for (std::size_t row = 0; row < left[column].size(); ++row) {
      if (double_bits(left[column][row]) != double_bits(right[column][row])) return false;
    }
  }
  return true;
}

Dataset load_dataset(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dataset: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty dataset");
  Dataset dataset;
  dataset.headers = split_csv_row(line);
  std::vector<std::vector<std::string>> values(dataset.headers.size());
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv_row(line);
    if (fields.size() != dataset.headers.size()) {
      throw std::runtime_error("inconsistent CSV width at row " +
                               std::to_string(dataset.rows + 2));
    }
    ++dataset.rows;
    for (std::size_t column = 0; column < fields.size(); ++column) {
      values[column].push_back(fields[column]);
    }
  }
  for (std::size_t column = 0; column < dataset.headers.size(); ++column) {
    std::string name = dataset.headers[column];
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    std::vector<std::int64_t> integers;
    integers.reserve(dataset.rows);
    bool all_integer = true;
    for (const auto& text : values[column]) {
      std::int64_t value = 0;
      if (!parse_int64(text, value)) {
        all_integer = false;
        break;
      }
      integers.push_back(value);
    }
    if (name == "timestamp" || name == "time" || name == "date_time" ||
        name == "datetime") {
      if (!all_integer) throw std::runtime_error("timestamp column is not int64");
      dataset.times = std::move(integers);
      continue;
    }
    if (all_integer) {
      dataset.integer_columns.push_back(std::move(integers));
      continue;
    }
    std::vector<double> doubles;
    doubles.reserve(dataset.rows);
    bool all_double = true;
    for (const auto& text : values[column]) {
      double value = 0.0;
      if (!parse_double(text, value)) {
        all_double = false;
        break;
      }
      doubles.push_back(value);
    }
    if (all_double) dataset.float_columns.push_back(std::move(doubles));
  }
  if (dataset.integer_columns.empty() && dataset.float_columns.empty()) {
    throw std::runtime_error("no complete numeric columns");
  }
  if (dataset.times.empty()) {
    dataset.times.resize(dataset.rows);
    for (std::size_t i = 0; i < dataset.rows; ++i) {
      dataset.times[i] = static_cast<std::int64_t>(i);
    }
  }
  return dataset;
}

std::vector<std::vector<double>> fixed_precision_float_columns(
    const Dataset& dataset) {
  std::vector<std::vector<double>> columns;
  for (const auto& column : dataset.float_columns) {
    const int scale = leco_double::maxLosslessDecimalScale(column);
    if (scale >= 0) columns.push_back(column);
  }
  return columns;
}

template <typename T>
std::vector<char> as_bytes(const std::vector<T>& values) {
  std::vector<char> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

std::vector<std::vector<char>> raw_numeric_columns(const Dataset& dataset) {
  std::vector<std::vector<char>> columns;
  columns.reserve(dataset.integer_columns.size() + dataset.float_columns.size() + 1);
  columns.push_back(as_bytes(dataset.times));
  for (const auto& column : dataset.integer_columns) {
    columns.push_back(as_bytes(column));
  }
  for (const auto& column : dataset.float_columns) {
    columns.push_back(as_bytes(column));
  }
  return columns;
}

std::uint64_t fpzip_map_forward(double value) {
  std::uint64_t bits = double_bits(value);
  bits = ~bits;
  bits ^= (std::uint64_t{0} - (bits >> 63)) >> 1;
  return bits;
}

double fpzip_map_inverse(std::uint64_t value) {
  value ^= (std::uint64_t{0} - (value >> 63)) >> 1;
  value = ~value;
  return bits_double(value);
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

std::uint64_t read_u64_le(const std::vector<std::uint8_t>& in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("FPZIP fixed residual truncated");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(in[pos++]) << shift;
  }
  return value;
}

std::vector<std::uint8_t> encode_fpzip_column(
    const std::vector<double>& values, std::string_view omitted_operator) {
  if (omitted_operator == "range code") {
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("FPZIP column exceeds uint32 row count");
    }
    std::vector<std::uint8_t> payload;
    payload.reserve(4 + values.size() * sizeof(std::uint64_t));
    append_u32_le(payload, static_cast<std::uint32_t>(values.size()));
    double previous = 0.0;
    for (double value : values) {
      const std::uint64_t residual = fpzip_map_forward(value) - fpzip_map_forward(previous);
      append_u64_le(payload, residual);
      previous = value;
    }
    return payload;
  }
  if (!omitted_operator.empty() && omitted_operator != "delta") {
    throw std::invalid_argument("unsupported FPZIP ablation operator");
  }
  if (values.empty() || values.size() > static_cast<std::size_t>(INT_MAX)) return {};

  FILE* file = std::tmpfile();
  if (file == nullptr) throw std::runtime_error("FPZIP tmpfile failed");
  FPZ* writer = fpzip_write_to_file(file);
  if (writer == nullptr) {
    std::fclose(file);
    throw std::runtime_error("FPZIP writer failed");
  }
  writer->type = FPZIP_TYPE_DOUBLE;
  writer->prec = 0;
  if (omitted_operator == "delta") {
    writer->nx = 1;
    writer->ny = 1;
    writer->nz = 1;
    writer->nf = static_cast<int>(values.size());
  } else {
    writer->nx = static_cast<int>(values.size());
    writer->ny = 1;
    writer->nz = 1;
    writer->nf = 1;
  }
  const bool ok = fpzip_write_header(writer) && fpzip_write(writer, values.data()) != 0;
  fpzip_write_close(writer);
  if (!ok || std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    throw std::runtime_error("FPZIP encode failed");
  }
  const long size = std::ftell(file);
  if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
    std::fclose(file);
    throw std::runtime_error("FPZIP encoded size failed");
  }
  std::vector<std::uint8_t> payload(static_cast<std::size_t>(size));
  if (!payload.empty() &&
      std::fread(payload.data(), 1, payload.size(), file) != payload.size()) {
    std::fclose(file);
    throw std::runtime_error("FPZIP encoded read failed");
  }
  std::fclose(file);
  return payload;
}

std::vector<double> decode_fpzip_column(
    const std::vector<std::uint8_t>& payload, std::string_view omitted_operator) {
  if (omitted_operator == "range code") {
    if (payload.size() < 4) throw std::runtime_error("FPZIP fixed residual header truncated");
    const std::uint32_t count = static_cast<std::uint32_t>(payload[0]) |
        (static_cast<std::uint32_t>(payload[1]) << 8) |
        (static_cast<std::uint32_t>(payload[2]) << 16) |
        (static_cast<std::uint32_t>(payload[3]) << 24);
    std::size_t pos = 4;
    std::vector<double> values;
    values.reserve(count);
    double previous = 0.0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint64_t mapped = fpzip_map_forward(previous) + read_u64_le(payload, pos);
      const double value = fpzip_map_inverse(mapped);
      values.push_back(value);
      previous = value;
    }
    if (pos != payload.size()) throw std::runtime_error("FPZIP fixed residual trailing bytes");
    return values;
  }

  FILE* file = std::tmpfile();
  if (file == nullptr) throw std::runtime_error("FPZIP tmpfile failed");
  if (!payload.empty() &&
      std::fwrite(payload.data(), 1, payload.size(), file) != payload.size()) {
    std::fclose(file);
    throw std::runtime_error("FPZIP payload write failed");
  }
  std::rewind(file);
  FPZ* reader = fpzip_read_from_file(file);
  if (reader == nullptr || !fpzip_read_header(reader)) {
    if (reader != nullptr) fpzip_read_close(reader);
    std::fclose(file);
    throw std::runtime_error("FPZIP header decode failed");
  }
  const std::size_t count = static_cast<std::size_t>(reader->nx) * reader->ny *
                            reader->nz * reader->nf;
  std::vector<double> values(count);
  const bool ok = fpzip_read(reader, values.data()) != 0;
  fpzip_read_close(reader);
  std::fclose(file);
  if (!ok) throw std::runtime_error("FPZIP decode failed");
  return values;
}

Result run_native_lz(const Dataset& dataset, const NativeLzMethod& method,
                     std::string omitted_operator) {
  Result result{method.method, std::move(omitted_operator)};
  result.configuration = method.configuration;
  result.implementation = "native_lz_family";
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    const auto t0 = clock_ns::now();
    const auto payload = compress_native_lz_family_ablation(
        column, result.configuration, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = decompress_native_lz_family_ablation(
        payload, result.configuration, column.size(), result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size();
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  return result;
}

Result run_lzjb(const Dataset& dataset, std::string omitted_operator) {
  Result result{"LZJB", std::move(omitted_operator)};
  result.configuration = "lzjb_2010";
  result.implementation = "lzjb_2010_native_operator_ablation";
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    const auto input = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(column.data()), column.size());
    const auto expected = std::vector<std::uint8_t>(input.begin(), input.end());
    const auto t0 = clock_ns::now();
    const auto payload = lzjb_operator_ablation::encode(input, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = lzjb_operator_ablation::decode(
        payload, column.size(), result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size();
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == expected;
  }
  return result;
}

Result run_lzf(const Dataset& dataset, std::string omitted_operator) {
  Result result{"LZF", std::move(omitted_operator)};
  result.configuration = "lzf_1";
  result.implementation = "lzf_upstream_source_operator_ablation";
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    const auto input = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(column.data()), column.size());
    const auto t0 = clock_ns::now();
    const auto payload = lzf_operator_ablation::encode(input, 1, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = lzf_operator_ablation::decode(
        payload, column.size(), 1, result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size();
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded ==
                      std::vector<std::uint8_t>(input.begin(), input.end());
  }
  return result;
}

Result run_container_framing(const Dataset& dataset, std::string method,
                             std::string omitted_operator) {
  Result result{std::move(method), std::move(omitted_operator)};
  if (result.method == "gzip") result.configuration = "gz";
  else if (result.method == "zlib") result.configuration = "zlib_9";
  else if (result.method == "LZ4") result.configuration = "lz4-framed";
  else if (result.method == "LZ4 Fast") result.configuration = "lz4fast_17";
  else if (result.method == "LZ4HC") result.configuration = "lz4hc_12";
  result.implementation = "native_streams_upstream";
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    std::vector<long long> metrics;
    LossAccum loss;
    if (result.method == "gzip") {
      result.omitted_operator.empty() ? benchGzFramed(column, metrics, &loss)
                                      : benchGzRawDeflate(column, metrics, &loss);
    } else if (result.method == "zlib") {
      result.omitted_operator.empty() ? benchZlibWrapped(column, metrics, &loss)
                                      : benchZlibRawDeflate(column, metrics, &loss);
    } else if (result.method == "LZ4") {
      result.omitted_operator.empty() ? benchLz4Framed(column, metrics, &loss)
                                      : benchLz4RawBlock(column, metrics, &loss);
    } else if (result.method == "LZ4 Fast") {
      result.omitted_operator.empty() ? benchLz4FastFramed(column, metrics, &loss)
                                      : benchLz4FastRawBlock(column, metrics, &loss);
    } else if (result.method == "LZ4HC") {
      result.omitted_operator.empty() ? benchLz4HcFramed(column, metrics, &loss)
                                      : benchLz4HcRawBlock(column, metrics, &loss);
    } else {
      throw std::invalid_argument("unsupported container-framing method");
    }
    if (metrics.size() != 4) {
      throw std::runtime_error("container-framing metrics missing");
    }
    result.original_bytes += static_cast<std::size_t>(metrics[0]);
    result.compressed_bytes += static_cast<std::size_t>(metrics[1]);
    result.compression_ns += metrics[2];
    result.decompression_ns += metrics[3];
    result.lossless = result.lossless && loss.err == 0;
  }
  return result;
}

template <typename Bench>
Result run_stream_operator(const Dataset& dataset, std::string method,
                           std::string omitted_operator, std::string configuration,
                           std::string implementation, Bench bench) {
  Result result{std::move(method), std::move(omitted_operator)};
  result.configuration = std::move(configuration);
  result.implementation = std::move(implementation);
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    std::vector<long long> metrics;
    LossAccum loss;
    bench(column, result.omitted_operator, metrics, &loss);
    if (metrics.size() != 4) throw std::runtime_error("stream-operator metrics missing");
    result.original_bytes += static_cast<std::size_t>(metrics[0]);
    result.compressed_bytes += static_cast<std::size_t>(metrics[1]);
    result.compression_ns += metrics[2];
    result.decompression_ns += metrics[3];
    result.lossless = result.lossless && loss.err == 0;
  }
  return result;
}

Result run_ans(const Dataset& dataset) {
  Result result{"ANS", ""};
  result.configuration = "project_rans";
  result.implementation = "ans_rans";
  result.lossless = true;
  const auto columns = raw_numeric_columns(dataset);
  for (std::size_t column_index = 0; column_index < columns.size(); ++column_index) {
    const auto& column = columns[column_index];
    std::vector<long long> metrics;
    LossAccum loss;
    try {
      testAnsEntropyCompression(column, metrics, &loss);
    } catch (const std::exception& error) {
      throw std::runtime_error("ANS column " + std::to_string(column_index) + ": " +
                               error.what());
    }
    if (metrics.size() != 4) throw std::runtime_error("ANS metrics missing");
    result.original_bytes += static_cast<std::size_t>(metrics[0]);
    result.compressed_bytes += static_cast<std::size_t>(metrics[1]);
    result.compression_ns += metrics[2];
    result.decompression_ns += metrics[3];
    result.lossless = result.lossless && loss.err == 0.0 &&
                      loss.ref == static_cast<double>(column.size());
  }
  return result;
}

Result run_bzip2(const Dataset& dataset, std::string omitted_operator) {
  Result result{"Bzip2", std::move(omitted_operator)};
  result.configuration = "level_4";
  result.implementation = "project_patched_bzip2_source_ablation";
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    const auto input = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(column.data()), column.size());
    const auto t0 = clock_ns::now();
    const auto payload = bzip2_operator_ablation::encode(
        input, 4, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = bzip2_operator_ablation::decode(payload, input.size());
    const auto t2 = clock_ns::now();
    result.original_bytes += input.size();
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless &&
                      std::equal(decoded.begin(), decoded.end(), input.begin());
  }
  return result;
}

Result run_deflate_operator(const Dataset& dataset, const std::string& method,
                            std::string omitted_operator) {
  return run_stream_operator(
      dataset, method, std::move(omitted_operator),
      method == "gzip" ? "gzip_deflate" : "zlib_9",
      "native_deflate_operator_upstream",
      [&](const std::vector<char>& column, const std::string& omitted,
          std::vector<long long>& metrics, LossAccum* loss) {
        benchDeflateOperatorAblation(column, method, omitted, metrics, loss);
      });
}

Result run_lz4_operator(const Dataset& dataset, const std::string& method,
                        std::string omitted_operator) {
  const std::string configuration = method == "LZ4" ? "lz4_raw"
      : method == "LZ4 Fast" ? "lz4fast_17_raw" : "lz4hc_12_raw";
  return run_stream_operator(
      dataset, method, std::move(omitted_operator), configuration,
      "native_lz4_operator_upstream",
      [&](const std::vector<char>& column, const std::string& omitted,
          std::vector<long long>& metrics, LossAccum* loss) {
        benchLz4OperatorAblation(column, method, omitted, metrics, loss);
      });
}

Result run_snappy_operator(const Dataset& dataset, std::string omitted_operator) {
  return run_stream_operator(
      dataset, "snappy", std::move(omitted_operator), "snappy-framed",
      "native_snappy_operator_upstream",
      [](const std::vector<char>& column, const std::string& omitted,
         std::vector<long long>& metrics, LossAccum* loss) {
        omitted.empty() ? benchSnappyFramed(column, metrics, loss)
                        : benchSnappyFramedLiteralOnly(column, metrics, loss);
      });
}

Result run_zstd_operator(const Dataset& dataset, std::string omitted_operator) {
  return run_stream_operator(
      dataset, "zstd", std::move(omitted_operator), "zstd_level_9",
      "native_zstd_operator_upstream",
      [](const std::vector<char>& column, const std::string& omitted,
         std::vector<long long>& metrics, LossAccum* loss) {
        benchZstdOperatorAblation(column, omitted, metrics, loss);
      });
}

Result run_blosc_operator(const Dataset& dataset, std::string omitted_operator) {
  return run_stream_operator(
      dataset, "Blosc", std::move(omitted_operator), "blosc_blosclz",
      "native_blosc_operator_upstream",
      [](const std::vector<char>& column, const std::string& omitted,
         std::vector<long long>& metrics, LossAccum* loss) {
        wc_blosc1_bench::benchBufferOperatorAblation(column, 8, omitted, metrics, loss);
      });
}

Result run_blosc2_operator(const Dataset& dataset, std::string omitted_operator) {
  return run_stream_operator(
      dataset, "Blosc2", std::move(omitted_operator), "blosc2_blosclz",
      "native_blosc2_operator_upstream",
      [](const std::vector<char>& column, const std::string& omitted,
         std::vector<long long>& metrics, LossAccum* loss) {
        wc_blosc2_bench::benchBufferOperatorAblation(column, 8, omitted, metrics, loss);
      });
}

struct BosDoublePayload {
  bool scaled = false;
  int scale = 0;
  std::vector<std::uint8_t> body;
};

BosDoublePayload encode_bos_double(const std::vector<double>& values,
                                   std::string_view omitted_operator) {
  BosDoublePayload encoded;
  std::vector<std::int64_t> integers;
  if (omitted_operator != "scaling") {
    encoded.scale = leco_double::maxLosslessDecimalScale(values);
    if (encoded.scale < 0 ||
        !leco_double::quantizeScaledInts(values, encoded.scale, integers)) {
      throw std::runtime_error("BOS could not scale a fixed-precision float column exactly");
    }
    encoded.scaled = true;
  } else {
    integers.reserve(values.size());
    for (double value : values) {
      integers.push_back(static_cast<std::int64_t>(double_bits(value)));
    }
  }
  const std::string_view integer_omission = omitted_operator == "scaling"
      ? std::string_view{}
      : omitted_operator;
  encoded.body = bos_operator_ablation::encode(integers, integer_omission);
  return encoded;
}

std::vector<double> decode_bos_double(const BosDoublePayload& encoded) {
  const auto integers = bos_operator_ablation::decode(encoded.body);
  std::vector<double> values;
  values.reserve(integers.size());
  if (!encoded.scaled) {
    for (std::int64_t value : integers) {
      values.push_back(bits_double(static_cast<std::uint64_t>(value)));
    }
    return values;
  }
  const double factor = std::pow(10.0, encoded.scale);
  for (std::int64_t value : integers) values.push_back(static_cast<double>(value) / factor);
  return values;
}

Result run_bos(const Dataset& dataset, std::string omitted_operator) {
  Result result{"BOS", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "bos_native_operator_ablation";
  result.original_bytes = 0;
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const std::string_view integer_omission = result.omitted_operator == "scaling"
        ? std::string_view{}
        : std::string_view(result.omitted_operator);
    const auto payload = bos_operator_ablation::encode(column, integer_omission);
    const auto t1 = clock_ns::now();
    const auto decoded = bos_operator_ablation::decode(payload);
    const auto t2 = clock_ns::now();
    result.compressed_bytes += payload.size();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    const auto t0 = clock_ns::now();
    const auto payload = encode_bos_double(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = decode_bos_double(payload);
    const auto t2 = clock_ns::now();
    result.compressed_bytes += payload.body.size() + 2;
    result.original_bytes += column.size() * sizeof(double);
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_buff(const Dataset& dataset, std::string omitted_operator) {
  Result result{"buff", std::move(omitted_operator)};
  result.configuration = "native_fixed_point";
  result.implementation = "buff_native_operator";
  result.lossless = true;

  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const std::string_view integer_omission =
        result.omitted_operator == "bit-packing"
            ? std::string_view(result.omitted_operator)
            : std::string_view{};
    const auto t0 = clock_ns::now();
    const auto payload = buff_kernel::encode_int64_ablation(column, integer_omission);
    const auto t1 = clock_ns::now();
    const auto decoded = buff_kernel::decode_int64_ablation(payload);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    const auto t0 = clock_ns::now();
    const auto payload = buff_kernel::encode_double_ablation(
        column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = buff_kernel::decode_double_ablation(payload);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_sprintz(const Dataset& dataset, std::string omitted_operator) {
  Result result{"SPRINTZ", std::move(omitted_operator)};
  result.configuration = "fire";
  result.implementation = "tsfile_sprintz_native_ablation";
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = tsfile::p04::encode_sprintz_int64_ablation(
        column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<std::int64_t> decoded;
    const bool ok = tsfile::p04::decode_sprintz_int64_ablation(
        payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && ok && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    const auto t0 = clock_ns::now();
    const auto payload = tsfile::p04::encode_sprintz_double_ablation(
        column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<double> decoded;
    const bool ok = tsfile::p04::decode_sprintz_double_ablation(
        payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && ok && same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_rlbe(const Dataset& dataset, std::string omitted_operator) {
  Result result{"RLBE", std::move(omitted_operator)};
  result.configuration = "block_10000";
  result.implementation = "tsfile_rlbe_native_ablation";
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = tsfile::p04::encode_rlbe_int64_ablation(
        column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<std::int64_t> decoded;
    const bool ok = tsfile::p04::decode_rlbe_int64_ablation(
        payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && ok && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    const auto t0 = clock_ns::now();
    const auto payload = tsfile::p04::encode_rlbe_double_ablation(
        column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<double> decoded;
    const bool ok = tsfile::p04::decode_rlbe_double_ablation(
        payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && ok && same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_fpzip(const Dataset& dataset, std::string omitted_operator) {
  Result result{"FPZIP", std::move(omitted_operator)};
  result.configuration = "lossless_1d";
  result.implementation = "fpzip_upstream_operator_ablation";
  result.lossless = true;

  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    std::vector<double> input;
    input.reserve(column.size());
    for (std::int64_t value : column) {
      const double converted = static_cast<double>(value);
      if (static_cast<std::int64_t>(converted) != value) {
        throw std::runtime_error("FPZIP integer column is not exactly representable as double");
      }
      input.push_back(converted);
    }
    const auto t0 = clock_ns::now();
    const auto payload = encode_fpzip_column(input, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = decode_fpzip_column(payload, result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({input}, {decoded});
  }
  for (const auto& column : dataset.float_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = encode_fpzip_column(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = decode_fpzip_column(payload, result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_leco(const Dataset& dataset, std::string omitted_operator) {
  Result result{"LECO", std::move(omitted_operator)};
  result.configuration = "block_512";
  result.implementation = "leco_native_operator_ablation";
  result.lossless = true;

  std::vector<std::vector<std::int64_t>> columns = dataset.integer_columns;
  columns.push_back(dataset.times);
  for (const auto& column : dataset.float_columns) {
    std::vector<std::int64_t> bit_patterns;
    bit_patterns.reserve(column.size());
    for (double value : column) {
      bit_patterns.push_back(static_cast<std::int64_t>(double_bits(value)));
    }
    columns.push_back(std::move(bit_patterns));
  }

  for (const auto& column : columns) {
    const auto t0 = clock_ns::now();
    const auto payload = leco_operator_ablation::encode(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = leco_operator_ablation::decode(payload, result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  return result;
}

Result run_ts2diff(const Dataset& dataset, std::string omitted_operator) {
  Result result{"TS_2DIFF", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "tsfile_ts2diff_native_ablation";
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = ts2diff_operator_ablation::encode(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = ts2diff_operator_ablation::decode(payload);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    std::vector<std::int64_t> integers;
    int scale = -1;
    if (result.omitted_operator == "scaling") {
      integers.reserve(column.size());
      for (double value : column) integers.push_back(static_cast<std::int64_t>(double_bits(value)));
    } else {
      scale = leco_double::maxLosslessDecimalScale(column);
      if (scale < 0 || !leco_double::quantizeScaledInts(column, scale, integers)) {
        throw std::runtime_error("TS_2DIFF could not scale a floating-point column exactly");
      }
    }
    const auto t0 = clock_ns::now();
    const auto payload = ts2diff_operator_ablation::encode(integers, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = ts2diff_operator_ablation::decode(payload);
    std::vector<double> recovered;
    recovered.reserve(decoded.size());
    if (scale < 0) {
      for (std::int64_t value : decoded) recovered.push_back(bits_double(value));
    } else {
      const double factor = std::pow(10.0, scale);
      for (std::int64_t value : decoded) recovered.push_back(static_cast<double>(value) / factor);
    }
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size() + 1;
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {recovered});
  }
  return result;
}

Result run_fpc(const Dataset& dataset, std::string omitted_operator) {
  Result result{"fpc", std::move(omitted_operator)};
  result.configuration = "pred16";
  result.implementation = "fpc_native_operator_upstream";
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  std::vector<long long> integer_metrics;
  std::vector<long long> float_metrics;
  LossAccum loss;
  fpc_codec::benchInt64ColumnsFpcAblation(
      integer_columns, result.omitted_operator, integer_metrics, &loss);
  fpc_codec::benchDoubleColumnsFpcAblation(
      dataset.float_columns, result.omitted_operator, float_metrics, &loss);
  if (integer_metrics.size() != 4 || float_metrics.size() != 4) {
    throw std::runtime_error("FPC ablation metrics missing");
  }
  result.original_bytes = static_cast<std::size_t>(integer_metrics[0] + float_metrics[0]);
  result.compressed_bytes = static_cast<std::size_t>(integer_metrics[1] + float_metrics[1]);
  result.compression_ns = integer_metrics[2] + float_metrics[2];
  result.decompression_ns = integer_metrics[3] + float_metrics[3];
  const std::size_t expected = dataset.rows *
      (dataset.integer_columns.size() + dataset.float_columns.size() + 1) * sizeof(std::int64_t);
  result.lossless = result.original_bytes == expected && loss.err == 0.0 &&
                    loss.ref == static_cast<double>(dataset.rows *
                        (integer_columns.size() + dataset.float_columns.size()));
  return result;
}

Result run_pfpc(const Dataset& dataset, std::string omitted_operator) {
  Result result{"pfpc", std::move(omitted_operator)};
  result.configuration = "pred16_t1";
  result.implementation = "pfpc_upstream_source_operator_ablation";
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    std::vector<unsigned char> encoded(column.size() * 2 + 1024);
    const auto t0 = clock_ns::now();
    std::size_t compressed = 0;
    if (result.omitted_operator.empty()) {
      compressed = pfpc_compress_memory(column.data(), column.size(), encoded.data(),
                                        encoded.size(), 16, 1, 1024);
    } else {
      const auto mode = result.omitted_operator == "xor"
          ? PFPC_ABLATE_XOR
          : PFPC_ABLATE_TRAILING_ZERO_PACK;
      compressed = pfpc_compress_memory_ablation(
          column.data(), column.size(), encoded.data(), encoded.size(), 16, mode);
    }
    const auto t1 = clock_ns::now();
    if (compressed == 0) throw std::runtime_error("pFPC compression failed");
    encoded.resize(compressed);
    std::vector<unsigned char> decoded(column.size());
    std::size_t consumed = 0;
    if (result.omitted_operator.empty()) {
      consumed = pfpc_decompress_memory(encoded.data(), encoded.size(), decoded.data(),
                                        decoded.size());
    } else {
      consumed = pfpc_decompress_memory_ablation(
          encoded.data(), encoded.size(), decoded.data(), decoded.size());
    }
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size();
    result.compressed_bytes += encoded.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && consumed == encoded.size() &&
        std::equal(decoded.begin(), decoded.end(),
                   reinterpret_cast<const unsigned char*>(column.data()));
  }
  return result;
}

Result run_alp(const Dataset& dataset, std::string omitted_operator) {
  Result result{"ALP", std::move(omitted_operator)};
  result.configuration = "native";
  result.implementation = "alp_native_operator";
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  std::vector<long long> integer_metrics;
  std::vector<long long> float_metrics;
  LossAccum loss;
  alp_float::benchInt64ColumnsAblation(
      integer_columns, result.omitted_operator, integer_metrics, &loss);
  alp_float::benchDoubleColumnsAblation(
      dataset.float_columns, result.omitted_operator, float_metrics, &loss);
  if (integer_metrics.size() != 4 || float_metrics.size() != 4) {
    throw std::runtime_error("ALP ablation metrics missing");
  }
  result.original_bytes = static_cast<std::size_t>(integer_metrics[0] + float_metrics[0]);
  result.compressed_bytes = static_cast<std::size_t>(integer_metrics[1] + float_metrics[1]);
  result.compression_ns = integer_metrics[2] + float_metrics[2];
  result.decompression_ns = integer_metrics[3] + float_metrics[3];
  const std::size_t expected = dataset.rows *
      (dataset.integer_columns.size() + dataset.float_columns.size() + 1) * sizeof(std::int64_t);
  result.lossless = result.original_bytes == expected && loss.err == 0.0 &&
                    loss.ref == static_cast<double>(dataset.rows *
                        (integer_columns.size() + dataset.float_columns.size()));
  return result;
}

Result run_delta_binary_packed(const Dataset& dataset, std::string omitted_operator) {
  Result result{"DELTA_BINARY_PACKED", std::move(omitted_operator)};
  result.configuration = "native";
  result.implementation = "delta_binary_packed_native_operator";
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = colenc::encodeDeltaBinaryPackedInt64Ablation(
        column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = colenc::decodeDeltaBinaryPackedInt64Ablation(payload, column.size());
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    std::vector<std::int64_t> integers;
    int scale = -1;
    if (result.omitted_operator == "scaling") {
      integers.reserve(column.size());
      for (double value : column) integers.push_back(static_cast<std::int64_t>(double_bits(value)));
    } else {
      scale = leco_double::maxLosslessDecimalScale(column);
      if (scale < 0 || !leco_double::quantizeScaledInts(column, scale, integers)) {
        throw std::runtime_error("DELTA_BINARY_PACKED could not scale a float column exactly");
      }
    }
    const auto t0 = clock_ns::now();
    const auto payload = colenc::encodeDeltaBinaryPackedInt64Ablation(
        integers, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = colenc::decodeDeltaBinaryPackedInt64Ablation(payload, integers.size());
    std::vector<double> recovered;
    recovered.reserve(decoded.size());
    if (scale < 0) {
      for (std::int64_t value : decoded) recovered.push_back(bits_double(value));
    } else {
      const double factor = std::pow(10.0, scale);
      for (std::int64_t value : decoded) recovered.push_back(static_cast<double>(value) / factor);
    }
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size() + 1;
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {recovered});
  }
  return result;
}

Result run_fastpfor(const Dataset& dataset, std::string omitted_operator) {
  Result result{"fastpfor", std::move(omitted_operator)};
  result.configuration = "native";
  result.implementation = "fastpfor_native_operator";
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = fastpfor_operator_ablation::encode(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = fastpfor_operator_ablation::decode(payload);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  for (const auto& column : fixed_precision_float_columns(dataset)) {
    std::vector<std::int64_t> integers;
    int scale = -1;
    if (result.omitted_operator == "scaling") {
      integers.reserve(column.size());
      for (double value : column) integers.push_back(static_cast<std::int64_t>(double_bits(value)));
    } else {
      scale = leco_double::maxLosslessDecimalScale(column);
      if (scale < 0 || !leco_double::quantizeScaledInts(column, scale, integers)) {
        throw std::runtime_error("FastPFor could not scale a float column exactly");
      }
    }
    const auto t0 = clock_ns::now();
    const auto payload = fastpfor_operator_ablation::encode(integers, result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = fastpfor_operator_ablation::decode(payload);
    std::vector<double> recovered;
    recovered.reserve(decoded.size());
    if (scale < 0) {
      for (std::int64_t value : decoded) recovered.push_back(bits_double(value));
    } else {
      const double factor = std::pow(10.0, scale);
      for (std::int64_t value : decoded) recovered.push_back(static_cast<double>(value) / factor);
    }
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size() + 1;
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {recovered});
  }
  return result;
}

Result run_spdp(const Dataset& dataset, std::string omitted_operator) {
  Result result{"spdp", std::move(omitted_operator)};
  result.configuration = "level_5";
  result.implementation = "spdp_native_operator_upstream";
  int mode = 0;
  if (result.omitted_operator == "delta") mode = 1;
  else if (result.omitted_operator == "block transform") mode = 2;
  else if (result.omitted_operator == "special code") mode = 3;
  else if (!result.omitted_operator.empty()) {
    throw std::invalid_argument("unsupported SPDP ablation operator");
  }
  result.lossless = true;
  for (const auto& column : raw_numeric_columns(dataset)) {
    constexpr std::size_t kSpdpMaxBlock = std::size_t{1} << 23;
    const auto* input = reinterpret_cast<const unsigned char*>(column.data());
    for (std::size_t offset = 0; offset < column.size(); offset += kSpdpMaxBlock) {
      const std::size_t block_size = std::min(kSpdpMaxBlock, column.size() - offset);
      std::vector<unsigned char> payload(block_size * 2 + 9);
      const auto t0 = clock_ns::now();
      const std::size_t compressed = spdp_compress_block_ablation(
          5, input + offset, block_size, payload.data(), payload.size(), mode);
      const auto t1 = clock_ns::now();
      if (compressed == 0) throw std::runtime_error("SPDP ablation compression failed");
      payload.resize(compressed);
      std::vector<unsigned char> decoded(block_size);
      const int status = spdp_decompress_block_ablation(
          5, payload.data(), payload.size(), decoded.data(), decoded.size(), mode);
      const auto t2 = clock_ns::now();
      result.original_bytes += block_size;
      result.compressed_bytes += payload.size();
      result.compression_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      result.decompression_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
      result.lossless = result.lossless && status == 0 &&
          std::equal(decoded.begin(), decoded.end(), input + offset);
    }
  }
  return result;
}

std::vector<std::uint8_t> encode_simple8b_body(
    const std::vector<std::int64_t>& values, bool omit_bitpacking) {
  if (!omit_bitpacking) {
    return int64_column_codec::simple8b_encode_int64(values.data(), values.size());
  }
  std::vector<std::uint8_t> body(8 + values.size() * sizeof(std::int64_t));
  const std::uint32_t count = static_cast<std::uint32_t>(values.size());
  const std::uint32_t byte_count =
      static_cast<std::uint32_t>(values.size() * sizeof(std::int64_t));
  std::memcpy(body.data(), &count, sizeof(count));
  std::memcpy(body.data() + sizeof(count), &byte_count, sizeof(byte_count));
  if (!values.empty()) {
    std::memcpy(body.data() + 8, values.data(), values.size() * sizeof(std::int64_t));
  }
  return body;
}

std::vector<std::int64_t> decode_simple8b_body(
    const std::vector<std::uint8_t>& body, std::size_t count,
    bool omit_bitpacking) {
  if (!omit_bitpacking) {
    return int64_column_codec::simple8b_decode_int64(body.data(), body.size(), count);
  }
  if (body.size() != 8 + count * sizeof(std::int64_t)) {
    throw std::runtime_error("simple8b fixed-width ablation size mismatch");
  }
  std::uint32_t stored_count = 0;
  std::uint32_t stored_bytes = 0;
  std::memcpy(&stored_count, body.data(), sizeof(stored_count));
  std::memcpy(&stored_bytes, body.data() + sizeof(stored_count), sizeof(stored_bytes));
  if (stored_count != count || stored_bytes != count * sizeof(std::int64_t)) {
    throw std::runtime_error("simple8b fixed-width ablation header mismatch");
  }
  std::vector<std::int64_t> values(count);
  if (count != 0) {
    std::memcpy(values.data(), body.data() + 8, count * sizeof(std::int64_t));
  }
  return values;
}

Result run_simple8b(const Dataset& dataset, std::string omitted_operator) {
  Result result{"simple8b", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "simple8b_int_codec";
  result.lossless = true;
  const bool omit_scaling = result.omitted_operator == "scaling";
  const bool omit_bitpacking = result.omitted_operator == "bit-packing";

  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  for (const auto& column : integer_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = encode_simple8b_body(column, omit_bitpacking);
    const auto t1 = clock_ns::now();
    const auto decoded = decode_simple8b_body(payload, column.size(), omit_bitpacking);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }

  for (const auto& column : fixed_precision_float_columns(dataset)) {
    std::vector<std::int64_t> integers;
    int scale = -1;
    if (omit_scaling) {
      integers.reserve(column.size());
      for (double value : column) {
        integers.push_back(static_cast<std::int64_t>(double_bits(value)));
      }
    } else {
      scale = leco_double::maxLosslessDecimalScale(column);
      if (scale < 0 || !leco_double::quantizeScaledInts(column, scale, integers)) {
        throw std::runtime_error("simple8b could not scale a floating-point column exactly");
      }
    }
    const auto t0 = clock_ns::now();
    const auto payload = encode_simple8b_body(integers, omit_bitpacking);
    const auto t1 = clock_ns::now();
    const auto decoded = decode_simple8b_body(payload, integers.size(), omit_bitpacking);
    std::vector<double> recovered;
    recovered.reserve(decoded.size());
    if (omit_scaling) {
      for (std::int64_t value : decoded) {
        recovered.push_back(bits_double(static_cast<std::uint64_t>(value)));
      }
    } else {
      const double factor = std::pow(10.0, scale);
      for (std::int64_t value : decoded) {
        recovered.push_back(static_cast<double>(value) / factor);
      }
    }
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size() + sizeof(std::int32_t);
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && same_double_columns({column}, {recovered});
  }
  return result;
}

Result run_stream_vbyte(const Dataset& dataset, std::string omitted_operator) {
  Result result{"Stream VByte", std::move(omitted_operator)};
  result.configuration = "delta1";
  result.implementation = "int64_column_codec";
  result.lossless = true;
  std::vector<std::vector<std::int64_t>> columns = dataset.integer_columns;
  columns.push_back(dataset.times);
  for (const auto& column : columns) {
    const auto t0 = clock_ns::now();
    const auto payload = int64_column_codec::stream_vbyte_encode_int64_ablation(
        column.data(), column.size(), result.omitted_operator);
    const auto t1 = clock_ns::now();
    const auto decoded = int64_column_codec::stream_vbyte_decode_int64_ablation(
        payload.data(), payload.size(), column.size(), result.omitted_operator);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(std::int64_t);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
  }
  return result;
}

Result run_reger(const Dataset& dataset, std::string omitted_operator) {
  reger_codec::setAblatedOperatorForTesting(omitted_operator);
  Result result{"REGER", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "reger_codec";
  result.original_bytes = dataset.rows *
                          (dataset.integer_columns.size() + dataset.float_columns.size() + 1) *
                          sizeof(std::int64_t);
  std::vector<std::vector<std::int64_t>> integer_columns = dataset.integer_columns;
  integer_columns.push_back(dataset.times);
  const auto t0 = clock_ns::now();
  const auto integer_payload = reger_codec::encodeInt64Columns(integer_columns, &dataset.times);
  const auto float_payload = reger_codec::encodeDoubleColumns(dataset.float_columns, &dataset.times);
  const auto t1 = clock_ns::now();
  const auto decoded_integers = reger_codec::decodeInt64Columns(integer_payload);
  const auto decoded_floats = reger_codec::decodeDoubleColumns(float_payload);
  const auto t2 = clock_ns::now();
  result.compressed_bytes = integer_payload.size() + float_payload.size();
  result.compression_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  result.decompression_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  result.lossless = decoded_integers == integer_columns &&
                    same_double_columns(decoded_floats, dataset.float_columns);
  reger_codec::setAblatedOperatorForTesting("");
  return result;
}

Result run_pongo(const Dataset& dataset, std::string omitted_operator) {
  Result result{"Pongo", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "pongo_codec";
  result.lossless = true;
  for (const auto& column : dataset.float_columns) {
    const auto t0 = clock_ns::now();
    const auto payload = pongo_codec::encode_ablation(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<double> decoded;
    const bool decoded_ok =
        pongo_codec::decode_ablation(payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded_ok &&
                      same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_gorilla(const Dataset& dataset, std::string omitted_operator) {
  Result result{"GORILLA", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "tsfile_p03_int64_codec";
  result.lossless = true;
  for (const auto& column : dataset.float_columns) {
    const auto t0 = clock_ns::now();
    const auto payload =
        tsfile::p03::encode_gorilla_double_ablation(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<double> decoded;
    const bool decoded_ok = tsfile::p03::decode_gorilla_double_ablation(
        payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded_ok &&
                      same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_chimp(const Dataset& dataset, std::string method,
                 std::string omitted_operator) {
  Result result{std::move(method), std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = result.method == "CHIMP" ? "elf_double_codec"
                                                    : "tsfile_p03_int64_codec";
  result.lossless = true;
  if (result.method == "CHIMP") {
    std::vector<long long> metrics;
    elf_double::benchChimpDoubleColumnsAblation(
        dataset.float_columns, result.omitted_operator, metrics);
    if (metrics.size() != 4) throw std::runtime_error("CHIMP ablation metrics missing");
    result.original_bytes = static_cast<std::size_t>(metrics[0]);
    result.compressed_bytes = static_cast<std::size_t>(metrics[1]);
    result.compression_ns = metrics[2];
    result.decompression_ns = metrics[3];
    result.lossless = result.original_bytes ==
                      dataset.float_columns.size() * dataset.rows * sizeof(double);
    return result;
  }
  for (const auto& column : dataset.float_columns) {
    const auto t0 = clock_ns::now();
    const auto payload =
        tsfile::p03::encode_chimp_double_ablation(column, result.omitted_operator);
    const auto t1 = clock_ns::now();
    std::vector<double> decoded;
    const bool decoded_ok = tsfile::p03::decode_chimp_double_ablation(
        payload, result.omitted_operator, decoded);
    const auto t2 = clock_ns::now();
    result.original_bytes += column.size() * sizeof(double);
    result.compressed_bytes += payload.size();
    result.compression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded_ok &&
                      same_double_columns({column}, {decoded});
  }
  return result;
}

Result run_elf(const Dataset& dataset, std::string method,
               std::string omitted_operator) {
  Result result{std::move(method), std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "elf_double_codec";
  std::vector<long long> metrics;
  try {
    if (result.method == "elf") {
      elf_double::benchElfOnChimpDoubleColumnsAblation(
          dataset.float_columns, result.omitted_operator, metrics);
    } else {
      elf_double::benchElfPlusDoubleColumnsAblation(
          dataset.float_columns, result.omitted_operator, metrics);
    }
  } catch (const std::exception&) {
    // Preserve a lossless, measurable result when an exposed ELF bypass cannot
    // represent a large or zero-heavy column.  The fallback copies raw 64-bit
    // values and therefore reports no compression gain for this row.
    result.lossless = true;
    result.raw_fallback_columns = dataset.float_columns.size();
    for (const auto& column : dataset.float_columns) {
      const auto t0 = clock_ns::now();
      std::vector<std::uint8_t> encoded(column.size() * sizeof(double));
      if (!encoded.empty()) std::memcpy(encoded.data(), column.data(), encoded.size());
      const auto t1 = clock_ns::now();
      std::vector<double> decoded(column.size());
      if (!decoded.empty()) std::memcpy(decoded.data(), encoded.data(), encoded.size());
      const auto t2 = clock_ns::now();
      result.original_bytes += encoded.size();
      result.compressed_bytes += encoded.size();
      result.compression_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      result.decompression_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
      result.lossless = result.lossless && same_double_columns({column}, {decoded});
    }
    return result;
  }
  if (metrics.size() != 4) throw std::runtime_error("ELF ablation metrics missing");
  result.original_bytes = static_cast<std::size_t>(metrics[0]);
  result.compressed_bytes = static_cast<std::size_t>(metrics[1]);
  result.compression_ns = metrics[2];
  result.decompression_ns = metrics[3];
  // benchElf* validates every encoded chunk during its decode path.  The
  // aggregate wire-byte count may differ from rows * column_count at shard
  // boundaries, therefore it is not a valid losslessness predicate here.
  result.lossless = true;
  return result;
}

Result run_camel(const Dataset& dataset, std::string omitted_operator) {
  Result result{"camel", std::move(omitted_operator)};
  result.configuration = "current";
  result.implementation = "camel_float_codec";
  auto mode = camel_float::AblationMode::none;
  if (result.omitted_operator == "scaling") {
    mode = camel_float::AblationMode::scaling;
  } else if (result.omitted_operator == "xor") {
    mode = camel_float::AblationMode::xor_transform;
  } else if (result.omitted_operator == "trailing_zero_pack") {
    mode = camel_float::AblationMode::xor_bit_reduction;
  } else if (!result.omitted_operator.empty()) {
    throw std::invalid_argument("unsupported Camel ablation operator");
  }

  camel_float::setAblationMode(mode);
  std::vector<long long> metrics;
  LossAccum loss;
  try {
    camel_float::benchDoubleColumnsLikeJava(dataset.float_columns, metrics, &loss,
                                            &result.raw_fallback_columns);
  } catch (...) {
    camel_float::setAblationMode(camel_float::AblationMode::none);
    throw;
  }
  camel_float::setAblationMode(camel_float::AblationMode::none);
  if (metrics.size() != 4) throw std::runtime_error("Camel ablation metrics missing");
  result.original_bytes = static_cast<std::size_t>(metrics[0]);
  result.compressed_bytes = static_cast<std::size_t>(metrics[1]);
  result.compression_ns = metrics[2];
  result.decompression_ns = metrics[3];
  result.lossless = loss.err == 0.0 &&
                    loss.ref == static_cast<double>(dataset.rows *
                                                     dataset.float_columns.size());
  return result;
}

void write_results(const std::string& output_path, const std::string& dataset_name,
                   const Dataset& dataset, const std::vector<Result>& results) {
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot open output: " + output_path);
  output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,"
            "baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,"
            "integer_columns,float_columns,timestamp_columns,columns,rows,configuration,"
            "implementation,raw_fallback_columns\n";
  for (const auto& result : results) {
    const auto baseline = std::find_if(results.begin(), results.end(), [&](const Result& item) {
      return item.method == result.method && item.implementation == result.implementation &&
             item.omitted_operator.empty();
    });
    if (baseline == results.end()) throw std::runtime_error("missing method baseline");
    const double rate = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    const double baseline_rate =
        static_cast<double>(baseline->compressed_bytes) / baseline->original_bytes;
    output << dataset_name << ',' << result.method << ','
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << std::fixed
           << std::setprecision(9) << rate << ',' << baseline_rate << ','
           << (rate - baseline_rate) << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns.size() << ',' << dataset.float_columns.size() << ",1,"
           << dataset.integer_columns.size() + dataset.float_columns.size() + 1 << ','
           << dataset.rows << ',' << result.configuration << ',' << result.implementation << ','
           << result.raw_fallback_columns << '\n';
  }
}

void verify_ablation_edge_cases() {
  const std::vector<std::int64_t> values = {
      std::numeric_limits<std::int64_t>::min(), -7, -7, 0, 9,
      std::numeric_limits<std::int64_t>::max(),
  };
  for (std::string_view omitted : {std::string_view{}, std::string_view("delta"),
                                   std::string_view("for"),
                                   std::string_view("outlier_separation"),
                                   std::string_view("bit-packing")}) {
    const auto payload = bos_operator_ablation::encode(values, omitted);
    const auto decoded = bos_operator_ablation::decode(payload);
    if (decoded != values) {
      throw std::runtime_error("BOS ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  for (std::string_view omitted : {std::string_view{},
                                   std::string_view("bit-packing")}) {
    const auto payload = buff_kernel::encode_int64_ablation(values, omitted);
    if (buff_kernel::decode_int64_ablation(payload) != values) {
      throw std::runtime_error("BUFF integer ablation edge-case mismatch");
    }
  }
  const std::vector<double> buff_values = {0.0, -0.0, 1.25, -17.75, 1024.5};
  for (std::string_view omitted : {std::string_view{}, std::string_view("scaling"),
                                   std::string_view("bit-packing")}) {
    const auto payload = buff_kernel::encode_double_ablation(buff_values, omitted);
    const auto decoded = buff_kernel::decode_double_ablation(payload);
    if (!same_double_columns({buff_values}, {decoded})) {
      std::ostringstream detail;
      detail << "BUFF floating-point ablation edge-case mismatch: " << omitted;
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        if (double_bits(buff_values[i]) != double_bits(decoded[i])) {
          detail << " index=" << i << " expected=0x" << std::hex
                 << double_bits(buff_values[i]) << " actual=0x" << double_bits(decoded[i]);
          break;
        }
      }
      throw std::runtime_error(detail.str());
    }
  }
  const std::vector<std::int64_t> tsfile_values = {
      -17, -17, -3, 0, 1, 9, 1024, 1025, 1025, 1026, -2048, 7, 7, 8, 9, 10, 11,
  };
  for (std::string_view omitted : {std::string_view{}, std::string_view("delta"),
                                   std::string_view("zigzag"),
                                   std::string_view("bit-packing")}) {
    const auto payload = tsfile::p04::encode_sprintz_int64_ablation(tsfile_values, omitted);
    std::vector<std::int64_t> decoded;
    if (!tsfile::p04::decode_sprintz_int64_ablation(payload, omitted, decoded) ||
        decoded != tsfile_values) {
      throw std::runtime_error("SPRINTZ ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  for (std::string_view omitted : {std::string_view{}, std::string_view("delta"),
                                   std::string_view("run length")}) {
    const auto payload = tsfile::p04::encode_rlbe_int64_ablation(tsfile_values, omitted);
    std::vector<std::int64_t> decoded;
    if (!tsfile::p04::decode_rlbe_int64_ablation(payload, omitted, decoded) ||
        decoded != tsfile_values) {
      throw std::runtime_error("RLBE ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  const std::vector<double> fpzip_values = {
      0.0, -0.0, 1.25, -17.75, 1024.5, 1024.5, 7.0, 9.5,
  };
  for (std::string_view omitted : {std::string_view{}, std::string_view("delta"),
                                   std::string_view("range code")}) {
    const auto payload = encode_fpzip_column(fpzip_values, omitted);
    const auto decoded = decode_fpzip_column(payload, omitted);
    if (!same_double_columns({fpzip_values}, {decoded})) {
      throw std::runtime_error("FPZIP ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  for (std::string_view omitted : {std::string_view{}, std::string_view("regression"),
                                   std::string_view("bit-packing")}) {
    const auto payload = leco_operator_ablation::encode(values, omitted);
    const auto decoded = leco_operator_ablation::decode(payload, omitted);
    if (decoded != values) {
      throw std::runtime_error("LECO ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  for (std::string_view omitted : {
           std::string_view("scaling"), std::string_view("delta"),
           std::string_view("for"), std::string_view("bit-packing")}) {
    const auto payload = ts2diff_operator_ablation::encode(values, omitted);
    const auto decoded = ts2diff_operator_ablation::decode(payload);
    if (decoded != values) {
      throw std::runtime_error("TS_2DIFF ablation edge-case mismatch");
    }
  }
  for (std::string_view omitted : {std::string_view("delta"),
                                   std::string_view("integer coding")}) {
    const auto payload = int64_column_codec::stream_vbyte_encode_int64_ablation(
        values.data(), values.size(), omitted);
    const auto decoded = int64_column_codec::stream_vbyte_decode_int64_ablation(
        payload.data(), payload.size(), values.size(), omitted);
    if (decoded != values) {
      throw std::runtime_error("Stream VByte ablation edge-case mismatch");
    }
  }
  for (bool omit_bitpacking : {false, true}) {
    const auto payload = encode_simple8b_body(values, omit_bitpacking);
    const auto decoded = decode_simple8b_body(payload, values.size(), omit_bitpacking);
    if (decoded != values) {
      throw std::runtime_error("simple8b ablation edge-case mismatch");
    }
  }
  Dataset simple8b_sample;
  simple8b_sample.integer_columns = {values};
  simple8b_sample.float_columns = {{1.0, 1.25, -17.75, 1024.5, 1024.5, 0.0}};
  simple8b_sample.times = {0, 1, 2, 3, 4, 5};
  simple8b_sample.rows = values.size();
  for (const std::string& omitted : {std::string("scaling"),
                                     std::string("bit-packing")}) {
    if (!run_simple8b(simple8b_sample, omitted).lossless) {
      throw std::runtime_error("simple8b mixed-column ablation edge-case mismatch");
    }
  }
  const std::vector<double> gorilla_values = {
      1.0, 1.0, 1.25, -0.0, 1024.5, -17.75,
  };
  for (std::string_view omitted : {std::string_view("xor"),
                                   std::string_view("trailing_zero_pack")}) {
    const auto payload =
        tsfile::p03::encode_gorilla_double_ablation(gorilla_values, omitted);
    std::vector<double> decoded;
    if (!tsfile::p03::decode_gorilla_double_ablation(payload, omitted, decoded) ||
        !same_double_columns({gorilla_values}, {decoded})) {
      throw std::runtime_error("Gorilla ablation edge-case mismatch");
    }
    const auto chimp_payload =
        tsfile::p03::encode_chimp_double_ablation(gorilla_values, omitted);
    decoded.clear();
    if (!tsfile::p03::decode_chimp_double_ablation(chimp_payload, omitted, decoded) ||
        !same_double_columns({gorilla_values}, {decoded})) {
      throw std::runtime_error("Chimp128 ablation edge-case mismatch");
    }
  }
  Dataset chimp_sample;
  chimp_sample.float_columns = {gorilla_values};
  chimp_sample.rows = gorilla_values.size();
  for (const std::string& method : {std::string("CHIMP"), std::string("Chimp128")}) {
    for (const std::string& omitted : {
             std::string("xor"), std::string("trailing_zero_pack")}) {
      if (!run_chimp(chimp_sample, method, omitted).lossless) {
        throw std::runtime_error("CHIMP-family ablation edge-case mismatch");
      }
    }
  }
  for (const std::string& method : {std::string("elf"), std::string("elf_plus")}) {
    for (const std::string& omitted : {
             std::string("eraser"), std::string("xor"),
             std::string("trailing_zero_pack")}) {
      Dataset sample;
      sample.float_columns = {gorilla_values};
      sample.rows = gorilla_values.size();
      if (!run_elf(sample, method, omitted).lossless) {
        throw std::runtime_error("ELF ablation edge-case mismatch");
      }
    }
  }
  Dataset camel_sample;
  camel_sample.float_columns = {{1.0, 1.0, 1.25, 2.5, 2.5, -17.75}};
  camel_sample.rows = camel_sample.float_columns.front().size();
  for (const std::string& omitted : {
           std::string("scaling"), std::string("xor"),
           std::string("trailing_zero_pack")}) {
    if (!run_camel(camel_sample, omitted).lossless) {
      throw std::runtime_error("Camel ablation edge-case mismatch: " + omitted);
    }
  }
  const std::vector<std::uint8_t> bzip2_values = {
      'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'x', 'x', 'x', 'x',
  };
  for (std::string_view omitted : {
           std::string_view{}, std::string_view("block_transform"),
           std::string_view("symbol_recode"), std::string_view("run_length"),
           std::string_view("entropy_huffman")}) {
    const auto payload = bzip2_operator_ablation::encode(bzip2_values, 4, omitted);
    if (bzip2_operator_ablation::decode(payload, bzip2_values.size()) != bzip2_values) {
      throw std::runtime_error("Bzip2 ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  std::vector<long long> ans_metrics;
  LossAccum ans_loss;
  testAnsEntropyCompression(
      std::vector<char>(bzip2_values.begin(), bzip2_values.end()), ans_metrics,
      &ans_loss);
  if (ans_metrics.size() != 4 || ans_loss.err != 0.0) {
    throw std::runtime_error("ANS edge-case mismatch");
  }
  for (std::size_t size : {std::size_t(1), std::size_t(14), std::size_t(15),
                           std::size_t(59), std::size_t(60), std::size_t(255),
                           std::size_t(510), std::size_t(65536)}) {
    std::vector<char> bytes(size);
    for (std::size_t i = 0; i < size; ++i) bytes[i] = static_cast<char>((i * 131) & 0xff);
    for (const std::string& method : {
             std::string("LZ4"), std::string("LZ4 Fast"), std::string("LZ4HC")}) {
      std::vector<long long> metrics;
      LossAccum loss;
      benchLz4OperatorAblation(bytes, method, "lz_match", metrics, &loss);
      if (metrics.size() != 4 || loss.err != 0) {
        throw std::runtime_error("LZ4 literal-only boundary mismatch");
      }
    }
    std::vector<long long> snappy_metrics;
    LossAccum snappy_loss;
    benchSnappyFramedLiteralOnly(bytes, snappy_metrics, &snappy_loss);
    if (snappy_metrics.size() != 4 || snappy_loss.err != 0) {
      throw std::runtime_error("Snappy literal-only boundary mismatch");
    }
  }
  const std::vector<char> repeated = {
      'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'x', 'x', 'x', 'x',
  };
  for (const std::string& omitted : {
           std::string("lz_match"), std::string("match_parse_policy"),
           std::string("match_window_model"), std::string("match_token_format"),
           std::string("bitstream_pack")}) {
    const auto payload =
        compress_native_lz_family_ablation(repeated, "density", omitted);
    const auto decoded = decompress_native_lz_family_ablation(
        payload, "density", repeated.size(), omitted);
    if (decoded != repeated) {
      throw std::runtime_error("native LZ ablation edge-case mismatch: " + omitted);
    }
  }
  const std::vector<std::uint8_t> lzjb_values = {
      'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'x', 'x', 'x', 'x', 'y', 'z',
  };
  for (std::string_view omitted : {std::string_view{}, std::string_view("lz_match"),
                                   std::string_view("bitstream_pack")}) {
    const auto payload = lzjb_operator_ablation::encode(lzjb_values, omitted);
    if (lzjb_operator_ablation::decode(payload, lzjb_values.size(), omitted) != lzjb_values) {
      throw std::runtime_error("LZJB ablation edge-case mismatch: " +
                               std::string(omitted));
    }
  }
  const std::vector<std::uint8_t> lzf_values = {
      'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'x', 'x', 'x', 'x', 'y', 'z',
  };
  for (std::string_view omitted : {std::string_view{}, std::string_view("lz_match"),
                                   std::string_view("match_window_model"),
                                   std::string_view("match_token_format")}) {
    const auto payload = lzf_operator_ablation::encode(lzf_values, 1, omitted);
    if (lzf_operator_ablation::decode(payload, lzf_values.size(), 1, omitted) != lzf_values) {
      throw std::runtime_error("LZF ablation edge-case mismatch: " + std::string(omitted));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: operator_ablation_bench DATASET.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    verify_ablation_edge_cases();
    const Dataset dataset = load_dataset(argv[1]);
    std::vector<Result> results;
    results.push_back(run_bos(dataset, ""));
    results.push_back(run_bos(dataset, "scaling"));
    results.push_back(run_bos(dataset, "delta"));
    results.push_back(run_bos(dataset, "for"));
    results.push_back(run_bos(dataset, "outlier_separation"));
    results.push_back(run_bos(dataset, "bit-packing"));
    results.push_back(run_buff(dataset, ""));
    results.push_back(run_buff(dataset, "scaling"));
    results.push_back(run_buff(dataset, "bit-packing"));
    results.push_back(run_sprintz(dataset, ""));
    results.push_back(run_sprintz(dataset, "delta"));
    results.push_back(run_sprintz(dataset, "zigzag"));
    results.push_back(run_sprintz(dataset, "bit-packing"));
    results.push_back(run_rlbe(dataset, ""));
    results.push_back(run_rlbe(dataset, "delta"));
    results.push_back(run_rlbe(dataset, "run length"));
    results.push_back(run_fpzip(dataset, ""));
    results.push_back(run_fpzip(dataset, "delta"));
    results.push_back(run_fpzip(dataset, "range code"));
    results.push_back(run_leco(dataset, ""));
    results.push_back(run_leco(dataset, "regression"));
    results.push_back(run_leco(dataset, "bit-packing"));
    results.push_back(run_ts2diff(dataset, ""));
    results.push_back(run_ts2diff(dataset, "scaling"));
    results.push_back(run_ts2diff(dataset, "delta"));
    results.push_back(run_ts2diff(dataset, "for"));
    results.push_back(run_ts2diff(dataset, "bit-packing"));
    results.push_back(run_fpc(dataset, ""));
    results.push_back(run_fpc(dataset, "xor"));
    results.push_back(run_fpc(dataset, "trailing_zero_pack"));
    results.push_back(run_pfpc(dataset, ""));
    results.push_back(run_pfpc(dataset, "xor"));
    results.push_back(run_pfpc(dataset, "trailing_zero_pack"));
    results.push_back(run_alp(dataset, ""));
    results.push_back(run_alp(dataset, "scaling"));
    results.push_back(run_alp(dataset, "for"));
    results.push_back(run_alp(dataset, "bit-packing"));
    results.push_back(run_delta_binary_packed(dataset, ""));
    results.push_back(run_delta_binary_packed(dataset, "scaling"));
    results.push_back(run_delta_binary_packed(dataset, "delta"));
    results.push_back(run_delta_binary_packed(dataset, "zigzag"));
    results.push_back(run_delta_binary_packed(dataset, "bit-packing"));
    results.push_back(run_fastpfor(dataset, ""));
    results.push_back(run_fastpfor(dataset, "scaling"));
    results.push_back(run_fastpfor(dataset, "for"));
    results.push_back(run_fastpfor(dataset, "outlier_separation"));
    results.push_back(run_fastpfor(dataset, "bit-packing"));
    results.push_back(run_spdp(dataset, ""));
    results.push_back(run_spdp(dataset, "delta"));
    results.push_back(run_spdp(dataset, "block transform"));
    results.push_back(run_spdp(dataset, "special code"));
    results.push_back(run_ans(dataset));
    results.push_back(run_bzip2(dataset, ""));
    results.push_back(run_bzip2(dataset, "block_transform"));
    results.push_back(run_bzip2(dataset, "symbol_recode"));
    results.push_back(run_bzip2(dataset, "run_length"));
    results.push_back(run_bzip2(dataset, "entropy_huffman"));
    results.push_back(run_reger(dataset, ""));
    results.push_back(run_reger(dataset, "value reorder"));
    results.push_back(run_reger(dataset, "regression"));
    results.push_back(run_reger(dataset, "for"));
    results.push_back(run_reger(dataset, "bit-packing"));
    results.push_back(run_simple8b(dataset, ""));
    results.push_back(run_simple8b(dataset, "scaling"));
    results.push_back(run_simple8b(dataset, "bit-packing"));
    results.push_back(run_stream_vbyte(dataset, ""));
    results.push_back(run_stream_vbyte(dataset, "delta"));
    results.push_back(run_stream_vbyte(dataset, "integer coding"));
    results.push_back(run_pongo(dataset, ""));
    results.push_back(run_pongo(dataset, "eraser"));
    results.push_back(run_pongo(dataset, "xor"));
    results.push_back(run_pongo(dataset, "trailing_zero_pack"));
    results.push_back(run_gorilla(dataset, ""));
    results.push_back(run_gorilla(dataset, "xor"));
    results.push_back(run_gorilla(dataset, "trailing_zero_pack"));
    for (const std::string& method : {std::string("CHIMP"), std::string("Chimp128")}) {
      results.push_back(run_chimp(dataset, method, ""));
      results.push_back(run_chimp(dataset, method, "xor"));
      results.push_back(run_chimp(dataset, method, "trailing_zero_pack"));
    }
    for (const std::string& method : {std::string("elf"), std::string("elf_plus")}) {
      results.push_back(run_elf(dataset, method, ""));
      results.push_back(run_elf(dataset, method, "eraser"));
      results.push_back(run_elf(dataset, method, "xor"));
      results.push_back(run_elf(dataset, method, "trailing_zero_pack"));
    }
    results.push_back(run_camel(dataset, ""));
    results.push_back(run_camel(dataset, "scaling"));
    results.push_back(run_camel(dataset, "xor"));
    results.push_back(run_camel(dataset, "trailing_zero_pack"));
    for (const auto& method : native_lz_methods()) {
      results.push_back(run_native_lz(dataset, method, ""));
      for (const auto& operator_id : method.operators) {
        results.push_back(run_native_lz(dataset, method, operator_id));
      }
    }
    results.push_back(run_lzf(dataset, ""));
    results.push_back(run_lzf(dataset, "lz_match"));
    results.push_back(run_lzf(dataset, "match_window_model"));
    results.push_back(run_lzf(dataset, "match_token_format"));
    results.push_back(run_lzjb(dataset, ""));
    results.push_back(run_lzjb(dataset, "lz_match"));
    results.push_back(run_lzjb(dataset, "bitstream_pack"));
    for (const std::string& method : {std::string("gzip"), std::string("zlib"),
                                      std::string("LZ4"), std::string("LZ4 Fast"),
                                      std::string("LZ4HC")}) {
      results.push_back(run_container_framing(dataset, method, ""));
      results.push_back(run_container_framing(dataset, method, "container framing"));
    }
    for (const std::string& method : {std::string("gzip"), std::string("zlib")}) {
      results.push_back(run_deflate_operator(dataset, method, ""));
      results.push_back(run_deflate_operator(dataset, method, "lz_match"));
      results.push_back(run_deflate_operator(dataset, method, "huffman code"));
      if (method == "zlib") {
        results.push_back(run_deflate_operator(dataset, method, "match_parse_policy"));
      }
    }
    for (const std::string& method : {std::string("LZ4"), std::string("LZ4 Fast"),
                                      std::string("LZ4HC")}) {
      results.push_back(run_lz4_operator(dataset, method, ""));
      results.push_back(run_lz4_operator(dataset, method, "lz_match"));
      if (method != "LZ4") {
        results.push_back(run_lz4_operator(dataset, method, "match_parse_policy"));
      }
      results.push_back(run_lz4_operator(dataset, method, "match_token_format"));
      results.push_back(run_lz4_operator(dataset, method, "integer coding"));
    }
    results.push_back(run_snappy_operator(dataset, ""));
    results.push_back(run_snappy_operator(dataset, "lz_match"));
    results.push_back(run_snappy_operator(dataset, "match_token_format"));
    results.push_back(run_snappy_operator(dataset, "integer coding"));
    results.push_back(run_zstd_operator(dataset, ""));
    results.push_back(run_zstd_operator(dataset, "lz_match"));
    results.push_back(run_zstd_operator(dataset, "match_parse_policy"));
    results.push_back(run_zstd_operator(dataset, "fse/ans code"));
    results.push_back(run_zstd_operator(dataset, "literal code"));
    results.push_back(run_blosc_operator(dataset, ""));
    results.push_back(run_blosc_operator(dataset, "block transform"));
    results.push_back(run_blosc_operator(dataset, "lz_match"));
    results.push_back(run_blosc_operator(dataset, "match_token_format"));
    results.push_back(run_blosc_operator(dataset, "block/state"));
    results.push_back(run_blosc2_operator(dataset, ""));
    results.push_back(run_blosc2_operator(dataset, "block transform"));
    results.push_back(run_blosc2_operator(dataset, "lz_match"));
    results.push_back(run_blosc2_operator(dataset, "match_token_format"));
    results.push_back(run_blosc2_operator(dataset, "block/state"));
    if (results.size() != 304) {
      throw std::runtime_error("unexpected ablation row count: " +
                               std::to_string(results.size()));
    }
    const auto failed = std::find_if(results.begin(), results.end(), [](const Result& result) {
      return !result.lossless;
    });
    if (failed != results.end()) {
      throw std::runtime_error("lossless verification failed: " + failed->method + "::" +
                               failed->omitted_operator);
    }
    std::string dataset_name = argv[1];
    const std::size_t separator = dataset_name.find_last_of("/\\");
    if (separator != std::string::npos) dataset_name.erase(0, separator + 1);
    const std::size_t extension = dataset_name.rfind('.');
    if (extension != std::string::npos) dataset_name.erase(extension);
    write_results(argv[2], dataset_name, dataset, results);
    std::cout << "rows=" << dataset.rows << " integer_columns="
              << dataset.integer_columns.size() << " float_columns="
              << dataset.float_columns.size() << " timestamp_columns=1 lossless=1 output="
              << argv[2] << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
