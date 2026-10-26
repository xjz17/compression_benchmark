// Air-Quality operator ablations for the original Pithy, Shrinker, and XPACK
// sources restored from lzbench history. Each result uses the paired upstream
// decoder and checks every reconstructed column byte for byte.

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
#include "lz/pithy/pithy.h"
#include "lz/shrinker/shrinker.h"
#include "lz/xpack/lib/libxpack.h"
#if defined(TSCB_DENSITY_C99)
#include "density_api.h"
#include "algorithms/lion/lion.h"
#endif
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
  std::string method;
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

Bytes encode_pithy(const Bytes& input, int mode) {
  pithy_SetAblationMode(mode);
  Bytes encoded(pithy_MaxCompressedLength(input.size()) + 64);
  const std::size_t size = pithy_Compress(
      reinterpret_cast<const char*>(input.data()), input.size(),
      reinterpret_cast<char*>(encoded.data()), encoded.size(), 9);
  if (size == 0) throw std::runtime_error("Pithy encoder failed");
  encoded.resize(size);
  return encoded;
}

Bytes decode_pithy(const Bytes& encoded, std::size_t expected_size) {
  Bytes decoded(expected_size);
  if (!pithy_Decompress(reinterpret_cast<const char*>(encoded.data()), encoded.size(),
                        reinterpret_cast<char*>(decoded.data()), decoded.size())) {
    throw std::runtime_error("Pithy decoder failed");
  }
  return decoded;
}

Bytes encode_shrinker(const Bytes& input, int mode) {
  shrinker_set_ablation_mode(mode);
  Bytes encoded(input.size() * 2 + 4096);
  const int size = shrinker_compress_with_capacity(
      const_cast<std::uint8_t*>(input.data()), encoded.data(), encoded.size(),
      static_cast<int>(input.size()));
  if (size <= 0) throw std::runtime_error("Shrinker encoder failed");
  encoded.resize(static_cast<std::size_t>(size));
  return encoded;
}

Bytes decode_shrinker(const Bytes& encoded, std::size_t expected_size) {
  Bytes decoded(expected_size);
  const int size = shrinker_decompress(
      const_cast<std::uint8_t*>(encoded.data()), decoded.data(), static_cast<int>(expected_size));
  if (size != static_cast<int>(expected_size)) {
    throw std::runtime_error("Shrinker decoder failed");
  }
  return decoded;
}

Bytes encode_xpack(const Bytes& input, int mode) {
  xpack_compressor* compressor = xpack_alloc_compressor(input.size(), 9);
  if (compressor == nullptr) throw std::runtime_error("XPACK compressor allocation failed");
  xpack_set_ablation_mode(compressor, mode);
  Bytes encoded(input.size() * 2 + 65536);
  const std::size_t size = xpack_compress(
      compressor, input.data(), input.size(), encoded.data(), encoded.size());
  xpack_free_compressor(compressor);
  if (size == 0) throw std::runtime_error("XPACK encoder failed");
  encoded.resize(size);
  return encoded;
}

Bytes decode_xpack(const Bytes& encoded, std::size_t expected_size) {
  xpack_decompressor* decompressor = xpack_alloc_decompressor();
  if (decompressor == nullptr) throw std::runtime_error("XPACK decompressor allocation failed");
  Bytes decoded(expected_size);
  const auto status = xpack_decompress(
      decompressor, encoded.data(), encoded.size(), decoded.data(), decoded.size(), nullptr);
  xpack_free_decompressor(decompressor);
  if (status != DECOMPRESS_SUCCESS) throw std::runtime_error("XPACK decoder failed");
  return decoded;
}

#if defined(TSCB_DENSITY_C99)
Bytes encode_density(const Bytes& input, int mode) {
  density_lion_set_ablation_mode(mode);
  Bytes encoded(density_compress_safe_size(input.size()));
  const density_processing_result result = density_compress(
      input.data(), input.size(), encoded.data(), encoded.size(), DENSITY_ALGORITHM_LION);
  if (result.state != DENSITY_STATE_OK || result.bytesRead != input.size()) {
    throw std::runtime_error("Density encoder failed");
  }
  encoded.resize(result.bytesWritten);
  return encoded;
}

Bytes decode_density(const Bytes& encoded, std::size_t expected_size) {
  Bytes decoded(density_decompress_safe_size(expected_size));
  const density_processing_result result = density_decompress(
      encoded.data(), encoded.size(), decoded.data(), decoded.size());
  if (result.state != DENSITY_STATE_OK || result.bytesWritten != expected_size) {
    throw std::runtime_error(
        "Density decoder failed state=" + std::to_string(result.state)
        + " read=" + std::to_string(result.bytesRead)
        + " encoded=" + std::to_string(encoded.size())
        + " written=" + std::to_string(result.bytesWritten)
        + " expected=" + std::to_string(expected_size));
  }
  decoded.resize(expected_size);
  return decoded;
}
#endif

using Encoder = Bytes (*)(const Bytes&, int);
using Decoder = Bytes (*)(const Bytes&, std::size_t);

Result run(const Dataset& dataset, const std::string& method,
           const std::string& omitted_operator, int mode,
           Encoder encode, Decoder decode) {
  Result result{method, omitted_operator};
  std::cerr << "running method=" << method << " omitted="
            << (omitted_operator.empty() ? "none" : omitted_operator) << '\n';
  std::size_t column_index = 0;
  for (const auto& column : dataset.columns) {
    std::cerr << "  column=" << column_index << " bytes=" << column.size() << '\n';
    const auto t0 = Clock::now();
    const Bytes encoded = encode(column, mode);
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
    ++column_index;
  }
  return result;
}

void write_results(const std::string& path, const Dataset& dataset,
                   const std::vector<Result>& results) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write output: " + path);
  output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
  output << std::setprecision(9);
  for (const auto& result : results) {
    const auto baseline_it = std::find_if(results.begin(), results.end(), [&](const Result& row) {
      return row.method == result.method && row.omitted_operator.empty();
    });
    if (baseline_it == results.end()) throw std::runtime_error("missing baseline");
    const double rate = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    const double baseline =
        static_cast<double>(baseline_it->compressed_bytes) / baseline_it->original_bytes;
    output << "Air-Quality," << result.method << ','
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << baseline << ',' << rate - baseline << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows;
    if (result.method == "Density") {
      output << ",source_c99_v0.14.2,density_c99_method_specific_operator_ablation\n";
    } else {
      output << ",upstream_level_9,lzbench_legacy_upstream_source_operator_ablation\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr << "usage: legacy_lz_source_air_bench DATASET OUTPUT [Density]\n";
      return 2;
    }
    const Dataset dataset = load_dataset(argv[1]);
    const bool density_only = argc == 4 && std::string_view(argv[3]) == "Density";
    std::vector<Result> results;
    if (!density_only) {
      results = {
          run(dataset, "Pithy", "", PITHY_ABLATION_NONE, encode_pithy, decode_pithy),
          run(dataset, "Pithy", "lz_match", PITHY_ABLATION_NO_LZ_MATCH,
              encode_pithy, decode_pithy),
          run(dataset, "Pithy", "match_parse_policy", PITHY_ABLATION_FIXED_GREEDY_SCAN,
              encode_pithy, decode_pithy),
          run(dataset, "Pithy", "match_token_format", PITHY_ABLATION_FIXED_TOKENS,
              encode_pithy, decode_pithy),
          run(dataset, "Shrinker", "", SHRINKER_ABLATION_NONE,
              encode_shrinker, decode_shrinker),
          run(dataset, "Shrinker", "lz_match", SHRINKER_ABLATION_NO_LZ_MATCH,
              encode_shrinker, decode_shrinker),
          run(dataset, "Shrinker", "match_token_format", SHRINKER_ABLATION_FIXED_TOKENS,
              encode_shrinker, decode_shrinker),
          run(dataset, "XPACK", "", XPACK_ABLATION_NONE, encode_xpack, decode_xpack),
          run(dataset, "XPACK", "lz_match", XPACK_ABLATION_NO_LZ_MATCH,
              encode_xpack, decode_xpack),
          run(dataset, "XPACK", "match_parse_policy", XPACK_ABLATION_GREEDY_PARSE,
              encode_xpack, decode_xpack),
          run(dataset, "XPACK", "match_token_format", XPACK_ABLATION_FIXED_TOKENS,
              encode_xpack, decode_xpack),
      };
    }
#if defined(TSCB_DENSITY_C99)
    if (density_only) {
      results = {
          run(dataset, "Density", "", 0, encode_density, decode_density),
          run(dataset, "Density", "dictionary", 1, encode_density, decode_density),
          run(dataset, "Density", "block_state_control", 2,
              encode_density, decode_density),
      };
    }
#else
    if (density_only) throw std::runtime_error("Density C99 support is not compiled");
#endif
    write_results(argv[2], dataset, results);
    for (const auto& result : results) {
      std::cout << result.method << ' '
                << (result.omitted_operator.empty() ? "baseline" : result.omitted_operator)
                << " lossless=" << result.lossless
                << " compressed=" << result.compressed_bytes << '\n';
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
