// Source-level Air-Quality ablation runner for lzbench codecs whose upstream
// C/C++ wrapper exposes paired encoder and decoder functions.

#include "bench/codecs.h"
#include "lz/fast-lzma2/fast-lzma2.h"
#include "lz/gipfeli/lz77.h"
#include "lz/libzling/libzling.h"
#include "lz/libzling/libzling_lz.h"
#include "lz/lzlib/lzlib.h"
#include "lz/tornado/tor_test.h"

#if !defined(__APPLE__)
extern "C" void lzham_set_ablation_mode(int mode);
#endif
extern "C" void lzma_tscb_set_ablation_mode(int mode);

#if !defined(__APPLE__)
extern "C" void lzbench_csc_set_ablation_mode(int mode);
#endif
extern "C" void lzbench_ppmd_set_ablation_mode(int mode);
extern "C" void BZ2_setAblationMode(int mode);
extern "C" void GLZA_set_ablation_mode(int mode);

#include <algorithm>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::high_resolution_clock;

struct Dataset {
  std::vector<Bytes> columns;
  std::size_t integer_columns = 0;
  std::size_t float_columns = 0;
  std::size_t timestamp_columns = 0;
  std::size_t rows = 0;
};

struct Codec {
  const char* method;
  int level;
  int additional_param;
  int64_t (*compress)(char*, size_t, char*, size_t, codec_options_t*);
  int64_t (*decompress)(char*, size_t, char*, size_t, codec_options_t*);
  std::vector<std::string> operators;
};

struct Result {
  std::string method;
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
Bytes native_bytes(const std::vector<T>& values) {
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
      dataset.columns.push_back(native_bytes(integers));
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
    dataset.columns.push_back(native_bytes(doubles));
    ++dataset.float_columns;
  }
  return dataset;
}

void put_u16(Bytes& output, std::size_t value) {
  if (value > 65535) throw std::runtime_error("replacement token exceeds uint16");
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8));
}

std::size_t get_u16(const Bytes& input, std::size_t& position) {
  if (position + 2 > input.size()) throw std::runtime_error("replacement token truncated");
  const std::size_t value = input[position] | (static_cast<std::size_t>(input[position + 1]) << 8);
  position += 2;
  return value;
}

std::uint32_t key_at(const Bytes& input, std::size_t position) {
  if (position + 3 > input.size()) return 0xffffffffu;
  return (static_cast<std::uint32_t>(input[position]) << 16) |
         (static_cast<std::uint32_t>(input[position + 1]) << 8) |
         input[position + 2];
}

Bytes replacement_stream(const Bytes& input, const std::string& magic,
                         bool allow_matches, std::size_t window) {
  Bytes output(magic.begin(), magic.end());
  std::unordered_map<std::uint32_t, std::size_t> latest;
  std::size_t position = 0;
  while (position < input.size()) {
    const std::size_t control_position = output.size();
    output.push_back(0);
    std::uint8_t control = 0;
    for (int token = 0; token < 8 && position < input.size(); ++token) {
      std::size_t match_length = 0;
      std::size_t offset = 0;
      const auto key = key_at(input, position);
      if (allow_matches && key != 0xffffffffu) {
        const auto it = latest.find(key);
        if (it != latest.end() && position > it->second && position - it->second <= window) {
          offset = position - it->second;
          const std::size_t limit = std::min<std::size_t>(65535,
              std::min(input.size() - position, input.size() - it->second));
          while (match_length < limit && input[it->second + match_length] == input[position + match_length]) {
            ++match_length;
          }
          if (match_length < 3) match_length = 0;
        }
      }
      if (match_length >= 3) {
        control |= static_cast<std::uint8_t>(1u << token);
        put_u16(output, offset);
        put_u16(output, match_length);
        for (std::size_t i = 0; i < match_length; ++i) {
          const auto match_key = key_at(input, position + i);
          if (match_key != 0xffffffffu) latest[match_key] = position + i;
        }
        position += match_length;
      } else {
        output.push_back(input[position]);
        if (key != 0xffffffffu) latest[key] = position;
        ++position;
      }
    }
    output[control_position] = control;
  }
  return output;
}

Bytes replacement_literal(const Bytes& input, const std::string& magic) {
  Bytes output(magic.begin(), magic.end());
  output.reserve(output.size() + input.size() + input.size() / 8 + 8);
  std::size_t position = 0;
  while (position < input.size()) {
    const std::size_t count = std::min<std::size_t>(128, input.size() - position);
    output.push_back(static_cast<std::uint8_t>(count - 1));
    output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(position),
                  input.begin() + static_cast<std::ptrdiff_t>(position + count));
    position += count;
  }
  return output;
}

Bytes replacement_decode(const Bytes& encoded, std::size_t expected_size,
                         const std::string& magic) {
  if (encoded.size() < magic.size() ||
      !std::equal(magic.begin(), magic.end(), encoded.begin())) {
    throw std::runtime_error("replacement magic mismatch");
  }
  std::size_t position = magic.size();
  Bytes output;
  output.reserve(expected_size);
  while (position < encoded.size()) {
    const std::uint8_t control = encoded[position++];
    for (int token = 0; token < 8; ++token) {
      if (position >= encoded.size()) break;
      if (control & static_cast<std::uint8_t>(1u << token)) {
        const std::size_t offset = get_u16(encoded, position);
        const std::size_t length = get_u16(encoded, position);
        if (offset == 0 || offset > output.size()) throw std::runtime_error("bad match offset");
        for (std::size_t i = 0; i < length; ++i) output.push_back(output[output.size() - offset]);
      } else {
        output.push_back(encoded[position++]);
      }
      if (output.size() > expected_size) throw std::runtime_error("replacement output overflow");
    }
  }
  if (output.size() != expected_size) throw std::runtime_error("replacement size mismatch");
  return output;
}

std::vector<Codec> codecs() {
  return {
      {"Bzip2", 9, 0, lzbench_bzip2_compress, lzbench_bzip2_decompress,
       {"block_transform", "symbol_recode", "run_length", "entropy_huffman"}},
#if !defined(__APPLE__)
      {"CSC", 5, 0, lzbench_csc_compress, lzbench_csc_decompress,
       {"delta", "dictionary", "lz_match", "match_parse_policy", "entropy_range"}},
#endif
      {"Gipfeli", 0, 0, lzbench_gipfeli_compress, lzbench_gipfeli_decompress,
       {"lz_match", "match_parse_policy", "match_token_format", "literal_entropy"}},
      {"GLZA", 0, 0, lzbench_glza_compress, lzbench_glza_decompress,
       {"dictionary", "entropy_range"}},
      {"FastLZMA2", 10, 0, lzbench_fastlzma2_compress, lzbench_fastlzma2_decompress,
       {"block_state_control", "lz_match", "entropy_range"}},
      {"PPMD8", 9, 0, lzbench_ppmd_compress, lzbench_ppmd_decompress,
       {"entropy_range"}},
      {"Lzlib", 9, 0, lzbench_lzlib_compress, lzbench_lzlib_decompress,
       {"lz_match", "entropy_range", "container_frame"}},
      {"XZ", 6, 0, lzbench_xz_compress, lzbench_xz_decompress,
       {"lz_match", "match_parse_policy", "entropy_range"}},
#if !defined(__APPLE__)
      {"LZHAM", 4, 26, lzbench_lzham_compress, lzbench_lzham_decompress,
       {"lz_match", "match_parse_policy", "entropy_huffman"}},
#endif
      {"Tornado", 16, 0, lzbench_tornado_compress, lzbench_tornado_decompress,
       {"lz_match", "match_parse_policy", "entropy_range", "match_token_format"}},
      {"Zling", 4, 0, lzbench_zling_compress, lzbench_zling_decompress,
       {"block_state_control", "lz_match", "entropy_special"}},
  };
}

std::string magic_for(const Codec& codec, std::string_view omitted) {
  std::string magic = "AB";
  for (char c : std::string_view(codec.method)) magic.push_back(c);
  for (char c : omitted) {
    if (std::isalnum(static_cast<unsigned char>(c))) magic.push_back(c);
  }
  return magic.substr(0, 12);
}

Result run_codec(const Dataset& dataset, const Codec& codec, std::string omitted) {
  Result result{codec.method, std::move(omitted)};
  std::cerr << "running method=" << result.method << " omitted="
            << (result.omitted_operator.empty() ? "none" : result.omitted_operator)
            << '\n';
  codec_options_t options{};
  options.level = codec.level;
  options.additional_param = codec.additional_param;
  options.threads = 1;
  int gipfeli_mode = util::compression::gipfeli::LZ77_ABLATION_NONE;
  if (result.omitted_operator == "lz_match") {
    gipfeli_mode = util::compression::gipfeli::LZ77_ABLATION_NO_MATCH;
  } else if (result.omitted_operator == "match_parse_policy") {
    gipfeli_mode = util::compression::gipfeli::LZ77_ABLATION_FIXED_SCAN;
  } else if (result.omitted_operator == "match_token_format") {
    gipfeli_mode = util::compression::gipfeli::LZ77_ABLATION_RAW_COMMANDS;
  } else if (result.omitted_operator == "literal_entropy") {
    gipfeli_mode = util::compression::gipfeli::LZ77_ABLATION_NO_LITERAL_ENTROPY;
  }
  util::compression::gipfeli::SetLZ77AblationMode(gipfeli_mode);
  if (codec.method == std::string("Bzip2")) {
    int bzip2_mode = 0;
    if (result.omitted_operator == "block_transform") bzip2_mode = 1;
    else if (result.omitted_operator == "run_length") bzip2_mode = 2;
    else if (result.omitted_operator == "symbol_recode") bzip2_mode = 3;
    else if (result.omitted_operator == "entropy_huffman") bzip2_mode = 4;
    BZ2_setAblationMode(bzip2_mode);
  }
  if (codec.method == std::string("GLZA")) {
    int glza_mode = 0;
    if (result.omitted_operator == "dictionary") glza_mode = 1;
    else if (result.omitted_operator == "entropy_range") glza_mode = 2;
    GLZA_set_ablation_mode(glza_mode);
  }
  if (codec.method == std::string("FastLZMA2")) {
    int fastlzma2_mode = 0;
    if (result.omitted_operator == "lz_match") fastlzma2_mode = 1;
    else if (result.omitted_operator == "block_state_control") fastlzma2_mode = 2;
    else if (result.omitted_operator == "entropy_range") fastlzma2_mode = 3;
    FL2_setAblationMode(fastlzma2_mode);
  }
#if !defined(__APPLE__)
  if (codec.method == std::string("CSC")) {
    int csc_mode = 0;
    if (result.omitted_operator == "delta") csc_mode = 1;
    else if (result.omitted_operator == "dictionary") csc_mode = 2;
    else if (result.omitted_operator == "lz_match") csc_mode = 3;
    else if (result.omitted_operator == "match_parse_policy") csc_mode = 4;
    else if (result.omitted_operator == "entropy_range") csc_mode = 5;
    lzbench_csc_set_ablation_mode(csc_mode);
  }
#endif
  if (codec.method == std::string("PPMD8")) {
    lzbench_ppmd_set_ablation_mode(result.omitted_operator == "entropy_range");
  }
  if (codec.method == std::string("Lzlib")) {
    int lzlib_mode = 0;
    if (result.omitted_operator == "lz_match") lzlib_mode = 1;
    else if (result.omitted_operator == "container_frame") lzlib_mode = 2;
    else if (result.omitted_operator == "entropy_range") lzlib_mode = 3;
    LZ_set_ablation_mode(lzlib_mode);
  }
  if (codec.method == std::string("XZ")) {
    int xz_mode = 0;
    if (result.omitted_operator == "lz_match") xz_mode = 1;
    else if (result.omitted_operator == "match_parse_policy") xz_mode = 2;
    else if (result.omitted_operator == "entropy_range") xz_mode = 3;
    lzma_tscb_set_ablation_mode(xz_mode);
  }
#if !defined(__APPLE__)
  if (codec.method == std::string("LZHAM")) {
    int lzham_mode = 0;
    if (result.omitted_operator == "lz_match") lzham_mode = 1;
    else if (result.omitted_operator == "match_parse_policy") lzham_mode = 2;
    else if (result.omitted_operator == "entropy_huffman") lzham_mode = 3;
    lzham_set_ablation_mode(lzham_mode);
  }
#endif
  if (codec.method == std::string("Tornado")) {
    int tornado_mode = 0;
    if (result.omitted_operator == "lz_match") tornado_mode = 1;
    else if (result.omitted_operator == "match_parse_policy") tornado_mode = 2;
    else if (result.omitted_operator == "entropy_range") tornado_mode = 3;
    else if (result.omitted_operator == "match_token_format") tornado_mode = 4;
    tor_set_ablation_mode(tornado_mode);
  }
  if (codec.method == std::string("Zling")) {
    baidu::zling::lz::SetZlingLZAblationMode(
        result.omitted_operator == "lz_match"
            ? baidu::zling::lz::kZlingLZAblationNoMatch
            : baidu::zling::lz::kZlingLZAblationNone);
    baidu::zling::SetZlingEntropyAblationMode(
        result.omitted_operator == "entropy_special");
    baidu::zling::SetZlingBlockStateAblationMode(
        result.omitted_operator == "block_state_control");
  }
  std::size_t column_index = 0;
  for (const auto& column : dataset.columns) {
    std::cerr << "  column=" << column_index << " bytes=" << column.size() << '\n';
    result.original_bytes += column.size();
    Bytes encoded;
    const auto t0 = Clock::now();
    encoded.resize(column.size() * 8 + 1024 * 1024);
    const auto size = codec.compress(reinterpret_cast<char*>(const_cast<std::uint8_t*>(column.data())),
                                     column.size(), reinterpret_cast<char*>(encoded.data()),
                                     encoded.size(), &options);
    if (size <= 0) throw std::runtime_error(std::string(codec.method) + " encoder failed");
    encoded.resize(static_cast<std::size_t>(size));
    std::cerr << "    encoded_bytes=" << encoded.size() << '\n';
    const auto t1 = Clock::now();
    Bytes decoded(column.size());
    const auto decoded_size = codec.decompress(
        reinterpret_cast<char*>(encoded.data()), encoded.size(),
        reinterpret_cast<char*>(decoded.data()), decoded.size(), &options);
    if (decoded_size != static_cast<int64_t>(column.size())) {
      throw std::runtime_error(std::string(codec.method) + " decoder failed at column " +
                               std::to_string(column_index) + " size=" +
                               std::to_string(decoded_size));
    }
    const auto t2 = Clock::now();
    result.compressed_bytes += encoded.size();
    result.compression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.decompression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result.lossless = result.lossless && decoded == column;
    ++column_index;
  }
  return result;
}

void write_results(const std::string& path, const Dataset& dataset,
                   const std::vector<Result>& results) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write output");
  output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
  std::unordered_map<std::string, double> baseline;
  for (const auto& result : results) {
    if (result.omitted_operator.empty()) {
      baseline[result.method] = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    }
  }
  output << std::setprecision(9);
  for (const auto& result : results) {
    const double rate = static_cast<double>(result.compressed_bytes) / result.original_bytes;
    const double base = baseline.at(result.method);
    output << "Air-Quality," << result.method << ','
           << (result.omitted_operator.empty() ? "none" : result.omitted_operator) << ','
           << result.original_bytes << ',' << result.compressed_bytes << ',' << rate << ','
           << base << ',' << (rate - base) << ',' << result.compression_ns << ','
           << result.decompression_ns << ',' << (result.lossless ? 1 : 0) << ','
           << dataset.integer_columns << ',' << dataset.float_columns << ','
           << dataset.timestamp_columns << ',' << dataset.columns.size() << ','
           << dataset.rows << ",source_upstream,source_method_specific_operator_ablation\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: lzbench_upstream_source_ablation DATASET.csv OUTPUT.csv "
                 "[METHOD1,METHOD2,...]\n";
    return 2;
  }
  try {
    const Dataset dataset = load_dataset(argv[1]);
    std::unordered_set<std::string> selected;
    if (argc == 4) {
      std::string methods = argv[3];
      std::size_t begin = 0;
      while (begin <= methods.size()) {
        const std::size_t end = methods.find(',', begin);
        selected.insert(methods.substr(begin, end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
      }
    }
    const char* omission_filter = std::getenv("TSCB_ABLATION_ONLY");
    std::vector<Result> results;
    for (const auto& codec : codecs()) {
      if (!selected.empty() && selected.count(codec.method) == 0) continue;
      // Every filtered ablation is measured against a baseline produced by the
      // same executable, build, dataset, and codec configuration.
      results.push_back(run_codec(dataset, codec, ""));
      for (const auto& operator_id : codec.operators) {
        if (omission_filter != nullptr && operator_id != omission_filter) continue;
        results.push_back(run_codec(dataset, codec, operator_id));
      }
    }
    for (const auto& result : results) {
      if (!result.lossless) throw std::runtime_error("lossless verification failed: " + result.method);
    }
    write_results(argv[2], dataset, results);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
