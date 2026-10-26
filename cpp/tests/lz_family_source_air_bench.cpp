// Source-level Air-Quality validation for five lzbench codecs.
// Baselines call the upstream encoder and decoder. Ablations either select an
// upstream alternative or use a method-tagged byte-aligned stream with its own
// decoder, so no native_lz_family result is reused as evidence.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lz/lzo/lzo1.h"
#include "lz/lzo/lzo2a.h"
#include "lz/lzav/lzav.h"
#include "lz/tamp/compressor.h"
#include "lz/tamp/decompressor.h"

#if defined(SOURCE_HAS_LZSSE)
#include "lz/lzsse/lzsse8/lzsse8.h"
#endif

namespace {

using Bytes = std::vector<std::uint8_t>;

struct Dataset {
  std::vector<Bytes> columns;
  std::size_t rows = 0;
  std::size_t integer_columns = 0;
  std::size_t float_columns = 0;
  std::size_t timestamp_columns = 0;
};

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
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
      out.push_back(std::move(field));
      field.clear();
    } else if (c != '\r') {
      field.push_back(c);
    }
  }
  out.push_back(std::move(field));
  return out;
}

bool parse_int(std::string_view text, std::int64_t& value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_double(std::string_view text, double& value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

template <typename T>
Bytes native_bytes(const std::vector<T>& values) {
  Bytes result(values.size() * sizeof(T));
  if (!result.empty()) std::memcpy(result.data(), values.data(), result.size());
  return result;
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
    std::vector<std::int64_t> integers;
    integers.reserve(dataset.rows);
    bool is_integer = true;
    for (const auto& text : values[column]) {
      std::int64_t value = 0;
      if (!parse_int(text, value)) {
        is_integer = false;
        break;
      }
      integers.push_back(value);
    }
    std::string header = headers[column];
    std::transform(header.begin(), header.end(), header.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (is_integer) {
      dataset.columns.push_back(native_bytes(integers));
      if (header == "timestamp" || header == "time") ++dataset.timestamp_columns;
      else ++dataset.integer_columns;
      continue;
    }
    std::vector<double> floats;
    floats.reserve(dataset.rows);
    for (const auto& text : values[column]) {
      double value = 0.0;
      if (!parse_double(text, value)) throw std::runtime_error("non-numeric column");
      floats.push_back(value);
    }
    dataset.columns.push_back(native_bytes(floats));
    ++dataset.float_columns;
  }
  return dataset;
}

void put_u16(Bytes& output, std::size_t value) {
  if (value > 65535) throw std::runtime_error("source ablation token exceeds uint16");
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8));
}

std::size_t get_u16(const Bytes& input, std::size_t& position) {
  if (position + 2 > input.size()) throw std::runtime_error("source ablation token truncated");
  const std::size_t value = input[position] | (static_cast<std::size_t>(input[position + 1]) << 8);
  position += 2;
  return value;
}

std::uint32_t key_at(const Bytes& input, std::size_t position) {
  if (position + 3 > input.size()) return 0xffffffffu;
  return (static_cast<std::uint32_t>(input[position]) << 16) |
         (static_cast<std::uint32_t>(input[position + 1]) << 8) | input[position + 2];
}

// This is a small source-level token writer used only when the selected
// token/control representation is explicitly bypassed.
Bytes byte_aligned_ablation(const Bytes& input, const std::string& magic,
                            bool allow_matches, int min_match, bool packed_controls) {
  Bytes output(magic.begin(), magic.end());
  std::unordered_map<std::uint32_t, std::size_t> latest;
  std::size_t position = 0;
  while (position < input.size()) {
    const std::size_t control_position = output.size();
    if (packed_controls) output.push_back(0);
    unsigned char control = 0;
    int token = 0;
    while (position < input.size() && (!packed_controls || token < 8)) {
      std::size_t match_length = 0;
      std::size_t offset = 0;
      const auto key = key_at(input, position);
      if (allow_matches && key != 0xffffffffu) {
        const auto it = latest.find(key);
        if (it != latest.end() && position > it->second && position - it->second <= 65535) {
          offset = position - it->second;
          const std::size_t limit = std::min<std::size_t>(65535, input.size() - position);
          while (match_length < limit && input[it->second + match_length] == input[position + match_length]) {
            ++match_length;
          }
          if (match_length < static_cast<std::size_t>(min_match)) match_length = 0;
        }
      }
      if (match_length >= static_cast<std::size_t>(min_match)) {
        if (packed_controls) {
          control = static_cast<unsigned char>(control | (1u << token));
        } else {
          output.push_back(1);
        }
        put_u16(output, offset);
        put_u16(output, match_length);
        for (std::size_t i = 0; i < match_length; ++i) {
          const auto k = key_at(input, position + i);
          if (k != 0xffffffffu) latest[k] = position + i;
        }
        position += match_length;
      } else {
        if (packed_controls) {
          output.push_back(input[position]);
        } else {
          output.push_back(0);
          put_u16(output, 1);
          output.push_back(input[position]);
        }
        if (key != 0xffffffffu) latest[key] = position;
        ++position;
      }
      ++token;
    }
    if (packed_controls) output[control_position] = control;
  }
  return output;
}

Bytes literal_ablation(const Bytes& input, const std::string& magic) {
  Bytes output(magic.begin(), magic.end());
  std::size_t position = 0;
  while (position < input.size()) {
    const std::size_t length = std::min<std::size_t>(65535, input.size() - position);
    output.push_back(0);
    put_u16(output, length);
    output.insert(output.end(), input.begin() + static_cast<std::ptrdiff_t>(position),
                  input.begin() + static_cast<std::ptrdiff_t>(position + length));
    position += length;
  }
  return output;
}

Bytes decode_byte_aligned(const Bytes& encoded, std::size_t expected_size,
                          const std::string& magic, bool packed_controls) {
  if (encoded.size() < magic.size() ||
      !std::equal(magic.begin(), magic.end(), encoded.begin())) {
    throw std::runtime_error("source ablation magic mismatch");
  }
  Bytes output;
  output.reserve(expected_size);
  std::size_t position = magic.size();
  while (output.size() < expected_size) {
    const unsigned char control = packed_controls ? encoded.at(position++) : 0;
    for (int token = 0; token < (packed_controls ? 8 : 1) && output.size() < expected_size; ++token) {
      const bool match = packed_controls ? ((control >> token) & 1u) != 0 : encoded.at(position++) != 0;
      if (!match) {
        const std::size_t length = packed_controls ? 1 : get_u16(encoded, position);
        if (position + length > encoded.size() || output.size() + length > expected_size) {
          throw std::runtime_error("source ablation literal invalid");
        }
        output.insert(output.end(), encoded.begin() + static_cast<std::ptrdiff_t>(position),
                     encoded.begin() + static_cast<std::ptrdiff_t>(position + length));
        position += length;
      } else {
        const std::size_t offset = get_u16(encoded, position);
        const std::size_t length = get_u16(encoded, position);
        if (offset == 0 || offset > output.size() || output.size() + length > expected_size) {
          throw std::runtime_error("source ablation match invalid");
        }
        const std::size_t start = output.size() - offset;
        for (std::size_t i = 0; i < length; ++i) output.push_back(output[start + i]);
      }
    }
  }
  if (position != encoded.size()) throw std::runtime_error("source ablation trailing bytes");
  return output;
}

Bytes decode_literal(const Bytes& encoded, std::size_t expected_size, const std::string& magic) {
  return decode_byte_aligned(encoded, expected_size, magic, false);
}

Bytes upstream_lzo1(const Bytes& input) {
  if (input.empty()) return {};
  Bytes output(input.size() + input.size() / 8 + 4096);
  std::vector<unsigned char> work(LZO1_MEM_COMPRESS);
  lzo_uint size = 0;
  if (lzo1_compress(input.data(), input.size(), output.data(), &size, work.data()) != LZO_E_OK)
    throw std::runtime_error("LZO1 encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_lzo2a(const Bytes& input) {
  if (input.empty()) return {};
  Bytes output(input.size() + input.size() / 8 + 4096);
  std::vector<unsigned char> work(LZO2A_999_MEM_COMPRESS);
  lzo_uint size = 0;
  if (lzo2a_999_compress(input.data(), input.size(), output.data(), &size, work.data()) != LZO_E_OK)
    throw std::runtime_error("LZO2A encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_lzav(const Bytes& input, bool high, bool mref6) {
  if (input.empty()) return {};
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("LZAV input too large");
  const int length = static_cast<int>(input.size());
  const int bound = (high ? lzav_compress_bound_hi(length) : lzav_compress_bound(length)) + 64;
  Bytes output(static_cast<std::size_t>(bound));
  int size = 0;
  if (high) {
    size = lzav_compress_hi(input.data(), output.data(), length, bound);
  } else if (mref6) {
    size = lzav_compress_mref6(input.data(), output.data(), length, bound, nullptr, 0);
  } else {
    size = lzav_compress_default(input.data(), output.data(), length, bound);
  }
  if (size <= 0) throw std::runtime_error("LZAV encoder failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes upstream_tamp(const Bytes& input, bool remove_frame) {
  if (input.empty()) return {};
  constexpr uint8_t window_bits = 10;
  TampConf conf{};
  conf.window = window_bits;
  conf.literal = 8;
  conf.use_custom_dictionary = false;
  conf.extended = true;
  conf.dictionary_reset = false;
  conf.append = false;
  std::vector<unsigned char> window(1u << window_bits);
  TampCompressor compressor{};
  if (tamp_compressor_init(&compressor, &conf, window.data()) != TAMP_OK)
    throw std::runtime_error("TAMP encoder init failed");
  Bytes output(input.size() + input.size() / 8 + 1024);
  size_t written = 0;
  if (tamp_compressor_compress_and_flush(&compressor, output.data(), output.size(), &written,
                                         input.data(), input.size(), nullptr, false) != TAMP_OK)
    throw std::runtime_error("TAMP encoder failed");
  output.resize(written);
  if (remove_frame) {
    if (output.empty()) throw std::runtime_error("TAMP frame missing");
    output.erase(output.begin());
  }
  return output;
}

Bytes decode_tamp(const Bytes& encoded, std::size_t expected_size, bool no_frame) {
  if (expected_size == 0) return {};
  constexpr uint8_t window_bits = 10;
  TampConf conf{};
  conf.window = window_bits;
  conf.literal = 8;
  conf.use_custom_dictionary = false;
  conf.extended = true;
  conf.dictionary_reset = false;
  conf.append = false;
  std::vector<unsigned char> window(1u << window_bits);
  TampDecompressor decompressor{};
  const unsigned char* input = encoded.data();
  size_t input_size = encoded.size();
  if (no_frame) {
    if (tamp_decompressor_init(&decompressor, &conf, window.data(), window_bits) != TAMP_OK)
      throw std::runtime_error("TAMP explicit decoder init failed");
  } else {
    if (tamp_decompressor_init(&decompressor, nullptr, window.data(), window_bits) != TAMP_OK)
      throw std::runtime_error("TAMP decoder init failed");
  }
  Bytes output(expected_size);
  size_t written = 0;
  const auto status = tamp_decompressor_decompress(&decompressor, output.data(), output.size(),
                                                   &written, input, input_size, nullptr);
  if (status != TAMP_OK && status != TAMP_INPUT_EXHAUSTED && status != TAMP_OUTPUT_FULL)
    throw std::runtime_error("TAMP decoder failed");
  if (written != expected_size) throw std::runtime_error("TAMP decoded size mismatch");
  return output;
}

#if defined(SOURCE_HAS_LZSSE)
Bytes upstream_lzsse(const Bytes& input, bool fast) {
  if (input.empty()) return {};
  Bytes output(input.size() + 64);
  size_t size = 0;
  if (fast) {
    auto* state = LZSSE8_MakeFastParseState();
    if (!state) throw std::runtime_error("LZSSE fast state allocation failed");
    size = LZSSE8_CompressFast(state, input.data(), input.size(), output.data(), output.size());
    LZSSE8_FreeFastParseState(state);
  } else {
    auto* state = LZSSE8_MakeOptimalParseState(input.size());
    if (!state) throw std::runtime_error("LZSSE optimal state allocation failed");
    size = LZSSE8_CompressOptimalParse(state, input.data(), input.size(), output.data(), output.size(), 17);
    LZSSE8_FreeOptimalParseState(state);
  }
  if (size == 0) throw std::runtime_error("LZSSE encoder failed");
  output.resize(size);
  return output;
}

Bytes decode_lzsse(const Bytes& encoded, std::size_t expected_size) {
  Bytes output(expected_size);
  if (LZSSE8_Decompress(encoded.data(), encoded.size(), output.data(), output.size()) != expected_size)
    throw std::runtime_error("LZSSE decoder failed");
  return output;
}
#endif

struct Codec {
  std::string method;
  std::vector<std::string> omissions;
  int min_match = 3;
  Bytes (*encode)(const Bytes&, const std::string&);
  Bytes (*decode)(const Bytes&, std::size_t, const std::string&);
};

Bytes encode_method(const std::string& method, const Bytes& input, const std::string& omission) {
  if (method == "LZO1") {
    if (omission.empty()) return upstream_lzo1(input);
    if (omission == "lz_match") return literal_ablation(input, "L1LT");
    if (omission == "bitstream_pack") return byte_aligned_ablation(input, "L1BP", true, 3, true);
    return byte_aligned_ablation(input, "L1TK", true, 3, false);
  }
  if (method == "LZO2A") {
    if (omission.empty()) return upstream_lzo2a(input);
    if (omission == "lz_match") return literal_ablation(input, "L2LT");
    if (omission == "bitstream_pack") return byte_aligned_ablation(input, "L2BP", true, 3, true);
    return byte_aligned_ablation(input, "L2TK", true, 3, false);
  }
  if (method == "LZAV") {
    if (omission.empty()) return upstream_lzav(input, true, false);
    if (omission == "lz_match") return literal_ablation(input, "AVLT");
    if (omission == "match_window_model") return upstream_lzav(input, false, true);
    return byte_aligned_ablation(input, "AVTK", true, 5, false);
  }
  if (method == "TAMP") {
    if (omission.empty()) return upstream_tamp(input, false);
    if (omission == "lz_match") return literal_ablation(input, "TMLT");
    if (omission == "container_frame") return upstream_tamp(input, true);
    return byte_aligned_ablation(input, "TMTK", true, 3, false);
  }
#if defined(SOURCE_HAS_LZSSE)
  if (method == "LZSSE") {
    if (omission.empty()) return upstream_lzsse(input, false);
    if (omission == "match_parse_policy") return upstream_lzsse(input, true);
    if (omission == "lz_match") return literal_ablation(input, "S8LT");
    if (omission == "block_state_control") return byte_aligned_ablation(input, "S8BS", true, 4, false);
    return byte_aligned_ablation(input, "S8TK", true, 4, false);
  }
#endif
  throw std::runtime_error("unsupported source codec or unavailable LZSSE: " + method);
}

Bytes decode_method(const std::string& method, const Bytes& encoded, std::size_t expected_size,
                    const std::string& omission) {
  if (expected_size == 0) return {};
  if (method == "LZO1") {
    if (omission.empty()) {
      Bytes output(expected_size); lzo_uint size = 0;
      if (lzo1_decompress(encoded.data(), encoded.size(), output.data(), &size, nullptr) != LZO_E_OK || size != expected_size)
        throw std::runtime_error("LZO1 decoder failed");
      return output;
    }
    return omission == "lz_match" ? decode_literal(encoded, expected_size, "L1LT")
                                    : decode_byte_aligned(encoded, expected_size, omission == "bitstream_pack" ? "L1BP" : "L1TK", omission == "bitstream_pack");
  }
  if (method == "LZO2A") {
    if (omission.empty()) {
      Bytes output(expected_size); lzo_uint size = 0;
      if (lzo2a_decompress(encoded.data(), encoded.size(), output.data(), &size, nullptr) != LZO_E_OK || size != expected_size)
        throw std::runtime_error("LZO2A decoder failed");
      return output;
    }
    return omission == "lz_match" ? decode_literal(encoded, expected_size, "L2LT")
                                    : decode_byte_aligned(encoded, expected_size, omission == "bitstream_pack" ? "L2BP" : "L2TK", omission == "bitstream_pack");
  }
  if (method == "LZAV") {
    if (omission.empty() || omission == "match_window_model") {
      Bytes output(expected_size);
      if (lzav_decompress(encoded.data(), output.data(), encoded.size(), output.size()) != static_cast<int>(expected_size))
        throw std::runtime_error("LZAV decoder failed");
      return output;
    }
    return omission == "lz_match" ? decode_literal(encoded, expected_size, "AVLT")
                                    : decode_byte_aligned(encoded, expected_size, "AVTK", false);
  }
  if (method == "TAMP") {
    if (omission.empty()) return decode_tamp(encoded, expected_size, false);
    if (omission == "container_frame") return decode_tamp(encoded, expected_size, true);
    return omission == "lz_match" ? decode_literal(encoded, expected_size, "TMLT")
                                    : decode_byte_aligned(encoded, expected_size, "TMTK", false);
  }
#if defined(SOURCE_HAS_LZSSE)
  if (method == "LZSSE") {
    if (omission.empty() || omission == "match_parse_policy") return decode_lzsse(encoded, expected_size);
    return omission == "lz_match" ? decode_literal(encoded, expected_size, "S8LT")
           : decode_byte_aligned(encoded, expected_size, omission == "block_state_control" ? "S8BS" : "S8TK", false);
  }
#endif
  throw std::runtime_error("unsupported source decoder: " + method);
}

struct Row {
  std::string method;
  std::string omission;
  std::size_t original = 0;
  std::size_t compressed = 0;
  std::int64_t compression_ns = 0;
  std::int64_t decompression_ns = 0;
  bool lossless = true;
};

Row run(const std::string& method, const std::string& omission, const Dataset& dataset) {
  Row row{method, omission};
  for (const auto& column : dataset.columns) {
    const auto compression_start = std::chrono::steady_clock::now();
    const Bytes encoded = encode_method(method, column, omission);
    const auto compression_end = std::chrono::steady_clock::now();
    const auto decompression_start = std::chrono::steady_clock::now();
    const Bytes decoded = decode_method(method, encoded, column.size(), omission);
    const auto decompression_end = std::chrono::steady_clock::now();
    if (decoded != column) throw std::runtime_error(method + " lossless check failed for " + omission);
    row.original += column.size();
    row.compressed += encoded.size();
    row.compression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(compression_end - compression_start).count();
    row.decompression_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(decompression_end - decompression_start).count();
  }
  return row;
}

void run_control_sequences() {
  const std::vector<Bytes> controls = {
      {}, Bytes{'a'}, Bytes{'a','b','c','a','b','c','a','b','c'},
      Bytes(4096, static_cast<std::uint8_t>('x'))};
  const std::vector<std::pair<std::string, std::vector<std::string>>> methods = {
      {"LZO1", {"", "lz_match", "match_token_format", "bitstream_pack"}},
      {"LZO2A", {"", "lz_match", "match_token_format", "bitstream_pack"}},
      {"LZAV", {"", "lz_match", "match_window_model", "match_token_format"}},
      {"TAMP", {"", "lz_match", "container_frame", "match_token_format"}},
#if defined(SOURCE_HAS_LZSSE)
      {"LZSSE", {"", "lz_match", "match_parse_policy", "match_token_format", "block_state_control"}},
#endif
  };
  for (const auto& method : methods) for (const auto& input : controls) for (const auto& omission : method.second) {
    const Bytes encoded = encode_method(method.first, input, omission);
    if (decode_method(method.first, encoded, input.size(), omission) != input)
      throw std::runtime_error("control sequence validation failed for " + method.first + ":" + omission);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: lz_family_source_air_bench DATASET.csv OUTPUT.csv\n";
    return 2;
  }
  try {
    run_control_sequences();
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<std::pair<std::string, std::vector<std::string>>> methods = {
        {"LZO1", {"", "lz_match", "match_token_format", "bitstream_pack"}},
        {"LZO2A", {"", "lz_match", "match_token_format", "bitstream_pack"}},
        {"LZAV", {"", "lz_match", "match_window_model", "match_token_format"}},
        {"TAMP", {"", "lz_match", "container_frame", "match_token_format"}},
#if defined(SOURCE_HAS_LZSSE)
        {"LZSSE", {"", "lz_match", "match_parse_policy", "match_token_format", "block_state_control"}},
#endif
    };
    std::ofstream output(argv[2]);
    output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
    for (const auto& method : methods) {
      const Row baseline = run(method.first, "", dataset);
      const double baseline_rate = static_cast<double>(baseline.compressed) / baseline.original;
      for (const auto& omission : method.second) {
        const Row row = omission.empty() ? baseline : run(method.first, omission, dataset);
        const double rate = static_cast<double>(row.compressed) / row.original;
        output << "Air-Quality," << row.method << ',' << (row.omission.empty() ? "none" : row.omission)
               << ',' << row.original << ',' << row.compressed << ',' << std::fixed << std::setprecision(9)
               << rate << ',' << baseline_rate << ',' << (rate - baseline_rate) << ','
               << row.compression_ns << ',' << row.decompression_ns << ",1,"
               << dataset.integer_columns << ',' << dataset.float_columns << ','
               << dataset.timestamp_columns << ',' << dataset.columns.size() << ',' << dataset.rows
               << ",source_upstream," << row.method << "_upstream_source_operator_ablation" << '\n';
        std::cerr << row.method << ',' << (row.omission.empty() ? "none" : row.omission)
                  << " rate=" << rate << " delta=" << (rate - baseline_rate) << '\n';
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
