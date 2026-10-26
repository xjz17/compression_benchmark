// Source-level Air-Quality ablation runner for codecs with self-contained
// upstream C/C++ interfaces. Baselines call the upstream encoder/decoder.
// Ablation streams are method-tagged and decoded by their paired source-level
// decoder; they are not native_lz_family estimates.

#include <cstdint>

#include "lz/brieflz/brieflz.h"
#include "lz/crush/crush.hpp"
#include "lz/fastlz/fastlz.h"
#include "lz/lzmat/lzmat.h"
#include "lz/liblzg/lzg.h"
#include "ucl/ucl.h"
#include "lz/yalz77/lz77.h"
extern "C" {
#include "lz/lzrw/lzrw.h"
#include "lz/wflz/wfLZ.h"
}
#include "lz/yappy/yappy.hpp"
#include "lz/libdeflate/libdeflate.h"
#include "lz/lizard/lizard_compress.h"
#include "lz/lizard/lizard_decompress.h"
#include "lz/quicklz/quicklz151b7.h"
extern "C" {
#include "lz/slz/src/slz.h"
}
#include "lz/zxc/include/zxc.h"
extern "C" {
#include "brotli/encode.h"
#include "brotli/decode.h"
#include "lz/lzfse/lzfse.h"
}
extern "C" {
#undef z_const
#undef Z_NULL
#define in_func zlibng_in_func
#include "lz/zlib-ng/zlib-ng.h"
#undef in_func
}
#include <zlib.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;

struct Dataset {
  std::vector<Bytes> columns;
  std::size_t integer_columns = 0;
  std::size_t float_columns = 0;
  std::size_t timestamp_columns = 0;
  std::size_t rows = 0;
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
    std::string name = headers[column];
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
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
  if (value > 65535) throw std::runtime_error("ablation token exceeds uint16");
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8));
}

std::size_t get_u16(const Bytes& input, std::size_t& position) {
  if (position + 2 > input.size()) throw std::runtime_error("ablation token truncated");
  const std::size_t value = input[position] | (static_cast<std::size_t>(input[position + 1]) << 8);
  position += 2;
  return value;
}

std::uint32_t key_at(const Bytes& input, std::size_t position) {
  if (position + 3 > input.size()) return 0xffffffffu;
  return (static_cast<std::uint32_t>(input[position]) << 16) |
         (static_cast<std::uint32_t>(input[position + 1]) << 8) | input[position + 2];
}

// Explicit byte-aligned replacement used when the selected upstream operator
// is bypassed. The control byte, offset, and length are all independently
// decoded, which makes encoder/decoder symmetry testable.
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
          const std::size_t limit = std::min<std::size_t>(
              65535, std::min(input.size() - position, input.size() - it->second));
          while (match_length < limit && input[it->second + match_length] == input[position + match_length]) ++match_length;
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
  return replacement_stream(input, magic, false, 1);
}

Bytes decode_replacement(const Bytes& encoded, std::size_t expected_size,
                         const std::string& magic) {
  if (encoded.size() < magic.size() || !std::equal(magic.begin(), magic.end(), encoded.begin()))
    throw std::runtime_error("replacement magic mismatch");
  Bytes output;
  output.reserve(expected_size);
  std::size_t position = magic.size();
  while (output.size() < expected_size) {
    if (position >= encoded.size()) throw std::runtime_error("replacement control truncated");
    const auto control = encoded[position++];
    for (int token = 0; token < 8 && output.size() < expected_size; ++token) {
      if ((control & (1u << token)) == 0) {
        if (position >= encoded.size()) throw std::runtime_error("replacement literal truncated");
        output.push_back(encoded[position++]);
      } else {
        const std::size_t offset = get_u16(encoded, position);
        const std::size_t length = get_u16(encoded, position);
        if (offset == 0 || offset > output.size() || output.size() + length > expected_size)
          throw std::runtime_error("replacement match invalid");
        const std::size_t start = output.size() - offset;
        for (std::size_t i = 0; i < length; ++i) output.push_back(output[start + i]);
      }
    }
  }
  if (position != encoded.size()) throw std::runtime_error("replacement trailing bytes");
  return output;
}

Bytes upstream_brieflz(const Bytes& input) {
  if (input.empty()) return {};
  Bytes output(input.size() + input.size() / 8 + 4096);
  std::vector<std::uint8_t> work(blz_workmem_size_level(input.size(), 9));
  const auto size = blz_pack_level(input.data(), output.data(), input.size(), work.data(), 9);
  if (size <= 0) throw std::runtime_error("BriefLZ encoder failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes upstream_fastlz(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  const int size = fastlz_compress_level(2, input.data(), static_cast<int>(input.size()), output.data());
  if (size <= 0) throw std::runtime_error("FastLZ encoder failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes upstream_lzmat(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 128);
  MP_U32 size = static_cast<MP_U32>(output.size());
  if (lzmat_encode(output.data(), &size, const_cast<MP_U8*>(input.data()), static_cast<MP_U32>(input.size())) != 0)
    throw std::runtime_error("LZMAT encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_lzrw(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  std::vector<std::uint8_t> work(lzrw2_req_mem());
  std::uint32_t size = 0;
  lzrw2_compress(COMPRESS_ACTION_COMPRESS, work.data(), const_cast<std::uint8_t*>(input.data()), input.size(), output.data(), &size);
  if (size == 0) throw std::runtime_error("LZRW encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_wflz(const Bytes& input, bool fast) {
  Bytes output(wfLZ_GetMaxCompressedSize(static_cast<std::uint32_t>(input.size())) + 64);
  std::vector<std::uint8_t> work(wfLZ_GetWorkMemSize());
  const auto size = fast ? wfLZ_CompressFast(input.data(), input.size(), output.data(), work.data(), 0)
                         : wfLZ_Compress(input.data(), input.size(), output.data(), work.data(), 0);
  if (size == 0) throw std::runtime_error("WFLZ encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_yappy(const Bytes& input, int level) {
  YappyFillTables();
  Bytes output(input.size() * 2 + 64);
  const auto* end = YappyCompress(input.data(), output.data(), input.size(), level);
  output.resize(static_cast<std::size_t>(end - output.data()));
  return output;
}

Bytes upstream_lzg(const Bytes& input, int level, int fast) {
  if (input.empty()) return {};
  Bytes output(LZG_MaxEncodedSize(static_cast<lzg_uint32_t>(input.size())));
  lzg_encoder_config_t config;
  LZG_InitEncoderConfig(&config);
  config.level = level;
  config.fast = fast;
  const auto size = LZG_Encode(input.data(), static_cast<lzg_uint32_t>(input.size()),
                               output.data(), static_cast<lzg_uint32_t>(output.size()), &config);
  if (size == 0) throw std::runtime_error("LZG encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_yalz77(const Bytes& input, std::size_t search_length) {
  const auto begin = reinterpret_cast<const unsigned char*>(input.data());
  const auto end = begin + input.size();
  lz77::compress_t compressor(search_length, lz77::DEFAULT_BLOCKSIZE);
  const std::string compressed = compressor.feed(begin, end);
  return Bytes(compressed.begin(), compressed.end());
}

Bytes upstream_ucl(const Bytes& input, bool nrv2d) {
  if (input.empty()) return {};
  Bytes output(input.size() + input.size() / 8 + 4096);
  ucl_uint size = static_cast<ucl_uint>(output.size());
  const int result = nrv2d
      ? ucl_nrv2d_99_compress(input.data(), static_cast<ucl_uint>(input.size()), output.data(), &size, nullptr, 9, nullptr, nullptr)
      : ucl_nrv2b_99_compress(input.data(), static_cast<ucl_uint>(input.size()), output.data(), &size, nullptr, 9, nullptr, nullptr);
  if (result != UCL_E_OK) throw std::runtime_error("UCL encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_crush(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  const auto size = crush::compress(2, const_cast<std::uint8_t*>(input.data()), input.size(), output.data());
  if (size == 0) throw std::runtime_error("Crush encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_libdeflate(const Bytes& input) {
  auto* compressor = libdeflate_alloc_compressor(12);
  if (!compressor) throw std::runtime_error("libdeflate compressor allocation failed");
  Bytes output(libdeflate_deflate_compress_bound(compressor, input.size()));
  const auto size = libdeflate_deflate_compress(compressor, input.data(), input.size(), output.data(), output.size());
  libdeflate_free_compressor(compressor);
  if (size == 0) throw std::runtime_error("libdeflate encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_lizard(const Bytes& input) {
  Bytes output(static_cast<std::size_t>(Lizard_compressBound(static_cast<int>(input.size()))));
  const auto size = Lizard_compress(reinterpret_cast<const char*>(input.data()),
                                    reinterpret_cast<char*>(output.data()), static_cast<int>(input.size()),
                                    static_cast<int>(output.size()), 19);
  if (size <= 0) throw std::runtime_error("Lizard encoder failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes upstream_quicklz(const Bytes& input) {
  const std::size_t state_size = std::max({static_cast<std::size_t>(qlz151_get_setting_1(1)),
                                           static_cast<std::size_t>(qlz151_get_setting_2(1)),
                                           static_cast<std::size_t>(qlz151_get_setting_3(1))});
  std::vector<std::uint8_t> state(state_size);
  Bytes output(input.size() * 2 + 1024);
  const auto size = qlz151_compress_1(input.data(), reinterpret_cast<char*>(output.data()), input.size(),
                                      reinterpret_cast<qlz_state_compress*>(state.data()));
  if (size == 0) throw std::runtime_error("QuickLZ encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_zlib_ng(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  size_t size = output.size();
  if (zng_compress2(output.data(), &size, input.data(), input.size(), 6) != Z_OK)
    throw std::runtime_error("zlib-ng encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_slz(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  slz_stream stream;
  if (slz_init(&stream, 1, SLZ_FMT_ZLIB) != 0) throw std::runtime_error("SLZ init failed");
  const long body = slz_encode(&stream, output.data(), input.data(), static_cast<long>(input.size()), 0);
  if (body < 0) throw std::runtime_error("SLZ encoder failed");
  const long trailer = slz_finish(&stream, output.data() + body);
  if (trailer < 0) throw std::runtime_error("SLZ finish failed");
  output.resize(static_cast<std::size_t>(body + trailer));
  return output;
}

Bytes upstream_zxc(const Bytes& input) {
  zxc_compress_opts_t options{};
  options.level = ZXC_LEVEL_DEFAULT;
  options.checksum_enabled = 1;
  Bytes output(static_cast<std::size_t>(zxc_compress_bound(input.size())));
  const auto size = zxc_compress(input.data(), input.size(), output.data(), output.size(), &options);
  if (size <= 0) throw std::runtime_error("ZXC encoder failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes upstream_brotli(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  size_t size = output.size();
  if (!BrotliEncoderCompress(11, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE,
                             input.size(), input.data(), &size, output.data()))
    throw std::runtime_error("Brotli encoder failed");
  output.resize(size);
  return output;
}

Bytes upstream_lzfse(const Bytes& input) {
  Bytes output(input.size() + input.size() / 8 + 4096);
  std::vector<std::uint8_t> scratch(std::max(lzfse_encode_scratch_size(), lzfse_decode_scratch_size()));
  const auto size = lzfse_encode_buffer(output.data(), output.size(), input.data(), input.size(), scratch.data());
  if (size == 0) throw std::runtime_error("LZFSE encoder failed");
  output.resize(size);
  return output;
}

Bytes decode_upstream(const std::string& method, const Bytes& encoded, std::size_t expected_size) {
  Bytes output(expected_size);
  if (method == "BriefLZ") {
    if (blz_depack(encoded.data(), output.data(), expected_size) != expected_size) throw std::runtime_error("BriefLZ decoder failed");
  } else if (method == "Crush") {
    if (crush::decompress(const_cast<std::uint8_t*>(encoded.data()), output.data(), expected_size) != expected_size) throw std::runtime_error("Crush decoder failed");
  } else if (method == "FastLZ") {
    if (fastlz_decompress(encoded.data(), encoded.size(), output.data(), expected_size) != static_cast<int>(expected_size)) throw std::runtime_error("FastLZ decoder failed");
  } else if (method == "LZMAT") {
    MP_U32 size = static_cast<MP_U32>(expected_size);
    if (lzmat_decode(output.data(), &size, const_cast<MP_U8*>(encoded.data()), encoded.size()) != 0 || size != expected_size) throw std::runtime_error("LZMAT decoder failed");
  } else if (method == "LZRW") {
    std::vector<std::uint8_t> work(lzrw2_req_mem());
    std::uint32_t size = 0;
    lzrw2_compress(COMPRESS_ACTION_DECOMPRESS, work.data(), const_cast<std::uint8_t*>(encoded.data()), encoded.size(), output.data(), &size);
    if (size != expected_size) throw std::runtime_error("LZRW decoder failed");
  } else if (method == "WFLZ") {
    if (wfLZ_GetDecompressedSize(encoded.data()) != expected_size) throw std::runtime_error("WFLZ size check failed");
    wfLZ_Decompress(encoded.data(), output.data());
  } else if (method == "Yappy") {
    output.resize(expected_size + 32);
    const auto* end = YappyUnCompress(encoded.data(), encoded.data() + encoded.size(), output.data());
    if (static_cast<std::size_t>(end - output.data()) != expected_size) throw std::runtime_error("Yappy decoder failed");
    output.resize(expected_size);
  } else if (method == "LZG") {
    if (LZG_DecodedSize(encoded.data(), encoded.size()) != expected_size ||
        LZG_Decode(encoded.data(), encoded.size(), output.data(), expected_size) != expected_size)
      throw std::runtime_error("LZG decoder failed");
  } else if (method == "YALZ77") {
    lz77::decompress_t decompressor;
    std::string extra;
    decompressor.feed(reinterpret_cast<const unsigned char*>(encoded.data()),
                      reinterpret_cast<const unsigned char*>(encoded.data() + encoded.size()), extra);
    const std::string& restored = decompressor.result();
    if (restored.size() != expected_size) throw std::runtime_error("YALZ77 decoder failed");
    std::copy(restored.begin(), restored.end(), output.begin());
  } else if (method == "UCL NRV") {
    ucl_uint size = static_cast<ucl_uint>(expected_size);
    const int result = ucl_nrv2b_decompress_safe_8(encoded.data(), static_cast<ucl_uint>(encoded.size()), output.data(), &size, nullptr);
    if (result != UCL_E_OK || size != expected_size) throw std::runtime_error("UCL decoder failed");
  } else if (method == "libdeflate") {
    auto* decompressor = libdeflate_alloc_decompressor();
    if (!decompressor) throw std::runtime_error("libdeflate decompressor allocation failed");
    size_t size = 0;
    const auto result = libdeflate_deflate_decompress(decompressor, encoded.data(), encoded.size(), output.data(), output.size(), &size);
    libdeflate_free_decompressor(decompressor);
    if (result != LIBDEFLATE_SUCCESS || size != expected_size) throw std::runtime_error("libdeflate decoder failed");
  } else if (method == "Lizard") {
    if (Lizard_decompress_safe(reinterpret_cast<const char*>(encoded.data()), reinterpret_cast<char*>(output.data()),
                               static_cast<int>(encoded.size()), static_cast<int>(expected_size)) != static_cast<int>(expected_size))
      throw std::runtime_error("Lizard decoder failed");
  } else if (method == "QuickLZ") {
    const std::size_t state_size = std::max({static_cast<std::size_t>(qlz151_get_setting_1(2)),
                                             static_cast<std::size_t>(qlz151_get_setting_2(2)),
                                             static_cast<std::size_t>(qlz151_get_setting_3(2))});
    std::vector<std::uint8_t> state(state_size);
    if (qlz151_decompress_1(reinterpret_cast<const char*>(encoded.data()), output.data(),
                             reinterpret_cast<qlz_state_decompress*>(state.data())) != expected_size)
      throw std::runtime_error("QuickLZ decoder failed");
  } else if (method == "zlib-ng") {
    size_t size = expected_size;
    if (zng_uncompress(output.data(), &size, encoded.data(), encoded.size()) != Z_OK || size != expected_size)
      throw std::runtime_error("zlib-ng decoder failed");
  } else if (method == "SLZ") {
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(encoded.data()));
    stream.avail_in = static_cast<uInt>(encoded.size());
    stream.next_out = reinterpret_cast<Bytef*>(output.data());
    stream.avail_out = static_cast<uInt>(output.size());
    if (inflateInit(&stream) != Z_OK || inflate(&stream, Z_FINISH) != Z_STREAM_END || stream.total_out != expected_size) {
      inflateEnd(&stream);
      throw std::runtime_error("SLZ decoder failed");
    }
    inflateEnd(&stream);
  } else if (method == "ZXC") {
    zxc_decompress_opts_t options{};
    options.checksum_enabled = 1;
    const auto size = zxc_decompress(encoded.data(), encoded.size(), output.data(), output.size(), &options);
    if (size != static_cast<int64_t>(expected_size)) throw std::runtime_error("ZXC decoder failed");
  } else if (method == "brotli") {
    size_t size = expected_size;
    if (BrotliDecoderDecompress(encoded.size(), encoded.data(), &size, output.data()) == BROTLI_DECODER_RESULT_ERROR || size != expected_size)
      throw std::runtime_error("Brotli decoder failed");
  } else if (method == "LZFSE") {
    std::vector<std::uint8_t> scratch(std::max(lzfse_encode_scratch_size(), lzfse_decode_scratch_size()));
    if (lzfse_decode_buffer(output.data(), output.size(), encoded.data(), encoded.size(), scratch.data()) != expected_size)
      throw std::runtime_error("LZFSE decoder failed");
  } else {
    throw std::runtime_error("unsupported upstream decoder: " + method);
  }
  return output;
}

Bytes encode(const std::string& method, const Bytes& input, const std::string& omission) {
  if (input.empty()) return {};
  if (method == "BriefLZ") {
    if (omission.empty()) return upstream_brieflz(input);
    if (omission == "lz_match") return replacement_literal(input, "BLZL");
    return replacement_stream(input, "BLZB", true, 65535);
  }
  if (method == "Crush") {
    if (omission.empty()) return upstream_crush(input);
    if (omission == "lz_match") return replacement_literal(input, "CRSL");
    return replacement_stream(input, "CRST", true, 65535);
  }
  if (method == "FastLZ") {
    if (omission.empty()) return upstream_fastlz(input);
    if (omission == "lz_match") return replacement_literal(input, "FLLT");
    if (omission == "match_policy") {
      Bytes output(input.size() + input.size() / 8 + 4096);
      const int size = fastlz_compress_level(1, input.data(), static_cast<int>(input.size()), output.data());
      if (size <= 0) throw std::runtime_error("FastLZ level-1 encoder failed");
      output.resize(size);
      return output;
    }
    return replacement_stream(input, "FLTK", true, 65535);
  }
  if (method == "LZMAT") {
    if (omission.empty()) return upstream_lzmat(input);
    if (omission == "lz_match") return replacement_literal(input, "LMLT");
    if (omission == "match_policy") return replacement_stream(input, "LMGP", true, 4096);
    return replacement_stream(input, "LMTK", true, 65535);
  }
  if (method == "LZRW") {
    if (omission.empty()) return upstream_lzrw(input);
    if (omission == "lz_match") return replacement_literal(input, "LRLT");
    if (omission == "window_model") return replacement_stream(input, "LRWM", true, 4096);
    return replacement_stream(input, "LRTK", true, 65535);
  }
  if (method == "WFLZ") {
    if (omission.empty()) return upstream_wflz(input, false);
    if (omission == "lz_match") return replacement_literal(input, "WFLT");
    if (omission == "window_model") return replacement_stream(input, "WFWM", true, 4096);
    return replacement_stream(input, "WFTK", true, 65535);
  }
  if (method == "Yappy") {
    if (omission.empty()) return upstream_yappy(input, 8);
    if (omission == "lz_match") return replacement_literal(input, "YPLT");
    if (omission == "match_policy") return upstream_yappy(input, 0);
    return replacement_stream(input, "YPTK", true, 4095);
  }
  if (method == "LZG") {
    if (omission.empty()) return upstream_lzg(input, 9, 1);
    if (omission == "match_parse_policy") return upstream_lzg(input, 1, 1);
    if (omission == "lz_match") return replacement_literal(input, "LZGL");
    if (omission == "match_window_model") return replacement_stream(input, "LZGW", true, 4096);
    return replacement_stream(input, "LZGT", true, 65535);
  }
  if (method == "YALZ77") {
    if (omission.empty()) return upstream_yalz77(input, 12);
    if (omission == "match_parse_policy") return upstream_yalz77(input, 4);
    if (omission == "lz_match") return replacement_literal(input, "Y7LT");
    return replacement_stream(input, "Y7TK", true, 65535);
  }
  if (method == "UCL NRV") {
    if (omission.empty()) return upstream_ucl(input, false);
    if (omission == "match_parse_policy") return upstream_ucl(input, true);
    if (omission == "lz_match") return replacement_literal(input, "UCLL");
    return replacement_stream(input, "UCLB", true, 65535);
  }
  if (method == "libdeflate") {
    if (omission.empty()) return upstream_libdeflate(input);
    if (omission == "lz_match") return replacement_literal(input, "LDFL");
    return replacement_stream(input, "LDFH", true, 65535);
  }
  if (method == "Lizard") {
    if (omission.empty()) return upstream_lizard(input);
    if (omission == "lz_match") return replacement_literal(input, "LZDL");
    if (omission == "match_window_model") return replacement_stream(input, "LZDW", true, 4096);
    return replacement_stream(input, "LZDT", true, 65535);
  }
  if (method == "QuickLZ") {
    if (omission.empty()) return upstream_quicklz(input);
    if (omission == "lz_match") return replacement_literal(input, "QLOL");
    if (omission == "match_window_model") return replacement_stream(input, "QLOW", true, 4096);
    return replacement_stream(input, "QLOT", true, 65535);
  }
  if (method == "zlib-ng") {
    if (omission.empty()) return upstream_zlib_ng(input);
    if (omission == "lz_match") return replacement_literal(input, "ZNGL");
    return replacement_stream(input, "ZNGH", true, 65535);
  }
  if (method == "SLZ") {
    if (omission.empty()) return upstream_slz(input);
    if (omission == "lz_match") return replacement_literal(input, "SLZL");
    return replacement_stream(input, "SLZH", true, 65535);
  }
  if (method == "ZXC") {
    if (omission.empty()) return upstream_zxc(input);
    if (omission == "lz_match") return replacement_literal(input, "ZXCL");
    return replacement_stream(input, "ZXCT", true, 65535);
  }
  if (method == "brotli") {
    if (omission.empty()) return upstream_brotli(input);
    if (omission == "lz_match") return replacement_literal(input, "BRTL");
    return replacement_stream(input, "BRTH", true, 65535);
  }
  if (method == "LZFSE") {
    if (omission.empty()) return upstream_lzfse(input);
    if (omission == "lz_match") return replacement_literal(input, "LSEL");
    return replacement_stream(input, "LSEH", true, 65535);
  }
  throw std::runtime_error("unsupported source codec: " + method);
}

Bytes decode(const std::string& method, const Bytes& encoded, std::size_t expected_size,
             const std::string& omission) {
  if (expected_size == 0) return {};
  if (method == "UCL NRV" && omission == "match_parse_policy") {
    Bytes output(expected_size);
    ucl_uint size = static_cast<ucl_uint>(expected_size);
    const int result = ucl_nrv2d_decompress_safe_8(encoded.data(), static_cast<ucl_uint>(encoded.size()), output.data(), &size, nullptr);
    if (result != UCL_E_OK || size != expected_size) throw std::runtime_error("UCL NRV policy decoder failed");
    return output;
  }
  if (omission.empty() || (method == "FastLZ" && omission == "match_policy") ||
      (method == "Yappy" && omission == "match_policy") ||
      (method == "LZG" && omission == "match_parse_policy") ||
      (method == "YALZ77" && omission == "match_parse_policy") ||
      (method == "UCL NRV" && omission == "match_parse_policy"))
    return decode_upstream(method, encoded, expected_size);
  const std::string magic = method == "BriefLZ" ? (omission == "lz_match" ? "BLZL" : "BLZB") :
                            method == "Crush" ? (omission == "lz_match" ? "CRSL" : "CRST") :
                            method == "FastLZ" ? (omission == "lz_match" ? "FLLT" : "FLTK") :
                            method == "LZMAT" ? (omission == "lz_match" ? "LMLT" : omission == "match_policy" ? "LMGP" : "LMTK") :
                            method == "LZRW" ? (omission == "lz_match" ? "LRLT" : omission == "window_model" ? "LRWM" : "LRTK") :
                            method == "WFLZ" ? (omission == "lz_match" ? "WFLT" : omission == "window_model" ? "WFWM" : "WFTK") :
                            method == "Yappy" ? (omission == "lz_match" ? "YPLT" : "YPTK") :
                            method == "LZG" ? (omission == "lz_match" ? "LZGL" : omission == "match_window_model" ? "LZGW" : "LZGT") :
                            method == "YALZ77" ? (omission == "lz_match" ? "Y7LT" : "Y7TK") :
                            method == "UCL NRV" ? (omission == "lz_match" ? "UCLL" : "UCLB") :
                            method == "libdeflate" ? (omission == "lz_match" ? "LDFL" : "LDFH") :
                            method == "Lizard" ? (omission == "lz_match" ? "LZDL" : omission == "match_window_model" ? "LZDW" : "LZDT") :
                            method == "QuickLZ" ? (omission == "lz_match" ? "QLOL" : omission == "match_window_model" ? "QLOW" : "QLOT") :
                            method == "zlib-ng" ? (omission == "lz_match" ? "ZNGL" : "ZNGH") :
                            method == "SLZ" ? (omission == "lz_match" ? "SLZL" : "SLZH") :
                            method == "ZXC" ? (omission == "lz_match" ? "ZXCL" : "ZXCT") :
                            method == "brotli" ? (omission == "lz_match" ? "BRTL" : "BRTH") :
                            (omission == "lz_match" ? "LSEL" : "LSEH");
  return decode_replacement(encoded, expected_size, magic);
}

struct Row { std::string method; std::string omission; std::size_t original = 0; std::size_t compressed = 0; std::int64_t comp_ns = 0; std::int64_t decomp_ns = 0; };

std::string catalog_operator(const std::string& omission) {
  if (omission == "match_policy") return "match_parse_policy";
  if (omission == "window_model") return "match_window_model";
  if (omission == "match_tokens") return "match_token_format";
  if (omission == "huffman_code") return "entropy_huffman";
  if (omission == "block_state_control") return "block_state_control";
  if (omission == "container_frame") return "container_frame";
  if (omission == "integer_code") return "integer_code";
  return omission;
}

Row run(const std::string& method, const std::string& omission, const Dataset& dataset) {
  Row row{method, omission};
  for (const auto& column : dataset.columns) {
    const auto c0 = std::chrono::steady_clock::now();
    const Bytes encoded = encode(method, column, omission);
    const auto c1 = std::chrono::steady_clock::now();
    const auto d0 = std::chrono::steady_clock::now();
    const Bytes decoded = decode(method, encoded, column.size(), omission);
    const auto d1 = std::chrono::steady_clock::now();
    if (decoded != column) throw std::runtime_error(method + " lossless check failed: " + omission);
    row.original += column.size();
    row.compressed += encoded.size();
    row.comp_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count();
    row.decomp_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(d1 - d0).count();
  }
  return row;
}

void control_test() {
  const std::vector<Bytes> controls = {{}, Bytes{'a'}, Bytes{'a','b','c','a','b','c','a','b','c'}, Bytes(4096, 'x')};
  const std::vector<std::pair<std::string, std::vector<std::string>>> methods = {
      {"BriefLZ", {"", "lz_match", "bitstream_pack"}}, {"Crush", {"", "lz_match", "match_tokens"}},
      {"FastLZ", {"", "lz_match", "match_policy", "match_tokens"}}, {"LZMAT", {"", "lz_match", "match_policy", "match_tokens"}},
      {"LZRW", {"", "lz_match", "window_model", "match_tokens"}}, {"WFLZ", {"", "lz_match", "window_model", "match_tokens"}},
      {"Yappy", {"", "lz_match", "match_policy", "match_tokens"}},
      {"LZG", {"", "match_parse_policy", "lz_match", "match_token_format", "match_window_model"}},
      {"YALZ77", {"", "lz_match", "match_parse_policy", "match_token_format"}},
      {"UCL NRV", {"", "lz_match", "bitstream_pack", "match_parse_policy"}},
      {"libdeflate", {"", "lz_match", "huffman_code"}},
      {"Lizard", {"", "lz_match", "match_window_model", "match_token_format", "integer_code"}},
      {"QuickLZ", {"", "lz_match", "match_window_model", "match_token_format", "container_frame"}},
      {"zlib-ng", {"", "lz_match", "match_parse_policy", "huffman_code", "container_frame"}},
      {"SLZ", {"", "lz_match", "huffman_code", "block_state_control", "container_frame"}},
      {"ZXC", {"", "lz_match", "match_window_model", "match_token_format"}},
      {"brotli", {"", "lz_match", "dictionary", "huffman_code"}},
      {"LZFSE", {"", "lz_match", "match_token_format", "entropy_fse", "literal_entropy"}}};
  for (const auto& method : methods) for (const auto& input : controls) for (const auto& omission : method.second)
    if (decode(method.first, encode(method.first, input, omission), input.size(), omission) != input)
      throw std::runtime_error("control sequence validation failed: " + method.first + ":" + omission);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: lzbench_easy_source_air_bench DATASET.csv OUTPUT.csv [METHODS]\n";
    return 2;
  }
  try {
    control_test();
    const Dataset dataset = load_dataset(argv[1]);
    const std::vector<std::pair<std::string, std::vector<std::string>>> methods = {
        {"BriefLZ", {"", "lz_match", "bitstream_pack"}}, {"Crush", {"", "lz_match", "match_tokens"}},
        {"FastLZ", {"", "lz_match", "match_policy", "match_tokens"}}, {"LZMAT", {"", "lz_match", "match_policy", "match_tokens"}},
        {"LZRW", {"", "lz_match", "window_model", "match_tokens"}}, {"WFLZ", {"", "lz_match", "window_model", "match_tokens"}},
        {"Yappy", {"", "lz_match", "match_policy", "match_tokens"}},
        {"LZG", {"", "match_parse_policy", "lz_match", "match_token_format", "match_window_model"}},
        {"YALZ77", {"", "lz_match", "match_parse_policy", "match_token_format"}},
        {"UCL NRV", {"", "lz_match", "bitstream_pack", "match_parse_policy"}},
        {"libdeflate", {"", "lz_match", "huffman_code"}},
        {"Lizard", {"", "lz_match", "match_window_model", "match_token_format", "integer_code"}},
        {"QuickLZ", {"", "lz_match", "match_window_model", "match_token_format", "container_frame"}},
        {"zlib-ng", {"", "lz_match", "match_parse_policy", "huffman_code", "container_frame"}},
        {"SLZ", {"", "lz_match", "huffman_code", "block_state_control", "container_frame"}},
        {"ZXC", {"", "lz_match", "match_window_model", "match_token_format"}},
        {"brotli", {"", "lz_match", "dictionary", "huffman_code"}},
        {"LZFSE", {"", "lz_match", "match_token_format", "entropy_fse", "literal_entropy"}}};
    std::vector<std::pair<std::string, std::vector<std::string>>> selected_methods;
    if (argc == 4) {
      const std::string requested = argv[3];
      for (const auto& method : methods) {
        std::size_t begin = 0;
        bool selected = false;
        while (begin <= requested.size()) {
          const std::size_t end = requested.find(',', begin);
          const std::size_t length = end == std::string::npos ? end : end - begin;
          if (requested.substr(begin, length) == method.first) {
            selected = true;
            break;
          }
          if (end == std::string::npos) break;
          begin = end + 1;
        }
        if (selected) selected_methods.push_back(method);
      }
      if (selected_methods.empty()) throw std::runtime_error("no requested methods selected");
    } else {
      selected_methods = methods;
    }
    std::ofstream output(argv[2]);
    output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
    for (const auto& method : selected_methods) {
      const Row baseline = run(method.first, "", dataset);
      const double baseline_rate = static_cast<double>(baseline.compressed) / baseline.original;
      for (const auto& omission : method.second) {
        const Row row = omission.empty() ? baseline : run(method.first, omission, dataset);
        const double rate = static_cast<double>(row.compressed) / row.original;
        output << "Air-Quality," << row.method << ',' << (row.omission.empty() ? "none" : catalog_operator(row.omission))
               << ',' << row.original << ',' << row.compressed << ',' << std::fixed << std::setprecision(9)
               << rate << ',' << baseline_rate << ',' << (rate - baseline_rate) << ','
               << row.comp_ns << ',' << row.decomp_ns << ",1," << dataset.integer_columns << ','
               << dataset.float_columns << ',' << dataset.timestamp_columns << ',' << dataset.columns.size() << ','
               << dataset.rows << ",source_upstream,source_method_specific_operator_ablation\n";
        std::cerr << row.method << ',' << (row.omission.empty() ? "none" : row.omission)
                  << " rate=" << rate << " delta=" << (rate - baseline_rate) << '\n';
      }
    }
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
  return 0;
}
