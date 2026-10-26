#include "vendored_lightweight_codecs_v1.hpp"
#include "vendored_entropy_codecs_v1.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>

extern "C" {
#include "brieflz.h"
#include "compressor.h"
#include "decompressor.h"
#include "fastlz.h"
#include "libdeflate.h"
#include "libbz3.h"
#include "lzf.h"
#include "lzfse.h"
#include "lzg.h"
#include "lzjb2010.h"
#include "lzmat.h"
#include "lzrw.h"
#include "quicklz151b7.h"
#include "slz.h"
#include "wfLZ.h"
#include "zlib.h"
#include "lzvn.h"
}
#include "lzav.h"
#include "lz77.h"
#include "yappy.hpp"

namespace native_baseline86_v1::vendored_lightweight {
namespace {

[[noreturn]] void fail(const std::string& label, const char* detail) {
  throw std::invalid_argument(label + ": " + detail);
}

void require(bool condition, const std::string& label, const char* detail) {
  if (!condition) fail(label, detail);
}

int as_int(std::size_t value, const std::string& label) {
  require(value <= static_cast<std::size_t>(INT_MAX), label,
          "input exceeds upstream int range");
  return static_cast<int>(value);
}

unsigned int as_uint(std::size_t value, const std::string& label) {
  require(value <= static_cast<std::size_t>(UINT_MAX), label,
          "input exceeds upstream unsigned range");
  return static_cast<unsigned int>(value);
}

std::uint32_t as_u32(std::size_t value, const std::string& label) {
  require(value <= static_cast<std::size_t>(UINT32_MAX), label,
          "input exceeds upstream uint32 range");
  return static_cast<std::uint32_t>(value);
}

Bytes raw_fallback(std::span<const std::uint8_t> input) {
  Bytes result;
  result.reserve(input.size() + 1);
  result.push_back(0);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

Bytes compressed_or_raw(std::span<const std::uint8_t> input, Bytes compressed,
                        std::size_t compressed_size) {
  if (compressed_size == 0 || compressed_size >= input.size())
    return raw_fallback(input);
  compressed.resize(compressed_size + 1);
  std::memmove(compressed.data() + 1, compressed.data(), compressed_size);
  compressed[0] = 1;
  return compressed;
}

std::span<const std::uint8_t> compressed_body(
    const std::string& label, std::span<const std::uint8_t> payload,
    std::size_t expected_size, Bytes& raw_result) {
  require(!payload.empty(), label, "empty adapter payload");
  if (payload[0] == 0) {
    require(payload.size() == expected_size + 1, label,
            "verbatim fallback length mismatch");
    raw_result.assign(payload.begin() + 1, payload.end());
    return {};
  }
  require(payload[0] == 1, label, "unknown adapter payload tag");
  require(payload.size() > 1, label, "empty upstream compressed stream");
  return payload.subspan(1);
}

Bytes encode_libdeflate(std::span<const std::uint8_t> input) {
  constexpr const char* label = "Libdeflate";
  auto* compressor = libdeflate_alloc_compressor(6);
  require(compressor != nullptr, label, "compressor allocation failed");
  Bytes output(libdeflate_deflate_compress_bound(compressor, input.size()));
  const auto size = libdeflate_deflate_compress(
      compressor, input.data(), input.size(), output.data(), output.size());
  libdeflate_free_compressor(compressor);
  return compressed_or_raw(input, std::move(output), size);
}

Bytes decode_libdeflate(std::span<const std::uint8_t> input,
                        std::size_t expected) {
  constexpr const char* label = "Libdeflate";
  auto* decompressor = libdeflate_alloc_decompressor();
  require(decompressor != nullptr, label, "decompressor allocation failed");
  Bytes output(expected);
  std::size_t consumed = 0, produced = 0;
  const auto status = libdeflate_deflate_decompress_ex(
      decompressor, input.data(), input.size(), output.data(), output.size(),
      &consumed, &produced);
  libdeflate_free_decompressor(decompressor);
  require(status == LIBDEFLATE_SUCCESS && consumed == input.size() &&
              produced == expected,
          label, "invalid, trailing, or wrong-sized deflate stream");
  return output;
}

Bytes encode_fastlz(std::span<const std::uint8_t> input) {
  constexpr const char* label = "FastLZ";
  if (input.size() < 16) return raw_fallback(input);
  const int size = as_int(input.size(), label);
  Bytes output(std::max<std::size_t>(66, input.size() + input.size() / 20 + 16));
  const int written = fastlz_compress_level(1, input.data(), size, output.data());
  require(written > 0 && static_cast<std::size_t>(written) <= output.size(), label,
          "compressor failed or exceeded its documented bound");
  return compressed_or_raw(input, std::move(output),
                           static_cast<std::size_t>(written));
}

Bytes decode_fastlz(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "FastLZ";
  Bytes output(expected);
  const int produced = fastlz_decompress(input.data(), as_int(input.size(), label),
                                         output.data(), as_int(expected, label));
  require(produced >= 0 && static_cast<std::size_t>(produced) == expected, label,
          "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzf(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZF";
  const auto in_size = as_uint(input.size(), label);
  Bytes output(input.size() + input.size() / 16 + 64 + 3);
  const auto written = lzf_compress_very(input.data(), in_size, output.data(),
                                         as_uint(output.size(), label));
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_lzf(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "LZF";
  Bytes output(expected);
  const auto produced = lzf_decompress(input.data(), as_uint(input.size(), label),
                                       output.data(), as_uint(expected, label));
  require(produced == expected, label, "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzfse(std::span<const std::uint8_t> input, bool lzvn) {
  const std::string label = lzvn ? "LZVN" : "LZFSE";
  Bytes output(input.size() * 2 + 4096);
  const auto scratch_size =
      lzvn ? lzvn_encode_scratch_size() : lzfse_encode_scratch_size();
  Bytes scratch(scratch_size);
  const auto written = lzvn
                           ? lzvn_encode_buffer(output.data(), output.size(),
                                                input.data(), input.size(),
                                                scratch.data())
                           : lzfse_encode_buffer(output.data(), output.size(),
                                                 input.data(), input.size(),
                                                 scratch.data());
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_lzfse(std::span<const std::uint8_t> input, std::size_t expected,
                   bool lzvn) {
  const std::string label = lzvn ? "LZVN" : "LZFSE";
  Bytes output(expected + 1);
  const auto scratch_size =
      lzvn ? lzvn_decode_scratch_size() : lzfse_decode_scratch_size();
  Bytes scratch(scratch_size);
  const auto produced = lzvn
                            ? lzvn_decode_buffer_scratch(
                                  output.data(), output.size(), input.data(),
                                  input.size(), scratch.data())
                            : lzfse_decode_buffer(output.data(), output.size(),
                                                  input.data(), input.size(),
                                                  scratch.data());
  require(produced == expected, label, "invalid or wrong-sized stream");
  output.resize(expected);
  return output;
}

Bytes encode_brieflz(std::span<const std::uint8_t> input) {
  constexpr const char* label = "BriefLZ";
  constexpr int level = 1;
  require(input.size() <= std::numeric_limits<unsigned long>::max(), label,
          "input exceeds upstream unsigned long range");
  Bytes output(blz_max_packed_size(input.size()));
  Bytes work(blz_workmem_size_level(input.size(), level));
  require(!work.empty(), label, "work memory allocation size is zero");
  const auto written = blz_pack_level(input.data(), output.data(),
                                      static_cast<unsigned long>(input.size()),
                                      work.data(), level);
  require(written <= output.size(), label, "compressor exceeded bound");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_brieflz(std::span<const std::uint8_t> input,
                     std::size_t expected) {
  constexpr const char* label = "BriefLZ";
  require(input.size() <= std::numeric_limits<unsigned long>::max() &&
              expected <= std::numeric_limits<unsigned long>::max(),
          label, "stream exceeds upstream unsigned long range");
  Bytes output(expected);
  const auto produced = blz_depack_safe(
      input.data(), static_cast<unsigned long>(input.size()), output.data(),
      static_cast<unsigned long>(expected));
  require(produced != BLZ_ERROR && produced == expected, label,
          "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzav(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZAV";
  const int input_size = as_int(input.size(), label);
  Bytes output(static_cast<std::size_t>(lzav_compress_bound(input_size)));
  const int written = lzav_compress_default(
      input.data(), output.data(), input_size, as_int(output.size(), label));
  require(written >= 0, label, "compressor returned a negative size");
  return compressed_or_raw(input, std::move(output),
                           static_cast<std::size_t>(written));
}

Bytes decode_lzav(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "LZAV";
  Bytes output(expected);
  const int produced = lzav_decompress(input.data(), output.data(),
                                       as_int(input.size(), label),
                                       as_int(expected, label));
  require(produced >= 0 && static_cast<std::size_t>(produced) == expected, label,
          "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzg(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZG";
  const auto input_size = as_u32(input.size(), label);
  Bytes output(LZG_MaxEncodedSize(input_size));
  lzg_encoder_config_t config;
  LZG_InitEncoderConfig(&config);
  const auto written = LZG_Encode(input.data(), input_size, output.data(),
                                  as_u32(output.size(), label), &config);
  require(written != 0, label, "compressor failed");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_lzg(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "LZG";
  require(LZG_DecodedSize(input.data(), as_u32(input.size(), label)) == expected,
          label, "header decoded size mismatch");
  Bytes output(expected);
  const auto produced = LZG_Decode(input.data(), as_u32(input.size(), label),
                                   output.data(), as_u32(expected, label));
  require(produced == expected, label, "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzjb(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZJB";
  Bytes output(input.size() + input.size() / 8 + 32);
  const auto written = lzjb_compress2010(
      const_cast<unsigned char*>(input.data()), output.data(), input.size(),
      output.size(), 0);
  require(written <= output.size(), label, "compressor exceeded output bound");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_lzjb(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "LZJB";
  // The 2010 decoder predates bounded-source APIs.  A padded private copy
  // keeps malformed input reads within allocated memory; NGF1 authenticates
  // the exact body before this function is entered.
  Bytes padded(input.begin(), input.end());
  padded.resize(input.size() + 16, 0);
  Bytes output(expected);
  const auto produced = lzjb_decompress2010(padded.data(), output.data(),
                                            input.size(), expected, 0);
  require(produced == expected, label, "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzmat(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZMAT";
  const auto input_size = as_u32(input.size(), label);
  Bytes output(MAX_LZMAT_ENCODED_SIZE(input_size));
  auto output_size = as_u32(output.size(), label);
  const int status = lzmat_encode(output.data(), &output_size,
                                  const_cast<std::uint8_t*>(input.data()),
                                  input_size);
  require(status == LZMAT_STATUS_OK, label, "compressor failed");
  return compressed_or_raw(input, std::move(output), output_size);
}

Bytes decode_lzmat(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "LZMAT";
  Bytes output(expected);
  auto output_size = as_u32(expected, label);
  const int status = lzmat_decode(output.data(), &output_size,
                                  const_cast<std::uint8_t*>(input.data()),
                                  as_u32(input.size(), label));
  require(status == LZMAT_STATUS_OK && output_size == expected, label,
          "invalid or wrong-sized stream");
  return output;
}

Bytes encode_lzrw(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZRW";
  if (input.size() < 32) return raw_fallback(input);
  Bytes output(input.size() + COMPRESS_OVERRUN);
  std::uint32_t written = 0;
  lzrw1_compress(const_cast<std::uint8_t*>(input.data()),
                  as_u32(input.size(), label), output.data(), &written);
  require(written <= output.size(), label, "compressor exceeded output bound");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_lzrw(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "LZRW";
  Bytes output(expected);
  std::uint32_t written = 0;
  lzrw1_decompress(const_cast<std::uint8_t*>(input.data()),
                    as_u32(input.size(), label), output.data(), &written);
  require(written == expected, label, "invalid or wrong-sized stream");
  return output;
}

Bytes encode_wflz(std::span<const std::uint8_t> input) {
  constexpr const char* label = "WFLZ";
  const auto input_size = as_u32(input.size(), label);
  Bytes output(wfLZ_GetMaxCompressedSize(input_size));
  Bytes work(wfLZ_GetWorkMemSize());
  require(!work.empty(), label, "work memory allocation size is zero");
  const auto written = wfLZ_CompressFast(input.data(), input_size,
                                          output.data(), work.data(), 0);
  require(written != 0 && written <= output.size(), label,
          "compressor failed or exceeded output bound");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_wflz(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "WFLZ";
  constexpr std::size_t header_size = 16;
  require(input.size() >= header_size, label, "truncated stream header");
  require(input.size() >= wfLZ_GetHeaderSize(input.data()), label,
          "truncated stream header");
  require(wfLZ_GetCompressedSize(input.data()) == input.size() &&
              wfLZ_GetDecompressedSize(input.data()) == expected,
          label, "header size mismatch");
  Bytes output(expected);
  wfLZ_Decompress(input.data(), output.data());
  return output;
}

void ensure_yappy_tables() {
  static const bool initialized = [] { YappyFillTables(); return true; }();
  (void)initialized;
}

Bytes encode_yappy(std::span<const std::uint8_t> input) {
  constexpr const char* label = "Yappy";
  ensure_yappy_tables();
  // Yappy's vectorized implementation reads complete 16-byte lanes and can
  // write a final lane past the logical end.  Keep that behavior inside
  // padded private buffers.
  Bytes padded(input.begin(), input.end());
  padded.resize(input.size() + 32, 0);
  Bytes output(input.size() * 2 + 64);
  auto* end = YappyCompress(padded.data(), output.data(), input.size(), 10);
  require(end >= output.data() && end <= output.data() + output.size(), label,
          "compressor exceeded output bound");
  const auto written = static_cast<std::size_t>(end - output.data());
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_yappy(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "Yappy";
  ensure_yappy_tables();
  Bytes padded(input.begin(), input.end());
  padded.resize(input.size() + 32, 0);
  Bytes output(expected + 32);
  auto* end = YappyUnCompress(padded.data(), padded.data() + input.size(),
                              output.data());
  require(end >= output.data() &&
              static_cast<std::size_t>(end - output.data()) == expected,
          label, "invalid or wrong-sized stream");
  output.resize(expected);
  return output;
}

Bytes encode_yalz77(std::span<const std::uint8_t> input) {
  constexpr const char* label = "YALZ77";
  lz77::compress_t compressor(lz77::DEFAULT_SEARCHLEN,
                              lz77::DEFAULT_BLOCKSIZE);
  const auto encoded = compressor.feed(input.data(), input.data() + input.size());
  Bytes output(encoded.begin(), encoded.end());
  return compressed_or_raw(input, std::move(output), encoded.size());
}

Bytes decode_yalz77(std::span<const std::uint8_t> input,
                    std::size_t expected) {
  constexpr const char* label = "YALZ77";
  lz77::decompress_t decompressor(expected);
  std::string remaining;
  require(decompressor.feed(input.data(), input.data() + input.size(), remaining),
          label, "truncated stream");
  require(remaining.empty(), label, "trailing stream bytes");
  const auto& decoded = decompressor.result();
  require(decoded.size() == expected, label, "wrong-sized stream");
  return Bytes(decoded.begin(), decoded.end());
}

Bytes encode_quicklz(std::span<const std::uint8_t> input) {
  constexpr const char* label = "QuickLZ";
  if (input.size() < 16) return raw_fallback(input);
  Bytes output(input.size() + 400);
  qlz_state_compress state{};
  const auto written = qlz151_compress_1(input.data(),
                                          reinterpret_cast<char*>(output.data()),
                                          input.size(), &state);
  require(written != 0 && written <= output.size(), label,
          "compressor failed or exceeded output bound");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_quicklz(std::span<const std::uint8_t> input,
                     std::size_t expected) {
  constexpr const char* label = "QuickLZ";
  require(input.size() >= 9, label, "truncated stream header");
  const auto* source = reinterpret_cast<const char*>(input.data());
  require(qlz151_size_compressed_1(source) == input.size() &&
              qlz151_size_decompressed_1(source) == expected,
          label, "header size mismatch");
  Bytes output(expected);
  qlz_state_decompress state{};
  const auto written = qlz151_decompress_1(source, output.data(), &state);
  require(written == expected, label, "invalid or wrong-sized stream");
  return output;
}

Bytes encode_slz(std::span<const std::uint8_t> input) {
  constexpr const char* label = "SLZ";
  require(input.size() <= static_cast<std::size_t>(LONG_MAX), label,
          "input exceeds upstream long range");
  slz_stream stream{};
  require(slz_init(&stream, 1, SLZ_FMT_GZIP) == 0, label,
          "compressor initialization failed");
  Bytes output(input.size() * 2 + 1024);
  const auto first = slz_encode(&stream, output.data(), input.data(),
                                static_cast<long>(input.size()), 0);
  require(first >= 0 && static_cast<std::size_t>(first) < output.size(), label,
          "compressor failed or exceeded output bound");
  const auto tail = slz_finish(&stream, output.data() + first);
  require(tail >= 0 && static_cast<std::size_t>(first + tail) <= output.size(),
          label, "compressor trailer exceeded output bound");
  return compressed_or_raw(input, std::move(output),
                           static_cast<std::size_t>(first + tail));
}

Bytes decode_slz(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "SLZ";
  require(input.size() <= UINT_MAX && expected < UINT_MAX, label,
          "stream exceeds zlib one-shot range");
  z_stream stream{};
  require(inflateInit2(&stream, 31) == Z_OK, label,
          "decompressor initialization failed");
  Bytes output(expected + 1);
  stream.next_in = const_cast<Bytef*>(input.data());
  stream.avail_in = static_cast<uInt>(input.size());
  stream.next_out = output.data();
  stream.avail_out = static_cast<uInt>(output.size());
  const int status = inflate(&stream, Z_FINISH);
  const auto produced = stream.total_out;
  const auto remaining = stream.avail_in;
  inflateEnd(&stream);
  require(status == Z_STREAM_END && produced == expected && remaining == 0,
          label, "invalid, trailing, or wrong-sized stream");
  output.resize(expected);
  return output;
}

Bytes encode_tamp(std::span<const std::uint8_t> input) {
  constexpr const char* label = "TAMP";
  constexpr std::uint8_t window_bits = 10;
  TampConf config{};
  config.window = window_bits;
  config.literal = 8;
  Bytes window(1U << window_bits);
  TampCompressor compressor{};
  require(tamp_compressor_init(&compressor, &config, window.data()) == TAMP_OK,
          label, "compressor initialization failed");
  Bytes output(input.size() * 2 + 64);
  std::size_t written = 0, consumed = 0;
  const auto status = tamp_compressor_compress_and_flush(
      &compressor, output.data(), output.size(), &written, input.data(),
      input.size(), &consumed, false);
  require(status == TAMP_OK && consumed == input.size(), label,
          "compressor did not consume the complete input");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes encode_bzip3(std::span<const std::uint8_t> input) {
  constexpr const char* label = "Bzip3";
  constexpr std::uint32_t block_size = 1U << 20;
  Bytes output(bz3_bound(input.size()));
  std::size_t written = output.size();
  require(bz3_compress(block_size, input.data(), output.data(), input.size(),
                       &written) == BZ3_OK,
          label, "compressor failed");
  require(written <= output.size(), label, "compressor exceeded output bound");
  return compressed_or_raw(input, std::move(output), written);
}

Bytes decode_bzip3(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "Bzip3";
  Bytes output(expected);
  std::size_t produced = output.size();
  require(bz3_decompress(input.data(), output.data(), input.size(), &produced) ==
              BZ3_OK &&
              produced == expected,
          label, "invalid or wrong-sized stream");
  return output;
}

Bytes encode_entropy(const std::string& label,
                     std::span<const std::uint8_t> input) {
  auto output = vendored_entropy::encode(label, input);
  const auto size = output.size();
  return compressed_or_raw(input, std::move(output), size);
}

Bytes decode_tamp(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "TAMP";
  constexpr std::uint8_t window_bits = 10;
  Bytes window(1U << window_bits);
  TampDecompressor decompressor{};
  require(tamp_decompressor_init(&decompressor, nullptr, window.data(),
                                 window_bits) == TAMP_OK,
          label, "decompressor initialization failed");
  Bytes output(expected + 1);
  std::size_t written = 0, consumed = 0;
  const auto status = tamp_decompressor_decompress(
      &decompressor, output.data(), output.size(), &written, input.data(),
      input.size(), &consumed);
  require((status == TAMP_INPUT_EXHAUSTED || status == TAMP_OK) &&
              consumed == input.size() && written == expected,
          label, "invalid, trailing, or wrong-sized stream");
  output.resize(expected);
  return output;
}

}  // namespace

Bytes encode(const std::string& label, std::span<const std::uint8_t> input) {
  if (label == "Libdeflate") return encode_libdeflate(input);
  if (label == "FastLZ") return encode_fastlz(input);
  if (label == "LZF") return encode_lzf(input);
  if (label == "LZFSE") return encode_lzfse(input, false);
  if (label == "LZVN") return encode_lzfse(input, true);
  if (label == "BriefLZ") return encode_brieflz(input);
  if (label == "LZAV") return encode_lzav(input);
  if (label == "LZG") return encode_lzg(input);
  if (label == "LZJB") return encode_lzjb(input);
  if (label == "LZMAT") return encode_lzmat(input);
  if (label == "LZRW") return encode_lzrw(input);
  if (label == "WFLZ") return encode_wflz(input);
  if (label == "Yappy") return encode_yappy(input);
  if (label == "YALZ77") return encode_yalz77(input);
  if (label == "QuickLZ") return encode_quicklz(input);
  if (label == "SLZ") return encode_slz(input);
  if (label == "TAMP") return encode_tamp(input);
  if (label == "Bzip3") return encode_bzip3(input);
  if (label == "FSE" || label == "Huff0") return encode_entropy(label, input);
  throw std::invalid_argument(label + ": unknown vendored lightweight codec");
}

Bytes decode(const std::string& label, std::span<const std::uint8_t> payload,
             std::size_t expected_size) {
  Bytes raw;
  const auto body = compressed_body(label, payload, expected_size, raw);
  if (payload[0] == 0) return raw;
  if (label == "Libdeflate") return decode_libdeflate(body, expected_size);
  if (label == "FastLZ") return decode_fastlz(body, expected_size);
  if (label == "LZF") return decode_lzf(body, expected_size);
  if (label == "LZFSE") return decode_lzfse(body, expected_size, false);
  if (label == "LZVN") return decode_lzfse(body, expected_size, true);
  if (label == "BriefLZ") return decode_brieflz(body, expected_size);
  if (label == "LZAV") return decode_lzav(body, expected_size);
  if (label == "LZG") return decode_lzg(body, expected_size);
  if (label == "LZJB") return decode_lzjb(body, expected_size);
  if (label == "LZMAT") return decode_lzmat(body, expected_size);
  if (label == "LZRW") return decode_lzrw(body, expected_size);
  if (label == "WFLZ") return decode_wflz(body, expected_size);
  if (label == "Yappy") return decode_yappy(body, expected_size);
  if (label == "YALZ77") return decode_yalz77(body, expected_size);
  if (label == "QuickLZ") return decode_quicklz(body, expected_size);
  if (label == "SLZ") return decode_slz(body, expected_size);
  if (label == "TAMP") return decode_tamp(body, expected_size);
  if (label == "Bzip3") return decode_bzip3(body, expected_size);
  if (label == "FSE" || label == "Huff0")
    return vendored_entropy::decode(label, body, expected_size);
  throw std::invalid_argument(label + ": unknown vendored lightweight codec");
}

}  // namespace native_baseline86_v1::vendored_lightweight
