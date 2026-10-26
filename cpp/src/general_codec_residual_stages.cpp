#include "general_codec_residual_stages.hpp"

#include "blosc2_codec.hpp"
#include "blosc_codec.hpp"
#include "bzip2_operator_ablation.hpp"
#include "deflate_container_stages.hpp"
#include "general_codec_reference_stages.hpp"
#include "parameterized_deflate.hpp"
#include "optimal_rle.hpp"
#include "xz_container_stage.hpp"

#include <brotli/decode.h>
#include <brotli/encode.h>
#include <bzlib.h>
#include <lzma.h>
#include <zlib.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace exact_actions::general_residual {
namespace {

using Bytes = std::vector<std::uint8_t>;
namespace ref = reference_actions::general;

Bytes require_nonempty(std::span<const std::uint8_t> raw, const char* codec) {
  if (raw.empty()) throw std::invalid_argument(std::string(codec) + " does not accept empty action input");
  return {raw.begin(), raw.end()};
}

Bytes brotli_prefix_encode(std::span<const std::uint8_t> raw) {
  return ref::huffman_forward(ref::lz_forward(ref::dictionary_forward(raw),
                                               65535, 4, 258, true));
}

Bytes brotli_prefix_decode(std::span<const std::uint8_t> prefix) {
  return ref::dictionary_inverse(ref::lz_inverse(ref::huffman_inverse(prefix)));
}

Bytes zstd_prefix_encode(std::span<const std::uint8_t> raw) {
  return ref::rans_forward(ref::huffman_forward(ref::parse_policy_forward(
      ref::lz_forward(raw, 65535, 3, 258, true))));
}

Bytes zstd_prefix_decode(std::span<const std::uint8_t> prefix) {
  return ref::lz_inverse(ref::parse_policy_inverse(
      ref::huffman_inverse(ref::rans_inverse(prefix))));
}

Bytes blosc_prefix_encode(std::span<const std::uint8_t> raw,
                          int type_size) {
  return ref::token_forward(
      ref::lz_forward(ref::shuffle_forward(raw, type_size), 65535, 4, 258, false));
}

Bytes blosc_prefix_decode(std::span<const std::uint8_t> prefix) {
  return ref::shuffle_inverse(ref::lz_inverse(ref::token_inverse(prefix)));
}

Bytes bzip2_prefix_encode(std::span<const std::uint8_t> raw) {
  return ref::huffman_forward(
      ref::rle_forward(ref::mtf_forward(ref::bwt_forward(raw))));
}

Bytes bzip2_prefix_decode(std::span<const std::uint8_t> prefix) {
  return ref::bwt_inverse(
      ref::mtf_inverse(ref::rle_inverse(ref::huffman_inverse(prefix))));
}

Bytes deflate_prefix_encode(std::span<const std::uint8_t> raw, bool zlib_kind) {
  auto lz = ref::lz_forward(raw, 32768, 3, 258, true);
  if (zlib_kind) lz = ref::parse_policy_forward(lz);
  return ref::huffman_forward(lz);
}

Bytes deflate_prefix_decode(std::span<const std::uint8_t> prefix,
                            bool zlib_kind) {
  auto lz = ref::huffman_inverse(prefix);
  if (zlib_kind) lz = ref::parse_policy_inverse(lz);
  return ref::lz_inverse(lz);
}

Bytes xz_prefix_encode(std::span<const std::uint8_t> raw) {
  return ref::range_forward(ref::parse_policy_forward(
      ref::lz_forward(raw, 16777216, 3, 273, true)));
}

Bytes xz_prefix_decode(std::span<const std::uint8_t> prefix) {
  return ref::lz_inverse(
      ref::parse_policy_inverse(ref::range_inverse(prefix)));
}

}  // namespace

Bytes brotli_encode(std::span<const std::uint8_t> raw) {
  std::size_t size = BrotliEncoderMaxCompressedSize(raw.size());
  Bytes output(size);
  if (!BrotliEncoderCompress(6, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_GENERIC,
                             raw.size(), raw.data(), &size, output.data())) {
    throw std::runtime_error("Brotli quality-6 compression failed");
  }
  output.resize(size);
  return output;
}

Bytes brotli_decode(std::span<const std::uint8_t> wire) {
  BrotliDecoderState* state = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
  if (state == nullptr) throw std::runtime_error("Brotli decoder allocation failed");
  const std::uint8_t* next_in = wire.data();
  std::size_t avail_in = wire.size();
  Bytes output;
  std::array<std::uint8_t, 65536> chunk{};
  for (;;) {
    std::uint8_t* next_out = chunk.data();
    std::size_t avail_out = chunk.size();
    const auto status = BrotliDecoderDecompressStream(
        state, &avail_in, &next_in, &avail_out, &next_out, nullptr);
    output.insert(output.end(), chunk.begin(),
                  chunk.end() - static_cast<std::ptrdiff_t>(avail_out));
    if (status == BROTLI_DECODER_RESULT_SUCCESS) break;
    if (status != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
      BrotliDecoderDestroyInstance(state);
      throw std::runtime_error("Brotli decompression failed");
    }
  }
  BrotliDecoderDestroyInstance(state);
  return output;
}

Bytes zstd_encode(std::span<const std::uint8_t> raw) {
  Bytes output(ZSTD_compressBound(raw.size()));
  const std::size_t size =
      ZSTD_compress(output.data(), output.size(), raw.data(), raw.size(), 9);
  if (ZSTD_isError(size)) throw std::runtime_error(ZSTD_getErrorName(size));
  output.resize(size);
  return output;
}

Bytes zstd_decode(std::span<const std::uint8_t> wire) {
  const unsigned long long size = ZSTD_getFrameContentSize(wire.data(), wire.size());
  if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN ||
      size > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("Zstd frame has no usable content size");
  }
  Bytes output(static_cast<std::size_t>(size));
  const std::size_t decoded = ZSTD_decompress(
      output.data(), output.size(), wire.data(), wire.size());
  if (ZSTD_isError(decoded) || decoded != output.size()) {
    throw std::runtime_error("Zstd decompression failed");
  }
  return output;
}

Bytes blosc1_encode(std::span<const std::uint8_t> raw, int type_size) {
  return wc_blosc1_bench::encodeBufferExact(raw, type_size);
}

Bytes blosc1_decode(std::span<const std::uint8_t> wire) {
  return wc_blosc1_bench::decodeBufferExact(wire);
}

Bytes blosc2_encode(std::span<const std::uint8_t> raw, int type_size) {
  return wc_blosc2_bench::encodeBufferExact(raw, type_size);
}

Bytes blosc2_decode(std::span<const std::uint8_t> wire, int type_size) {
  return wc_blosc2_bench::decodeBufferExact(wire, type_size);
}

Bytes bzip2_encode(std::span<const std::uint8_t> raw) {
  return bzip2_operator_ablation::encode(raw, 4);
}

Bytes bzip2_decode(std::span<const std::uint8_t> wire) {
  if (wire.size() > std::numeric_limits<unsigned int>::max()) {
    throw std::runtime_error("Bzip2 payload too large");
  }
  std::size_t capacity = std::max<std::size_t>(65536, wire.size() * 4);
  for (int attempt = 0; attempt < 30; ++attempt) {
    if (capacity > std::numeric_limits<unsigned int>::max()) break;
    Bytes output(capacity);
    unsigned int output_size = static_cast<unsigned int>(capacity);
    unsigned int input_size = static_cast<unsigned int>(wire.size());
    const int status = BZ2_bzBuffToBuffDecompress(
        reinterpret_cast<char*>(output.data()), &output_size,
        const_cast<char*>(reinterpret_cast<const char*>(wire.data())), input_size,
        0, 0);
    if (status == BZ_OK) {
      output.resize(output_size);
      return output;
    }
    if (status != BZ_OUTBUFF_FULL) break;
    capacity *= 2;
  }
  throw std::runtime_error("Bzip2 decompression failed");
}

Bytes gzip_encode(std::span<const std::uint8_t> raw, int level) {
  return parameterized_deflate::encode(raw, level, parameterized_deflate::Wrapper::gzip);
}

Bytes gzip_decode(std::span<const std::uint8_t> wire) {
  return parameterized_deflate::decode(wire, parameterized_deflate::Wrapper::gzip);
}

Bytes zlib_encode(std::span<const std::uint8_t> raw, int level) {
  return parameterized_deflate::encode(raw, level, parameterized_deflate::Wrapper::zlib);
}

Bytes zlib_decode(std::span<const std::uint8_t> wire) {
  return parameterized_deflate::decode(wire, parameterized_deflate::Wrapper::zlib);
}

Bytes xz_encode(std::span<const std::uint8_t> raw) {
  // Match the streaming encoder path used by Python's lzma.compress rather
  // than lzma_easy_buffer_encode.  The one-shot helper is allowed to record
  // block sizes in the header and can therefore produce a different, though
  // equivalent, XZ stream.
  lzma_stream stream = LZMA_STREAM_INIT;
  if (lzma_easy_encoder(&stream, 5, LZMA_CHECK_CRC64) != LZMA_OK) {
    throw std::runtime_error("XZ preset-5 encoder initialization failed");
  }
  stream.next_in = raw.data();
  stream.avail_in = raw.size();
  Bytes output;
  std::array<std::uint8_t, 65536> chunk{};
  for (;;) {
    stream.next_out = chunk.data();
    stream.avail_out = chunk.size();
    const lzma_ret status = lzma_code(&stream, LZMA_FINISH);
    output.insert(output.end(), chunk.begin(),
                  chunk.end() - static_cast<std::ptrdiff_t>(stream.avail_out));
    if (status == LZMA_STREAM_END) break;
    if (status != LZMA_OK) {
      lzma_end(&stream);
      throw std::runtime_error("XZ preset-5 compression failed");
    }
  }
  lzma_end(&stream);
  return output;
}

Bytes xz_decode(std::span<const std::uint8_t> wire) {
  std::size_t capacity = std::max<std::size_t>(65536, wire.size() * 4);
  for (int attempt = 0; attempt < 30; ++attempt) {
    Bytes output(capacity);
    std::uint64_t memlimit = std::numeric_limits<std::uint64_t>::max();
    std::size_t in_pos = 0;
    std::size_t out_pos = 0;
    const lzma_ret status = lzma_stream_buffer_decode(
        &memlimit, 0, nullptr, wire.data(), &in_pos, wire.size(), output.data(),
        &out_pos, output.size());
    if (status == LZMA_OK && in_pos == wire.size()) {
      output.resize(out_pos);
      return output;
    }
    if (status != LZMA_BUF_ERROR) break;
    capacity *= 2;
  }
  throw std::runtime_error("XZ decompression failed");
}

Bytes brotli_forward(std::span<const std::uint8_t> prefix) {
  return brotli_encode(brotli_prefix_decode(prefix));
}
Bytes brotli_inverse(std::span<const std::uint8_t> wire) {
  return brotli_prefix_encode(brotli_decode(wire));
}
Bytes zstd_forward(std::span<const std::uint8_t> prefix) {
  return zstd_encode(zstd_prefix_decode(prefix));
}
Bytes zstd_inverse(std::span<const std::uint8_t> wire) {
  return zstd_prefix_encode(zstd_decode(wire));
}
Bytes blosc1_forward(std::span<const std::uint8_t> prefix, int type_size) {
  return blosc1_encode(blosc_prefix_decode(prefix), type_size);
}
Bytes blosc1_inverse(std::span<const std::uint8_t> wire, int type_size) {
  return blosc_prefix_encode(blosc1_decode(wire), type_size);
}
Bytes blosc2_forward(std::span<const std::uint8_t> prefix, int type_size) {
  return blosc2_encode(blosc_prefix_decode(prefix), type_size);
}
Bytes blosc2_inverse(std::span<const std::uint8_t> wire, int type_size) {
  return blosc_prefix_encode(blosc2_decode(wire, type_size), type_size);
}
Bytes bzip2_forward(std::span<const std::uint8_t> prefix) {
  return bzip2_encode(bzip2_prefix_decode(prefix));
}
Bytes bzip2_inverse(std::span<const std::uint8_t> wire) {
  return bzip2_prefix_encode(bzip2_decode(wire));
}

Bytes bzip2_optimal_forward(std::span<const std::uint8_t> prefix) {
  const auto runs = ref::huffman_inverse(prefix);
  const auto symbols = unified_rle::decode(runs);
  if (unified_rle::symbol_width(runs) != 1)
    throw std::runtime_error("Bzip2 optimal RLE prefix requires uint8 symbols");
  return bzip2_encode(ref::bwt_inverse(ref::mtf_inverse(symbols)));
}

Bytes bzip2_optimal_inverse(std::span<const std::uint8_t> wire) {
  const auto symbols = ref::mtf_forward(ref::bwt_forward(bzip2_decode(wire)));
  return ref::huffman_forward(unified_rle::encode(symbols, 1));
}
Bytes gzip_forward(std::span<const std::uint8_t> prefix, int level) {
  parameterized_deflate::validate_level(level);
  return exact_actions::deflate_container::gzip_inverse(
      gzip_encode(deflate_prefix_decode(prefix, false), level));
}
Bytes gzip_inverse(std::span<const std::uint8_t> packet) {
  return deflate_prefix_encode(
      gzip_decode(exact_actions::deflate_container::gzip_forward(packet)), false);
}
Bytes zlib_forward(std::span<const std::uint8_t> prefix, int level) {
  parameterized_deflate::validate_level(level);
  return exact_actions::deflate_container::zlib_inverse(
      zlib_encode(deflate_prefix_decode(prefix, true), level));
}
Bytes zlib_inverse(std::span<const std::uint8_t> packet) {
  return deflate_prefix_encode(
      zlib_decode(exact_actions::deflate_container::zlib_forward(packet)), true);
}
Bytes xz_forward(std::span<const std::uint8_t> prefix) {
  return exact_actions::xz_container::inverse(xz_encode(xz_prefix_decode(prefix)));
}
Bytes xz_inverse(std::span<const std::uint8_t> packet) {
  return xz_prefix_encode(xz_decode(exact_actions::xz_container::forward(packet)));
}

}  // namespace exact_actions::general_residual
