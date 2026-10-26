#include "upstream_general_b.hpp"

#include "density_api.h"
#include "lz/lizard/lizard_compress.h"
#include "lz/lizard/lizard_decompress.h"
#include "lz/lzham/include/lzham.h"
#include "lz/lzlib/lzlib.h"
#include "lz/libzling/libzling.h"
#include "lz/tornado/tor_test.h"
#include "lz/zxc/include/zxc.h"
#include "misc/7-zip/Ppmd8.h"
#include "misc/kanzi-cpp/src/InputStream.hpp"
#include "misc/kanzi-cpp/src/OutputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedInputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedOutputStream.hpp"
#include "misc/kanzi-cpp/src/util/fixedbuf.hpp"

extern "C" {
#undef z_const
#undef Z_NULL
#define in_func baseline86_zlibng_in_func
#include "lz/zlib-ng/zlib-ng.h"
#undef in_func
}

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <string>

namespace native_baseline86_v1::upstream_general_b {
namespace {

extern "C" void density_lion_set_ablation_mode(int mode);

constexpr std::size_t kMaximumInput = 256ULL << 20;
constexpr std::uint8_t kRaw = 0;
constexpr std::uint8_t kUpstream = 1;

std::mutex density_mutex;
std::mutex tornado_mutex;

constexpr std::array<Descriptor, 10> kDescriptors{{
    {"Density", "density_c_0.14.2_lion", "external/operator_ablation_sources/density_c_0_14_2",
     "lion_default_single_thread_tagged_raw_fallback", "Legacy upstream C99 implementation; not the newer Rust rewrite."},
    {"Kanzi", "kanzi_cpp_level5", "external/compression/lzbench/misc/kanzi-cpp",
     "level5_TEXT_UTF_BWT_RANK_ZRLT_ANS0_block4MiB_single_thread_tagged_raw_fallback", "Memory stream API; one worker."},
    {"Lizard", "lizard_upstream", "external/compression/lzbench/lz/lizard",
     "level19_single_thread_tagged_raw_fallback", "Raw Lizard block API; exact expected length required for decode."},
    {"LZHAM", "lzham_upstream", "external/compression/lzbench/lz/lzham",
     "level2_dict20_single_thread_tagged_raw_fallback", "One-shot memory API with helper threads disabled."},
    {"Lzlib", "lzlib_1.15", "external/compression/lzbench/lz/lzlib",
     "level6_dict8MiB_match36_single_thread_tagged_raw_fallback", "Streaming library API driven entirely in process."},
    {"PPMD8", "7zip_ppmd8", "external/compression/lzbench/misc/7-zip",
     "order8_mem16MiB_restart_single_thread_tagged_raw_fallback", "Bounded in-memory range-coded PPMdI stream."},
    {"Tornado", "tornado_legacy", "external/compression/lzbench/lz/tornado",
     "method5_single_thread_tagged_raw_fallback", "Legacy GPL implementation has process-global state; caller must serialize invocations."},
    {"Zlib-ng", "zlib_ng_native", "external/compression/lzbench/lz/zlib-ng",
     "native_api_level6_single_thread_tagged_raw_fallback", "Native zng API with generic architecture functions."},
    {"Zling", "libzling_upstream", "external/compression/lzbench/lz/libzling",
     "level0_single_thread_tagged_raw_fallback", "Strict bounded memory Outputter rejects truncation."},
    {"ZXC", "zxc_upstream", "external/compression/lzbench/lz/zxc",
     "level3_default_block_checksum_single_thread_tagged_raw_fallback", "One-shot buffer API; SIMD dispatch is isolated by the build."},
}};

[[noreturn]] void fail(std::string_view label, std::string_view message) {
  throw std::runtime_error(std::string(label) + ": " + std::string(message));
}

void validate_input(std::span<const std::uint8_t> input) {
  if (input.size() > kMaximumInput) throw std::invalid_argument("general-b input exceeds 256 MiB");
}

Bytes raw(std::span<const std::uint8_t> input) {
  Bytes result;
  result.reserve(input.size() + 1);
  result.push_back(kRaw);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

Bytes tagged(std::span<const std::uint8_t> input, Bytes encoded) {
  if (encoded.size() >= input.size()) return raw(input);
  encoded.insert(encoded.begin(), kUpstream);
  return encoded;
}

Bytes encode_density(std::span<const std::uint8_t> input) {
  const std::scoped_lock lock(density_mutex);
  density_lion_set_ablation_mode(0);
  Bytes output(static_cast<std::size_t>(density_compress_safe_size(input.size())));
  const auto result = density_compress(input.data(), input.size(), output.data(), output.size(),
                                       DENSITY_ALGORITHM_LION);
  if (result.state != DENSITY_STATE_OK || result.bytesRead != input.size() ||
      result.bytesWritten > output.size()) fail("Density", "encode failed");
  output.resize(static_cast<std::size_t>(result.bytesWritten));
  if (output.size() < input.size()) {
    Bytes restored(static_cast<std::size_t>(
        density_decompress_safe_size(input.size())));
    const auto decoded = density_decompress(output.data(), output.size(),
        restored.data(), restored.size());
    if (decoded.state != DENSITY_STATE_OK ||
        decoded.bytesWritten != input.size() ||
        !std::equal(input.begin(), input.end(), restored.begin()))
      return Bytes(input.begin(), input.end());
  }
  return output;
}

Bytes decode_density(std::span<const std::uint8_t> input, std::size_t expected) {
  const std::scoped_lock lock(density_mutex);
  density_lion_set_ablation_mode(0);
  Bytes output(static_cast<std::size_t>(density_decompress_safe_size(expected)));
  const auto result = density_decompress(input.data(), input.size(), output.data(), output.size());
  // Density reports algorithm-body consumption rather than necessarily the
  // full container byte count; its public example and legacy source harness
  // define success by state plus the exact reconstructed length.
  if (result.state != DENSITY_STATE_OK || result.bytesWritten != expected)
    fail("Density", "decode failed or length mismatch");
  output.resize(expected);
  return output;
}

Bytes encode_kanzi(std::span<const std::uint8_t> input) {
  Bytes output(input.size() * 2 + 65536);
  ofixedbuf buffer(reinterpret_cast<char*>(output.data()), output.size());
  std::iostream stream(&buffer);
  kanzi::CompressedOutputStream compressed(stream, 1, "ANS0", "TEXT+UTF+BWT+RANK+ZRLT",
                                           4 * 1024 * 1024, 0, input.size());
  compressed.write(reinterpret_cast<const char*>(input.data()),
                   static_cast<std::streamsize>(input.size()));
  compressed.close();
  const auto size = static_cast<std::size_t>(compressed.getWritten());
  if (size == 0 || size > output.size()) fail("Kanzi", "encode failed");
  output.resize(size);
  return output;
}

Bytes decode_kanzi(std::span<const std::uint8_t> input, std::size_t expected) {
  ifixedbuf buffer(reinterpret_cast<char*>(const_cast<std::uint8_t*>(input.data())), input.size());
  std::iostream stream(&buffer);
  kanzi::CompressedInputStream compressed(stream, 1);
  Bytes output(expected);
  compressed.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(expected));
  const auto decoded = static_cast<std::size_t>(compressed.gcount());
  compressed.close();
  if (decoded != expected) fail("Kanzi", "decode length mismatch");
  return output;
}

Bytes encode_lizard(std::span<const std::uint8_t> input) {
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    fail("Lizard", "input length exceeds int API");
  Bytes output(static_cast<std::size_t>(Lizard_compressBound(static_cast<int>(input.size()))));
  const int size = Lizard_compress(reinterpret_cast<const char*>(input.data()),
      reinterpret_cast<char*>(output.data()), static_cast<int>(input.size()),
      static_cast<int>(output.size()), 19);
  if (size <= 0) fail("Lizard", "encode failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes decode_lizard(std::span<const std::uint8_t> input, std::size_t expected) {
  Bytes output(expected);
  const int size = Lizard_decompress_safe(reinterpret_cast<const char*>(input.data()),
      reinterpret_cast<char*>(output.data()), static_cast<int>(input.size()), static_cast<int>(expected));
  if (size < 0 || static_cast<std::size_t>(size) != expected) fail("Lizard", "decode failed or length mismatch");
  return output;
}

Bytes encode_lzham(std::span<const std::uint8_t> input) {
  Bytes output(input.size() * 2 + 65536);
  lzham_compress_params params{};
  params.m_struct_size = sizeof(params);
  params.m_dict_size_log2 = 20;
  params.m_max_helper_threads = 0;
  params.m_level = LZHAM_COMP_LEVEL_DEFAULT;
  std::size_t size = output.size();
  lzham_uint32 adler = 0;
  const auto status = lzham_compress_memory(&params, output.data(), &size, input.data(), input.size(), &adler);
  if (status != LZHAM_COMP_STATUS_SUCCESS || size > output.size()) fail("LZHAM", "encode failed");
  output.resize(size);
  return output;
}

Bytes decode_lzham(std::span<const std::uint8_t> input, std::size_t expected) {
  Bytes output(expected);
  lzham_decompress_params params{};
  params.m_struct_size = sizeof(params);
  params.m_dict_size_log2 = 20;
  std::size_t size = output.size();
  lzham_uint32 adler = 0;
  const auto status = lzham_decompress_memory(&params, output.data(), &size, input.data(), input.size(), &adler);
  if (status != LZHAM_DECOMP_STATUS_SUCCESS || size != expected) fail("LZHAM", "decode failed or length mismatch");
  return output;
}

Bytes encode_lzlib(std::span<const std::uint8_t> input) {
  int dictionary = std::max<int>(LZ_min_dictionary_size(),
      static_cast<int>(std::min<std::size_t>(input.size(), 1U << 23)));
  LZ_Encoder* encoder = LZ_compress_open(dictionary, 36, 0x7fffffffffffffffULL);
  if (!encoder || LZ_compress_errno(encoder) != LZ_ok) {
    if (encoder) LZ_compress_close(encoder);
    fail("Lzlib", "encoder allocation failed");
  }
  Bytes output(input.size() * 2 + 65536);
  std::size_t written = 0, produced = 0;
  bool finished = false;
  while (!finished) {
    if (LZ_compress_write_size(encoder) > 0 && written < input.size()) {
      const int amount = static_cast<int>(std::min<std::size_t>(input.size() - written,
          static_cast<std::size_t>(std::numeric_limits<int>::max())));
      const int n = LZ_compress_write(encoder, input.data() + written, amount);
      if (n < 0) { LZ_compress_close(encoder); fail("Lzlib", "write failed"); }
      written += static_cast<std::size_t>(n);
    }
    if (written == input.size()) LZ_compress_finish(encoder);
    if (produced == output.size()) { LZ_compress_close(encoder); fail("Lzlib", "output bound exhausted"); }
    const int n = LZ_compress_read(encoder, output.data() + produced,
        static_cast<int>(std::min<std::size_t>(output.size() - produced, std::numeric_limits<int>::max())));
    if (n < 0) { LZ_compress_close(encoder); fail("Lzlib", "read failed"); }
    produced += static_cast<std::size_t>(n);
    finished = LZ_compress_finished(encoder) == 1;
  }
  if (LZ_compress_close(encoder) < 0) fail("Lzlib", "close failed");
  output.resize(produced);
  return output;
}

Bytes decode_lzlib(std::span<const std::uint8_t> input, std::size_t expected) {
  LZ_Decoder* decoder = LZ_decompress_open();
  if (!decoder || LZ_decompress_errno(decoder) != LZ_ok) {
    if (decoder) LZ_decompress_close(decoder);
    fail("Lzlib", "decoder allocation failed");
  }
  Bytes output(expected);
  std::size_t written = 0, produced = 0;
  bool finished = false;
  while (!finished) {
    if (LZ_decompress_write_size(decoder) > 0 && written < input.size()) {
      const int amount = static_cast<int>(std::min<std::size_t>(input.size() - written,
          static_cast<std::size_t>(std::numeric_limits<int>::max())));
      const int n = LZ_decompress_write(decoder, input.data() + written, amount);
      if (n < 0) { LZ_decompress_close(decoder); fail("Lzlib", "write failed"); }
      written += static_cast<std::size_t>(n);
    }
    if (written == input.size()) LZ_decompress_finish(decoder);
    const int capacity = static_cast<int>(std::min<std::size_t>(expected - produced,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int n = LZ_decompress_read(decoder, output.data() + produced, capacity);
    if (n < 0) { LZ_decompress_close(decoder); fail("Lzlib", "read failed"); }
    produced += static_cast<std::size_t>(n);
    finished = LZ_decompress_finished(decoder) == 1;
    if (!finished && produced == expected) { LZ_decompress_close(decoder); fail("Lzlib", "decoded output exceeds expected length"); }
  }
  if (LZ_decompress_close(decoder) < 0 || produced != expected || written != input.size())
    fail("Lzlib", "decode failed or length mismatch");
  return output;
}

struct PpmdWriter {
  IByteOut stream{};
  Bytes* output{};
  bool overflow = false;
  static void write(const IByteOut* base, Byte value) {
    auto* self = reinterpret_cast<PpmdWriter*>(const_cast<IByteOut*>(base));
    if (self->output->size() == self->output->capacity()) { self->overflow = true; return; }
    self->output->push_back(value);
  }
};

struct PpmdReader {
  IByteIn stream{};
  const std::uint8_t* current{};
  const std::uint8_t* end{};
  bool overread = false;
  static Byte read(const IByteIn* base) {
    auto* self = reinterpret_cast<PpmdReader*>(const_cast<IByteIn*>(base));
    if (self->current == self->end) { self->overread = true; return 0; }
    return *self->current++;
  }
};

void* ppmd_alloc(ISzAllocPtr, std::size_t size) { return std::malloc(size); }
void ppmd_free(ISzAllocPtr, void* address) { std::free(address); }

Bytes encode_ppmd(std::span<const std::uint8_t> input) {
  Bytes output;
  output.reserve(input.size() * 2 + 65536);
  PpmdWriter writer;
  writer.stream.Write = &PpmdWriter::write;
  writer.output = &output;
  CPpmd8 ppmd{};
  ppmd.Stream.Out = &writer.stream;
  ISzAlloc allocator{ppmd_alloc, ppmd_free};
  Ppmd8_Construct(&ppmd);
  if (!Ppmd8_Alloc(&ppmd, 16U << 20, &allocator)) fail("PPMD8", "model allocation failed");
  Ppmd8_Init_RangeEnc(&ppmd);
  Ppmd8_Init(&ppmd, 8, PPMD8_RESTORE_METHOD_RESTART);
  for (std::uint8_t value : input) Ppmd8_EncodeSymbol(&ppmd, value);
  Ppmd8_EncodeSymbol(&ppmd, -1);
  Ppmd8_Flush_RangeEnc(&ppmd);
  Ppmd8_Free(&ppmd, &allocator);
  if (writer.overflow) fail("PPMD8", "bounded output exhausted");
  return output;
}

Bytes decode_ppmd(std::span<const std::uint8_t> input, std::size_t expected) {
  PpmdReader reader;
  reader.stream.Read = &PpmdReader::read;
  reader.current = input.data();
  reader.end = input.data() + input.size();
  CPpmd8 ppmd{};
  ppmd.Stream.In = &reader.stream;
  ISzAlloc allocator{ppmd_alloc, ppmd_free};
  Ppmd8_Construct(&ppmd);
  if (!Ppmd8_Alloc(&ppmd, 16U << 20, &allocator)) fail("PPMD8", "model allocation failed");
  if (!Ppmd8_Init_RangeDec(&ppmd)) { Ppmd8_Free(&ppmd, &allocator); fail("PPMD8", "range header invalid"); }
  Ppmd8_Init(&ppmd, 8, PPMD8_RESTORE_METHOD_RESTART);
  Bytes output;
  output.reserve(expected);
  int symbol = 0;
  while ((symbol = Ppmd8_DecodeSymbol(&ppmd)) >= 0) {
    if (output.size() == expected) { Ppmd8_Free(&ppmd, &allocator); fail("PPMD8", "decoded output exceeds expected length"); }
    output.push_back(static_cast<std::uint8_t>(symbol));
  }
  const bool ok = symbol == PPMD8_SYM_END && output.size() == expected &&
      Ppmd8_RangeDec_IsFinishedOK(&ppmd) && !reader.overread;
  Ppmd8_Free(&ppmd, &allocator);
  if (!ok) fail("PPMD8", "decode failed or length mismatch");
  return output;
}

Bytes encode_tornado(std::span<const std::uint8_t> input) {
  const std::scoped_lock lock(tornado_mutex);
  tor_set_ablation_mode(0);
  Bytes output(input.size() * 2 + 65536);
  const auto size = tor_compress(5, const_cast<std::uint8_t*>(input.data()),
      static_cast<std::uint32_t>(input.size()), output.data(), static_cast<std::uint32_t>(output.size()));
  if (size == 0 || size > output.size()) fail("Tornado", "encode failed");
  output.resize(size);
  return output;
}

Bytes decode_tornado(std::span<const std::uint8_t> input, std::size_t expected) {
  const std::scoped_lock lock(tornado_mutex);
  tor_set_ablation_mode(0);
  Bytes output(expected);
  const auto size = tor_decompress(const_cast<std::uint8_t*>(input.data()),
      static_cast<std::uint32_t>(input.size()), output.data(), static_cast<std::uint32_t>(output.size()));
  if (size != expected) fail("Tornado", "decode failed or length mismatch");
  return output;
}

Bytes encode_zlib_ng(std::span<const std::uint8_t> input) {
  Bytes output(input.size() + input.size() / 8 + 65536);
  std::size_t size = output.size();
  if (zng_compress2(output.data(), &size, input.data(), input.size(), 6) != Z_OK)
    fail("Zlib-ng", "encode failed");
  output.resize(size);
  return output;
}

Bytes decode_zlib_ng(std::span<const std::uint8_t> input, std::size_t expected) {
  Bytes output(expected);
  std::size_t size = output.size();
  if (zng_uncompress(output.data(), &size, input.data(), input.size()) != Z_OK || size != expected)
    fail("Zlib-ng", "decode failed or length mismatch");
  return output;
}

class ZlingInput final : public baidu::zling::Inputter {
 public:
  explicit ZlingInput(std::span<const std::uint8_t> data) : data_(data) {}
  std::size_t GetData(unsigned char* output, std::size_t size) override {
    size = std::min(size, data_.size() - position_);
    if (size) std::memcpy(output, data_.data() + position_, size);
    position_ += size;
    return size;
  }
  bool IsEnd() override { return position_ == data_.size(); }
  bool IsErr() override { return false; }
 private:
  std::span<const std::uint8_t> data_;
  std::size_t position_ = 0;
};

class ZlingOutput final : public baidu::zling::Outputter {
 public:
  explicit ZlingOutput(std::span<std::uint8_t> data) : data_(data) {}
  std::size_t PutData(unsigned char* input, std::size_t size) override {
    if (size > data_.size() - position_) { failed_ = true; return 0; }
    if (size) std::memcpy(data_.data() + position_, input, size);
    position_ += size;
    return size;
  }
  bool IsErr() override { return failed_; }
  std::size_t size() const { return position_; }
 private:
  std::span<std::uint8_t> data_;
  std::size_t position_ = 0;
  bool failed_ = false;
};

Bytes encode_zling(std::span<const std::uint8_t> input) {
  Bytes output(input.size() * 2 + 65536);
  ZlingInput source(input);
  ZlingOutput sink(output);
  const int status = baidu::zling::Encode(&source, &sink, nullptr, 0);
  if (status != 0 || sink.IsErr()) fail("Zling", "encode failed or output truncated");
  output.resize(sink.size());
  return output;
}

Bytes decode_zling(std::span<const std::uint8_t> input, std::size_t expected) {
  Bytes output(expected);
  ZlingInput source(input);
  ZlingOutput sink(output);
  const int status = baidu::zling::Decode(&source, &sink);
  if (status != 0 || sink.IsErr() || sink.size() != expected) fail("Zling", "decode failed or length mismatch");
  return output;
}

Bytes encode_zxc(std::span<const std::uint8_t> input) {
  zxc_compress_opts_t options{};
  options.n_threads = 1;
  options.level = ZXC_LEVEL_DEFAULT;
  options.checksum_enabled = 1;
  Bytes output(static_cast<std::size_t>(zxc_compress_bound(input.size())));
  const auto size = zxc_compress(input.data(), input.size(), output.data(), output.size(), &options);
  if (size <= 0 || static_cast<std::size_t>(size) > output.size()) fail("ZXC", "encode failed");
  output.resize(static_cast<std::size_t>(size));
  return output;
}

Bytes decode_zxc(std::span<const std::uint8_t> input, std::size_t expected) {
  if (zxc_get_decompressed_size(input.data(), input.size()) != expected)
    fail("ZXC", "stored decoded length mismatch");
  zxc_decompress_opts_t options{};
  options.n_threads = 1;
  options.checksum_enabled = 1;
  Bytes output(expected);
  const auto size = zxc_decompress(input.data(), input.size(), output.data(), output.size(), &options);
  if (size <= 0 || static_cast<std::size_t>(size) != expected) fail("ZXC", "decode failed or length mismatch");
  return output;
}

Bytes encode_real(std::string_view label, std::span<const std::uint8_t> input) {
  if (label == "Density") return encode_density(input);
  if (label == "Kanzi") return encode_kanzi(input);
  if (label == "Lizard") return encode_lizard(input);
  if (label == "LZHAM") return encode_lzham(input);
  if (label == "Lzlib") return encode_lzlib(input);
  if (label == "PPMD8") return encode_ppmd(input);
  if (label == "Tornado") return encode_tornado(input);
  if (label == "Zlib-ng") return encode_zlib_ng(input);
  if (label == "Zling") return encode_zling(input);
  if (label == "ZXC") return encode_zxc(input);
  throw std::invalid_argument("unknown general-b label");
}

Bytes decode_real(std::string_view label, std::span<const std::uint8_t> input, std::size_t expected) {
  if (label == "Density") return decode_density(input, expected);
  if (label == "Kanzi") return decode_kanzi(input, expected);
  if (label == "Lizard") return decode_lizard(input, expected);
  if (label == "LZHAM") return decode_lzham(input, expected);
  if (label == "Lzlib") return decode_lzlib(input, expected);
  if (label == "PPMD8") return decode_ppmd(input, expected);
  if (label == "Tornado") return decode_tornado(input, expected);
  if (label == "Zlib-ng") return decode_zlib_ng(input, expected);
  if (label == "Zling") return decode_zling(input, expected);
  if (label == "ZXC") return decode_zxc(input, expected);
  throw std::invalid_argument("unknown general-b label");
}

}  // namespace

std::span<const Descriptor> descriptors() { return kDescriptors; }

bool supports(std::string_view label) {
  return std::ranges::any_of(kDescriptors, [label](const Descriptor& value) { return value.label == label; });
}

Bytes encode(std::string_view label, std::span<const std::uint8_t> input) {
  validate_input(input);
  if (!supports(label)) throw std::invalid_argument("unknown general-b label");
  if (input.empty()) return raw(input);
  return tagged(input, encode_real(label, input));
}

Bytes decode(std::string_view label, std::span<const std::uint8_t> payload,
             std::size_t expected_size) {
  if (!supports(label)) throw std::invalid_argument("unknown general-b label");
  if (expected_size > kMaximumInput) throw std::invalid_argument("general-b expected size exceeds 256 MiB");
  if (payload.empty()) throw std::invalid_argument("general-b payload is empty");
  if (payload.front() == kRaw) {
    if (payload.size() - 1 != expected_size) throw std::invalid_argument("general-b raw length mismatch");
    return Bytes(payload.begin() + 1, payload.end());
  }
  if (payload.front() != kUpstream) throw std::invalid_argument("general-b invalid adapter tag");
  return decode_real(label, payload.subspan(1), expected_size);
}

}  // namespace native_baseline86_v1::upstream_general_b
