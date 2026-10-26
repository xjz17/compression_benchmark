#include "upstream_general_a.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

#if defined(BASELINE86_GENERAL_A_GIPFELI)
#include "gipfeli.h"
#endif
#if defined(BASELINE86_GENERAL_A_GLZA)
extern "C" {
#include "GLZAcomp.h"
#include "GLZAdecode.h"
}
#if defined(BASELINE86_GENERAL_A_GLZA_SERIAL_SCHEDULER)
#include "glza_serial_scheduler.h"
#endif
#endif
#if defined(BASELINE86_GENERAL_A_LZO)
extern "C" {
#include "lzo/lzo1.h"
#include "lzo/lzo2a.h"
}
#endif
#if defined(BASELINE86_GENERAL_A_LZSSE)
extern "C" {
#include "lzsse8.h"
}
#endif
#if defined(BASELINE86_GENERAL_A_NRV)
extern "C" {
#include "ucl.h"
}
#endif
#if defined(BASELINE86_GENERAL_A_BSC)
extern "C" {
#include "libbsc.h"
}
#endif
#if defined(BASELINE86_GENERAL_A_CRUSH)
#include "crush.hpp"
#endif
#if defined(BASELINE86_GENERAL_A_CSC)
#include "csc_dec.h"
#include "csc_enc.h"
int g_csc_ablation_mode = CSC_ABLATION_NONE;
#endif
#if defined(BASELINE86_GENERAL_A_FASTLZMA2)
extern "C" {
#include "fast-lzma2.h"
}
#endif

namespace native_baseline86_v1::upstream_general_a {
namespace {

constexpr std::size_t kMaximumInput = 256ULL << 20;

[[noreturn]] void fail(const std::string& label, const char* detail) {
  throw std::invalid_argument(label + ": " + detail);
}

void require(bool condition, const std::string& label, const char* detail) {
  if (!condition) fail(label, detail);
}

int as_int(std::size_t value, const std::string& label) {
  require(value <= static_cast<std::size_t>(INT_MAX), label,
          "size exceeds upstream int range");
  return static_cast<int>(value);
}

Bytes raw_payload(std::span<const std::uint8_t> input) {
  Bytes result;
  result.reserve(input.size() + 1);
  result.push_back(0);
  result.insert(result.end(), input.begin(), input.end());
  return result;
}

Bytes compressed_or_raw(std::span<const std::uint8_t> input, Bytes result,
                        std::size_t written, const std::string& label) {
  require(written <= result.size(), label, "compressor exceeded output bound");
  if (written == 0 || written >= input.size()) return raw_payload(input);
  result.resize(written + 1);
  std::memmove(result.data() + 1, result.data(), written);
  result[0] = 1;
  return result;
}

std::span<const std::uint8_t> compressed_body(
    const std::string& label, std::span<const std::uint8_t> payload,
    std::size_t expected_size, Bytes& raw) {
  require(!payload.empty(), label, "empty adapter payload");
  if (payload[0] == 0) {
    require(payload.size() == expected_size + 1, label,
            "verbatim payload length mismatch");
    raw.assign(payload.begin() + 1, payload.end());
    return {};
  }
  require(payload[0] == 1, label, "unknown adapter payload tag");
  require(payload.size() > 1, label, "empty upstream stream");
  return payload.subspan(1);
}

std::vector<Descriptor> make_inventory() {
  std::vector<Descriptor> result;
  auto add = [&](const char* label, const char* implementation,
                 const char* source, const char* configuration,
                 const char* limitations, bool available) {
    result.push_back({label, implementation, source, configuration,
                      limitations, available});
  };
#if defined(BASELINE86_GENERAL_A_GIPFELI)
  constexpr bool gipfeli = true;
#else
  constexpr bool gipfeli = false;
#endif
#if defined(BASELINE86_GENERAL_A_GLZA) && \
    defined(BASELINE86_GENERAL_A_GLZA_SERIAL_SCHEDULER)
  constexpr bool glza = true;
#else
  constexpr bool glza = false;
#endif
#if defined(BASELINE86_GENERAL_A_LZO)
  constexpr bool lzo = true;
#else
  constexpr bool lzo = false;
#endif
#if defined(BASELINE86_GENERAL_A_LZSSE)
  constexpr bool lzsse = true;
#else
  constexpr bool lzsse = false;
#endif
#if defined(BASELINE86_GENERAL_A_NRV)
  constexpr bool nrv = true;
#else
  constexpr bool nrv = false;
#endif
#if defined(BASELINE86_GENERAL_A_BSC)
  constexpr bool bsc = true;
#else
  constexpr bool bsc = false;
#endif
#if defined(BASELINE86_GENERAL_A_CRUSH)
  constexpr bool crush = true;
#else
  constexpr bool crush = false;
#endif
#if defined(BASELINE86_GENERAL_A_CSC)
  constexpr bool csc = true;
#else
  constexpr bool csc = false;
#endif
#if defined(BASELINE86_GENERAL_A_FASTLZMA2)
  constexpr bool fastlzma2 = true;
#else
  constexpr bool fastlzma2 = false;
#endif
  add("Gipfeli", "google_gipfeli_2016_07_13",
      "external/compression/lzbench/lz/gipfeli",
      "string_api_default_single_thread+encode_decode_admission+tagged_raw_fallback",
      "Vendored Gipfeli is upstream-marked potentially unstable; a compressed candidate is admitted only after exact decode, otherwise the block uses the tagged-raw fallback.",
      gipfeli);
  add("GLZA", "glza_0_12",
      "external/compression/lzbench/misc/glza",
#if defined(BASELINE86_GENERAL_A_GLZA_SERIAL_SCHEDULER)
      "library_defaults_cooperative_single_os_thread+tagged_raw_when_not_smaller",
      "The unmodified GLZA worker graph runs through a bounded ucontext scheduler on the caller OS thread; decoder two_threads remains disabled. Inputs below 1024 bytes use the explicit tagged-raw branch.",
#else
      "library_defaults_internal_pthread_workers+tagged_raw_when_not_smaller",
      "The vendored GLZA encoder starts internal pthread workers, so builds without the cooperative scheduler fail closed: GLZA is unavailable and cannot be listed as a primary strict-single-thread method.",
#endif
      glza);
  add("LZO1", "lzo_2_10_lzo1",
      "external/compression/lzbench/lz/lzo",
      "lzo1_level1_single_thread+tagged_raw_when_not_smaller",
      "The public family label is fixed to its LZO1 level-1 entry; no post-hoc variant selection.",
      lzo);
  add("LZO2A", "lzo_2_10_lzo2a",
      "external/compression/lzbench/lz/lzo",
      "lzo2a_999_single_thread+control_bit_worst_case_bound+encode_decode_admission+tagged_raw_fallback",
      "The adapter reserves one LZO2A control bit per possible literal and admits a compressed candidate only after exact decode.", lzo);
  add("LZSSE", "lzsse_2019_04_18_lzsse8fast",
      "external/compression/lzbench/lz/lzsse",
      "lzsse8_fast_single_thread_sse4_1+tagged_raw_when_not_smaller",
      "Requires x86-64 with SSE4.1; disabled on other architectures.", lzsse);
  add("NRV", "ucl_1_03_nrv2b",
      "external/compression/lzbench/lz/ucl",
      "nrv2b_level6_single_thread+tagged_raw_when_not_smaller",
      "The NRV family label is fixed to NRV2B level 6; no post-hoc NRV2B/2D/2E selection.",
      nrv);
  add("BSC", "libbsc_3_3_11",
      "external/compression/lzbench/bwt/libbsc",
      "m0_e2_bwt_qlfc_adaptive_fastmode_single_thread+tagged_raw_when_not_smaller",
      "CPU-only fixed bsc0 profile; CUDA and OpenMP are disabled.", bsc);
  add("Crush", "crush_1_0", "external/compression/lzbench/lz/crush",
      "level0_single_thread+zero_padded_workspace+encode_decode_admission+tagged_raw_fallback",
      "Upstream buffer API has no explicit output capacity; the adapter uses zero-padded bounded workspaces and admits only candidates that decode exactly.",
      crush);
  add("CSC", "csc_2016_10_13",
      "external/compression/lzbench/lz/libcsc",
      "level2_dict_min_64MiB_input_single_thread+zeroed_allocator+encode_decode_admission+tagged_raw_fallback",
      "Vendored CSC is upstream-marked potentially unstable; the adapter zero-initializes codec allocations and admits only candidates that decode exactly through bounded streams.",
      csc);
  add("FastLZMA2", "fast_lzma2_1_0_1",
      "external/compression/lzbench/lz/fast-lzma2",
      "level5_single_thread+tagged_raw_when_not_smaller",
      "Built with FL2_SINGLETHREAD and NO_XXHASH; fixed level 5.", fastlzma2);
  return result;
}

#if defined(BASELINE86_GENERAL_A_GIPFELI)
Bytes encode_gipfeli(std::span<const std::uint8_t> input) {
  constexpr const char* label = "Gipfeli";
  std::unique_ptr<util::compression::Compressor> codec(
      util::compression::NewGipfeliCompressor());
  require(codec != nullptr, label, "compressor allocation failed");
  std::string source(reinterpret_cast<const char*>(input.data()), input.size());
  std::string output;
  const auto written = codec->Compress(source, &output);
  require(written == output.size(), label, "compressor length mismatch");
  if (written != 0 && written < input.size()) {
    std::string restored;
    if (!codec->Uncompress(output, &restored) || restored.size() != input.size() ||
        !std::equal(restored.begin(), restored.end(), source.begin()))
      return raw_payload(input);
  }
  Bytes result(output.begin(), output.end());
  return compressed_or_raw(input, std::move(result), written, label);
}

Bytes decode_gipfeli(std::span<const std::uint8_t> input,
                     std::size_t expected) {
  constexpr const char* label = "Gipfeli";
  std::unique_ptr<util::compression::Compressor> codec(
      util::compression::NewGipfeliCompressor());
  require(codec != nullptr, label, "decompressor allocation failed");
  std::string source(reinterpret_cast<const char*>(input.data()), input.size());
  std::string output;
  require(codec->Uncompress(source, &output), label, "invalid stream");
  require(output.size() == expected, label, "decoded size mismatch");
  return Bytes(output.begin(), output.end());
}
#endif

#if defined(BASELINE86_GENERAL_A_GLZA)
std::mutex glza_mutex;
Bytes encode_glza(std::span<const std::uint8_t> input) {
  constexpr const char* label = "GLZA";
  std::lock_guard<std::mutex> lock(glza_mutex);
  Bytes mutable_input(input.begin(), input.end());
  Bytes output(input.size() + input.size() / 8 + 1024);
  std::size_t written = output.size();
#if defined(BASELINE86_GENERAL_A_GLZA_SERIAL_SCHEDULER)
  require(glza_serial_scheduler_begin() == 0, label,
          "serial scheduler initialization failed");
#endif
  const auto status = GLZAcomp(input.size(), mutable_input.data(), &written,
                               output.data(), nullptr, nullptr);
#if defined(BASELINE86_GENERAL_A_GLZA_SERIAL_SCHEDULER)
  glza_serial_scheduler_stats scheduler_stats{};
  const auto scheduler_status =
      glza_serial_scheduler_finish(&scheduler_stats);
  require(scheduler_status == 0, label,
          "serial scheduler did not finish cleanly");
#endif
  require(status != 0,
          label, "compressor failed");
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_glza(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "GLZA";
  std::lock_guard<std::mutex> lock(glza_mutex);
  Bytes mutable_input(input.begin(), input.end());
  Bytes output(expected);
  std::size_t written = expected;
  require(GLZAdecode(input.size(), mutable_input.data(), &written, output.data(),
                     nullptr, nullptr) != nullptr,
          label, "invalid stream");
  require(written == expected, label, "decoded size mismatch");
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_LZO)
Bytes encode_lzo(const std::string& label,
                 std::span<const std::uint8_t> input) {
  require(lzo_init() == LZO_E_OK, label, "library initialization failed");
  // LZO2A emits one control bit for each literal.  Its incompressible bound is
  // therefore input + ceil(input/8) plus the final code, not the LZO1 1/16
  // allowance.  The old shared bound let the 999 encoder overwrite the heap.
  const auto overhead = label == "LZO2A" ? (input.size() + 7) / 8 + 256
                                          : input.size() / 16 + 256;
  Bytes output(input.size() + overhead);
  lzo_uint written = 0;
  int status = LZO_E_ERROR;
  if (label == "LZO1") {
    Bytes work(LZO1_MEM_COMPRESS);
    status = lzo1_compress(input.data(), input.size(), output.data(), &written,
                           work.data());
  } else {
    Bytes work(LZO2A_999_MEM_COMPRESS);
    status = lzo2a_999_compress(input.data(), input.size(), output.data(),
                                &written, work.data());
  }
  require(status == LZO_E_OK, label, "compressor failed");
  if (written != 0 && written < input.size()) {
    Bytes restored(input.size());
    lzo_uint restored_size = restored.size();
    const int decoded = label == "LZO1"
                            ? lzo1_decompress(output.data(), written,
                                              restored.data(), &restored_size,
                                              nullptr)
                            : lzo2a_decompress_safe(output.data(), written,
                                                    restored.data(),
                                                    &restored_size, nullptr);
    if (decoded != LZO_E_OK || restored_size != input.size() ||
        !std::equal(restored.begin(), restored.end(), input.begin()))
      return raw_payload(input);
  }
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_lzo(const std::string& label, std::span<const std::uint8_t> input,
                 std::size_t expected) {
  Bytes output(expected);
  lzo_uint written = expected;
  const int status = label == "LZO1"
                         ? lzo1_decompress(input.data(), input.size(),
                                           output.data(), &written, nullptr)
                         : lzo2a_decompress_safe(input.data(), input.size(),
                                                 output.data(), &written, nullptr);
  require(status == LZO_E_OK && written == expected, label,
          "invalid or wrong-sized stream");
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_LZSSE)
Bytes encode_lzsse(std::span<const std::uint8_t> input) {
  constexpr const char* label = "LZSSE";
  auto* state = LZSSE8_MakeFastParseState();
  require(state != nullptr, label, "parse-state allocation failed");
  Bytes output(input.size());
  const auto written = LZSSE8_CompressFast(state, input.data(), input.size(),
                                           output.data(), output.size());
  LZSSE8_FreeFastParseState(state);
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_lzsse(std::span<const std::uint8_t> input,
                   std::size_t expected) {
  constexpr const char* label = "LZSSE";
  Bytes output(expected);
  const auto written =
      LZSSE8_Decompress(input.data(), input.size(), output.data(), expected);
  require(written == expected, label, "invalid or wrong-sized stream");
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_NRV)
Bytes encode_nrv(std::span<const std::uint8_t> input) {
  constexpr const char* label = "NRV";
  require(ucl_init() == UCL_E_OK, label, "library initialization failed");
  Bytes output(input.size() + input.size() / 8 + 1024);
  ucl_uint written = output.size();
  const int status = ucl_nrv2b_99_compress(
      input.data(), input.size(), output.data(), &written, nullptr, 6, nullptr,
      nullptr);
  require(status == UCL_E_OK, label, "compressor failed");
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_nrv(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "NRV";
  Bytes output(expected);
  ucl_uint written = expected;
  const int status = ucl_nrv2b_decompress_safe_8(
      input.data(), input.size(), output.data(), &written, nullptr);
  require(status == UCL_E_OK && written == expected, label,
          "invalid or wrong-sized stream");
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_BSC)
Bytes encode_bsc(std::span<const std::uint8_t> input) {
  constexpr const char* label = "BSC";
  require(input.size() <= static_cast<std::size_t>(INT_MAX), label,
          "input exceeds libbsc range");
  constexpr int features = LIBBSC_FEATURE_FASTMODE;
  require(bsc_init(features) == LIBBSC_NO_ERROR, label,
          "library initialization failed");
  Bytes output(input.size() + LIBBSC_HEADER_SIZE);
  const int written = bsc_compress(
      input.data(), output.data(), as_int(input.size(), label),
      LIBBSC_DEFAULT_LZPHASHSIZE, LIBBSC_DEFAULT_LZPMINLEN,
      LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE, features);
  if (written == LIBBSC_NOT_COMPRESSIBLE) return raw_payload(input);
  require(written >= 0, label, "compressor failed");
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_bsc(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "BSC";
  require(input.size() >= LIBBSC_HEADER_SIZE, label, "truncated stream");
  constexpr int features = LIBBSC_FEATURE_FASTMODE;
  require(bsc_init(features) == LIBBSC_NO_ERROR, label,
          "decoder initialization failed");
  int compressed = 0, decompressed = 0;
  require(bsc_block_info(input.data(), LIBBSC_HEADER_SIZE, &compressed,
                         &decompressed, features) == LIBBSC_NO_ERROR,
          label, "invalid header");
  require(compressed == as_int(input.size(), label) &&
              decompressed == as_int(expected, label),
          label, "stream length mismatch");
  Bytes output(expected);
  require(bsc_decompress(input.data(), compressed, output.data(), decompressed,
                         features) == LIBBSC_NO_ERROR,
          label, "invalid stream");
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_CRUSH)
Bytes encode_crush(std::span<const std::uint8_t> input) {
  constexpr const char* label = "Crush";
  Bytes mutable_input(input.begin(), input.end());
  mutable_input.resize(input.size() + 64, 0);
  Bytes output(input.size() + input.size() / 8 + 1024 + 64, 0);
  const auto written = crush::compress(0, mutable_input.data(),
                                       as_int(input.size(), label),
                                       output.data());
  require(written <= output.size() - 64, label,
          "compressor exceeded padded output bound");
  if (written != 0 && written < input.size()) {
    Bytes encoded(output.begin(), output.begin() + written);
    encoded.resize(written + 64, 0);
    Bytes restored(input.size() + 64, 0);
    const auto restored_size = crush::decompress(
        encoded.data(), restored.data(), as_int(input.size(), label));
    if (restored_size != input.size() ||
        !std::equal(input.begin(), input.end(), restored.begin()))
      return raw_payload(input);
  }
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_crush(std::span<const std::uint8_t> input,
                   std::size_t expected) {
  constexpr const char* label = "Crush";
  Bytes mutable_input(input.begin(), input.end());
  mutable_input.resize(input.size() + 64, 0);
  Bytes output(expected + 64, 0);
  const auto written = crush::decompress(mutable_input.data(), output.data(),
                                         as_int(expected, label));
  require(written == expected, label, "invalid or wrong-sized stream");
  output.resize(expected);
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_CSC)
void* csc_zero_alloc(void*, std::size_t size) {
  // libcsc allocates C++ implementation objects through this C interface and
  // its fixed wire can otherwise observe uninitialised table bytes.  A
  // zero-initialised allocator makes the fixed configuration reproducible.
  return std::calloc(size == 0 ? 1 : size, 1);
}
void csc_zero_free(void*, void* address) { std::free(address); }
ISzAlloc csc_allocator{csc_zero_alloc, csc_zero_free};

struct CscInput {
  ISeqInStream api{};
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  std::size_t position = 0;
  bool failed = false;
};
struct CscOutput {
  ISeqOutStream api{};
  std::uint8_t* data = nullptr;
  std::size_t capacity = 0;
  std::size_t position = 0;
  bool failed = false;
};
SRes csc_read(void* opaque, void* destination, std::size_t* count) {
  auto& stream = *static_cast<CscInput*>(opaque);
  const auto available = stream.size - stream.position;
  const auto take = std::min(*count, available);
  if (take != 0)
    std::memcpy(destination, stream.data + stream.position, take);
  stream.position += take;
  *count = take;
  return SZ_OK;
}
std::size_t csc_write(void* opaque, const void* source, std::size_t count) {
  auto& stream = *static_cast<CscOutput*>(opaque);
  if (count > stream.capacity - stream.position) {
    stream.failed = true;
    return 0;
  }
  if (count != 0)
    std::memcpy(stream.data + stream.position, source, count);
  stream.position += count;
  return count;
}

Bytes decode_csc(std::span<const std::uint8_t> input, std::size_t expected);

Bytes encode_csc(std::span<const std::uint8_t> input) {
  constexpr const char* label = "CSC";
  Bytes output(input.size() + input.size() / 8 + (1U << 20));
  CSCProps properties{};
  CSCEncProps_Init(&properties,
                   static_cast<std::uint32_t>(std::min<std::size_t>(
                       input.size(), static_cast<std::size_t>(1U << 26))),
                   2);
  CSCEnc_WriteProperties(&properties, output.data(), 0);
  CscInput source{{csc_read}, input.data(), input.size(), 0, false};
  CscOutput destination{{csc_write}, output.data() + CSC_PROP_SIZE,
                        output.size() - CSC_PROP_SIZE, 0, false};
  CSCEncHandle encoder =
      CSCEnc_Create(&properties, &destination.api, &csc_allocator);
  require(encoder != nullptr, label, "encoder allocation failed");
  const int encoded = CSCEnc_Encode(encoder, &source.api, nullptr);
  const int flushed = encoded == 0 ? CSCEnc_Encode_Flush(encoder) : encoded;
  CSCEnc_Destroy(encoder);
  require(encoded == 0 && flushed == 0 && !destination.failed &&
              source.position == source.size,
          label, "compressor failed or exceeded output bound");
  auto candidate = compressed_or_raw(
      input, std::move(output), destination.position + CSC_PROP_SIZE, label);
  if (candidate.front() == 1) {
    try {
      const auto restored = decode_csc(
          std::span<const std::uint8_t>(candidate).subspan(1), input.size());
      if (!std::equal(input.begin(), input.end(), restored.begin()))
        return raw_payload(input);
    } catch (const std::exception&) {
      return raw_payload(input);
    }
  }
  return candidate;
}

Bytes decode_csc(std::span<const std::uint8_t> input, std::size_t expected) {
  constexpr const char* label = "CSC";
  require(input.size() >= CSC_PROP_SIZE, label, "truncated stream");
  CSCProps properties{};
  CSCDec_ReadProperties(&properties,
                        const_cast<std::uint8_t*>(input.data()));
  CscInput source{{csc_read}, input.data() + CSC_PROP_SIZE,
                  input.size() - CSC_PROP_SIZE, 0, false};
  Bytes output(expected);
  CscOutput destination{{csc_write}, output.data(), output.size(), 0, false};
  CSCDecHandle decoder =
      CSCDec_Create(&properties, &source.api, &csc_allocator);
  require(decoder != nullptr, label, "decoder allocation failed");
  const int status = CSCDec_Decode(decoder, &destination.api, nullptr);
  CSCDec_Destroy(decoder);
  require(status == 0 && !destination.failed &&
              destination.position == expected && source.position == source.size,
          label, "invalid, trailing, or wrong-sized stream");
  return output;
}
#endif

#if defined(BASELINE86_GENERAL_A_FASTLZMA2)
Bytes encode_fastlzma2(std::span<const std::uint8_t> input) {
  constexpr const char* label = "FastLZMA2";
  Bytes output(FL2_compressBound(input.size()));
  const auto written = FL2_compress(output.data(), output.size(), input.data(),
                                    input.size(), 5);
  require(!FL2_isError(written), label, FL2_getErrorName(written));
  return compressed_or_raw(input, std::move(output), written, label);
}

Bytes decode_fastlzma2(std::span<const std::uint8_t> input,
                       std::size_t expected) {
  constexpr const char* label = "FastLZMA2";
  require(FL2_findDecompressedSize(input.data(), input.size()) == expected,
          label, "stream declares a different decoded size");
  Bytes output(expected);
  const auto written = FL2_decompress(output.data(), output.size(), input.data(),
                                      input.size());
  require(!FL2_isError(written) && written == expected, label,
          "invalid or wrong-sized stream");
  return output;
}
#endif

}  // namespace

const std::vector<Descriptor>& inventory() {
  static const auto records = make_inventory();
  return records;
}

const Descriptor& descriptor(const std::string& label) {
  const auto& records = inventory();
  const auto it = std::find_if(records.begin(), records.end(),
                               [&](const auto& r) { return r.label == label; });
  if (it == records.end()) fail(label, "unknown general-A label");
  return *it;
}

Bytes encode(const std::string& label, std::span<const std::uint8_t> input) {
  require(input.size() <= kMaximumInput, label, "input exceeds 256 MiB limit");
  require(descriptor(label).available, label,
          "direct upstream implementation unavailable in this build");
  if (input.empty()) return raw_payload(input);
#if defined(BASELINE86_GENERAL_A_GLZA)
  if (label == "GLZA" && input.size() < 1024) return raw_payload(input);
#endif
#if defined(BASELINE86_GENERAL_A_GIPFELI)
  if (label == "Gipfeli") return encode_gipfeli(input);
#endif
#if defined(BASELINE86_GENERAL_A_GLZA)
  if (label == "GLZA") return encode_glza(input);
#endif
#if defined(BASELINE86_GENERAL_A_LZO)
  if (label == "LZO1" || label == "LZO2A") return encode_lzo(label, input);
#endif
#if defined(BASELINE86_GENERAL_A_LZSSE)
  if (label == "LZSSE") return encode_lzsse(input);
#endif
#if defined(BASELINE86_GENERAL_A_NRV)
  if (label == "NRV") return encode_nrv(input);
#endif
#if defined(BASELINE86_GENERAL_A_BSC)
  if (label == "BSC") return encode_bsc(input);
#endif
#if defined(BASELINE86_GENERAL_A_CRUSH)
  if (label == "Crush") return encode_crush(input);
#endif
#if defined(BASELINE86_GENERAL_A_CSC)
  if (label == "CSC") return encode_csc(input);
#endif
#if defined(BASELINE86_GENERAL_A_FASTLZMA2)
  if (label == "FastLZMA2") return encode_fastlzma2(input);
#endif
  fail(label, "missing encoder dispatch");
}

Bytes decode(const std::string& label, std::span<const std::uint8_t> payload,
             std::size_t expected_size) {
  require(expected_size <= kMaximumInput, label,
          "decoded size exceeds 256 MiB limit");
  require(descriptor(label).available, label,
          "direct upstream implementation unavailable in this build");
  Bytes raw;
  const auto input = compressed_body(label, payload, expected_size, raw);
  if (input.empty()) return raw;
#if defined(BASELINE86_GENERAL_A_GIPFELI)
  if (label == "Gipfeli") return decode_gipfeli(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_GLZA)
  if (label == "GLZA") return decode_glza(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_LZO)
  if (label == "LZO1" || label == "LZO2A")
    return decode_lzo(label, input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_LZSSE)
  if (label == "LZSSE") return decode_lzsse(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_NRV)
  if (label == "NRV") return decode_nrv(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_BSC)
  if (label == "BSC") return decode_bsc(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_CRUSH)
  if (label == "Crush") return decode_crush(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_CSC)
  if (label == "CSC") return decode_csc(input, expected_size);
#endif
#if defined(BASELINE86_GENERAL_A_FASTLZMA2)
  if (label == "FastLZMA2") return decode_fastlzma2(input, expected_size);
#endif
  fail(label, "missing decoder dispatch");
}

}  // namespace native_baseline86_v1::upstream_general_a
