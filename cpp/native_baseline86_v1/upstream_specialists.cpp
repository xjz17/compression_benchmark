#include "upstream_specialists.hpp"

#include "ans_rans.hpp"
#include "arithmetic_bytes.hpp"
#include "huffman_bytes.hpp"
#include "chimp/chimp.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

// These headers are the exact CPU stages used by the vendored FPCompress
// DPratio/DPspeed programs.  They are included in a private namespace so their
// file-program helper names cannot collide with another target.
namespace fpcompress_source {
using byte = unsigned char;
inline constexpr int CS = 1024 * 16;
#include "../../external/compression/FPCompress/components/h_DIFFMS_8.h"
#include "../../external/compression/FPCompress/components/h_HCLOG_8.h"
#include "../../external/compression/FPCompress/components/h_RAZE_8.h"
#include "../../external/compression/FPCompress/components/h_RARE_8.h"
}  // namespace fpcompress_source

namespace native_baseline86_v1::upstream_specialists {
namespace {

static_assert(std::endian::native == std::endian::little,
              "specialist sources are frozen for little-endian hosts");
constexpr std::size_t kMaximum = 256ULL << 20;
constexpr std::size_t kEnvelope = 32;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

void put_u16(Bytes& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}
void put_u32(Bytes& out, std::uint32_t value) {
  for (int shift = 0; shift != 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void put_u64(Bytes& out, std::uint64_t value) {
  for (int shift = 0; shift != 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

struct Cursor {
  std::span<const std::uint8_t> input;
  std::size_t position = 0;
  std::uint8_t u8() {
    require(position < input.size(), "specialist payload truncated byte");
    return input[position++];
  }
  std::uint16_t u16() {
    std::uint16_t value = 0;
    for (int shift = 0; shift != 16; shift += 8)
      value |= static_cast<std::uint16_t>(u8()) << shift;
    return value;
  }
  std::uint32_t u32() {
    std::uint32_t value = 0;
    for (int shift = 0; shift != 32; shift += 8)
      value |= static_cast<std::uint32_t>(u8()) << shift;
    return value;
  }
  std::uint64_t u64() {
    std::uint64_t value = 0;
    for (int shift = 0; shift != 64; shift += 8)
      value |= static_cast<std::uint64_t>(u8()) << shift;
    return value;
  }
  std::span<const std::uint8_t> take(std::size_t count) {
    require(count <= input.size() - position,
            "specialist payload truncated span");
    auto result = input.subspan(position, count);
    position += count;
    return result;
  }
  void end() const {
    require(position == input.size(), "specialist payload has trailing bytes");
  }
};

std::uint32_t crc32(std::span<const std::uint8_t> data) {
  std::uint32_t crc = 0xffffffffU;
  for (const auto byte : data) {
    crc ^= byte;
    for (int bit = 0; bit != 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
  }
  return ~crc;
}

enum class Method : std::uint8_t {
  chimp = 1,
  rle = 2,
  dictionary = 3,
  corad = 4,
  buff = 5,
  ans = 6,
  huffman = 7,
  arithmetic = 8,
  fpcompress_ratio = 9,
  fpcompress_speed = 10,
};

Method method(std::string_view label) {
  if (label == "Chimp") return Method::chimp;
  if (label == "RLE") return Method::rle;
  if (label == "Dict.") return Method::dictionary;
  if (label == "CORAD") return Method::corad;
  if (label == "BUFF") return Method::buff;
  if (label == "ANS") return Method::ans;
  if (label == "Huffman") return Method::huffman;
  if (label == "Arithmetic") return Method::arithmetic;
  if (label == "FPCmp-R") return Method::fpcompress_ratio;
  if (label == "FPCmp-S") return Method::fpcompress_speed;
  throw NotApplicable("unknown upstream specialist label: " +
                      std::string(label));
}

Bytes envelope(Method id, std::span<const std::uint8_t> raw,
               std::span<const std::uint8_t> body) {
  require(raw.size() <= kMaximum && body.size() <= kMaximum,
          "specialist payload exceeds byte cap");
  Bytes out{'U', 'S', 'P', '1', static_cast<std::uint8_t>(id), 0, 0, 0};
  put_u64(out, raw.size());
  put_u64(out, body.size());
  put_u32(out, crc32(raw));
  put_u32(out, crc32(body));
  require(out.size() == kEnvelope, "specialist envelope size drift");
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

struct Opened {
  std::uint64_t raw_size;
  std::uint32_t raw_crc;
  std::span<const std::uint8_t> body;
};

Opened open(Method expected, std::span<const std::uint8_t> payload,
            std::size_t expected_bytes) {
  require(payload.size() >= kEnvelope, "specialist envelope truncated");
  Cursor cursor{payload};
  require(cursor.u8() == 'U' && cursor.u8() == 'S' && cursor.u8() == 'P' &&
              cursor.u8() == '1',
          "specialist envelope bad magic");
  require(cursor.u8() == static_cast<std::uint8_t>(expected),
          "specialist envelope method mismatch");
  require(cursor.u8() == 0 && cursor.u8() == 0 && cursor.u8() == 0,
          "specialist envelope reserved bytes");
  const auto raw_size = cursor.u64();
  const auto body_size = cursor.u64();
  const auto raw_crc = cursor.u32();
  const auto body_crc = cursor.u32();
  require(raw_size == expected_bytes && raw_size <= kMaximum,
          "specialist original size mismatch");
  require(body_size == payload.size() - kEnvelope,
          "specialist body size mismatch");
  auto body = cursor.take(static_cast<std::size_t>(body_size));
  cursor.end();
  require(crc32(body) == body_crc, "specialist body checksum mismatch");
  return {raw_size, raw_crc, body};
}

void put_varint(Bytes& out, std::uint64_t value) {
  do {
    auto byte = static_cast<std::uint8_t>(value & 0x7fU);
    value >>= 7;
    out.push_back(static_cast<std::uint8_t>(byte | (value ? 0x80U : 0U)));
  } while (value);
}

std::uint64_t get_varint(Cursor& cursor) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 10; ++index) {
    const auto byte = cursor.u8();
    require(index != 9 || byte <= 1, "specialist varint overflow");
    value |= static_cast<std::uint64_t>(byte & 0x7fU) << (index * 7);
    if (!(byte & 0x80U)) {
      require(index == 0 || (byte & 0x7fU) != 0,
              "specialist noncanonical varint");
      return value;
    }
  }
  throw std::invalid_argument("specialist varint overflow");
}

Bytes rle_encode(std::span<const std::uint8_t> raw) {
  Bytes body;
  for (std::size_t first = 0; first < raw.size();) {
    std::size_t last = first + 1;
    while (last < raw.size() && raw[last] == raw[first]) ++last;
    put_varint(body, last - first);
    body.push_back(raw[first]);
    first = last;
  }
  return body;
}

Bytes rle_decode(std::span<const std::uint8_t> body, std::size_t expected) {
  Cursor cursor{body};
  Bytes raw;
  raw.reserve(expected);
  int previous = -1;
  while (cursor.position != body.size()) {
    const auto count = get_varint(cursor);
    require(count != 0 && count <= expected - raw.size(),
            "RLE invalid run length");
    const auto value = cursor.u8();
    require(previous < 0 || value != previous,
            "RLE adjacent equal runs are noncanonical");
    raw.insert(raw.end(), static_cast<std::size_t>(count), value);
    previous = value;
  }
  require(raw.size() == expected, "RLE decoded size mismatch");
  return raw;
}

Bytes dictionary_encode(std::span<const std::uint8_t> raw) {
  std::array<int, 256> ids;
  ids.fill(-1);
  Bytes dictionary;
  std::vector<std::uint8_t> symbols;
  symbols.reserve(raw.size());
  for (const auto value : raw) {
    if (ids[value] < 0) {
      ids[value] = static_cast<int>(dictionary.size());
      dictionary.push_back(value);
    }
    symbols.push_back(static_cast<std::uint8_t>(ids[value]));
  }
  const unsigned bits = dictionary.size() <= 1
                            ? 0
                            : std::bit_width(dictionary.size() - 1);
  Bytes body;
  put_u16(body, static_cast<std::uint16_t>(dictionary.size()));
  body.push_back(static_cast<std::uint8_t>(bits));
  body.insert(body.end(), dictionary.begin(), dictionary.end());
  const auto packed = (raw.size() * bits + 7) / 8;
  const auto start = body.size();
  body.resize(start + packed, 0);
  std::size_t bit_position = 0;
  for (const auto symbol : symbols) {
    for (unsigned bit = 0; bit != bits; ++bit) {
      if ((symbol >> bit) & 1U)
        body[start + bit_position / 8] |=
            static_cast<std::uint8_t>(1U << (bit_position % 8));
      ++bit_position;
    }
  }
  return body;
}

Bytes dictionary_decode(std::span<const std::uint8_t> body,
                        std::size_t expected) {
  Cursor cursor{body};
  const auto count = cursor.u16();
  const auto bits = cursor.u8();
  require(count <= 256, "dictionary symbol count exceeds byte alphabet");
  require(bits == (count <= 1 ? 0U :
                    std::bit_width(static_cast<unsigned>(count - 1))),
          "dictionary bit width is noncanonical");
  auto dictionary = cursor.take(count);
  std::array<bool, 256> seen{};
  for (const auto value : dictionary) {
    require(!seen[value], "dictionary contains duplicate symbols");
    seen[value] = true;
  }
  require((expected == 0) == (count == 0),
          "dictionary empty/nonempty mismatch");
  const auto packed_size = (expected * bits + 7) / 8;
  auto packed = cursor.take(packed_size);
  cursor.end();
  if (expected * bits % 8 && !packed.empty())
    require((packed.back() >> (expected * bits % 8)) == 0,
            "dictionary nonzero packed padding");
  Bytes raw;
  raw.reserve(expected);
  std::size_t bit_position = 0;
  for (std::size_t index = 0; index != expected; ++index) {
    unsigned symbol = 0;
    for (unsigned bit = 0; bit != bits; ++bit) {
      symbol |= ((packed[bit_position / 8] >> (bit_position % 8)) & 1U)
                << bit;
      ++bit_position;
    }
    require(symbol < dictionary.size(), "dictionary id out of range");
    raw.push_back(dictionary[symbol]);
  }
  return raw;
}

Bytes corad_constant_encode(std::span<const std::uint8_t> raw) {
  require(raw.size() % 8 == 0, "CORAD direct subset requires float64 words");
  if (raw.empty())
    throw NotApplicable("CORAD source does not admit an empty column");
  double first = 0;
  std::memcpy(&first, raw.data(), 8);
  if (!std::isfinite(first))
    throw NotApplicable(
        "CORAD constant fallback is not reached for nonfinite values");
  for (std::size_t offset = 8; offset != raw.size(); offset += 8)
    if (!std::equal(raw.begin(), raw.begin() + 8, raw.begin() + offset))
      throw NotApplicable(
          "CORAD is error-bounded for nonconstant inputs; only its exact "
          "constant-column branch is admitted in a bit-exact comparison");
  Bytes plain;
  put_u64(plain, raw.size() / 8);
  plain.insert(plain.end(), raw.begin(), raw.begin() + 8);
  const auto bound = compressBound(static_cast<uLong>(plain.size()));
  Bytes body(static_cast<std::size_t>(bound));
  uLongf compressed_size = bound;
  require(compress2(body.data(), &compressed_size, plain.data(),
                    static_cast<uLong>(plain.size()), 6) == Z_OK,
          "CORAD zlib level-6 compression failed");
  body.resize(static_cast<std::size_t>(compressed_size));
  return body;
}

Bytes corad_constant_decode(std::span<const std::uint8_t> body,
                            std::size_t expected) {
  require(expected != 0 && expected % 8 == 0,
          "CORAD decoded length is outside the exact constant subset");
  Bytes plain(16);
  uLongf plain_size = static_cast<uLongf>(plain.size());
  require(body.size() <= std::numeric_limits<uLong>::max() &&
              uncompress(plain.data(), &plain_size, body.data(),
                         static_cast<uLong>(body.size())) == Z_OK &&
              plain_size == plain.size(),
          "CORAD zlib level-6 payload is invalid");
  Cursor cursor{plain};
  const auto count = cursor.u64();
  require(count <= kMaximum / 8 && count * 8 == expected,
          "CORAD constant count mismatch");
  Bytes value;
  if (count) {
    auto word = cursor.take(8);
    value.assign(word.begin(), word.end());
  }
  cursor.end();
  Bytes raw;
  raw.reserve(expected);
  for (std::uint64_t index = 0; index != count; ++index)
    raw.insert(raw.end(), value.begin(), value.end());
  return raw;
}

std::int64_t buff_fixed(double value) {
  constexpr int precision_exp = -15;  // exponent of BUFF scale=10000 bound
  constexpr int decimal_bits = 14;
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, 8);
  const int exponent = static_cast<int>((bits >> 52) & 0x7ffU) - 1023;
  if (exponent < precision_exp) return 0;
  const int shift = 63 - exponent - decimal_bits;
  if (shift < 0 || shift > 63)
    throw NotApplicable("BUFF scale=10000 exponent is outside native wire");
  std::uint64_t fixed = ((bits << 11) | (1ULL << 63)) >> shift;
  if (bits >> 63) fixed = ~(fixed - 1);
  std::int64_t result = 0;
  std::memcpy(&result, &fixed, 8);
  return result;
}

Bytes buff_encode(std::span<const std::uint8_t> raw) {
  require(raw.size() % 8 == 0, "BUFF requires float64 words");
  const auto count = raw.size() / 8;
  if (!count) throw NotApplicable("BUFF upstream profile rejects empty input");
  require(count <= std::numeric_limits<std::uint32_t>::max(),
          "BUFF row count exceeds uint32");
  std::vector<std::int64_t> fixed;
  fixed.reserve(count);
  for (std::size_t index = 0; index != count; ++index) {
    double value = 0;
    std::memcpy(&value, raw.data() + index * 8, 8);
    if (!std::isfinite(value))
      throw NotApplicable(
          "BUFF bounded float wire does not preserve nonfinite payload bits");
    fixed.push_back(buff_fixed(value));
  }
  const auto [minimum, maximum] =
      std::minmax_element(fixed.begin(), fixed.end());
  const auto ordered = [](std::int64_t value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, 8);
    return bits ^ (1ULL << 63);
  };
  const auto base_ordered = ordered(*minimum);
  const auto delta = ordered(*maximum) - base_ordered;
  const unsigned width = std::max<unsigned>(14, std::bit_width(delta));
  if (width > 32)
    throw NotApplicable(
        "BUFF vendored decoder supports at most four byte slices");
  const unsigned slices = (width + 7) / 8;
  const unsigned padding = slices * 8 - width;
  Bytes body;
  std::uint64_t base_bits = 0;
  std::memcpy(&base_bits, &*minimum, 8);
  put_u64(body, base_bits);
  put_u32(body, static_cast<std::uint32_t>(count));
  body.push_back(static_cast<std::uint8_t>(width));
  for (unsigned slice = 0; slice != slices; ++slice) {
    const unsigned shift = (slices - 1 - slice) * 8;
    for (const auto value : fixed) {
      const auto offset = ordered(value) - base_ordered;
      const auto aligned = padding ? offset << padding : offset;
      body.push_back(static_cast<std::uint8_t>(aligned >> shift) ^ 0x80U);
    }
  }
  return body;
}

Bytes buff_decode(std::span<const std::uint8_t> body, std::size_t expected) {
  require(expected != 0, "BUFF upstream profile rejects empty input");
  Cursor cursor{body};
  std::uint64_t base_bits = cursor.u64();
  std::int64_t base = 0;
  std::memcpy(&base, &base_bits, 8);
  const auto ordered = [](std::int64_t value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, 8);
    return bits ^ (1ULL << 63);
  };
  const auto base_ordered = ordered(base);
  const auto count = cursor.u32();
  const auto width = cursor.u8();
  require(static_cast<std::uint64_t>(count) * 8 == expected,
          "BUFF decoded count mismatch");
  require(width >= 14 && width <= 32, "BUFF invalid fixed width");
  const unsigned slices = (width + 7) / 8;
  const unsigned padding = slices * 8 - width;
  auto packed = cursor.take(static_cast<std::size_t>(count) * slices);
  cursor.end();
  std::vector<std::uint64_t> offsets(count, 0);
  for (unsigned slice = 0; slice != slices; ++slice)
    for (std::size_t index = 0; index != count; ++index)
      offsets[index] = (offsets[index] << 8) |
                       static_cast<std::uint8_t>(packed[slice * count + index] ^
                                                 0x80U);
  Bytes raw;
  raw.resize(expected);
  for (std::size_t index = 0; index != count; ++index) {
    if (padding) {
      require((offsets[index] & ((1ULL << padding) - 1)) == 0,
              "BUFF nonzero fixed padding");
      offsets[index] >>= padding;
    }
    require(offsets[index] <=
                std::numeric_limits<std::uint64_t>::max() - base_ordered,
            "BUFF reconstructed fixed integer overflow");
    const auto fixed_ordered = base_ordered + offsets[index];
    const auto fixed_bits = fixed_ordered ^ (1ULL << 63);
    std::int64_t fixed = 0;
    std::memcpy(&fixed, &fixed_bits, 8);
    const double value = std::ldexp(static_cast<double>(fixed), -14);
    std::memcpy(raw.data() + index * 8, &value, 8);
  }
  return raw;
}

Bytes chimp_encode(std::span<const std::uint8_t> raw) {
  require(raw.size() % 8 == 0, "Chimp requires float64 words");
  const auto count = raw.size() / 8;
  Bytes body;
  put_u64(body, count);
  if (!count) {
    put_u64(body, 0);
    put_u64(body, 0);
    put_u64(body, 0);
    return body;
  }
  Bytes data(count * 9 + 16, 0);
  Bytes flags((count + 3) / 4 + 8, 0);
  Bytes leading((count * 3 + 7) / 8 + 8, 0);
  alp_bench::ChimpCompressionState<std::uint64_t, false> state;
  state.Reset();
  state.output.SetStream(data.data());
  state.flag_buffer.SetBuffer(flags.data());
  state.leading_zero_buffer.SetBuffer(leading.data());
  for (std::size_t index = 0; index != count; ++index) {
    std::uint64_t word = 0;
    std::memcpy(&word, raw.data() + index * 8, 8);
    alp_bench::ChimpCompression<std::uint64_t, false>::Store(word, state);
  }
  state.Flush();
  alp_bench::ChimpCompression<std::uint64_t, false>::Flush(state);
  const auto data_bits = state.output.BitsWritten();
  const auto flag_count = state.flag_buffer.FlagCount();
  const auto leading_count = state.leading_zero_buffer.GetCount();
  require(flag_count == count - 1 && leading_count <= count - 1,
          "Chimp source buffer count mismatch");
  const auto data_bytes = (data_bits + 7) / 8;
  const auto flag_bytes = (flag_count + 3) / 4;
  const auto leading_bytes = (leading_count * 3 + 7) / 8;
  put_u64(body, data_bits);
  put_u64(body, flag_count);
  put_u64(body, leading_count);
  body.insert(body.end(), data.begin(), data.begin() + data_bytes);
  body.insert(body.end(), flags.begin(), flags.begin() + flag_bytes);
  body.insert(body.end(), leading.begin(), leading.begin() + leading_bytes);
  return body;
}

Bytes chimp_decode(std::span<const std::uint8_t> body, std::size_t expected) {
  Cursor cursor{body};
  const auto count = cursor.u64();
  const auto data_bits = cursor.u64();
  const auto flag_count = cursor.u64();
  const auto leading_count = cursor.u64();
  require(count <= kMaximum / 8 && count * 8 == expected,
          "Chimp decoded count mismatch");
  require(flag_count == (count ? count - 1 : 0) &&
              leading_count <= flag_count,
          "Chimp side-buffer count mismatch");
  require(data_bits <= count * 70 + 64,
          "Chimp data bit count exceeds bound");
  auto data_wire = cursor.take((data_bits + 7) / 8);
  auto flag_wire = cursor.take((flag_count + 3) / 4);
  auto leading_wire = cursor.take((leading_count * 3 + 7) / 8);
  cursor.end();
  if (!count) {
    require(data_bits == 0 && flag_count == 0 && leading_count == 0,
            "Chimp empty stream has state");
    return {};
  }
  if (data_bits % 8)
    require((data_wire.back() & ((1U << (8 - data_bits % 8)) - 1U)) == 0,
            "Chimp nonzero data padding");
  Bytes data(data_wire.begin(), data_wire.end());
  data.resize(data.size() + 8, 0);
  Bytes flags_bytes(flag_wire.begin(), flag_wire.end());
  flags_bytes.resize(flags_bytes.size() + 4, 0);
  Bytes leading_bytes(leading_wire.begin(), leading_wire.end());
  leading_bytes.resize(leading_bytes.size() + 4, 0);
  alp_bench::FlagBuffer<false> flag_buffer;
  flag_buffer.SetBuffer(flags_bytes.data());
  alp_bench::LeadingZeroBuffer<false> leading_buffer;
  leading_buffer.SetBuffer(leading_bytes.data());
  std::vector<alp_bench::ChimpConstants::Flags> flags(count);
  flags[0] = alp_bench::ChimpConstants::Flags::VALUE_IDENTICAL;
  for (std::size_t index = 1; index != count; ++index)
    flags[index] = static_cast<alp_bench::ChimpConstants::Flags>(
        flag_buffer.Extract());
  std::vector<std::uint8_t> leading(leading_count);
  for (auto& value : leading)
    value = alp_bench::ChimpConstants::Decompression::LEADING_REPRESENTATION
        [leading_buffer.Extract()];
  alp_bench::ChimpDecompressionState<std::uint64_t> state;
  state.input.SetStream(data.data());
  std::uint32_t leading_index = 0;
  Bytes raw(expected);
  for (std::size_t index = 0; index != count; ++index) {
    const auto word = alp_bench::ChimpDecompression<std::uint64_t>::Load(
        flags[index], leading.data(), leading_index, state);
    std::memcpy(raw.data() + index * 8, &word, 8);
  }
  require(state.input.index == data_bits && leading_index == leading_count,
          "Chimp side-buffer consumption mismatch");
  return raw;
}

enum class FpVariant { ratio, speed };

std::uint32_t fp_hash(std::uint64_t value) {
  return static_cast<std::uint32_t>((value >> 32) ^ value);
}

Bytes fp_ratio_preprocess(std::span<const std::uint8_t> raw) {
  require(raw.size() % 8 == 0, "FPCmp-R requires float64 words");
  const auto count = raw.size() / 8;
  require(count <= std::numeric_limits<std::uint32_t>::max(),
          "FPCmp-R word count exceeds source wire");
  std::vector<std::uint64_t> values(count);
  if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  std::vector<std::uint64_t> positions(count);
  for (std::size_t index = 0; index != count; ++index) {
    std::uint32_t context_hash = 0;
    for (std::size_t back = 1; back <= 3 && back <= index; ++back)
      context_hash ^= fp_hash(values[index - back]);
    positions[index] = (static_cast<std::uint64_t>(context_hash) << 32) |
                       static_cast<std::uint32_t>(index);
  }
  std::sort(positions.begin(), positions.end());
  std::vector<std::uint64_t> preprocessed(count * 2, 0);
  for (std::size_t sorted = 0; sorted != count; ++sorted) {
    const auto position = static_cast<std::uint32_t>(positions[sorted]);
    const auto hash = static_cast<std::uint32_t>(positions[sorted] >> 32);
    auto value = values[position];
    std::uint64_t distance = 0;
    for (unsigned back = 1; back <= 4 && back <= sorted; ++back) {
      const auto previous = positions[sorted - back];
      if (static_cast<std::uint32_t>(previous >> 32) != hash) break;
      const auto previous_position = static_cast<std::uint32_t>(previous);
      if (value == values[previous_position]) {
        distance = position - previous_position;
        value = 0;
        break;
      }
    }
    preprocessed[position] = value;
    preprocessed[position + count] = distance;
  }
  Bytes result(preprocessed.size() * 8);
  if (!result.empty()) std::memcpy(result.data(), preprocessed.data(), result.size());
  return result;
}

Bytes fp_ratio_postprocess(std::span<const std::uint8_t> input,
                           std::size_t expected) {
  require(input.size() == expected * 2 && expected % 8 == 0,
          "FPCmp-R preprocessed size mismatch");
  const auto count = expected / 8;
  std::vector<std::uint64_t> words(count * 2);
  if (!input.empty()) std::memcpy(words.data(), input.data(), input.size());
  for (std::size_t index = 0; index != count; ++index) {
    const auto distance = words[index + count];
    if (distance) {
      require(distance <= index, "FPCmp-R invalid dictionary distance");
      words[index] = words[index - distance];
    }
  }
  Bytes raw(expected);
  if (!raw.empty()) std::memcpy(raw.data(), words.data(), raw.size());
  return raw;
}

Bytes fpcompress_encode(std::span<const std::uint8_t> raw, FpVariant variant) {
  require(raw.size() % 8 == 0, "FPCompress double profile needs float64 words");
  Bytes input = variant == FpVariant::ratio ? fp_ratio_preprocess(raw)
                                             : Bytes(raw.begin(), raw.end());
  const auto chunks = (input.size() + fpcompress_source::CS - 1) /
                      fpcompress_source::CS;
  require(chunks <= std::numeric_limits<std::uint32_t>::max(),
          "FPCompress chunk count overflow");
  Bytes body;
  put_u64(body, input.size());
  put_u32(body, static_cast<std::uint32_t>(chunks));
  for (std::size_t chunk = 0; chunk != chunks; ++chunk) {
    const auto base = chunk * fpcompress_source::CS;
    const auto raw_size = std::min<std::size_t>(fpcompress_source::CS,
                                                input.size() - base);
    alignas(8) std::array<fpcompress_source::byte, fpcompress_source::CS> a{};
    alignas(8) std::array<fpcompress_source::byte, fpcompress_source::CS> b{};
    std::memcpy(a.data(), input.data() + base, raw_size);
    auto* current = a.data();
    auto* scratch = b.data();
    int size = static_cast<int>(raw_size);
    bool good = fpcompress_source::h_DIFFMS_8(size, current, scratch);
    if (good) std::swap(current, scratch);
    if (variant == FpVariant::ratio) {
      if (good) good = fpcompress_source::h_RAZE_8(size, current, scratch);
      if (good) std::swap(current, scratch);
      if (good) good = fpcompress_source::h_RARE_8(size, current, scratch);
      if (good) std::swap(current, scratch);
    } else {
      if (good) good = fpcompress_source::h_HCLOG_8(size, current, scratch);
      if (good) std::swap(current, scratch);
    }
    const bool compressed = good && size < static_cast<int>(raw_size);
    put_u16(body, static_cast<std::uint16_t>(raw_size));
    put_u16(body, static_cast<std::uint16_t>(compressed ? size : raw_size));
    body.push_back(compressed ? 1 : 0);
    if (compressed)
      body.insert(body.end(), current, current + size);
    else
      body.insert(body.end(), input.begin() + base,
                  input.begin() + base + raw_size);
  }
  return body;
}

Bytes fpcompress_decode(std::span<const std::uint8_t> body,
                        std::size_t expected, FpVariant variant) {
  Cursor cursor{body};
  const auto preprocessed_size = cursor.u64();
  const auto count = cursor.u32();
  const auto wanted = variant == FpVariant::ratio ? expected * 2 : expected;
  require(expected <= kMaximum && preprocessed_size == wanted,
          "FPCompress preprocessed length mismatch");
  const auto wanted_chunks =
      (wanted + fpcompress_source::CS - 1) / fpcompress_source::CS;
  require(count == wanted_chunks, "FPCompress chunk count mismatch");
  Bytes preprocessed;
  preprocessed.reserve(wanted);
  for (std::uint32_t chunk = 0; chunk != count; ++chunk) {
    const auto raw_size = cursor.u16();
    const auto encoded_size = cursor.u16();
    const auto compressed = cursor.u8();
    const auto expected_chunk = std::min<std::size_t>(
        fpcompress_source::CS, wanted - preprocessed.size());
    require(raw_size == expected_chunk && encoded_size <= raw_size &&
                (compressed == 0 || compressed == 1),
            "FPCompress invalid chunk header");
    auto wire = cursor.take(encoded_size);
    if (!compressed) {
      require(encoded_size == raw_size, "FPCompress raw chunk size mismatch");
      preprocessed.insert(preprocessed.end(), wire.begin(), wire.end());
      continue;
    }
    require(encoded_size < raw_size, "FPCompress noncanonical compressed chunk");
    alignas(8) std::array<fpcompress_source::byte, fpcompress_source::CS> a{};
    alignas(8) std::array<fpcompress_source::byte, fpcompress_source::CS> b{};
    std::memcpy(a.data(), wire.data(), wire.size());
    auto* current = a.data();
    auto* scratch = b.data();
    int size = encoded_size;
    if (variant == FpVariant::ratio) {
      fpcompress_source::h_iRARE_8(size, current, scratch);
      std::swap(current, scratch);
      fpcompress_source::h_iRAZE_8(size, current, scratch);
      std::swap(current, scratch);
    } else {
      fpcompress_source::h_iHCLOG_8(size, current, scratch);
      std::swap(current, scratch);
    }
    fpcompress_source::h_iDIFFMS_8(size, current, scratch);
    std::swap(current, scratch);
    require(size == raw_size, "FPCompress decoded chunk size mismatch");
    preprocessed.insert(preprocessed.end(), current, current + size);
  }
  cursor.end();
  require(preprocessed.size() == wanted,
          "FPCompress reconstructed length mismatch");
  return variant == FpVariant::ratio
             ? fp_ratio_postprocess(preprocessed, expected)
             : preprocessed;
}

Bytes encode_body(Method id, std::span<const std::uint8_t> raw) {
  switch (id) {
    case Method::chimp: return chimp_encode(raw);
    case Method::rle: return rle_encode(raw);
    case Method::dictionary: return dictionary_encode(raw);
    case Method::corad: return corad_constant_encode(raw);
    case Method::buff: {
      auto body = buff_encode(raw);
      auto restored = buff_decode(body, raw.size());
      if (!std::equal(raw.begin(), raw.end(), restored.begin(), restored.end()))
        throw NotApplicable(
            "BUFF scale=10000 changed an IEEE-754 bit; block is outside the "
            "bit-exact subset");
      return body;
    }
    case Method::ans: return ans_rans::encode(raw);
    case Method::huffman: return huffman_bytes::encode(raw);
    case Method::arithmetic: return arithmetic_bytes::encode(raw);
    case Method::fpcompress_ratio:
      return fpcompress_encode(raw, FpVariant::ratio);
    case Method::fpcompress_speed:
      return fpcompress_encode(raw, FpVariant::speed);
  }
  throw std::logic_error("unknown specialist method");
}

Bytes decode_body(Method id, std::span<const std::uint8_t> body,
                  std::size_t expected) {
  switch (id) {
    case Method::chimp: return chimp_decode(body, expected);
    case Method::rle: return rle_decode(body, expected);
    case Method::dictionary: return dictionary_decode(body, expected);
    case Method::corad: return corad_constant_decode(body, expected);
    case Method::buff: return buff_decode(body, expected);
    case Method::ans: return ans_rans::decode(body);
    case Method::huffman: return huffman_bytes::decode(body);
    case Method::arithmetic: return arithmetic_bytes::decode(body);
    case Method::fpcompress_ratio:
      return fpcompress_decode(body, expected, FpVariant::ratio);
    case Method::fpcompress_speed:
      return fpcompress_decode(body, expected, FpVariant::speed);
  }
  throw std::logic_error("unknown specialist method");
}

}  // namespace

const std::vector<Profile>& profiles() {
  static const std::vector<Profile> value = {
      {"Chimp", "alp_publication_chimp", "external/compression/ALP/publication/source_code/include/chimp", "Chimp_float64_separate_data_flags_leading_zero_buffers", {"float64"}, Admission::direct_upstream, "Exact IEEE-754 bits; this is Chimp, distinct from the existing Chimp128 implementation."},
      {"RLE", "baseline86_byte_run_length", "cpp/native_baseline86_v1/upstream_specialists.cpp", "canonical_byte_run_value_varint_count", {"int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"}, Admission::direct_project_generic, "The label is generic; this fixed profile is actual byte RLE and never selects bit-pack/raw modes."},
      {"Dict.", "baseline86_first_seen_byte_dictionary", "cpp/native_baseline86_v1/upstream_specialists.cpp", "first_seen_byte_dictionary_minimal_bitpacked_ids", {"int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"}, Admission::direct_project_generic, "The label is generic; source/profile identify the concrete dictionary algorithm."},
      {"CORAD", "corad_cpp_exact_constant_branch", "cpp/native_baseline86_v1/upstream_specialists.cpp; branch provenance cpp/src/corad_tricklet_float_codec.cpp", "constant_float64_count_exact_word_zlib6", {"float64"}, Admission::direct_exact_subset, "CORAD is error-bounded for nonconstant inputs. Only its nonempty finite bitwise-constant fallback is admitted; all other blocks fail as unsupported rather than using repair words."},
      {"BUFF", "buff_bounded_float_scale10000", "cpp/native_baseline86_v1/upstream_specialists.cpp; external/operator_ablation_sources/buff-rs; cpp/src/buff_kernel/buff_precision_bound.cpp", "scale10000_decimal_bits14_byte_sliced_max4", {"float64"}, Admission::direct_exact_subset, "Original BUFF is precision-bounded. Encoding is admitted only when its at-most-four-slice decode reproduces every IEEE-754 bit; no raw64 fallback is used."},
      {"ANS", "project_static_rans", "cpp/src/ans_rans.cpp", "static_order0_rans_prob_bits15", {"int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"}, Admission::direct_project_generic, "The public label is generic; this row names the concrete static rANS profile."},
      {"Huffman", "project_static_huffman", "cpp/src/huffman_bytes.cpp", "static_order0_frequency_tree", {"int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"}, Admission::direct_project_generic, "The public label is generic; this row names the concrete deterministic tree profile."},
      {"Arithmetic", "project_static_arithmetic", "cpp/src/arithmetic_bytes.cpp", "static_order0_range_coder_15bit_model", {"int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"}, Admission::direct_project_generic, "The public label is generic; this row names the concrete static range-coder profile."},
      {"FPCmp-R", "fpcompress_1_2_dpratio", "external/compression/FPCompress/double_src/ratio-*.cpp", "FCMp_8_DIFFMS_8_RAZE_8_RARE_8_serial_memory", {"float64"}, Admission::direct_upstream, "File main/OpenMP scheduling was replaced by a bounded single-thread memory adapter; algorithm stages and 16 KiB chunk fallback semantics are retained."},
      {"FPCmp-S", "fpcompress_1_2_dpspeed", "external/compression/FPCompress/double_src/speed-*.cpp", "DIFFMS_8_HCLOG_8_serial_memory", {"float64"}, Admission::direct_upstream, "File main/OpenMP scheduling was replaced by a bounded single-thread memory adapter; algorithm stages and 16 KiB chunk fallback semantics are retained."},
  };
  return value;
}

const Profile& profile(std::string_view label) {
  const auto& all = profiles();
  const auto iterator = std::find_if(all.begin(), all.end(), [&](const auto& item) {
    return item.label == label;
  });
  if (iterator == all.end())
    throw NotApplicable("unknown upstream specialist profile: " +
                        std::string(label));
  return *iterator;
}

Bytes encode(std::string_view label, std::span<const std::uint8_t> raw) {
  (void)profile(label);
  require(raw.size() <= kMaximum, "specialist original bytes exceed cap");
  const auto id = method(label);
  const auto body = encode_body(id, raw);
  return envelope(id, raw, body);
}

Bytes decode(std::string_view label, std::span<const std::uint8_t> payload,
             std::size_t expected_bytes) {
  (void)profile(label);
  const auto id = method(label);
  const auto opened = open(id, payload, expected_bytes);
  auto raw = decode_body(id, opened.body, expected_bytes);
  require(raw.size() == expected_bytes,
          "specialist reconstructed size mismatch");
  require(crc32(raw) == opened.raw_crc,
          "specialist reconstructed checksum mismatch");
  return raw;
}

}  // namespace native_baseline86_v1::upstream_specialists
