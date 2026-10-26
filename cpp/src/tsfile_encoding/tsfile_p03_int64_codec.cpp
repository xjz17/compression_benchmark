#include "tsfile_encode_p03.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace tsfile::p03 {

namespace {

constexpr int kValueBits64 = 64;
constexpr int kLeadingZeroBits64 = 6;
constexpr int kMeaningfulXorBits64 = 6;
constexpr std::int64_t kGorillaEndLong = std::numeric_limits<std::int64_t>::min();
constexpr std::uint64_t kGorillaEndDoubleBits = 0x7ff8000000000000ull;

constexpr int kChimpPreviousValues64 = 128;
constexpr int kChimpPrevLog2_64 = 7;
constexpr int kChimpThreshold64 = 6 + kChimpPrevLog2_64;
constexpr int kChimpSetLsb64 = (1 << (kChimpThreshold64 + 1)) - 1;
constexpr int kCaseZeroMetaLen64 = kChimpPrevLog2_64 + 2;
constexpr int kCaseOneMetaLen64 = kChimpPrevLog2_64 + 11;
constexpr int kCaseOneMetaLen64Decoder = kChimpPrevLog2_64 + 9;

static const int kChimpDecodeLeading[] = {0, 8, 12, 16, 18, 20, 22, 24};

struct GorillaBitStream {
  int stored_leading_zeros = std::numeric_limits<int>::max();
  int stored_trailing_zeros = 0;
  std::uint8_t buffer = 0;
  int bits_left = 8;
  std::vector<std::uint8_t> out;

  void flip_byte() {
    if (bits_left == 0) {
      out.push_back(buffer);
      buffer = 0;
      bits_left = 8;
    }
  }
  void skip_bit() {
    bits_left--;
    flip_byte();
  }
  void write_bit() {
    buffer = static_cast<std::uint8_t>(buffer | static_cast<std::uint8_t>(1u << (bits_left - 1)));
    bits_left--;
    flip_byte();
  }
  void write_bits(std::uint64_t value, int bits) {
    while (bits > 0) {
      int shift = bits - bits_left;
      if (shift >= 0) {
        buffer = static_cast<std::uint8_t>(
            buffer | static_cast<std::uint8_t>((value >> shift) & ((1ull << bits_left) - 1)));
        bits -= bits_left;
        bits_left = 0;
      } else {
        shift = bits_left - bits;
        buffer = static_cast<std::uint8_t>(buffer | static_cast<std::uint8_t>(value << shift));
        bits_left -= bits;
        bits = 0;
      }
      flip_byte();
    }
  }
  void finish_stream() {
    bits_left = 0;
    flip_byte();
  }
};

struct LongGorillaEncoder {
  GorillaBitStream b;
  bool first_written = false;
  std::int64_t stored_value = 0;

  void write_first(std::int64_t value) {
    stored_value = value;
    b.write_bits(static_cast<std::uint64_t>(value), kValueBits64);
  }
  void write_existing_leading(std::uint64_t xorv) {
    b.skip_bit();
    int significant_bits = kValueBits64 - b.stored_leading_zeros - b.stored_trailing_zeros;
    b.write_bits(xorv >> b.stored_trailing_zeros, significant_bits);
  }
  void write_new_leading(std::uint64_t xorv, int leading_zeros, int trailing_zeros) {
    b.write_bit();
    int significant_bits = kValueBits64 - leading_zeros - trailing_zeros;
    b.write_bits(static_cast<std::uint64_t>(leading_zeros), kLeadingZeroBits64);
    b.write_bits(static_cast<std::uint64_t>(significant_bits - 1), kMeaningfulXorBits64);
    b.write_bits(xorv >> trailing_zeros, significant_bits);
    b.stored_leading_zeros = leading_zeros;
    b.stored_trailing_zeros = trailing_zeros;
  }
  __attribute__((noinline)) void compress_value(std::int64_t value) {
    const std::uint64_t xors = static_cast<std::uint64_t>(stored_value ^ value);
    stored_value = value;
    if (xors == 0) {
      b.skip_bit();
    } else {
      b.write_bit();
      const int lz = __builtin_clzll(xors);
      const int tz = __builtin_ctzll(xors);
      if (lz >= b.stored_leading_zeros && tz >= b.stored_trailing_zeros) {
        write_existing_leading(xors);
      } else {
        write_new_leading(xors, lz, tz);
      }
    }
  }
  void encode(std::int64_t value) {
    if (first_written) {
      compress_value(value);
    } else {
      write_first(value);
      first_written = true;
    }
  }
  void flush_ending(std::int64_t end_val) {
    encode(end_val);
    b.finish_stream();
  }
};

static const std::int16_t kLeadingRepresentation[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

struct DoubleGorillaEncoder : LongGorillaEncoder {
  void encode_d(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(double));
    encode(static_cast<std::int64_t>(bits));
  }
  void flush_ending_double() {
    flush_ending(static_cast<std::int64_t>(kGorillaEndDoubleBits));
  }
};

struct LongChimpEncoder {
  GorillaBitStream b;
  bool first_written = false;
  std::int64_t stored_values[kChimpPreviousValues64]{};
  int indices[1 << (kChimpThreshold64 + 1)]{};
  int index = 0;
  int current = 0;

  static int leading_round_u64(std::uint64_t xorv) {
    const int nlz = __builtin_clzll(xorv);
    // Must be 64 entries (index = numberOfLeadingZeros); same as Java LongChimpEncoder.LEADING_ROUND.
    static const int kLeadingRound[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};
    static_assert(sizeof(kLeadingRound) / sizeof(kLeadingRound[0]) == 64,
                  "LongChimp LEADING_ROUND must match Java (64 entries)");
    return kLeadingRound[nlz < 64 ? nlz : 63];
  }

  void write_first(std::int64_t value) {
    stored_values[current] = value;
    b.write_bits(static_cast<std::uint64_t>(value), kValueBits64);
    indices[static_cast<int>(value & kChimpSetLsb64)] = index;
  }

  __attribute__((noinline)) void compress_value(std::int64_t value) {
    const int key = static_cast<int>(value & kChimpSetLsb64);
    std::uint64_t xorv = 0;
    int previous_index = 0;
    int trailing_zeros = 0;
    const int curr_index = indices[key];
    if ((index - curr_index) < kChimpPreviousValues64) {
      const std::uint64_t temp_xor =
          static_cast<std::uint64_t>(value ^ stored_values[curr_index % kChimpPreviousValues64]);
      trailing_zeros = (temp_xor == 0) ? kValueBits64 : __builtin_ctzll(temp_xor);
      if (trailing_zeros > kChimpThreshold64) {
        previous_index = curr_index % kChimpPreviousValues64;
        xorv = temp_xor;
      } else {
        previous_index = index % kChimpPreviousValues64;
        xorv = static_cast<std::uint64_t>(stored_values[previous_index] ^ value);
      }
    } else {
      previous_index = index % kChimpPreviousValues64;
      xorv = static_cast<std::uint64_t>(stored_values[previous_index] ^ value);
    }

    if (xorv == 0) {
      b.write_bits(static_cast<std::uint64_t>(previous_index), kCaseZeroMetaLen64);
      b.stored_leading_zeros = kValueBits64 + 1;
    } else {
      const int leading_zeros = leading_round_u64(xorv);
      if (trailing_zeros > kChimpThreshold64) {
        const int significant_bits = kValueBits64 - leading_zeros - trailing_zeros;
        const std::uint64_t meta =
            512ull * (kChimpPreviousValues64 + previous_index) +
            64ull * static_cast<std::uint64_t>(kLeadingRepresentation[leading_zeros]) +
            static_cast<std::uint64_t>(significant_bits);
        b.write_bits(meta, kCaseOneMetaLen64);
        b.write_bits(xorv >> trailing_zeros, significant_bits);
        b.stored_leading_zeros = kValueBits64 + 1;
      } else if (leading_zeros == b.stored_leading_zeros) {
        b.write_bit();
        b.skip_bit();
        const int significant_bits = kValueBits64 - leading_zeros;
        b.write_bits(xorv, significant_bits);
      } else {
        b.stored_leading_zeros = leading_zeros;
        const int significant_bits = kValueBits64 - leading_zeros;
        b.write_bits(24ull + static_cast<std::uint64_t>(kLeadingRepresentation[leading_zeros]), 5);
        b.write_bits(xorv, significant_bits);
      }
    }
    current = (current + 1) % kChimpPreviousValues64;
    stored_values[current] = value;
    index++;
    indices[key] = index;
  }

  void encode(std::int64_t value) {
    if (first_written) {
      compress_value(value);
    } else {
      write_first(value);
      first_written = true;
    }
  }
  void flush_ending(std::int64_t end_marker) {
    encode(end_marker);
    b.finish_stream();
  }
};

struct DoubleChimpEncoder : LongChimpEncoder {
  void encode_d(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(double));
    encode(static_cast<std::int64_t>(bits));
  }
  void flush_ending_double() {
    flush_ending(static_cast<std::int64_t>(kGorillaEndDoubleBits));
  }
};

}  // namespace

std::vector<std::uint8_t> encode_gorilla_int64(std::span<const std::int64_t> values) {
  LongGorillaEncoder enc;
  for (std::int64_t v : values) {
    enc.encode(v);
  }
  enc.flush_ending(kGorillaEndLong);
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_gorilla_double(std::span<const double> values) {
  DoubleGorillaEncoder enc;
  for (double d : values) {
    enc.encode_d(d);
  }
  enc.flush_ending_double();
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_gorilla_double_ablation(
    std::span<const double> values, std::string_view omitted_operator) {
  if (omitted_operator.empty() || omitted_operator == "none") {
    return encode_gorilla_double(values);
  }
  if (omitted_operator != "xor" && omitted_operator != "trailing_zero_pack") {
    throw std::invalid_argument("unsupported Gorilla ablation operator");
  }

  GorillaBitStream stream;
  auto bits_of = [](double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  if (omitted_operator == "xor") {
    for (double value : values) stream.write_bits(bits_of(value), kValueBits64);
    stream.write_bits(kGorillaEndDoubleBits, kValueBits64);
    stream.finish_stream();
    return std::move(stream.out);
  }

  std::uint64_t previous = values.empty() ? kGorillaEndDoubleBits : bits_of(values.front());
  stream.write_bits(previous, kValueBits64);
  const std::size_t start = values.empty() ? 0 : 1;
  for (std::size_t i = start; i <= values.size(); ++i) {
    const std::uint64_t current =
        i == values.size() ? kGorillaEndDoubleBits : bits_of(values[i]);
    const std::uint64_t xor_value = previous ^ current;
    if (xor_value == 0) {
      stream.skip_bit();
    } else {
      stream.write_bit();
      stream.write_bits(xor_value, kValueBits64);
    }
    previous = current;
  }
  stream.finish_stream();
  return std::move(stream.out);
}

std::vector<std::uint8_t> encode_chimp_int64(std::span<const std::int64_t> values) {
  LongChimpEncoder enc;
  for (std::int64_t v : values) {
    enc.encode(v);
  }
  enc.flush_ending(kGorillaEndLong);
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_chimp_double(std::span<const double> values) {
  DoubleChimpEncoder enc;
  for (double d : values) {
    enc.encode_d(d);
  }
  enc.flush_ending_double();
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_chimp_double_ablation(
    std::span<const double> values, std::string_view omitted_operator) {
  if (omitted_operator.empty() || omitted_operator == "none") {
    return encode_chimp_double(values);
  }
  return encode_gorilla_double_ablation(values, omitted_operator);
}

namespace {

struct GorillaReader {
  const std::uint8_t* p = nullptr;
  std::size_t len = 0;
  std::size_t pos = 0;
  std::uint8_t buffer = 0;
  int bits_left = 0;
  bool ok = true;

  void flip_byte() {
    if (bits_left == 0) {
      if (pos >= len) {
        ok = false;
        return;
      }
      buffer = p[pos++];
      bits_left = 8;
    }
  }

  bool read_bit() {
    flip_byte();
    if (!ok) return false;
    bool bit = ((buffer >> (bits_left - 1)) & 1) == 1;
    bits_left--;
    return bit;
  }

  std::uint64_t read_long(int bits) {
    std::uint64_t value = 0;
    while (bits > 0) {
      flip_byte();
      if (!ok) return 0;
      if (bits > bits_left || bits == 8) {
        std::uint8_t d = static_cast<std::uint8_t>(buffer & ((1 << bits_left) - 1));
        value = (value << bits_left) + d;
        bits -= bits_left;
        bits_left = 0;
      } else {
        std::uint8_t d =
            static_cast<std::uint8_t>((buffer >> (bits_left - bits)) & ((1 << bits) - 1));
        value = (value << bits) + d;
        bits_left -= bits;
        bits = 0;
      }
    }
    return value;
  }

  std::uint8_t read_next_clear_bit(int max_bits) {
    std::uint8_t value = 0;
    for (int i = 0; i < max_bits; i++) {
      value = static_cast<std::uint8_t>(value << 1);
      if (read_bit()) {
        value = static_cast<std::uint8_t>(value | 1);
      } else {
        break;
      }
    }
    return value;
  }

  std::uint8_t read_next_n_bits(int n) {
    std::uint8_t value = 0;
    for (int i = 0; i < n; i++) {
      value = static_cast<std::uint8_t>(value << 1);
      if (read_bit()) value = static_cast<std::uint8_t>(value | 1);
    }
    return value;
  }
};

struct GorillaLongDecoder {
  GorillaReader r{};
  bool first_read = false;
  int stored_leading_zeros = std::numeric_limits<int>::max();
  int stored_trailing_zeros = 0;
  std::int64_t stored_value = 0;
  bool has_next = true;
  std::uint64_t end_bits_u64 = static_cast<std::uint64_t>(kGorillaEndLong);

  void reset(std::span<const std::uint8_t> enc, std::uint64_t end_u64) {
    r = GorillaReader{enc.data(), enc.size(), 0, 0, 0, true};
    first_read = false;
    stored_leading_zeros = std::numeric_limits<int>::max();
    stored_trailing_zeros = 0;
    stored_value = 0;
    has_next = true;
    end_bits_u64 = end_u64;
  }

  void read_next_after_control(std::uint8_t control_bits) {
    switch (control_bits) {
      case 3:
        stored_leading_zeros = static_cast<int>(r.read_long(kLeadingZeroBits64));
        if (!r.ok) return;
        {
          std::uint64_t sig64 = r.read_long(kMeaningfulXorBits64);
          if (!r.ok) return;
          auto significant_bits = static_cast<int>(static_cast<std::uint8_t>(sig64) + 1);
          stored_trailing_zeros = kValueBits64 - significant_bits - stored_leading_zeros;
        }
        [[fallthrough]];
      case 2: {
        int sig = kValueBits64 - stored_leading_zeros - stored_trailing_zeros;
        std::uint64_t xorv = r.read_long(sig);
        if (!r.ok) return;
        xorv <<= stored_trailing_zeros;
        stored_value ^= static_cast<std::int64_t>(xorv);
      } break;
      default:
        break;
    }
    if (static_cast<std::uint64_t>(stored_value) == end_bits_u64) has_next = false;
  }

  std::int64_t read_value() {
    std::int64_t ret = stored_value;
    if (!first_read) {
      r.flip_byte();
      stored_value = static_cast<std::int64_t>(r.read_long(kValueBits64));
      if (!r.ok) return 0;
      ret = stored_value;
      first_read = true;
    } else {
      ret = stored_value;
    }
    std::uint8_t ctrl = r.read_next_clear_bit(2);
    if (!r.ok) return ret;
    read_next_after_control(ctrl);
    return ret;
  }
};

struct LongChimpDecoder {
  GorillaReader r{};
  bool first_read = false;
  int stored_leading_zeros = std::numeric_limits<int>::max();
  int stored_trailing_zeros = 0;
  std::int64_t stored_value = 0;
  std::int64_t stored_values[kChimpPreviousValues64]{};
  int current = 0;
  bool has_next = true;
  std::uint64_t end_bits_u64 = static_cast<std::uint64_t>(kGorillaEndLong);

  void reset(std::span<const std::uint8_t> enc, std::uint64_t end_u64) {
    r = GorillaReader{enc.data(), enc.size(), 0, 0, 0, true};
    first_read = false;
    stored_leading_zeros = std::numeric_limits<int>::max();
    stored_trailing_zeros = 0;
    stored_value = 0;
    current = 0;
    has_next = true;
    end_bits_u64 = end_u64;
    for (std::int64_t& v : stored_values) v = 0;
  }

  __attribute__((noinline)) void read_next() {
    std::uint8_t control = r.read_next_n_bits(2);
    if (!r.ok) return;
    switch (control) {
      case 3:
        stored_leading_zeros = kChimpDecodeLeading[static_cast<int>(r.read_long(3))];
        if (!r.ok) return;
        {
          std::uint64_t xorv = r.read_long(kValueBits64 - stored_leading_zeros);
          if (!r.ok) return;
          stored_value ^= static_cast<std::int64_t>(xorv);
        }
        current = (current + 1) % kChimpPreviousValues64;
        stored_values[current] = stored_value;
        break;
      case 2: {
        std::uint64_t xorv = r.read_long(kValueBits64 - stored_leading_zeros);
        if (!r.ok) return;
        stored_value ^= static_cast<std::int64_t>(xorv);
        current = (current + 1) % kChimpPreviousValues64;
        stored_values[current] = stored_value;
      } break;
      case 1: {
        int fill = kCaseOneMetaLen64Decoder;
        std::uint64_t temp = r.read_long(fill);
        if (!r.ok) return;
        int index = static_cast<int>((temp >> (fill -= kChimpPrevLog2_64)) &
                                     ((1 << kChimpPrevLog2_64) - 1));
        stored_leading_zeros = kChimpDecodeLeading[(temp >> (fill -= 3)) & 7];
        int significant_bits =
            static_cast<int>((temp >> (fill -= 6)) & ((1 << 6) - 1));
        if (significant_bits == 0) significant_bits = kValueBits64;
        stored_value = stored_values[index];
        stored_trailing_zeros = kValueBits64 - significant_bits - stored_leading_zeros;
        std::uint64_t xorv =
            r.read_long(kValueBits64 - stored_leading_zeros - stored_trailing_zeros);
        if (!r.ok) return;
        xorv <<= stored_trailing_zeros;
        stored_value ^= static_cast<std::int64_t>(xorv);
        current = (current + 1) % kChimpPreviousValues64;
        stored_values[current] = stored_value;
      } break;
      default: {
        int previous_index = static_cast<int>(r.read_long(kChimpPrevLog2_64));
        if (!r.ok) return;
        stored_value = stored_values[previous_index];
        current = (current + 1) % kChimpPreviousValues64;
        stored_values[current] = stored_value;
      } break;
    }
    if (static_cast<std::uint64_t>(stored_values[current]) == end_bits_u64) has_next = false;
  }

  std::int64_t read_value() {
    std::int64_t ret = stored_value;
    if (!first_read) {
      r.flip_byte();
      stored_value = static_cast<std::int64_t>(r.read_long(kValueBits64));
      if (!r.ok) return 0;
      stored_values[current] = stored_value;
      first_read = true;
      ret = stored_value;
    } else {
      ret = stored_value;
    }
    read_next();
    return ret;
  }
};

bool decode_gorilla_long_stream(std::span<const std::uint8_t> enc, std::vector<std::int64_t>& out) {
  out.clear();
  GorillaLongDecoder dec;
  dec.reset(enc, static_cast<std::uint64_t>(kGorillaEndLong));
  while (dec.has_next) {
    out.push_back(dec.read_value());
    if (!dec.r.ok) return false;
  }
  return dec.r.ok;
}

bool decode_gorilla_double_stream(std::span<const std::uint8_t> enc, std::vector<double>& out) {
  out.clear();
  GorillaLongDecoder dec;
  dec.reset(enc, kGorillaEndDoubleBits);
  while (dec.has_next) {
    std::int64_t bits = dec.read_value();
    if (!dec.r.ok) return false;
    double d;
    std::memcpy(&d, &bits, sizeof(double));
    out.push_back(d);
  }
  return dec.r.ok;
}

bool decode_chimp_long_stream(std::span<const std::uint8_t> enc, std::vector<std::int64_t>& out) {
  out.clear();
  LongChimpDecoder dec;
  dec.reset(enc, static_cast<std::uint64_t>(kGorillaEndLong));
  while (dec.has_next) {
    out.push_back(dec.read_value());
    if (!dec.r.ok) return false;
  }
  return dec.r.ok;
}

bool decode_chimp_double_stream(std::span<const std::uint8_t> enc, std::vector<double>& out) {
  out.clear();
  LongChimpDecoder dec;
  dec.reset(enc, kGorillaEndDoubleBits);
  while (dec.has_next) {
    std::int64_t bits = dec.read_value();
    if (!dec.r.ok) return false;
    double d;
    std::memcpy(&d, &bits, sizeof(double));
    out.push_back(d);
  }
  return dec.r.ok;
}

}  // namespace

bool decode_gorilla_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  return decode_gorilla_long_stream(encoded, out);
}

bool decode_gorilla_double(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  return decode_gorilla_double_stream(encoded, out);
}

bool decode_gorilla_double_ablation(std::span<const std::uint8_t> encoded,
                                    std::string_view omitted_operator,
                                    std::vector<double>& out) {
  if (omitted_operator.empty() || omitted_operator == "none") {
    return decode_gorilla_double(encoded, out);
  }
  if (omitted_operator != "xor" && omitted_operator != "trailing_zero_pack") {
    return false;
  }

  out.clear();
  GorillaReader reader{encoded.data(), encoded.size(), 0, 0, 0, true};
  std::uint64_t stored = reader.read_long(kValueBits64);
  if (!reader.ok) return false;
  auto append = [&](std::uint64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    out.push_back(value);
  };
  if (stored == kGorillaEndDoubleBits) return true;
  append(stored);
  while (reader.ok) {
    if (omitted_operator == "xor") {
      stored = reader.read_long(kValueBits64);
    } else if (reader.read_bit()) {
      stored ^= reader.read_long(kValueBits64);
    }
    if (!reader.ok) return false;
    if (stored == kGorillaEndDoubleBits) return true;
    append(stored);
  }
  return false;
}

bool decode_chimp_int64(std::span<const std::uint8_t> encoded, std::vector<std::int64_t>& out) {
  return decode_chimp_long_stream(encoded, out);
}

bool decode_chimp_double(std::span<const std::uint8_t> encoded, std::vector<double>& out) {
  return decode_chimp_double_stream(encoded, out);
}

bool decode_chimp_double_ablation(std::span<const std::uint8_t> encoded,
                                  std::string_view omitted_operator,
                                  std::vector<double>& out) {
  if (omitted_operator.empty() || omitted_operator == "none") {
    return decode_chimp_double(encoded, out);
  }
  return decode_gorilla_double_ablation(encoded, omitted_operator, out);
}


}  // namespace tsfile::p03
