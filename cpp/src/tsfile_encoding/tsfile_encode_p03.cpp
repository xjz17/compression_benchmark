#include "tsfile_encode_p03.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace tsfile::p03 {

namespace {

constexpr int kValueBits32 = 32;
constexpr int kLeadingZeroBits32 = 5;
constexpr int kMeaningfulXorBits32 = 5;
constexpr std::int32_t kGorillaEndInt = std::numeric_limits<std::int32_t>::min();  // TSFileConfig.GORILLA_ENCODING_ENDING_INTEGER
constexpr std::uint32_t kGorillaEndFloatBits = 0x7fc00000u;  // Float.floatToRawIntBits(Float.NaN) on HotSpot

constexpr int kChimpPreviousValues = 64;
constexpr int kChimpPrevLog2 = 6;  // (int)(log(64)/log(2))
constexpr int kChimpThreshold = 5 + kChimpPrevLog2;  // 11
constexpr int kChimpSetLsb = (1 << (kChimpThreshold + 1)) - 1;  // 4095
constexpr int kCaseZeroMetaLen = kChimpPrevLog2 + 2;             // 8
constexpr int kCaseOneMetaLen = kChimpPrevLog2 + 10;             // 16
constexpr std::uint32_t kChimpEndFloatBits = 0x7fc00000u;  // Float.floatToRawIntBits(NaN)

// org.apache.tsfile.encoding.encoder.IntChimpEncoder
static const std::int16_t kLeadingRepresentation[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

static const std::int16_t kLeadingRound[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};

/** Shared bit stream writer — org.apache.tsfile.encoding.encoder.GorillaEncoderV2 */
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

/** org.apache.tsfile.encoding.encoder.IntGorillaEncoder */
struct IntGorillaEncoder {
  GorillaBitStream b;
  bool first_written = false;
  std::int32_t stored_value = 0;

  void write_first(std::int32_t value) {
    stored_value = value;
    b.write_bits(static_cast<std::uint32_t>(value), kValueBits32);
  }
  void write_existing_leading(std::uint32_t xorv) {
    b.skip_bit();
    int significant_bits = kValueBits32 - b.stored_leading_zeros - b.stored_trailing_zeros;
    b.write_bits(xorv >> b.stored_trailing_zeros, significant_bits);
  }
  void write_new_leading(std::uint32_t xorv, int leading_zeros, int trailing_zeros) {
    b.write_bit();
    int significant_bits = kValueBits32 - leading_zeros - trailing_zeros;
    b.write_bits(static_cast<std::uint64_t>(leading_zeros), kLeadingZeroBits32);
    b.write_bits(static_cast<std::uint64_t>(significant_bits - 1), kMeaningfulXorBits32);
    b.write_bits(xorv >> trailing_zeros, significant_bits);
    b.stored_leading_zeros = leading_zeros;
    b.stored_trailing_zeros = trailing_zeros;
  }
  void compress_value(std::int32_t value) {
    std::int32_t xors = stored_value ^ value;
    std::uint32_t xoru = static_cast<std::uint32_t>(xors);
    stored_value = value;
    if (xoru == 0) {
      b.skip_bit();
    } else {
      b.write_bit();
      int lz = __builtin_clz(xoru);
      int tz = __builtin_ctz(xoru);
      if (lz >= b.stored_leading_zeros && tz >= b.stored_trailing_zeros) {
        write_existing_leading(xoru);
      } else {
        write_new_leading(xoru, lz, tz);
      }
    }
  }
  void encode(std::int32_t value) {
    if (first_written) {
      compress_value(value);
    } else {
      write_first(value);
      first_written = true;
    }
  }
  void flush_ending(std::int32_t end_int) {
    encode(end_int);
    b.finish_stream();
  }
};

/** org.apache.tsfile.encoding.encoder.SinglePrecisionEncoderV2 */
struct FloatGorillaEncoder : IntGorillaEncoder {
  void encode_f(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::memcpy(&bits, &value, sizeof(float));
    encode(static_cast<std::int32_t>(bits));
  }
  void flush_ending_float() { flush_ending(static_cast<std::int32_t>(kGorillaEndFloatBits)); }
};

/** org.apache.tsfile.encoding.encoder.IntChimpEncoder */
struct IntChimpEncoder {
  GorillaBitStream b;
  bool first_written = false;
  std::int32_t stored_values[kChimpPreviousValues]{};
  int indices[1 << (kChimpThreshold + 1)]{};
  int index = 0;
  int current = 0;

  void write_first(std::int32_t value) {
    stored_values[current] = value;
    b.write_bits(static_cast<std::uint32_t>(value), kValueBits32);
    indices[value & kChimpSetLsb] = index;
  }

  __attribute__((noinline)) void compress_value(std::int32_t value) {
    int key = value & kChimpSetLsb;
    std::uint32_t xorv;
    int previous_index;
    int trailing_zeros = 0;
    int curr_index = indices[key];
    if ((index - curr_index) < kChimpPreviousValues) {
      int temp_xor = value ^ stored_values[curr_index % kChimpPreviousValues];
      trailing_zeros = (temp_xor == 0) ? kValueBits32 : __builtin_ctz(static_cast<std::uint32_t>(temp_xor));
      if (trailing_zeros > kChimpThreshold) {
        previous_index = curr_index % kChimpPreviousValues;
        xorv = static_cast<std::uint32_t>(temp_xor);
      } else {
        previous_index = index % kChimpPreviousValues;
        xorv = static_cast<std::uint32_t>(stored_values[previous_index] ^ value);
      }
    } else {
      previous_index = index % kChimpPreviousValues;
      xorv = static_cast<std::uint32_t>(stored_values[previous_index] ^ value);
    }

    if (xorv == 0) {
      b.write_bits(static_cast<std::uint64_t>(previous_index), kCaseZeroMetaLen);
      b.stored_leading_zeros = kValueBits32 + 1;
    } else {
      int nlz = __builtin_clz(xorv);
      int leading_zeros = kLeadingRound[nlz];
      if (trailing_zeros > kChimpThreshold) {
        int significant_bits = kValueBits32 - leading_zeros - trailing_zeros;
        std::uint64_t meta = 256ull * (kChimpPreviousValues + previous_index) +
                             32ull * static_cast<std::uint64_t>(kLeadingRepresentation[leading_zeros]) +
                             static_cast<std::uint64_t>(significant_bits);
        b.write_bits(meta, kCaseOneMetaLen);
        b.write_bits(xorv >> trailing_zeros, significant_bits);
        b.stored_leading_zeros = kValueBits32 + 1;
      } else if (leading_zeros == b.stored_leading_zeros) {
        b.write_bit();
        b.skip_bit();
        int significant_bits = kValueBits32 - leading_zeros;
        b.write_bits(xorv, significant_bits);
      } else {
        b.stored_leading_zeros = leading_zeros;
        int significant_bits = kValueBits32 - leading_zeros;
        b.write_bits(24ull + static_cast<std::uint64_t>(kLeadingRepresentation[leading_zeros]), 5);
        b.write_bits(xorv, significant_bits);
      }
    }
    current = (current + 1) % kChimpPreviousValues;
    stored_values[current] = value;
    index++;
    indices[key] = index;
  }

  void encode(std::int32_t value) {
    if (first_written) {
      compress_value(value);
    } else {
      write_first(value);
      first_written = true;
    }
  }

  void flush_ending(std::int32_t end_marker) {
    encode(end_marker);
    b.finish_stream();
  }
};

/** org.apache.tsfile.encoding.encoder.SinglePrecisionChimpEncoder */
struct FloatChimpEncoder : IntChimpEncoder {
  void encode_f(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    encode(static_cast<std::int32_t>(bits));
  }
  void flush_ending_float() { flush_ending(static_cast<std::int32_t>(kChimpEndFloatBits)); }
};

}  // namespace

std::vector<std::uint8_t> encode_gorilla_int32(std::span<const std::int32_t> values) {
  IntGorillaEncoder enc;
  for (std::int32_t v : values) {
    enc.encode(v);
  }
  enc.flush_ending(kGorillaEndInt);
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_gorilla_float(std::span<const float> values) {
  FloatGorillaEncoder enc;
  for (float f : values) {
    enc.encode_f(f);
  }
  enc.flush_ending_float();
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_chimp_int32(std::span<const std::int32_t> values) {
  IntChimpEncoder enc;
  for (std::int32_t v : values) {
    enc.encode(v);
  }
  enc.flush_ending(kGorillaEndInt);
  return std::move(enc.b.out);
}

std::vector<std::uint8_t> encode_chimp_float(std::span<const float> values) {
  FloatChimpEncoder enc;
  for (float f : values) {
    enc.encode_f(f);
  }
  enc.flush_ending_float();
  return std::move(enc.b.out);
}

}  // namespace tsfile::p03
