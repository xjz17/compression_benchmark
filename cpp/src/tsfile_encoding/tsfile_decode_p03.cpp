#include "tsfile_decode_p03.hpp"

#include <cstring>
#include <span>
#include <limits>
#include <vector>

namespace tsfile::p03 {

namespace {

constexpr int kValueBits32 = 32;
constexpr int kLeadingZeroBits32 = 5;
constexpr int kMeaningfulXorBits32 = 5;
constexpr std::int32_t kGorillaEndInt = std::numeric_limits<std::int32_t>::min();
constexpr std::uint32_t kGorillaEndFloatBits = 0x7fc00000u;
constexpr int kChimpPreviousValues = 64;
constexpr int kChimpPrevLog2 = 6;
constexpr int kCaseOneMetaLen = kChimpPrevLog2 + 10;
// IntChimpDecoder.java case-1 metadata bit width (PREVIOUS_VALUES_LOG2 + 8); required for float
// Chimp streams with NaN terminator to match tsfile Java I/O. Int streams use kCaseOneMetaLen (+10).
constexpr int kCaseOneMetaLenDecoderJava = kChimpPrevLog2 + 8;
constexpr std::uint32_t kChimpEndFloatBits = 0x7fc00000u;

// IntChimpDecoder.java LEADING_REPRESENTATION: 3-bit class -> leading zero count for cases 1 and 3.
static const int kChimpDecodeLeading[] = {0, 8, 12, 16, 18, 20, 22, 24};

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

struct GorillaIntDecoder {
  GorillaReader r{};
  bool first_read = false;
  int stored_leading_zeros = std::numeric_limits<int>::max();
  int stored_trailing_zeros = 0;
  std::int32_t stored_value = 0;
  bool has_next = true;
  std::uint32_t end_bits_u32 = static_cast<std::uint32_t>(kGorillaEndInt);

  void reset(std::span<const std::uint8_t> enc, std::uint32_t end_u32) {
    r = GorillaReader{enc.data(), enc.size(), 0, 0, 0, true};
    first_read = false;
    stored_leading_zeros = std::numeric_limits<int>::max();
    stored_trailing_zeros = 0;
    stored_value = 0;
    has_next = true;
    end_bits_u32 = end_u32;
  }

  void read_next_after_control(std::uint8_t control_bits) {
    switch (control_bits) {
      case 3:
        stored_leading_zeros = static_cast<int>(r.read_long(kLeadingZeroBits32));
        if (!r.ok) return;
        {
          std::uint64_t sig64 = r.read_long(kMeaningfulXorBits32);
          if (!r.ok) return;
          auto significant_bits = static_cast<int>(static_cast<std::uint8_t>(sig64) + 1);
          stored_trailing_zeros = kValueBits32 - significant_bits - stored_leading_zeros;
        }
        [[fallthrough]];
      case 2: {
        int sig = kValueBits32 - stored_leading_zeros - stored_trailing_zeros;
        int xorv = static_cast<int>(r.read_long(sig));
        if (!r.ok) return;
        xorv <<= stored_trailing_zeros;
        stored_value ^= xorv;
      } break;
      default:
        break;
    }
    if (static_cast<std::uint32_t>(stored_value) == end_bits_u32) has_next = false;
  }

  std::int32_t read_int() {
    std::int32_t ret = stored_value;
    if (!first_read) {
      r.flip_byte();
      stored_value = static_cast<std::int32_t>(r.read_long(kValueBits32));
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

bool decode_gorilla_int_stream(std::span<const std::uint8_t> enc, std::vector<std::int32_t>& out) {
  out.clear();
  GorillaIntDecoder dec;
  dec.reset(enc, static_cast<std::uint32_t>(kGorillaEndInt));
  while (dec.has_next) {
    out.push_back(dec.read_int());
    if (!dec.r.ok) return false;
  }
  return dec.r.ok;
}

bool decode_gorilla_float_stream(std::span<const std::uint8_t> enc, std::vector<float>& out) {
  out.clear();
  GorillaIntDecoder dec;
  dec.reset(enc, kGorillaEndFloatBits);
  while (dec.has_next) {
    std::int32_t bits = dec.read_int();
    if (!dec.r.ok) return false;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    out.push_back(f);
  }
  return dec.r.ok;
}

struct IntChimpDecoder {
  GorillaReader r{};
  bool first_read = false;
  int stored_leading_zeros = std::numeric_limits<int>::max();
  int stored_trailing_zeros = 0;
  std::int32_t stored_value = 0;
  std::int32_t stored_values[kChimpPreviousValues]{};
  int current = 0;
  bool has_next = true;
  std::uint32_t end_bits_u32 = static_cast<std::uint32_t>(kGorillaEndInt);
  int case_one_meta_bits = kCaseOneMetaLen;

  void reset(std::span<const std::uint8_t> enc, std::uint32_t end_u32) {
    r = GorillaReader{enc.data(), enc.size(), 0, 0, 0, true};
    first_read = false;
    stored_leading_zeros = std::numeric_limits<int>::max();
    stored_trailing_zeros = 0;
    stored_value = 0;
    current = 0;
    has_next = true;
    end_bits_u32 = end_u32;
    case_one_meta_bits =
        (end_u32 == kChimpEndFloatBits) ? kCaseOneMetaLenDecoderJava : kCaseOneMetaLen;
    for (std::int32_t& v : stored_values) v = 0;
  }

  __attribute__((noinline)) void read_next() {
    std::uint8_t control = r.read_next_n_bits(2);
    if (!r.ok) return;
    switch (control) {
      case 3:
        stored_leading_zeros = kChimpDecodeLeading[static_cast<int>(r.read_long(3))];
        if (!r.ok) return;
        {
          int xorv = static_cast<int>(r.read_long(kValueBits32 - stored_leading_zeros));
          if (!r.ok) return;
          stored_value ^= xorv;
        }
        current = (current + 1) % kChimpPreviousValues;
        stored_values[current] = stored_value;
        break;
      case 2: {
        int xorv = static_cast<int>(r.read_long(kValueBits32 - stored_leading_zeros));
        if (!r.ok) return;
        stored_value ^= xorv;
        current = (current + 1) % kChimpPreviousValues;
        stored_values[current] = stored_value;
      } break;
      case 1: {
        int fill = case_one_meta_bits;
        int temp = static_cast<int>(r.read_long(fill));
        if (!r.ok) return;
        int index = (temp >> (fill -= kChimpPrevLog2)) & ((1 << kChimpPrevLog2) - 1);
        stored_leading_zeros = kChimpDecodeLeading[(temp >> (fill -= 3)) & 7];
        int significant_bits = (temp >> (fill -= 5)) & 31;
        stored_value = stored_values[index];
        if (significant_bits == 0) significant_bits = kValueBits32;
        stored_trailing_zeros = kValueBits32 - significant_bits - stored_leading_zeros;
        int xorv =
            static_cast<int>(r.read_long(kValueBits32 - stored_leading_zeros - stored_trailing_zeros));
        if (!r.ok) return;
        xorv <<= stored_trailing_zeros;
        stored_value ^= xorv;
        current = (current + 1) % kChimpPreviousValues;
        stored_values[current] = stored_value;
      } break;
      default: {
        // Match encoder kCaseZeroMetaLen (= prevLog2 + 2): 2-bit control + prevLog2 index bits.
        int previous_index = static_cast<int>(r.read_long(kChimpPrevLog2));
        if (!r.ok) return;
        stored_value = stored_values[previous_index % kChimpPreviousValues];
        current = (current + 1) % kChimpPreviousValues;
        stored_values[current] = stored_value;
      } break;
    }
    if (static_cast<std::uint32_t>(stored_values[current]) == end_bits_u32) has_next = false;
  }

  std::int32_t read_int() {
    std::int32_t ret = stored_value;
    if (!first_read) {
      r.flip_byte();
      stored_value = static_cast<std::int32_t>(r.read_long(kValueBits32));
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

bool decode_chimp_int_stream(std::span<const std::uint8_t> enc, std::vector<std::int32_t>& out) {
  out.clear();
  IntChimpDecoder dec;
  dec.reset(enc, static_cast<std::uint32_t>(kGorillaEndInt));
  while (dec.has_next) {
    out.push_back(dec.read_int());
    if (!dec.r.ok) return false;
  }
  return dec.r.ok;
}

bool decode_chimp_float_stream(std::span<const std::uint8_t> enc, std::vector<float>& out) {
  out.clear();
  IntChimpDecoder dec;
  dec.reset(enc, kChimpEndFloatBits);
  while (dec.has_next) {
    std::int32_t bits = dec.read_int();
    if (!dec.r.ok) return false;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    out.push_back(f);
  }
  return dec.r.ok;
}

}  // namespace

bool decode_gorilla_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out) {
  return decode_gorilla_int_stream(encoded, out);
}

bool decode_gorilla_float(std::span<const std::uint8_t> encoded, std::vector<float>& out) {
  return decode_gorilla_float_stream(encoded, out);
}

bool decode_chimp_int32(std::span<const std::uint8_t> encoded, std::vector<std::int32_t>& out) {
  return decode_chimp_int_stream(encoded, out);
}

bool decode_chimp_float(std::span<const std::uint8_t> encoded, std::vector<float>& out) {
  return decode_chimp_float_stream(encoded, out);
}

}  // namespace tsfile::p03
