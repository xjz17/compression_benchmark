#include "camel_exact_stages.hpp"
#include "adjacent_residual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace exact_actions::camel {
namespace {

constexpr std::array<std::uint8_t, 4> kPacketMagic{'C', 'E', 'R', '1'};
constexpr std::array<std::uint8_t, 4> kXorPacketMagic{'C', 'X', 'R', '1'};
constexpr std::array<std::uint8_t, 4> kWireMagic{'C', 'A', 'M', '1'};
constexpr std::size_t kChunk = 1000;
constexpr int kPowers[] = {1, 10, 100, 1000, 10000, 100000};
constexpr int kThreshold[] = {5, 25, 125, 625};
constexpr int kMValueBits[] = {3, 5, 7, 10, 15};

enum class RecordTag : std::uint8_t { first = 0, decimal = 1, literal = 2 };

struct Record {
  RecordTag tag{};
  std::uint64_t raw_bits{};
  std::int64_t integer{};
  std::uint8_t signal{};
  std::uint8_t decimal_count{};
  std::uint32_t decimal_value{};
};

struct XorRecord {
  RecordTag tag{};
  std::uint64_t bits{};
  std::int64_t integer_delta{};
  std::uint8_t signal{};
  std::uint8_t decimal_count{};
  std::uint32_t decimal_value{};
};

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double bits_double(std::uint64_t bits) {
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t int64_bits(std::int64_t value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::int64_t signed_bits(std::uint64_t value) {
  std::int64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

constexpr double kInt64MinAsDouble = -0x1p63;
constexpr double kInt64MaxExclusiveAsDouble = 0x1p63;

bool truncating_double_to_int64(double value, std::int64_t& converted) {
  if (!std::isfinite(value) || value < kInt64MinAsDouble ||
      value >= kInt64MaxExclusiveAsDouble)
    return false;
  converted = static_cast<std::int64_t>(value);
  return true;
}

bool exact_double_to_int64(double value, std::int64_t& converted) {
  return truncating_double_to_int64(value, converted) &&
         static_cast<double>(converted) == value;
}

// Java's narrowing double-to-long conversion is defined for every bit pattern:
// it truncates in range, saturates at the two endpoints, and maps NaN to zero.
// Camel uses that conversion only as the integer history beside literal values.
std::int64_t java_double_to_int64(double value) {
  if (std::isnan(value)) return 0;
  if (value >= kInt64MaxExclusiveAsDouble)
    return std::numeric_limits<std::int64_t>::max();
  if (value <= kInt64MinAsDouble)
    return std::numeric_limits<std::int64_t>::min();
  return static_cast<std::int64_t>(value);
}

std::int64_t java_round(double value) {
  if (!std::isfinite(value)) throw std::runtime_error("Camel decimal path requires finite input");
  const double rounded =
      value >= 0 ? std::floor(value + 0.5) : std::ceil(value - 0.5);
  return java_double_to_int64(rounded);
}

std::pair<int, int> decimal_count(double value) {
  value = std::fabs(value);
  if (!std::isfinite(value)) throw std::runtime_error("Camel non-finite decimal value");
  double factor = 1.0;
  int count = 0;
  while (std::fabs(value * factor - static_cast<double>(java_round(value * factor))) > 1e-7) {
    factor *= 10.0;
    if (++count > 64 || !std::isfinite(factor) || !std::isfinite(value * factor))
      throw std::runtime_error("Camel decimal scale did not converge");
  }
  if (count == 0) count = 1;
  const int dc = std::min(count, 3);
  const auto scaled = java_round(value * static_cast<double>(kPowers[dc]));
  return {dc, static_cast<int>(scaled % kPowers[dc])};
}

double reconstruct_decimal(std::int64_t integer, std::uint8_t signal,
                           int dc, int decimal) {
  const double scale = static_cast<double>(kPowers[dc]);
  const double sign = signal == 0 ? -1.0 : 1.0;
  const auto magnitude = integer >= 0
                             ? static_cast<std::uint64_t>(integer)
                             : static_cast<std::uint64_t>(
                                   -static_cast<__int128>(integer));
  double decimal_part = static_cast<double>(decimal) / scale;
  decimal_part = static_cast<double>(java_round(decimal_part * scale)) / scale;
  return sign * ((static_cast<double>(magnitude) * scale +
                  decimal_part * scale) /
                 scale);
}

Record make_record(double value, bool first) {
  const auto raw = double_bits(value);
  if (first) return {RecordTag::first, raw, 0, 0, 0, 0};
  // Camel's decimal path is only an optimization.  NaN and infinities are
  // still valid float64 payloads and must use the literal path so that their
  // exact IEEE-754 bit pattern survives the round trip.
  if (!std::isfinite(value))
    return {RecordTag::literal, raw, 0, 0, 0, 0};
  std::int64_t integer = 0;
  if (!truncating_double_to_int64(value, integer)) {
    return {RecordTag::literal, raw, 0, 0, 0, 0};
  }
  const auto [dc, decimal] = decimal_count(value);
  const auto signal = static_cast<std::uint8_t>(value < 0 ? 0 : 1);
  // Match CamelCompressor._decimal_path_lossless exactly.  That preflight uses
  // the signed truncated integer and then applies the sign again, so negative
  // values intentionally fall back to the literal-XOR path.
  const double scale = static_cast<double>(kPowers[dc]);
  double decimal_part = static_cast<double>(decimal) / scale;
  decimal_part = static_cast<double>(java_round(decimal_part * scale)) / scale;
  const double sign = signal == 0 ? -1.0 : 1.0;
  const auto reconstructed =
      sign * ((static_cast<double>(integer) * scale + decimal_part * scale) /
              scale);
  if (double_bits(reconstructed) != raw)
    return {RecordTag::literal, raw, 0, 0, 0, 0};
  return {RecordTag::decimal, 0, integer, signal,
          static_cast<std::uint8_t>(dc), static_cast<std::uint32_t>(decimal)};
}

double record_value(const Record& record) {
  if (record.tag == RecordTag::first || record.tag == RecordTag::literal)
    return bits_double(record.raw_bits);
  if (record.tag != RecordTag::decimal || record.decimal_count < 1 ||
      record.decimal_count > 3 || record.signal > 1)
    throw std::runtime_error("invalid Camel eraser record");
  return reconstruct_decimal(record.integer, record.signal,
                             record.decimal_count, record.decimal_value);
}

void put_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32_le(std::span<const std::uint8_t> data, std::size_t& pos,
                         const char* label) {
  if (data.size() - pos < 4) throw std::runtime_error(std::string("truncated ") + label);
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(data[pos++]) << shift;
  return value;
}

std::uint64_t get_u64_le(std::span<const std::uint8_t> data, std::size_t& pos,
                         const char* label) {
  if (data.size() - pos < 8) throw std::runtime_error(std::string("truncated ") + label);
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

std::vector<std::uint8_t> serialize(std::span<const Record> records) {
  std::vector<std::uint8_t> out(kPacketMagic.begin(), kPacketMagic.end());
  put_u64_le(out, records.size());
  for (const auto& record : records) {
    out.push_back(static_cast<std::uint8_t>(record.tag));
    if (record.tag == RecordTag::first || record.tag == RecordTag::literal) {
      put_u64_le(out, record.raw_bits);
    } else {
      put_u64_le(out, int64_bits(record.integer));
      out.push_back(record.signal);
      out.push_back(record.decimal_count);
      put_u32_le(out, record.decimal_value);
    }
  }
  return out;
}

std::vector<Record> parse(std::span<const std::uint8_t> packet) {
  if (packet.size() < 12 ||
      !std::equal(kPacketMagic.begin(), kPacketMagic.end(), packet.begin()))
    throw std::runtime_error("wrong Camel eraser packet type");
  std::size_t pos = 4;
  const auto count = get_u64_le(packet, pos, "Camel eraser count");
  std::vector<Record> records;
  records.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    if (pos >= packet.size()) throw std::runtime_error("truncated Camel eraser tag");
    const auto tag = static_cast<RecordTag>(packet[pos++]);
    Record record{tag};
    if (tag == RecordTag::first || tag == RecordTag::literal) {
      record.raw_bits = get_u64_le(packet, pos, "Camel literal bits");
    } else if (tag == RecordTag::decimal) {
      record.integer = signed_bits(get_u64_le(packet, pos, "Camel integer part"));
      if (packet.size() - pos < 2) throw std::runtime_error("truncated Camel decimal metadata");
      record.signal = packet[pos++];
      record.decimal_count = packet[pos++];
      record.decimal_value = get_u32_le(packet, pos, "Camel decimal value");
      if (record.signal > 1 || record.decimal_count < 1 || record.decimal_count > 3 ||
          record.decimal_value >= static_cast<std::uint32_t>(kPowers[record.decimal_count]))
        throw std::runtime_error("invalid Camel decimal metadata");
    } else {
      throw std::runtime_error("invalid Camel eraser tag");
    }
    records.push_back(record);
  }
  if (pos != packet.size()) throw std::runtime_error("Camel eraser packet has trailing bytes");
  for (std::size_t i = 0; i < records.size(); ++i) {
    const bool should_be_first = i % kChunk == 0;
    if ((records[i].tag == RecordTag::first) != should_be_first)
      throw std::runtime_error("Camel eraser chunk boundary mismatch");
  }
  return records;
}

std::vector<std::uint8_t> serialize_xor(
    std::span<const XorRecord> records) {
  std::vector<std::uint8_t> out(kXorPacketMagic.begin(), kXorPacketMagic.end());
  put_u64_le(out, records.size());
  for (const auto& record : records) {
    out.push_back(static_cast<std::uint8_t>(record.tag));
    if (record.tag == RecordTag::first || record.tag == RecordTag::literal) {
      put_u64_le(out, record.bits);
    } else if (record.tag == RecordTag::decimal) {
      put_u64_le(out, int64_bits(record.integer_delta));
      out.push_back(record.signal);
      out.push_back(record.decimal_count);
      put_u32_le(out, record.decimal_value);
    } else {
      throw std::runtime_error("invalid Camel XOR record tag");
    }
  }
  return out;
}

std::vector<XorRecord> parse_xor(std::span<const std::uint8_t> packet) {
  if (packet.size() < 12 ||
      !std::equal(kXorPacketMagic.begin(), kXorPacketMagic.end(), packet.begin()))
    throw std::runtime_error("wrong Camel XOR packet type");
  std::size_t pos = 4;
  const auto count = get_u64_le(packet, pos, "Camel XOR count");
  std::vector<XorRecord> records;
  records.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    if (pos >= packet.size()) throw std::runtime_error("truncated Camel XOR tag");
    XorRecord record;
    record.tag = static_cast<RecordTag>(packet[pos++]);
    if (record.tag == RecordTag::first || record.tag == RecordTag::literal) {
      record.bits = get_u64_le(packet, pos, "Camel XOR bits");
    } else if (record.tag == RecordTag::decimal) {
      record.integer_delta = signed_bits(
          get_u64_le(packet, pos, "Camel integer delta"));
      if (packet.size() - pos < 2)
        throw std::runtime_error("truncated Camel XOR decimal metadata");
      record.signal = packet[pos++];
      record.decimal_count = packet[pos++];
      record.decimal_value = get_u32_le(packet, pos, "Camel XOR decimal value");
      if (record.signal > 1 || record.decimal_count < 1 ||
          record.decimal_count > 3 ||
          record.decimal_value >=
              static_cast<std::uint32_t>(kPowers[record.decimal_count]))
        throw std::runtime_error("invalid Camel XOR decimal metadata");
    } else {
      throw std::runtime_error("invalid Camel XOR tag");
    }
    const bool should_be_first = i % kChunk == 0;
    if ((record.tag == RecordTag::first) != should_be_first)
      throw std::runtime_error("Camel XOR chunk boundary mismatch");
    records.push_back(record);
  }
  if (pos != packet.size())
    throw std::runtime_error("Camel XOR packet has trailing bytes");
  return records;
}

class BitWriter {
 public:
  void write(std::uint64_t value, int width) {
    if (width < 0 || width > 64) throw std::runtime_error("invalid Camel write width");
    for (int shift = width - 1; shift >= 0; --shift) {
      current_ = static_cast<std::uint8_t>(
          (current_ << 1) | ((value >> shift) & 1U));
      if (++used_ == 8) {
        bytes_.push_back(current_);
        current_ = 0;
        used_ = 0;
      }
    }
  }
  std::vector<std::uint8_t> finish() {
    if (used_) bytes_.push_back(static_cast<std::uint8_t>(current_ << (8 - used_)));
    return std::move(bytes_);
  }

 private:
  std::vector<std::uint8_t> bytes_;
  std::uint8_t current_{};
  int used_{};
};

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}
  std::uint64_t read(int width) {
    if (width < 0 || width > 64 ||
        static_cast<std::size_t>(width) > data_.size() * 8 - bit_)
      throw std::runtime_error("truncated Camel bitstream");
    std::uint64_t value = 0;
    for (int i = 0; i < width; ++i) {
      value = (value << 1) |
              ((data_[bit_ / 8] >> (7 - bit_ % 8)) & 1U);
      ++bit_;
    }
    return value;
  }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t bit_{};
};

void write_integer(BitWriter& out, std::int64_t integer, std::uint8_t signal,
                   std::int64_t& stored) {
  const __int128 wide_diff =
      static_cast<__int128>(integer) - static_cast<__int128>(stored);
  if (wide_diff < -0xffff || wide_diff > 0xffff)
    throw std::runtime_error("Camel integer delta exceeds wire range");
  const auto diff = static_cast<std::int64_t>(wide_diff);
  out.write(signal, 1);
  if (diff >= -1 && diff <= 1) {
    out.write(static_cast<std::uint64_t>(diff + 1), 2);
  } else {
    out.write(3, 2);
    const bool positive = diff >= 0;
    const auto magnitude = static_cast<std::uint64_t>(
        positive ? static_cast<__int128>(diff) : -static_cast<__int128>(diff));
    out.write(positive ? 1 : 0, 1);
    if (magnitude >= 2 && magnitude < 8) {
      out.write(0, 1);
      out.write(magnitude, 3);
    } else {
      out.write(1, 1);
      out.write(magnitude, 16);
    }
  }
  stored = integer;
}

void write_integer_delta(BitWriter& out, std::int64_t diff,
                         std::uint8_t signal) {
  out.write(signal, 1);
  if (diff >= -1 && diff <= 1) {
    out.write(static_cast<std::uint64_t>(diff + 1), 2);
    return;
  }
  out.write(3, 2);
  const bool positive = diff >= 0;
  const auto magnitude = static_cast<std::uint64_t>(
      positive ? static_cast<__int128>(diff) : -static_cast<__int128>(diff));
  out.write(positive ? 1 : 0, 1);
  if (magnitude >= 2 && magnitude < 8) {
    out.write(0, 1);
    out.write(magnitude, 3);
  } else {
    if (magnitude > 0xffffU)
      throw std::runtime_error("Camel integer delta exceeds wire range");
    out.write(1, 1);
    out.write(magnitude, 16);
  }
}

void write_decimal(BitWriter& out, int decimal, int dc) {
  out.write(static_cast<std::uint64_t>(dc - 1), 2);
  const int threshold = kThreshold[dc - 1];
  int m = decimal;
  if (decimal >= threshold) {
    out.write(1, 1);
    m = decimal % threshold;
    const double power = static_cast<double>(kPowers[dc]);
    const auto x = double_bits(static_cast<double>(decimal) / power + 1.0) ^
                   double_bits(static_cast<double>(m) / power + 1.0);
    out.write(x >> (52 - dc), dc);
  } else {
    out.write(0, 1);
  }
  if (dc == 1) {
    out.write(static_cast<std::uint64_t>(m), 3);
  } else if (dc == 2) {
    out.write(m < 8 ? 0 : 1, 1);
    out.write(static_cast<std::uint64_t>(m), m < 8 ? 3 : 5);
  } else {
    if (m < 2) { out.write(0, 2); out.write(m, 1); }
    else if (m < 8) { out.write(1, 2); out.write(m, 3); }
    else if (m < 32) { out.write(2, 2); out.write(m, 5); }
    else { out.write(3, 2); out.write(m, kMValueBits[dc - 1]); }
  }
}

std::vector<std::uint8_t> encode_chunk(std::span<const Record> records) {
  if (records.empty() || records.front().tag != RecordTag::first)
    throw std::runtime_error("Camel chunk has no first record");
  BitWriter out;
  out.write(records.front().raw_bits, 64);
  auto stored_bits = records.front().raw_bits;
  std::int64_t stored_integer = java_double_to_int64(bits_double(stored_bits));
  for (std::size_t i = 1; i < records.size(); ++i) {
    const auto& record = records[i];
    if (record.tag == RecordTag::literal) {
      out.write(1, 1);
      out.write(record.raw_bits ^ stored_bits, 64);
      stored_bits = record.raw_bits;
      stored_integer = java_double_to_int64(bits_double(stored_bits));
    } else if (record.tag == RecordTag::decimal) {
      out.write(0, 1);
      write_integer(out, record.integer, record.signal, stored_integer);
      write_decimal(out, static_cast<int>(record.decimal_value),
                    record.decimal_count);
      stored_bits = double_bits(record_value(record));
    } else {
      throw std::runtime_error("unexpected Camel record inside chunk");
    }
  }
  out.write(0, 1);
  return out.finish();
}

std::int64_t read_integer(BitReader& input, std::int64_t& stored,
                          std::uint8_t& signal) {
  signal = static_cast<std::uint8_t>(input.read(1));
  const int code = static_cast<int>(input.read(2));
  std::int64_t diff = 0;
  if (code < 3) {
    diff = code - 1;
  } else {
    const bool positive = input.read(1) != 0;
    const bool wide = input.read(1) != 0;
    diff = static_cast<std::int64_t>(input.read(wide ? 16 : 3));
    if (!positive) diff = -diff;
  }
  stored = adjacent_residual::checked_restore(diff, stored);
  return stored;
}

std::pair<int, int> read_decimal(BitReader& input) {
  const int dc = static_cast<int>(input.read(2)) + 1;
  const bool is_m = input.read(1) != 0;
  std::uint64_t xor_string = is_m ? input.read(dc) << (52 - dc) : 0;
  int m = 0;
  if (dc == 1) {
    m = static_cast<int>(input.read(3));
  } else if (dc == 2) {
    const bool wide = input.read(1) != 0;
    m = static_cast<int>(input.read(wide ? 5 : 3));
  } else if (dc == 3) {
    const int kind = static_cast<int>(input.read(2));
    const int widths[] = {1, 3, 5, kMValueBits[2]};
    m = static_cast<int>(input.read(widths[kind]));
  } else {
    throw std::runtime_error("unsupported Camel decimal count");
  }
  int decimal = m;
  if (is_m) {
    const double power = static_cast<double>(kPowers[dc]);
    const auto m_bits = double_bits(static_cast<double>(m) / power + 1.0);
    const double value = bits_double(xor_string ^ m_bits) - 1.0;
    const auto rounded = java_round(value * power);
    if (rounded < 0 || rounded >= kPowers[dc])
      throw std::runtime_error("invalid Camel decoded decimal value");
    decimal = static_cast<int>(rounded);
  }
  return {dc, decimal};
}

std::vector<std::uint64_t> decode_chunk(std::span<const std::uint8_t> body,
                                        std::size_t count) {
  if (count == 0) throw std::runtime_error("Camel chunk count is zero");
  BitReader input(body);
  std::vector<std::uint64_t> values;
  values.reserve(count);
  auto stored_bits = input.read(64);
  std::int64_t stored_integer = java_double_to_int64(bits_double(stored_bits));
  values.push_back(stored_bits);
  for (std::size_t i = 1; i < count; ++i) {
    if (input.read(1) != 0) {
      stored_bits ^= input.read(64);
      stored_integer = java_double_to_int64(bits_double(stored_bits));
    } else {
      std::uint8_t signal = 0;
      const auto integer = read_integer(input, stored_integer, signal);
      const auto [dc, decimal] = read_decimal(input);
      stored_bits = double_bits(reconstruct_decimal(integer, signal, dc, decimal));
    }
    values.push_back(stored_bits);
  }
  if (input.read(1) != 0) throw std::runtime_error("Camel close marker mismatch");
  return values;
}

std::vector<std::uint8_t> encode_xor_chunk(
    std::span<const XorRecord> records) {
  if (records.empty() || records.front().tag != RecordTag::first)
    throw std::runtime_error("Camel XOR chunk has no first record");
  BitWriter out;
  out.write(records.front().bits, 64);
  for (std::size_t i = 1; i < records.size(); ++i) {
    const auto& record = records[i];
    if (record.tag == RecordTag::literal) {
      out.write(1, 1);
      out.write(record.bits, 64);
    } else if (record.tag == RecordTag::decimal) {
      out.write(0, 1);
      write_integer_delta(out, record.integer_delta, record.signal);
      write_decimal(out, static_cast<int>(record.decimal_value),
                    record.decimal_count);
    } else {
      throw std::runtime_error("unexpected Camel XOR record inside chunk");
    }
  }
  out.write(0, 1);
  return out.finish();
}

std::vector<XorRecord> decode_xor_chunk(
    std::span<const std::uint8_t> body, std::size_t count) {
  if (count == 0) throw std::runtime_error("Camel XOR chunk count is zero");
  BitReader input(body);
  std::vector<XorRecord> records;
  records.reserve(count);
  std::uint64_t stored_bits = input.read(64);
  std::int64_t stored_integer = java_double_to_int64(bits_double(stored_bits));
  records.push_back({RecordTag::first, stored_bits});
  for (std::size_t i = 1; i < count; ++i) {
    if (input.read(1) != 0) {
      const auto residual = input.read(64);
      stored_bits ^= residual;
      stored_integer = java_double_to_int64(bits_double(stored_bits));
      records.push_back({RecordTag::literal, residual});
    } else {
      const auto previous_integer = stored_integer;
      std::uint8_t signal = 0;
      const auto integer = read_integer(input, stored_integer, signal);
      const auto [dc, decimal] = read_decimal(input);
      const __int128 delta = static_cast<__int128>(integer) - previous_integer;
      if (delta < std::numeric_limits<std::int64_t>::min() ||
          delta > std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("Camel decoded integer delta overflow");
      records.push_back(
          {RecordTag::decimal, 0, static_cast<std::int64_t>(delta), signal,
           static_cast<std::uint8_t>(dc), static_cast<std::uint32_t>(decimal)});
      stored_bits = double_bits(reconstruct_decimal(integer, signal, dc, decimal));
    }
  }
  if (input.read(1) != 0)
    throw std::runtime_error("Camel close marker mismatch");
  return records;
}

}  // namespace

std::vector<std::uint8_t> eraser_forward(
    std::span<const std::int64_t> values) {
  std::vector<Record> records;
  records.reserve(values.size());
  std::int64_t stored_integer = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = bits_double(int64_bits(values[i]));
    const bool first = i % kChunk == 0;
    auto record = make_record(value, first);
    if (record.tag == RecordTag::decimal) {
      // The public Camel wire reserves at most 16 bits for an integer delta.
      // A lossless literal is the format-defined escape for a larger jump.
      // Decide here, before serializing the eraser stage, so xor_forward can
      // remain an exact inverse of xor_inverse as its public contract requires.
      const __int128 delta = static_cast<__int128>(record.integer) -
                             static_cast<__int128>(stored_integer);
      if (delta < -0xffff || delta > 0xffff)
        record = {RecordTag::literal, double_bits(value), 0, 0, 0, 0};
    }
    if (record.tag == RecordTag::decimal)
      stored_integer = record.integer;
    else
      stored_integer = java_double_to_int64(value);
    records.push_back(record);
  }
  return serialize(records);
}

std::vector<std::uint8_t> eraser_forward_int64_numeric(
    std::span<const std::int64_t> values) {
  std::vector<Record> records;
  records.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double converted = static_cast<double>(values[i]);
    std::int64_t round_trip = 0;
    if (!exact_double_to_int64(converted, round_trip) ||
        round_trip != values[i])
      throw std::runtime_error("Camel integer is not exactly representable as float64");
    records.push_back(make_record(converted, i % kChunk == 0));
  }
  return serialize(records);
}

std::vector<std::int64_t> eraser_inverse(
    std::span<const std::uint8_t> packet) {
  const auto records = parse(packet);
  std::vector<std::int64_t> values;
  values.reserve(records.size());
  for (const auto& record : records)
    values.push_back(signed_bits(double_bits(record_value(record))));
  return values;
}

std::vector<std::int64_t> eraser_inverse_int64_numeric(
    std::span<const std::uint8_t> packet) {
  const auto records = parse(packet);
  std::vector<std::int64_t> values;
  values.reserve(records.size());
  for (const auto& record : records) {
    const double value = record_value(record);
    std::int64_t converted = 0;
    if (!exact_double_to_int64(value, converted))
      throw std::runtime_error(
          "Camel decoded value is not an exactly representable int64");
    values.push_back(converted);
  }
  return values;
}

std::vector<std::uint8_t> xor_forward(
    std::span<const std::uint8_t> eraser_packet) {
  const auto records = parse(eraser_packet);
  std::vector<XorRecord> output;
  output.reserve(records.size());
  std::uint64_t stored_bits = 0;
  std::int64_t stored_integer = 0;
  for (const auto& record : records) {
    if (record.tag == RecordTag::first) {
      stored_bits = record.raw_bits;
      stored_integer = java_double_to_int64(bits_double(stored_bits));
      output.push_back({RecordTag::first, stored_bits});
    } else if (record.tag == RecordTag::literal) {
      output.push_back({RecordTag::literal,
          adjacent_residual::xor_word(record.raw_bits, stored_bits)});
      stored_bits = record.raw_bits;
      stored_integer = java_double_to_int64(bits_double(stored_bits));
    } else {
      const auto delta = adjacent_residual::checked_delta(record.integer, stored_integer);
      output.push_back(
          {RecordTag::decimal, 0, static_cast<std::int64_t>(delta),
           record.signal, record.decimal_count, record.decimal_value});
      stored_integer = record.integer;
      stored_bits = double_bits(record_value(record));
    }
  }
  return serialize_xor(output);
}

std::vector<std::uint8_t> xor_inverse(
    std::span<const std::uint8_t> xor_packet) {
  const auto input = parse_xor(xor_packet);
  std::vector<Record> records;
  records.reserve(input.size());
  std::uint64_t stored_bits = 0;
  std::int64_t stored_integer = 0;
  for (const auto& record : input) {
    if (record.tag == RecordTag::first) {
      stored_bits = record.bits;
      stored_integer = java_double_to_int64(bits_double(stored_bits));
      records.push_back({RecordTag::first, stored_bits});
    } else if (record.tag == RecordTag::literal) {
      stored_bits = adjacent_residual::xor_word(record.bits, stored_bits);
      stored_integer = java_double_to_int64(bits_double(stored_bits));
      records.push_back({RecordTag::literal, stored_bits});
    } else {
      const auto integer = adjacent_residual::checked_restore(record.integer_delta, stored_integer);
      Record restored{RecordTag::decimal};
      restored.integer = static_cast<std::int64_t>(integer);
      restored.signal = record.signal;
      restored.decimal_count = record.decimal_count;
      restored.decimal_value = record.decimal_value;
      records.push_back(restored);
      stored_integer = restored.integer;
      stored_bits = double_bits(record_value(restored));
    }
  }
  return serialize(records);
}

std::vector<std::uint8_t> bitstream_pack_forward(
    std::span<const std::uint8_t> xor_packet) {
  const auto records = parse_xor(xor_packet);
  if (records.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Camel column exceeds uint32 count");
  std::vector<std::uint8_t> out(kWireMagic.begin(), kWireMagic.end());
  put_u32_le(out, static_cast<std::uint32_t>(records.size()));
  for (std::size_t start = 0; start < records.size(); start += kChunk) {
    const auto count = std::min<std::size_t>(kChunk, records.size() - start);
    const auto body = encode_xor_chunk(
        std::span<const XorRecord>(records.data() + start, count));
    put_u32_le(out, static_cast<std::uint32_t>(count));
    put_u32_le(out, static_cast<std::uint32_t>(body.size()));
    out.insert(out.end(), body.begin(), body.end());
  }
  return out;
}

std::vector<std::uint8_t> bitstream_pack_inverse(
    std::span<const std::uint8_t> wire) {
  if (wire.size() < 8 ||
      !std::equal(kWireMagic.begin(), kWireMagic.end(), wire.begin()))
    throw std::runtime_error("wrong Camel public wire type");
  std::size_t pos = 4;
  const auto total = get_u32_le(wire, pos, "Camel public count");
  std::vector<XorRecord> records;
  records.reserve(total);
  while (records.size() < total) {
    const auto count = get_u32_le(wire, pos, "Camel chunk count");
    const auto size = get_u32_le(wire, pos, "Camel chunk size");
    if (count == 0 || count > kChunk || count > total - records.size() ||
        wire.size() - pos < size)
      throw std::runtime_error("invalid Camel chunk header");
    auto chunk = decode_xor_chunk(wire.subspan(pos, size), count);
    records.insert(records.end(), chunk.begin(), chunk.end());
    pos += size;
  }
  if (pos != wire.size()) throw std::runtime_error("Camel public wire has trailing bytes");
  return serialize_xor(records);
}

std::vector<std::uint8_t> compose(std::span<const std::int64_t> values) {
  return bitstream_pack_forward(xor_forward(eraser_forward(values)));
}

std::vector<std::int64_t> decompose(std::span<const std::uint8_t> wire) {
  return eraser_inverse(xor_inverse(bitstream_pack_inverse(wire)));
}

std::vector<std::uint8_t> compose_int64_numeric(
    std::span<const std::int64_t> values) {
  return bitstream_pack_forward(
      xor_forward(eraser_forward_int64_numeric(values)));
}

std::vector<std::int64_t> decompose_int64_numeric(
    std::span<const std::uint8_t> wire) {
  return eraser_inverse_int64_numeric(
      xor_inverse(bitstream_pack_inverse(wire)));
}

}  // namespace exact_actions::camel
