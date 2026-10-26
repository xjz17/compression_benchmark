#include "learning_codec_exact_stages.hpp"
#include "minimum_for.hpp"
#include "max_value_bitpack.hpp"
#include "zigzag.hpp"
#include "dictionary_recode.hpp"

#include "alp_float_codec.hpp"
#include "leco_double_codec.hpp"
#include "reger_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace exact_actions::learning {
namespace {

constexpr std::uint8_t kLecoResidual[] = {'L', 'R', 'D', '1'};
constexpr std::uint8_t kLecoAr[] = {'L', 'A', 'R', '1'};
constexpr std::uint8_t kAlpScale[] = {'A', 'S', 'C', '1'};
constexpr std::uint8_t kAlpFor[] = {'A', 'F', 'O', '1'};
constexpr std::uint8_t kAlpWire[] = {'A', 'L', 'P', '1'};
constexpr std::uint8_t kAlpDictionary[] = {'A', 'D', 'I', '1'};
constexpr std::uint8_t kRegerReorder[] = {'R', 'R', 'O', '1'};
constexpr std::uint8_t kRegerDelta[] = {'R', 'S', 'D', '1'};
constexpr std::uint8_t kRegerAr[] = {'R', 'A', 'R', '1'};
constexpr std::uint8_t kRegerFor[] = {'R', 'F', 'O', '1'};
constexpr std::uint8_t kRegerWire[] = {'R', 'G', 'R', '1'};
constexpr std::uint8_t kBuffScale[] = {'B', 'S', 'C', '1'};
constexpr std::uint8_t kBuffFor[] = {'B', 'F', 'R', '1'};
constexpr std::uint8_t kBuffSplit[] = {'B', 'S', 'F', '1'};
constexpr std::size_t kLecoBlock = 512;
constexpr std::size_t kAlpBlock = 1024;

void put_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}
void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint16_t get_u16(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 2 > in.size()) throw std::runtime_error("stage packet truncated u16");
  const auto value = static_cast<std::uint16_t>(in[pos]) |
                     static_cast<std::uint16_t>(in[pos + 1] << 8);
  pos += 2;
  return value;
}
std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("stage packet truncated u32");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(in[pos++]) << shift;
  return value;
}
std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("stage packet truncated u64");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(in[pos++]) << shift;
  return value;
}

void put_i64(std::vector<std::uint8_t>& out, std::int64_t value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  put_u64(out, bits);
}
std::int64_t get_i64(std::span<const std::uint8_t> in, std::size_t& pos) {
  const auto bits = get_u64(in, pos);
  std::int64_t value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
void put_double(std::vector<std::uint8_t>& out, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  put_u64(out, bits);
}
double get_double(std::span<const std::uint8_t> in, std::size_t& pos) {
  const auto bits = get_u64(in, pos);
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <std::size_t N>
void require_magic(std::span<const std::uint8_t> in,
                   const std::uint8_t (&magic)[N]) {
  if (in.size() < N || std::memcmp(in.data(), magic, N) != 0)
    throw std::runtime_error("bad learning-codec stage packet magic");
}

std::uint64_t to_u64(std::int64_t value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
std::int64_t to_i64(std::uint64_t value) {
  std::int64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
int bit_width(std::uint64_t value) {
  return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

class BitWriter {
 public:
  void write(std::uint64_t value, int width) {
    if (width < 0 || width > 64) throw std::runtime_error("invalid bit width");
    for (int bit = width - 1; bit >= 0; --bit) {
      current_ = static_cast<std::uint8_t>(
          (current_ << 1) | ((value >> bit) & 1U));
      if (++used_ == 8) {
        bytes_.push_back(current_);
        current_ = 0;
        used_ = 0;
      }
    }
  }
  std::vector<std::uint8_t> finish() {
    if (used_ != 0)
      bytes_.push_back(static_cast<std::uint8_t>(current_ << (8 - used_)));
    return std::move(bytes_);
  }

 private:
  std::vector<std::uint8_t> bytes_;
  std::uint8_t current_ = 0;
  int used_ = 0;
};

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}
  std::uint64_t read(int width) {
    if (width < 0 || width > 64) throw std::runtime_error("invalid bit width");
    std::uint64_t value = 0;
    for (int i = 0; i < width; ++i) {
      if (bit_ >= bytes_.size() * 8)
        throw std::runtime_error("bitpacked stage body truncated");
      value = (value << 1) |
              ((bytes_[bit_ / 8] >> (7 - bit_ % 8)) & 1U);
      ++bit_;
    }
    return value;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_ = 0;
};

std::vector<std::int64_t> raw_ints(std::span<const std::uint8_t> bytes) {
  if (bytes.size() % 8 != 0) throw std::runtime_error("int64 input is not aligned");
  std::vector<std::int64_t> values(bytes.size() / 8);
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
std::vector<double> raw_doubles(std::span<const std::uint8_t> bytes) {
  if (bytes.size() % 8 != 0) throw std::runtime_error("float64 input is not aligned");
  std::vector<double> values(bytes.size() / 8);
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
std::vector<std::uint8_t> int_bytes(std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> out(values.size() * 8);
  if (!out.empty()) std::memcpy(out.data(), values.data(), out.size());
  return out;
}
std::vector<std::uint8_t> double_bytes(std::span<const double> values) {
  std::vector<std::uint8_t> out(values.size() * 8);
  if (!out.empty()) std::memcpy(out.data(), values.data(), out.size());
  return out;
}

bool exact_decimal_scale(const std::vector<double>& values,
                         std::uint8_t& scale,
                         std::vector<std::int64_t>& integers) {
  const int candidate = leco_double::maxLosslessDecimalScale(values);
  if (candidate < 0 || candidate > 15 ||
      !leco_double::quantizeScaledInts(values, candidate, integers))
    return false;
  const double factor = std::pow(10.0, candidate);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double reconstructed = static_cast<double>(integers[i]) / factor;
    std::uint64_t expected = 0;
    std::uint64_t actual = 0;
    std::memcpy(&expected, &values[i], sizeof(expected));
    std::memcpy(&actual, &reconstructed, sizeof(actual));
    if (expected != actual) return false;
  }
  scale = static_cast<std::uint8_t>(candidate);
  return true;
}

std::pair<double, double> fit_line(std::span<const std::int64_t> values) {
  if (values.empty()) return {0.0, 0.0};
  if (values.size() == 1) return {static_cast<double>(values.front()), 0.0};
  const long double mean_x = static_cast<long double>(values.size() - 1) / 2;
  long double mean_y = 0;
  for (const auto value : values) mean_y += static_cast<long double>(value);
  mean_y /= static_cast<long double>(values.size());
  long double numerator = 0;
  long double denominator = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto dx = static_cast<long double>(i) - mean_x;
    numerator += dx * (static_cast<long double>(values[i]) - mean_y);
    denominator += dx * dx;
  }
  const auto slope = static_cast<double>(numerator / denominator);
  const auto intercept = static_cast<double>(
      mean_y - static_cast<long double>(slope) * mean_x);
  return {intercept, slope};
}

std::int64_t predict(double intercept, double slope, std::size_t index) {
  const long double estimate = static_cast<long double>(intercept) +
                               static_cast<long double>(slope) * index;
  if (estimate <= static_cast<long double>(std::numeric_limits<std::int64_t>::min()))
    return std::numeric_limits<std::int64_t>::min();
  if (estimate >= static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
    return std::numeric_limits<std::int64_t>::max();
  return static_cast<std::int64_t>(std::llround(estimate));
}

struct LecoBlock {
  std::uint16_t length = 0;
  double intercept = 0;
  double slope = 0;
  std::vector<std::uint64_t> residuals;
};

std::pair<std::uint32_t, std::vector<LecoBlock>> parse_leco_packet(
    std::span<const std::uint8_t> packet, const std::uint8_t (&magic)[4]) {
  require_magic(packet, magic);
  std::size_t pos = 4;
  const auto count = get_u32(packet, pos);
  std::vector<LecoBlock> blocks;
  std::size_t total = 0;
  while (total < count) {
    LecoBlock block;
    block.length = get_u16(packet, pos);
    block.intercept = get_double(packet, pos);
    block.slope = get_double(packet, pos);
    if (block.length == 0 || total + block.length > count)
      throw std::runtime_error("invalid LECO autoregression block");
    block.residuals.reserve(block.length);
    for (std::uint16_t i = 0; i < block.length; ++i)
      block.residuals.push_back(get_u64(packet, pos));
    total += block.length;
    blocks.push_back(std::move(block));
  }
  if (pos != packet.size()) throw std::runtime_error("LECO AR trailing bytes");
  return {count, std::move(blocks)};
}

std::vector<std::uint8_t> make_leco_packet(
    std::uint32_t count, const std::vector<LecoBlock>& blocks,
    const std::uint8_t (&magic)[4]) {
  std::vector<std::uint8_t> out(std::begin(magic), std::end(magic));
  put_u32(out, count);
  for (const auto& block : blocks) {
    put_u16(out, block.length);
    put_double(out, block.intercept);
    put_double(out, block.slope);
    for (const auto residual : block.residuals) put_u64(out, residual);
  }
  return out;
}

std::pair<std::uint32_t, std::vector<LecoBlock>> parse_leco_ar(
    std::span<const std::uint8_t> packet) {
  return parse_leco_packet(packet, kLecoAr);
}

std::vector<std::uint8_t> make_leco_ar(
    std::uint32_t count, const std::vector<LecoBlock>& blocks) {
  return make_leco_packet(count, blocks, kLecoAr);
}

struct AlpException {
  std::uint16_t index = 0;
  std::uint64_t bits = 0;
};
struct AlpBlock {
  std::uint16_t length = 0;
  bool scaled = false;
  std::uint8_t scale = 0;
  std::vector<std::int64_t> integers;
  std::int64_t base = 0;
  std::vector<std::uint64_t> offsets;
  std::vector<AlpException> exceptions;
};

std::vector<std::uint8_t> make_alp_scale(
    std::uint32_t count, const std::vector<AlpBlock>& blocks) {
  std::vector<std::uint8_t> out(std::begin(kAlpScale), std::end(kAlpScale));
  put_u32(out, count);
  for (const auto& block : blocks) {
    put_u16(out, block.length);
    out.push_back(block.scaled ? 1 : 0);
    out.push_back(block.scale);
    put_u16(out, static_cast<std::uint16_t>(block.exceptions.size()));
    for (const auto value : block.integers) put_i64(out, value);
    for (const auto& exception : block.exceptions) {
      put_u16(out, exception.index);
      put_u64(out, exception.bits);
    }
  }
  return out;
}

std::pair<std::uint32_t, std::vector<AlpBlock>> parse_alp_scale(
    std::span<const std::uint8_t> packet) {
  require_magic(packet, kAlpScale);
  std::size_t pos = 4;
  const auto count = get_u32(packet, pos);
  std::vector<AlpBlock> blocks;
  std::size_t total = 0;
  while (total < count) {
    AlpBlock block;
    block.length = get_u16(packet, pos);
    if (pos + 2 > packet.size()) throw std::runtime_error("ALP scale header truncated");
    block.scaled = packet[pos++] != 0;
    block.scale = packet[pos++];
    const auto exception_count = get_u16(packet, pos);
    if (block.length == 0 || total + block.length > count)
      throw std::runtime_error("invalid ALP scale block");
    block.integers.reserve(block.length);
    for (std::uint16_t i = 0; i < block.length; ++i)
      block.integers.push_back(get_i64(packet, pos));
    block.exceptions.reserve(exception_count);
    for (std::uint16_t i = 0; i < exception_count; ++i) {
      AlpException exception{get_u16(packet, pos), get_u64(packet, pos)};
      if (exception.index >= block.length)
        throw std::runtime_error("ALP exception index out of range");
      block.exceptions.push_back(exception);
    }
    total += block.length;
    blocks.push_back(std::move(block));
  }
  if (pos != packet.size()) throw std::runtime_error("ALP scale trailing bytes");
  return {count, std::move(blocks)};
}

std::vector<std::uint8_t> make_alp_for(
    std::uint32_t count, const std::vector<AlpBlock>& blocks) {
  std::vector<std::uint8_t> out(std::begin(kAlpFor), std::end(kAlpFor));
  put_u32(out, count);
  for (const auto& block : blocks) {
    put_u16(out, block.length);
    out.push_back(block.scaled ? 1 : 0);
    out.push_back(block.scale);
    put_u16(out, static_cast<std::uint16_t>(block.exceptions.size()));
    put_i64(out, block.base);
    for (const auto value : block.offsets) put_u64(out, value);
    for (const auto& exception : block.exceptions) {
      put_u16(out, exception.index);
      put_u64(out, exception.bits);
    }
  }
  return out;
}

std::pair<std::uint32_t, std::vector<AlpBlock>> parse_alp_for(
    std::span<const std::uint8_t> packet) {
  require_magic(packet, kAlpFor);
  std::size_t pos = 4;
  const auto count = get_u32(packet, pos);
  std::vector<AlpBlock> blocks;
  std::size_t total = 0;
  while (total < count) {
    AlpBlock block;
    block.length = get_u16(packet, pos);
    if (pos + 2 > packet.size()) throw std::runtime_error("ALP FOR header truncated");
    block.scaled = packet[pos++] != 0;
    block.scale = packet[pos++];
    const auto exception_count = get_u16(packet, pos);
    block.base = get_i64(packet, pos);
    if (block.length == 0 || total + block.length > count)
      throw std::runtime_error("invalid ALP FOR block");
    block.offsets.reserve(block.length);
    for (std::uint16_t i = 0; i < block.length; ++i)
      block.offsets.push_back(get_u64(packet, pos));
    block.exceptions.reserve(exception_count);
    for (std::uint16_t i = 0; i < exception_count; ++i) {
      AlpException exception{get_u16(packet, pos), get_u64(packet, pos)};
      if (exception.index >= block.length)
        throw std::runtime_error("ALP exception index out of range");
      block.exceptions.push_back(exception);
    }
    total += block.length;
    blocks.push_back(std::move(block));
  }
  if (pos != packet.size()) throw std::runtime_error("ALP FOR trailing bytes");
  return {count, std::move(blocks)};
}

std::vector<double> recover_alp(const std::vector<AlpBlock>& blocks) {
  std::vector<double> values;
  for (const auto& block : blocks) {
    std::vector<double> decoded(block.length);
    const double factor = std::pow(10.0, block.scale);
    for (std::size_t i = 0; i < block.length; ++i) {
      const auto integer = block.integers.empty()
          ? to_i64(to_u64(block.base) + block.offsets[i])
          : block.integers[i];
      if (block.scaled) {
        decoded[i] = integer == 0 && to_u64(integer) != 0
                         ? 0.0
                         : static_cast<double>(integer) / factor;
      } else {
        const auto bits = to_u64(integer);
        std::memcpy(&decoded[i], &bits, sizeof(bits));
      }
    }
    for (const auto& exception : block.exceptions)
      std::memcpy(&decoded[exception.index], &exception.bits, sizeof(double));
    values.insert(values.end(), decoded.begin(), decoded.end());
  }
  return values;
}

struct ReorderPacket {
  std::vector<std::int64_t> values;
  std::vector<std::uint32_t> original_index;
};

std::vector<std::uint8_t> make_reorder(const ReorderPacket& packet) {
  std::vector<std::uint8_t> out(std::begin(kRegerReorder), std::end(kRegerReorder));
  put_u32(out, static_cast<std::uint32_t>(packet.values.size()));
  for (const auto value : packet.values) put_i64(out, value);
  for (const auto index : packet.original_index) put_u32(out, index);
  return out;
}
ReorderPacket parse_reorder(std::span<const std::uint8_t> packet) {
  require_magic(packet, kRegerReorder);
  std::size_t pos = 4;
  const auto count = get_u32(packet, pos);
  ReorderPacket out;
  out.values.reserve(count);
  out.original_index.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) out.values.push_back(get_i64(packet, pos));
  std::vector<bool> seen(count, false);
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto index = get_u32(packet, pos);
    if (index >= count || seen[index]) throw std::runtime_error("invalid REGER permutation");
    seen[index] = true;
    out.original_index.push_back(index);
  }
  if (pos != packet.size()) throw std::runtime_error("REGER reorder trailing bytes");
  return out;
}

struct RegerResidualPacket {
  std::vector<std::uint32_t> original_index;
  std::int64_t first = 0;
  std::vector<std::uint64_t> residuals;
  std::uint64_t base = 0;
  std::vector<std::uint64_t> offsets;
};

std::vector<std::uint8_t> make_reger_residual(
    const RegerResidualPacket& packet, bool normalized,
    bool signed_residual = false) {
  const auto& magic = normalized ? kRegerFor
                                 : (signed_residual ? kRegerDelta : kRegerAr);
  std::vector<std::uint8_t> out(magic, magic + 4);
  put_u32(out, static_cast<std::uint32_t>(packet.original_index.size()));
  put_i64(out, packet.first);
  if (normalized) put_u64(out, packet.base);
  for (const auto index : packet.original_index) put_u32(out, index);
  for (const auto value : normalized ? packet.offsets : packet.residuals)
    put_u64(out, value);
  return out;
}
RegerResidualPacket parse_reger_residual(
    std::span<const std::uint8_t> packet, bool normalized,
    bool signed_residual = false) {
  if (normalized) require_magic(packet, kRegerFor);
  else if (signed_residual) require_magic(packet, kRegerDelta);
  else require_magic(packet, kRegerAr);
  std::size_t pos = 4;
  const auto count = get_u32(packet, pos);
  RegerResidualPacket out;
  out.first = get_i64(packet, pos);
  if (normalized) out.base = get_u64(packet, pos);
  out.original_index.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i)
    out.original_index.push_back(get_u32(packet, pos));
  const auto residual_count = count == 0 ? 0 : count - 1;
  auto& values = normalized ? out.offsets : out.residuals;
  values.reserve(residual_count);
  for (std::uint32_t i = 0; i < residual_count; ++i)
    values.push_back(get_u64(packet, pos));
  if (pos != packet.size()) throw std::runtime_error("REGER residual trailing bytes");
  return out;
}

struct BuffPacket {
  bool scaled = false;
  std::uint8_t scale = 0;
  std::vector<std::int64_t> integers;
  std::vector<std::uint64_t> offsets;
  std::int64_t base = 0;
  std::uint8_t width = 0;
  std::vector<std::uint8_t> planes;
};

std::vector<std::uint8_t> make_buff_scale(const BuffPacket& packet) {
  std::vector<std::uint8_t> out(std::begin(kBuffScale), std::end(kBuffScale));
  out.push_back(packet.scaled ? 1 : 0);
  out.push_back(packet.scale);
  put_u32(out, static_cast<std::uint32_t>(packet.integers.size()));
  for (const auto value : packet.integers) put_i64(out, value);
  return out;
}
BuffPacket parse_buff_scale(std::span<const std::uint8_t> packet) {
  require_magic(packet, kBuffScale);
  std::size_t pos = 4;
  if (pos + 2 > packet.size()) throw std::runtime_error("BUFF scale header truncated");
  BuffPacket out;
  out.scaled = packet[pos++] != 0;
  out.scale = packet[pos++];
  const auto count = get_u32(packet, pos);
  out.integers.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) out.integers.push_back(get_i64(packet, pos));
  if (pos != packet.size()) throw std::runtime_error("BUFF scale trailing bytes");
  return out;
}
std::vector<std::uint8_t> make_buff_for(const BuffPacket& packet) {
  if (packet.offsets.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("BUFF FOR count exceeds uint32");
  std::vector<std::uint8_t> out(std::begin(kBuffFor), std::end(kBuffFor));
  out.push_back(packet.scaled ? 1 : 0);
  out.push_back(packet.scale);
  put_u32(out, static_cast<std::uint32_t>(packet.offsets.size()));
  put_i64(out, packet.base);
  for (const auto offset : packet.offsets) put_u64(out, offset);
  return out;
}
BuffPacket parse_buff_for(std::span<const std::uint8_t> packet) {
  require_magic(packet, kBuffFor);
  std::size_t pos = 4;
  if (pos + 2 > packet.size()) throw std::runtime_error("BUFF FOR header truncated");
  BuffPacket out;
  out.scaled = packet[pos++] != 0;
  out.scale = packet[pos++];
  const auto count = get_u32(packet, pos);
  out.base = get_i64(packet, pos);
  if (packet.size() - pos != static_cast<std::size_t>(count) * 8)
    throw std::runtime_error("BUFF FOR offset body mismatch");
  out.integers.resize(count);  // Count only; the plane adapter never restores values.
  out.offsets.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) out.offsets.push_back(get_u64(packet, pos));
  return out;
}
std::vector<std::uint8_t> make_buff_split(const BuffPacket& packet) {
  std::vector<std::uint8_t> out(std::begin(kBuffSplit), std::end(kBuffSplit));
  out.push_back(packet.scaled ? 1 : 0);
  out.push_back(packet.scale);
  put_u32(out, static_cast<std::uint32_t>(packet.integers.size()));
  put_i64(out, packet.base);
  out.push_back(packet.width);
  put_u32(out, static_cast<std::uint32_t>(packet.planes.size()));
  out.insert(out.end(), packet.planes.begin(), packet.planes.end());
  return out;
}
BuffPacket parse_buff_split(std::span<const std::uint8_t> packet) {
  require_magic(packet, kBuffSplit);
  std::size_t pos = 4;
  if (pos + 2 > packet.size()) throw std::runtime_error("BUFF split header truncated");
  BuffPacket out;
  out.scaled = packet[pos++] != 0;
  out.scale = packet[pos++];
  const auto count = get_u32(packet, pos);
  out.integers.resize(count);
  out.base = get_i64(packet, pos);
  if (pos >= packet.size()) throw std::runtime_error("BUFF split width truncated");
  out.width = packet[pos++];
  if (out.width > 64) throw std::runtime_error("BUFF invalid split width");
  const auto plane_size = get_u32(packet, pos);
  if (pos + plane_size != packet.size()) throw std::runtime_error("BUFF split body mismatch");
  out.planes.assign(packet.begin() + static_cast<std::ptrdiff_t>(pos), packet.end());
  out.offsets = buff_byte_planes_join(out.planes, count, out.width);
  return out;
}

}  // namespace

std::vector<std::uint8_t> alp_original_encode(
    std::span<const std::uint8_t> float64_le) {
  return alp_float::encodeDoubleColumnExact(raw_doubles(float64_le));
}

std::vector<std::uint8_t> reger_original_encode(
    std::span<const std::uint8_t> int64_le) {
  return reger_codec::encodeInt64Column(raw_ints(int64_le));
}

std::vector<std::uint8_t> leco_regression_forward(
    std::span<const std::uint8_t> int64_le) {
  const auto values = raw_ints(int64_le);
  if (values.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("LECO input exceeds uint32 count");
  std::vector<LecoBlock> blocks;
  for (std::size_t offset = 0; offset < values.size(); offset += kLecoBlock) {
    const auto length = std::min(kLecoBlock, values.size() - offset);
    const auto view = std::span<const std::int64_t>(values).subspan(offset, length);
    const auto [intercept, slope] = fit_line(view);
    LecoBlock block;
    block.length = static_cast<std::uint16_t>(length);
    block.intercept = intercept;
    block.slope = slope;
    block.residuals.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
      const auto residual = to_i64(to_u64(view[i]) - to_u64(predict(intercept, slope, i)));
      block.residuals.push_back(to_u64(residual));
    }
    blocks.push_back(std::move(block));
  }
  return make_leco_packet(static_cast<std::uint32_t>(values.size()), blocks,
                          kLecoResidual);
}

std::vector<std::uint8_t> leco_regression_inverse(
    std::span<const std::uint8_t> packet) {
  const auto [count, blocks] = parse_leco_packet(packet, kLecoResidual);
  std::vector<std::int64_t> values;
  values.reserve(count);
  for (const auto& block : blocks)
    for (std::size_t i = 0; i < block.length; ++i)
      values.push_back(to_i64(to_u64(predict(block.intercept, block.slope, i)) +
                              block.residuals[i]));
  return int_bytes(values);
}

std::vector<std::uint8_t> leco_zigzag_forward(
    std::span<const std::uint8_t> packet) {
  auto [count, blocks] = parse_leco_packet(packet, kLecoResidual);
  for (auto& block : blocks)
    for (auto& residual : block.residuals)
      residual = compression_operators::zigzag::encode64(to_i64(residual));
  return make_leco_ar(count, blocks);
}

std::vector<std::uint8_t> leco_zigzag_inverse(
    std::span<const std::uint8_t> packet) {
  auto [count, blocks] = parse_leco_ar(packet);
  for (auto& block : blocks)
    for (auto& residual : block.residuals)
      residual = to_u64(compression_operators::zigzag::decode64(residual));
  return make_leco_packet(count, blocks, kLecoResidual);
}

std::vector<std::uint8_t> leco_autoregression_forward(
    std::span<const std::uint8_t> int64_le) {
  return leco_zigzag_forward(leco_regression_forward(int64_le));
}

std::vector<std::uint8_t> leco_autoregression_inverse(
    std::span<const std::uint8_t> packet) {
  return leco_regression_inverse(leco_zigzag_inverse(packet));
}

std::vector<std::uint8_t> leco_bitpack_forward(
    std::span<const std::uint8_t> packet) {
  const auto [count, blocks] = parse_leco_ar(packet);
  std::vector<std::uint8_t> out;
  put_u32(out, count);
  for (const auto& block : blocks) {
    put_u16(out, block.length);
    put_double(out, block.intercept);
    put_double(out, block.slope);
    const auto packed = max_value_bitpack::pack(block.residuals);
    out.push_back(packed.width);
    const auto& body = packed.payload;
    out.insert(out.end(), body.begin(), body.end());
  }
  return out;
}

std::vector<std::uint8_t> leco_bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  std::size_t pos = 0;
  const auto count = get_u32(wire, pos);
  std::vector<LecoBlock> blocks;
  std::size_t total = 0;
  while (total < count) {
    LecoBlock block;
    block.length = get_u16(wire, pos);
    block.intercept = get_double(wire, pos);
    block.slope = get_double(wire, pos);
    if (pos >= wire.size()) throw std::runtime_error("LECO wire width truncated");
    const int width = wire[pos++];
    if (block.length == 0 || total + block.length > count || width > 64)
      throw std::runtime_error("invalid LECO packed block");
    const auto body_size = max_value_bitpack::payload_size(block.length, width);
    if (pos + body_size > wire.size()) throw std::runtime_error("LECO wire body truncated");
    block.residuals = max_value_bitpack::unpack(
        wire.subspan(pos, body_size), block.length, width);
    pos += body_size;
    total += block.length;
    blocks.push_back(std::move(block));
  }
  if (pos != wire.size()) throw std::runtime_error("LECO wire trailing bytes");
  return make_leco_ar(count, blocks);
}

std::vector<std::uint8_t> alp_scaling_forward(
    std::span<const std::uint8_t> float64_le) {
  const auto values = raw_doubles(float64_le);
  if (values.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("ALP input exceeds uint32 count");
  std::vector<AlpBlock> blocks;
  for (std::size_t offset = 0; offset < values.size(); offset += kAlpBlock) {
    const auto length = std::min(kAlpBlock, values.size() - offset);
    const std::vector<double> window(
        values.begin() + static_cast<std::ptrdiff_t>(offset),
        values.begin() + static_cast<std::ptrdiff_t>(offset + length));
    AlpBlock block;
    block.length = static_cast<std::uint16_t>(length);
    block.scaled = exact_decimal_scale(window, block.scale, block.integers);
    if (!block.scaled) {
      block.integers.reserve(length);
      block.integers.clear();
      for (const auto value : window) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        block.integers.push_back(to_i64(bits));
      }
    }
    blocks.push_back(std::move(block));
  }
  return make_alp_scale(static_cast<std::uint32_t>(values.size()), blocks);
}

std::vector<std::uint8_t> alp_scaling_inverse(
    std::span<const std::uint8_t> packet) {
  const auto [count, blocks] = parse_alp_scale(packet);
  auto values = recover_alp(blocks);
  if (values.size() != count) throw std::runtime_error("ALP scaling count mismatch");
  return double_bytes(values);
}

std::vector<std::uint8_t> alp_for_forward(
    std::span<const std::uint8_t> packet) {
  auto [count, blocks] = parse_alp_scale(packet);
  for (auto& block : blocks) {
    auto encoded = minimum_for::encode<std::int64_t>(block.integers);
    block.base = encoded.minimum;
    block.offsets = std::move(encoded.offsets);
    block.integers.clear();
  }
  return make_alp_for(count, blocks);
}

std::vector<std::uint8_t> alp_for_inverse(
    std::span<const std::uint8_t> packet) {
  auto [count, blocks] = parse_alp_for(packet);
  for (auto& block : blocks) {
    block.integers.reserve(block.offsets.size());
    for (const auto offset : block.offsets)
      block.integers.push_back(minimum_for::restore(block.base, offset));
    block.offsets.clear();
  }
  return make_alp_scale(count, blocks);
}

std::vector<std::uint8_t> alp_bitpack_forward(
    std::span<const std::uint8_t> packet) {
  const auto [count, blocks] = parse_alp_for(packet);
  std::vector<std::uint8_t> out(std::begin(kAlpWire), std::end(kAlpWire));
  put_u32(out, count);
  for (const auto& block : blocks) {
    put_u16(out, block.length);
    out.push_back(block.scaled ? 1 : 0);
    out.push_back(block.scale);
    put_u16(out, static_cast<std::uint16_t>(block.exceptions.size()));
    put_i64(out, block.base);
    const auto packed = max_value_bitpack::pack(block.offsets);
    out.push_back(packed.width);
    const auto& body = packed.payload;
    put_u32(out, static_cast<std::uint32_t>(body.size()));
    out.insert(out.end(), body.begin(), body.end());
    for (const auto& exception : block.exceptions) {
      put_u16(out, exception.index);
      put_u64(out, exception.bits);
    }
  }
  return out;
}

std::vector<std::uint8_t> alp_bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  require_magic(wire, kAlpWire);
  std::size_t pos = 4;
  const auto count = get_u32(wire, pos);
  std::vector<AlpBlock> blocks;
  std::size_t total = 0;
  while (total < count) {
    AlpBlock block;
    block.length = get_u16(wire, pos);
    if (pos + 2 > wire.size()) throw std::runtime_error("ALP wire header truncated");
    block.scaled = wire[pos++] != 0;
    block.scale = wire[pos++];
    const auto exception_count = get_u16(wire, pos);
    block.base = get_i64(wire, pos);
    if (pos >= wire.size()) throw std::runtime_error("ALP wire width truncated");
    const int width = wire[pos++];
    const auto body_size = get_u32(wire, pos);
    const auto expected = max_value_bitpack::payload_size(block.length, width);
    if (width > 64 || body_size != expected || pos + body_size > wire.size())
      throw std::runtime_error("invalid ALP packed body");
    block.offsets = max_value_bitpack::unpack(
        wire.subspan(pos, body_size), block.length, width);
    pos += body_size;
    for (std::uint16_t i = 0; i < exception_count; ++i)
      block.exceptions.push_back({get_u16(wire, pos), get_u64(wire, pos)});
    total += block.length;
    blocks.push_back(std::move(block));
  }
  if (pos != wire.size()) throw std::runtime_error("ALP wire trailing bytes");
  return make_alp_for(count, blocks);
}

std::vector<std::uint8_t> alp_residual_core_forward(
    std::span<const std::uint8_t> prefix) {
  const auto raw = alp_scaling_inverse(
      alp_for_inverse(alp_bitpack_inverse(prefix)));
  return alp_float::encodeDoubleColumnExact(raw_doubles(raw));
}

std::vector<std::uint8_t> alp_residual_core_inverse(
    std::span<const std::uint8_t> wire) {
  const auto raw = double_bytes(alp_float::decodeDoubleColumnExact(wire));
  return alp_bitpack_forward(alp_for_forward(alp_scaling_forward(raw)));
}

std::vector<std::uint8_t> alp_dictionary_forward(
    std::span<const std::uint8_t> float64_le) {
  const auto values = raw_doubles(float64_le);
  if (values.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("ALP dictionary input exceeds uint32 count");
  std::vector<std::uint64_t> words;
  words.reserve(values.size());
  for (const auto value : values) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    words.push_back(bits);
  }
  const auto encoded = dictionary_recode::encode<std::uint64_t>(words, dictionary_recode::Order::first_seen);
  const auto& dictionary = encoded.dictionary;
  const int width = bit_width(dictionary.empty() ? 0 : dictionary.size() - 1);
  const auto body = max_value_bitpack::pack_fixed(encoded.ids, width).payload;
  std::vector<std::uint8_t> out(std::begin(kAlpDictionary), std::end(kAlpDictionary));
  put_u32(out, static_cast<std::uint32_t>(values.size()));
  put_u32(out, static_cast<std::uint32_t>(dictionary.size()));
  out.push_back(static_cast<std::uint8_t>(width));
  for (const auto bits : dictionary) put_u64(out, bits);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> alp_dictionary_inverse(
    std::span<const std::uint8_t> packet) {
  require_magic(packet, kAlpDictionary);
  std::size_t pos = 4;
  const auto count = get_u32(packet, pos);
  const auto dictionary_count = get_u32(packet, pos);
  if (pos >= packet.size()) throw std::runtime_error("ALP dictionary width truncated");
  const int width = packet[pos++];
  if (width > 32 || (dictionary_count == 0 && count != 0))
    throw std::runtime_error("invalid ALP dictionary header");
  std::vector<std::uint64_t> dictionary;
  dictionary.reserve(dictionary_count);
  for (std::uint32_t i = 0; i < dictionary_count; ++i)
    dictionary.push_back(get_u64(packet, pos));
  const auto body_size = (static_cast<std::size_t>(width) * count + 7) / 8;
  if (pos + body_size != packet.size()) throw std::runtime_error("ALP dictionary body mismatch");
  const auto ids = max_value_bitpack::unpack(packet.subspan(pos), count, width);
  const auto words = dictionary_recode::decode<std::uint64_t>(dictionary, ids);
  std::vector<double> values(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    std::memcpy(&values[i], &words[i], sizeof(double));
  }
  return double_bytes(values);
}

std::vector<std::uint8_t> reger_reorder_forward(
    std::span<const std::uint8_t> int64_le) {
  const auto values = raw_ints(int64_le);
  if (values.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("REGER input exceeds uint32 count");
  std::vector<std::uint32_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](const auto left, const auto right) {
    return values[left] < values[right];
  });
  ReorderPacket packet;
  packet.values.reserve(values.size());
  packet.original_index = order;
  for (const auto index : order) packet.values.push_back(values[index]);
  return make_reorder(packet);
}

std::vector<std::uint8_t> reger_reorder_inverse(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_reorder(packet);
  std::vector<std::int64_t> values(parsed.values.size());
  for (std::size_t i = 0; i < parsed.values.size(); ++i)
    values[parsed.original_index[i]] = parsed.values[i];
  return int_bytes(values);
}

std::vector<std::uint8_t> reger_delta_forward(
    std::span<const std::uint8_t> packet) {
  const auto reorder = parse_reorder(packet);
  RegerResidualPacket out;
  out.original_index = reorder.original_index;
  if (!reorder.values.empty()) out.first = reorder.values.front();
  out.residuals.reserve(reorder.values.empty() ? 0 : reorder.values.size() - 1);
  for (std::size_t i = 1; i < reorder.values.size(); ++i) {
    const auto delta = to_i64(to_u64(reorder.values[i]) - to_u64(reorder.values[i - 1]));
    out.residuals.push_back(to_u64(delta));
  }
  return make_reger_residual(out, false, true);
}

std::vector<std::uint8_t> reger_delta_inverse(
    std::span<const std::uint8_t> packet) {
  const auto residual = parse_reger_residual(packet, false, true);
  ReorderPacket out;
  out.original_index = residual.original_index;
  if (!out.original_index.empty()) out.values.push_back(residual.first);
  for (const auto code : residual.residuals)
    out.values.push_back(to_i64(to_u64(out.values.back()) + code));
  return make_reorder(out);
}

std::vector<std::uint8_t> reger_zigzag_forward(
    std::span<const std::uint8_t> packet) {
  auto residual = parse_reger_residual(packet, false, true);
  for (auto& value : residual.residuals)
    value = compression_operators::zigzag::encode64(to_i64(value));
  return make_reger_residual(residual, false);
}

std::vector<std::uint8_t> reger_zigzag_inverse(
    std::span<const std::uint8_t> packet) {
  auto residual = parse_reger_residual(packet, false);
  for (auto& value : residual.residuals)
    value = to_u64(compression_operators::zigzag::decode64(value));
  return make_reger_residual(residual, false, true);
}

std::vector<std::uint8_t> reger_autoregression_forward(
    std::span<const std::uint8_t> packet) {
  return reger_zigzag_forward(reger_delta_forward(packet));
}

std::vector<std::uint8_t> reger_autoregression_inverse(
    std::span<const std::uint8_t> packet) {
  return reger_delta_inverse(reger_zigzag_inverse(packet));
}

std::vector<std::uint8_t> reger_for_forward(
    std::span<const std::uint8_t> packet) {
  auto residual = parse_reger_residual(packet, false);
  auto encoded = minimum_for::encode<std::uint64_t>(residual.residuals);
  residual.base = encoded.minimum;
  residual.offsets = std::move(encoded.offsets);
  residual.residuals.clear();
  return make_reger_residual(residual, true);
}

std::vector<std::uint8_t> reger_for_inverse(
    std::span<const std::uint8_t> packet) {
  auto residual = parse_reger_residual(packet, true);
  residual.residuals.reserve(residual.offsets.size());
  for (const auto value : residual.offsets) residual.residuals.push_back(minimum_for::restore(residual.base, value));
  residual.offsets.clear();
  return make_reger_residual(residual, false);
}

std::vector<std::uint8_t> reger_bitpack_forward(
    std::span<const std::uint8_t> packet) {
  const auto residual = parse_reger_residual(packet, true);
  const auto packed = max_value_bitpack::pack(residual.offsets);
  const auto width = packed.width;
  const auto& body = packed.payload;
  std::vector<std::uint8_t> out(std::begin(kRegerWire), std::end(kRegerWire));
  put_u32(out, static_cast<std::uint32_t>(residual.original_index.size()));
  put_i64(out, residual.first);
  put_u64(out, residual.base);
  out.push_back(static_cast<std::uint8_t>(width));
  for (const auto index : residual.original_index) put_u32(out, index);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> reger_bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  require_magic(wire, kRegerWire);
  std::size_t pos = 4;
  const auto count = get_u32(wire, pos);
  RegerResidualPacket residual;
  residual.first = get_i64(wire, pos);
  residual.base = get_u64(wire, pos);
  if (pos >= wire.size()) throw std::runtime_error("REGER wire width truncated");
  const int width = wire[pos++];
  if (width > 64) throw std::runtime_error("invalid REGER wire width");
  residual.original_index.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i)
    residual.original_index.push_back(get_u32(wire, pos));
  const auto residual_count = count == 0 ? 0 : count - 1;
  const auto body_size = max_value_bitpack::payload_size(residual_count, width);
  if (pos + body_size != wire.size()) throw std::runtime_error("REGER wire body mismatch");
  residual.offsets = max_value_bitpack::unpack(wire.subspan(pos), residual_count, width);
  return make_reger_residual(residual, true);
}

std::vector<std::uint8_t> reger_residual_core_forward(
    std::span<const std::uint8_t> prefix) {
  const auto raw = reger_reorder_inverse(reger_delta_inverse(
      reger_zigzag_inverse(reger_for_inverse(reger_bitpack_inverse(prefix)))));
  return reger_codec::encodeInt64Column(raw_ints(raw));
}

std::vector<std::uint8_t> reger_residual_core_inverse(
    std::span<const std::uint8_t> wire) {
  const std::vector<std::uint8_t> copy(wire.begin(), wire.end());
  const auto raw = int_bytes(reger_codec::decodeInt64Column(copy));
  return reger_bitpack_forward(reger_for_forward(reger_zigzag_forward(
      reger_delta_forward(reger_reorder_forward(raw)))));
}

std::vector<std::uint8_t> buff_scaling_forward(
    std::span<const std::uint8_t> float64_le) {
  const auto values = raw_doubles(float64_le);
  if (values.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("BUFF input exceeds uint32 count");
  BuffPacket packet;
  packet.scaled = exact_decimal_scale(values, packet.scale, packet.integers);
  if (!packet.scaled) {
    packet.integers.clear();
    packet.integers.reserve(values.size());
    for (const auto value : values) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      packet.integers.push_back(to_i64(bits));
    }
  }
  return make_buff_scale(packet);
}

std::vector<std::uint8_t> buff_scaling_inverse(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_buff_scale(packet);
  std::vector<double> values;
  values.reserve(parsed.integers.size());
  if (parsed.scaled) {
    const double factor = std::pow(10.0, parsed.scale);
    for (const auto integer : parsed.integers)
      values.push_back(static_cast<double>(integer) / factor);
  } else {
    for (const auto integer : parsed.integers) {
      const auto bits = to_u64(integer);
      double value = 0;
      std::memcpy(&value, &bits, sizeof(value));
      values.push_back(value);
    }
  }
  return double_bytes(values);
}

std::vector<std::uint8_t> buff_for_forward(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse_buff_scale(packet);
  auto encoded = minimum_for::encode<std::int64_t>(parsed.integers);
  parsed.base = encoded.minimum;
  parsed.offsets = std::move(encoded.offsets);
  return make_buff_for(parsed);
}

std::vector<std::uint8_t> buff_for_inverse(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse_buff_for(packet);
  minimum_for::validate_canonical<std::int64_t>(parsed.base, parsed.offsets);
  parsed.integers = minimum_for::decode<std::int64_t>(parsed.base, parsed.offsets);
  return make_buff_scale(parsed);
}

unsigned buff_byte_plane_max_width(std::span<const std::uint64_t> offsets) {
  const auto maximum = offsets.empty()
                           ? 0
                           : *std::max_element(offsets.begin(), offsets.end());
  return static_cast<unsigned>(bit_width(maximum));
}

std::vector<std::uint8_t> buff_byte_planes_split(
    std::span<const std::uint64_t> offsets, unsigned width) {
  if (width > 64) throw std::invalid_argument("BUFF byte-plane width must be 0..64");
  const std::size_t slices = (width + 7) / 8;
  if (slices != 0 && offsets.size() > std::numeric_limits<std::size_t>::max() / slices)
    throw std::invalid_argument("BUFF byte-plane size overflow");
  for (const auto offset : offsets)
    if (width < 64 && (width == 0 ? offset != 0 : offset >> width != 0))
      throw std::invalid_argument("BUFF byte-plane offset exceeds explicit width");
  const unsigned padding = static_cast<unsigned>(slices * 8) - width;
  std::vector<std::uint8_t> planes;
  planes.reserve(slices * offsets.size());
  for (std::size_t slice = 0; slice < slices; ++slice) {
    const unsigned shift = static_cast<unsigned>((slices - 1 - slice) * 8);
    for (const auto offset : offsets) {
      const auto aligned = padding == 0 ? offset : offset << padding;
      planes.push_back(static_cast<std::uint8_t>(aligned >> shift));
    }
  }
  return planes;
}

std::vector<std::uint64_t> buff_byte_planes_join(
    std::span<const std::uint8_t> planes, std::size_t count, unsigned width) {
  if (width > 64) throw std::invalid_argument("BUFF byte-plane width must be 0..64");
  const std::size_t slices = (width + 7) / 8;
  if ((slices != 0 && count > std::numeric_limits<std::size_t>::max() / slices) ||
      planes.size() != slices * count)
    throw std::invalid_argument("BUFF byte-plane size mismatch");
  const unsigned padding = static_cast<unsigned>(slices * 8) - width;
  std::vector<std::uint64_t> offsets(count, 0);
  std::size_t body = 0;
  for (std::size_t slice = 0; slice < slices; ++slice)
    for (std::size_t i = 0; i < count; ++i)
      offsets[i] = (offsets[i] << 8) | planes[body++];
  if (padding != 0)
    for (auto& offset : offsets) {
      if ((offset & ((std::uint64_t{1} << padding) - 1)) != 0)
        throw std::invalid_argument("BUFF byte-plane padding must be zero");
      offset >>= padding;
    }
  return offsets;
}

unsigned buff_byte_plane_width(std::span<const std::uint8_t> packet) {
  return buff_byte_plane_max_width(parse_buff_for(packet).offsets);
}

std::vector<std::uint8_t> buff_byte_plane_forward(
    std::span<const std::uint8_t> packet, unsigned width) {
  auto parsed = parse_buff_for(packet);
  parsed.planes = buff_byte_planes_split(parsed.offsets, width);
  parsed.width = static_cast<std::uint8_t>(width);
  if (parsed.planes.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("BUFF byte-plane packet exceeds uint32 size");
  return make_buff_split(parsed);
}

std::vector<std::uint8_t> buff_byte_plane_inverse(
    std::span<const std::uint8_t> packet) {
  return make_buff_for(parse_buff_split(packet));
}

std::vector<std::uint8_t> buff_column_split_forward(
    std::span<const std::uint8_t> packet) {
  const auto normalized = buff_for_forward(packet);
  return buff_byte_plane_forward(normalized, buff_byte_plane_width(normalized));
}

std::vector<std::uint8_t> buff_column_split_inverse(
    std::span<const std::uint8_t> packet) {
  // Historical adapters permit a supplied base and retain modular restoration;
  // the new public FOR inverse validates the minimum/offset contract instead.
  auto parsed = parse_buff_for(buff_byte_plane_inverse(packet));
  parsed.integers = minimum_for::decode<std::int64_t>(parsed.base, parsed.offsets);
  return make_buff_scale(parsed);
}

std::vector<std::uint8_t> buff_plane_frame_forward(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse_buff_split(packet);
  std::vector<std::uint8_t> out;
  out.push_back(parsed.scaled ? 1 : 0);
  out.push_back(parsed.scale);
  put_u32(out, static_cast<std::uint32_t>(parsed.integers.size()));
  put_i64(out, parsed.base);
  out.push_back(parsed.width);
  out.push_back(1);
  out.insert(out.end(), parsed.planes.begin(), parsed.planes.end());
  return out;
}

std::vector<std::uint8_t> buff_plane_frame_inverse(
    std::span<const std::uint8_t> wire) {
  if (wire.size() < 16) throw std::runtime_error("BUFF wire truncated");
  std::size_t pos = 0;
  BuffPacket parsed;
  parsed.scaled = wire[pos++] != 0;
  parsed.scale = wire[pos++];
  const auto count = get_u32(wire, pos);
  parsed.integers.resize(count);
  parsed.base = get_i64(wire, pos);
  if (pos + 2 > wire.size()) throw std::runtime_error("BUFF wire header truncated");
  parsed.width = wire[pos++];
  const bool packed = wire[pos++] != 0;
  if (!packed || parsed.width > 64) throw std::runtime_error("BUFF wire is not split-packed");
  const std::size_t expected = ((parsed.width + 7) / 8) * count;
  if (pos + expected != wire.size()) throw std::runtime_error("BUFF wire body mismatch");
  parsed.planes.assign(wire.begin() + static_cast<std::ptrdiff_t>(pos), wire.end());
  return make_buff_split(parsed);
}

std::vector<std::uint8_t> buff_bitpack_forward(
    std::span<const std::uint8_t> packet) {
  return buff_plane_frame_forward(packet);
}

std::vector<std::uint8_t> buff_bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  return buff_plane_frame_inverse(wire);
}

}  // namespace exact_actions::learning
