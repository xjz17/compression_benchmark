#include "dbp_exact_stages.hpp"
#include "unsigned_leb128.hpp"
#include "zigzag.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::dbp {
namespace {

constexpr std::array<std::uint8_t, 4> kScalingMagic{'D', 'B', 'S', '1'};
constexpr std::array<std::uint8_t, 4> kDeltaMagic{'D', 'B', 'D', '1'};
constexpr std::array<std::uint8_t, 4> kZigzagMagic{'D', 'B', 'Z', '1'};

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t read_u32(std::span<const std::uint8_t> data, std::size_t& pos) {
  if (data.size() - pos < 4) throw std::runtime_error("truncated DBP packet");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(data[pos++]) << shift;
  return value;
}

std::uint64_t read_u64(std::span<const std::uint8_t> data, std::size_t& pos) {
  if (data.size() - pos < 8) throw std::runtime_error("truncated DBP packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(data[pos++]) << shift;
  return value;
}

void append_varint(std::vector<std::uint8_t>& out, std::uint64_t value) {
  compression_operators::unsigned_leb128::append(out, value);
}

std::uint64_t read_varint(std::span<const std::uint8_t> data, std::size_t& pos) {
  return compression_operators::unsigned_leb128::read(data, pos);
}

std::uint64_t zigzag(std::int64_t value) {
  return compression_operators::zigzag::encode64(value);
}

std::int64_t unzigzag(std::uint64_t value) {
  return compression_operators::zigzag::decode64(value);
}

struct Packet {
  int scale;
  std::vector<std::int64_t> values;
};

enum class ScalingRejection {
  None,
  InvalidMaxPoint,
  Nonfinite,
  Overflow,
  NotBitwiseReversible,
};

struct ScalingCheck {
  ScalingRejection rejection = ScalingRejection::None;
  int scale = 0;
};

const std::array<double, maximum_scaling_point + 1>& decimal_powers() {
  static const auto powers = [] {
    std::array<double, maximum_scaling_point + 1> result{};
    for (int scale = 0; scale <= maximum_scaling_point; ++scale)
      result[scale] = std::pow(10.0, scale);
    return result;
  }();
  return powers;
}

bool scaling_product_fits(double value, int scale) noexcept {
  const long double product = static_cast<long double>(value) *
      decimal_powers()[scale];
  return product >=
          static_cast<long double>(std::numeric_limits<std::int64_t>::min()) &&
      product <=
          static_cast<long double>(std::numeric_limits<std::int64_t>::max());
}

bool scale_value_exact(double value, int scale,
                       std::int64_t* scaled_value = nullptr) noexcept {
  const double power = decimal_powers()[scale];
  const long double product = static_cast<long double>(value) * power;
  const auto scaled = static_cast<std::int64_t>(std::llround(product));
  const double recovered = static_cast<double>(scaled) / power;
  if (std::bit_cast<std::uint64_t>(recovered) !=
      std::bit_cast<std::uint64_t>(value))
    return false;
  if (scaled_value) *scaled_value = scaled;
  return true;
}

// all_scaling_feasibility() has already proved that every product at every
// scale up to safe_scale is finite and inside int64_t.  On platforms whose
// long double can represent every int64_t exactly, implement llround's
// halfway-away-from-zero rule inline and avoid a libm call per value/scale.
// Keep the existing helper on binary64-long-double platforms: there
// INT64_MAX is not distinguishable from +2^63, so a direct cast is unsafe.
bool scale_value_exact_after_fit_check(double value, int scale) noexcept {
  if constexpr (std::numeric_limits<long double>::digits < 64) {
    return scale_value_exact(value, scale);
  } else {
    const double power = decimal_powers()[scale];
    const long double product = static_cast<long double>(value) * power;
    auto scaled = static_cast<std::int64_t>(product);
    const long double fraction = product - static_cast<long double>(scaled);
    if (fraction >= 0.5L &&
        scaled != std::numeric_limits<std::int64_t>::max())
      ++scaled;
    else if (fraction <= -0.5L &&
             scaled != std::numeric_limits<std::int64_t>::min())
      --scaled;
    const double recovered = static_cast<double>(scaled) / power;
    return std::bit_cast<std::uint64_t>(recovered) ==
        std::bit_cast<std::uint64_t>(value);
  }
}

std::uint64_t scaling_cache_hash(std::uint64_t value) noexcept {
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

std::uint32_t exact_scale_subset(double value,
                                 std::uint32_t scales,
                                 int safe_scale) noexcept {
  if (!scales) return 0;
  std::uint32_t exact = scales;
  for (int scale = 0; scale <= safe_scale; ++scale) {
    const auto bit = std::uint32_t{1} << scale;
    if ((scales & bit) &&
        !scale_value_exact_after_fit_check(value, scale))
      exact &= ~bit;
  }
  return exact;
}

template <class Get>
ScalingCheck check_scaling_exact(std::size_t count, Get&& get, int max_point,
                                 std::span<std::int64_t> scaled) noexcept {
  if (max_point < 0 || max_point > maximum_scaling_point)
    return {ScalingRejection::InvalidMaxPoint, 0};
  int safe_scale = max_point;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = get(i);
    if (!std::isfinite(value))
      return {ScalingRejection::Nonfinite, 0};
    while (safe_scale >= 0 && !scaling_product_fits(value, safe_scale))
      --safe_scale;
    if (safe_scale < 0) return {ScalingRejection::Overflow, 0};
  }
  const bool write_scaled = scaled.size() == count;
  for (std::size_t i = 0; i < count; ++i) {
    std::int64_t value = 0;
    if (!scale_value_exact(get(i), safe_scale,
                           write_scaled ? &value : nullptr))
      return {ScalingRejection::NotBitwiseReversible, safe_scale};
    if (write_scaled) scaled[i] = value;
  }
  return {ScalingRejection::None, safe_scale};
}

template <class Get>
ScalingFeasibility all_scaling_feasibility(std::size_t count,
                                            Get&& get) noexcept {
  ScalingFeasibility result;
  struct ProbeEntry {
    std::uint64_t key = 0;
    bool occupied = false;
  };
  std::array<ProbeEntry, 64> repeat_probe{};
  std::size_t repeat_probe_hits = 0;
  // The fit pass already visits every value.  Inspect only its first 64 loads
  // to choose the exactness loop, without adding a third value pass.  The
  // uncached loop below stays identical for small or high-cardinality input.
  const bool probe_repetition = count >= 1024;
  int safe_scale = maximum_scaling_point;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = get(i);
    ++result.input_value_visits;
    if (!std::isfinite(value)) return result;
    if (probe_repetition && i < repeat_probe.size()) {
      const auto key = std::bit_cast<std::uint64_t>(value);
      auto& entry =
          repeat_probe[scaling_cache_hash(key) & (repeat_probe.size() - 1)];
      if (entry.occupied && entry.key == key)
        ++repeat_probe_hits;
      else
        entry = {key, true};
    }
    while (safe_scale >= 0 && !scaling_product_fits(value, safe_scale))
      --safe_scale;
    if (safe_scale < 0) return result;
  }

  std::uint32_t exact_scales =
      (std::uint32_t{1} << (safe_scale + 1)) - 1;
  if (repeat_probe_hits < 4) {
    for (std::size_t i = 0; i < count && exact_scales; ++i) {
      const double value = get(i);
      ++result.input_value_visits;
      for (int scale = 0; scale <= safe_scale; ++scale) {
        const auto bit = std::uint32_t{1} << scale;
        if ((exact_scales & bit) &&
            !scale_value_exact_after_fit_check(value, scale))
          exact_scales &= ~bit;
      }
    }
  } else {
    // Decimal and categorical time-series columns frequently contain a small
    // repeating value alphabet.  Cache the exact-scale result by the complete
    // binary64 bit pattern, including the sign bit of zero.  Direct mapping
    // makes collisions a performance-only event: a colliding entry is simply
    // replaced and its value is evaluated normally.
    struct CacheEntry {
      std::uint64_t key = 0;
      std::uint32_t tested_scales = 0;
      std::uint32_t exact_scales = 0;
      bool occupied = false;
    };
    std::array<CacheEntry, 256> cache{};
    for (std::size_t i = 0; i < count && exact_scales; ++i) {
      const double value = get(i);
      ++result.input_value_visits;
      const auto key = std::bit_cast<std::uint64_t>(value);
      auto& entry = cache[scaling_cache_hash(key) & (cache.size() - 1)];
      if (!entry.occupied || entry.key != key) {
        entry = {key, exact_scales,
                 exact_scale_subset(value, exact_scales, safe_scale), true};
      } else {
        const auto missing = exact_scales & ~entry.tested_scales;
        entry.exact_scales |= exact_scale_subset(value, missing, safe_scale);
        entry.tested_scales |= missing;
      }
      exact_scales &= entry.exact_scales;
    }
  }
  for (int max_point = 0; max_point <= maximum_scaling_point; ++max_point) {
    const int selected_scale = std::min(max_point, safe_scale);
    if (exact_scales & (std::uint32_t{1} << selected_scale))
      result.legal_max_points |= std::uint32_t{1} << max_point;
  }
  return result;
}

double little_double(std::span<const std::uint8_t> bytes,
                     std::size_t index) noexcept {
  std::uint64_t bits = 0;
  const auto offset = index * 8;
  for (unsigned byte = 0; byte < 8; ++byte)
    bits |= std::uint64_t(bytes[offset + byte]) << (byte * 8);
  return std::bit_cast<double>(bits);
}

std::vector<std::uint8_t> serialize(const std::array<std::uint8_t, 4>& magic,
                                    int scale,
                                    std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  append_u32(out, static_cast<std::uint32_t>(scale));
  append_u64(out, values.size());
  for (const auto value : values) append_u64(out, static_cast<std::uint64_t>(value));
  return out;
}

Packet parse(std::span<const std::uint8_t> packet,
             const std::array<std::uint8_t, 4>& magic) {
  if (packet.size() < 16 ||
      !std::equal(magic.begin(), magic.end(), packet.begin()))
    throw std::runtime_error("wrong DBP stage packet type");
  std::size_t pos = 4;
  const int scale = static_cast<int>(static_cast<std::int32_t>(read_u32(packet, pos)));
  const auto count = read_u64(packet, pos);
  if (count > (packet.size() - pos) / 8 || count * 8 != packet.size() - pos)
    throw std::runtime_error("invalid DBP stage packet count");
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i)
    values.push_back(static_cast<std::int64_t>(read_u64(packet, pos)));
  return {scale, std::move(values)};
}

std::vector<std::uint8_t> delta_packet(int scale,
                                       std::span<const std::int64_t> values) {
  std::vector<std::int64_t> deltas(values.begin(), values.end());
  for (std::size_t i = values.size(); i-- > 1;)
    deltas[i] = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(values[i]) -
        static_cast<std::uint64_t>(values[i - 1]));
  return serialize(kDeltaMagic, scale, deltas);
}

Packet inverse_delta(std::span<const std::uint8_t> packet) {
  auto parsed = parse(packet, kDeltaMagic);
  for (std::size_t i = 1; i < parsed.values.size(); ++i)
    parsed.values[i] = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(parsed.values[i - 1]) +
        static_cast<std::uint64_t>(parsed.values[i]));
  return parsed;
}

}  // namespace

bool can_scale_bitwise_exact(std::span<const double> values,
                             int max_point) noexcept {
  return check_scaling_exact(
      values.size(), [&](std::size_t i) { return values[i]; }, max_point,
      {}).rejection == ScalingRejection::None;
}

ScalingFeasibility scaling_feasibility(
    std::span<const double> values) noexcept {
  return all_scaling_feasibility(
      values.size(), [&](std::size_t i) { return values[i]; });
}

ScalingFeasibility scaling_feasibility_le64(
    std::span<const std::uint8_t> bytes) noexcept {
  if (bytes.size() % 8 != 0) return {};
  return all_scaling_feasibility(bytes.size() / 8, [&](std::size_t i) {
    return little_double(bytes, i);
  });
}

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int max_point) {
  std::vector<std::int64_t> scaled(values.size());
  const auto check = check_scaling_exact(
      values.size(), [&](std::size_t i) { return values[i]; }, max_point,
      scaled);
  switch (check.rejection) {
    case ScalingRejection::None:
      break;
    case ScalingRejection::InvalidMaxPoint:
      throw std::invalid_argument("DBP max_point must be in [0,18]");
    case ScalingRejection::Nonfinite:
      throw std::invalid_argument("DBP scaling requires finite values");
    case ScalingRejection::Overflow:
      throw std::overflow_error("DBP decimal scaling exceeds int64");
    case ScalingRejection::NotBitwiseReversible:
      throw std::invalid_argument(
          "DBP decimal scaling is not bitwise reversible");
  }
  return serialize(kScalingMagic, check.scale, scaled);
}

std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet, kScalingMagic);
  if (parsed.scale < 0 || parsed.scale > 18)
    throw std::runtime_error("invalid DBP scale");
  const double power = std::pow(10.0, parsed.scale);
  std::vector<double> values;
  values.reserve(parsed.values.size());
  for (const auto value : parsed.values)
    values.push_back(static_cast<double>(value) / power);
  return values;
}

std::pair<int, std::vector<std::int64_t>> scaling_values(
    std::span<const std::uint8_t> packet) {
  auto parsed = parse(packet, kScalingMagic);
  return {parsed.scale, std::move(parsed.values)};
}

std::vector<std::uint8_t> make_scaling_packet(
    int scale, std::span<const std::int64_t> values) {
  if (scale < 0 || scale > 18) throw std::invalid_argument("invalid DBP scale");
  return serialize(kScalingMagic, scale, values);
}

std::vector<std::uint8_t> delta_forward_int64(
    std::span<const std::int64_t> values) {
  return delta_packet(-1, values);
}

std::vector<std::uint8_t> delta_forward_scaled(
    std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet, kScalingMagic);
  return delta_packet(parsed.scale, parsed.values);
}

std::vector<std::int64_t> delta_inverse_int64(
    std::span<const std::uint8_t> packet) {
  auto parsed = inverse_delta(packet);
  if (parsed.scale != -1) throw std::runtime_error("DBP delta packet is not integer");
  return parsed.values;
}

std::vector<std::uint8_t> delta_inverse_scaled(
    std::span<const std::uint8_t> packet) {
  auto parsed = inverse_delta(packet);
  if (parsed.scale < 0) throw std::runtime_error("DBP delta packet is not scaled");
  return serialize(kScalingMagic, parsed.scale, parsed.values);
}

std::vector<std::uint8_t> zigzag_forward(std::span<const std::uint8_t> packet) {
  auto parsed = parse(packet, kDeltaMagic);
  for (std::size_t i = 1; i < parsed.values.size(); ++i)
    parsed.values[i] = static_cast<std::int64_t>(zigzag(parsed.values[i]));
  return serialize(kZigzagMagic, parsed.scale, parsed.values);
}

std::vector<std::uint8_t> zigzag_inverse(std::span<const std::uint8_t> packet) {
  auto parsed = parse(packet, kZigzagMagic);
  for (std::size_t i = 1; i < parsed.values.size(); ++i)
    parsed.values[i] = unzigzag(static_cast<std::uint64_t>(parsed.values[i]));
  return serialize(kDeltaMagic, parsed.scale, parsed.values);
}

std::vector<std::uint8_t> varint_forward(std::span<const std::uint8_t> packet) {
  const auto parsed = parse(packet, kZigzagMagic);
  std::vector<std::uint8_t> out;
  if (parsed.scale < 0) {
    out.push_back('I');
  } else {
    out.push_back('F');
    append_u32(out, static_cast<std::uint32_t>(parsed.scale));
  }
  if (!parsed.values.empty()) {
    append_u64(out, static_cast<std::uint64_t>(parsed.values.front()));
    for (std::size_t i = 1; i < parsed.values.size(); ++i)
      append_varint(out, static_cast<std::uint64_t>(parsed.values[i]));
  }
  return out;
}

std::vector<std::uint8_t> varint_inverse(std::span<const std::uint8_t> wire) {
  if (wire.empty() || (wire[0] != 'I' && wire[0] != 'F'))
    throw std::runtime_error("invalid DBP public wire tag");
  std::size_t pos = 1;
  const int scale = wire[0] == 'I' ? -1 : static_cast<int>(read_u32(wire, pos));
  std::vector<std::int64_t> values;
  if (pos < wire.size()) {
    values.push_back(static_cast<std::int64_t>(read_u64(wire, pos)));
    while (pos < wire.size())
      values.push_back(static_cast<std::int64_t>(read_varint(wire, pos)));
  }
  return serialize(kZigzagMagic, scale, values);
}

std::vector<std::uint8_t> compose_int64(std::span<const std::int64_t> values) {
  return varint_forward(zigzag_forward(delta_forward_int64(values)));
}

std::vector<std::int64_t> decompose_int64(std::span<const std::uint8_t> wire) {
  return delta_inverse_int64(zigzag_inverse(varint_inverse(wire)));
}

std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int max_point) {
  return varint_forward(zigzag_forward(delta_forward_scaled(
      scaling_forward(values, max_point))));
}

std::vector<double> decompose_double(std::span<const std::uint8_t> wire) {
  return scaling_inverse(delta_inverse_scaled(zigzag_inverse(varint_inverse(wire))));
}

}  // namespace exact_actions::dbp
