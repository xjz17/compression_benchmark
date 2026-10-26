#include "fastpfor_operator_ablation.hpp"

#include "int64_column_codec.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fastpfor_operator_ablation {
namespace {

enum class Mode : std::uint8_t { Baseline = 0, NoFor = 1, NoOutliers = 2, NoBitPacking = 3 };

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("FastPFor ablation truncated u32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value = (value << 8) | in[pos++];
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("FastPFor ablation truncated u64");
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value = (value << 8) | in[pos++];
  return value;
}

std::uint64_t zigzag(std::int64_t value) {
  const auto bits = static_cast<std::uint64_t>(value);
  return (bits << 1) ^ static_cast<std::uint64_t>(value >> 63);
}

std::int64_t unzigzag(std::uint64_t value) {
  return static_cast<std::int64_t>((value >> 1) ^ (0 - (value & 1)));
}

int width_of(std::uint64_t value) { return value == 0 ? 0 : 64 - __builtin_clzll(value); }

std::vector<std::uint8_t> pack(std::span<const std::uint64_t> values, int width) {
  if (values.empty() || width == 0) return {};
  std::vector<std::uint8_t> out((values.size() * static_cast<std::size_t>(width) + 7) / 8, 0);
  std::size_t bit = 0;
  for (std::uint64_t value : values) {
    for (int shift = width - 1; shift >= 0; --shift, ++bit) {
      out[bit / 8] |= static_cast<std::uint8_t>(((value >> shift) & 1) << (7 - bit % 8));
    }
  }
  return out;
}

std::vector<std::uint64_t> unpack(std::span<const std::uint8_t> bytes,
                                  std::size_t count, int width) {
  std::vector<std::uint64_t> values(count, 0);
  if (width == 0) return values;
  const std::size_t required = (count * static_cast<std::size_t>(width) + 7) / 8;
  if (bytes.size() != required) throw std::runtime_error("FastPFor ablation packed-size mismatch");
  std::size_t bit = 0;
  for (auto& value : values) {
    for (int i = 0; i < width; ++i, ++bit) {
      value = (value << 1) | ((bytes[bit / 8] >> (7 - bit % 8)) & 1);
    }
  }
  return values;
}

}  // namespace

std::vector<std::uint8_t> encode(std::span<const std::int64_t> values,
                                 std::string_view omitted_operator) {
  if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("FastPFor ablation column too long");
  }
  Mode mode = Mode::Baseline;
  if (omitted_operator == "for") mode = Mode::NoFor;
  else if (omitted_operator == "outlier_separation") mode = Mode::NoOutliers;
  else if (omitted_operator == "bit-packing") mode = Mode::NoBitPacking;
  else if (!omitted_operator.empty() && omitted_operator != "scaling") {
    throw std::invalid_argument("unsupported FastPFor ablation operator");
  }
  std::vector<std::uint8_t> out{static_cast<std::uint8_t>(mode)};
  put_u32(out, static_cast<std::uint32_t>(values.size()));
  if (mode == Mode::Baseline) {
    const auto body = int64_column_codec::fast_pfor_encode_int64(values.data(), values.size());
    out.insert(out.end(), body.begin(), body.end());
    return out;
  }
  if (values.empty()) return out;

  if (mode == Mode::NoBitPacking) {
    const std::int64_t base = *std::min_element(values.begin(), values.end());
    put_u64(out, static_cast<std::uint64_t>(base));
    for (std::int64_t value : values) {
      put_u64(out, static_cast<std::uint64_t>(value) - static_cast<std::uint64_t>(base));
    }
    return out;
  }

  std::vector<std::uint64_t> codes;
  codes.reserve(values.size());
  if (mode == Mode::NoOutliers) {
    const std::int64_t base = *std::min_element(values.begin(), values.end());
    put_u64(out, static_cast<std::uint64_t>(base));
    int width = 0;
    for (std::int64_t value : values) {
      const auto code = static_cast<std::uint64_t>(value) - static_cast<std::uint64_t>(base);
      codes.push_back(code);
      width = std::max(width, width_of(code));
    }
    out.push_back(static_cast<std::uint8_t>(width));
    const auto body = pack(codes, width);
    out.insert(out.end(), body.begin(), body.end());
    return out;
  }

  std::vector<int> widths;
  widths.reserve(values.size());
  for (std::int64_t value : values) {
    const auto code = zigzag(value);
    codes.push_back(code);
    widths.push_back(width_of(code));
  }
  auto sorted_widths = widths;
  std::sort(sorted_widths.begin(), sorted_widths.end());
  const int regular_width = sorted_widths[(sorted_widths.size() - 1) * 9 / 10];
  out.push_back(static_cast<std::uint8_t>(regular_width));
  std::vector<std::uint8_t> bitmap((values.size() + 7) / 8, 0);
  std::vector<std::uint64_t> regular(values.size(), 0);
  std::vector<std::pair<std::uint32_t, std::uint64_t>> exceptions;
  for (std::size_t i = 0; i < codes.size(); ++i) {
    if (widths[i] > regular_width) {
      bitmap[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
      exceptions.push_back({static_cast<std::uint32_t>(i), codes[i]});
    } else {
      regular[i] = codes[i];
    }
  }
  put_u32(out, static_cast<std::uint32_t>(exceptions.size()));
  out.insert(out.end(), bitmap.begin(), bitmap.end());
  const auto body = pack(regular, regular_width);
  out.insert(out.end(), body.begin(), body.end());
  for (const auto& [index, code] : exceptions) {
    put_u32(out, index);
    put_u64(out, code);
  }
  return out;
}

std::vector<std::int64_t> decode(std::span<const std::uint8_t> payload) {
  if (payload.size() < 5) throw std::runtime_error("FastPFor ablation truncated header");
  std::size_t pos = 0;
  const auto mode = static_cast<Mode>(payload[pos++]);
  const std::size_t count = get_u32(payload, pos);
  if (mode == Mode::Baseline) {
    return int64_column_codec::fast_pfor_decode_int64(
        payload.data() + pos, payload.size() - pos, count);
  }
  if (count == 0) return {};
  if (mode == Mode::NoBitPacking) {
    const std::uint64_t base = get_u64(payload, pos);
    std::vector<std::int64_t> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      values.push_back(static_cast<std::int64_t>(base + get_u64(payload, pos)));
    }
    if (pos != payload.size()) throw std::runtime_error("FastPFor no-bitpacking trailing bytes");
    return values;
  }
  if (mode == Mode::NoOutliers) {
    const std::uint64_t base = get_u64(payload, pos);
    if (pos >= payload.size()) throw std::runtime_error("FastPFor no-outlier width missing");
    const int width = payload[pos++];
    const std::size_t body_size = (count * static_cast<std::size_t>(width) + 7) / 8;
    if (pos + body_size != payload.size()) throw std::runtime_error("FastPFor no-outlier body mismatch");
    const auto codes = unpack(payload.subspan(pos, body_size), count, width);
    std::vector<std::int64_t> values;
    values.reserve(count);
    for (auto code : codes) values.push_back(static_cast<std::int64_t>(base + code));
    return values;
  }
  if (mode != Mode::NoFor || pos >= payload.size()) {
    throw std::runtime_error("FastPFor ablation mode invalid");
  }
  const int width = payload[pos++];
  const std::size_t exception_count = get_u32(payload, pos);
  const std::size_t bitmap_size = (count + 7) / 8;
  const std::size_t body_size = (count * static_cast<std::size_t>(width) + 7) / 8;
  if (pos + bitmap_size + body_size + exception_count * 12 != payload.size()) {
    throw std::runtime_error("FastPFor no-FOR payload mismatch");
  }
  pos += bitmap_size;
  auto codes = unpack(payload.subspan(pos, body_size), count, width);
  pos += body_size;
  for (std::size_t i = 0; i < exception_count; ++i) {
    const std::size_t index = get_u32(payload, pos);
    const std::uint64_t code = get_u64(payload, pos);
    if (index >= codes.size()) throw std::runtime_error("FastPFor exception index invalid");
    codes[index] = code;
  }
  std::vector<std::int64_t> values;
  values.reserve(count);
  for (auto code : codes) values.push_back(unzigzag(code));
  return values;
}

}  // namespace fastpfor_operator_ablation
