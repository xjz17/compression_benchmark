#include "ts2diff_operator_ablation.hpp"

#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ts2diff_operator_ablation {
namespace {

enum class Mode : std::uint8_t { Baseline = 0, NoDelta = 1, NoFor = 2, NoBitPacking = 3 };

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("TS_2DIFF ablation truncated u32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value = (value << 8) | in[pos++];
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("TS_2DIFF ablation truncated u64");
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

int width_of(std::uint64_t value) {
  return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

std::vector<std::uint8_t> pack(std::span<const std::uint64_t> values, int width) {
  if (width == 0 || values.empty()) return {};
  if (width == 64) {
    std::vector<std::uint8_t> out;
    out.reserve(values.size() * 8);
    for (std::uint64_t value : values) put_u64(out, value);
    return out;
  }
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
  if (bytes.size() != required) throw std::runtime_error("TS_2DIFF ablation packed size mismatch");
  std::size_t bit = 0;
  for (std::uint64_t& value : values) {
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
    throw std::runtime_error("TS_2DIFF ablation column too long");
  }
  Mode mode = Mode::Baseline;
  if (omitted_operator == "delta") mode = Mode::NoDelta;
  else if (omitted_operator == "for") mode = Mode::NoFor;
  else if (omitted_operator == "bit-packing") mode = Mode::NoBitPacking;
  else if (!omitted_operator.empty() && omitted_operator != "scaling") {
    throw std::invalid_argument("unsupported TS_2DIFF operator ablation");
  }

  std::vector<std::uint8_t> out{static_cast<std::uint8_t>(mode)};
  put_u32(out, static_cast<std::uint32_t>(values.size()));
  if (mode == Mode::Baseline) {
    const auto body = tsfile::p02::encode_ts_2diff_int64(values);
    out.insert(out.end(), body.begin(), body.end());
    return out;
  }
  if (values.empty()) return out;

  if (mode == Mode::NoDelta) {
    const std::int64_t base = *std::min_element(values.begin(), values.end());
    std::vector<std::uint64_t> residuals;
    residuals.reserve(values.size());
    int width = 0;
    for (std::int64_t value : values) {
      const std::uint64_t residual = static_cast<std::uint64_t>(value) -
                                     static_cast<std::uint64_t>(base);
      residuals.push_back(residual);
      width = std::max(width, width_of(residual));
    }
    put_u64(out, static_cast<std::uint64_t>(base));
    out.push_back(static_cast<std::uint8_t>(width));
    const auto body = pack(residuals, width);
    out.insert(out.end(), body.begin(), body.end());
    return out;
  }

  std::vector<std::int64_t> deltas;
  deltas.reserve(values.size() - 1);
  std::int64_t previous = values.front();
  for (std::size_t i = 1; i < values.size(); ++i) {
    const std::uint64_t bits = static_cast<std::uint64_t>(values[i]) -
                               static_cast<std::uint64_t>(previous);
    deltas.push_back(static_cast<std::int64_t>(bits));
    previous = values[i];
  }
  put_u64(out, static_cast<std::uint64_t>(values.front()));
  if (mode == Mode::NoFor) {
    std::vector<std::uint64_t> encoded;
    encoded.reserve(deltas.size());
    int width = 0;
    for (std::int64_t delta : deltas) {
      const std::uint64_t value = zigzag(delta);
      encoded.push_back(value);
      width = std::max(width, width_of(value));
    }
    out.push_back(static_cast<std::uint8_t>(width));
    const auto body = pack(encoded, width);
    out.insert(out.end(), body.begin(), body.end());
    return out;
  }

  const std::int64_t base = deltas.empty()
                                ? 0
                                : *std::min_element(deltas.begin(), deltas.end());
  put_u64(out, static_cast<std::uint64_t>(base));
  for (std::int64_t delta : deltas) {
    put_u64(out, static_cast<std::uint64_t>(delta) - static_cast<std::uint64_t>(base));
  }
  return out;
}

std::vector<std::int64_t> decode(std::span<const std::uint8_t> payload) {
  if (payload.size() < 5) throw std::runtime_error("TS_2DIFF ablation truncated header");
  std::size_t pos = 0;
  const auto mode = static_cast<Mode>(payload[pos++]);
  const std::size_t count = get_u32(payload, pos);
  if (mode == Mode::Baseline) {
    std::vector<std::int64_t> values;
    if (!tsfile::p02::decode_ts_2diff_int64(payload.subspan(pos), values) ||
        values.size() != count) {
      throw std::runtime_error("TS_2DIFF baseline decode failed");
    }
    return values;
  }
  if (count == 0) {
    if (pos != payload.size()) throw std::runtime_error("TS_2DIFF empty payload has trailing bytes");
    return {};
  }

  if (mode == Mode::NoDelta) {
    const std::uint64_t base = get_u64(payload, pos);
    if (pos >= payload.size()) throw std::runtime_error("TS_2DIFF no-delta width missing");
    const int width = payload[pos++];
    const auto residuals = unpack(payload.subspan(pos), count, width);
    std::vector<std::int64_t> values;
    values.reserve(count);
    for (std::uint64_t residual : residuals) {
      values.push_back(static_cast<std::int64_t>(base + residual));
    }
    return values;
  }

  std::vector<std::int64_t> values(count);
  values[0] = static_cast<std::int64_t>(get_u64(payload, pos));
  if (mode == Mode::NoFor) {
    if (pos >= payload.size()) throw std::runtime_error("TS_2DIFF no-FOR width missing");
    const int width = payload[pos++];
    const auto encoded = unpack(payload.subspan(pos), count - 1, width);
    std::uint64_t previous = static_cast<std::uint64_t>(values[0]);
    for (std::size_t i = 1; i < count; ++i) {
      previous += static_cast<std::uint64_t>(unzigzag(encoded[i - 1]));
      values[i] = static_cast<std::int64_t>(previous);
    }
    return values;
  }
  if (mode != Mode::NoBitPacking) throw std::runtime_error("TS_2DIFF ablation mode invalid");
  const std::uint64_t base = get_u64(payload, pos);
  if (payload.size() - pos != (count - 1) * 8) {
    throw std::runtime_error("TS_2DIFF no-bitpacking size mismatch");
  }
  std::uint64_t previous = static_cast<std::uint64_t>(values[0]);
  for (std::size_t i = 1; i < count; ++i) {
    previous += base + get_u64(payload, pos);
    values[i] = static_cast<std::int64_t>(previous);
  }
  return values;
}

}  // namespace ts2diff_operator_ablation
