#include "p04_exact_stages.hpp"
#include "zigzag.hpp"
#include "max_value_bitpack.hpp"

#include "tsfile_encoding/tsfile_decode_p04.hpp"
#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_decode_p03.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace exact_actions::p04 {
namespace {

constexpr std::array<std::uint8_t, 4> kRlbeDeltaMagic{'R', 'L', 'D', '1'};
constexpr std::array<std::uint8_t, 4> kSprintzPredictorIntMagic{'S', 'P', 'I', '1'};
constexpr std::array<std::uint8_t, 4> kSprintzPredictorFloatMagic{'S', 'P', 'F', '1'};
constexpr std::array<std::uint8_t, 4> kSprintzZigzagIntMagic{'S', 'Z', 'I', '1'};
constexpr std::array<std::uint8_t, 4> kSprintzZigzagFloatMagic{'S', 'Z', 'F', '1'};
constexpr std::size_t kRlbeBlock = 10000;
constexpr std::size_t kSprintzBlockValues = 9;

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

std::uint64_t read_u64_le(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  if (bytes.size() - pos < 8) throw std::runtime_error("truncated p04 stage packet");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(bytes[pos++]) << shift;
  }
  return value;
}

void append_i64_be(std::vector<std::uint8_t>& out, std::int64_t value) {
  const auto bits = static_cast<std::uint64_t>(value);
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xffu));
  }
}

std::int64_t read_i64_be(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  if (bytes.size() - pos < 8) throw std::runtime_error("truncated Sprintz public wire");
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value = (value << 8) | bytes[pos++];
  return static_cast<std::int64_t>(value);
}

std::vector<std::uint8_t> serialize(
    const std::array<std::uint8_t, 4>& magic,
    std::span<const std::int64_t> values) {
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  append_u64_le(out, values.size());
  for (const auto value : values) append_u64_le(out, static_cast<std::uint64_t>(value));
  return out;
}

std::vector<std::int64_t> parse(
    std::span<const std::uint8_t> packet,
    const std::array<std::uint8_t, 4>& magic) {
  if (packet.size() < 12 || !std::equal(magic.begin(), magic.end(), packet.begin())) {
    throw std::runtime_error("wrong p04 stage packet type");
  }
  std::size_t pos = 4;
  const auto count = read_u64_le(packet, pos);
  if (count > (packet.size() - pos) / 8 || count * 8 != packet.size() - pos) {
    throw std::runtime_error("invalid p04 stage packet count");
  }
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    values.push_back(static_cast<std::int64_t>(read_u64_le(packet, pos)));
  }
  return values;
}

enum class SprintzKind { integer, floating };

std::pair<std::vector<std::int64_t>, SprintzKind> parse_sprintz(
    std::span<const std::uint8_t> packet,
    const std::array<std::uint8_t, 4>& integer_magic,
    const std::array<std::uint8_t, 4>& floating_magic) {
  if (packet.size() >= 4 &&
      std::equal(integer_magic.begin(), integer_magic.end(), packet.begin())) {
    return {parse(packet, integer_magic), SprintzKind::integer};
  }
  if (packet.size() >= 4 &&
      std::equal(floating_magic.begin(), floating_magic.end(), packet.begin())) {
    return {parse(packet, floating_magic), SprintzKind::floating};
  }
  throw std::runtime_error("wrong typed Sprintz stage packet");
}

const std::array<std::uint8_t, 4>& predictor_magic(SprintzKind kind) {
  return kind == SprintzKind::integer ? kSprintzPredictorIntMagic
                                      : kSprintzPredictorFloatMagic;
}

const std::array<std::uint8_t, 4>& zigzag_magic(SprintzKind kind) {
  return kind == SprintzKind::integer ? kSprintzZigzagIntMagic
                                      : kSprintzZigzagFloatMagic;
}

struct LongFire {
  std::int64_t accumulator = 0;
  std::int64_t delta = 0;

  std::int64_t predict(std::int64_t value) const {
    const std::int64_t alpha = accumulator >> 2;
    const std::uint64_t product = static_cast<std::uint64_t>(alpha) *
                                  static_cast<std::uint64_t>(delta);
    const std::int64_t diff = static_cast<std::int64_t>(product) >> 16;
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(value) +
                                     static_cast<std::uint64_t>(diff));
  }

  void train(std::int64_t previous, std::int64_t value, std::int64_t error) {
    const std::int64_t gradient =
        error > 0
            ? static_cast<std::int64_t>(std::uint64_t{0} - static_cast<std::uint64_t>(delta))
            : delta;
    accumulator = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(accumulator) - static_cast<std::uint64_t>(gradient));
    delta = static_cast<std::int64_t>(static_cast<std::uint64_t>(value) -
                                      static_cast<std::uint64_t>(previous));
  }
};

std::int64_t zigzag(std::int64_t value) {
  return std::bit_cast<std::int64_t>(compression_operators::zigzag::encode64(value));
}

std::int64_t unzigzag(std::int64_t value) {
  return compression_operators::zigzag::decode64(std::bit_cast<std::uint64_t>(value));
}

int bit_width(std::int64_t value) {
  const auto bits = static_cast<std::uint64_t>(value);
  return bits == 0 ? 0 : 64 - __builtin_clzll(bits);
}

void pack8(int width, const std::int64_t* values, std::vector<std::uint8_t>& out) {
  std::array<std::uint64_t, 8> codes;
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = std::bit_cast<std::uint64_t>(values[i]);
  const auto packed = max_value_bitpack::pack_fixed(codes, width);
  out.insert(out.end(), packed.payload.begin(), packed.payload.end());
}

void unpack8(int width, std::span<const std::uint8_t> payload,
             std::int64_t* values) {
  const auto codes = max_value_bitpack::unpack(payload, 8, width);
  for (std::size_t i = 0; i < codes.size(); ++i) values[i] = std::bit_cast<std::int64_t>(codes[i]);
}

}  // namespace

std::vector<std::uint8_t> rlbe_delta_forward(std::span<const std::int64_t> values) {
  std::vector<std::int64_t> deltas(values.size());
  for (std::size_t block = 0; block < values.size(); block += kRlbeBlock) {
    const std::size_t end = std::min(values.size(), block + kRlbeBlock);
    if (block == end) continue;
    deltas[block] = values[block];
    for (std::size_t i = block + 1; i < end; ++i) {
      deltas[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(values[i]) -
                                            static_cast<std::uint64_t>(values[i - 1]));
    }
  }
  return serialize(kRlbeDeltaMagic, deltas);
}

std::vector<std::int64_t> rlbe_delta_inverse(std::span<const std::uint8_t> packet) {
  auto values = parse(packet, kRlbeDeltaMagic);
  for (std::size_t block = 0; block < values.size(); block += kRlbeBlock) {
    const std::size_t end = std::min(values.size(), block + kRlbeBlock);
    for (std::size_t i = block + 1; i < end; ++i) {
      values[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(values[i - 1]) +
                                            static_cast<std::uint64_t>(values[i]));
    }
  }
  return values;
}

std::vector<std::uint8_t> rlbe_run_length_forward(std::span<const std::uint8_t> packet) {
  const auto deltas = parse(packet, kRlbeDeltaMagic);
  return tsfile::p04::encode_rlbe_int64_ablation(deltas, "delta");
}

std::vector<std::uint8_t> rlbe_run_length_inverse(std::span<const std::uint8_t> wire) {
  std::vector<std::int64_t> deltas;
  if (!tsfile::p04::decode_rlbe_int64_ablation(wire, "delta", deltas)) {
    throw std::runtime_error("invalid RLBE public wire");
  }
  return serialize(kRlbeDeltaMagic, deltas);
}

std::vector<std::uint8_t> rlbe_compose(std::span<const std::int64_t> values) {
  return rlbe_run_length_forward(rlbe_delta_forward(values));
}

std::vector<std::int64_t> rlbe_decompose(std::span<const std::uint8_t> wire) {
  return rlbe_delta_inverse(rlbe_run_length_inverse(wire));
}

std::vector<std::uint8_t> sprintz_predictor_forward(
    std::span<const std::int64_t> values) {
  std::vector<std::int64_t> residuals(values.begin(), values.end());
  const std::size_t full_blocks = values.size() / kSprintzBlockValues;
  for (std::size_t block = 0; block < full_blocks; ++block) {
    const std::size_t base = block * kSprintzBlockValues;
    LongFire fire;
    std::int64_t previous = values[base];
    for (std::size_t offset = 1; offset < kSprintzBlockValues; ++offset) {
      const auto current = values[base + offset];
      const auto predicted = fire.predict(previous);
      const auto error = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(current) - static_cast<std::uint64_t>(predicted));
      residuals[base + offset] = error;
      fire.train(previous, current, error);
      previous = current;
    }
  }
  return serialize(kSprintzPredictorIntMagic, residuals);
}

std::vector<std::uint8_t> sprintz_predictor_forward_double_bits(
    std::span<const std::int64_t> values) {
  auto packet = sprintz_predictor_forward(values);
  std::copy(kSprintzPredictorFloatMagic.begin(), kSprintzPredictorFloatMagic.end(),
            packet.begin());
  return packet;
}

std::vector<std::int64_t> sprintz_predictor_inverse(
    std::span<const std::uint8_t> packet) {
  auto [values, kind] = parse_sprintz(
      packet, kSprintzPredictorIntMagic, kSprintzPredictorFloatMagic);
  const std::size_t full_blocks = values.size() / kSprintzBlockValues;
  for (std::size_t block = 0; block < full_blocks; ++block) {
    const std::size_t base = block * kSprintzBlockValues;
    LongFire fire;
    for (std::size_t offset = 1; offset < kSprintzBlockValues; ++offset) {
      const auto previous = values[base + offset - 1];
      const auto error = values[base + offset];
      const auto predicted = fire.predict(previous);
      const auto current = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(predicted) + static_cast<std::uint64_t>(error));
      values[base + offset] = current;
      fire.train(previous, current, error);
    }
  }
  return values;
}

std::vector<std::uint8_t> sprintz_zigzag_forward(
    std::span<const std::uint8_t> packet) {
  auto [values, kind] = parse_sprintz(
      packet, kSprintzPredictorIntMagic, kSprintzPredictorFloatMagic);
  const std::size_t full_blocks = values.size() / kSprintzBlockValues;
  for (std::size_t block = 0; block < full_blocks; ++block) {
    const std::size_t base = block * kSprintzBlockValues;
    for (std::size_t offset = 1; offset < kSprintzBlockValues; ++offset) {
      values[base + offset] = zigzag(values[base + offset]);
    }
  }
  return serialize(zigzag_magic(kind), values);
}

std::vector<std::uint8_t> sprintz_zigzag_inverse(
    std::span<const std::uint8_t> packet) {
  auto [values, kind] = parse_sprintz(
      packet, kSprintzZigzagIntMagic, kSprintzZigzagFloatMagic);
  const std::size_t full_blocks = values.size() / kSprintzBlockValues;
  for (std::size_t block = 0; block < full_blocks; ++block) {
    const std::size_t base = block * kSprintzBlockValues;
    for (std::size_t offset = 1; offset < kSprintzBlockValues; ++offset) {
      values[base + offset] = unzigzag(values[base + offset]);
    }
  }
  return serialize(predictor_magic(kind), values);
}

static std::vector<std::uint8_t> sprintz_bitpack_forward_impl(
    std::span<const std::uint8_t> packet, bool current_rle_tail) {
  const auto parsed = parse_sprintz(
      packet, kSprintzZigzagIntMagic, kSprintzZigzagFloatMagic);
  const auto& values = parsed.first;
  const auto kind = parsed.second;
  std::vector<std::uint8_t> out;
  const std::size_t full_blocks = values.size() / kSprintzBlockValues;
  for (std::size_t block = 0; block < full_blocks; ++block) {
    const std::size_t base = block * kSprintzBlockValues;
    int width = 1;
    for (std::size_t offset = 1; offset < kSprintzBlockValues; ++offset) {
      width = std::max(width, bit_width(values[base + offset]));
    }
    out.push_back(static_cast<std::uint8_t>(width));
    append_i64_be(out, values[base]);
    const std::size_t payload_begin = out.size();
    pack8(width, values.data() + base + 1, out);
    if (out.size() - payload_begin != static_cast<std::size_t>(width)) {
      throw std::runtime_error("Sprintz bitpack emitted wrong byte count");
    }
  }
  const std::size_t tail = values.size() - full_blocks * kSprintzBlockValues;
  if (tail > 0) {
    out.push_back(static_cast<std::uint8_t>(0x80u | tail));
    const auto tail_values = std::span<const std::int64_t>(values).last(tail);
    if (kind == SprintzKind::integer) {
      const auto encoded = current_rle_tail
          ? tsfile::p02::encode_rle_int64(tail_values)
          : tsfile::p02::encode_rle_int64_legacy(tail_values);
      out.insert(out.end(), encoded.begin(), encoded.end());
    } else {
      std::vector<double> doubles(tail);
      std::memcpy(doubles.data(), tail_values.data(), tail * sizeof(double));
      const auto encoded = tsfile::p03::encode_gorilla_double(doubles);
      out.insert(out.end(), encoded.begin(), encoded.end());
    }
  }
  return out;
}

std::vector<std::uint8_t> sprintz_bitpack_forward(
    std::span<const std::uint8_t> packet) {
  return sprintz_bitpack_forward_impl(packet, false);
}

std::vector<std::uint8_t> sprintz_bitpack_forward_current(
    std::span<const std::uint8_t> packet) {
  return sprintz_bitpack_forward_impl(packet, true);
}

namespace {

std::vector<std::uint8_t> sprintz_bitpack_inverse_typed(
    std::span<const std::uint8_t> wire, SprintzKind kind) {
  std::vector<std::int64_t> values;
  std::size_t pos = 0;
  while (pos < wire.size()) {
    const std::uint8_t marker = wire[pos++];
    if ((marker & 0x80u) != 0) {
      const std::size_t tail = marker & 0x7fu;
      if (tail == 0 || tail > 8) {
        throw std::runtime_error("invalid Sprintz tail");
      }
      const auto encoded_tail = wire.subspan(pos);
      if (kind == SprintzKind::integer) {
        std::vector<std::int64_t> decoded;
        if (!tsfile::p02::decode_rle_int64(encoded_tail, decoded) || decoded.size() != tail) {
          throw std::runtime_error("invalid Sprintz integer RLE tail");
        }
        values.insert(values.end(), decoded.begin(), decoded.end());
      } else {
        std::vector<double> decoded;
        if (!tsfile::p03::decode_gorilla_double(encoded_tail, decoded) ||
            decoded.size() != tail) {
          throw std::runtime_error("invalid Sprintz floating Gorilla tail");
        }
        const auto old_size = values.size();
        values.resize(old_size + tail);
        std::memcpy(values.data() + old_size, decoded.data(), tail * sizeof(double));
      }
      pos = wire.size();
      break;
    }
    const int width = marker;
    if (width < 1 || width > 64 || wire.size() - pos < 8 + static_cast<std::size_t>(width)) {
      throw std::runtime_error("invalid Sprintz block");
    }
    values.push_back(read_i64_be(wire, pos));
    std::int64_t unpacked[8]{};
    unpack8(width, wire.subspan(pos, static_cast<std::size_t>(width)), unpacked);
    pos += static_cast<std::size_t>(width);
    values.insert(values.end(), std::begin(unpacked), std::end(unpacked));
  }
  if (pos != wire.size()) throw std::runtime_error("trailing Sprintz bytes");
  return serialize(zigzag_magic(kind), values);
}

}  // namespace

std::vector<std::uint8_t> sprintz_bitpack_inverse(
    std::span<const std::uint8_t> wire) {
  return sprintz_bitpack_inverse_typed(wire, SprintzKind::integer);
}

std::vector<std::uint8_t> sprintz_bitpack_inverse_double(
    std::span<const std::uint8_t> wire) {
  return sprintz_bitpack_inverse_typed(wire, SprintzKind::floating);
}

std::vector<std::uint8_t> sprintz_bitpack_inverse_current(
    std::span<const std::uint8_t> wire) {
  return sprintz_bitpack_inverse_typed(wire, SprintzKind::integer);
}

std::vector<std::uint8_t> sprintz_bitpack_inverse_double_current(
    std::span<const std::uint8_t> wire) {
  return sprintz_bitpack_inverse_typed(wire, SprintzKind::floating);
}

std::vector<std::uint8_t> sprintz_compose(std::span<const std::int64_t> values) {
  return sprintz_bitpack_forward(sprintz_zigzag_forward(sprintz_predictor_forward(values)));
}

std::vector<std::uint8_t> sprintz_compose_double_bits(
    std::span<const std::int64_t> values) {
  return sprintz_bitpack_forward(
      sprintz_zigzag_forward(sprintz_predictor_forward_double_bits(values)));
}

std::vector<std::int64_t> sprintz_decompose(std::span<const std::uint8_t> wire) {
  return sprintz_predictor_inverse(
      sprintz_zigzag_inverse(sprintz_bitpack_inverse(wire)));
}

std::vector<std::int64_t> sprintz_decompose_double_bits(
    std::span<const std::uint8_t> wire) {
  return sprintz_predictor_inverse(sprintz_zigzag_inverse(
      sprintz_bitpack_inverse_typed(wire, SprintzKind::floating)));
}

}  // namespace exact_actions::p04
