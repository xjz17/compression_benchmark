#include "max_value_bitpack.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace bp = exact_actions::max_value_bitpack;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F function, const char* message) {
  try { function(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}

// Frozen per-bit reference is independent of the optimized byte fragments.
std::vector<std::uint8_t> reference_pack(std::span<const std::uint64_t> values,
                                       unsigned width, bp::Format format) {
  std::vector<std::uint8_t> payload(bp::payload_size(values.size(), width, format), 0);
  std::size_t position = 0;
  for (auto value : values) {
    for (unsigned i = 0; i < width; ++i, ++position) {
      auto from = format.order == bp::BitOrder::msb_first ? width - 1 - i : i;
      auto to = format.order == bp::BitOrder::msb_first ? 7 - position % 8 : position % 8;
      payload[position / 8] |= static_cast<std::uint8_t>(((value >> from) & 1U) << to);
    }
  }
  return payload;
}

std::vector<std::uint8_t> raw_bytes(const std::vector<std::uint64_t>& values) {
  std::vector<std::uint8_t> out;
  for (const auto value : values)
    for (unsigned shift = 0; shift < 64; shift += 8)
      out.push_back(static_cast<std::uint8_t>(value >> shift));
  return out;
}
}  // namespace

int main() {
  std::mt19937_64 random(0x4d565731);
  std::size_t roundtrips = 0;
  for (unsigned width = 0; width <= 64; ++width) {
    const auto maximum = width == 64 ? std::numeric_limits<std::uint64_t>::max()
                                    : (std::uint64_t{1} << width) - 1;
    for (const auto count : {1, 2, 3, 7, 8, 9, 63, 64, 65, 128, 1024}) {
      std::vector<std::uint64_t> values(count);
      for (auto& value : values) value = random() & maximum;
      values.back() = maximum;
      for (const auto order : {bp::BitOrder::msb_first, bp::BitOrder::lsb_first}) {
        for (const auto alignment : {1U, 8U}) {
          const bp::Format format{order, alignment};
          const auto packed = bp::pack(values, format);
          require(packed.payload == reference_pack(values, width, format),
                  "optimized pack changed the frozen per-bit wire");
          require(packed.width == width, "width differs from maximum numeric value");
          require(bp::unpack(packed.payload, values.size(), width, format) == values,
                  "core roundtrip mismatch");
          ++roundtrips;
        }
      }
      const auto raw = raw_bytes(values);
      require(bp::decode(bp::encode(raw)) == raw, "MVW1 uint64 LE roundtrip mismatch");
      ++roundtrips;
    }
  }
  require(bp::pack({}).width == 0 && bp::pack({}).payload.empty(), "empty core");
  const auto empty = bp::encode({});
  require(empty.size() == 13 && bp::decode(empty).empty(), "empty MVW1");
  const std::array<std::uint64_t, 3> small{1, 2, 7};
  const std::array<std::uint64_t, 3> supplied{0, 3, 1};
  require(bp::pack_fixed(supplied, 2).payload == std::vector<std::uint8_t>{0x34},
          "explicit-width known wire");
  rejects([&] { bp::pack_fixed(small, 2); }, "accepted undersized supplied width");
  rejects([&] { bp::pack_fixed(small, 65); }, "accepted supplied width 65");
  rejects([&] { bp::allowed_width(small, 57, 56); }, "accepted invalid width profile");
  const std::array<std::uint64_t, 2> zeros{0, 0};
  require(bp::allowed_width(zeros, 1, 56) == 1, "BOS64 zero-width profile");
  for (unsigned bits : {1U, 55U, 56U, 57U, 63U, 64U}) {
    const std::array<std::uint64_t, 1> value{std::uint64_t{1} << (bits - 1)};
    const auto width = bp::allowed_width(value, 1, 56);
    require(width == (bits > 56 ? 64 : bits), "BOS64 width rounding boundary");
    const auto packed = bp::pack_fixed(value, width);
    require(bp::unpack(packed.payload, 1, width)[0] == value[0], "rounded-width decode");
  }
  require(bp::pack(small).payload == std::vector<std::uint8_t>({0x2b, 0x80}),
          "MSB known wire");
  const bp::Format lsb{bp::BitOrder::lsb_first, 8};
  require(bp::pack(small, lsb).payload ==
              std::vector<std::uint8_t>({0xd1, 1, 0, 0, 0, 0, 0, 0}), "LSB word wire");
  auto bad_padding = bp::pack(small).payload;
  bad_padding.back() |= 1;
  rejects([&] { bp::unpack(bad_padding, small.size(), 3); }, "accepted MSB padding");
  bad_padding = bp::pack(small, lsb).payload;
  bad_padding[1] |= 2;
  rejects([&] { bp::unpack(bad_padding, small.size(), 3, lsb); }, "accepted LSB padding");
  bad_padding = bp::pack(small, lsb).payload;
  bad_padding.back() = 1;
  rejects([&] { bp::unpack(bad_padding, small.size(), 3, lsb); }, "accepted word padding");
  rejects([&] { bp::payload_size(1, 65); }, "accepted width 65");
  rejects([&] { bp::payload_size(std::numeric_limits<std::size_t>::max(), 64); },
          "accepted bit count overflow");
  rejects([&] { bp::unpack({}, std::numeric_limits<std::size_t>::max(), 0); },
          "accepted decoded vector size overflow");
  rejects([&] { bp::pack(small, {bp::BitOrder::msb_first, 3}); }, "accepted alignment 3");
  rejects([&] { bp::pack(small, {static_cast<bp::BitOrder>(9), 1}); }, "accepted bit order 9");
  const std::array<std::uint8_t, 1> unaligned{1};
  rejects([&] { bp::encode(unaligned); }, "accepted unaligned raw input");
  auto wire = bp::encode(raw_bytes({1, 2, 7}));
  for (std::size_t i = 0; i < wire.size(); ++i)
    rejects([&] { bp::decode(std::span(wire).first(i)); }, "accepted truncated MVW1");
  auto corrupted = wire;
  corrupted.push_back(0);
  rejects([&] { bp::decode(corrupted); }, "accepted trailing MVW1 bytes");
  corrupted = wire; corrupted[0] = 'X';
  rejects([&] { bp::decode(corrupted); }, "accepted wrong magic");
  corrupted = wire; corrupted[12] = 65;
  rejects([&] { bp::decode(corrupted); }, "accepted invalid MVW1 width");
  corrupted = wire; corrupted.back() |= 1;
  rejects([&] { bp::decode(corrupted); }, "accepted MVW1 nonzero padding");
  corrupted = wire;
  for (std::size_t i = 4; i < 12; ++i) corrupted[i] = 0xff;
  rejects([&] { bp::decode(corrupted); }, "accepted MVW1 length overflow");
  corrupted = empty; corrupted[12] = 1;
  rejects([&] { bp::decode(corrupted); }, "accepted nonminimal empty width");
  corrupted = bp::encode(raw_bytes({0})); corrupted[12] = 1; corrupted.push_back(0);
  rejects([&] { bp::decode(corrupted); }, "accepted nonminimal zero width");
  std::cout << "max_value_bitpack=PASS widths=0..64 roundtrips=" << roundtrips
            << " formats=MSB/LSB+byte/word invalid_inputs=PASS\n";
}
