#include "learning_codec_exact_stages.hpp"

#include "alp_float_codec.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"
#include "leco_operator_ablation.hpp"
#include "reger_codec.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<std::uint8_t> bytes(const std::vector<T>& values) {
  std::vector<std::uint8_t> out(values.size() * sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), values.data(), out.size());
  return out;
}

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template <typename Fn>
void require_rejected(Fn operation, const char* message) {
  try { operation(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}

std::vector<std::uint8_t> buff_integer_packet(const std::vector<std::int64_t>& values) {
  std::vector<std::uint8_t> packet = {'B', 'S', 'C', '1', 1, 3};
  const auto count = static_cast<std::uint32_t>(values.size());
  for (unsigned shift = 0; shift < 32; shift += 8)
    packet.push_back(static_cast<std::uint8_t>(count >> shift));
  const auto raw = bytes(values);
  packet.insert(packet.end(), raw.begin(), raw.end());
  return packet;
}

}  // namespace

int main() {
  using namespace exact_actions::learning;

  const std::vector<std::int64_t> integers = {
      19, -4, 19, 8, 1000, -4, 23, 24, 25,
      std::numeric_limits<std::int64_t>::min() + 8,
      std::numeric_limits<std::int64_t>::max() - 8,
  };
  const auto int_wire = bytes(integers);

  const auto leco_regression = leco_regression_forward(int_wire);
  require(leco_regression_inverse(leco_regression) == int_wire,
          "LECO pure regression roundtrip");
  const auto leco_ar = leco_zigzag_forward(leco_regression);
  require(leco_zigzag_inverse(leco_ar) == leco_regression,
          "LECO ZigZag adapter roundtrip");
  require(leco_ar == leco_autoregression_forward(int_wire),
          "LECO compatibility packet parity");
  require(leco_autoregression_inverse(leco_ar) == int_wire,
          "LECO autoregression roundtrip");
  const auto leco = leco_bitpack_forward(leco_ar);
  require(leco_bitpack_inverse(leco) == leco_ar, "LECO bitpack roundtrip");
  require(leco_autoregression_inverse(leco_bitpack_inverse(leco)) == int_wire,
          "LECO chain roundtrip");
  require(leco == leco_operator_ablation::encode(integers, ""),
          "LECO exact source-wire parity");

  const std::vector<double> floats = {
      1.25, 1.25, -0.5, 100.125, 42.0, 42.0, 7.75,
  };
  const auto float_wire = bytes(floats);

  const auto alp_scale = alp_scaling_forward(float_wire);
  require(alp_scaling_inverse(alp_scale) == float_wire, "ALP scaling roundtrip");
  const auto alp_for = alp_for_forward(alp_scale);
  require(alp_for_inverse(alp_for) == alp_scale, "ALP FOR roundtrip");
  const auto alp = alp_bitpack_forward(alp_for);
  require(alp_bitpack_inverse(alp) == alp_for, "ALP bitpack roundtrip");
  require(alp_scaling_inverse(alp_for_inverse(alp_bitpack_inverse(alp))) ==
              float_wire,
          "ALP main chain roundtrip");
  const auto alp_exact = alp_residual_core_forward(alp);
  require(alp_exact == alp_float::encodeDoubleColumnExact(floats),
          "ALP residual exact source-wire parity");
  require(alp_residual_core_inverse(alp_exact) == alp,
          "ALP residual inverse");
  const auto dictionary = alp_dictionary_forward(float_wire);
  require(alp_dictionary_inverse(dictionary) == float_wire,
          "ALP-RD dictionary roundtrip");

  const auto reorder = reger_reorder_forward(int_wire);
  require(reger_reorder_inverse(reorder) == int_wire, "REGER reorder roundtrip");
  const auto delta = reger_delta_forward(reorder);
  require(reger_delta_inverse(delta) == reorder,
          "REGER pure delta roundtrip");
  const auto regression = reger_zigzag_forward(delta);
  require(reger_zigzag_inverse(regression) == delta,
          "REGER ZigZag adapter roundtrip");
  require(regression == reger_autoregression_forward(reorder),
          "REGER compatibility packet parity");
  require(reger_autoregression_inverse(regression) == reorder,
          "REGER autoregression roundtrip");
  const auto reger_for = reger_for_forward(regression);
  require(reger_for_inverse(reger_for) == regression, "REGER FOR roundtrip");
  const auto reger = reger_bitpack_forward(reger_for);
  require(reger_bitpack_inverse(reger) == reger_for, "REGER bitpack roundtrip");
  require(reger_reorder_inverse(reger_autoregression_inverse(
              reger_for_inverse(reger_bitpack_inverse(reger)))) == int_wire,
          "REGER chain roundtrip");
  const auto reger_exact = reger_residual_core_forward(reger);
  require(reger_exact == reger_codec::encodeInt64Column(integers),
          "REGER residual exact source-wire parity");
  require(reger_residual_core_inverse(reger_exact) == reger,
          "REGER residual inverse");

  const auto buff_scale = buff_scaling_forward(float_wire);
  require(buff_scaling_inverse(buff_scale) == float_wire, "BUFF scaling roundtrip");
  const auto buff_for = buff_for_forward(buff_scale);
  require(buff_for_inverse(buff_for) == buff_scale, "BUFF independently callable FOR roundtrip");
  const auto buff_width = buff_byte_plane_width(buff_for);
  const auto pure_split = buff_byte_plane_forward(buff_for, buff_width);
  require(buff_byte_plane_inverse(pure_split) == buff_for,
          "BUFF pure plane inverse returns offsets, not restored integers");
  const auto split = buff_column_split_forward(buff_scale);
  require(pure_split == split, "BUFF explicit FOR and planes legacy packet parity");
  require(buff_column_split_inverse(split) == buff_scale,
          "BUFF column-split roundtrip");
  const auto buff = buff_bitpack_forward(split);
  require(buff_bitpack_inverse(buff) == split, "BUFF bitpack roundtrip");
  require(buff_scaling_inverse(buff_column_split_inverse(
              buff_bitpack_inverse(buff))) == float_wire,
          "BUFF chain roundtrip");
  require(buff == buff_kernel::encode_double_ablation(floats, ""),
          "BUFF exact source-wire parity");

  // Plane order is most-significant byte first, with low padding only in the
  // final slice.  It never chooses a base or subtracts a value.
  const std::vector<std::uint64_t> offsets_9 = {0, 1, 0x1ff};
  const std::vector<std::uint8_t> expected_planes_9 = {0, 0, 255, 0, 128, 128};
  require(buff_byte_planes_split(offsets_9, 9) == expected_planes_9,
          "BUFF 9-bit byte-plane layout");
  require(buff_byte_planes_join(expected_planes_9, offsets_9.size(), 9) == offsets_9,
          "BUFF 9-bit byte-plane inverse");
  for (unsigned width = 0; width <= 64; ++width) {
    const auto maximum = width == 64 ? std::numeric_limits<std::uint64_t>::max()
                                     : (std::uint64_t{1} << width) - 1;
    const std::vector<std::uint64_t> values = {0, maximum / 2, maximum};
    const auto planes = buff_byte_planes_split(values, width);
    require(buff_byte_planes_join(planes, values.size(), width) == values,
            "BUFF explicit-width plane roundtrip");
    require(buff_byte_plane_max_width(values) == width, "BUFF separate width configuration");
    require(buff_byte_planes_join(buff_byte_planes_split({}, width), 0, width).empty(),
            "BUFF empty plane roundtrip");
  }
  for (const auto& values : std::vector<std::vector<std::int64_t>>{
           {}, {8}, {17,17,17}, {-17,-16,0,5},
           {std::numeric_limits<std::int64_t>::min(),
            std::numeric_limits<std::int64_t>::max(), 0}}) {
    const auto scale_packet = buff_integer_packet(values);
    const auto normalized = buff_for_forward(scale_packet);
    require(buff_for_inverse(normalized) == scale_packet, "BUFF FOR full-domain restoration");
    const auto planes = buff_byte_plane_forward(normalized, buff_byte_plane_width(normalized));
    require(buff_byte_plane_inverse(planes) == normalized, "BUFF plane full-domain offsets");
    require(planes == buff_column_split_forward(scale_packet), "BUFF full-domain legacy parity");
    auto other_base = planes;
    other_base[10] ^= 1;  // base occupies bytes10..17 in BSF1, outside the plane data.
    const auto other_for = buff_byte_plane_inverse(other_base);
    require(std::vector<std::uint8_t>(other_for.begin()+18, other_for.end()) ==
                std::vector<std::uint8_t>(normalized.begin()+18, normalized.end()),
            "BUFF byte-plane inverse must not use base");
  }
  require_rejected([&]{buff_byte_planes_split(offsets_9, 8);}, "BUFF narrow width must reject");
  require_rejected([&]{buff_byte_planes_split(offsets_9, 65);}, "BUFF width65 must reject");
  require_rejected([&]{buff_byte_planes_split(std::vector<std::uint64_t>{1}, 0);},
                   "BUFF nonzero width0 must reject");
  require_rejected([&]{buff_byte_planes_join(expected_planes_9, 3, 65);},
                   "BUFF inverse width65 must reject");
  require_rejected([&]{buff_byte_planes_join(expected_planes_9, 4, 9);},
                   "BUFF inverse incorrect count must reject");
  auto bad_padding = expected_planes_9;
  bad_padding.back() |= 1;
  require_rejected([&]{buff_byte_planes_join(bad_padding, 3, 9);},
                   "BUFF nonzero padding must reject");

  const std::vector<double> raw_fallback = {
      -0.0, std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
  };
  const auto fallback_wire = bytes(raw_fallback);
  require(alp_scaling_inverse(alp_scaling_forward(fallback_wire)) == fallback_wire,
          "ALP raw-bit fallback roundtrip");
  require(buff_scaling_inverse(buff_scaling_forward(fallback_wire)) == fallback_wire,
          "BUFF raw-bit fallback roundtrip");

  std::cout << "learning_codec_exact_stages=PASS leco_source_parity=PASS "
               "alp_source_parity=PASS reger_source_parity=PASS "
               "buff_source_parity=PASS\n";
}
