#include "p04_exact_stages.hpp"

#include "tsfile_encoding/tsfile_encode_p04.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"

#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::int64_t> as_bits(const std::vector<double>& values) {
  std::vector<std::int64_t> bits(values.size());
  std::memcpy(bits.data(), values.data(), values.size() * sizeof(double));
  return bits;
}

void check(const std::vector<std::int64_t>& values) {
  using namespace exact_actions::p04;

  const auto rlbe_delta = rlbe_delta_forward(values);
  if (rlbe_delta_inverse(rlbe_delta) != values) throw std::runtime_error("RLBE delta round trip");
  const auto rlbe_wire = rlbe_run_length_forward(rlbe_delta);
  if (rlbe_run_length_inverse(rlbe_wire) != rlbe_delta) throw std::runtime_error("RLBE run stage round trip");
  if (rlbe_wire != tsfile::p04::encode_rlbe_int64_legacy(values)) throw std::runtime_error("RLBE legacy wire parity");
  if (rlbe_decompose(rlbe_wire) != values) throw std::runtime_error("RLBE chain round trip");

  const auto predictor = sprintz_predictor_forward(values);
  if (sprintz_predictor_inverse(predictor) != values) throw std::runtime_error("Sprintz predictor round trip");
  const auto zigzag = sprintz_zigzag_forward(predictor);
  if (sprintz_zigzag_inverse(zigzag) != predictor) throw std::runtime_error("Sprintz zigzag round trip");
  const auto sprintz_wire = sprintz_bitpack_forward(zigzag);
  if (sprintz_bitpack_inverse(sprintz_wire) != zigzag) throw std::runtime_error("Sprintz bitpack round trip");
  const std::size_t full_size = values.size() / 9 * 9;
  // Complete 9-value blocks have no tail. The retained ablation entry with
  // no omitted operator preserves their original bitstream independently
  // of the new reference encoder's URL1 tail selection.
  auto expected_sprintz = tsfile::p04::encode_sprintz_int64_ablation(
      std::span<const std::int64_t>(values).first(full_size), {});
  if (full_size != values.size()) {
    expected_sprintz.push_back(static_cast<std::uint8_t>(0x80u | (values.size() - full_size)));
    const auto tail = tsfile::p02::encode_rle_int64_legacy(
        std::span<const std::int64_t>(values).subspan(full_size));
    expected_sprintz.insert(expected_sprintz.end(), tail.begin(), tail.end());
  }
  if (sprintz_wire != expected_sprintz) throw std::runtime_error("Sprintz benchmark wire parity");
  if (sprintz_decompose(sprintz_wire) != values) throw std::runtime_error("Sprintz chain round trip");
}

}  // namespace

int main() {
  check({});
  check({-9, -9, -3, 0, 1, 1, 17, 17, 17, 99, 3, -7, 42});

  std::vector<std::int64_t> multi_block(10017);
  for (std::size_t i = 0; i < multi_block.size(); ++i) {
    multi_block[i] = static_cast<std::int64_t>((i * 104729u) ^ (i >> 3));
  }
  check(multi_block);

  const std::vector<double> doubles = {
      -19.25, -1.0, -0.0, 0.0, 0.125, 3.5, 17.0, 17.25, 1000.0,
      -8.75, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
      512.25, -1024.5};
  const auto bits = as_bits(doubles);
  check(bits);
  if (exact_actions::p04::rlbe_compose(bits) != tsfile::p04::encode_rlbe_double_legacy(doubles)) {
    throw std::runtime_error("RLBE double legacy wire parity");
  }
  const auto sprintz_double = exact_actions::p04::sprintz_compose_double_bits(bits);
  const std::size_t full_double_size = doubles.size() / 9 * 9;
  auto expected_double = tsfile::p04::encode_sprintz_double_ablation(
      std::span<const double>(doubles).first(full_double_size), {});
  if (full_double_size != doubles.size()) {
    expected_double.push_back(
        static_cast<std::uint8_t>(0x80u | (doubles.size() - full_double_size)));
    const auto tail = tsfile::p03::encode_gorilla_double(
        std::span<const double>(doubles).subspan(full_double_size));
    expected_double.insert(expected_double.end(), tail.begin(), tail.end());
  }
  if (sprintz_double != expected_double) {
    throw std::runtime_error("Sprintz double benchmark wire parity");
  }
  if (exact_actions::p04::sprintz_decompose_double_bits(sprintz_double) != bits) {
    throw std::runtime_error("Sprintz double chain round trip");
  }

  std::cout << "p04_exact_stage_actions=5 status=PASS\n";
  return 0;
}
