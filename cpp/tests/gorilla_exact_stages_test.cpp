#include "gorilla_exact_stages.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  using namespace exact_actions::gorilla;
  for (const std::vector<std::int64_t> values : {
           std::vector<std::int64_t>{},
           std::vector<std::int64_t>{-9, -9, -3, 0, 1, 17, 17, 99}}) {
    const auto xors = xor_forward_int64(values);
    if (xor_inverse(xors) != values) throw std::runtime_error("Gorilla XOR round trip");
    const auto wire = trailing_pack_forward(xors);
    if (wire != tsfile::p03::encode_gorilla_int64(values)) throw std::runtime_error("Gorilla int wire parity");
    if (trailing_pack_inverse_int64(wire) != xors) throw std::runtime_error("Gorilla pack inverse");
    if (decompose_int64(wire) != values) throw std::runtime_error("Gorilla int chain");
  }
  const std::vector<double> doubles{-19.25, -1.0, -0.0, 0.0, 0.125, 3.5, 17.25};
  std::vector<std::int64_t> bits(doubles.size());
  std::memcpy(bits.data(), doubles.data(), doubles.size() * 8);
  const auto xors = xor_forward_double_bits(bits);
  const auto wire = trailing_pack_forward(xors);
  if (wire != tsfile::p03::encode_gorilla_double(doubles)) throw std::runtime_error("Gorilla double wire parity");
  if (decompose_double_bits(wire) != bits) throw std::runtime_error("Gorilla double chain");
  std::cout << "gorilla_exact_stage_actions=2 status=PASS\n";
}
