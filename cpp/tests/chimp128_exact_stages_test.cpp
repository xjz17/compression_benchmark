#include "chimp128_exact_stages.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  using namespace exact_actions::chimp128;
  for (const std::vector<std::int64_t> values : {
           std::vector<std::int64_t>{},
           std::vector<std::int64_t>{-9, -9, -3, 0, 1, 17, 17, 99, -9, 0}}) {
    const auto xors = xor_forward_int64(values);
    if (xor_inverse(xors) != values) throw std::runtime_error("Chimp128 XOR round trip");
    const auto wire = trailing_pack_forward(xors);
    if (wire != tsfile::p03::encode_chimp_int64(values)) throw std::runtime_error("Chimp128 int wire parity");
    if (trailing_pack_inverse_int64(wire) != xors) throw std::runtime_error("Chimp128 pack inverse");
    if (decompose_int64(wire) != values) throw std::runtime_error("Chimp128 int chain");
  }
  const std::vector<double> doubles{-19.25, -1.0, -0.0, 0.0, 0.125, 3.5, 17.25, -19.25};
  std::vector<std::int64_t> bits(doubles.size());
  std::memcpy(bits.data(), doubles.data(), doubles.size() * 8);
  const auto xors = xor_forward_double_bits(bits);
  const auto wire = trailing_pack_forward(xors);
  if (wire != tsfile::p03::encode_chimp_double(doubles)) throw std::runtime_error("Chimp128 double wire parity");
  if (decompose_double_bits(wire) != bits) throw std::runtime_error("Chimp128 double chain");

  // 0x7ff8000000000000 is the historical Chimp double end marker. It is also
  // a valid, payload-bearing IEEE-754 input word and must round-trip at every
  // position without being mistaken for end-of-stream.
  constexpr std::int64_t marker = static_cast<std::int64_t>(0x7ff8000000000000ull);
  for (const std::vector<std::int64_t> marker_bits : {
           std::vector<std::int64_t>{marker},
           std::vector<std::int64_t>{marker, 0, 1},
           std::vector<std::int64_t>{0, marker, 1},
           std::vector<std::int64_t>{0, 1, marker},
           std::vector<std::int64_t>{marker, marker, marker}}) {
    const auto marker_wire = compose_double_bits(marker_bits);
    if (decompose_double_bits(marker_wire) != marker_bits)
      throw std::runtime_error("Chimp128 escaped marker double chain");
  }
  auto corrupted = compose_double_bits(bits);
  corrupted.push_back(1);
  bool trailing_rejected = false;
  try {
    static_cast<void>(decompose_double_bits(corrupted));
  } catch (const std::runtime_error& error) {
    trailing_rejected =
        std::string(error.what()) == "non-zero Chimp128 trailing bits";
  }
  if (!trailing_rejected)
    throw std::runtime_error("Chimp128 non-zero trailing bits were accepted");
  std::cout << "chimp128_exact_stage_actions=2 status=PASS\n";
}
