#include "fpc_exact_stages.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
size_t FPC_Compress_Memory(const void*, size_t, void*, long);
size_t FPC_Decompress_Memory(const void*, size_t, void*);
}

int main() {
  std::vector<std::uint64_t> values{
      0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};
  std::vector<std::uint8_t> raw(values.size() * 8);
  std::memcpy(raw.data(), values.data(), raw.size());
  using namespace exact_actions::fpc;
  const auto residuals = predictor_xor_forward(raw);
  if (predictor_xor_inverse(residuals) != raw)
    throw std::runtime_error("FPC predictor round trip");
  const auto wire = trailing_pack_forward(residuals);
  if (trailing_pack_inverse(wire) != residuals)
    throw std::runtime_error("FPC pack round trip");
  std::vector<std::uint8_t> reference(raw.size() * 2 + 1024);
  const auto size = FPC_Compress_Memory(raw.data(), raw.size(), reference.data(), 16);
  reference.resize(size);
  if (wire != reference) throw std::runtime_error("FPC public wire mismatch");
  if (decompose(wire) != raw) throw std::runtime_error("FPC chain round trip");

  // Empty input has a native, unambiguous FPC representation: the public
  // encoder writes only its one-byte predictor-size header.  This catches an
  // adapter guard that used to reject a wire the native encoder can emit.
  const std::vector<std::uint8_t> empty;
  for (unsigned predictor_lg2 = 0; predictor_lg2 <= 24; ++predictor_lg2) {
    const auto empty_residuals = predictor_xor_forward(empty, predictor_lg2);
    if (!predictor_xor_inverse(empty_residuals).empty())
      throw std::runtime_error("FPC empty predictor round trip");
    const auto empty_wire = trailing_pack_forward(empty_residuals);
    if (empty_wire !=
        std::vector<std::uint8_t>{static_cast<std::uint8_t>(predictor_lg2)})
      throw std::runtime_error("FPC empty native wire mismatch");
    if (trailing_pack_inverse(empty_wire) != empty_residuals)
      throw std::runtime_error("FPC empty pack round trip");
    if (!decompose(empty_wire).empty())
      throw std::runtime_error("FPC empty chain round trip");
  }

  const auto expect_bad_empty_wire = [](std::vector<std::uint8_t> malformed) {
    try {
      (void)trailing_pack_inverse(malformed);
    } catch (const std::exception&) {
      return;
    }
    throw std::runtime_error("malformed FPC empty wire was accepted");
  };
  expect_bad_empty_wire({});
  expect_bad_empty_wire({25});
  expect_bad_empty_wire({16, 0});
  expect_bad_empty_wire({16, 0, 0, 0, 0, 0, 0});

  std::uint8_t empty_input_sentinel = 0;
  std::vector<std::uint8_t> empty_reference(16, 0xff);
  const auto empty_reference_size = FPC_Compress_Memory(
      &empty_input_sentinel, 0, empty_reference.data(), 16);
  empty_reference.resize(empty_reference_size);
  if (empty_reference != std::vector<std::uint8_t>{16})
    throw std::runtime_error("FPC public empty wire mismatch");
  std::uint8_t empty_output = 0xa5;
  if (FPC_Decompress_Memory(empty_reference.data(), empty_reference.size(),
                            &empty_output) != empty_reference.size() ||
      empty_output != 0xa5)
    throw std::runtime_error("FPC public empty decode mismatch");
  for (const auto& malformed :
       std::vector<std::vector<std::uint8_t>>{std::vector<std::uint8_t>(7)}) {
    bool rejected = false;
    try { (void)predictor_xor_forward(malformed, 16); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unaligned FPC input accepted");
  }
  bool bad_predictor_rejected = false;
  try { (void)predictor_xor_forward(empty, 25); }
  catch (const std::invalid_argument&) { bad_predictor_rejected = true; }
  if (!bad_predictor_rejected)
    throw std::runtime_error("invalid FPC predictor accepted for empty input");
  return 0;
}
