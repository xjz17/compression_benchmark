#include "lz4_exact_stages.hpp"

#include <lz4frame.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::uint8_t> reference(const std::vector<std::uint8_t>& input) {
  LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
  prefs.frameInfo.blockSizeID = LZ4F_max64KB;
  prefs.frameInfo.blockMode = LZ4F_blockLinked;
  prefs.frameInfo.contentChecksumFlag = LZ4F_noContentChecksum;
  prefs.frameInfo.blockChecksumFlag = LZ4F_noBlockChecksum;
  prefs.frameInfo.contentSize = input.size();
  std::vector<std::uint8_t> out(LZ4F_compressFrameBound(input.size(), &prefs));
  const auto size = LZ4F_compressFrame(out.data(), out.size(), input.data(), input.size(), &prefs);
  if (LZ4F_isError(size)) throw std::runtime_error("reference LZ4F failed");
  out.resize(size);
  return out;
}
}
int main() {
  using namespace exact_actions::lz4;
  std::vector<std::vector<std::uint8_t>> probes{{}, {'a'}, std::vector<std::uint8_t>(4096, 7)};
  std::vector<std::uint8_t> ramp(140000);
  for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<std::uint8_t>(i % 251);
  probes.push_back(std::move(ramp));
  for (const auto& input : probes) {
    const auto m = lz_match_forward(input);
    if (lz_match_inverse(m) != input) throw std::runtime_error("LZ4 match roundtrip");
    const auto t = token_format_forward(m);
    if (token_format_inverse(t) != m) throw std::runtime_error("LZ4 token roundtrip");
    const auto i = integer_code_forward(t);
    if (integer_code_inverse(i) != t) throw std::runtime_error("LZ4 integer roundtrip");
    const auto wire = container_forward(i);
    if (container_inverse(wire) != i) throw std::runtime_error("LZ4 frame roundtrip");
    const auto expected = reference(input);
    if (wire != expected) throw std::runtime_error("LZ4 public wire parity");
    if (decompose(wire) != input) throw std::runtime_error("LZ4 chain roundtrip");
  }
  std::cout << "lz4_exact_stage_actions=4 status=PASS\n";
}
