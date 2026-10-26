#include "zstd_exact_stages.hpp"

#include <zstd.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  using namespace exact_actions::zstd;
  int public_wire_mismatches = 0;
  std::vector<std::vector<std::uint8_t>> probes{{}, {'a'}, std::vector<std::uint8_t>(4096, 7)};
  std::vector<std::uint8_t> ramp(140000);
  for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<std::uint8_t>(i % 251);
  probes.push_back(std::move(ramp));
  for (const auto& input : probes) {
    const auto sequences = sequence_forward(input);
    if (sequence_inverse(sequences) != input) throw std::runtime_error("Zstd sequence roundtrip");
    const auto wire = entropy_frame_forward(sequences);
    if (sequence_inverse(entropy_frame_inverse(wire)) != input)
      throw std::runtime_error("Zstd entropy roundtrip");
    if (decompose(wire) != input) throw std::runtime_error("Zstd chain roundtrip");
    std::vector<std::uint8_t> reference(ZSTD_compressBound(input.size()));
    const auto size = ZSTD_compress(reference.data(), reference.size(), input.data(), input.size(), 9);
    if (ZSTD_isError(size)) throw std::runtime_error("Zstd reference compression");
    reference.resize(size);
    if (wire != reference) ++public_wire_mismatches;
  }
  std::cout << "zstd_sequence_stage_roundtrip=PASS public_wire_mismatches="
            << public_wire_mismatches << "/" << probes.size() << "\n";
}
