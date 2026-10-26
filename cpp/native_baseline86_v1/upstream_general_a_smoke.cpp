#include "upstream_general_a.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using native_baseline86_v1::upstream_general_a::Bytes;

int main() {
  Bytes repetitive(32768);
  for (std::size_t i = 0; i < repetitive.size(); ++i)
    repetitive[i] = static_cast<std::uint8_t>((i / 64) & 7);
  Bytes noisy(4093);
  std::uint32_t state = 0x12345678U;
  for (auto& value : noisy) {
    state = state * 1664525U + 1013904223U;
    value = static_cast<std::uint8_t>(state >> 24);
  }
  Bytes empty;
  Bytes one{0xa5};
  Bytes odd(37);
  for (std::size_t i = 0; i < odd.size(); ++i)
    odd[i] = static_cast<std::uint8_t>(i * 29 + 7);

  std::size_t tested = 0;
  for (const auto& record :
       native_baseline86_v1::upstream_general_a::inventory()) {
    if (!record.available) continue;
    bool first = true;
    for (const auto* input : {&repetitive, &noisy, &empty, &one, &odd}) {
      const auto frame =
          native_baseline86_v1::upstream_general_a::encode(record.label, *input);
      if (first && (frame.empty() || frame[0] != 1))
        throw std::runtime_error(record.label +
                                 " did not exercise its upstream stream");
      const auto decoded = native_baseline86_v1::upstream_general_a::decode(
          record.label, frame, input->size());
      if (decoded != *input) throw std::runtime_error(record.label + " mismatch");
      auto corrupt = frame;
      corrupt[0] = 2;
      bool rejected = false;
      try {
        (void)native_baseline86_v1::upstream_general_a::decode(
            record.label, corrupt, input->size());
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error(record.label + " accepted an unknown tag");
      first = false;
      ++tested;
    }
    std::cout << "PASS " << record.label << " direct_stream\n";
  }
  if (tested == 0) throw std::runtime_error("no direct implementation enabled");

  // LZO2A can emit one control bit for every incompressible literal.  A large
  // deterministic pseudo-random block exercises the adapter's dedicated
  // ceil(n/8) worst-case allowance rather than the smaller generic LZO bound.
  const auto inventory = native_baseline86_v1::upstream_general_a::inventory();
  const auto lzo2a = std::find_if(
      inventory.begin(), inventory.end(),
      [](const auto& record) { return record.label == "LZO2A"; });
  if (lzo2a != inventory.end() && lzo2a->available) {
    Bytes incompressible(1U << 20);
    std::uint32_t random_state = 0x6d2b79f5U;
    for (auto& value : incompressible) {
      random_state ^= random_state << 13;
      random_state ^= random_state >> 17;
      random_state ^= random_state << 5;
      value = static_cast<std::uint8_t>(random_state >> 24);
    }
    const auto frame =
        native_baseline86_v1::upstream_general_a::encode("LZO2A", incompressible);
    const auto decoded = native_baseline86_v1::upstream_general_a::decode(
        "LZO2A", frame, incompressible.size());
    if (decoded != incompressible)
      throw std::runtime_error("LZO2A incompressible bound mismatch");
    std::cout << "PASS LZO2A incompressible_worst_case_bound\n";
  }
  std::cout << "PASS cases=" << tested << '\n';
}
