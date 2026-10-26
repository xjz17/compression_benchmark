#include "vendored_lightweight_codecs_v1.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using native_baseline86_v1::vendored_lightweight::Bytes;

namespace {
constexpr std::array<const char*, 20> labels = {
    "Libdeflate", "FastLZ", "LZF",   "LZFSE", "LZVN",
    "BriefLZ",    "LZAV",   "LZG",   "LZJB",  "TAMP", "Bzip3",
    "FSE", "Huff0", "LZMAT", "LZRW", "WFLZ", "Yappy",
    "YALZ77", "QuickLZ", "SLZ"};
constexpr std::array<const char*, 7> second_wave = {
    "LZMAT", "LZRW", "WFLZ", "Yappy", "YALZ77", "QuickLZ", "SLZ"};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void roundtrip(const char* label, const Bytes& input, bool require_compressed) {
  const auto wire = native_baseline86_v1::vendored_lightweight::encode(label, input);
  require(!wire.empty(), std::string(label) + ": empty adapter wire");
  if (require_compressed)
    require(wire[0] == 1, std::string(label) +
                                ": repetitive input did not exercise upstream stream; tag=" +
                                std::to_string(wire[0]) + " bytes=" +
                                std::to_string(wire.size()) + " input=" +
                                std::to_string(input.size()));
  require(native_baseline86_v1::vendored_lightweight::decode(label, wire,
                                                              input.size()) == input,
          std::string(label) + ": adapter roundtrip mismatch");
}
}  // namespace

int main() try {
  Bytes repetitive(8192);
  for (std::size_t i = 0; i < repetitive.size(); ++i)
    repetitive[i] = static_cast<std::uint8_t>((i / 64) & 3);
  Bytes pseudo_random(1021);
  std::uint32_t state = 0x12345678U;
  for (auto& byte : pseudo_random) {
    state = state * 1664525U + 1013904223U;
    byte = static_cast<std::uint8_t>(state >> 24);
  }
  const Bytes tiny{0, 1, 2, 3, 4, 5, 6, 7};
  for (const auto* label : labels) {
    roundtrip(label, repetitive, std::string(label) != "Yappy");
    roundtrip(label, pseudo_random, false);
    roundtrip(label, tiny, false);
  }
  roundtrip("Yappy", Bytes(8192, 0), true);
  constexpr std::array<std::size_t, 21> boundary_sizes = {
      1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
      127, 255, 256, 257, 1023, 4095, 8191};
  for (const std::size_t size : boundary_sizes) {
    Bytes patterned(size);
    for (std::size_t i = 0; i < size; ++i)
      patterned[i] = static_cast<std::uint8_t>(((i / 11) ^ (i / 97)) & 0xff);
    for (const auto* label : second_wave) roundtrip(label, patterned, false);
  }
  for (const auto* label : {"FSE", "Huff0"}) {
    auto wire = native_baseline86_v1::vendored_lightweight::encode(label, repetitive);
    require(wire.size() > 2 && wire[0] == 1,
            std::string(label) + ": entropy corruption fixture is not compressed");
    wire.pop_back();
    bool entropy_rejected = false;
    try {
      (void)native_baseline86_v1::vendored_lightweight::decode(
          label, wire, repetitive.size());
    } catch (const std::exception&) {
      entropy_rejected = true;
    }
    require(entropy_rejected,
            std::string(label) + ": truncated entropy stream was accepted");
  }
  bool rejected = false;
  try {
    (void)native_baseline86_v1::vendored_lightweight::decode(
        "FastLZ", Bytes{2, 0}, 1);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "unknown adapter tag was accepted");
  std::cout << "vendored lightweight codec adapters passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
