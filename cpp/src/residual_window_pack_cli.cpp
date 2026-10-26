#include "residual_window_pack.hpp"
#include <array>
#include <bit>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
struct Bits {
  std::vector<std::uint8_t> bytes;
  std::size_t pos = 0;
  void write(std::uint64_t value, int width) {
    if (width < 0 || width > 64) throw std::runtime_error("invalid bit width");
    for (int i = width; i-- > 0; ++pos) {
      if (pos % 8 == 0) bytes.push_back(0);
      bytes.back() |= ((value >> i) & 1U) << (7 - pos % 8);
    }
  }
  std::uint64_t read(int width) {
    if (width < 0 || width > 64 || static_cast<std::size_t>(width) > bytes.size() * 8 - pos)
      throw std::runtime_error("truncated window payload");
    std::uint64_t value = 0;
    for (int i = 0; i < width; ++i, ++pos) value = (value << 1) | ((bytes[pos / 8] >> (7 - pos % 8)) & 1U);
    return value;
  }
};
constexpr std::array<int, 8> chimp_buckets{0, 8, 12, 16, 18, 20, 22, 24};
constexpr std::array<int, 8> plus32_buckets{0, 6, 10, 12, 14, 16, 18, 20};
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: residual_window_pack_cli encode|decode chimp32|chimp64|elfplus32|elfplus64");
    const std::string mode = argv[1], profile = argv[2];
    if (mode != "encode" && mode != "decode") throw std::runtime_error("invalid operation");
    if (profile != "chimp32" && profile != "chimp64" && profile != "elfplus32" && profile != "elfplus64")
      throw std::runtime_error("invalid window profile");
    const bool plus = profile.starts_with("elfplus");
    const int width = profile.ends_with("32") ? 32 : 64;
    const int lane_bytes = width / 8;
    const auto& buckets = plus && width == 32 ? plus32_buckets : chimp_buckets;
    const exact_actions::residual_window_pack::ChimpProfile cp{width, 0, width == 32 ? 5 : 6, width == 32 ? 5 : 6};
    const exact_actions::residual_window_pack::ElfPlusProfile ep{width, width == 32 ? 3 : 4, width == 32 ? 5 : 6};
    const std::vector<std::uint8_t> input((std::istreambuf_iterator<char>(std::cin)), {});
    std::vector<std::uint8_t> output;
    int sl = width + 1, st = width + 1;
    if (mode == "encode") {
      if (input.size() % lane_bytes) throw std::runtime_error("unaligned residual input");
      Bits bits;
      for (std::size_t i = 0; i < input.size(); i += lane_bytes) {
        std::uint64_t value = 0;
        for (int j = 0; j < lane_bytes; ++j) value |= std::uint64_t(input[i + j]) << (8 * j);
        const int leading = value == 0 ? width : std::countl_zero(value) - (64 - width);
        const int trailing = value == 0 ? width : std::countr_zero(value);
        unsigned bucket = 0;
        while (bucket + 1 < buckets.size() && buckets[bucket + 1] <= leading) ++bucket;
        auto write = [&](std::uint64_t v, int n) { bits.write(v, n); };
        if (plus) exact_actions::residual_window_pack::elf_plus(write, value, buckets[bucket], trailing, bucket, sl, st, ep);
        else exact_actions::residual_window_pack::chimp(write, value, 0, buckets[bucket], trailing, bucket, sl, cp);
      }
      output = {'R', 'W', 'P', '1', static_cast<std::uint8_t>(width), static_cast<std::uint8_t>(plus)};
      const std::uint64_t count = input.size() / lane_bytes;
      for (int j = 0; j < 8; ++j) output.push_back(count >> (8 * j));
      output.insert(output.end(), bits.bytes.begin(), bits.bytes.end());
    } else {
      if (input.size() < 14 || std::string(input.begin(), input.begin() + 4) != "RWP1" ||
          input[4] != width || input[5] != plus) throw std::runtime_error("wrong RWP1 profile/header");
      std::uint64_t count = 0;
      for (int j = 0; j < 8; ++j) count |= std::uint64_t(input[6 + j]) << (8 * j);
      if (count > (input.size() - 14) * 4) throw std::runtime_error("impossible residual count");
      Bits bits{{input.begin() + 14, input.end()}};
      for (std::uint64_t i = 0; i < count; ++i) {
        auto read = [&](int n) { return bits.read(n); };
        const auto value = plus ? exact_actions::residual_window_pack::elf_plus_inverse(read, buckets, sl, st, ep)
                               : exact_actions::residual_window_pack::chimp_inverse(read, buckets, sl, cp).value;
        for (int j = 0; j < lane_bytes; ++j) output.push_back(value >> (8 * j));
      }
      if ((bits.pos + 7) / 8 != bits.bytes.size()) throw std::runtime_error("trailing window bytes");
      while (bits.pos % 8) if (bits.read(1)) throw std::runtime_error("nonzero window padding");
    }
    std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
    return std::cout ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
