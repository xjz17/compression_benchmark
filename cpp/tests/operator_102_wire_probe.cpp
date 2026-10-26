#include "elf_double_codec.hpp"
#include "elf_float_codec.hpp"
#include "camel_exact_stages.hpp"
#include "chimp128_exact_stages.hpp"
#include <bit>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::filesystem::create_directories(argv[1]);
  std::mt19937_64 random(102);
  for (unsigned sample = 0; sample < 12; ++sample) {
    std::vector<double> values;
    const unsigned n = sample < 4 ? sample : (sample == 11 ? 2050 : 128 + sample);
    for (unsigned i = 0; i < n; ++i) {
      double x = static_cast<double>(static_cast<int>(random() % 20000) - 10000) / 16;
      if (sample == 11) x = (i % 31) * 0.25;
      if (i % 7 == 0) x = 0.0;
      if (i % 11 == 0) x = -0.0;
      if (i % 13 == 0 && !values.empty()) x = values.back();
      values.push_back(x);
    }
    std::vector<std::int64_t> bits;
    std::vector<std::uint8_t> raw;
    for (double v : values) {
      auto word = std::bit_cast<std::uint64_t>(v);
      bits.push_back(std::bit_cast<std::int64_t>(v));
      for (unsigned j = 0; j < 8; ++j) raw.push_back(word >> (8 * j));
    }
    auto save = [&](const char* name, const auto& wire) {
      std::ofstream out(std::filesystem::path(argv[1]) /
          (std::string(name) + "_" + std::to_string(sample) + ".bin"), std::ios::binary);
      out.write(reinterpret_cast<const char*>(wire.data()), wire.size());
    };
    for (bool plus : {false, true}) {
      auto w64 = elf_double::encodeDoubleColumnExact(raw, plus);
      if (elf_double::decodeDoubleColumnExact(w64, plus) != raw)
        throw std::runtime_error("Elf64 roundtrip");
      save(plus ? "elfplus64" : "elf64", w64);
      auto w32 = plus ? exact_actions::elf::compose_elf_plus(bits) : exact_actions::elf::compose_elf(bits);
      auto decoded = plus ? exact_actions::elf::decompose_elf_plus(w32) : exact_actions::elf::decompose_elf(w32);
      if (decoded != bits) throw std::runtime_error("Elf32 roundtrip");
      save(plus ? "elfplus32" : "elf32", w32);
    }
    // Camel's decimal/literal transitions share state across every record.
    auto cw = exact_actions::camel::compose(bits);
    if (exact_actions::camel::decompose(cw) != bits) throw std::runtime_error("Camel roundtrip");
    save("camel", cw);
    if (!bits.empty()) {
      auto hw = exact_actions::chimp128::compose_double_bits(bits);
      if (exact_actions::chimp128::decompose_double_bits(hw) != bits) throw std::runtime_error("Chimp roundtrip");
      save("chimp128", hw);
    }
  }
}
