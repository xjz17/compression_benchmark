// Probability-only protocol over the shared actual cmix v21 adapters.
// This executable is a GPL-3.0 combined work; upstream PAQ code is GPL-2.0-or-later.
// Preserve external/compression/cmix/COPYING and the upstream source notices.
// These are related-work models, NOT OpenZL built-in codecs or reversible codecs.
#include "prediction_probability_model.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using prediction_probability::Options;
using prediction_probability::ProbabilityModel;

namespace {
Options parse_options(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("expected provider: ppmd, dmc, paq, cmix");
  Options o;
  o.provider = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--allow-large-memory") { o.allow_large_memory = true; continue; }
    if (i + 1 >= argc) throw std::runtime_error("missing option value");
    std::string value = argv[++i];
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error("option values must be unsigned decimal integers");
    unsigned long n = std::stoul(value);
    if (n > std::numeric_limits<unsigned>::max()) throw std::runtime_error("option too large");
    if (key == "--ppmd-order") o.ppmd_order = static_cast<unsigned>(n);
    else if (key == "--ppmd-memory-mib") o.ppmd_memory_mib = static_cast<unsigned>(n);
    else if (key == "--paq-level") o.paq_level = static_cast<unsigned>(n);
    else throw std::runtime_error("unknown option: " + key);
  }
  prediction_probability::validate_options(o);
  return o;
}

void write_f64(double value) {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                "requires IEEE-754 binary64");
  std::uint64_t bits;
  std::memcpy(&bits, &value, 8);
  char bytes[8];
  for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>(bits >> (8 * i));
  std::cout.write(bytes, 8);
}

std::uint64_t read_u64() {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) {
    int c = std::cin.get();
    if (c == EOF) throw std::runtime_error("truncated length");
    value |= std::uint64_t(static_cast<unsigned char>(c)) << (8 * i);
  }
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    ProbabilityModel model(parse_options(argc, argv));
    for (int command; (command = std::cin.get()) != EOF;) {
      if (command == 'Q') break;
      if (command == 'P') {
        write_f64(model.predict());
      } else if (command == 'O') {
        int bit = std::cin.get();
        if (bit == EOF) throw std::runtime_error("missing observation");
        model.observe(static_cast<unsigned>(bit));
      } else if (command == 'D') {
        const auto& distribution = model.byte_distribution();
        for (unsigned i = 0; i < 256; ++i) write_f64(distribution[i]);
      } else if (command == 'T') {
        const std::uint64_t size = read_u64();
        if (size > 16777216) throw std::runtime_error("trace limited to 16 MiB input per request");
        for (std::uint64_t i = 0; i < size; ++i) {
          int c = std::cin.get();
          if (c == EOF) throw std::runtime_error("truncated trace input");
          for (int shift = 7; shift >= 0; --shift) {
            write_f64(model.predict());
            model.observe((c >> shift) & 1);
          }
        }
      } else {
        throw std::runtime_error("unknown protocol command");
      }
      std::cout.flush();
      if (!std::cout) throw std::runtime_error("probability output failed");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prediction_probability: " << error.what() << '\n';
    return 2;
  }
}
