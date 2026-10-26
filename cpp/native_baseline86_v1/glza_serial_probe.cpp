#include "upstream_general_a.hpp"
#if defined(GLZA_SERIAL_SCHEDULER_PROBE)
#include "glza_serial_scheduler.h"
#endif

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using native_baseline86_v1::upstream_general_a::Bytes;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Bytes input_for(const std::string& profile) {
  constexpr std::size_t size = 64 * 1024;
  Bytes input(size);
  if (profile == "repeated") {
    for (std::size_t i = 0; i < size; ++i)
      input[i] = static_cast<std::uint8_t>((i / 64) & 7U);
  } else if (profile == "random") {
    std::uint32_t state = 0x8d12e3a7U;
    for (auto& byte : input) {
      state = state * 1664525U + 1013904223U;
      byte = static_cast<std::uint8_t>(state >> 24);
    }
  } else if (profile == "text") {
    static constexpr char sentence[] =
        "GLZA cooperative single-thread validation keeps the upstream "
        "grammar stages and their producer consumer ordering.\n";
    for (std::size_t i = 0; i < size; ++i)
      input[i] = static_cast<std::uint8_t>(sentence[i % (sizeof(sentence) - 1)]);
  } else {
    throw std::runtime_error("unknown fixture profile");
  }
  return input;
}

Bytes read_file(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(input.good(), std::string("cannot open ") + path);
  const auto end = input.tellg();
  require(end >= 0, "cannot size input file");
  Bytes bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!bytes.empty())
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  require(input.good(), std::string("cannot read ") + path);
  return bytes;
}

void write_file(const char* path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.good(), std::string("cannot open ") + path);
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  require(output.good(), std::string("cannot write ") + path);
}
}  // namespace

int main(int argc, char** argv) try {
  using namespace native_baseline86_v1::upstream_general_a;
  if (argc == 5 && std::string(argv[1]) == "encode") {
    const auto raw = input_for(argv[2]);
    const auto encoded = encode("GLZA", raw);
    require(decode("GLZA", encoded, raw.size()) == raw,
            "same-process roundtrip mismatch");
    write_file(argv[3], raw);
    write_file(argv[4], encoded);
    std::cout << "PASS encode profile=" << argv[2]
              << " raw=" << raw.size() << " frame=" << encoded.size()
              << " tag=" << static_cast<unsigned>(encoded.at(0));
#if defined(GLZA_SERIAL_SCHEDULER_PROBE)
    glza_serial_scheduler_stats stats{};
    require(glza_serial_scheduler_last_stats(&stats) == 0,
            "missing serial scheduler statistics");
    std::cout << " tasks=" << stats.tasks_created
              << " switches=" << stats.context_switches
              << " max_live=" << stats.maximum_live_tasks
              << " stack_bytes=" << stats.task_stack_bytes;
#endif
    std::cout << '\n';
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "decode") {
    const auto raw = read_file(argv[2]);
    const auto encoded = read_file(argv[3]);
    require(decode("GLZA", encoded, raw.size()) == raw,
            "cold decode mismatch");
    std::cout << "PASS cold_decode raw=" << raw.size()
              << " frame=" << encoded.size() << '\n';
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "stress") {
    const auto raw = input_for(argv[2]);
    const auto repeats = std::strtoul(argv[3], nullptr, 10);
    require(repeats >= 1 && repeats <= 1000, "invalid stress repeat count");
    Bytes reference;
    for (unsigned long run = 0; run < repeats; ++run) {
      const auto encoded = encode("GLZA", raw);
      require(decode("GLZA", encoded, raw.size()) == raw,
              "stress roundtrip mismatch");
      if (run == 0)
        reference = encoded;
      else
        require(encoded == reference, "stress determinism mismatch");
    }
    std::cout << "PASS stress profile=" << argv[2]
              << " runs=" << repeats << " frame=" << reference.size()
              << '\n';
    return 0;
  }
  throw std::runtime_error(
      "usage: glza_serial_probe encode PROFILE RAW FRAME | decode RAW FRAME | "
      "stress PROFILE RUNS");
} catch (const std::exception& error) {
  std::cerr << "FAIL " << error.what() << '\n';
  return 1;
}
