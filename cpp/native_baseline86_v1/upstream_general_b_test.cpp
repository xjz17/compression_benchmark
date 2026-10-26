#include "upstream_general_b.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using native_baseline86_v1::upstream_general_b::Bytes;

namespace {
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

Bytes read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read cold-decode fixture");
  return Bytes(std::istreambuf_iterator<char>(stream), {});
}

void write_file(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("cannot write cold-decode fixture");
}

std::string shell_quote(std::string_view value) {
  std::string result("'");
  for (const char ch : value) result += ch == '\'' ? "'\\''" : std::string(1, ch);
  return result + "'";
}
}

int main(int argc, char** argv) {
  try {
    if (argc == 5 && std::string_view(argv[1]) == "--cold-decode") {
      const Bytes payload = read_file(argv[3]);
      const Bytes original = read_file(argv[4]);
      const Bytes decoded = native_baseline86_v1::upstream_general_b::decode(
          argv[2], payload, original.size());
      require(decoded == original, std::string(argv[2]) + " cold decode byte mismatch");
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--in-process-roundtrip") {
      Bytes original(32768);
      for (std::size_t i = 0; i < original.size(); ++i)
        original[i] = static_cast<std::uint8_t>((i / 97) & 7);
      const auto encoded = native_baseline86_v1::upstream_general_b::encode(argv[2], original);
      const auto decoded = native_baseline86_v1::upstream_general_b::decode(
          argv[2], encoded, original.size());
      require(decoded == original, std::string(argv[2]) + " in-process byte mismatch");
      return 0;
    }
    require(argc == 1, "invalid test arguments");
    Bytes compressible(32768);
    for (std::size_t i = 0; i < compressible.size(); ++i)
      compressible[i] = static_cast<std::uint8_t>((i / 97) & 7);
    Bytes incompressible(4096);
    std::uint32_t state = 0x12345678U;
    for (auto& value : incompressible) {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      value = static_cast<std::uint8_t>(state);
    }
    const auto list = native_baseline86_v1::upstream_general_b::descriptors();
    require(list.size() == 10, "descriptor count changed");
    std::size_t ordinal = 0;
    for (const auto& descriptor : list) {
      for (const Bytes* original : {&compressible, &incompressible}) {
        const auto encoded = native_baseline86_v1::upstream_general_b::encode(descriptor.label, *original);
        if (original == &compressible)
          require(encoded.front() == 1,
                  std::string(descriptor.label) + " did not exercise its upstream stream");
        const auto decoded = native_baseline86_v1::upstream_general_b::decode(
            descriptor.label, encoded, original->size());
        require(decoded == *original, std::string(descriptor.label) + " round trip failed");
        bool wrong_length_rejected = false;
        try {
          (void)native_baseline86_v1::upstream_general_b::decode(
              descriptor.label, encoded, original->size() + 1);
        } catch (const std::exception&) {
          wrong_length_rejected = true;
        }
        require(wrong_length_rejected,
                std::string(descriptor.label) + " accepted a wrong expected length");

        if (original == &compressible) {
          const std::string stem = "baseline86-general-b-" + std::to_string(getpid()) +
              "-" + std::to_string(ordinal++);
          const auto payload_path = std::filesystem::temp_directory_path() / (stem + ".payload");
          const auto original_path = std::filesystem::temp_directory_path() / (stem + ".original");
          write_file(payload_path, encoded);
          write_file(original_path, *original);
          const std::string command = shell_quote(argv[0]) + " --cold-decode " +
              shell_quote(descriptor.label) + " " + shell_quote(payload_path.string()) + " " +
              shell_quote(original_path.string());
          const int status = std::system(command.c_str());
          std::filesystem::remove(payload_path);
          std::filesystem::remove(original_path);
          require(status == 0, std::string(descriptor.label) + " independent cold decode failed");
        }
      }
      const Bytes empty;
      const auto encoded_empty = native_baseline86_v1::upstream_general_b::encode(descriptor.label, empty);
      require(native_baseline86_v1::upstream_general_b::decode(descriptor.label, encoded_empty, 0).empty(),
              std::string(descriptor.label) + " empty round trip failed");
    }
    bool rejected = false;
    try { (void)native_baseline86_v1::upstream_general_b::decode("ZXC", Bytes{7}, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid adapter tag was accepted");
    std::cout << "upstream_general_b_roundtrip=PASS labels=" << list.size() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
