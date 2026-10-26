#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_all() {
  return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
}

void write_bytes(const void* data, std::size_t size) {
  std::cout.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("usage: rle_int64_action_cli encode|decode");
  const std::string command = argv[1];
  const auto raw = read_all();
  if (command == "encode") {
    if (raw.size() % sizeof(std::int64_t) != 0) {
      throw std::runtime_error("int64 input byte count mismatch");
    }
    std::vector<std::int64_t> values(raw.size() / sizeof(std::int64_t));
    if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
    const auto encoded = tsfile::p02::encode_rle_int64_legacy(values);
    write_bytes(encoded.data(), encoded.size());
  } else if (command == "decode") {
    std::vector<std::int64_t> values;
    if (!tsfile::p02::decode_rle_int64(raw, values)) {
      throw std::runtime_error("invalid TsFile int64 RLE wire");
    }
    write_bytes(values.data(), values.size() * sizeof(std::int64_t));
  } else {
    throw std::invalid_argument("unknown command: " + command);
  }
  if (!std::cout) throw std::runtime_error("failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
