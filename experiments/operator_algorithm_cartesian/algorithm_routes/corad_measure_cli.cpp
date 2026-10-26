#include "corad_lossless.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using J = nlohmann::json;
using corad_lossless_exact_v1::Block;
using corad_lossless_exact_v1::Bytes;

namespace {
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::uint64_t number(const char* text) {
  std::size_t used = 0;
  const std::string value(text ? text : "");
  const auto result = std::stoull(value, &used);
  if (used != value.size()) fail("invalid unsigned argument");
  return result;
}

Bytes read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open input");
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) fail("cannot size input");
  input.seekg(0);
  Bytes bytes(static_cast<std::size_t>(size));
  if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!input) fail("cannot read input");
  return bytes;
}

void write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) fail("cannot create output");
  if (!bytes.empty()) output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!output) fail("cannot persist output");
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  auto* context = EVP_MD_CTX_new();
  if (!context) fail("cannot allocate SHA-256 context");
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned length = 0;
  const auto* pointer = bytes.empty()
      ? reinterpret_cast<const unsigned char*>("") : bytes.data();
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, pointer, bytes.size()) == 1 &&
      EVP_DigestFinal_ex(context, digest, &length) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || length != 32) fail("SHA-256 failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned index = 0; index < length; ++index)
    output << std::setw(2) << static_cast<unsigned>(digest[index]);
  return output.str();
}

double median(std::vector<std::uint64_t> values) {
  std::sort(values.begin(), values.end());
  return static_cast<double>(values.at(values.size() / 2));
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) fail("expected measure or decode");
  const std::string command = argv[1];
  if (command == "measure") {
    if (argc != 10) fail("measure CORAD DTYPE ROWS VALUES MISSING WARMUPS REPEATS FRAME");
    if (std::string(argv[2]) != "CORAD") fail("this isolated CLI only serves CORAD");
    Block block{argv[3], number(argv[4]), read(argv[5]), read(argv[6])};
    corad_lossless_exact_v1::validate(block);
    const auto warmups = number(argv[7]);
    const auto repeats = number(argv[8]);
    if (warmups != 1 || repeats != 3) fail("frozen contract requires W1/R3");
    auto measurement = corad_lossless_exact_v1::measure(block, warmups, repeats);
    write(argv[9], measurement.frame);
    std::cout << J{{"status", measurement.original_numeric_bytes ? "ok" : "N/A"},
                   {"method", "CORAD"}, {"dtype", block.dtype},
                   {"row_count", block.row_count},
                   {"original_numeric_bytes", measurement.original_numeric_bytes},
                   {"complete_frame_bytes", measurement.frame.size()},
                   {"complete_frame_sha256", sha256(measurement.frame)},
                   {"compression_ns_samples", measurement.compression_ns},
                   {"decompression_ns_samples", measurement.decompression_ns},
                   {"compression_ns", median(measurement.compression_ns)},
                   {"decompression_ns", median(measurement.decompression_ns)},
                   {"profile_version", corad_lossless_exact_v1::profile_version},
                   {"configuration", corad_lossless_exact_v1::configuration},
                   {"timing_scope", "CORAD exact complete frame in-process W1/R3/T1"}}
                     .dump() << '\n';
  } else if (command == "decode") {
    if (argc != 5) fail("decode FRAME VALUES_OUT MISSING_OUT");
    const auto block = corad_lossless_exact_v1::decode(read(argv[2]));
    write(argv[3], block.values);
    write(argv[4], block.missing);
    std::cout << J{{"status", "ok"}, {"dtype", block.dtype},
                   {"row_count", block.row_count},
                   {"values_sha256", sha256(block.values)},
                   {"missing_bitmap_sha256", sha256(block.missing)}}.dump() << '\n';
  } else {
    fail("unknown command");
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
