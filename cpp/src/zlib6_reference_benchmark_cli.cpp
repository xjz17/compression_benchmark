// Complete-frame zlib level-6 reference and in-process timing benchmark.
//
// The benchmark timer excludes process startup and stdin/stdout transfer.  Its
// encoded frame includes the original length and CRC32 so `decode` can run in a
// fresh process with no encoder-side state.
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Bytes = std::vector<std::uint8_t>;
constexpr std::string_view kMagic = "ZR61";
constexpr std::uint8_t kVersion = 1;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

void append_u32(Bytes& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(Bytes& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t read_u32(const Bytes& input, std::size_t& offset) {
  require(offset + 4 <= input.size(), "truncated zlib6 frame");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= std::uint32_t(input[offset++]) << shift;
  return value;
}

std::uint64_t read_u64(const Bytes& input, std::size_t& offset) {
  require(offset + 8 <= input.size(), "truncated zlib6 frame");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= std::uint64_t(input[offset++]) << shift;
  return value;
}

std::uint32_t checksum(const Bytes& input) {
  require(input.size() <= std::numeric_limits<uInt>::max(),
          "zlib6 reference input exceeds zlib checksum limit");
  return static_cast<std::uint32_t>(
      crc32(0, reinterpret_cast<const Bytef*>(input.data()),
            static_cast<uInt>(input.size())));
}

Bytes encode_frame(const Bytes& raw) {
  require(!raw.empty(), "zlib6 reference input must be non-empty");
  require(raw.size() <= std::numeric_limits<uLong>::max(),
          "zlib6 reference input is too large");
  uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
  Bytes compressed(compressed_size);
  const int status = compress2(
      compressed.data(), &compressed_size, raw.data(),
      static_cast<uLong>(raw.size()), 6);
  require(status == Z_OK, "zlib6 reference compression failed");
  compressed.resize(compressed_size);

  Bytes frame;
  frame.reserve(4 + 1 + 8 + 4 + 8 + compressed.size());
  frame.insert(frame.end(), kMagic.begin(), kMagic.end());
  frame.push_back(kVersion);
  append_u64(frame, raw.size());
  append_u32(frame, checksum(raw));
  append_u64(frame, compressed.size());
  frame.insert(frame.end(), compressed.begin(), compressed.end());
  return frame;
}

Bytes decode_frame(const Bytes& frame) {
  require(frame.size() >= 25, "truncated zlib6 frame");
  require(std::equal(kMagic.begin(), kMagic.end(), frame.begin()),
          "invalid zlib6 frame magic");
  std::size_t offset = 4;
  require(frame[offset++] == kVersion, "unsupported zlib6 frame version");
  const auto raw_size64 = read_u64(frame, offset);
  const auto expected_crc = read_u32(frame, offset);
  const auto compressed_size64 = read_u64(frame, offset);
  require(raw_size64 > 0 && raw_size64 <= std::numeric_limits<uLongf>::max(),
          "invalid zlib6 frame raw size");
  require(compressed_size64 <= frame.size() - offset,
          "truncated zlib6 frame payload");
  require(compressed_size64 == frame.size() - offset,
          "trailing bytes after zlib6 frame");
  require(compressed_size64 <= std::numeric_limits<uLong>::max(),
          "zlib6 frame payload is too large");

  Bytes raw(static_cast<std::size_t>(raw_size64));
  uLongf actual_size = static_cast<uLongf>(raw.size());
  const int status = uncompress(
      raw.data(), &actual_size, frame.data() + offset,
      static_cast<uLong>(compressed_size64));
  require(status == Z_OK && actual_size == raw.size(),
          "zlib6 reference decompression failed");
  require(checksum(raw) == expected_crc, "zlib6 frame CRC32 mismatch");
  return raw;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
  require(!values.empty(), "cannot take an empty timing median");
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2) return values[middle];
  return values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

int parse_positive(const char* text, std::string_view label) {
  std::size_t used = 0;
  const auto value = std::stoi(text, &used);
  require(used == std::strlen(text) && value > 0 && value <= 1001,
          std::string(label) + " must be in 1..1001");
  return value;
}

void print_samples(const std::vector<std::uint64_t>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void benchmark(const Bytes& raw, int warmups, int repeats) {
  Bytes frame;
  for (int index = 0; index < warmups; ++index) {
    frame = encode_frame(raw);
    require(decode_frame(frame) == raw, "zlib6 warmup round trip mismatch");
  }
  std::vector<std::uint64_t> encode_ns, decode_ns;
  encode_ns.reserve(repeats);
  decode_ns.reserve(repeats);
  for (int index = 0; index < repeats; ++index) {
    const auto encode_started = std::chrono::steady_clock::now();
    const auto candidate = encode_frame(raw);
    const auto encode_stopped = std::chrono::steady_clock::now();
    if (!frame.empty()) require(candidate == frame, "zlib6 frame is nondeterministic");
    frame = candidate;
    encode_ns.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            encode_stopped - encode_started).count()));

    const auto decode_started = std::chrono::steady_clock::now();
    const auto restored = decode_frame(frame);
    const auto decode_stopped = std::chrono::steady_clock::now();
    require(restored == raw, "zlib6 benchmark round trip mismatch");
    decode_ns.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            decode_stopped - decode_started).count()));
  }
  std::cout << "{\"schema\":\"zlib6-reference-benchmark-v1\","
            << "\"zlib_runtime\":\"" << zlibVersion() << "\","
            << "\"zlib_compile\":\"" << ZLIB_VERSION << "\","
            << "\"level\":6,\"raw_bytes\":" << raw.size()
            << ",\"complete_frame_bytes\":" << frame.size()
            << ",\"warmups\":" << warmups << ",\"repeats\":" << repeats
            << ",\"compression_median_ns\":" << median(encode_ns)
            << ",\"decompression_median_ns\":" << median(decode_ns)
            << ",\"compression_samples_ns\":";
  print_samples(encode_ns);
  std::cout << ",\"decompression_samples_ns\":";
  print_samples(decode_ns);
  std::cout << ",\"timing_boundary\":\"in-process complete ZR61 frame; excludes process startup and stdin/stdout transfer\"}\n";
}
}  // namespace

int main(int argc, char** argv) try {
  require(argc >= 2, "expected encode, decode, benchmark, or version");
  const std::string command = argv[1];
  Bytes input{std::istreambuf_iterator<char>(std::cin), {}};
  if (command == "encode") {
    require(argc == 2, "encode takes no options");
    const auto output = encode_frame(input);
    std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  } else if (command == "decode") {
    require(argc == 2, "decode takes no options");
    const auto output = decode_frame(input);
    std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
  } else if (command == "benchmark") {
    int warmups = 3, repeats = 15;
    for (int index = 2; index < argc; index += 2) {
      require(index + 1 < argc, "benchmark option requires a value");
      const std::string option = argv[index];
      if (option == "--warmups") warmups = parse_positive(argv[index + 1], "warmups");
      else if (option == "--repeats") repeats = parse_positive(argv[index + 1], "repeats");
      else throw std::invalid_argument("unknown benchmark option");
    }
    benchmark(input, warmups, repeats);
  } else if (command == "version") {
    require(argc == 2 && input.empty(), "version takes no options or stdin");
    std::cout << "zlib6-reference-benchmark-v1 zlib=" << zlibVersion()
              << " compile-zlib=" << ZLIB_VERSION << " level=6\n";
  } else {
    throw std::invalid_argument("unknown zlib6 reference command");
  }
  if (!std::cout) throw std::runtime_error("zlib6 reference output write failed");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
