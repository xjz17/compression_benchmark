#include "nncp_probability.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace {
constexpr std::uint64_t kLimit = NNCP_PROBABILITY_SYMBOL_LIMIT;
using Bytes = std::vector<unsigned char>;
using File = std::unique_ptr<FILE, decltype(&fclose)>;

void require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
unsigned byte(FILE* file = stdin) {
  const int result = fgetc(file);
  require(result != EOF, "truncated NNCP command or frame");
  return static_cast<unsigned>(result);
}
std::uint64_t read_integer(unsigned width, FILE* file = stdin) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < width; ++i) value |= std::uint64_t(byte(file)) << (8 * i);
  return value;
}
void write_integer(std::uint64_t value, unsigned width, FILE* file = stdout) {
  for (unsigned i = 0; i < width; ++i) require(fputc(value >> (8 * i) & 255, file) != EOF, "NNCP write failed");
}
File temporary() {
  File result(tmpfile(), fclose);
  require(result != nullptr, "NNCP temporary file failed");
  return result;
}
Bytes read_all(FILE* file, std::size_t limit) {
  Bytes result;
  for (int next; (next = fgetc(file)) != EOF;) {
    require(result.size() < limit, "NNCP bounded input limit exceeded");
    result.push_back(static_cast<unsigned char>(next));
  }
  require(!ferror(file), "NNCP read failed");
  return result;
}
void write_all(FILE* file, const Bytes& data) {
  require(fwrite(data.data(), 1, data.size(), file) == data.size(), "NNCP write failed");
}

void coding(bool decode) {
  auto input = temporary();
  auto output = temporary();
  auto data = read_all(stdin, decode ? 1024 * 1024 : kLimit);
  std::uint64_t count = data.size();
  std::uint32_t expected_crc = 0;
  if (decode) {
    require(data.size() >= 24 && std::memcmp(data.data(), "NNP1", 4) == 0,
            "invalid NNCP frame magic");
    write_all(input.get(), data);
    rewind(input.get());
    require(read_integer(4, input.get()) == 0x31504e4e, "invalid NNCP magic");
    require(read_integer(4, input.get()) == 1, "unsupported NNCP streaming profile");
    count = read_integer(8, input.get());
    expected_crc = read_integer(4, input.get());
    const auto body_size = read_integer(4, input.get());
    require(count <= kLimit && body_size == data.size() - 24, "NNCP frame length mismatch");
    require(nncp_probability_decode(input.get(), output.get(), count) == 0, "NNCP decode failed");
    rewind(output.get());
    auto raw = read_all(output.get(), kLimit);
    require(raw.size() == count && crc32(0, raw.data(), raw.size()) == expected_crc,
            "NNCP decoded checksum mismatch");
    write_all(stdout, raw);
  } else {
    write_all(input.get(), data);
    rewind(input.get());
    require(nncp_probability_encode(input.get(), output.get(), count) == 0, "NNCP encode failed");
    rewind(output.get());
    const auto body = read_all(output.get(), 1024 * 1024);
    write_integer(0x31504e4e, 4);
    write_integer(1, 4);
    write_integer(count, 8);
    write_integer(crc32(0, data.data(), data.size()), 4);
    write_integer(body.size(), 4);
    write_all(stdout, body);
  }
}

void interactive() {
  std::unique_ptr<NNCPProbability, decltype(&nncp_probability_destroy)> state(
      nncp_probability_create(), nncp_probability_destroy);
  require(state != nullptr, "NNCP model allocation failed");
  std::uint64_t observed = 0;
  const auto predict = [&] {
    require(observed < kLimit, "NNCP session symbol limit exceeded");
    std::array<float, 256> values;
    require(nncp_probability_predict(state.get(), values.data()) == 0, "invalid NNCP probabilities");
    for (float value : values) write_integer(std::bit_cast<std::uint32_t>(value), 4);
  };
  for (int command; (command = fgetc(stdin)) != EOF;) {
    if (command == 'Q') return;
    if (command == 'D') {
      predict();
    } else if (command == 'O') {
      require(observed < kLimit, "NNCP session symbol limit exceeded");
      require(nncp_probability_observe(state.get(), byte()) == 0, "predict must precede observe");
      ++observed;
      write_integer(1, 1);
    } else if (command == 'T') {
      const auto count = read_integer(8);
      require(count <= kLimit - observed, "NNCP trace symbol limit exceeded");
      for (std::uint64_t i = 0; i < count; ++i) {
        predict();
        require(nncp_probability_observe(state.get(), byte()) == 0, "NNCP observe failed");
        ++observed;
      }
    } else {
      throw std::runtime_error("unknown NNCP command; use D/O/T/Q symbol protocol");
    }
    require(fflush(stdout) == 0, "NNCP flush failed");
  }
}
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "encode") coding(false);
    else if (argc == 2 && std::string(argv[1]) == "decode") coding(true);
    else if (argc == 1) interactive();
    else throw std::runtime_error("usage: nncp_probability_cli [encode|decode]");
    return 0;
  } catch (const std::exception& error) {
    fprintf(stderr, "NNCP: %s\n", error.what());
    return 1;
  }
}
