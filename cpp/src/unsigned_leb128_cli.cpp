#include "unsigned_leb128.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>

namespace {
using Bytes = std::vector<std::uint8_t>;
using U64 = std::uint64_t;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void put(Bytes& out, U64 value) {
  for (unsigned i = 0; i < 8; ++i) out.push_back(value >> (8 * i));
}
U64 get(std::span<const std::uint8_t> input, std::size_t offset) {
  require(offset <= input.size() && input.size() - offset >= 8, "truncated uint64");
  U64 value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= U64{input[offset+i]} << (8*i);
  return value;
}
std::vector<U64> words(std::span<const std::uint8_t> input) {
  require(input.size() % 8 == 0, "ULEB128 input must be complete uint64 LE words");
  std::vector<U64> values;
  values.reserve(input.size()/8);
  for (std::size_t i = 0; i < input.size(); i += 8) values.push_back(get(input,i));
  return values;
}
Bytes serialize(std::span<const U64> values) {
  require(values.size() <= std::numeric_limits<std::size_t>::max()/8,
          "uint64 output size overflow");
  Bytes output;
  output.reserve(values.size()*8);
  for (const auto value : values) put(output,value);
  return output;
}
}  // namespace

int main(int argc, char** argv) try {
  require(argc == 2, "expected raw-forward|raw-inverse|packet-forward|packet-inverse|lengths");
  const std::string command = argv[1];
  const Bytes input{std::istreambuf_iterator<char>(std::cin), {}};
  Bytes output;
  namespace leb = compression_operators::unsigned_leb128;
  if (command == "raw-forward" || command == "packet-forward" || command == "lengths") {
    const auto values = words(input);
    if (command == "lengths") {
      for (const auto value : values) output.push_back(leb::encoded_length(value));
    } else {
      auto payload = leb::encode(values);
      if (command == "packet-forward") {
        output = {'U','V','L','1'};
        put(output,values.size());
        put(output,payload.size());
      }
      output.insert(output.end(),payload.begin(),payload.end());
    }
  } else if (command == "raw-inverse") {
    output = serialize(leb::decode(input));
  } else if (command == "packet-inverse") {
    require(input.size() >= 20 && std::equal(input.begin(),input.begin()+4,"UVL1"),
            "invalid unsigned ULEB128 packet");
    const U64 count = get(input,4), payload_size = get(input,12);
    require(payload_size == input.size()-20 && count <= payload_size,
            "invalid unsigned ULEB128 packet lengths");
    const auto values = leb::decode(std::span(input).subspan(20));
    require(values.size() == count,"unsigned ULEB128 value count mismatch");
    output = serialize(values);
  } else {
    throw std::invalid_argument("unknown unsigned ULEB128 command");
  }
  std::cout.write(reinterpret_cast<const char*>(output.data()),output.size());
  require(bool(std::cout),"failed to write stdout");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
