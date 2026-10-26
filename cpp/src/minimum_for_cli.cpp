#include "minimum_for.hpp"

#include <cstring>
#include <iostream>
#include <iterator>
#include <string>

namespace {
using Bytes = std::vector<std::uint8_t>;
void put(Bytes& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) out.push_back(value >> shift);
}
std::uint64_t get(const Bytes& in, std::size_t& pos) {
  if (pos > in.size() || in.size() - pos < 8) throw std::invalid_argument("truncated FOR word");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) value |= std::uint64_t{in[pos++]} << shift;
  return value;
}
template<minimum_for::Word T> T from_word(std::uint64_t value) {
  if constexpr (std::same_as<T, std::int64_t>) return std::bit_cast<std::int64_t>(value);
  else return value;
}
template<minimum_for::Word T> Bytes encode(const Bytes& input) {
  if (input.size() % 8) throw std::invalid_argument("FOR requires complete 64-bit values");
  std::vector<T> values; values.reserve(input.size() / 8);
  for (std::size_t pos = 0; pos < input.size();) values.push_back(from_word<T>(get(input, pos)));
  const auto result = minimum_for::encode<T>(values);
  Bytes out{'M','F','R','1',std::same_as<T, std::int64_t> ? std::uint8_t{0} : std::uint8_t{1}};
  put(out, values.size()); put(out, static_cast<std::uint64_t>(result.minimum));
  for (auto value : result.offsets) put(out, value);
  return out;
}
template<minimum_for::Word T> Bytes restore(const Bytes& input, std::size_t count, std::size_t pos) {
  const auto minimum = from_word<T>(get(input, pos));
  std::vector<std::uint64_t> offsets; offsets.reserve(count);
  while (pos < input.size()) offsets.push_back(get(input, pos));
  minimum_for::validate_canonical<T>(minimum, offsets);
  const auto values = minimum_for::decode<T>(minimum, offsets);
  Bytes out; out.reserve(count * 8);
  for (auto value : values) put(out, static_cast<std::uint64_t>(value));
  return out;
}
Bytes decode(const Bytes& input) {
  if (input.size() < 21 || std::memcmp(input.data(), "MFR1", 4) || input[4] > 1)
    throw std::invalid_argument("invalid MFR1 header or integer domain");
  std::size_t pos = 5;
  const auto count = get(input, pos);
  if ((input.size() - 21) % 8 || count != (input.size() - 21) / 8)
    throw std::invalid_argument("FOR count does not match payload length");
  return input[4] == 0 ? restore<std::int64_t>(input, count, pos)
                       : restore<std::uint64_t>(input, count, pos);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("expected encode-int64, encode-uint64 or decode");
  const Bytes input{std::istreambuf_iterator<char>(std::cin), {}};
  const std::string command = argv[1];
  const auto out = command == "encode-int64" ? encode<std::int64_t>(input) :
                   command == "encode-uint64" ? encode<std::uint64_t>(input) :
                   command == "decode" ? decode(input) :
                   throw std::invalid_argument("unknown FOR command");
  std::cout.write(reinterpret_cast<const char*>(out.data()), out.size());
  return std::cout ? 0 : 2;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n'; return 1;
}
