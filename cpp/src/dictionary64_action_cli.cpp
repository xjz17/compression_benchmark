// Legacy wire uses u64 indices; the existing C++ column codec uses u32.
// Int64 and float64 share this bit-preserving kernel (no numeric conversion).
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "dictionary_recode.hpp"
using Bytes = std::vector<std::uint8_t>;
static void put(Bytes& out, std::uint64_t v, unsigned n) {
  for (unsigned i = 0; i < n; ++i) out.push_back(v >> (i * 8));
}
static std::uint64_t get(const Bytes& in, std::size_t& pos, unsigned n) {
  if (pos > in.size() || n > in.size() - pos) throw std::runtime_error("truncated dictionary64");
  std::uint64_t v = 0;
  for (unsigned i = 0; i < n; ++i) v |= std::uint64_t(in[pos++]) << (i * 8);
  return v;
}
static Bytes encode(const Bytes& in) {
  if (in.size() % 8 || in.size() / 8 > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("dictionary64 requires complete uint32-counted words");
  std::vector<std::uint64_t> words;
  std::size_t pos = 0;
  while (pos < in.size()) {
    words.push_back(get(in, pos, 8));
  }
  const auto encoded = exact_actions::dictionary_recode::encode<std::uint64_t>(
      words, exact_actions::dictionary_recode::Order::first_seen);
  Bytes out; put(out, encoded.dictionary.size(), 4);
  for (auto w : encoded.dictionary) put(out, w, 8);
  put(out, words.size(), 4);
  for (auto id : encoded.ids) put(out, id, 8);
  return out;
}
static Bytes decode(const Bytes& in) {
  std::size_t pos = 0; const auto size = get(in, pos, 4);
  if (size > (in.size() - pos) / 8) throw std::runtime_error("invalid dictionary size");
  std::vector<std::uint64_t> dictionary;
  for (std::uint64_t i = 0; i < size; ++i) dictionary.push_back(get(in, pos, 8));
  const auto count = get(in, pos, 4);
  if (count != (in.size() - pos) / 8 || (in.size() - pos) % 8)
    throw std::runtime_error("dictionary64 index size mismatch");
  std::vector<std::uint64_t> ids;
  for (std::uint64_t i = 0; i < count; ++i) {
    ids.push_back(get(in, pos, 8));
  }
  Bytes out;
  for (const auto word : exact_actions::dictionary_recode::decode<std::uint64_t>(dictionary, ids)) put(out, word, 8);
  return out;
}
int main(int argc, char** argv) try {
  if (argc != 2) throw std::invalid_argument("dictionary64_action_cli encode|decode");
  const Bytes in{std::istreambuf_iterator<char>(std::cin), {}};
  const std::string command = argv[1]; Bytes out;
  if (command == "encode") out = encode(in);
  else if (command == "decode") out = decode(in);
  else throw std::invalid_argument("unknown dictionary64 command");
  std::cout.write(reinterpret_cast<const char*>(out.data()), out.size());
  return std::cout ? 0 : 2;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
