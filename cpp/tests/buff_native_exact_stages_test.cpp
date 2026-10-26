#include "buff_native_exact_stages.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"

#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void check(const std::vector<std::uint64_t>& bits) {
  using namespace exact_actions::buff_native;
  std::vector<std::uint8_t> raw(bits.size() * 8);
  std::vector<double> values(bits.size());
  if (!raw.empty()) {
    std::memcpy(raw.data(), bits.data(), raw.size());
    std::memcpy(values.data(), bits.data(), raw.size());
  }
  const auto words = word_order_forward(raw);
  const auto wire = frame_forward(words);
  require(word_order_inverse(words) == raw, "BUFF native word-order inverse");
  require(frame_inverse(wire) == words, "BUFF native frame inverse");
  require(decompose(wire) == raw, "BUFF native complete roundtrip");
  require(wire == buff_kernel::encode_doubles(values, 10000),
          "BUFF native exact original encoder byte parity");
  const auto decoded = buff_kernel::decode_doubles(wire);
  require(decoded.values.size() == values.size(), "BUFF native original decoder count");
  require(raw.empty() || std::memcmp(decoded.values.data(), raw.data(), raw.size()) == 0,
          "BUFF native original decoder bitwise parity");
}

template <typename F> void reject(F function) {
  bool threw = false;
  try { function(); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, "BUFF native accepted malformed packet");
}

}  // namespace

int main() try {
  check({});
  check({0});
  check({0, 0x8000000000000000ULL, 0x7ff0000000000000ULL,
         0xfff0000000000000ULL, 0x7ff800000000002aULL,
         0x7ff0000000000001ULL, 1, 0x8000000000000001ULL});
  std::mt19937_64 rng(20260905);
  for (const auto count : {2, 7, 79, 80, 81, 128, 2048}) {
    std::vector<std::uint64_t> bits(count);
    for (auto& value : bits) value = rng();
    check(bits);
  }
  using namespace exact_actions::buff_native;
  reject([] { word_order_forward(std::vector<std::uint8_t>(7)); });
  reject([] { frame_forward(std::vector<std::uint8_t>(9)); });
  reject([] { frame_inverse(std::vector<std::uint8_t>(11)); });
  auto wire = compose(std::vector<std::uint8_t>(8));
  auto bad = wire;
  bad[4] = 63;
  reject([&] { frame_inverse(bad); });
  bad = wire;
  bad.pop_back();
  reject([&] { frame_inverse(bad); });
  bad = wire;
  bad.push_back(0);
  reject([&] { frame_inverse(bad); });
  std::cout << "BUFF native: 10 exact source-wire/roundtrip cases and 6 malformed-input cases passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
