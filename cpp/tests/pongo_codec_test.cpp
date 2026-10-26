#include "pongo_codec.hpp"
#include "tsfile_encoding/tsfile_decode_p03.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"

#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

bool same_bits(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (std::bit_cast<std::uint64_t>(left[i]) != std::bit_cast<std::uint64_t>(right[i])) {
      return false;
    }
  }
  return true;
}

bool check_pongo(const std::vector<double>& values, const char* label) {
  const auto encoded = pongo_codec::encode(values);
  std::vector<double> decoded;
  if (!pongo_codec::decode(encoded, decoded) || !same_bits(values, decoded)) {
    std::cerr << "Pongo round trip failed: " << label << '\n';
    return false;
  }
  return true;
}

bool check_pongo_ablation(const std::vector<double>& values,
                          const char* omitted_operator) {
  const auto encoded = pongo_codec::encode_ablation(values, omitted_operator);
  std::vector<double> decoded;
  if (!pongo_codec::decode_ablation(encoded, omitted_operator, decoded) ||
      !same_bits(values, decoded)) {
    std::cerr << "Pongo ablation round trip failed: " << omitted_operator << '\n';
    return false;
  }
  return true;
}

std::string to_hex(const std::vector<std::uint8_t>& bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (std::uint8_t byte : bytes) {
    hex.push_back(digits[byte >> 4]);
    hex.push_back(digits[byte & 0x0f]);
  }
  return hex;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check_pongo({}, "empty");
  ok &= check_pongo(
      {65007.28, 65011.83, 65011.83, 65015.50, 64998.12, 0.03215, 0.03218,
       100.0, 100.5, 100.55, 100.555, -42.125},
      "decimal-native");
  std::vector<double> financial(1000);
  financial[0] = 65000.0;
  for (std::size_t i = 1; i < financial.size(); ++i) {
    const double change = static_cast<double>(static_cast<int>(i % 17) - 8) * 0.01;
    financial[i] = std::round((financial[i - 1] + change) * 100.0) / 100.0;
  }
  ok &= check_pongo(financial, "financial-series");
  ok &= check_pongo_ablation(financial, "eraser");
  ok &= check_pongo_ablation(financial, "xor");
  ok &= check_pongo_ablation(financial, "trailing_zero_pack");
  const auto pongo_financial = pongo_codec::encode(financial);
  const auto gorilla_financial = tsfile::p03::encode_gorilla_double(financial);
  if (pongo_financial.size() >= gorilla_financial.size()) {
    std::cerr << "Pongo did not improve the decimal-series payload: "
              << pongo_financial.size() << " >= " << gorilla_financial.size() << '\n';
    ok = false;
  }
  ok &= check_pongo(
      {0.0, -0.0, std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(),
       std::bit_cast<double>(std::uint64_t{0x7ff8000000000042ull}), 3.141592653589793,
       2.718281828459045},
      "special-and-high-precision");

  const std::vector<double> parity_values = {
      0.0,
      -0.0,
      1.2,
      1.23,
      0.00123,
      -42.125,
      1e20,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::denorm_min(),
  };
  const std::string java_golden =
      "0a0000000000000000000000000000007001f00ebff7982d42c82e8df018ff18bf703a8450f01d78"
      "b58c58efb6bc75e2d6316000800000000001c1ffff80000000000008";
  if (to_hex(pongo_codec::encode(parity_values)) != java_golden) {
    std::cerr << "Pongo C++ payload differs from the Java golden vector\n";
    ok = false;
  }

  std::mt19937_64 random(20260725);
  std::vector<double> random_bits;
  random_bits.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    random_bits.push_back(std::bit_cast<double>(random()));
  }
  ok &= check_pongo(random_bits, "random-bit-patterns");

  const std::vector<std::int64_t> integers = {
      std::numeric_limits<std::int64_t>::min() + 1, -(std::int64_t{1} << 60), -1, 0, 1,
      std::int64_t{1} << 53, (std::int64_t{1} << 60) + 12345,
      std::numeric_limits<std::int64_t>::max()};
  const auto chimp = tsfile::p03::encode_chimp_int64(integers);
  std::vector<std::int64_t> decoded_integers;
  if (!tsfile::p03::decode_chimp_int64(chimp, decoded_integers) ||
      decoded_integers != integers) {
    std::cerr << "CHIMP128 int64 round trip failed\n";
    ok = false;
  }
  return ok ? 0 : 1;
}
