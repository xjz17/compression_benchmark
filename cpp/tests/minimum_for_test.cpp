#include "minimum_for.hpp"
#include <array>
#include <iostream>
#include <random>

template<minimum_for::Word T> void check(const std::vector<T>& input) {
  auto result = minimum_for::encode<T>(input);
  minimum_for::validate_canonical<T>(result.minimum, result.offsets);
  if (minimum_for::decode<T>(result.minimum, result.offsets) != input)
    throw std::runtime_error("FOR round trip failed");
  if (input.size() != result.offsets.size()) throw std::runtime_error("FOR length changed");
  for (std::size_t i = 0; i < input.size(); ++i)
    if (static_cast<__int128>(input[i]) - static_cast<__int128>(result.minimum) != result.offsets[i])
      throw std::runtime_error("FOR offset differs from exact mathematical subtraction");
}
template<minimum_for::Word T> void domain() {
  using U = std::make_unsigned_t<T>;
  check<T>({}); check<T>({0}); check<T>({7,7,7});
  check<T>({std::numeric_limits<T>::min(),std::numeric_limits<T>::max(),0,1});
  auto bounds = minimum_for::encode<T>(std::array<T,2>{std::numeric_limits<T>::min(),std::numeric_limits<T>::max()});
  if constexpr (std::is_signed_v<T>)
    if (bounds.offsets[1] != std::numeric_limits<U>::max()) throw std::runtime_error("full signed domain narrowed");
  std::mt19937_64 rng(191);
  for (int k = 0; k < 80; ++k) {
    std::vector<T> values;
    for (int n = 0; n < k; ++n) {
      const U bits = static_cast<U>(rng());
      if constexpr (std::is_signed_v<T>) values.push_back(std::bit_cast<T>(bits));
      else values.push_back(bits);
    }
    check(values);
  }
  for (const auto& bad : std::vector<std::vector<U>>{{1},{0,std::numeric_limits<U>::max()}}) {
    bool rejected = false;
    try { minimum_for::validate_canonical<T>(T{1}, bad); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("malformed canonical FOR accepted");
  }
}
int main() {
  domain<std::int64_t>(); domain<std::uint64_t>();
  domain<std::int32_t>(); domain<std::uint32_t>();
  std::cout << "minimum_for=PASS signed/unsigned 32/64-bit full domains and exact residuals\n";
}
