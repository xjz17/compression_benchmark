#include "ts2diff_exact_stages.hpp"

#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <class T>
void require_equal(const T& left, const T& right, const char* message) {
  if (left != right) throw std::runtime_error(message);
}

}  // namespace

int main() {
  std::vector<std::vector<std::int64_t>> integer_cases = {
      {}, {7}, {1, 2, 4, 8, 16, 32}, std::vector<std::int64_t>(300, 42)};
  for (std::size_t i = 0; i < 300; ++i) {
    integer_cases.back()[i] = static_cast<std::int64_t>(i * i - 31 * i + (i % 7));
  }
  for (const auto& values : integer_cases) {
    const auto delta = exact_actions::ts2diff::delta_forward_int64(values);
    require_equal(exact_actions::ts2diff::delta_inverse_int64(delta), values,
                  "delta round trip");
    const auto normalized = exact_actions::ts2diff::frame_of_reference_forward(delta);
    require_equal(exact_actions::ts2diff::frame_of_reference_inverse(normalized), delta,
                  "FOR packet round trip");
    const auto composed = exact_actions::ts2diff::bitpack_forward(normalized);
    const auto direct = tsfile::p02::encode_ts_2diff_int64(values);
    require_equal(composed, direct, "integer public-wire parity");
    require_equal(exact_actions::ts2diff::decompose_int64(composed), values,
                  "integer full-chain round trip");
    std::vector<std::int64_t> direct_decoded;
    if (!tsfile::p02::decode_ts_2diff_int64(composed, direct_decoded)) {
      throw std::runtime_error("project integer decoder rejected staged wire");
    }
    require_equal(direct_decoded, values, "project integer decoder parity");
  }

  const std::vector<double> decimals = {-12.34, -1.25, 0.0, 0.01, 7.5, 91.75};
  const auto scaled = exact_actions::ts2diff::scaling_forward(decimals, 2);
  const auto restored_scaled = exact_actions::ts2diff::scaling_inverse(scaled);
  if (restored_scaled.size() != decimals.size()) throw std::runtime_error("scaling size");
  for (std::size_t i = 0; i < decimals.size(); ++i) {
    if (std::bit_cast<std::uint64_t>(restored_scaled[i]) !=
        std::bit_cast<std::uint64_t>(decimals[i])) {
      throw std::runtime_error("scaling bitwise round trip");
    }
  }
  const auto delta = exact_actions::ts2diff::delta_forward_scaled(scaled);
  require_equal(exact_actions::ts2diff::delta_inverse_scaled(delta), scaled,
                "scaled delta packet round trip");
  const auto normalized = exact_actions::ts2diff::frame_of_reference_forward(delta);
  const auto composed = exact_actions::ts2diff::bitpack_forward(normalized);
  const auto direct = tsfile::p02::encode_ts_2diff_double(decimals, 2);
  require_equal(composed, direct, "double public-wire parity");
  const auto decoded = exact_actions::ts2diff::decompose_double(composed, 2);
  if (decoded.size() != decimals.size()) throw std::runtime_error("double decoded size");
  for (std::size_t i = 0; i < decimals.size(); ++i) {
    if (std::bit_cast<std::uint64_t>(decoded[i]) !=
        std::bit_cast<std::uint64_t>(decimals[i])) {
      throw std::runtime_error("double full-chain bitwise round trip");
    }
  }

  bool rejected = false;
  try {
    const std::vector<double> not_exact = {0.1 + 0.2};
    (void)exact_actions::ts2diff::scaling_forward(not_exact, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("non-reversible scaling was not rejected");

  std::cout << "ts2diff_exact_stage_actions=4 status=PASS\n";
  return 0;
}
