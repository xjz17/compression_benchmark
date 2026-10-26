#include "dbp_exact_stages.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint8_t> raw_doubles(const std::vector<double>& values) {
  std::vector<std::uint8_t> out;
  out.reserve(values.size() * 8);
  for (const auto value : values) {
    const auto bytes = std::bit_cast<std::array<std::uint8_t, 8>>(value);
    out.insert(out.end(), bytes.begin(), bytes.end());
  }
  return out;
}

bool encoder_accepts(const std::vector<double>& values, int max_point) {
  try {
    const auto encoded = exact_actions::dbp::scaling_forward(values, max_point);
    const auto decoded = exact_actions::dbp::scaling_inverse(encoded);
    if (decoded.size() != values.size()) return false;
    for (std::size_t i = 0; i < values.size(); ++i)
      if (std::bit_cast<std::uint64_t>(decoded[i]) !=
          std::bit_cast<std::uint64_t>(values[i]))
        return false;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

void verify_predicate_parity(const std::vector<double>& values) {
  using namespace exact_actions::dbp;
  const auto all = scaling_feasibility(values);
  const auto raw = raw_doubles(values);
  const auto bytes = scaling_feasibility_le64(raw);
  if (all.legal_max_points != bytes.legal_max_points)
    throw std::runtime_error("DBP span/LE64 feasibility mismatch");
  if (all.input_value_visits > values.size() * 2 ||
      bytes.input_value_visits > values.size() * 2)
    throw std::runtime_error("DBP feasibility rescanned a value more than twice");
  for (int max_point = 0; max_point <= maximum_scaling_point; ++max_point) {
    const bool encoded = encoder_accepts(values, max_point);
    if (can_scale_bitwise_exact(values, max_point) != encoded ||
        all.legal(max_point) != encoded)
      throw std::runtime_error("DBP feasibility disagrees with encoder");
  }
}

std::uint64_t splitmix64(std::uint64_t& state) {
  state += UINT64_C(0x9e3779b97f4a7c15);
  auto value = state;
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

void verify_random_predicate_parity() {
  std::uint64_t state = UINT64_C(0x4d595df4d0f33173);
  for (std::size_t case_index = 0; case_index < 1200; ++case_index) {
    std::vector<double> values;
    const auto count = 1 + (splitmix64(state) % 37);
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto selector = splitmix64(state) & 7;
      if (selector == 0) {
        values.push_back(static_cast<double>(
            static_cast<std::int64_t>(splitmix64(state) % 2000001) -
            1000000) / 1000.0);
      } else if (selector == 1 && !values.empty()) {
        values.push_back(values[splitmix64(state) % values.size()]);
      } else {
        values.push_back(std::bit_cast<double>(splitmix64(state)));
      }
    }
    verify_predicate_parity(values);
  }
}

void run_benchmark() {
  constexpr std::size_t count = 131072;
  constexpr int repetitions = 15;
  std::array<std::pair<const char*, std::vector<double>>, 3> workloads;
  workloads[0].first = "decimal_uniqueish";
  workloads[1].first = "repeated_32";
  workloads[2].first = "monotonic_unique";
  for (auto& workload : workloads) workload.second.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    workloads[0].second.push_back(
        static_cast<double>((i * 104729) % 10000019) / 1000.0);
    workloads[1].second.push_back(
        static_cast<double>(static_cast<std::int64_t>(i % 32) - 16) / 100.0);
    workloads[2].second.push_back(static_cast<double>(i) / 100.0);
  }
  std::uint32_t checksum = 0;
  for (const auto& [name, values] : workloads) {
    for (int warmup = 0; warmup < 3; ++warmup)
      checksum ^= exact_actions::dbp::scaling_feasibility(values)
                      .legal_max_points;
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      const auto begin = std::chrono::steady_clock::now();
      checksum ^= exact_actions::dbp::scaling_feasibility(values)
                      .legal_max_points;
      const auto elapsed = std::chrono::duration_cast<
          std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin)
                               .count();
      samples.push_back(static_cast<double>(elapsed) /
                        static_cast<double>(values.size() * sizeof(double)));
    }
    std::sort(samples.begin(), samples.end());
    std::cout << name << " median_ns_per_byte="
              << samples[samples.size() / 2] << '\n';
  }
  std::cout << "benchmark_checksum=" << checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
    run_benchmark();
    return 0;
  }
  using namespace exact_actions::dbp;
  const std::vector<std::int64_t> ints = {
      std::numeric_limits<std::int64_t>::min(), -9, -9, -3, 0, 1, 17,
      std::numeric_limits<std::int64_t>::max()};
  const auto delta = delta_forward_int64(ints);
  if (delta_inverse_int64(delta) != ints) throw std::runtime_error("DBP delta round trip");
  const auto zz = zigzag_forward(delta);
  if (zigzag_inverse(zz) != delta) throw std::runtime_error("DBP ZigZag round trip");
  const auto wire = varint_forward(zz);
  if (varint_inverse(wire) != zz) throw std::runtime_error("DBP varint round trip");
  if (decompose_int64(wire) != ints) throw std::runtime_error("DBP int chain round trip");

  const std::vector<double> doubles = {-19.25, -1.0, 0.0, 0.125, 3.5, 17.25};
  const auto scaled = scaling_forward(doubles, 3);
  if (scaling_inverse(scaled) != doubles) throw std::runtime_error("DBP scaling round trip");
  const auto float_wire = compose_double(doubles, 3);
  const auto restored = decompose_double(float_wire);
  if (restored.size() != doubles.size()) throw std::runtime_error("DBP double count");
  for (std::size_t i = 0; i < doubles.size(); ++i)
    if (std::bit_cast<std::uint64_t>(restored[i]) != std::bit_cast<std::uint64_t>(doubles[i]))
      throw std::runtime_error("DBP double bitwise round trip");

  verify_predicate_parity({});
  verify_predicate_parity(doubles);
  verify_predicate_parity({0.0, 1.0, -1.0,
                           std::numeric_limits<double>::max()});
  verify_predicate_parity({-0.0});
  verify_predicate_parity({std::numeric_limits<double>::quiet_NaN()});
  verify_predicate_parity({std::numeric_limits<double>::infinity()});
  verify_predicate_parity({1.0e20, -1.0e20});
  verify_predicate_parity({std::nextafter(1.0, 2.0), 0.1, -1234.5678});
  std::vector<double> repeated(4096);
  std::vector<double> high_cardinality(4096);
  for (std::size_t i = 0; i < repeated.size(); ++i) {
    repeated[i] = static_cast<double>(
        static_cast<std::int64_t>(i % 32) - 16) / 100.0;
    high_cardinality[i] =
        static_cast<double>((i * 104729) % 10000019) / 1000.0;
  }
  verify_predicate_parity(repeated);
  verify_predicate_parity(high_cardinality);
  verify_random_predicate_parity();
  if (can_scale_bitwise_exact(doubles, -1) ||
      can_scale_bitwise_exact(doubles, maximum_scaling_point + 1))
    throw std::runtime_error("DBP feasibility accepted invalid max_point");
  auto malformed = raw_doubles(doubles);
  malformed.pop_back();
  if (scaling_feasibility_le64(malformed).legal_max_points != 0)
    throw std::runtime_error("DBP feasibility accepted partial binary64 lane");
  std::cout << "dbp_exact_stage_actions=4 status=PASS\n";
}
