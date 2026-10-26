// In-process old/new task-feature microbenchmark for the v6 fast path.
//
// Build the old source with -Dfeatures_v1=features_before_v6, the new source
// normally, then link both objects with this harness. No process launch or I/O
// occurs inside a timed interval.
#include "features.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace wc::features_before_v6 {
inline constexpr std::size_t task_width = 18;
struct FlowView {
  std::string kind;
  std::uint32_t width;
  std::span<const std::uint8_t> bytes;
  std::string logical_view;
};
std::array<double, task_width> task(std::span<const FlowView> flows);
}  // namespace wc::features_before_v6

namespace {
using Clock = std::chrono::steady_clock;
volatile double sink_value = 0;

struct OwnedFlow {
  std::string kind;
  std::uint32_t width;
  std::string logical_view;
  std::vector<std::uint8_t> bytes;
};
struct Workload {
  std::string name;
  std::vector<OwnedFlow> owned;
  std::vector<wc::features_v1::FlowView> current;
  std::vector<wc::features_before_v6::FlowView> before;

  void finish() {
    current.clear(); before.clear();
    for (const auto& item : owned) {
      current.push_back({item.kind, item.width, item.bytes, item.logical_view});
      before.push_back({item.kind, item.width, item.bytes, item.logical_view});
    }
  }
  std::size_t byte_count() const {
    std::size_t result = 0;
    for (const auto& item : owned) result += item.bytes.size();
    return result;
  }
};

template <typename UInt> void append_little(std::vector<std::uint8_t>& out, UInt value) {
  for (std::size_t index = 0; index < sizeof(UInt); ++index)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
}

std::uint64_t splitmix_hash(std::uint64_t value) {
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

Workload smooth_int64() {
  Workload work{"smooth_int64_8193"};
  OwnedFlow item{"numeric", 8, "int64", {}};
  item.bytes.reserve(8193 * 8);
  for (std::int64_t index = 0; index < 8193; ++index) {
    const auto value = index * 17 - (index % 29) * 31;
    append_little(item.bytes, static_cast<std::uint64_t>(value));
  }
  work.owned.push_back(std::move(item)); work.finish(); return work;
}

Workload float64_specials() {
  Workload work{"float64_zeros_nonfinite_8193"};
  OwnedFlow item{"numeric", 8, "float64", {}};
  item.bytes.reserve(8193 * 8);
  for (std::uint64_t index = 0; index < 8193; ++index) {
    std::uint64_t bits;
    if (index % 257 == 0) bits = UINT64_C(0x7ff8000000000001); // NaN
    else if (index % 131 == 0) bits = UINT64_C(0x7ff0000000000000); // +Inf
    else if (index % 17 == 0) bits = index & 1 ? UINT64_C(0x8000000000000000) : 0; // +/-0
    else bits = std::bit_cast<std::uint64_t>(std::sin(double(index) * 0.001));
    append_little(item.bytes, bits);
  }
  work.owned.push_back(std::move(item)); work.finish(); return work;
}

Workload collision_chain() {
  Workload work{"float64_hash_collision_chain_24"};
  OwnedFlow item{"numeric", 8, "float64", {}};
  std::array<std::vector<double>, 64> buckets;
  for (std::uint64_t integer = 1; integer < 200000; ++integer) {
    const double value = static_cast<double>(integer);
    auto& group = buckets[splitmix_hash(std::bit_cast<std::uint64_t>(value)) & 63];
    group.push_back(value);
    if (group.size() == 24) {
      for (const double selected : group)
        append_little(item.bytes, std::bit_cast<std::uint64_t>(selected));
      work.owned.push_back(std::move(item)); work.finish(); return work;
    }
  }
  throw std::runtime_error("failed to construct collision chain");
}

Workload multiple_roots() {
  Workload work{"multiple_roots_mixed_8x1024"};
  for (unsigned root = 0; root < 8; ++root) {
    OwnedFlow item;
    if (root % 3 == 0) item = {"numeric", 8, "uint64", {}};
    else if (root % 3 == 1) item = {"numeric", 4, "int32", {}};
    else item = {"numeric", 4, "float32", {}};
    for (std::uint64_t index = 0; index < 1024; ++index) {
      if (item.logical_view == "uint64") append_little(item.bytes, index * UINT64_C(11400714819323198485));
      else if (item.logical_view == "int32") append_little(item.bytes, std::uint32_t(index * 7919 - root));
      else append_little(item.bytes, std::bit_cast<std::uint32_t>(float(std::sin(double(index + root)))));
    }
    work.owned.push_back(std::move(item));
  }
  work.finish(); return work;
}

Workload bytes_only_roots() {
  Workload work{"bytes_only_32x3"};
  for (unsigned root = 0; root < 32; ++root)
    work.owned.push_back({"serial", 1, "bytes", {std::uint8_t(root), std::uint8_t(root), std::uint8_t(255 - root)}});
  work.finish(); return work;
}

Workload tiny_numeric() {
  Workload work{"tiny_int32_1"};
  OwnedFlow item{"numeric", 4, "int32", {}};
  append_little(item.bytes, std::uint32_t(17));
  work.owned.push_back(std::move(item)); work.finish(); return work;
}

bool exact_equal(const std::array<double, 18>& left, const std::array<double, 18>& right) {
  for (std::size_t index = 0; index < left.size(); ++index)
    if (std::bit_cast<std::uint64_t>(left[index]) != std::bit_cast<std::uint64_t>(right[index])) return false;
  return true;
}

template <typename Function> double measure(Function&& function, std::size_t iterations) {
  const auto begin = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    const auto output = function();
    sink_value = sink_value + output[iteration % output.size()];
  }
  const auto end = Clock::now();
  return double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) /
         double(iterations);
}

double middle(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t base_iterations = 40;
    if (argc == 2) base_iterations = std::stoull(argv[1]);
    else if (argc != 1) throw std::invalid_argument("usage: benchmark [base_iterations]");
    if (!base_iterations) throw std::invalid_argument("iterations must be positive");

    std::vector<Workload> workloads;
    workloads.push_back(smooth_int64());
    workloads.push_back(float64_specials());
    workloads.push_back(collision_chain());
    workloads.push_back(multiple_roots());
    workloads.push_back(bytes_only_roots());
    workloads.push_back(tiny_numeric());

    std::cout << std::setprecision(10);
    std::cout << "{\n  \"schema\": \"native-features-inprocess-benchmark-v6\",\n"
                 "  \"clock\": \"std::chrono::steady_clock\",\n"
                 "  \"timed_scope\": \"direct task() calls only; input construction, process startup and output excluded\",\n"
                 "  \"rounds\": 7,\n  \"workloads\": [\n";
    bool all_equal = true;
    for (std::size_t case_index = 0; case_index < workloads.size(); ++case_index) {
      const auto& work = workloads[case_index];
      const auto old_call = [&] { return wc::features_before_v6::task(work.before); };
      const auto new_call = [&] { return wc::features_v1::task(work.current); };
      const auto old_output = old_call();
      const auto new_output = new_call();
      const bool equal = exact_equal(old_output, new_output);
      all_equal = all_equal && equal;
      for (unsigned warmup = 0; warmup < 10; ++warmup) { old_call(); new_call(); }

      const std::size_t iterations = work.byte_count() < 1024 ? base_iterations * 20 : base_iterations;
      std::vector<double> old_times, new_times;
      for (unsigned round = 0; round < 7; ++round) {
        if (round & 1) {
          new_times.push_back(measure(new_call, iterations));
          old_times.push_back(measure(old_call, iterations));
        } else {
          old_times.push_back(measure(old_call, iterations));
          new_times.push_back(measure(new_call, iterations));
        }
      }
      const double old_ns = middle(old_times), new_ns = middle(new_times);
      if (case_index) std::cout << ",\n";
      std::cout << "    {\"name\": \"" << work.name << "\", \"input_bytes\": " << work.byte_count()
                << ", \"iterations_per_round\": " << iterations
                << ", \"exact_feature_bits_equal\": " << (equal ? "true" : "false")
                << ", \"old_ns_per_call_median\": " << old_ns
                << ", \"new_ns_per_call_median\": " << new_ns
                << ", \"old_over_new\": " << old_ns / new_ns << "}";
    }
    std::cout << "\n  ],\n  \"all_exact_feature_bits_equal\": " << (all_equal ? "true" : "false")
              << "\n}\n";
    return all_equal ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 2;
  }
}
