#include "operator102_special_bridges.hpp"

#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using native_graph_v1::operator102::Bytes;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T>
Bytes bytes(const std::vector<T>& values) {
  static_assert(std::endian::native == std::endian::little);
  Bytes out(values.size() * sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), values.data(), out.size());
  return out;
}

}  // namespace

int main() {
  using namespace native_graph_v1::operator102;
  std::vector<std::int64_t> integers;
  std::vector<double> floats;
  for (std::int64_t i = 0; i < 128; ++i) {
    integers.push_back((i % 11) * (i % 7) - 19);
    floats.push_back(static_cast<double>((i % 17) - 8) / 8.0);
  }
  const auto int_raw = bytes(integers);
  const auto float_raw = bytes(floats);
  for (const auto id : {"A057", "A065", "A086", "A094", "A096",
                        "A097", "A098", "A100", "A102"}) {
    require(supports_special_bridge(id), "bridge inventory missing ID");
    const auto result = run_special_bridge(id, int_raw);
    require(result.fully_restored == int_raw, "full integer round trip");
    require(result.target_inverse == result.target_before, "focal round trip");
    require(!result.target_after_ports.empty(), "missing target output port");
    require(result.target_after_ports.size() == result.output_kinds.size(),
            "missing output kind descriptor");
    require(result.target_after_ports.size() == result.output_widths.size(),
            "missing output width descriptor");
    require(result.target_after_ports.size() == result.output_logical_views.size(),
            "missing output logical view");
  }
  const auto alp = run_special_bridge("A099", float_raw);
  require(alp.fully_restored == float_raw, "full ALP2 round trip");
  require(alp.prefix_inverse_metadata.size() == 4,
          "ALP2 prefix metadata must retain four stage packets");
  const auto legacy = run_special_bridge("A098", int_raw);
  require(legacy.prefix_inverse_metadata.size() == 4,
          "legacy limb prefix metadata must retain four stage packets");
  for (const auto& metadata : legacy.prefix_inverse_metadata)
    require(!metadata.empty(), "legacy prefix metadata packet missing");
  require(!supports_special_bridge("A095"), "ordinary dispatcher stage claimed special");
  std::cout << "operator102 special bridge contract PASS\n";
}
