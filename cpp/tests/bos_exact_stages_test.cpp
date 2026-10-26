#include "bos_exact_stages.hpp"
#include "bos_operator_ablation.hpp"
#include "dbp_exact_stages.hpp"
#include "ts2diff_bos_native.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
  using namespace exact_actions::bos;
  const std::vector<std::vector<std::int64_t>> probes{
      {},
      {7},
      {-9, -9, -3, 0, 1, 17, 17, 99},
      {1000, 1001, 1002, 1003, 1000000, 1004, 1005, -900000},
      {std::numeric_limits<std::int64_t>::min(),
       std::numeric_limits<std::int64_t>::max(), 0, -1, 1},
  };
  for (const auto& values : probes) {
    const auto d = delta_forward(values);
    if (delta_inverse(d) != values) throw std::runtime_error("BOS delta round trip");
    const auto f1 = first_for_forward(d);
    if (first_for_inverse(f1) != d) throw std::runtime_error("BOS first FOR round trip");
    const auto separated = outlier_forward(f1);
    if (outlier_inverse(separated) != f1) throw std::runtime_error("BOS outlier round trip");
    const auto f2 = second_for_forward(separated);
    if (second_for_inverse(f2) != separated) throw std::runtime_error("BOS second FOR round trip");
    const auto wire = bitpack_forward(f2);
    if (bitpack_inverse(wire) != f2) throw std::runtime_error("BOS bitpack round trip");
    if (wire != bos_operator_ablation::encode(values, ""))
      throw std::runtime_error("BOS five-stage public wire parity");
    if (decompose(wire) != values) throw std::runtime_error("BOS five-stage chain");

    const auto plain_delta = adapter_plain_delta_forward(values);
    if (adapter_plain_delta_inverse(plain_delta) != values)
      throw std::runtime_error("BOS64 adapter pure delta round trip");
    const auto ad = adapter_zigzag_forward(plain_delta);
    if (adapter_zigzag_inverse(ad) != plain_delta)
      throw std::runtime_error("BOS64 adapter ZigZag round trip");
    if (ad != adapter_delta_forward(values))
      throw std::runtime_error("BOS64 compatibility packet parity");
    const auto scaling = exact_actions::dbp::make_scaling_packet(3, values);
    const auto plain_scaled = adapter_plain_delta_forward_scaled(scaling);
    if (adapter_plain_delta_inverse_scaled(plain_scaled) != scaling)
      throw std::runtime_error("BOS64 scaled pure delta round trip");
    const auto zigzag_scaled = adapter_zigzag_forward_scaled(plain_scaled);
    if (adapter_zigzag_inverse_scaled(zigzag_scaled) != plain_scaled)
      throw std::runtime_error("BOS64 scaled ZigZag round trip");
    if (zigzag_scaled != adapter_delta_forward_scaled(scaling))
      throw std::runtime_error("BOS64 scaled compatibility packet parity");
    if (adapter_delta_inverse(ad) != values)
      throw std::runtime_error("BOS64 adapter delta round trip");
    if (!values.empty()) {
      const auto adapter_wire = adapter_bitpack_forward(ad);
      if (adapter_wire != ts2diff_bos_native::encode_int64_column_payload(values))
        throw std::runtime_error("BOS64 adapter public wire parity");
      if (adapter_bitpack_inverse(adapter_wire) != ad)
        throw std::runtime_error("BOS64 adapter bitpack round trip");
      if (adapter_decompose(adapter_wire) != values)
        throw std::runtime_error("BOS64 adapter chain");
      const auto public_wire = adapter_wire_forward_int(ad);
      if (public_wire.size() != adapter_wire.size() + 1 || public_wire.front() != 'I')
        throw std::runtime_error("BOS64 typed adapter wire");
      if (adapter_wire_inverse_int(public_wire) != ad)
        throw std::runtime_error("BOS64 typed adapter inverse");
    }
  }
  std::cout << "bos_exact_stages=PASS adapter_delta_zigzag_split=PASS\n";
}
