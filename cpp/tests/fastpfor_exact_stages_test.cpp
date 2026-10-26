#include "fastpfor_exact_stages.hpp"
#include "dbp_exact_stages.hpp"
#include "int64_column_codec.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
  using namespace exact_actions::fastpfor;
  std::vector<std::vector<std::int64_t>> probes{
           std::vector<std::int64_t>{},
           std::vector<std::int64_t>{7},
           std::vector<std::int64_t>{-9, -9, -3, 0, 1, 17, 17, 99},
           std::vector<std::int64_t>{
               std::numeric_limits<std::int64_t>::min(),
               std::numeric_limits<std::int64_t>::max(), 0, -1, 1}};
  std::vector<std::int64_t> block_tail(777);
  for (std::size_t i = 0; i < block_tail.size(); ++i)
    block_tail[i] = static_cast<std::int64_t>(i * i) - 90000;
  probes.push_back(std::move(block_tail));

  for (const auto& values : probes) {
    const auto stage = pfor_stage_forward_int(values);
    if (pfor_stage_inverse_int(stage) != values)
      throw std::runtime_error("FastPFor PFor-stage round trip");
    const auto wire = variable_byte_forward(stage);
    const auto reference_body = int64_column_codec::fast_pfor_encode_int64(
        values.data(), values.size());
    std::vector<std::uint8_t> reference{'I'};
    reference.insert(reference.end(), reference_body.begin(), reference_body.end());
    if (wire != reference) throw std::runtime_error("FastPFor staged wire parity");
    if (variable_byte_inverse(wire, values.size()) != stage)
      throw std::runtime_error("FastPFor VariableByte-stage round trip");
    const auto adapter = adapter_forward_int(values);
    if (adapter != wire || adapter.empty() || adapter.front() != 'I' ||
        adapter_inverse_int(adapter, values.size()) != values)
      throw std::runtime_error("FastPFor typed adapter round trip");
  }

  const std::vector<std::int64_t> scaled_values{-125, 0, 250, 9999};
  const auto scaling = exact_actions::dbp::make_scaling_packet(3, scaled_values);
  const auto scaled_stage = pfor_stage_forward_scaled(scaling);
  if (pfor_stage_inverse_scaled(scaled_stage) != scaling)
    throw std::runtime_error("FastPFor scaled PFor-stage round trip");
  const auto scaled_wire = variable_byte_forward(scaled_stage);
  if (scaled_wire != adapter_forward_scaled(scaling))
    throw std::runtime_error("FastPFor scaled staged wire parity");
  if (variable_byte_inverse(scaled_wire, scaled_values.size()) != scaled_stage)
    throw std::runtime_error("FastPFor scaled VariableByte-stage round trip");

  std::cout << "fastpfor_exact_stage_actions=2 status=PASS\n";
}
