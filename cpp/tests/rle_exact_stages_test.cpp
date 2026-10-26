#include "rle_exact_stages.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  using namespace exact_actions::rle;
  const std::vector<double> values{-3.5, -3.5, 0.0, 1.25, 1.25, 9.0};
  const auto scaled = scaling_forward(values, 2);
  if (scaling_inverse(scaled) != values)
    throw std::runtime_error("RLE scaling round trip");
  const auto wire = run_length_forward(scaled);
  if (run_length_inverse(wire) != scaled)
    throw std::runtime_error("RLE run-length stage round trip");
  if (decompose_double(wire) != values)
    throw std::runtime_error("RLE stage chain round trip");

  const std::vector<std::int64_t> integers{-350, -350, 0, 125, 125, 900};
  // This stage is the frozen decimal-scaling protocol. The public TsFile
  // encoder now emits URL1, so parity must use the preserved legacy writer.
  auto expected = tsfile::p02::encode_rle_int64_legacy(integers);
  expected.insert(expected.begin(), 2);
  if (wire != expected) throw std::runtime_error("RLE legacy public wire parity");
  std::cout << "rle_exact_stage_actions=2 status=PASS\n";
}
