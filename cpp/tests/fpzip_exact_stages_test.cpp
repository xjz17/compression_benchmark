#include "fpzip_exact_stages.hpp"

#include <fpzip.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint8_t> raw(const std::vector<double>& values) {
  std::vector<std::uint8_t> out(values.size() * 8);
  if (!out.empty()) std::memcpy(out.data(), values.data(), out.size());
  return out;
}

std::vector<std::uint8_t> reference(const std::vector<double>& values) {
  std::vector<std::uint8_t> out(64 + values.size() * 16);
  FPZ* writer = fpzip_write_to_buffer(out.data(), out.size());
  if (writer == nullptr) throw std::runtime_error("FPZIP writer creation failed");
  writer->type = FPZIP_TYPE_DOUBLE;
  writer->prec = 0;
  writer->nx = static_cast<int>(values.size());
  writer->ny = writer->nz = writer->nf = 1;
  const bool header_ok = fpzip_write_header(writer) != 0;
  const std::size_t size = header_ok ? fpzip_write(writer, values.data()) : 0;
  fpzip_write_close(writer);
  if (!header_ok || size == 0) throw std::runtime_error("FPZIP reference failed");
  out.resize(size);
  return out;
}

}  // namespace

int main() {
  using namespace exact_actions::fpzip;
  const std::vector<std::vector<double>> probes{
      {0.0},
      {-0.0, 1.0, 1.0, -3.5, 9.25},
      {1.0, 1.5, 2.25, 4.0, 8.5, 17.0, 34.25},
  };
  for (const auto& values : probes) {
    const auto source = raw(values);
    const auto residuals = predictor_forward(source);
    if (predictor_inverse(residuals) != source)
      throw std::runtime_error("FPZIP predictor roundtrip failed");
    const auto wire = range_forward(residuals);
    if (wire != reference(values))
      throw std::runtime_error("FPZIP public wire mismatch");
    if (range_inverse(wire) != residuals)
      throw std::runtime_error("FPZIP range-stage inverse mismatch");
    if (decompose(compose(source)) != source)
      throw std::runtime_error("FPZIP chain roundtrip failed");
  }
  std::cout << "fpzip_exact_stage_actions=2 status=PASS\n";
}
