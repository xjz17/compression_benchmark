#include "bzip2_operator_ablation.hpp"

#include <bzlib.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

Bytes native_encode(const Bytes& raw) {
  unsigned int output_size =
      static_cast<unsigned int>(raw.size() + raw.size() / 100 + 601);
  Bytes output(output_size);
  std::uint8_t empty_input = 0;
  const auto* input_data = raw.empty() ? &empty_input : raw.data();
  const int status = BZ2_bzBuffToBuffCompress(
      reinterpret_cast<char*>(output.data()), &output_size,
      const_cast<char*>(reinterpret_cast<const char*>(input_data)),
      static_cast<unsigned int>(raw.size()), 4, 0, 30);
  require(status == BZ_OK, "native Bzip2 reference encode failed");
  output.resize(output_size);
  return output;
}

void check_empty_native_stream() {
  const Bytes raw;
  const auto wire = bzip2_operator_ablation::encode(raw, 4);
  require(wire == native_encode(raw), "Bzip2 empty native byte parity");
  require(wire.size() == 14, "Bzip2 empty stream canonical size");
  require(wire.size() >= 4 && wire[0] == 'B' && wire[1] == 'Z' &&
              wire[2] == 'h' && wire[3] == '4',
          "Bzip2 empty stream native header");
  require(bzip2_operator_ablation::decode(wire, 0).empty(),
          "Bzip2 empty stream roundtrip");

  bool wrong_size_rejected = false;
  try {
    (void)bzip2_operator_ablation::decode(wire, 1);
  } catch (const std::runtime_error&) {
    wrong_size_rejected = true;
  }
  require(wrong_size_rejected, "Bzip2 empty stream wrong decoded size accepted");
}

void check_nonempty_unchanged() {
  const Bytes raw = {0x00, 0x01, 0x02, 0x7f, 0x80, 0xfe, 0xff};
  const auto wire = bzip2_operator_ablation::encode(raw, 4);
  require(!wire.empty(), "Bzip2 non-empty stream missing");
  require(wire == native_encode(raw), "Bzip2 non-empty native byte parity");
  require(bzip2_operator_ablation::decode(wire, raw.size()) == raw,
          "Bzip2 non-empty roundtrip changed");
}

}  // namespace

int main() {
  check_empty_native_stream();
  check_nonempty_unchanged();
  std::cout << "bzip2_empty_exact=PASS native_empty_bytes=14 "
               "empty_roundtrip=PASS nonempty_roundtrip=PASS "
               "wrong_size_rejected=PASS\n";
}
