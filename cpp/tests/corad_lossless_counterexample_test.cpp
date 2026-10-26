// Capture the payload/wire from the unchanged benchmark encoder.  The shim
// forwards every call to the same zlib compress2 function and only copies bytes.
#include <zlib.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::uint8_t> captured_payload;
std::vector<std::uint8_t> captured_wire;
}

int audit_compress2(Bytef* dest, uLongf* dest_len, const Bytef* source,
                    uLong source_len, int level) {
  captured_payload.assign(source, source + source_len);
  const int rc = ::compress2(dest, dest_len, source, source_len, level);
  if (rc == Z_OK) captured_wire.assign(dest, dest + *dest_len);
  return rc;
}

#define compress2 audit_compress2
#include "../src/corad_tricklet_float_codec.cpp"
#undef compress2

int main() try {
  std::vector<double> x(80), y(80);
  for (int i = 0; i != 80; ++i) {
    x[i] = i;
    y[i] = 2 * i + 8;
  }
  std::vector<long long> first, second;
  corad_tricklet_float::benchDoubleColumnsCoradTrickletLegacyForensics({x}, first);
  const auto first_payload = captured_payload;
  const auto first_wire = captured_wire;
  corad_tricklet_float::benchDoubleColumnsCoradTrickletLegacyForensics({y}, second);
  if (x == y || first_payload != captured_payload || first_wire != captured_wire)
    throw std::runtime_error("CORAD collision did not reproduce");
  if (first.size() != 4 || first[0] != 640 || first[1] != second[1])
    throw std::runtime_error("unexpected CORAD benchmark counters");
  std::cout << "CORAD confirmed non-injective: x[i]=i, y[i]=2*i+8, n=80; "
            << "distinct 640-byte inputs, identical " << first_payload.size()
            << "-byte normalized payload and " << first_wire.size()
            << "-byte zlib wire\n";
  const std::vector<long long> disabled(4, 0);
  corad_tricklet_float::benchDoubleColumnsCoradTricklet({x}, first);
  if (first != disabled) throw std::runtime_error("lossy C++ double benchmark still enabled");
  corad_tricklet_float::benchFloatColumnsCoradTricklet({{1.0f, 2.0f}}, first);
  if (first != disabled) throw std::runtime_error("lossy C++ float benchmark still enabled");
  corad_tricklet_float::benchInt64ColumnsCoradTricklet({{1, 2}}, first);
  if (first != disabled) throw std::runtime_error("lossy C++ int benchmark still enabled");
  std::cout << "CORAD production C++ entrypoints: all three fail closed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
