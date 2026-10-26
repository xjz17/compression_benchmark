// Private versioned codec DSO: the three Blosc ABIs never share C symbols.
#include <cstdint>
#include <span>
#include <vector>
#include <stdexcept>

#if OPERATOR102_BLOSC_VARIANT == 1
#include "blosc_codec.hpp"
#elif OPERATOR102_BLOSC_VARIANT == 2
#include "blosc2_codec.hpp"
#else
namespace blosc2_binding_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes core_forward(const Bytes&);
Bytes core_inverse(const Bytes&);
Bytes frame_forward(const Bytes&);
Bytes frame_inverse(const Bytes&);
}
#endif

extern "C" __attribute__((visibility("default")))
std::vector<std::uint8_t> operator102_blosc_call(
    std::span<const std::uint8_t> bytes, unsigned operation, int type_size) {
#if OPERATOR102_BLOSC_VARIANT == 1
  if (operation == 0) return wc_blosc1_bench::encodeBufferExact(bytes, type_size);
  if (operation == 1) return wc_blosc1_bench::decodeBufferExact(bytes);
#elif OPERATOR102_BLOSC_VARIANT == 2
  if (operation == 0) return wc_blosc2_bench::encodeBufferExact(bytes, type_size);
  if (operation == 1) return wc_blosc2_bench::decodeBufferExact(bytes, type_size);
#else
  const std::vector<std::uint8_t> raw(bytes.begin(), bytes.end());
  if (operation == 0) return blosc2_binding_cpp::core_forward(raw);
  if (operation == 1) return blosc2_binding_cpp::core_inverse(raw);
  if (operation == 2) return blosc2_binding_cpp::frame_forward(raw);
  if (operation == 3) return blosc2_binding_cpp::frame_inverse(raw);
#endif
  throw std::invalid_argument("unknown private Blosc operation");
}
