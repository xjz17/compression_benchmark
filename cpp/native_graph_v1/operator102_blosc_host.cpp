#include "blosc_codec.hpp"
#include "blosc2_codec.hpp"
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace {
using Bytes = std::vector<std::uint8_t>;
using Function = Bytes (*)(std::span<const std::uint8_t>, unsigned, int);
struct Plugin {
  void* handle;
  Function function;
  explicit Plugin(const char* path) {
    // Every codec's private C symbols are localized by the export map.
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(std::string("Blosc DSO: ") + dlerror());
    function = reinterpret_cast<Function>(dlsym(handle, "operator102_blosc_call"));
    if (!function) throw std::runtime_error("Blosc DSO entry missing");
  }
  ~Plugin() { if (handle) dlclose(handle); }
};
Bytes invoke(unsigned variant, std::span<const std::uint8_t> bytes,
             unsigned operation, int type_size = 1) {
  if (variant == 1) {
    static Plugin plugin(OPERATOR102_BLOSC1_PLUGIN);
    return plugin.function(bytes, operation, type_size);
  }
  if (variant == 2) {
    static Plugin plugin(OPERATOR102_BLOSC2_PLUGIN);
    return plugin.function(bytes, operation, type_size);
  }
  static Plugin plugin(OPERATOR102_BLOSC333_PLUGIN);
  return plugin.function(bytes, operation, type_size);
}
}
namespace wc_blosc1_bench {
std::vector<std::uint8_t> encodeBufferExact(std::span<const std::uint8_t> bytes, int width) {
  return invoke(1, bytes, 0, width);
}
std::vector<std::uint8_t> decodeBufferExact(std::span<const std::uint8_t> bytes) {
  return invoke(1, bytes, 1);
}
}
namespace wc_blosc2_bench {
std::vector<std::uint8_t> encodeBufferExact(std::span<const std::uint8_t> bytes, int width) {
  return invoke(2, bytes, 0, width);
}
std::vector<std::uint8_t> decodeBufferExact(std::span<const std::uint8_t> bytes, int width) {
  return invoke(2, bytes, 1, width);
}
}
namespace blosc2_binding_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes core_forward(const Bytes& bytes) { return invoke(3, bytes, 0); }
Bytes core_inverse(const Bytes& bytes) { return invoke(3, bytes, 1); }
Bytes frame_forward(const Bytes& bytes) { return invoke(3, bytes, 2); }
Bytes frame_inverse(const Bytes& bytes) { return invoke(3, bytes, 3); }
}
