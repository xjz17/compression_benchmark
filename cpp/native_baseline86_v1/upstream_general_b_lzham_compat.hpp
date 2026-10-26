#pragma once

// LZHAM's last public release names an old libc++ implementation detail on
// Apple.  Keep the vendored source unchanged and provide the removed trait in
// the isolated LZHAM translation units.
#include <type_traits>

#if defined(__APPLE__)
namespace std {
template <class T>
struct __is_pod {
  static constexpr bool __value = is_standard_layout<T>::value && is_trivial<T>::value;
};
}  // namespace std
#endif
