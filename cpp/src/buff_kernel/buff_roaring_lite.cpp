#include "buff_roaring_lite.hpp"

#include <algorithm>

namespace buff_kernel {

void BuffRoaringLite::clear() {
    ids_.clear();
}

void BuffRoaringLite::add(std::uint32_t x) {
    auto it = std::lower_bound(ids_.begin(), ids_.end(), x);
    if (it == ids_.end() || *it != x) {
        ids_.insert(it, x);
    }
}

void BuffRoaringLite::or_with(const BuffRoaringLite& other) {
    for (std::uint32_t x : other.ids_) {
        add(x);
    }
}

}  // namespace buff_kernel
