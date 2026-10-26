#pragma once

/**
 * Minimal Roaring-style set of ``uint32`` row ids (sorted unique), for parity with upstream
 * ``croaring::Bitmap`` iterator usage in ``buff_simd.rs`` filter paths. Not wire-compatible with
 * CRoaring bitstreams; use for native kernel bookkeeping and filter simulations.
 */

#include <cstdint>
#include <vector>

namespace buff_kernel {

class BuffRoaringLite {
   public:
    void clear();
    void add(std::uint32_t x);
    void or_with(const BuffRoaringLite& other);
    [[nodiscard]] std::size_t cardinality() const { return ids_.size(); }
    [[nodiscard]] const std::vector<std::uint32_t>& sorted_ids() const { return ids_; }

   private:
    std::vector<std::uint32_t> ids_{};
};

}  // namespace buff_kernel
