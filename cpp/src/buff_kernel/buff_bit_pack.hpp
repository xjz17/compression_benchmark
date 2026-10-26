#pragma once

/**
 * LSB-first bit writer aligned with upstream {@code database/src/methods/bit_packing.rs}
 * {@code BitPack::<Vec<u8>>::write}.
 */

#include <cstdint>
#include <vector>

namespace buff_kernel {

/** LSB-first bit reader matching upstream {@code BitPack<&[u8]>::read}. */
class BuffBitPackReader {
   public:
    BuffBitPackReader(const std::uint8_t* data, std::size_t len);

    [[nodiscard]] std::uint32_t read(int bits);
    [[nodiscard]] std::size_t bit_position() const { return bit_pos_; }

   private:
    const std::uint8_t* data_{nullptr};
    std::size_t len_{0};
    std::size_t bit_pos_{0};
};

class BuffBitPackWriter {
   public:
    static constexpr int kMaxBits = 32;
    static constexpr int kByteBits = 8;

    void write(std::uint32_t value, int bits);
    [[nodiscard]] std::vector<std::uint8_t> to_bytes() const;

   private:
    void ensure_capacity();

    std::vector<std::uint8_t> buf_{};
    std::size_t cursor_{0};
    int bits_in_byte_{0};
};

}  // namespace buff_kernel
