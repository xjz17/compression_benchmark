#include "buff_bit_pack.hpp"

#include <stdexcept>

namespace buff_kernel {

BuffBitPackReader::BuffBitPackReader(const std::uint8_t* data, std::size_t len) : data_(data), len_(len) {}

std::uint32_t BuffBitPackReader::read(int bits) {
    if (bits <= 0 || bits > BuffBitPackWriter::kMaxBits) {
        throw std::out_of_range("BuffBitPackReader::read bits");
    }
    std::uint32_t v = 0;
    for (int i = 0; i < bits; ++i) {
        std::size_t bi = bit_pos_ / BuffBitPackWriter::kByteBits;
        int off = static_cast<int>(bit_pos_ % BuffBitPackWriter::kByteBits);
        if (bi >= len_) {
            throw std::out_of_range("BuffBitPackReader::read past end");
        }
        int b = (static_cast<int>(data_[bi]) >> off) & 1;
        v |= static_cast<std::uint32_t>(b) << i;
        bit_pos_++;
    }
    return v;
}

void BuffBitPackWriter::write(std::uint32_t value, int bits) {
    if (buf_.empty()) {
        buf_.resize(64);
    }
    int mut_bits = bits;
    std::uint32_t v = value;
    if (mut_bits < kMaxBits) {
        v &= (1U << mut_bits) - 1U;
    }
    while (mut_bits > 0) {
        int bits_left = kByteBits - bits_in_byte_;
        if (mut_bits <= bits_left) {
            buf_[cursor_] |= static_cast<std::uint8_t>((v & 0xFFU) << bits_in_byte_);
            bits_in_byte_ += mut_bits;
            if (bits_in_byte_ >= kByteBits) {
                cursor_++;
                bits_in_byte_ = 0;
            }
            break;
        }
        int take = bits_left;
        std::uint32_t bb = v & ((1U << take) - 1U);
        buf_[cursor_] |= static_cast<std::uint8_t>(bb << bits_in_byte_);
        cursor_++;
        bits_in_byte_ = 0;
        v >>= take;
        mut_bits -= take;
        ensure_capacity();
    }
    ensure_capacity();
}

void BuffBitPackWriter::ensure_capacity() {
    if (cursor_ >= buf_.size() - 4) {
        buf_.resize(buf_.size() * 2);
    }
}

std::vector<std::uint8_t> BuffBitPackWriter::to_bytes() const {
    std::size_t n = cursor_;
    if (bits_in_byte_ > 0) {
        n++;
    }
    return std::vector<std::uint8_t>(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(n));
}

}  // namespace buff_kernel
