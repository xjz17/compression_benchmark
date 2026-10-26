#pragma once

/**
 * C++ port of Tranway1/buff {@code database/src/methods/prec_double.rs} ({@code PrecisionBound}),
 * aligned with {@code org.example.buff.BuffPrecisionBound} (Java) and the Rust source.
 */

#include <cmath>
#include <cstdint>

namespace buff_kernel {

inline constexpr std::uint64_t kExpMask = 0x7FF0000000000000ULL;
inline constexpr std::uint64_t kFirstOne = 0x8000000000000000ULL;
inline constexpr std::uint64_t kNegOne = 0xFFFFFFFFFFFFFFFFULL;

double get_precision_bound(int precision);

class BuffPrecisionBound {
   public:
    explicit BuffPrecisionBound(double prec_delta);

    double precision_bound(double orig);
    void cal_length(double x);
    std::uint64_t int_length() const { return int_length_; }
    std::uint64_t decimal_length() const { return decimal_length_; }
    void set_length(std::uint64_t ilen, std::uint64_t dlen);

    /** Returns (signed integer part, decimal part after shift). */
    void fetch_components(double bd, std::int64_t& out_int, std::uint64_t& out_dec) const;

    bool is_bounded(double a, double b) const;

   private:
    std::uint64_t position_{0};
    double precision_{};
    int precision_exp_{0};
    std::uint64_t int_length_{0};
    std::uint64_t decimal_length_{0};
};

}  // namespace buff_kernel
