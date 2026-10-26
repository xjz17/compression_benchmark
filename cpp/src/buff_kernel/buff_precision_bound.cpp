#include "buff_precision_bound.hpp"

#include <bit>
#include <cstring>
#include <string>

namespace buff_kernel {

namespace {

double bits_to_double(std::uint64_t u) {
    double d;
    std::memcpy(&d, &u, sizeof(double));
    return d;
}

std::uint64_t double_to_bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(double));
    return u;
}

}  // namespace

double get_precision_bound(int precision) {
    std::string s = "0.";
    s.append(static_cast<std::size_t>(precision), '0');
    s += "49";
    return std::stod(s);
}

BuffPrecisionBound::BuffPrecisionBound(double prec_delta) : precision_(prec_delta) {
    std::uint64_t xu = double_to_bits(prec_delta);
    precision_exp_ = static_cast<int>((xu & kExpMask) >> 52) - 1023;
}

double BuffPrecisionBound::precision_bound(double orig) {
    std::uint64_t mask = ~0ULL;
    double pre = 0.0;
    double cur = 0.0;
    std::uint64_t origu = double_to_bits(orig);
    std::uint64_t curu = (origu & (mask << position_)) | (1ULL << position_);
    cur = bits_to_double(curu);
    pre = cur;
    bool bounded = is_bounded(orig, cur);
    if (bounded) {
        while (bounded) {
            if (position_ == 52) {
                return pre;
            }
            position_ += 1;
            curu = (origu & (mask << position_)) | (1ULL << position_);
            cur = bits_to_double(curu);
            if (!is_bounded(orig, cur)) {
                bounded = false;
                break;
            }
            pre = cur;
        }
    } else {
        while (!bounded) {
            if (position_ == 0) {
                break;
            }
            position_ -= 1;
            curu = (origu & (mask << position_)) | (1ULL << position_);
            cur = bits_to_double(curu);
            if (is_bounded(orig, cur)) {
                bounded = true;
                pre = cur;
                break;
            }
        }
    }
    return pre;
}

void BuffPrecisionBound::cal_length(double x) {
    std::uint64_t xu = double_to_bits(x);
    int trailing_zeros = static_cast<int>(std::countr_zero(xu));
    int exp = static_cast<int>((xu & kExpMask) >> 52) - 1023;
    std::uint64_t dec_length = 0;
    if (52 <= trailing_zeros) {
        if (exp < 0) {
            dec_length = static_cast<std::uint64_t>(-exp);
            if (exp < precision_exp_) {
                dec_length = 0;
            }
        }
    } else if ((52 - trailing_zeros) > exp) {
        dec_length = static_cast<std::uint64_t>((52 - trailing_zeros) - exp);
    }
    if (exp >= 0) {
        if (static_cast<std::uint64_t>(exp + 1) > int_length_) {
            int_length_ = static_cast<std::uint64_t>(exp + 1);
        }
    }
    if (dec_length > decimal_length_) {
        decimal_length_ = dec_length;
    }
}

void BuffPrecisionBound::set_length(std::uint64_t ilen, std::uint64_t dlen) {
    int_length_ = ilen;
    decimal_length_ = dlen;
}

void BuffPrecisionBound::fetch_components(double bd, std::int64_t& out_int, std::uint64_t& out_dec) const {
    std::uint64_t bdu = double_to_bits(bd);
    int exp = static_cast<int>((bdu & kExpMask) >> 52) - 1023;
    std::uint64_t sign = bdu & kFirstOne;
    std::uint64_t int_part = 0;
    std::uint64_t dec_part = 0;
    if (exp >= 0) {
        dec_part = bdu << static_cast<unsigned>(12 + exp);
        int_part = ((bdu << 11) | kFirstOne) >> static_cast<unsigned>(63 - exp);
        if (sign != 0) {
            int_part = ~int_part;
            dec_part = ~dec_part + 1;
        }
    } else if (exp < precision_exp_) {
        dec_part = 0;
        if (sign != 0) {
            int_part = kNegOne;
            dec_part = ~dec_part;
        }
    } else {
        dec_part = ((bdu << 11) | kFirstOne) >> static_cast<unsigned>(-exp - 1);
        if (sign != 0) {
            int_part = kNegOne;
            dec_part = ~dec_part;
        }
    }
    std::int64_t signed_int = 0;
    std::memcpy(&signed_int, &int_part, sizeof(std::int64_t));
    std::uint64_t shift = 64ULL - decimal_length_;
    std::uint64_t dec_out = (shift > 0) ? (dec_part >> shift) : dec_part;
    out_int = signed_int;
    out_dec = dec_out;
}

bool BuffPrecisionBound::is_bounded(double a, double b) const {
    double delta = a - b;
    return std::fabs(delta) < precision_;
}

}  // namespace buff_kernel
