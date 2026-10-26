#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "leco_double_codec.hpp"

/** int64 as double with zero fractional part. */
inline std::vector<std::vector<double>> integers_as_double_columns(
    const std::vector<std::vector<std::int64_t>> &int_cols) {
    std::vector<std::vector<double>> out;
    out.reserve(int_cols.size());
    for (const auto &col : int_cols) {
        std::vector<double> dcol;
        dcol.reserve(col.size());
        for (std::int64_t v : col) {
            dcol.push_back(static_cast<double>(v));
        }
        out.push_back(std::move(dcol));
    }
    return out;
}

inline std::vector<std::vector<std::int64_t>> scale_double_columns_to_int64(
    const std::vector<std::vector<double>> &float_cols,
    const std::vector<int> *max_point_per_column = nullptr) {
    std::vector<std::vector<std::int64_t>> out;
    out.reserve(float_cols.size());
    for (std::size_t ci = 0; ci < float_cols.size(); ++ci) {
        const auto &col = float_cols[ci];
        if (col.empty()) {
            continue;
        }
        std::vector<std::int64_t> data;
        if (max_point_per_column != nullptr && ci < max_point_per_column->size()) {
            const int cap = (*max_point_per_column)[ci];
            if (cap >= 0) {
                if (!leco_double::quantizeScaledIntsWithMaxPointFallback(col, cap, data, nullptr)) {
                    data.clear();
                }
            } else {
                int auto_scale = leco_double::maxLosslessDecimalScale(col);
                if (auto_scale < 0 || !leco_double::quantizeScaledInts(col, auto_scale, data)) {
                    data.clear();
                }
            }
        } else {
            int auto_scale = leco_double::maxLosslessDecimalScale(col);
            if (auto_scale < 0 || !leco_double::quantizeScaledInts(col, auto_scale, data)) {
                data.clear();
            }
        }
        // DELTA_BINARY_PACKED and Stream VByte are integer codecs.  Decimal
        // scaling is preferable when it is exactly reversible, but arbitrary
        // doubles (large magnitudes, non-finite values, or more than 15 decimal
        // digits) cannot always be represented as scaled int64 values.  Keep
        // those columns instead of silently dropping them: the IEEE-754 bit
        // pattern is a lossless int64 representation and the caller's codec can
        // round-trip it unchanged.
        if (data.empty() && !col.empty()) {
            data.resize(col.size());
            for (std::size_t i = 0; i < col.size(); ++i) {
                std::uint64_t bits = 0;
                std::memcpy(&bits, &col[i], sizeof(bits));
                data[i] = static_cast<std::int64_t>(bits);
            }
        }
        out.push_back(std::move(data));
    }
    return out;
}

inline std::vector<std::vector<float>> integers_as_float_columns(
    const std::vector<std::vector<std::int64_t>> &int_cols) {
    std::vector<std::vector<float>> out;
    out.reserve(int_cols.size());
    for (const auto &col : int_cols) {
        std::vector<float> fcol;
        fcol.reserve(col.size());
        for (std::int64_t v : col) {
            fcol.push_back(static_cast<float>(static_cast<double>(v)));
        }
        out.push_back(std::move(fcol));
    }
    return out;
}

/** Narrow double→float for legacy float research codecs (no exact-roundtrip filter). */
inline std::vector<std::vector<float>> double_columns_as_float(
    const std::vector<std::vector<double>> &cols) {
    std::vector<std::vector<float>> out;
    out.reserve(cols.size());
    for (const auto &col : cols) {
        if (col.empty()) {
            continue;
        }
        std::vector<float> fcol;
        fcol.reserve(col.size());
        for (double v : col) {
            fcol.push_back(static_cast<float>(v));
        }
        out.push_back(std::move(fcol));
    }
    return out;
}

inline std::vector<std::vector<int>> int64_columns_as_int32(
    const std::vector<std::vector<std::int64_t>> &in) {
    std::vector<std::vector<int>> out;
    out.reserve(in.size());
    for (const auto &col : in) {
        std::vector<int> icol;
        bool ok = true;
        for (std::int64_t v : col) {
            if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
                ok = false;
                break;
            }
            icol.push_back(static_cast<int>(v));
        }
        if (ok && !icol.empty()) {
            out.push_back(std::move(icol));
        }
    }
    return out;
}
