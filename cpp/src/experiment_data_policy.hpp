#pragma once

#include <cmath>
#include <string_view>

namespace experiment_data_policy {
// Keep infinities and the sign bit of zero; only NaN is excluded from inputs.
template <class Number> inline bool keep_numeric_value(Number value) {
    return !std::isnan(value);
}

inline bool is_nan_cell(std::string_view value) {
    if (!value.empty() && (value.front() == '+' || value.front() == '-')) {
        value.remove_prefix(1);
    }
    return value.size() == 3 && (value[0] == 'n' || value[0] == 'N') &&
           (value[1] == 'a' || value[1] == 'A') &&
           (value[2] == 'n' || value[2] == 'N');
}
} // namespace experiment_data_policy
