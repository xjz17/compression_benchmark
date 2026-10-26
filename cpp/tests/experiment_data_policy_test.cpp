#include "experiment_data_policy.hpp"
#include <cassert>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string>

int main() {
    using namespace experiment_data_policy;
    assert(!keep_numeric_value(std::numeric_limits<double>::quiet_NaN()));
    assert(keep_numeric_value(std::numeric_limits<double>::infinity()));
    assert(keep_numeric_value(-std::numeric_limits<double>::infinity()));
    assert(keep_numeric_value(-0.0) && std::signbit(-0.0));
    assert(is_nan_cell("NaN") && is_nan_cell("-nAn"));
    assert(!is_nan_cell("infinity"));
    std::int64_t integer = 0;
    const std::string text = "9007199254740993";
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), integer);
    assert(parsed.ec == std::errc{} && integer == INT64_C(9007199254740993));
}
