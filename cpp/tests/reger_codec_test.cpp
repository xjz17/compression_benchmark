#include "reger_codec.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

std::uint64_t double_bits(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

}  // namespace

int main() {
  std::vector<std::int64_t> ints;
  for (int i = 0; i < 128; ++i) {
    ints.push_back(1000 + i * 5 + (i % 7));
  }
  ints.push_back(-(1LL << 40));
  ints.push_back(1LL << 42);
  ints.push_back(-17);
  ints.push_back(0);

  const auto encoded_ints = reger_codec::encodeInt64Column(ints);
  const auto decoded_ints = reger_codec::decodeInt64Column(encoded_ints);
  assert(decoded_ints == ints);

  std::vector<double> doubles = {
      -0.0, 1.25, -13.5, 3.141592653589793, std::ldexp(1.0, 40),
      -std::ldexp(1.0, -20), INFINITY, -INFINITY,
  };
  std::vector<double> repeated;
  for (int i = 0; i < 16; ++i) {
    repeated.insert(repeated.end(), doubles.begin(), doubles.end());
  }

  const auto encoded_doubles = reger_codec::encodeDoubleColumn(repeated);
  const auto decoded_doubles = reger_codec::decodeDoubleColumn(encoded_doubles);
  assert(decoded_doubles.size() == repeated.size());
  for (std::size_t i = 0; i < repeated.size(); ++i) {
    assert(double_bits(decoded_doubles[i]) == double_bits(repeated[i]));
  }

  std::vector<std::int64_t> double_times;
  std::vector<std::vector<double>> mixed_double_columns(2);
  for (int i = 0; i < 257; ++i) {
    double_times.push_back(1'700'000'000'000LL + ((i * 73) % 257) * 1000LL);
    mixed_double_columns[0].push_back(static_cast<double>((i % 31) - 15) / 100.0);
    mixed_double_columns[1].push_back(std::nextafter(
        3.141592653589793 + static_cast<double>(i) / 17.0,
        std::numeric_limits<double>::infinity()));
  }
  mixed_double_columns[1][0] = -0.0;
  const auto mixed_double_payload =
      reger_codec::encodeDoubleColumns(mixed_double_columns, &double_times);
  const auto mixed_double_decoded = reger_codec::decodeDoubleColumns(mixed_double_payload);
  assert(mixed_double_decoded.size() == mixed_double_columns.size());
  for (std::size_t column = 0; column < mixed_double_columns.size(); ++column) {
    assert(mixed_double_decoded[column].size() == mixed_double_columns[column].size());
    for (std::size_t row = 0; row < mixed_double_columns[column].size(); ++row) {
      assert(double_bits(mixed_double_decoded[column][row]) ==
             double_bits(mixed_double_columns[column][row]));
    }
  }

  std::vector<std::vector<std::int64_t>> int_columns{ints};
  std::vector<std::vector<double>> double_columns{repeated};
  std::vector<std::vector<std::int64_t>> multi_int_columns{
      {},
      {},
      {},
  };
  for (int i = 0; i < 180; ++i) {
    const std::int64_t base = 10000 - i * 17 + (i % 3);
    multi_int_columns[0].push_back(base);
    multi_int_columns[1].push_back(base * 2 + 7);
    multi_int_columns[2].push_back(i % 11);
  }
  const auto encoded_multi = reger_codec::encodeInt64Columns(multi_int_columns);
  const auto decoded_multi = reger_codec::decodeInt64Columns(encoded_multi);
  assert(decoded_multi == multi_int_columns);

  std::vector<std::int64_t> times;
  std::vector<std::vector<std::int64_t>> reordered_columns{{}, {}, {}};
  for (int i = 0; i < 512; ++i) {
    times.push_back(1700000000000000LL + static_cast<std::int64_t>(i) * 60000000LL);
    const std::int64_t a = 50000 - (i / 4) * 11 + (i % 4);
    reordered_columns[0].push_back(a);
    reordered_columns[1].push_back(a * 3 - 17);
    reordered_columns[2].push_back((i % 16) * 5);
  }
  const auto encoded_reordered = reger_codec::encodeInt64Columns(reordered_columns, &times);
  const auto decoded_reordered = reger_codec::decodeInt64Columns(encoded_reordered);
  assert(decoded_reordered == reordered_columns);
  assert(encoded_reordered.size() < static_cast<std::size_t>(512 * 3 * 8 / 5));

  std::vector<std::int64_t> zigzag_times;
  std::vector<std::vector<std::int64_t>> zigzag_columns{{}, {}};
  for (int i = 0; i < 512; ++i) {
    zigzag_times.push_back(i);
    const std::int64_t value = (i % 2 == 0) ? (i / 2) : (1000000 + i / 2);
    zigzag_columns[0].push_back(value);
    zigzag_columns[1].push_back(value * 3 + 11);
  }
  const auto encoded_zigzag = reger_codec::encodeInt64Columns(zigzag_columns, &zigzag_times);
  const auto decoded_zigzag = reger_codec::decodeInt64Columns(encoded_zigzag);
  assert(decoded_zigzag == zigzag_columns);
  assert(encoded_zigzag.size() < static_cast<std::size_t>(512 * 2 * 8 / 4));

  std::vector<std::int64_t> duplicate_times;
  std::vector<std::vector<std::int64_t>> duplicate_time_columns{{}};
  for (int i = 0; i < 512; ++i) {
    duplicate_times.push_back(i / 2);
    duplicate_time_columns[0].push_back(
        i % 2 == 0 ? 1'000'000LL + i / 2 : static_cast<std::int64_t>(i / 2));
  }
  const auto duplicate_time_payload =
      reger_codec::encodeInt64Columns(duplicate_time_columns, &duplicate_times);
  assert(reger_codec::decodeInt64Columns(duplicate_time_payload) == duplicate_time_columns);
  assert(duplicate_time_payload.size() < duplicate_time_columns[0].size() * sizeof(std::int64_t));

  std::vector<std::int64_t> unsorted_times;
  std::vector<std::vector<std::int64_t>> unsorted_time_columns(3);
  for (int i = 0; i < 257; ++i) {
    const std::int64_t time = 1'700'000'000'000LL + ((i * 73) % 257) * 1000LL;
    unsorted_times.push_back(time);
    unsorted_time_columns[0].push_back(time / 1000LL + 17);
    unsorted_time_columns[1].push_back(9'000'000LL - 3 * time / 1000LL);
    unsorted_time_columns[2].push_back(i % 11 == 0 ? i * 101LL : i / 3LL);
  }
  const auto unsorted_time_payload =
      reger_codec::encodeInt64Columns(unsorted_time_columns, &unsorted_times);
  assert(reger_codec::decodeInt64Columns(unsorted_time_payload) == unsorted_time_columns);

  std::vector<std::int64_t> autoregressive;
  autoregressive.reserve(48);
  autoregressive.push_back(2);
  for (int i = 1; i < 48; ++i) autoregressive.push_back(2 * autoregressive.back() + 3);
  const auto encoded_autoregressive = reger_codec::encodeInt64Column(autoregressive);
  assert(reger_codec::decodeInt64Column(encoded_autoregressive) == autoregressive);
  assert(encoded_autoregressive.size() < autoregressive.size() * sizeof(std::int64_t) / 2);

  std::vector<std::vector<std::int64_t>> pruning_columns(20);
  std::vector<std::int64_t> pruning_times;
  for (int row = 0; row < 257; ++row) {
    pruning_times.push_back(1'700'000'000'000LL + row * 1000LL);
    for (std::size_t col = 0; col < pruning_columns.size(); ++col) {
      pruning_columns[col].push_back(
          static_cast<std::int64_t>((row * (col + 3)) % (41 + col * 7)) *
              static_cast<std::int64_t>(col + 1) +
          row / static_cast<std::int64_t>(col + 2));
    }
  }
  setenv("WEB_COMPRESSION_REGER_REORDER_ITERS", "2", 1);
  unsetenv("WEB_COMPRESSION_REGER_EXHAUSTIVE");
  const auto optimized_payload = reger_codec::encodeInt64Columns(pruning_columns, &pruning_times);
  setenv("WEB_COMPRESSION_REGER_EXHAUSTIVE", "1", 1);
  const auto exhaustive_payload = reger_codec::encodeInt64Columns(pruning_columns, &pruning_times);
  assert(optimized_payload == exhaustive_payload);
  assert(reger_codec::decodeInt64Columns(optimized_payload) == pruning_columns);
  unsetenv("WEB_COMPRESSION_REGER_EXHAUSTIVE");
  unsetenv("WEB_COMPRESSION_REGER_REORDER_ITERS");

  std::vector<std::vector<std::int64_t>> high_column_values(33);
  std::vector<std::int64_t> high_column_times;
  for (int row = 0; row < 65; ++row) {
    high_column_times.push_back(10'000LL + ((row * 19) % 65));
    for (std::size_t column = 0; column < high_column_values.size(); ++column) {
      high_column_values[column].push_back(
          static_cast<std::int64_t>((row * (column + 5)) % (23 + column)) +
          static_cast<std::int64_t>(column * 100));
    }
  }
  setenv("WEB_COMPRESSION_REGER_REORDER_ITERS", "0", 1);
  setenv("WEB_COMPRESSION_REGER_HIGH_COLUMN_CANDIDATES", "4", 1);
  unsetenv("WEB_COMPRESSION_REGER_EXHAUSTIVE");
  const auto budgeted_payload =
      reger_codec::encodeInt64Columns(high_column_values, &high_column_times);
  assert(reger_codec::decodeInt64Columns(budgeted_payload) == high_column_values);
  setenv("WEB_COMPRESSION_REGER_EXHAUSTIVE", "1", 1);
  const auto high_column_exhaustive_payload =
      reger_codec::encodeInt64Columns(high_column_values, &high_column_times);
  assert(reger_codec::decodeInt64Columns(high_column_exhaustive_payload) == high_column_values);
  assert(high_column_exhaustive_payload.size() <= budgeted_payload.size());
  unsetenv("WEB_COMPRESSION_REGER_EXHAUSTIVE");
  unsetenv("WEB_COMPRESSION_REGER_HIGH_COLUMN_CANDIDATES");
  unsetenv("WEB_COMPRESSION_REGER_REORDER_ITERS");

  std::vector<long long> cr(4);
  LossAccum loss;
  reger_codec::benchInt64ColumnsReger(int_columns, cr, &loss);
  assert(cr[0] == static_cast<long long>(ints.size() * sizeof(std::int64_t)));
  assert(cr[1] > 0);
  assert(loss.err == 0.0);
  reger_codec::benchDoubleColumnsReger(double_columns, cr, &loss);
  assert(cr[0] == static_cast<long long>(repeated.size() * sizeof(double)));
  assert(cr[1] > 0);
  assert(loss.err == 0.0);
  reger_codec::benchInt64ColumnsReger(multi_int_columns, cr, &loss);
  assert(cr[0] == static_cast<long long>(180 * 3 * sizeof(std::int64_t)));
  assert(cr[1] > 0);
  assert(loss.err == 0.0);
  reger_codec::benchInt64ColumnsReger(reordered_columns, cr, &loss, &times);
  assert(cr[0] == static_cast<long long>(512 * 3 * sizeof(std::int64_t)));
  assert(cr[1] == static_cast<long long>(encoded_reordered.size()));
  assert(loss.err == 0.0);
  reger_codec::benchInt64ColumnsReger(zigzag_columns, cr, &loss, &zigzag_times);
  assert(cr[0] == static_cast<long long>(512 * 2 * sizeof(std::int64_t)));
  assert(cr[1] == static_cast<long long>(encoded_zigzag.size()));
  assert(loss.err == 0.0);

  setenv("WEB_COMPRESSION_REGER_PROFILE", "fast", 1);
  const auto fast_reordered = reger_codec::encodeInt64Columns(reordered_columns, &times);
  assert(reger_codec::decodeInt64Columns(fast_reordered) == reordered_columns);
  const auto fast_zigzag = reger_codec::encodeInt64Columns(zigzag_columns, &zigzag_times);
  assert(reger_codec::decodeInt64Columns(fast_zigzag) == zigzag_columns);
  unsetenv("WEB_COMPRESSION_REGER_PROFILE");
  return 0;
}
