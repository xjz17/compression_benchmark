#include "benchmark_loss.hpp"
#include "benchmark_columns.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"
#include "corad_tricklet_float_codec.hpp"
#include "leco_double_codec.hpp"
#include "native_lz_family.hpp"
#include "simple8b_int_codec.hpp"
#include "subcolumn_codec.hpp"
#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {

double from_bits(std::uint64_t bits) {
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool bit_equal(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

}  // namespace

int main() {
  std::vector<std::int64_t> integers = {
      std::numeric_limits<std::int64_t>::min(), -1, 0, 1,
      std::numeric_limits<std::int64_t>::max()};
  for (std::int64_t i = 0; i < 10050; ++i) {
    integers.push_back((i % 17 == 0) ? std::numeric_limits<std::int64_t>::max() - i
                                    : (i * 6364136223846793005ULL + 1442695040888963407ULL));
  }
  const std::vector<double> values = {
      0.0, -0.0, 1.0, -1.0, 0.1, std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
      from_bits(0x7ff8000000000042ULL), from_bits(0xfff8000000001234ULL),
      from_bits(0x3fd5555555555555ULL), from_bits(0x43b0000000000001ULL)};
  const std::vector<double> finite_values = {
      0.0, -0.0, 1.0, -1.0, 0.125, 0.25, -2.5, 12.5, -98765.25,
      -2.664974, 0.380366859955132, -73.992028196156};

  auto check_direct = [&](const char* name, const std::vector<std::uint8_t>& encoded,
                          auto decode) {
    std::vector<double> decoded;
    if (!decode(encoded, decoded) || !bit_equal(values, decoded)) {
      std::cerr << name << " failed bit-exact round trip\n";
      return false;
    }
    return true;
  };

  bool ok = true;
  auto check_int = [&](const char* name, const std::vector<std::uint8_t>& encoded, auto decode) {
    std::vector<std::int64_t> decoded;
    if (!decode(encoded, decoded) || decoded != integers) {
      std::cerr << name << " failed int64 round trip\n";
      return false;
    }
    return true;
  };
  ok &= check_int("RLE", tsfile::p02::encode_rle_int64(integers),
                  tsfile::p02::decode_rle_int64);
  ok &= check_int("SPRINTZ", tsfile::p04::encode_sprintz_int64(integers),
                  tsfile::p04::decode_sprintz_int64);
  ok &= check_int("RLBE", tsfile::p04::encode_rlbe_int64(integers),
                  tsfile::p04::decode_rlbe_int64);
  ok &= check_direct("RLE", tsfile::p02::encode_rle_double(values, 4),
                     tsfile::p02::decode_rle_double);
  ok &= check_direct("SPRINTZ", tsfile::p04::encode_sprintz_double(values),
                     tsfile::p04::decode_sprintz_double);
  ok &= check_direct("RLBE", tsfile::p04::encode_rlbe_double(values),
                     tsfile::p04::decode_rlbe_double);

  const auto buff_encoded = buff_kernel::encode_doubles(values, 10000);
  const auto buff_decoded = buff_kernel::decode_doubles(buff_encoded);
  if (!bit_equal(values, buff_decoded.values)) {
    std::cerr << "BUFF failed bit-exact round trip\n";
    ok = false;
  }

  const std::vector<std::vector<double>> columns{values};
  const std::vector<std::vector<double>> finite_columns{finite_values};
  const std::vector<int> max_points{4};
  const std::vector<int> finite_max_points{4};
  auto check_bench = [&](const char* name, std::size_t count, auto bench) {
    std::vector<long long> result;
    LossAccum loss;
    bench(result, loss);
    const long long expected = static_cast<long long>(count * sizeof(double));
    if (result.size() != 4 || result[0] != expected || loss.err != 0.0 ||
        loss.ref != static_cast<double>(count)) {
      std::cerr << name << " failed benchmark round trip coverage\n";
      ok = false;
    }
  };
  check_bench("LECO", values.size(), [&](auto& result, auto& loss) {
    leco_double::benchDoubleColumnsLikeJava(columns, result, &max_points, &loss);
  });
  check_bench("Simple8b", finite_values.size(), [&](auto& result, auto& loss) {
    simple8b_int::benchDoubleColumnsSimple8b(finite_columns, result, &finite_max_points, &loss);
  });
  check_bench("Sub-column", values.size(), [&](auto& result, auto& loss) {
    subcolumn_codec::benchDoubleColumnsSubcolumn(columns, result, &max_points, &loss);
  });

  const auto delta_input = scale_double_columns_to_int64(columns, &max_points);
  if (delta_input.size() != 1 || delta_input.front().size() != values.size()) {
    std::cerr << "DELTA_BINARY_PACKED dropped an unscalable double column\n";
    ok = false;
  } else {
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &values[i], sizeof(bits));
      if (delta_input.front()[i] != static_cast<std::int64_t>(bits)) {
        std::cerr << "DELTA_BINARY_PACKED fallback changed an IEEE-754 bit pattern\n";
        ok = false;
        break;
      }
    }
  }

  const std::vector<std::vector<std::int64_t>> constant_int_columns{
      std::vector<std::int64_t>(10000, 1)};
  check_bench("CORAD constant int", constant_int_columns.front().size(),
              [&](auto& result, auto& loss) {
                corad_tricklet_float::benchInt64ColumnsCoradTricklet(
                    constant_int_columns, result, &loss);
              });

  std::vector<char> patch_bytes(10000 * sizeof(double));
  for (std::size_t i = 0; i < patch_bytes.size(); ++i) {
    patch_bytes[i] = static_cast<char>((i * 17 + i / 97) & 0xff);
  }
  for (const char* algorithm : {"elf_on_chimp", "elf_plus", "fpcompress_dpratio",
                                "fpcompress_dpspeed"}) {
    std::vector<long long> result;
    LossAccum loss;
    bench_native_lz_family_bytes({patch_bytes}, algorithm, result, &loss);
    if (result.size() != 4 || result[0] != static_cast<long long>(patch_bytes.size()) ||
        loss.err != 0.0 || loss.ref != static_cast<double>(patch_bytes.size())) {
      std::cerr << algorithm << " failed native-family patch round trip\n";
      ok = false;
    }
  }

  return ok ? 0 : 1;
}
