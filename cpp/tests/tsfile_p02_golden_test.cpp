// Compares preserved TSFile C++ encoders against frozen Java goldens (all twelve: TS_2DIFF, RLE, GORILLA, CHIMP,
// SPRINTZ, RLBE × INT32/FLOAT). RLBE float matches tsfile 2.0.2 FloatRLBE (float deltas).
// Public RLE/RLBE now emit versioned formats. Never regenerate these files to
// make a current writer pass: the explicit legacy writers retain byte parity.

#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_decode_p03.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p03.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#if defined(TSFILE_GOLDEN_DIR)
#define TSFILE_GOLDEN_DIR_STR TSFILE_GOLDEN_DIR
#else
#define TSFILE_GOLDEN_DIR_STR ""
#endif

static std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto sz = f.tellg();
  f.seekg(0);
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
  f.read(reinterpret_cast<char*>(buf.data()), sz);
  return buf;
}

static std::vector<std::int32_t> golden_int_column() {
  std::vector<std::int32_t> v;
  v.reserve(64);
  for (int i = 0; i < 64; i++) v.push_back(10000 + i * 3);
  return v;
}

static std::vector<float> golden_float_column() {
  std::vector<float> v;
  v.reserve(64);
  for (int i = 0; i < 64; i++) v.push_back(1.0f + static_cast<float>(i) * 0.03125f);
  return v;
}

// Matches convert_float_to_int in tsfile_encode_p02 (max_point 0: llround to integer).
static std::vector<float> expected_ts2diff_floats(std::span<const float> raw) {
  std::vector<float> v;
  v.reserve(raw.size());
  for (float f : raw) v.push_back(static_cast<float>(std::llround(static_cast<double>(f))));
  return v;
}

// Matches RLE float default precision 2 in encode_p02.
static std::vector<float> expected_rle_floats(std::span<const float> raw) {
  constexpr int kMaxPoint = 2;
  const double scale = std::pow(10.0, kMaxPoint);
  std::vector<float> v;
  v.reserve(raw.size());
  for (float f : raw) {
    auto q = static_cast<std::int32_t>(std::llround(static_cast<double>(f) * scale));
    v.push_back(static_cast<float>(static_cast<double>(q) / scale));
  }
  return v;
}

static bool same(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b, const char* name) {
  if (a.size() != b.size()) {
    std::cerr << name << ": size " << a.size() << " vs golden " << b.size() << '\n';
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      std::cerr << name << ": mismatch at " << i << " got " << static_cast<int>(a[i]) << " want "
                << static_cast<int>(b[i]) << '\n';
      return false;
    }
  }
  return true;
}

static bool same_int_col(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b,
                         const char* name) {
  if (a.size() != b.size()) {
    std::cerr << name << ": decode len " << a.size() << " want " << b.size() << '\n';
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      std::cerr << name << ": decode diff at " << i << " got " << a[i] << " want " << b[i] << '\n';
      return false;
    }
  }
  return true;
}

static bool same_float_col(const std::vector<float>& a, const std::vector<float>& b, const char* name) {
  if (a.size() != b.size()) {
    std::cerr << name << ": decode len " << a.size() << " want " << b.size() << '\n';
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      std::cerr << name << ": decode float diff at " << i << '\n';
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  std::string root;
  if (argc >= 2) {
    root = argv[1];
  } else {
    const char* e = std::getenv("TSFILE_GOLDEN_DIR");
    if (e && e[0]) root = e;
  }
  if (root.empty()) {
    std::string d(TSFILE_GOLDEN_DIR_STR);
    if (!d.empty()) root = std::move(d);
  }
  if (root.empty()) {
    std::cerr << "Usage: tsfile_p02_golden_test <golden/tsfile/dir>\n";
    return 2;
  }
  if (!root.empty() && root.back() == '/') root.pop_back();

  auto ints = golden_int_column();
  auto floats = golden_float_column();
  auto exp_ts2diff_f = expected_ts2diff_floats(floats);
  auto exp_rle_f = expected_rle_floats(floats);

  int fails = 0;
  const char* cases[] = {"TS_2DIFF_INT32_default.bin",
                         "TS_2DIFF_FLOAT_default.bin",
                         "RLE_INT32_default.bin",
                         "RLE_FLOAT_default.bin",
                         "GORILLA_INT32_default.bin",
                         "GORILLA_FLOAT_default.bin",
                         "CHIMP_INT32_default.bin",
                         "CHIMP_FLOAT_default.bin",
                         "SPRINTZ_INT32_default.bin",
                         "SPRINTZ_FLOAT_default.bin",
                         "RLBE_INT32_default.bin",
                         "RLBE_FLOAT_default.bin"};

  for (const char* fn : cases) {
    std::vector<std::uint8_t> golden = read_file(root + "/" + fn);
    if (golden.empty()) {
      std::cerr << "Missing " << root << "/" << fn << '\n';
      fails++;
      continue;
    }
    std::vector<std::uint8_t> got;
    std::string label = fn;
    if (label == "TS_2DIFF_INT32_default.bin")
      got = tsfile::p02::encode_ts_2diff_int32(ints);
    else if (label == "TS_2DIFF_FLOAT_default.bin")
      got = tsfile::p02::encode_ts_2diff_float(floats);
    else if (label == "RLE_INT32_default.bin")
      got = tsfile::p02::encode_rle_int32_legacy(ints);
    else if (label == "RLE_FLOAT_default.bin")
      got = tsfile::p02::encode_rle_float_legacy(floats);
    else if (label == "GORILLA_INT32_default.bin")
      got = tsfile::p03::encode_gorilla_int32(ints);
    else if (label == "GORILLA_FLOAT_default.bin")
      got = tsfile::p03::encode_gorilla_float(floats);
    else if (label == "CHIMP_INT32_default.bin")
      got = tsfile::p03::encode_chimp_int32(ints);
    else if (label == "CHIMP_FLOAT_default.bin")
      got = tsfile::p03::encode_chimp_float(floats);
    else if (label == "SPRINTZ_INT32_default.bin")
      got = tsfile::p04::encode_sprintz_int32_legacy(ints);
    else if (label == "SPRINTZ_FLOAT_default.bin")
      got = tsfile::p04::encode_sprintz_float_legacy(floats);
    else if (label == "RLBE_INT32_default.bin")
      got = tsfile::p04::encode_rlbe_int32_legacy(ints);
    else if (label == "RLBE_FLOAT_default.bin")
      got = tsfile::p04::encode_rlbe_float_legacy(floats);

    if (got.empty() && !golden.empty()) {
      std::cerr << fn << ": encoder returned empty\n";
      fails++;
      continue;
    }
    if (!same(got, golden, fn)) {
      fails++;
      continue;
    }

    std::vector<std::int32_t> di;
    std::vector<float> df;
    bool ok_dec = false;
    if (label == "TS_2DIFF_INT32_default.bin")
      ok_dec = tsfile::p02::decode_ts_2diff_int32(got, di) && same_int_col(di, ints, fn);
    else if (label == "RLE_INT32_default.bin")
      ok_dec = tsfile::p02::decode_rle_int32(got, di) && same_int_col(di, ints, fn);
    else if (label == "GORILLA_INT32_default.bin")
      ok_dec = tsfile::p03::decode_gorilla_int32(got, di) && same_int_col(di, ints, fn);
    else if (label == "CHIMP_INT32_default.bin")
      ok_dec = tsfile::p03::decode_chimp_int32(got, di) && same_int_col(di, ints, fn);
    else if (label == "SPRINTZ_INT32_default.bin")
      ok_dec = tsfile::p04::decode_sprintz_int32(got, di) && same_int_col(di, ints, fn);
    else if (label == "RLBE_INT32_default.bin")
      ok_dec = tsfile::p04::decode_rlbe_int32(got, di) && same_int_col(di, ints, fn);
    else if (label == "TS_2DIFF_FLOAT_default.bin")
      ok_dec = tsfile::p02::decode_ts_2diff_float(got, df) && same_float_col(df, exp_ts2diff_f, fn);
    else if (label == "RLE_FLOAT_default.bin")
      ok_dec = tsfile::p02::decode_rle_float(got, df) && same_float_col(df, exp_rle_f, fn);
    else if (label == "GORILLA_FLOAT_default.bin")
      ok_dec = tsfile::p03::decode_gorilla_float(got, df) && same_float_col(df, floats, fn);
    else if (label == "CHIMP_FLOAT_default.bin")
      ok_dec = tsfile::p03::decode_chimp_float(got, df) && same_float_col(df, floats, fn);
    else if (label == "SPRINTZ_FLOAT_default.bin")
      ok_dec = tsfile::p04::decode_sprintz_float(got, df) && same_float_col(df, floats, fn);
    else if (label == "RLBE_FLOAT_default.bin")
      ok_dec = tsfile::p04::decode_rlbe_float(got, df) && same_float_col(df, floats, fn);

    if (!ok_dec) {
      std::cerr << fn << ": round-trip decode mismatch or decoder error\n";
      fails++;
    }
  }

  if (fails) {
    std::cerr << fails << " failure(s)\n";
    return 1;
  }
  std::cout << "OK: all twelve TSFile encodings match Java goldens and round-trip decode\n";
  return 0;
}
