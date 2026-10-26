#include "tsfile_encoding/tsfile_encode_p03.hpp"
#include "tsfile_encoding/tsfile_decode_p03.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"

#include <cstdio>
#include <span>
#include <vector>

static bool test_gorilla() {
  std::vector<std::int64_t> v = {1, 2, 3, 4, 5};
  auto enc = tsfile::p03::encode_gorilla_int64(v);
  std::vector<std::int64_t> dec;
  if (!tsfile::p03::decode_gorilla_int64(enc, dec)) return false;
  return dec.size() >= v.size();
}

static bool test_chimp() {
  std::vector<std::int64_t> v = {1, 2, 3, 4, 5};
  auto enc = tsfile::p03::encode_chimp_int64(v);
  std::vector<std::int64_t> dec;
  return tsfile::p03::decode_chimp_int64(enc, dec) && dec.size() >= v.size();
}

static bool test_chimp_i64_ramp(int n) {
  std::vector<std::int64_t> v;
  v.reserve(static_cast<std::size_t>(n));
  for (int i = 1; i <= n; ++i) {
    v.push_back(static_cast<std::int64_t>(i));
  }
  auto enc = tsfile::p03::encode_chimp_int64(v);
  std::vector<std::int64_t> dec;
  const bool ok = tsfile::p03::decode_chimp_int64(enc, dec);
  std::printf("chimp_i64_ramp n=%d enc=%zu ok=%d dec=%zu\n", n, enc.size(), ok ? 1 : 0, dec.size());
  return ok && dec.size() >= v.size();
}

static bool test_chimp_double_ramp() {
  std::vector<double> v;
  v.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    v.push_back(static_cast<double>(i) * 1.25);
  }
  auto enc = tsfile::p03::encode_chimp_double(std::span<const double>(v.data(), v.size()));
  std::vector<double> dec;
  if (!tsfile::p03::decode_chimp_double(enc, dec)) {
    return false;
  }
  return dec.size() >= v.size();
}

static bool test_sprintz() {
  std::vector<std::int64_t> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto enc = tsfile::p04::encode_sprintz_int64(v);
  std::vector<std::int64_t> dec;
  return tsfile::p04::decode_sprintz_int64(enc, dec) && !dec.empty();
}

static bool test_rlbe() {
  std::vector<std::int64_t> v = {1, 2, 3, 4, 5};
  auto enc = tsfile::p04::encode_rlbe_int64(v);
  std::vector<std::int64_t> dec;
  return tsfile::p04::decode_rlbe_int64(enc, dec) && dec.size() == v.size();
}

int main() {
  bool g = test_gorilla();
  bool c = test_chimp();
  bool cr = test_chimp_i64_ramp(10000);
  bool cd = test_chimp_double_ramp();
  bool s = test_sprintz();
  bool r = test_rlbe();
  std::printf("gorilla=%d chimp=%d chimp_ramp=%d chimp_d=%d sprintz=%d rlbe=%d\n", g ? 1 : 0,
              c ? 1 : 0, cr ? 1 : 0, cd ? 1 : 0, s ? 1 : 0, r ? 1 : 0);
  return (g && c && cr && cd && s && r) ? 0 : 1;
}
