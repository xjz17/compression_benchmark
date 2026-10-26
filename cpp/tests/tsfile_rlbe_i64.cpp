#include "tsfile_encoding/tsfile_encode_p04.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"
#include <cstdio>
#include <vector>
int main() {
  std::vector<std::int64_t> v = {1,2,3,4,5};
  auto enc = tsfile::p04::encode_rlbe_int64(v);
  std::vector<std::int64_t> dec;
  bool ok = tsfile::p04::decode_rlbe_int64(enc, dec);
  printf("ok=%d enc=%zu dec=%zu\n", ok, enc.size(), dec.size());
  for (auto x : dec) printf("%lld ", (long long)x);
  return ok ? 0 : 1;
}
