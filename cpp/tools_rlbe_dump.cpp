// Optional local debug: same 64 floats as golden test / org.example.RlbeFloatDump — not built by CMake.

#include "tsfile_encoding/tsfile_encode_p04.hpp"

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  std::vector<float> f;
  f.reserve(64);
  for (int i = 0; i < 64; i++) {
    f.push_back(1.0f + static_cast<float>(i) * 0.03125f);
  }
  auto v = tsfile::p04::encode_rlbe_float(f);
  std::cout << "size " << v.size() << '\n';
  const char* out_path = (argc >= 2 && argv[1][0]) ? argv[1] : "rlbe_cpp_out.bin";
  std::ofstream o(out_path, std::ios::binary);
  if (!o) {
    std::cerr << "cannot open " << out_path << '\n';
    return 1;
  }
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
  return 0;
}
