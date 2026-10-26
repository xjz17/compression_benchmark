#pragma once
#include "graph.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace pipeline_test {
using native_graph_v1::Bytes;
inline void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
inline void put(Bytes& out, std::uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i) out.push_back(value >> (8 * i));
}
inline void text(Bytes& out, const std::string& value) {
  put(out, value.size(), 4); out.insert(out.end(), value.begin(), value.end());
}
inline void write(const std::filesystem::path& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  check(bool(out), "cannot write test fixture");
}
inline Bytes read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  check(bool(in), "cannot open fixture");
  const auto size = in.tellg(); check(size >= 0 && size <= 32 * 1024 * 1024, "oversized fixture");
  Bytes out(static_cast<std::size_t>(size)); in.seekg(0);
  in.read(reinterpret_cast<char*>(out.data()), out.size());
  check(bool(in), "cannot read fixture"); return out;
}
// Test-only expected-root representation. It is not a compressed frame and is
// never supplied to the decoder until after decode() returns.
inline Bytes expected(const std::vector<native_graph_v1::RootInput>& roots) {
  Bytes out; put(out, roots.size(), 4);
  for (const auto& root : roots) {
    put(out, static_cast<unsigned>(root.stream.kind), 1);
    put(out, root.stream.width, 4); put(out, root.original_numeric_payload, 1);
    text(out, root.logical_view); put(out, root.stream.data.size(), 8);
    out.insert(out.end(), root.stream.data.begin(), root.stream.data.end());
  }
  return out;
}
}  // namespace pipeline_test
