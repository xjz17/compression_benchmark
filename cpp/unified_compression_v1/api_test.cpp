#include "api.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>

namespace uc = unified_compression_v1;

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

template <class F> void rejected(F f, std::string_view code = {}) {
  try { f(); }
  catch (const uc::Error& e) {
    if (!code.empty()) require(e.code() == code, e.what());
    return;
  }
  throw std::runtime_error("invalid request unexpectedly succeeded");
}

void metrics(const uc::CompressionResult& result, const uc::Column& source,
             std::size_t samples = 1) {
  require(result.bit_exact, "missing bit-exact verification");
  require(result.compressed_bytes == result.compressed_data.size(), "frame bytes mismatch");
  require(result.original_numeric_bytes == source.values.size(), "source bytes mismatch");
  require(result.compression_samples_ns.size() == samples &&
          result.decompression_samples_ns.size() == samples, "sample count mismatch");
  if (source.values.empty()) require(!result.compression_ratio, "empty ratio must be null");
  else require(result.compression_ratio &&
      *result.compression_ratio == double(result.compressed_bytes) / source.values.size(),
      "ratio must charge the complete frame");
  require(uc::decompress_column(result.compressed_data) == source, "frame-only roundtrip mismatch");
}

int main() {
  try {
    uc::Registry registry;
    require(registry.methods().size() == 218, "must register 86 algorithms, 102 operators and 30 combinations");
    std::set<std::string> names;
    std::size_t algorithms = 0, operators = 0, combinations = 0;
    for (const auto& method : registry.methods()) {
      require(names.insert(method.name).second, "duplicate canonical method");
      algorithms += method.kind == "algorithm";
      operators += method.kind == "operator";
      combinations += method.kind == "combo";
    }
    require(algorithms == 86 && operators == 102 && combinations == 30, "inventory kind count mismatch");
    require(registry.resolve(" zstd ").name == "algorithm:Zstd", "casefold alias failed");
    require(registry.resolve("A013").name == "operator:A013", "operator alias failed");
    rejected([&] { registry.resolve("does-not-exist"); }, "unknown_method");

    const std::vector<std::int64_t> integer_values{
        std::numeric_limits<std::int64_t>::min(), -1, 0,
        std::numeric_limits<std::int64_t>::max()};
    auto integers = uc::make_column(std::span<const std::int64_t>(integer_values));
    auto measured = registry.compress(integers, "Zstd", {1, 3});
    metrics(measured, integers, 3);

    auto nullable = integers;
    nullable.row_count = 6;
    nullable.missing = {0x12};
    metrics(registry.compress(nullable, "LZ4"), nullable);

    const std::vector<double> floats{
        0.0, -0.0, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::bit_cast<double>(UINT64_C(0x7ff8000000001234))};
    auto floating = uc::make_column(std::span<const double>(floats));
    metrics(registry.compress(floating, "Zstd"), floating);
    const std::vector<float> f32{1.25F, -0.0F,
        std::bit_cast<float>(UINT32_C(0x7fc01234))};
    auto float32 = uc::make_column(std::span<const float>(f32));
    metrics(registry.compress(float32, "Zstd"), float32);
    const std::vector<std::int32_t> i32{INT32_MIN, 0, INT32_MAX};
    auto int32 = uc::make_column(std::span<const std::int32_t>(i32));
    metrics(registry.compress(int32, "Zstd"), int32);

    uc::Column empty{"float64", 0, {}, {}};
    uc::Column all_null{"int64", 3, {}, {7}};
    metrics(registry.compress(empty, "Zstd"), empty);
    metrics(registry.compress(all_null, "Zstd"), all_null);
    metrics(registry.compress(empty, "A048"), empty);

    const std::vector<std::int64_t> ordinary{10, 11, 11, 13, 19, 20};
    auto for_column = uc::make_column(std::span<const std::int64_t>(ordinary));
    metrics(registry.compress(for_column, "A013"), for_column);
    metrics(registry.compress(for_column, "A048"), for_column);

    for (const auto& disabled : {"PPMD8", "GLZA", "A106", "A107", "A108", "A109", "A110"})
      rejected([&] { registry.compress(integers, disabled); }, "method_disabled");
    rejected([&] { registry.compress(float32, "A013"); }, "domain_rejected");
    auto broken = nullable;
    broken.missing = {0x80};
    rejected([&] { registry.compress(broken, "Zstd"); });
    broken = integers;
    broken.values.pop_back();
    rejected([&] { registry.compress(broken, "Zstd"); });
    rejected([&] { registry.compress(integers, "Zstd", {0, 0}); }, "invalid_options");

    auto corrupt = measured.compressed_data;
    corrupt[corrupt.size() / 2] ^= 1;
    rejected([&] { uc::decompress_column(corrupt); }, "invalid_frame");
    corrupt = measured.compressed_data;
    corrupt.resize(30);
    rejected([&] { uc::decompress_column(corrupt); }, "invalid_frame");

    // Even a correctly checksummed outer frame may not rename its inner codec.
    auto forged = measured.compressed_data;
    const std::string impostor = "algorithm:Zlib";
    const std::string original = "algorithm:Zstd";
    require(impostor.size() == original.size(), "test fixture name width mismatch");
    std::copy(impostor.begin(), impostor.end(), forged.begin() + 20);
    std::array<unsigned char, 32> hash{};
    unsigned hash_size = 0;
    require(EVP_Digest(forged.data(), forged.size() - 32, hash.data(), &hash_size,
                       EVP_sha256(), nullptr) == 1 && hash_size == 32,
            "fixture digest failed");
    std::copy(hash.begin(), hash.end(), forged.end() - 32);
    rejected([&] { uc::decompress_column(forged); }, "inner_binding_mismatch");
    std::cout << "unified API: 218 registered; typed/null/boundary/frame/timing checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
