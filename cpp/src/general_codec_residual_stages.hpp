#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::general_residual {

// Canonical vendored-library encoders used by the 86-method audit.
std::vector<std::uint8_t> brotli_encode(std::span<const std::uint8_t> raw);
std::vector<std::uint8_t> brotli_decode(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> zstd_encode(std::span<const std::uint8_t> raw);
std::vector<std::uint8_t> zstd_decode(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> blosc1_encode(std::span<const std::uint8_t> raw,
                                        int type_size);
std::vector<std::uint8_t> blosc1_decode(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> blosc2_encode(std::span<const std::uint8_t> raw,
                                        int type_size);
std::vector<std::uint8_t> blosc2_decode(std::span<const std::uint8_t> wire,
                                        int type_size);
std::vector<std::uint8_t> bzip2_encode(std::span<const std::uint8_t> raw);
std::vector<std::uint8_t> bzip2_decode(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> gzip_encode(std::span<const std::uint8_t> raw, int level = 6);
std::vector<std::uint8_t> gzip_decode(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> zlib_encode(std::span<const std::uint8_t> raw, int level = 9);
std::vector<std::uint8_t> zlib_decode(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> xz_encode(std::span<const std::uint8_t> raw);
std::vector<std::uint8_t> xz_decode(std::span<const std::uint8_t> wire);

// Each residual action maps the output of the explicitly split reference
// prefix to the exact original codec wire, and reverses that mapping.
std::vector<std::uint8_t> brotli_forward(std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> brotli_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> zstd_forward(std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> zstd_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> blosc1_forward(std::span<const std::uint8_t> prefix,
                                         int type_size);
std::vector<std::uint8_t> blosc1_inverse(std::span<const std::uint8_t> wire,
                                         int type_size);
std::vector<std::uint8_t> blosc2_forward(std::span<const std::uint8_t> prefix,
                                         int type_size);
std::vector<std::uint8_t> blosc2_inverse(std::span<const std::uint8_t> wire,
                                         int type_size);
std::vector<std::uint8_t> bzip2_forward(std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> bzip2_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> bzip2_optimal_forward(std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> bzip2_optimal_inverse(std::span<const std::uint8_t> wire);

// gzip/zlib/XZ residuals return the exact container packet consumed by the
// already split container_frame action, so the final action still owns the
// original header, trailer, checksum, and stream framing bytes.
// Level is an input parameter in [0, 9]; defaults retain each legacy wire.
std::vector<std::uint8_t> gzip_forward(std::span<const std::uint8_t> prefix, int level = 6);
std::vector<std::uint8_t> gzip_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> zlib_forward(std::span<const std::uint8_t> prefix, int level = 9);
std::vector<std::uint8_t> zlib_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> xz_forward(std::span<const std::uint8_t> prefix);
std::vector<std::uint8_t> xz_inverse(std::span<const std::uint8_t> packet);

}  // namespace exact_actions::general_residual
