#pragma once

#include "../../native_baseline86_v1/baseline.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace unified_compression_v1::bos_bfe_bridge {

inline constexpr const char* kRouteId = "bos-bfe-ngf-hybrid-v2";
inline constexpr const char* kIntegerRouteId = "project_bos2_partitioned";
inline constexpr const char* kBfePolicy =
    "bos-float-exception-ngf-profile-v4";
inline constexpr const char* kDsc3Policy =
    "baseline86-decimal-scaling-int64-exact-v3";
inline constexpr std::uint64_t kMaximumBlockValues = 10000;

struct FrameInfo {
  std::string dtype;
  std::uint64_t row_count = 0;
  std::uint16_t whole_column_decimal_places = 0;
  std::string selected_frontend;
  std::uint64_t block_count = 0;
  std::uint64_t raw_block_count = 0;
  std::uint64_t bfe_block_count = 0;
  std::uint64_t dsc3_block_count = 0;
  std::uint64_t bos2_block_count = 0;
};

// int64 uses the version-isolated Project-150 BOS2 kernel.  Floating columns
// are scanned in bounded 10,000-value pieces to obtain the exact column
// maximum P, then encoded as independently decodable complete-NGF blocks using
// that same P.  A one-block float column retains the reviewed NGF wire byte for
// byte; multiple float blocks use the checksummed BHC1 envelope.
native_baseline86_v1::Bytes encode(
    const native_baseline86_v1::Block& block);

native_baseline86_v1::Block decode(
    std::span<const std::uint8_t> frame);

// True only for the exact reviewed BOS2, DSC3/BOS2, or BFE/BOS2
// dispatcher/policy/action/configuration tuples produced by encode().  It
// never accepts an unrelated baseline method or configuration frame.
bool is_frame(std::span<const std::uint8_t> frame);

FrameInfo inspect(std::span<const std::uint8_t> frame);

}  // namespace unified_compression_v1::bos_bfe_bridge
