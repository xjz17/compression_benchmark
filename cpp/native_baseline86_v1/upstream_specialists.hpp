#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace native_baseline86_v1::upstream_specialists {

using Bytes = std::vector<std::uint8_t>;

// These are payload kernels.  The baseline86 NGF1 layer remains responsible
// for dtype/row-count/missing-mask metadata and the complete-frame checksum.
enum class Admission {
  direct_upstream,
  direct_project_generic,
  direct_exact_subset,
};

struct Profile {
  std::string label;
  std::string implementation;
  std::string source;
  std::string configuration;
  std::vector<std::string> dtypes;
  Admission admission;
  std::string limitations;
};

class NotApplicable final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

const std::vector<Profile>& profiles();
const Profile& profile(std::string_view label);

// Returns a deterministic, self-identifying USP1 payload.  decode() verifies
// the method id, exact expected byte count, payload CRC and reconstructed CRC.
// CORAD and BUFF deliberately fail with NotApplicable when their native,
// lossy/bounded representation cannot reproduce every input bit.
Bytes encode(std::string_view label, std::span<const std::uint8_t> raw);
Bytes decode(std::string_view label, std::span<const std::uint8_t> payload,
             std::size_t expected_bytes);

}  // namespace native_baseline86_v1::upstream_specialists
