#pragma once

#include "graph.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace operator_recommend135_v1 {

// values contains only non-null values, in source order and in the exact
// little-endian source dtype. missing has ceil(row_count / 8) bytes; a set bit
// means NULL and bits are assigned least-significant first.
struct Block {
  std::string dtype;
  std::uint64_t row_count = 0;
  native_graph_v1::Bytes values;
  native_graph_v1::Bytes missing;
  bool operator==(const Block&) const = default;
};

// A non-owning execution certificate is intentionally not exposed: every
// public preparation call validates the untrusted Block before returning
// metadata or candidate IDs.  Callers can retain the resulting IDs/descriptor
// for planning, but cannot use them to bypass encode_candidate validation.
struct ValidatedBlockDescriptor {
  std::string dtype;
  std::uint32_t dtype_width = 0;
  std::uint64_t row_count = 0;
  std::uint64_t value_count = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t missing_bytes = 0;
};

struct PreparedEligibility {
  ValidatedBlockDescriptor block;
  std::vector<std::string> candidate_ids;
  std::uint64_t validation_passes = 0;
};

// PreparedRegistry parses the immutable candidate metadata and constructs its
// audit catalog once per process.  prepare() validates a Block exactly once,
// then applies only precomputed dtype/nonempty metadata in its candidate loop.
class PreparedRegistry {
 public:
  PreparedRegistry();

  const nlohmann::json& catalog() const noexcept;
  PreparedEligibility prepare(const Block& block) const;
  std::vector<std::string> eligible_candidates(const Block& block) const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

nlohmann::json candidate_catalog();
bool candidate_eligible(std::string_view id, const Block& block);
native_graph_v1::Bytes encode_candidate(std::string_view id, const Block& block);
Block decode_candidate(std::span<const std::uint8_t> frame);

// Runs one strict encode/decode audit outside the timed path.  The
// decoded_from_frame_only field means this API passed only frame bytes to
// decode_candidate; process isolation is established by the campaign runner.
nlohmann::json inspect_candidate_run(std::string_view id, const Block& block,
                                     bool strict = true);

}  // namespace operator_recommend135_v1
