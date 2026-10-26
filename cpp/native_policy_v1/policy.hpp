#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace weighted_grpo::native_policy_v1 {
inline constexpr const char* feature_schema = "task-state-plus-simplex-weights-v1";
inline constexpr const char* checkpoint_schema = "weight-conditioned-grpo-policy-v1";
inline constexpr std::size_t maximum_model_bytes = 256 * 1024 * 1024;

// The integrating graph runtime supplies these values independently of the model.
// No action remapping, feature migration, or inference fallback is performed.
struct RuntimeIdentity {
    std::string registry_fingerprint, feature_schema;
    std::uint32_t action_count{}, stop_action{}, task_feature_dim{}, state_feature_dim{};
    bool operator==(const RuntimeIdentity&) const = default;
    std::size_t input_dim() const { return std::size_t(task_feature_dim) + state_feature_dim + 3; }
};
struct Identity : RuntimeIdentity {
    std::array<std::uint32_t, 2> hidden_sizes{};
    std::vector<std::pair<std::string, std::string>> data_fingerprints, root_fingerprints;
};
struct Origin {
    std::string checkpoint_schema, checkpoint_payload_sha256, checkpoint_file_sha256;
};
struct Result {
    std::vector<double> logits, probabilities;
    // Legal IDs only; descending logit, exact ties in ascending unchanged action ID.
    std::vector<std::uint32_t> ordered_actions;
};

class Policy {
public:
    static Policy load(const std::filesystem::path&, const RuntimeIdentity& expected);
    // Inspection validates the complete model but does not permit inference until
    // load() has checked an independently supplied runtime identity.
    static std::pair<Identity, Origin> inspect(const std::filesystem::path&);
    const Identity& identity() const noexcept { return identity_; }
    const Origin& origin() const noexcept { return origin_; }
    Result infer(std::span<const double> features, std::span<const std::uint8_t> legal_mask) const;
    // Return the highest-logit legal action without materializing all logits,
    // probabilities, or a sorted ranking. Legal IDs must be ascending/unique.
    std::uint32_t greedy(std::span<const double> features,
                         std::span<const std::uint32_t> legal_action_ids) const;
private:
    Policy() = default;
    static Policy parse(const std::filesystem::path&);
    Identity identity_;
    Origin origin_;
    std::array<std::vector<double>, 6> parameters_;
    // W3 transposed once by load(), so a legal action's 128 weights are contiguous.
    std::vector<double> output_weights_by_action_;
};

// Portable standard SHA-256, used for integrity (not a signature/authentication).
std::array<std::uint8_t, 32> sha256(std::span<const std::uint8_t>);
} // namespace weighted_grpo::native_policy_v1
