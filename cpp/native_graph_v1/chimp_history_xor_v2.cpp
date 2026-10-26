#include "chimp_history_xor_v2.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace native_graph_v1::chimp_history_xor_v2 {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'C', 'H', 'X', '2'};
constexpr std::size_t kMetadataBytes = 24;
constexpr std::size_t kHistorySize = 128;
constexpr std::size_t kKeyCount = 1U << 14U;
constexpr std::uint64_t kKeyMask = kKeyCount - 1;
constexpr int kTrailingThreshold = 13;

[[noreturn]] void bad(const char* why) { throw std::invalid_argument(why); }

void require(bool condition, const char* why) {
  if (!condition) bad(why);
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint64_t read_u64_le(std::span<const std::uint8_t> input,
                          std::size_t& position) {
  require(position <= input.size() && input.size() - position >= 8,
          "truncated Chimp history128 uint64");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(input[position++]) << shift;
  return value;
}

void validate_profile(Profile profile) {
  require(profile == Profile::Int64 || profile == Profile::Float64,
          "invalid Chimp history128 profile");
}

void validate_encoded_shape(const Encoded& encoded) {
  validate_profile(encoded.profile);
  const std::uint64_t residual_count =
      encoded.value_count == 0 ? 0 : encoded.value_count - 1;
  require(residual_count <= std::numeric_limits<std::size_t>::max(),
          "Chimp history128 value count exceeds address space");
  const auto expected = static_cast<std::size_t>(residual_count);
  require(encoded.reference_slots.size() == expected &&
              encoded.xor_residuals.size() == expected,
          "Chimp history128 port length mismatch");
  require(encoded.value_count != 0 || encoded.first_value == 0,
          "non-canonical empty Chimp history128 metadata");
  for (const auto slot : encoded.reference_slots)
    require(slot < kHistorySize, "Chimp history128 reference slot out of range");
}

std::vector<std::uint8_t> residual_bytes(
    std::span<const std::uint64_t> residuals) {
  require(residuals.size() <= std::numeric_limits<std::size_t>::max() / 8,
          "Chimp history128 residual stream exceeds address space");
  std::vector<std::uint8_t> out;
  out.reserve(residuals.size() * 8);
  for (const auto residual : residuals) append_u64_le(out, residual);
  return out;
}

}  // namespace

Encoded encode_words(std::span<const std::uint64_t> words, Profile profile) {
  validate_profile(profile);
  Encoded encoded;
  encoded.profile = profile;
  encoded.value_count = words.size();
  if (words.empty()) return encoded;

  encoded.first_value = words.front();
  encoded.reference_slots.reserve(words.size() - 1);
  encoded.xor_residuals.reserve(words.size() - 1);

  std::array<std::uint64_t, kHistorySize> stored{};
  std::array<std::size_t, kKeyCount> indices{};
  stored[0] = words.front();
  indices[words.front() & kKeyMask] = 0;

  // index is the number of residual records already emitted.  Therefore
  // index % 128 is also the immediately preceding value's ring slot.
  std::size_t index = 0;
  std::size_t current = 0;
  for (std::size_t input_index = 1; input_index < words.size(); ++input_index) {
    const auto value = words[input_index];
    const auto key = static_cast<std::size_t>(value & kKeyMask);
    const auto candidate_index = indices[key];
    std::size_t reference = index % kHistorySize;

    if (index - candidate_index < kHistorySize) {
      const auto candidate_slot = candidate_index % kHistorySize;
      const auto candidate_xor = value ^ stored[candidate_slot];
      if (std::countr_zero(candidate_xor) > kTrailingThreshold)
        reference = candidate_slot;
    }

    encoded.reference_slots.push_back(static_cast<std::uint8_t>(reference));
    encoded.xor_residuals.push_back(stored[reference] ^ value);

    current = (current + 1) % kHistorySize;
    stored[current] = value;
    ++index;
    indices[key] = index;
  }
  return encoded;
}

std::vector<std::uint64_t> decode_words(const Encoded& encoded) {
  validate_encoded_shape(encoded);
  if (encoded.value_count == 0) return {};

  std::vector<std::uint64_t> words;
  words.reserve(static_cast<std::size_t>(encoded.value_count));
  words.push_back(encoded.first_value);

  std::array<std::uint64_t, kHistorySize> stored{};
  stored[0] = encoded.first_value;
  std::size_t current = 0;
  for (std::size_t index = 0; index < encoded.reference_slots.size(); ++index) {
    const auto value =
        stored[encoded.reference_slots[index]] ^ encoded.xor_residuals[index];
    current = (current + 1) % kHistorySize;
    stored[current] = value;
    words.push_back(value);
  }
  require(words.size() == encoded.value_count,
          "Chimp history128 decoded value count mismatch");
  return words;
}

std::vector<std::uint8_t> serialize_metadata(const Encoded& encoded) {
  validate_encoded_shape(encoded);
  std::vector<std::uint8_t> metadata(kMagic.begin(), kMagic.end());
  metadata.push_back(static_cast<std::uint8_t>(encoded.profile));
  metadata.insert(metadata.end(), 3, 0);  // Reserved; required to remain zero.
  append_u64_le(metadata, encoded.value_count);
  append_u64_le(metadata, encoded.first_value);
  require(metadata.size() == kMetadataBytes,
          "internal Chimp history128 metadata size mismatch");
  return metadata;
}

Encoded parse_stage_payload(std::span<const std::uint8_t> metadata,
                            std::span<const std::uint8_t> reference_slots,
                            std::span<const std::uint8_t> xor_residuals_le,
                            Profile expected_profile) {
  validate_profile(expected_profile);
  require(metadata.size() == kMetadataBytes,
          "invalid Chimp history128 metadata size");
  require(std::equal(kMagic.begin(), kMagic.end(), metadata.begin()),
          "invalid Chimp history128 metadata magic/version");
  const auto profile = static_cast<Profile>(metadata[4]);
  validate_profile(profile);
  require(profile == expected_profile, "Chimp history128 profile mismatch");
  require(metadata[5] == 0 && metadata[6] == 0 && metadata[7] == 0,
          "non-zero Chimp history128 reserved metadata");

  std::size_t position = 8;
  const auto value_count = read_u64_le(metadata, position);
  const auto first_value = read_u64_le(metadata, position);
  require(position == metadata.size(), "trailing Chimp history128 metadata");

  const std::uint64_t residual_count = value_count == 0 ? 0 : value_count - 1;
  require(residual_count <= std::numeric_limits<std::size_t>::max(),
          "Chimp history128 value count exceeds address space");
  const auto expected = static_cast<std::size_t>(residual_count);
  require(reference_slots.size() == expected,
          "Chimp history128 reference port length mismatch");
  require(expected <= std::numeric_limits<std::size_t>::max() / 8 &&
              xor_residuals_le.size() == expected * 8,
          "Chimp history128 residual port length mismatch");

  Encoded encoded;
  encoded.profile = profile;
  encoded.value_count = value_count;
  encoded.first_value = first_value;
  encoded.reference_slots.assign(reference_slots.begin(), reference_slots.end());
  encoded.xor_residuals.reserve(expected);
  position = 0;
  for (std::size_t i = 0; i < expected; ++i)
    encoded.xor_residuals.push_back(read_u64_le(xor_residuals_le, position));
  require(position == xor_residuals_le.size(),
          "trailing Chimp history128 residual bytes");
  validate_encoded_shape(encoded);
  return encoded;
}

StagePayload encode_bytes(std::span<const std::uint8_t> input, Profile profile) {
  validate_profile(profile);
  require(input.size() % 8 == 0,
          "Chimp history128 input byte count is not a multiple of 8");
  std::vector<std::uint64_t> words(input.size() / 8);
  if (!input.empty()) std::memcpy(words.data(), input.data(), input.size());
  const auto encoded = encode_words(words, profile);
  return {serialize_metadata(encoded), encoded.reference_slots,
          residual_bytes(encoded.xor_residuals)};
}

std::vector<std::uint8_t> decode_bytes(const StagePayload& payload,
                                       Profile expected_profile) {
  const auto encoded = parse_stage_payload(payload.metadata,
                                           payload.reference_slots,
                                           payload.xor_residuals_le,
                                           expected_profile);
  const auto words = decode_words(encoded);
  require(words.size() <= std::numeric_limits<std::size_t>::max() / 8,
          "Chimp history128 decoded stream exceeds address space");
  std::vector<std::uint8_t> output(words.size() * 8);
  if (!output.empty()) std::memcpy(output.data(), words.data(), output.size());
  return output;
}

}  // namespace native_graph_v1::chimp_history_xor_v2
