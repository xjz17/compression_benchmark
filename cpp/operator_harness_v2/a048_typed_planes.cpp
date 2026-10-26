#include "a048_typed_planes.hpp"

#include "learning_codec_exact_stages.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace operator_harness_v2 {
namespace {

using native_graph_v1::Bytes;
using native_graph_v1::Kind;
using native_graph_v1::StageRequest;
using native_graph_v1::StageResult;
using native_graph_v1::Stream;
using Json = nlohmann::json;

static_assert(std::endian::native == std::endian::little,
              "A048 typed uint64 byte contract requires a little-endian host");
constexpr std::uint64_t kMaxReconstructedBytes = 256ULL << 20;
constexpr std::size_t kMaxMetadataBytes = 4096;

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) invalid(message);
}

unsigned requested_bit_width(const StageRequest& request) {
  require(request.action == kA048TypedPlaneAction,
          "A048_TYPED_ACTION: wrong stage action");
  require(request.profile == "uint64",
          "A048_TYPED_PROFILE: profile must be uint64");
  require(request.parameters.size() == 1 &&
              request.parameters.contains("bit_width"),
          "A048_TYPED_PARAMETERS: exactly bit_width is required");
  const auto& text = request.parameters.at("bit_width");
  require(!text.empty() && text.size() <= 2 &&
              std::all_of(text.begin(), text.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
              }) &&
              (text == "0" || text.front() != '0'),
          "A048_TYPED_BIT_WIDTH: bit_width must be a canonical integer in [0,64]");
  unsigned result = 0;
  for (const auto c : text)
    result = result * 10 + static_cast<unsigned>(c - '0');
  require(result <= 64,
          "A048_TYPED_BIT_WIDTH: bit_width must be a canonical integer in [0,64]");
  return result;
}

std::vector<std::uint64_t> words(const Stream& input) {
  require(input.kind == Kind::Numeric && input.width == 8 &&
              input.data.size() % sizeof(std::uint64_t) == 0,
          "A048_TYPED_INPUT: expected an aligned numeric uint64 stream");
  require(input.data.size() <= kMaxReconstructedBytes,
          "A048_TYPED_SIZE: input exceeds the 256 MiB typed-stage bound");
  std::vector<std::uint64_t> result(input.data.size() / sizeof(std::uint64_t));
  if (!result.empty())
    std::memcpy(result.data(), input.data.data(), input.data.size());
  return result;
}

Bytes raw_words(const std::vector<std::uint64_t>& input) {
  require(input.size() <=
              std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t),
          "A048_TYPED_SIZE: reconstructed stream is too large");
  Bytes result(input.size() * sizeof(std::uint64_t));
  if (!result.empty())
    std::memcpy(result.data(), input.data(), result.size());
  return result;
}

void require_exact_metadata_keys(const Json& metadata) {
  static constexpr std::array<std::string_view, 5> keys{
      "bit_width", "count", "schema", "semantic_plane_count",
      "zero_plane_carrier"};
  require(metadata.is_object() && metadata.size() == keys.size(),
          "A048_TYPED_METADATA: metadata must contain exactly five fields");
  for (const auto key : keys)
    require(metadata.contains(std::string(key)),
            "A048_TYPED_METADATA: required metadata field is missing");
}

}  // namespace

bool handles_a048_typed_planes(const StageRequest& request) {
  return request.action == kA048TypedPlaneAction;
}

StageResult forward_a048_typed_planes(
    const StageRequest& request, const std::vector<Stream>& inputs) {
  require(inputs.size() == 1,
          "A048_TYPED_INPUT: exactly one input is required");
  const auto bit_width = requested_bit_width(request);
  const auto values = words(inputs.front());
  if (bit_width < 64) {
    const auto exceeds = [bit_width](std::uint64_t value) {
      return bit_width == 0 ? value != 0 : (value >> bit_width) != 0;
    };
    if (std::any_of(values.begin(), values.end(), exceeds))
      throw A048InputDomainError(
          "A048 value exceeds the requested pure typed-plane bit_width");
  }
  const auto packed =
      exact_actions::learning::buff_byte_planes_split(values, bit_width);
  const std::size_t plane_count = (bit_width + 7) / 8;

  StageResult result;
  if (plane_count == 0) {
    // The graph runtime requires at least one physical output.  This empty
    // port is a carrier only; metadata records that there are zero semantic
    // planes, and inverse accepts no other zero-width representation.
    result.outputs.push_back({Kind::Serial, 1, {}});
    result.output_logical_views.push_back("a048_zero_plane_physical_carrier");
  } else {
    require(values.empty() ||
                plane_count <=
                    std::numeric_limits<std::size_t>::max() / values.size(),
            "A048_TYPED_SIZE: plane payload size overflow");
    require(packed.size() == plane_count * values.size(),
            "A048_TYPED_INTERNAL: split result size mismatch");
    for (std::size_t plane = 0; plane < plane_count; ++plane) {
      const auto begin = packed.begin() +
                         static_cast<std::ptrdiff_t>(plane * values.size());
      const auto end = begin + static_cast<std::ptrdiff_t>(values.size());
      result.outputs.push_back({Kind::Serial, 1, Bytes(begin, end)});
      result.output_logical_views.push_back(
          "a048_aligned_msb_plane_" + std::to_string(plane));
    }
  }

  const Json metadata{{"schema", "operator-harness-a048-typed-plane-v2"},
                      {"count", values.size()},
                      {"bit_width", bit_width},
                      {"semantic_plane_count", plane_count},
                      {"zero_plane_carrier", plane_count == 0}};
  const auto text = metadata.dump();
  result.metadata.assign(text.begin(), text.end());
  return result;
}

std::vector<Stream> inverse_a048_typed_planes(
    const StageRequest& request, const std::vector<Stream>& outputs,
    std::span<const std::uint8_t> metadata_bytes) {
  const auto requested_width = requested_bit_width(request);
  require(metadata_bytes.size() <= kMaxMetadataBytes,
          "A048_TYPED_METADATA: metadata exceeds the size bound");
  const std::string text(reinterpret_cast<const char*>(metadata_bytes.data()),
                         metadata_bytes.size());
  const auto metadata = Json::parse(text);
  require_exact_metadata_keys(metadata);
  require(metadata.at("schema").is_string() &&
              metadata.at("schema") ==
                  "operator-harness-a048-typed-plane-v2",
          "A048_TYPED_METADATA: invalid metadata schema");
  require(metadata.at("count").is_number_unsigned(),
          "A048_TYPED_METADATA: count must be unsigned");
  require(metadata.at("bit_width").is_number_unsigned(),
          "A048_TYPED_METADATA: bit_width must be unsigned");
  require(metadata.at("semantic_plane_count").is_number_unsigned(),
          "A048_TYPED_METADATA: semantic_plane_count must be unsigned");
  require(metadata.at("zero_plane_carrier").is_boolean(),
          "A048_TYPED_METADATA: zero_plane_carrier must be boolean");

  const auto count = metadata.at("count").get<std::uint64_t>();
  const auto stored_width = metadata.at("bit_width").get<std::uint64_t>();
  const auto plane_count =
      metadata.at("semantic_plane_count").get<std::uint64_t>();
  require(stored_width == requested_width && stored_width <= 64 &&
              plane_count == (stored_width + 7) / 8,
          "A048_TYPED_METADATA: metadata differs from the requested stage");
  require(count <= kMaxReconstructedBytes / sizeof(std::uint64_t) &&
              count <= std::numeric_limits<std::size_t>::max(),
          "A048_TYPED_METADATA: reconstructed size exceeds the bound");

  Bytes packed;
  if (plane_count == 0) {
    require(metadata.at("zero_plane_carrier").get<bool>() &&
                outputs.size() == 1 && outputs.front().kind == Kind::Serial &&
                outputs.front().width == 1 && outputs.front().data.empty(),
            "A048_TYPED_OUTPUT: invalid zero-plane physical carrier");
  } else {
    require(!metadata.at("zero_plane_carrier").get<bool>(),
            "A048_TYPED_METADATA: nonzero planes cannot use the carrier");
    require(outputs.size() == plane_count,
            "A048_TYPED_OUTPUT: physical plane count mismatch");
    require(count == 0 ||
                plane_count <=
                    std::numeric_limits<std::size_t>::max() / count,
            "A048_TYPED_SIZE: joined plane payload size overflow");
    packed.reserve(static_cast<std::size_t>(plane_count * count));
    for (const auto& output : outputs) {
      require(output.kind == Kind::Serial && output.width == 1 &&
                  output.data.size() == count,
              "A048_TYPED_OUTPUT: each plane must be a serial byte stream of count bytes");
      packed.insert(packed.end(), output.data.begin(), output.data.end());
    }
  }

  auto joined = exact_actions::learning::buff_byte_planes_join(
      packed, static_cast<std::size_t>(count),
      static_cast<unsigned>(stored_width));
  return {{Kind::Numeric, 8, raw_words(joined)}};
}

}  // namespace operator_harness_v2
