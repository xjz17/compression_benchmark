#include "harness_dispatch.hpp"

#include "learning_codec_exact_stages.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace operator_harness_v1 {
namespace {
using native_graph_v1::Bytes;
using native_graph_v1::Kind;
using native_graph_v1::StageRequest;
using native_graph_v1::StageResult;
using native_graph_v1::Stream;
using Json = nlohmann::json;

static_assert(std::endian::native == std::endian::little,
              "operator_harness_v1 typed byte contracts require a little-endian host");
constexpr std::uint64_t kMaxA048ReconstructedBytes = 256ULL << 20;

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) invalid(message);
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool is_nncp(const std::string& action) {
  return lower(action).find("nncp") != std::string::npos;
}

bool is_context_heavy(const std::string& action) {
  const auto name = lower(action);
  return name.find(".paq.") != std::string::npos ||
         name.find(".cmix.") != std::string::npos;
}

unsigned bit_width(const StageRequest& request) {
  require(request.profile == "uint64",
          "HARNESS_A048_PROFILE: harness.a048.byte_planes.v1 requires profile=uint64");
  require(request.parameters.size() == 1 && request.parameters.contains("bit_width"),
          "HARNESS_A048_PARAMETERS: exactly bit_width is required");
  const auto& text = request.parameters.at("bit_width");
  require(!text.empty() && text.size() <= 2 &&
              std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); }),
          "HARNESS_A048_BIT_WIDTH: bit_width must be an integer in [0,64]");
  unsigned value = 0;
  for (const auto c : text) value = value * 10 + static_cast<unsigned>(c - '0');
  require(value <= 64, "HARNESS_A048_BIT_WIDTH: bit_width must be an integer in [0,64]");
  return value;
}

std::vector<std::uint64_t> words(const Stream& input) {
  require(input.kind == Kind::Numeric && input.width == 8 && input.data.size() % 8 == 0,
          "HARNESS_A048_INPUT: expected an aligned uint64 numeric stream");
  require(input.data.size() <= kMaxA048ReconstructedBytes,
          "HARNESS_A048_SIZE: input exceeds the 256 MiB typed-stage bound");
  std::vector<std::uint64_t> result(input.data.size() / 8);
  if (!result.empty()) std::memcpy(result.data(), input.data.data(), input.data.size());
  return result;
}

Bytes raw_words(const std::vector<std::uint64_t>& input) {
  require(input.size() <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t),
          "HARNESS_A048_SIZE: reconstructed stream is too large");
  Bytes result(input.size() * sizeof(std::uint64_t));
  if (!result.empty()) std::memcpy(result.data(), input.data(), result.size());
  return result;
}

StageResult byte_plane_forward(const StageRequest& request,
                               const std::vector<Stream>& inputs) {
  require(inputs.size() == 1, "HARNESS_A048_INPUT: exactly one input is required");
  const auto width = bit_width(request);
  const auto values = words(inputs.front());
  const auto packed = exact_actions::learning::buff_byte_planes_split(values, width);
  const std::size_t plane_count = (width + 7) / 8;

  StageResult result;
  if (plane_count == 0) {
    // GraphSession requires at least one physical output. This empty carrier
    // represents zero semantic planes and is checked on inverse.
    result.outputs.push_back({Kind::Serial, 1, {}});
    result.output_logical_views.push_back("a048_zero_plane_carrier");
  } else {
    require(values.empty() || plane_count <= std::numeric_limits<std::size_t>::max() / values.size(),
            "HARNESS_A048_SIZE: plane size overflow");
    require(packed.size() == plane_count * values.size(),
            "HARNESS_A048_INTERNAL: split result size mismatch");
    for (std::size_t plane = 0; plane < plane_count; ++plane) {
      const auto begin = packed.begin() + static_cast<std::ptrdiff_t>(plane * values.size());
      const auto end = begin + static_cast<std::ptrdiff_t>(values.size());
      result.outputs.push_back({Kind::Serial, 1, Bytes(begin, end)});
      result.output_logical_views.push_back("a048_byte_plane_" + std::to_string(plane));
    }
  }
  const Json metadata{{"schema", "harness-a048-metadata-v1"},
                      {"count", values.size()},
                      {"bit_width", width},
                      {"semantic_plane_count", plane_count},
                      {"zero_plane_carrier", plane_count == 0}};
  const auto dumped = metadata.dump();
  result.metadata.assign(dumped.begin(), dumped.end());
  return result;
}

std::vector<Stream> byte_plane_inverse(const StageRequest& request,
                                       const std::vector<Stream>& outputs,
                                       std::span<const std::uint8_t> metadata_bytes) {
  const auto requested_width = bit_width(request);
  const std::string metadata_text(reinterpret_cast<const char*>(metadata_bytes.data()),
                                  metadata_bytes.size());
  const auto metadata = Json::parse(metadata_text);
  require(metadata.is_object() && metadata.value("schema", "") == "harness-a048-metadata-v1",
          "HARNESS_A048_METADATA: invalid metadata schema");
  require(metadata.contains("count") && metadata.at("count").is_number_unsigned(),
          "HARNESS_A048_METADATA: count must be an unsigned integer");
  require(metadata.contains("bit_width") && metadata.at("bit_width").is_number_unsigned(),
          "HARNESS_A048_METADATA: bit_width must be an unsigned integer");
  require(metadata.contains("semantic_plane_count") &&
              metadata.at("semantic_plane_count").is_number_unsigned(),
          "HARNESS_A048_METADATA: semantic_plane_count must be an unsigned integer");
  require(metadata.contains("zero_plane_carrier") &&
              metadata.at("zero_plane_carrier").is_boolean(),
          "HARNESS_A048_METADATA: zero_plane_carrier must be boolean");
  const auto count = metadata.at("count").get<std::uint64_t>();
  const auto stored_width = metadata.at("bit_width").get<std::uint64_t>();
  const auto plane_count = metadata.at("semantic_plane_count").get<std::uint64_t>();
  require(stored_width == requested_width && stored_width <= 64 &&
              plane_count == (stored_width + 7) / 8,
          "HARNESS_A048_METADATA: metadata/request mismatch");
  require(count <= kMaxA048ReconstructedBytes / sizeof(std::uint64_t) &&
              count <= std::numeric_limits<std::size_t>::max(),
          "HARNESS_A048_METADATA: count exceeds the 256 MiB typed-stage bound");

  Bytes packed;
  if (plane_count == 0) {
    require(metadata.value("zero_plane_carrier", false) && outputs.size() == 1 &&
                outputs.front().kind == Kind::Serial && outputs.front().width == 1 &&
                outputs.front().data.empty(),
            "HARNESS_A048_METADATA: invalid zero-plane physical carrier");
  } else {
    require(!metadata.at("zero_plane_carrier").get<bool>(),
            "HARNESS_A048_METADATA: nonzero planes cannot use the zero-plane carrier");
    require(outputs.size() == plane_count,
            "HARNESS_A048_OUTPUT: physical plane count mismatch");
    require(count == 0 || plane_count <= std::numeric_limits<std::size_t>::max() / count,
            "HARNESS_A048_SIZE: joined plane size overflow");
    packed.reserve(static_cast<std::size_t>(plane_count * count));
    for (const auto& output : outputs) {
      require(output.kind == Kind::Serial && output.width == 1 && output.data.size() == count,
              "HARNESS_A048_OUTPUT: expected equal-size serial plane streams");
      packed.insert(packed.end(), output.data.begin(), output.data.end());
    }
  }
  auto joined = exact_actions::learning::buff_byte_planes_join(
      packed, static_cast<std::size_t>(count), static_cast<unsigned>(stored_width));
  return {{Kind::Numeric, 8, raw_words(joined)}};
}
}  // namespace

HarnessDispatcher::HarnessDispatcher(bool allow_context_heavy)
    : allow_context_heavy_(allow_context_heavy),
      inner_(native_graph_v1::make_default_dispatcher()) {}

std::string HarnessDispatcher::identity() const {
  return "operator_harness_v1.policy_dispatch.v1";
}

void HarnessDispatcher::enforce_policy(const StageRequest& request) const {
  if (is_nncp(request.action))
    invalid("NNCP_DISABLED_BY_HARNESS: NNCP is permanently disabled for forward and inverse");
  if (!allow_context_heavy_ && is_context_heavy(request.action))
    invalid("CONTEXT_HEAVY_STAGE_DISABLED_BY_HARNESS: set allow_context_heavy=true for PAQ/CMIX exploratory runs");
}

StageResult HarnessDispatcher::forward(const StageRequest& request,
                                       const std::vector<Stream>& inputs) const {
  enforce_policy(request);
  if (request.action == kBytePlaneAction) return byte_plane_forward(request, inputs);
  auto result = inner_->forward(request, inputs);
  // The underlying dispatcher historically defaults ordinary outputs to their
  // physical view. The harness planner needs the semantic type of these typed
  // ports so Split/Concat chains retain the correct contract.
  if (request.action == "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1" ||
      request.action == "op.frame_of_reference.minimum.cpp_v1") {
    require(result.outputs.size() == 2, "HARNESS_TYPED_VIEW: FOR must have two outputs");
    result.output_logical_views = {request.profile, "uint64"};
  } else if (request.action == "op.zigzag.signed64.raw.cpp_v2") {
    require(result.outputs.size() == 1, "HARNESS_TYPED_VIEW: ZigZag must have one output");
    result.output_logical_views = {"uint64"};
  } else if (request.action == "op.xor.previous_word.semantic_ports.cpp_v1") {
    require(result.outputs.size() == 2, "HARNESS_TYPED_VIEW: semantic XOR must have two outputs");
    result.output_logical_views = {"uint64", "uint64"};
  }
  return result;
}

std::vector<Stream> HarnessDispatcher::inverse(
    const StageRequest& request, const std::vector<Stream>& outputs,
    std::span<const std::uint8_t> metadata) const {
  enforce_policy(request);
  if (request.action == kBytePlaneAction)
    return byte_plane_inverse(request, outputs, metadata);
  return inner_->inverse(request, outputs, metadata);
}

std::vector<HarnessCapability> harness_capabilities() {
  std::vector<HarnessCapability> result;
  for (const auto& capability : native_graph_v1::native_stage_capabilities()) {
    if (is_nncp(capability.action)) continue;
    const bool heavy = is_context_heavy(capability.action);
    result.push_back({capability, !heavy, heavy});
  }
  result.push_back({{kBytePlaneAction, "cpp/operator_harness_v1",
                     "A048 typed uint64 to independent serial byte-plane ports", true},
                    true, false});
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
    return left.stage.action < right.stage.action;
  });
  return result;
}

}  // namespace operator_harness_v1
