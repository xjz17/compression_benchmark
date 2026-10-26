#include "special_stages.hpp"

#include "general_codec_reference_stages.hpp"
#include "lz4_exact_stages.hpp"
#include "native_lz_action_stages.hpp"
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
#include "deflate_container_stages.hpp"
#include "general_codec_residual_stages.hpp"
#endif

#include <nlohmann/json.hpp>

#include <array>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
// The frozen REGER source predates a public library header.  The optional
// operator102_support_deps target compiles this translation unit with its CLI
// main renamed and exports these real forward/inverse entry points.
namespace reger_legacy_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes core_forward(const Bytes&);
Bytes core_inverse(const Bytes&);
Bytes frame_forward(const Bytes&);
Bytes frame_inverse(const Bytes&);
}  // namespace reger_legacy_cpp

// The support-deps target exposes the pinned C-Blosc2 3.3.3 implementation
// through its private DSO host.  These functions keep all three A102 stages
// reversible across processes: logical B2P1 core, physical chunk bridge, and
// the focal physical-to-logical container normalization.
namespace blosc2_binding_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes core_forward(const Bytes&);
Bytes core_inverse(const Bytes&);
Bytes frame_forward(const Bytes&);
Bytes frame_inverse(const Bytes&);
}  // namespace blosc2_binding_cpp
#endif

namespace operator_harness_v2 {
namespace {

using Json = nlohmann::json;
using native_graph_v1::Bytes;
using native_graph_v1::Kind;
using native_graph_v1::StageRequest;
using native_graph_v1::StageResult;
using native_graph_v1::Stream;

constexpr std::string_view kWindowAction =
    "op.match_window_model.native_lz.window.cpp_v2";
constexpr std::string_view kMatchLimitsAction =
    "op.lz_match.native_lz.aligned8.cpp_v2";
constexpr std::string_view kMagicLengthAction =
    "op.container_frame.magic_length.cpp_v3";
constexpr std::string_view kLz4ContainerAction =
    "op.container_frame.lz4.cpp_v1";
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
constexpr std::string_view kRegerCoreAction =
    "op.block_state_control.reger_legacy_block_core.cpp_v1";
constexpr std::string_view kRegerFrameAction =
    "op.container_frame.reger3_legacy.cpp_v1";
constexpr std::string_view kGzipResidualAction =
    "op.block_state_control.gzip_deflate_residual_core.cpp_v1";
constexpr std::string_view kGzipStoredPartsAction =
    "op.container_frame.stored_parts.cpp_v2";
constexpr std::string_view kBlosc2CoreAction =
    "op.block_state_control.blosc2_3_3_3_binding_macro.cpp_v1";
constexpr std::string_view kBlosc2PhysicalChunkAction =
    "build.blosc2_physical_chunk_from_b2p1";
constexpr std::string_view kBlosc2ContainerAction =
    "op.container_frame.blosc2_legacy_chunk.cpp_v1";
#endif

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("HARNESS_V2_SPECIAL_STAGE: " + message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::string kind_name(Kind kind) {
  if (kind == Kind::Serial) return "serial";
  if (kind == Kind::Struct) return "struct";
  if (kind == Kind::Numeric) return "numeric";
  fail("unknown stream kind");
}

Kind parse_kind(const Json& value) {
  require(value.is_string(), "metadata stream kind is not a string");
  const auto kind = value.get<std::string>();
  if (kind == "serial") return Kind::Serial;
  if (kind == "struct") return Kind::Struct;
  if (kind == "numeric") return Kind::Numeric;
  fail("metadata stream kind is invalid");
}

void validate_shape(const Stream& stream, std::string_view role) {
  require(stream.width > 0, std::string(role) + " width is zero");
  require(stream.data.size() % stream.width == 0,
          std::string(role) + " payload is not width aligned");
  if (stream.kind == Kind::Serial)
    require(stream.width == 1, std::string(role) + " serial width is not one");
  else if (stream.kind == Kind::Numeric)
    require(stream.width == 1 || stream.width == 2 || stream.width == 4 ||
                stream.width == 8,
            std::string(role) + " numeric width is unsupported");
}

Json stream_spec(const Stream& stream) {
  return {{"kind", kind_name(stream.kind)},
          {"width", stream.width},
          {"bytes", stream.data.size()}};
}

void require_spec(const Stream& stream, const Json& spec,
                  std::string_view role) {
  require(spec.is_object(), std::string(role) + " metadata spec is not an object");
  require(spec.contains("kind") && spec.contains("width") &&
              spec.contains("bytes"),
          std::string(role) + " metadata spec is incomplete");
  require(parse_kind(spec.at("kind")) == stream.kind,
          std::string(role) + " kind differs from metadata");
  require(spec.at("width").is_number_unsigned() &&
              spec.at("width").get<std::uint64_t>() == stream.width,
          std::string(role) + " width differs from metadata");
  require(spec.at("bytes").is_number_unsigned() &&
              spec.at("bytes").get<std::uint64_t>() == stream.data.size(),
          std::string(role) + " byte length differs from metadata");
}

void require_one_input(const std::vector<Stream>& inputs, Kind kind,
                       std::uint32_t width) {
  require(inputs.size() == 1, "stage requires exactly one input stream");
  validate_shape(inputs.front(), "input");
  require(inputs.front().kind == kind && inputs.front().width == width,
          "input stream shape does not match the source action");
}

void require_request(const StageRequest& request, std::string_view profile,
                     const std::map<std::string, std::string>& parameters) {
  require(request.profile == profile, "stage profile differs from frozen catalog");
  require(request.parameters == parameters,
          "stage parameters differ from frozen catalog");
}

StageResult result_with_shape_metadata(const StageRequest& request,
                                       const Stream& input, Bytes output) {
  Stream output_stream{Kind::Serial, 1, std::move(output)};
  validate_shape(output_stream, "output");
  const Json metadata{{"schema", "operator-harness-v2-special-stage-metadata-v1"},
                      {"action", request.action},
                      {"profile", request.profile},
                      {"parameters", request.parameters},
                      {"input", stream_spec(input)},
                      {"output", stream_spec(output_stream)}};
  const auto serialized = metadata.dump();
  return {{std::move(output_stream)},
          Bytes(serialized.begin(), serialized.end()),
          {"bytes"}};
}

Json parse_metadata(const StageRequest& request,
                    const std::vector<Stream>& outputs,
                    std::span<const std::uint8_t> metadata) {
  require(metadata.size() <= 64 * 1024,
          "special-stage metadata exceeds 64 KiB");
  require(outputs.size() == 1, "stage requires exactly one output stream");
  validate_shape(outputs.front(), "output");
  Json decoded;
  try {
    decoded = Json::parse(metadata.begin(), metadata.end());
  } catch (const Json::exception&) {
    fail("metadata is not valid JSON");
  }
  require(decoded.is_object() &&
              decoded.value("schema", "") ==
                  "operator-harness-v2-special-stage-metadata-v1",
          "metadata schema is invalid");
  require(decoded.value("action", "") == request.action &&
              decoded.value("profile", "") == request.profile,
          "request identity differs from metadata");
  require(decoded.contains("parameters") &&
              decoded.at("parameters") == Json(request.parameters),
          "request parameters differ from metadata");
  require(decoded.contains("input") && decoded.contains("output"),
          "metadata shape is incomplete");
  require_spec(outputs.front(), decoded.at("output"), "output");
  return decoded;
}

Stream restored_stream(const Json& input_spec, Bytes restored) {
  require(input_spec.is_object() && input_spec.contains("kind") &&
              input_spec.contains("width") && input_spec.contains("bytes"),
          "input metadata spec is incomplete");
  require(input_spec.at("width").is_number_unsigned() &&
              input_spec.at("bytes").is_number_unsigned(),
          "input metadata shape is invalid");
  const auto width64 = input_spec.at("width").get<std::uint64_t>();
  require(width64 > 0 && width64 <= UINT32_MAX,
          "input metadata width is invalid");
  const auto expected_bytes = input_spec.at("bytes").get<std::uint64_t>();
  require(restored.size() == expected_bytes,
          "inverse byte length differs from input metadata");
  Stream result{parse_kind(input_spec.at("kind")),
                static_cast<std::uint32_t>(width64), std::move(restored)};
  validate_shape(result, "restored input");
  return result;
}

#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
bool is_support_deps_action(std::string_view action) {
  return action == kRegerCoreAction || action == kRegerFrameAction ||
         action == kGzipResidualAction ||
         action == kGzipStoredPartsAction || action == kBlosc2CoreAction ||
         action == kBlosc2PhysicalChunkAction ||
         action == kBlosc2ContainerAction;
}

bool explicit_reger_parameters(
    const std::map<std::string, std::string>& parameters) {
  return parameters ==
         std::map<std::string, std::string>{{"block_size", "1000"},
                                            {"search", "balanced"},
                                            {"segment_size", "20"}};
}

void require_reger_core_request(const StageRequest& request) {
  require(request.profile == "int64",
          "stage profile differs from frozen catalog");
  // A094 records its explicit benchmark settings.  A100 uses the same source
  // core as a prefix with that source's frozen defaults.
  require(request.parameters.empty() ||
              explicit_reger_parameters(request.parameters),
          "stage parameters differ from frozen catalog");
}

class ScopedEnvironmentValue {
 public:
  ScopedEnvironmentValue(std::string name, std::string value)
      : name_(std::move(name)) {
    if (const char* current = std::getenv(name_.c_str())) previous_ = current;
    require(::setenv(name_.c_str(), value.c_str(), 1) == 0,
            "cannot pin REGER environment");
  }

  ScopedEnvironmentValue(const ScopedEnvironmentValue&) = delete;
  ScopedEnvironmentValue& operator=(const ScopedEnvironmentValue&) = delete;

  ~ScopedEnvironmentValue() {
    if (previous_)
      (void)::setenv(name_.c_str(), previous_->c_str(), 1);
    else
      (void)::unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

Bytes reger_core_forward(const StageRequest& request, const Bytes& input) {
  require_reger_core_request(request);
  const bool explicit_parameters =
      explicit_reger_parameters(request.parameters);
  ScopedEnvironmentValue profile("COMPRESSION_BENCHMARK_REGER_PROFILE", "balanced");
  ScopedEnvironmentValue block_size("COMPRESSION_BENCHMARK_REGER_BLOCK_SIZE",
                                    explicit_parameters ? "1000" : "513");
  ScopedEnvironmentValue segment_size("COMPRESSION_BENCHMARK_REGER_SEGMENT_SIZE",
                                      explicit_parameters ? "20" : "16");
  ScopedEnvironmentValue reorder_iterations(
      "COMPRESSION_BENCHMARK_REGER_REORDER_ITERS", "4");
  ScopedEnvironmentValue exhaustive("COMPRESSION_BENCHMARK_REGER_EXHAUSTIVE", "0");
  ScopedEnvironmentValue high_column_candidates(
      "COMPRESSION_BENCHMARK_REGER_HIGH_COLUMN_CANDIDATES", "16");
  return reger_legacy_cpp::core_forward(input);
}

Bytes reger_core_inverse(const StageRequest& request, const Bytes& packet) {
  require_reger_core_request(request);
  const bool explicit_parameters =
      explicit_reger_parameters(request.parameters);
  // core_inverse currently restores block/segment values from the packet.
  // Pin and restore the same environment anyway so forward and cold inverse
  // retain one closed execution contract if the frozen helper evolves.
  ScopedEnvironmentValue profile("COMPRESSION_BENCHMARK_REGER_PROFILE", "balanced");
  ScopedEnvironmentValue block_size("COMPRESSION_BENCHMARK_REGER_BLOCK_SIZE",
                                    explicit_parameters ? "1000" : "513");
  ScopedEnvironmentValue segment_size("COMPRESSION_BENCHMARK_REGER_SEGMENT_SIZE",
                                      explicit_parameters ? "20" : "16");
  ScopedEnvironmentValue reorder_iterations(
      "COMPRESSION_BENCHMARK_REGER_REORDER_ITERS", "4");
  ScopedEnvironmentValue exhaustive("COMPRESSION_BENCHMARK_REGER_EXHAUSTIVE", "0");
  ScopedEnvironmentValue high_column_candidates(
      "COMPRESSION_BENCHMARK_REGER_HIGH_COLUMN_CANDIDATES", "16");
  return reger_legacy_cpp::core_inverse(packet);
}
#endif

}  // namespace

bool handles_special(const StageRequest& request) {
  const std::string_view action = request.action;
  if (action == kWindowAction || action == kMatchLimitsAction ||
      action == kMagicLengthAction || action == kLz4ContainerAction)
    return true;
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  return is_support_deps_action(action);
#else
  return false;
#endif
}

StageResult forward_special(const StageRequest& request,
                            const std::vector<Stream>& inputs) {
  require(handles_special(request), "unsupported special-stage action");

  if (request.action == kWindowAction) {
    require_request(request, "bytes", {{"window_bytes", "32768"}});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        native_lz_action::attach_window(inputs.front().data, 32768));
  }
  if (request.action == kMatchLimitsAction) {
    require_request(request, "bytes",
                    {{"maximum", "175"}, {"minimum", "5"}});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        native_lz_action::attach_match_limits(inputs.front().data, 5, 175));
  }
  if (request.action == kMagicLengthAction) {
    require_request(request, "bytes", {{"profile", "B1F1"}});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        reference_actions::general::frame_forward(inputs.front().data, false));
  }
  if (request.action == kLz4ContainerAction) {
    require_request(request, "bytes", {});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        exact_actions::lz4::container_forward(inputs.front().data));
  }

#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  if (request.action == kRegerCoreAction) {
    require_reger_core_request(request);
    require_one_input(inputs, Kind::Numeric, 8);
    return result_with_shape_metadata(
        request, inputs.front(),
        reger_core_forward(request, inputs.front().data));
  }
  if (request.action == kRegerFrameAction) {
    require_request(request, "int64", {});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        reger_legacy_cpp::frame_forward(inputs.front().data));
  }
  if (request.action == kGzipResidualAction) {
    require_request(request, "bytes", {});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        exact_actions::general_residual::gzip_forward(inputs.front().data, 6));
  }
  if (request.action == kGzipStoredPartsAction) {
    require_request(request, "bytes", {{"profile", "gzip"}});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        exact_actions::deflate_container::gzip_forward(inputs.front().data));
  }
  if (request.action == kBlosc2CoreAction) {
    require_request(request, "bytes", {});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        blosc2_binding_cpp::core_forward(inputs.front().data));
  }
  if (request.action == kBlosc2PhysicalChunkAction) {
    require_request(request, "bytes", {});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        blosc2_binding_cpp::frame_forward(inputs.front().data));
  }
  if (request.action == kBlosc2ContainerAction) {
    require_request(request, "bytes", {});
    require_one_input(inputs, Kind::Serial, 1);
    return result_with_shape_metadata(
        request, inputs.front(),
        blosc2_binding_cpp::frame_inverse(inputs.front().data));
  }
#endif
  fail("action passed handles_special but has no forward implementation");
}

std::vector<Stream> inverse_special(
    const StageRequest& request, const std::vector<Stream>& outputs,
    std::span<const std::uint8_t> metadata) {
  require(handles_special(request), "unsupported special-stage action");
  const auto decoded = parse_metadata(request, outputs, metadata);
  Bytes restored;

  if (request.action == kWindowAction) {
    require_request(request, "bytes", {{"window_bytes", "32768"}});
    restored = native_lz_action::inverse(
        "op.match_window_model.native_lz.window32768", outputs.front().data);
  } else if (request.action == kMatchLimitsAction) {
    require_request(request, "bytes",
                    {{"maximum", "175"}, {"minimum", "5"}});
    restored = native_lz_action::inverse(
        "op.lz_match.native_lz.aligned8.min5.max175", outputs.front().data);
  } else if (request.action == kMagicLengthAction) {
    require_request(request, "bytes", {{"profile", "B1F1"}});
    restored = reference_actions::general::frame_inverse(
        outputs.front().data, false);
  } else if (request.action == kLz4ContainerAction) {
    require_request(request, "bytes", {});
    restored = exact_actions::lz4::container_inverse(outputs.front().data);
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  } else if (request.action == kRegerCoreAction) {
    require_reger_core_request(request);
    restored = reger_core_inverse(request, outputs.front().data);
  } else if (request.action == kRegerFrameAction) {
    require_request(request, "int64", {});
    restored = reger_legacy_cpp::frame_inverse(outputs.front().data);
  } else if (request.action == kGzipResidualAction) {
    require_request(request, "bytes", {});
    restored = exact_actions::general_residual::gzip_inverse(
        outputs.front().data);
  } else if (request.action == kGzipStoredPartsAction) {
    require_request(request, "bytes", {{"profile", "gzip"}});
    restored = exact_actions::deflate_container::gzip_inverse(
        outputs.front().data);
  } else if (request.action == kBlosc2CoreAction) {
    require_request(request, "bytes", {});
    restored = blosc2_binding_cpp::core_inverse(outputs.front().data);
  } else if (request.action == kBlosc2PhysicalChunkAction) {
    require_request(request, "bytes", {});
    restored = blosc2_binding_cpp::frame_inverse(outputs.front().data);
  } else if (request.action == kBlosc2ContainerAction) {
    require_request(request, "bytes", {});
    restored = blosc2_binding_cpp::frame_forward(outputs.front().data);
#endif
  } else {
    fail("action passed handles_special but has no inverse implementation");
  }

  return {restored_stream(decoded.at("input"), std::move(restored))};
}

}  // namespace operator_harness_v2
