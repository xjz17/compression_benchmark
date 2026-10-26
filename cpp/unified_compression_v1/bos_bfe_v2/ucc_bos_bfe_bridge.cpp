#include "ucc_bos_bfe_bridge.hpp"

#include "bos_float_exception.hpp"
#include "bos2_refined.hpp"
#include "decimal_scaling_v2.hpp"
#include "../../native_graph_v1/graph_internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace unified_compression_v1::bos_bfe_bridge {
namespace {

using native_baseline86_v1::Block;
using native_baseline86_v1::Bytes;
using namespace native_graph_v1;

constexpr std::size_t kMaximumBytes = 256ULL << 20;
constexpr std::array<std::uint8_t, 8> kBundleMagic = {
    'B', 'O', 'S', 'B', 'H', 'C', '1', 0};
constexpr std::uint16_t kBundleVersion = 1;
constexpr std::string_view kBfeAction =
    "bos-float-exception/bfe1-pilot-v3";
constexpr std::string_view kBfeConfiguration =
    "bfe1-decimal-xor-sparse-bos2-pilot256-wholep-v3";
constexpr std::string_view kBfeIntegerConfiguration =
    "project_bos2_partitioned";
constexpr std::string_view kBfePrecisions = "0,1,2,3,4,6";
constexpr std::string_view kBfeDispatcher =
    "bos-float-exception-ngf-dispatch-v3";
constexpr std::string_view kDsc3Action =
    "baseline86/scaled-int64-v2/BOS";
constexpr std::string_view kDsc3Configuration =
    "typed_shortest_decimal_int64_planar_words_v3";
constexpr std::string_view kDsc3Dispatcher =
    "native-baseline86-full-frame-dispatch-v1";
constexpr std::string_view kBos2Policy =
    "baseline86-fixed-profiles-20260919-v3";
constexpr std::string_view kBos2Action = "baseline86/BOS";
constexpr std::string_view kBos2Configuration =
    "delta_zigzag_split_partition_for_bitpack_v1";
constexpr std::string_view kBos2Dispatcher =
    "native-baseline86-full-frame-dispatch-v1";

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::uint64_t natural(const std::string& text) {
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
          "invalid BOS hybrid natural number");
  return value;
}

template <class T>
std::vector<T> words(std::span<const std::uint8_t> bytes) {
  require(bytes.size() % sizeof(T) == 0, "unaligned BOS integer words");
  std::vector<T> result(bytes.size() / sizeof(T));
  if (!bytes.empty())
    std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

template <class T>
Bytes bytes(const std::vector<T>& values) {
  require(values.size() <= kMaximumBytes / sizeof(T),
          "BOS decoded words exceed byte bound");
  Bytes result(values.size() * sizeof(T));
  if (!result.empty())
    std::memcpy(result.data(), values.data(), result.size());
  return result;
}

std::uint64_t valid_values(std::uint64_t rows, const Bytes& bitmap) {
  require(rows <= kMaximumBytes * 8 && bitmap.size() == (rows + 7) / 8,
          "BOS hybrid bitmap count mismatch");
  std::uint64_t missing = 0;
  for (const auto byte : bitmap) missing += std::popcount(byte);
  if (rows % 8 != 0 && !bitmap.empty()) {
    const auto used = static_cast<unsigned>(rows % 8);
    const auto padding = static_cast<unsigned>(~((1U << used) - 1U)) & 0xffU;
    require((bitmap.back() & padding) == 0,
            "BOS hybrid bitmap padding is nonzero");
  }
  require(missing <= rows, "BOS hybrid missing count exceeds row count");
  return rows - missing;
}

const bos_float_exception::EncodeInt kIntegerEncode =
    [](std::span<const std::int64_t> values) {
      return bos_bfe_v2::integer::optimized_compose(values);
    };

const bos_float_exception::DecodeInt kIntegerDecode =
    [](std::span<const std::uint8_t> wire) {
      return bos_bfe_v2::integer::optimized_decompose(wire);
    };

void validate_integer_block(const Block& block) {
  native_baseline86_v1::validate(block);
  require(block.dtype == "int64", "BOS2 route requires int64");
}

void validate_float_block(const Block& block) {
  native_baseline86_v1::validate(block);
  require(block.dtype == "float32" || block.dtype == "float64",
          "BOS hybrid requires float32 or float64");
  const auto count =
      block.values.size() / native_baseline86_v1::width(block.dtype);
  require(count <= kMaximumBlockValues,
          "BOS hybrid block exceeds the frozen 10000-valid-value contract");
}

bool missing_at(const Bytes& bitmap, std::uint64_t row) {
  return (bitmap[static_cast<std::size_t>(row / 8)] &
          static_cast<std::uint8_t>(1U << (row % 8))) != 0;
}

void append_u16(Bytes& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void append_u32(Bytes& out, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(Bytes& out, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t crc32(std::span<const std::uint8_t> input) {
  std::uint32_t crc = 0xffffffffU;
  for (const auto byte : input) {
    crc ^= byte;
    for (unsigned bit = 0; bit != 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
  }
  return ~crc;
}

class Cursor {
 public:
  explicit Cursor(std::span<const std::uint8_t> input) : input_(input) {}

  std::span<const std::uint8_t> take(std::size_t size) {
    require(size <= input_.size() - position_, "truncated BOS hybrid bundle");
    auto result = input_.subspan(position_, size);
    position_ += size;
    return result;
  }

  std::uint16_t u16() {
    const auto value = take(2);
    return static_cast<std::uint16_t>(value[0]) |
           (static_cast<std::uint16_t>(value[1]) << 8);
  }

  std::uint32_t u32() {
    const auto value = take(4);
    std::uint32_t result = 0;
    for (unsigned index = 0; index != 4; ++index)
      result |= static_cast<std::uint32_t>(value[index]) << (8 * index);
    return result;
  }

  std::uint64_t u64() {
    const auto value = take(8);
    std::uint64_t result = 0;
    for (unsigned index = 0; index != 8; ++index)
      result |= static_cast<std::uint64_t>(value[index]) << (8 * index);
    return result;
  }

  std::size_t position() const { return position_; }
  bool empty() const { return position_ == input_.size(); }

 private:
  std::span<const std::uint8_t> input_;
  std::size_t position_ = 0;
};

std::vector<Block> split_blocks(const Block& block) {
  require(block.dtype == "float32" || block.dtype == "float64",
          "BOS hybrid requires float32 or float64");
  const auto width = native_baseline86_v1::width(block.dtype);
  require(block.values.size() % width == 0,
          "BOS hybrid numeric bytes are not width aligned");
  const auto present = valid_values(block.row_count, block.missing);
  require(present <= std::numeric_limits<std::size_t>::max() / width &&
              block.values.size() == static_cast<std::size_t>(present) * width,
          "BOS hybrid valid numeric bytes differ from bitmap");
  std::vector<Block> result;
  std::uint64_t row_start = 0;
  std::size_t value_offset = 0;
  while (row_start < block.row_count) {
    auto row_end = row_start;
    std::uint64_t present = 0;
    while (row_end < block.row_count && present < kMaximumBlockValues) {
      if (!missing_at(block.missing, row_end)) ++present;
      ++row_end;
    }
    const auto local_rows = row_end - row_start;
    Bytes local_missing(static_cast<std::size_t>((local_rows + 7) / 8), 0);
    for (std::uint64_t local = 0; local != local_rows; ++local)
      if (missing_at(block.missing, row_start + local))
        local_missing[static_cast<std::size_t>(local / 8)] |=
            static_cast<std::uint8_t>(1U << (local % 8));
    require(present <= (block.values.size() - value_offset) / width,
            "BOS hybrid split numeric length mismatch");
    const auto byte_count = static_cast<std::size_t>(present) * width;
    Bytes local_values(block.values.begin() + value_offset,
                       block.values.begin() + value_offset + byte_count);
    value_offset += byte_count;
    result.push_back(
        {block.dtype, local_rows, std::move(local_values),
         std::move(local_missing)});
    row_start = row_end;
  }
  if (block.row_count == 0)
    result.push_back({block.dtype, 0, {}, {}});
  require(value_offset == block.values.size(),
          "BOS hybrid split did not consume all numeric bytes");
  return result;
}

std::uint16_t exact_column_decimal_places(const std::vector<Block>& blocks) {
  std::uint16_t result = 0;
  for (const auto& block : blocks)
    result = std::max(
        result,
        native_baseline86_v1::decimal_scaling_v2::required_decimal_places(
            block.values, block.dtype));
  return result;
}

std::map<std::string, std::string> bfe_parameters(
    const Block& block, std::uint16_t whole_p) {
  return {{"configuration", std::string(kBfeConfiguration)},
          {"integer_configuration", std::string(kBfeIntegerConfiguration)},
          {"candidate_precisions", std::string(kBfePrecisions)},
          {"pilot_values", "256"},
          {"whole_column_decimal_places", std::to_string(whole_p)},
          {"row_count", std::to_string(block.row_count)}};
}

std::map<std::string, std::string> dsc3_parameters(
    const Block& block, std::uint16_t whole_p) {
  return {{"configuration",
           std::string(kBos2Configuration)},
          {"scaling_configuration", std::string(kDsc3Configuration)},
          {"row_count", std::to_string(block.row_count)},
          {"decimal_places", std::to_string(whole_p)}};
}

void validate_bfe_request(const StageRequest& request) {
  require(request.action == kBfeAction,
          "unexpected BOS BFE graph action");
  require(request.profile == "float32" || request.profile == "float64",
          "BOS BFE graph requires float32/float64");
  require(request.parameters.size() == 6 &&
              request.parameters.contains("configuration") &&
              request.parameters.at("configuration") == kBfeConfiguration &&
              request.parameters.contains("integer_configuration") &&
              request.parameters.at("integer_configuration") ==
                  kBfeIntegerConfiguration &&
              request.parameters.contains("candidate_precisions") &&
              request.parameters.at("candidate_precisions") ==
                  kBfePrecisions &&
              request.parameters.contains("pilot_values") &&
              natural(request.parameters.at("pilot_values")) == 256 &&
              request.parameters.contains("whole_column_decimal_places") &&
              natural(request.parameters.at("whole_column_decimal_places")) <=
                  324 &&
              request.parameters.contains("row_count"),
          "BOS BFE graph fixed configuration mismatch");
}

void validate_dsc3_request(const StageRequest& request) {
  require(request.action == kDsc3Action,
          "unexpected BOS DSC3 graph action");
  require(request.profile == "float32" || request.profile == "float64",
          "BOS DSC3 graph requires float32/float64");
  require(request.parameters.size() == 4 &&
              request.parameters.contains("configuration") &&
              request.parameters.at("configuration") == kBos2Configuration &&
              request.parameters.contains("scaling_configuration") &&
              request.parameters.at("scaling_configuration") ==
                  kDsc3Configuration &&
              request.parameters.contains("row_count") &&
              request.parameters.contains("decimal_places") &&
              natural(request.parameters.at("decimal_places")) <= 324,
          "BOS DSC3 graph fixed configuration mismatch");
}

void validate_bos2_request(const StageRequest& request) {
  require(request.action == kBos2Action && request.profile == "int64" &&
              request.parameters.size() == 2 &&
              request.parameters.contains("configuration") &&
              request.parameters.at("configuration") == kBos2Configuration &&
              request.parameters.contains("row_count"),
          "BOS2 graph fixed configuration mismatch");
}

class Bos2Dispatcher final : public StageDispatcher {
 public:
  std::string identity() const override { return std::string(kBos2Dispatcher); }

  StageResult forward(const StageRequest& request,
                      const std::vector<Stream>& inputs) const override {
    validate_bos2_request(request);
    require(inputs.size() == 2 && inputs[0].kind == Kind::Numeric &&
                inputs[0].width == 8 && inputs[1].kind == Kind::Serial &&
                inputs[1].width == 1,
            "BOS2 graph root shape mismatch");
    Block block{"int64", natural(request.parameters.at("row_count")),
                inputs[0].data, inputs[1].data};
    validate_integer_block(block);
    Bytes payload;
    if (!block.values.empty())
      payload = bos_bfe_v2::integer::optimized_compose(
          words<std::int64_t>(block.values));
    require(payload.size() <= kMaximumBytes, "BOS2 payload exceeds byte bound");
    return {{{Kind::Serial, 1, std::move(payload)}, inputs[1]},
            {}, {"bytes", "bytes"}};
  }

  std::vector<Stream> inverse(
      const StageRequest& request, const std::vector<Stream>& outputs,
      std::span<const std::uint8_t> metadata) const override {
    validate_bos2_request(request);
    require(metadata.empty() && outputs.size() == 2 &&
                outputs[0].kind == Kind::Serial && outputs[0].width == 1 &&
                outputs[1].kind == Kind::Serial && outputs[1].width == 1,
            "BOS2 inverse shape or metadata mismatch");
    const auto rows = natural(request.parameters.at("row_count"));
    const auto count = valid_values(rows, outputs[1].data);
    if (count == 0) {
      require(outputs[0].data.empty(),
              "all-missing BOS2 block has a nonempty method payload");
      return {{Kind::Numeric, 8, {}}, outputs[1]};
    }
    auto restored = bos_bfe_v2::integer::optimized_decompose(outputs[0].data);
    require(restored.size() == count, "BOS2 decoded integer count mismatch");
    return {{Kind::Numeric, 8, bytes(restored)}, outputs[1]};
  }
};

class BfeDispatcher final : public StageDispatcher {
 public:
  std::string identity() const override { return std::string(kBfeDispatcher); }

  StageResult forward(const StageRequest& request,
                      const std::vector<Stream>& inputs) const override {
    validate_bfe_request(request);
    const auto width = native_baseline86_v1::width(request.profile);
    require(inputs.size() == 2 && inputs[0].kind == Kind::Numeric &&
                inputs[0].width == width && inputs[1].kind == Kind::Serial &&
                inputs[1].width == 1,
            "BOS BFE graph root shape mismatch");
    Block block{request.profile, natural(request.parameters.at("row_count")),
                inputs[0].data, inputs[1].data};
    validate_float_block(block);
    bos_float_exception::EncodeOptions options;
    options.candidate_precisions = {0, 1, 2, 3, 4, 6};
    options.pilot_values = 256;
    auto payload = bos_float_exception::encode(
        block.values,
        request.profile == "float32" ? bos_float_exception::DType::Float32
                                     : bos_float_exception::DType::Float64,
        kIntegerEncode, nullptr, options);
    require(payload.size() <= kMaximumBytes,
            "BOS BFE payload exceeds byte bound");
    return {{{Kind::Serial, 1, std::move(payload)}, inputs[1]},
            {}, {"bytes", "bytes"}};
  }

  std::vector<Stream> inverse(
      const StageRequest& request, const std::vector<Stream>& outputs,
      std::span<const std::uint8_t> metadata) const override {
    validate_bfe_request(request);
    require(metadata.empty() && outputs.size() == 2 &&
                outputs[0].kind == Kind::Serial && outputs[0].width == 1 &&
                outputs[1].kind == Kind::Serial && outputs[1].width == 1,
            "BOS BFE inverse shape or metadata mismatch");
    const auto rows = natural(request.parameters.at("row_count"));
    const auto count = valid_values(rows, outputs[1].data);
    auto restored =
        bos_float_exception::decode(outputs[0].data, kIntegerDecode);
    const auto width = native_baseline86_v1::width(request.profile);
    require(count <= kMaximumBytes / width &&
                restored.size() == count * width,
            "BOS BFE inverse numeric length mismatch");
    return {{Kind::Numeric, width, std::move(restored)}, outputs[1]};
  }
};

class Dsc3Dispatcher final : public StageDispatcher {
 public:
  std::string identity() const override { return std::string(kDsc3Dispatcher); }

  StageResult forward(const StageRequest& request,
                      const std::vector<Stream>& inputs) const override {
    validate_dsc3_request(request);
    const auto width = native_baseline86_v1::width(request.profile);
    require(inputs.size() == 2 && inputs[0].kind == Kind::Numeric &&
                inputs[0].width == width && inputs[1].kind == Kind::Serial &&
                inputs[1].width == 1,
            "BOS DSC3 graph root shape mismatch");
    Block block{request.profile, natural(request.parameters.at("row_count")),
                inputs[0].data, inputs[1].data};
    validate_float_block(block);
    auto converted = native_baseline86_v1::decimal_scaling_v2::forward(
        block.values, block.dtype,
        static_cast<std::uint16_t>(
            natural(request.parameters.at("decimal_places"))));
    require(block.values.empty() || !converted.values.empty(),
            "nonempty BOS DSC3 input has no integer words");
    auto payload = bos_bfe_v2::integer::optimized_compose(converted.values);
    require(payload.size() <= kMaximumBytes,
            "BOS DSC3 payload exceeds byte bound");
    return {{{Kind::Serial, 1, std::move(payload)}, inputs[1]},
            std::move(converted.metadata), {"bytes", "bytes"}};
  }

  std::vector<Stream> inverse(
      const StageRequest& request, const std::vector<Stream>& outputs,
      std::span<const std::uint8_t> metadata) const override {
    validate_dsc3_request(request);
    require(outputs.size() == 2 && outputs[0].kind == Kind::Serial &&
                outputs[0].width == 1 && outputs[1].kind == Kind::Serial &&
                outputs[1].width == 1,
            "BOS DSC3 inverse shape mismatch");
    const auto rows = natural(request.parameters.at("row_count"));
    const auto count = valid_values(rows, outputs[1].data);
    const auto info = native_baseline86_v1::decimal_scaling_v2::inspect(
        metadata, request.profile);
    require(info.original_count == count &&
                info.decimal_places ==
                    natural(request.parameters.at("decimal_places")) &&
                (count == 0 || info.integer_word_count > 0),
            "BOS DSC3 metadata differs from input contract");
    auto integers = bos_bfe_v2::integer::optimized_decompose(outputs[0].data);
    require(integers.size() == info.integer_word_count,
            "BOS DSC3 decoded integer count mismatch");
    auto restored = native_baseline86_v1::decimal_scaling_v2::inverse(
        integers, metadata, request.profile);
    require(restored.size() ==
                count * native_baseline86_v1::width(request.profile),
            "BOS DSC3 inverse numeric length mismatch");
    return {{Kind::Numeric, native_baseline86_v1::width(request.profile),
             std::move(restored)},
            outputs[1]};
  }
};

Bytes encode_graph(const Block& block, const std::string& policy,
                   const std::string& action,
                   std::map<std::string, std::string> parameters,
                   const StageDispatcher& dispatcher) {
  Recipe recipe;
  recipe.policy_registry = policy;
  Step stage;
  stage.kind = StepKind::Stage;
  stage.inputs = {0, 1};
  stage.stage = {action, block.dtype, std::move(parameters)};
  recipe.steps.push_back(std::move(stage));
  Step stop;
  stop.kind = StepKind::Stop;
  recipe.steps.push_back(stop);
  return native_graph_v1::encode(
      {{{Kind::Numeric, native_baseline86_v1::width(block.dtype),
         block.values},
        true, block.dtype},
       {{Kind::Serial, 1, block.missing}, false,
        "missing_bitmap_lsb_first:rows=" +
            std::to_string(block.row_count)}},
      recipe, dispatcher);
}

Block finish_decode(Decoded decoded) {
  require(decoded.roots.size() == 2 &&
              decoded.roots[0].original_numeric_payload &&
              !decoded.roots[1].original_numeric_payload,
          "BOS hybrid decoded root flags mismatch");
  auto& numeric = decoded.roots[0];
  auto& bitmap = decoded.roots[1];
  const std::string prefix = "missing_bitmap_lsb_first:rows=";
  require(bitmap.logical_view.starts_with(prefix),
          "BOS hybrid missing bitmap row metadata");
  Block block{numeric.logical_view,
              natural(bitmap.logical_view.substr(prefix.size())),
              std::move(numeric.stream.data), std::move(bitmap.stream.data)};
  require(numeric.stream.kind == Kind::Numeric &&
              numeric.stream.width == native_baseline86_v1::width(block.dtype) &&
              bitmap.stream.kind == Kind::Serial && bitmap.stream.width == 1,
          "BOS hybrid decoded root type mismatch");
  native_baseline86_v1::validate(block);
  require(block.dtype == "int64" || block.dtype == "float32" ||
              block.dtype == "float64",
          "BOS route decoded an unsupported dtype");
  require(decoded.original_numeric_bytes == block.values.size(),
          "BOS hybrid numeric denominator mismatch");
  return block;
}

enum class Wire { None, Bos2, Dsc3, Bfe };

Wire wire(const native_graph_v1::detail::Frame& frame) {
  if (frame.policy_registry == kBos2Policy &&
      frame.dispatcher == kBos2Dispatcher)
    return Wire::Bos2;
  if (frame.policy_registry == kBfePolicy &&
      frame.dispatcher == kBfeDispatcher)
    return Wire::Bfe;
  if (frame.policy_registry == kDsc3Policy &&
      frame.dispatcher == kDsc3Dispatcher)
    return Wire::Dsc3;
  return Wire::None;
}

native_graph_v1::detail::Frame parse(std::span<const std::uint8_t> frame) {
  return native_graph_v1::detail::parse_frame(frame, {});
}

FrameInfo inspect_parsed(const native_graph_v1::detail::Frame& frame) {
  require(frame.nodes.size() == 1 &&
              frame.nodes.front().step.kind == StepKind::Stage,
          "BOS hybrid must contain one stage");
  const auto& request = frame.nodes.front().step.stage;
  if (wire(frame) == Wire::Bos2) {
    validate_bos2_request(request);
    return {"int64", natural(request.parameters.at("row_count")), 0,
            "bos2", 1, 0, 0, 0, 1};
  }
  if (wire(frame) == Wire::Bfe) {
    validate_bfe_request(request);
    const auto& node = frame.nodes.front();
    require(!node.outputs.empty(), "BOS BFE frame has no payload output");
    const auto leaf = frame.leaves.find(node.outputs.front());
    require(leaf != frame.leaves.end() && leaf->second.kind == Kind::Serial &&
                leaf->second.width == 1 && leaf->second.data.size() >= 8 &&
                leaf->second.data[0] == 'B' && leaf->second.data[1] == 'F' &&
                leaf->second.data[2] == 'E' && leaf->second.data[3] == '1',
            "BOS BFE payload leaf is malformed");
    const bool raw = leaf->second.data[6] ==
                     static_cast<std::uint8_t>(bos_float_exception::Mode::Raw);
    require(raw || leaf->second.data[6] == static_cast<std::uint8_t>(
                                             bos_float_exception::Mode::Exception),
            "BOS BFE payload mode is invalid");
    return {request.profile, natural(request.parameters.at("row_count")),
            static_cast<std::uint16_t>(natural(
                request.parameters.at("whole_column_decimal_places"))),
            raw ? "bfe1_raw" : "bfe1", 1, raw ? 1U : 0U, 1, 0, 0};
  }
  if (wire(frame) == Wire::Dsc3) {
    validate_dsc3_request(request);
    return {request.profile, natural(request.parameters.at("row_count")),
            static_cast<std::uint16_t>(
                natural(request.parameters.at("decimal_places"))),
            "dsc3", 1, 0, 0, 1, 0};
  }
  fail("frame is outside the BOS BFE hybrid route");
}

bool has_bundle_magic(std::span<const std::uint8_t> frame) {
  return frame.size() >= kBundleMagic.size() &&
         std::equal(kBundleMagic.begin(), kBundleMagic.end(), frame.begin());
}

struct BundleView {
  std::string dtype;
  std::uint16_t whole_p = 0;
  std::uint64_t row_count = 0;
  std::uint64_t valid_count = 0;
  std::vector<std::span<const std::uint8_t>> frames;
};

BundleView parse_bundle(std::span<const std::uint8_t> frame) {
  require(has_bundle_magic(frame), "not a BOS hybrid bundle");
  require(frame.size() >= 44, "truncated BOS hybrid bundle header");
  Cursor cursor(frame);
  const auto magic = cursor.take(kBundleMagic.size());
  require(std::equal(magic.begin(), magic.end(), kBundleMagic.begin()),
          "invalid BOS hybrid bundle magic");
  require(cursor.u16() == kBundleVersion,
          "unsupported BOS hybrid bundle version");
  const auto dtype_code = cursor.u16();
  require(dtype_code == 1 || dtype_code == 2,
          "invalid BOS hybrid bundle dtype");
  BundleView result;
  result.dtype = dtype_code == 1 ? "float32" : "float64";
  result.whole_p = cursor.u16();
  require(result.whole_p <= 324 && cursor.u16() == 0,
          "invalid BOS hybrid bundle precision or reserved field");
  result.row_count = cursor.u64();
  result.valid_count = cursor.u64();
  const auto block_count = cursor.u64();
  require(block_count > 1 && block_count <= result.row_count + 1,
          "invalid BOS hybrid bundle block count");
  require(block_count <= std::numeric_limits<std::size_t>::max(),
          "BOS hybrid bundle block count exceeds platform size");
  result.frames.reserve(static_cast<std::size_t>(block_count));
  for (std::uint64_t index = 0; index != block_count; ++index) {
    const auto size = cursor.u64();
    require(size > 0 && size <= std::numeric_limits<std::size_t>::max(),
            "invalid BOS hybrid bundle member size");
    result.frames.push_back(cursor.take(static_cast<std::size_t>(size)));
  }
  const auto checksum_offset = cursor.position();
  const auto expected = cursor.u32();
  require(cursor.empty(), "trailing bytes in BOS hybrid bundle");
  require(crc32(frame.first(checksum_offset)) == expected,
          "BOS hybrid bundle checksum mismatch");
  return result;
}

Bytes encode_one(const Block& block, std::uint16_t whole_p) {
  validate_float_block(block);
  Dsc3Dispatcher old_dispatcher;
  auto old_frame = encode_graph(block, kDsc3Policy, std::string(kDsc3Action),
                                dsc3_parameters(block, whole_p),
                                old_dispatcher);
  BfeDispatcher bfe_dispatcher;
  auto bfe_frame = encode_graph(block, kBfePolicy, std::string(kBfeAction),
                                bfe_parameters(block, whole_p),
                                bfe_dispatcher);
  return bfe_frame.size() < old_frame.size() ? std::move(bfe_frame)
                                              : std::move(old_frame);
}

Block decode_one(std::span<const std::uint8_t> frame) {
  const auto parsed = parse(frame);
  (void)inspect_parsed(parsed);
  if (wire(parsed) == Wire::Bos2) {
    Bos2Dispatcher dispatcher;
    return finish_decode(native_graph_v1::decode(frame, dispatcher));
  }
  if (wire(parsed) == Wire::Bfe) {
    BfeDispatcher dispatcher;
    return finish_decode(native_graph_v1::decode(frame, dispatcher));
  }
  Dsc3Dispatcher dispatcher;
  return finish_decode(native_graph_v1::decode(frame, dispatcher));
}

Bytes encode_bundle(const Block& original, const std::vector<Block>& blocks,
                    std::uint16_t whole_p) {
  std::vector<Bytes> frames;
  frames.reserve(blocks.size());
  std::size_t total = 44;
  for (const auto& block : blocks) {
    auto frame = encode_one(block, whole_p);
    require(frame.size() <= std::numeric_limits<std::size_t>::max() - total - 8,
            "BOS hybrid bundle size overflow");
    total += 8 + frame.size();
    frames.push_back(std::move(frame));
  }
  Bytes result;
  result.reserve(total);
  result.insert(result.end(), kBundleMagic.begin(), kBundleMagic.end());
  append_u16(result, kBundleVersion);
  append_u16(result, original.dtype == "float32" ? 1 : 2);
  append_u16(result, whole_p);
  append_u16(result, 0);
  append_u64(result, original.row_count);
  append_u64(result,
             original.values.size() / native_baseline86_v1::width(original.dtype));
  append_u64(result, frames.size());
  for (const auto& frame : frames) {
    append_u64(result, frame.size());
    result.insert(result.end(), frame.begin(), frame.end());
  }
  append_u32(result, crc32(result));
  return result;
}

Block decode_bundle(std::span<const std::uint8_t> frame) {
  const auto bundle = parse_bundle(frame);
  const auto width = native_baseline86_v1::width(bundle.dtype);
  require(bundle.valid_count <=
              std::numeric_limits<std::size_t>::max() / width,
          "BOS hybrid decoded numeric size overflow");
  Block result{bundle.dtype, bundle.row_count, {},
               Bytes(static_cast<std::size_t>((bundle.row_count + 7) / 8), 0)};
  result.values.reserve(static_cast<std::size_t>(bundle.valid_count) * width);
  std::uint64_t row_offset = 0;
  for (const auto member : bundle.frames) {
    const auto info = inspect_parsed(parse(member));
    require(info.dtype == bundle.dtype &&
                info.whole_column_decimal_places == bundle.whole_p,
            "BOS hybrid bundle member contract mismatch");
    auto block = decode_one(member);
    require(block.row_count <= bundle.row_count - row_offset,
            "BOS hybrid bundle member rows exceed total");
    result.values.insert(result.values.end(), block.values.begin(),
                         block.values.end());
    for (std::uint64_t local = 0; local != block.row_count; ++local)
      if (missing_at(block.missing, local)) {
        const auto global = row_offset + local;
        result.missing[static_cast<std::size_t>(global / 8)] |=
            static_cast<std::uint8_t>(1U << (global % 8));
      }
    row_offset += block.row_count;
  }
  require(row_offset == result.row_count &&
              result.values.size() ==
                  static_cast<std::size_t>(bundle.valid_count) * width,
          "BOS hybrid bundle reconstructed length mismatch");
  require(valid_values(result.row_count, result.missing) == bundle.valid_count,
          "BOS hybrid bundle reconstructed bitmap mismatch");
  return result;
}

FrameInfo inspect_bundle(std::span<const std::uint8_t> frame) {
  const auto bundle = parse_bundle(frame);
  std::uint64_t rows = 0;
  FrameInfo result{bundle.dtype, bundle.row_count, bundle.whole_p, "chunked",
                   bundle.frames.size(), 0, 0, 0, 0};
  for (const auto member : bundle.frames) {
    const auto info = inspect_parsed(parse(member));
    require(info.dtype == bundle.dtype &&
                info.whole_column_decimal_places == bundle.whole_p &&
                info.row_count <= bundle.row_count - rows,
            "BOS hybrid bundle member metadata mismatch");
    rows += info.row_count;
    result.raw_block_count += info.raw_block_count;
    result.bfe_block_count += info.bfe_block_count;
    result.dsc3_block_count += info.dsc3_block_count;
    result.bos2_block_count += info.bos2_block_count;
  }
  require(rows == bundle.row_count,
          "BOS hybrid bundle row total mismatch");
  return result;
}

}  // namespace

Bytes encode(const Block& block) {
  if (block.dtype == "int64") {
    validate_integer_block(block);
    Bos2Dispatcher dispatcher;
    return encode_graph(
        block, std::string(kBos2Policy), std::string(kBos2Action),
        {{"configuration", std::string(kBos2Configuration)},
         {"row_count", std::to_string(block.row_count)}},
        dispatcher);
  }
  const auto blocks = split_blocks(block);
  const auto whole_p = exact_column_decimal_places(blocks);
  if (blocks.size() == 1) return encode_one(blocks.front(), whole_p);
  return encode_bundle(block, blocks, whole_p);
}

Block decode(std::span<const std::uint8_t> frame) {
  if (has_bundle_magic(frame)) return decode_bundle(frame);
  return decode_one(frame);
}

bool is_frame(std::span<const std::uint8_t> frame) {
  try {
    if (has_bundle_magic(frame)) {
      (void)inspect_bundle(frame);
      return true;
    }
    const auto parsed = parse(frame);
    if (wire(parsed) == Wire::None) return false;
    (void)inspect_parsed(parsed);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

FrameInfo inspect(std::span<const std::uint8_t> frame) {
  if (has_bundle_magic(frame)) return inspect_bundle(frame);
  return inspect_parsed(parse(frame));
}

}  // namespace unified_compression_v1::bos_bfe_bridge
