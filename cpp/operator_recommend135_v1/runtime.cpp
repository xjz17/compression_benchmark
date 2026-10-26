#include "runtime.hpp"

#include "graph_internal.hpp"
#include "policy_dispatch.hpp"

#include "../../external/compression/lz4/lib/lz4frame.h"
#include "../../external/compression/zstd/lib/zstd.h"
#include <zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace operator_recommend135_v1 {
namespace {

using Json = nlohmann::json;
using native_graph_v1::Bytes;
using native_graph_v1::Kind;
using native_graph_v1::RootInput;
using native_graph_v1::StageRequest;
using native_graph_v1::StageResult;
using native_graph_v1::Step;
using native_graph_v1::StepKind;
using native_graph_v1::Stream;
using native_graph_v1::StreamId;

static_assert(std::endian::native == std::endian::little,
              "recommend135 typed values require a little-endian host");

constexpr std::string_view kRegistry = "operator-recommend135-candidates-v1";
constexpr std::string_view kBindingView = "recommend135_binding_v1";
constexpr std::string_view kMissingView = "missing_bitmap_lsb1";
constexpr std::size_t kBindingBytes = 32;
constexpr std::uint64_t kMaxStreamBytes = 256ULL << 20;

constexpr std::string_view kZstd1 = "recommend135.codec.zstd.level1.v1";
constexpr std::string_view kZstd3 = "recommend135.codec.zstd.level3.v1";
constexpr std::string_view kZstd9 = "recommend135.codec.zstd.level9.v1";
constexpr std::string_view kLz4 = "recommend135.codec.lz4frame.level0.v1";
constexpr std::string_view kZlib6 = "recommend135.codec.zlib.level6.v1";
constexpr std::string_view kHuffman =
    "op.entropy_huffman.project_bytes.cpp_v1";
constexpr std::string_view kRans =
    "op.entropy_fse.zstd_rans.cpp_reference_v1";
constexpr std::string_view kColumnize = "columnize";
constexpr std::string_view kXor = "op.xor.previous_word.cpp_v2";
constexpr std::string_view kFor =
    "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1";
constexpr std::string_view kBitpack = "op.bitpack_int.max_value_width.cpp_v1";
constexpr std::string_view kFloatDeconstruct = "float_deconstruct";
constexpr std::string_view kA048Planes = "harness.a048.byte_planes.v1";

constexpr std::int64_t kActionZstd1 = 135001;
constexpr std::int64_t kActionZstd3 = 135003;
constexpr std::int64_t kActionZstd9 = 135009;
constexpr std::int64_t kActionLz4 = 135010;
constexpr std::int64_t kActionZlib6 = 135011;
constexpr std::int64_t kActionColumnize = 135020;
constexpr std::int64_t kActionA048Planes = 135048;
constexpr std::int64_t kActionHuffman = 6212;
constexpr std::int64_t kActionRans = 6215;
constexpr std::int64_t kActionForInt64 = 6137;
constexpr std::int64_t kActionBitpackBytes = 6145;
constexpr std::int64_t kActionXorFloat64 = 6165;
constexpr std::int64_t kActionXorInt64 = 6166;
constexpr std::int64_t kActionFloat32 = 6219;

enum class Plan {
  raw,
  codec,
  operator_only,
  columnize,
  adjacent_xor,
  for_bitpack,
  float_deconstruct,
  a048_planes,
  split_halves,
};

struct Candidate {
  std::uint16_t ordinal;
  const char* id;
  const char* kind;
  Plan plan;
  std::vector<const char*> component_ids;
  std::vector<std::int64_t> action_ids;
  std::vector<std::string_view> terminal_actions;
  std::vector<const char*> dtypes;
  bool requires_nonempty = false;
  const char* primary_action = nullptr;
};

const std::vector<Candidate>& candidates() {
  static const std::vector<Candidate> value{
      {0, "raw", "raw", Plan::raw, {"NGF1_RAW"}, {}, {},
       {"int32", "int64", "float32", "float64"}},
      {1, "zstd1", "algorithm", Plan::codec, {"ZSTD_NATIVE_LEVEL1"},
       {kActionZstd1}, {kZstd1}, {"int32", "int64", "float32", "float64"}},
      {2, "zstd3", "algorithm", Plan::codec, {"ZSTD_NATIVE_LEVEL3"},
       {kActionZstd3}, {kZstd3}, {"int32", "int64", "float32", "float64"}},
      {3, "zstd9", "algorithm", Plan::codec, {"ZSTD_NATIVE_LEVEL9"},
       {kActionZstd9}, {kZstd9}, {"int32", "int64", "float32", "float64"}},
      {4, "lz4frame", "algorithm", Plan::codec, {"LZ4FRAME_NATIVE_LEVEL0"},
       {kActionLz4}, {kLz4}, {"int32", "int64", "float32", "float64"}},
      {5, "zlib6", "algorithm", Plan::codec, {"ZLIB_NATIVE_LEVEL6"},
       {kActionZlib6}, {kZlib6}, {"int32", "int64", "float32", "float64"}},
      {6, "huffman_a077", "operator", Plan::operator_only, {"A077"},
       {kActionHuffman}, {}, {"int32", "int64", "float32", "float64"}, true,
       kHuffman.data()},
      {7, "rans_a080", "operator", Plan::operator_only, {"A080"},
       {kActionRans}, {}, {"int32", "int64", "float32", "float64"}, true,
       kRans.data()},
      {8, "columnize_zstd3", "operator_tree", Plan::columnize,
       {"OPENZL_COLUMNIZE", "ZSTD_NATIVE_LEVEL3"},
       {kActionColumnize, kActionZstd3}, {kZstd3},
       {"int32", "int64", "float32", "float64"}},
      {9, "columnize_lz4", "operator_tree", Plan::columnize,
       {"OPENZL_COLUMNIZE", "LZ4FRAME_NATIVE_LEVEL0"},
       {kActionColumnize, kActionLz4}, {kLz4},
       {"int32", "int64", "float32", "float64"}},
      {10, "columnize_zlib6", "operator_tree", Plan::columnize,
       {"OPENZL_COLUMNIZE", "ZLIB_NATIVE_LEVEL6"},
       {kActionColumnize, kActionZlib6}, {kZlib6},
       {"int32", "int64", "float32", "float64"}},
      {11, "xor_zstd3", "operator_tree", Plan::adjacent_xor,
       {"A031", "ZSTD_NATIVE_LEVEL3"}, {kActionXorFloat64, kActionXorInt64,
        kActionZstd3}, {kZstd3}, {"int64", "float64"}},
      {12, "xor_lz4", "operator_tree", Plan::adjacent_xor,
       {"A031", "LZ4FRAME_NATIVE_LEVEL0"}, {kActionXorFloat64, kActionXorInt64,
        kActionLz4}, {kLz4}, {"int64", "float64"}},
      {13, "xor_zlib6", "operator_tree", Plan::adjacent_xor,
       {"A031", "ZLIB_NATIVE_LEVEL6"}, {kActionXorFloat64, kActionXorInt64,
        kActionZlib6}, {kZlib6}, {"int64", "float64"}},
      {14, "float_deconstruct_zstd3", "operator_tree", Plan::float_deconstruct,
       {"A103", "ZSTD_NATIVE_LEVEL3"}, {kActionFloat32, kActionZstd3},
       {kZstd3}, {"float32"}, true},
      {15, "float_deconstruct_lz4", "operator_tree", Plan::float_deconstruct,
       {"A103", "LZ4FRAME_NATIVE_LEVEL0"}, {kActionFloat32, kActionLz4},
       {kLz4}, {"float32"}, true},
      {16, "float_deconstruct_zstd3_lz4", "algorithm_combination",
       Plan::float_deconstruct,
       {"A103", "ZSTD_NATIVE_LEVEL3", "LZ4FRAME_NATIVE_LEVEL0"},
       {kActionFloat32, kActionZstd3, kActionLz4}, {kZstd3, kLz4},
       {"float32"}, true},
      {17, "for_bitpack_zstd3", "operator_tree", Plan::for_bitpack,
       {"A013", "A020", "ZSTD_NATIVE_LEVEL3"},
       {kActionForInt64, kActionBitpackBytes, kActionZstd3}, {kZstd3},
       {"int64"}, true},
      {18, "for_bitpack_lz4", "operator_tree", Plan::for_bitpack,
       {"A013", "A020", "LZ4FRAME_NATIVE_LEVEL0"},
       {kActionForInt64, kActionBitpackBytes, kActionLz4}, {kLz4},
       {"int64"}, true},
      {19, "a048_planes_zstd3", "operator_tree", Plan::a048_planes,
       {"A048_TYPED_PLANES", "ZSTD_NATIVE_LEVEL3"},
       {kActionA048Planes, kActionZstd3}, {kZstd3}, {"int64", "float64"}},
      {20, "a048_planes_lz4", "operator_tree", Plan::a048_planes,
       {"A048_TYPED_PLANES", "LZ4FRAME_NATIVE_LEVEL0"},
       {kActionA048Planes, kActionLz4}, {kLz4}, {"int64", "float64"}},
      {21, "a048_planes_zstd3_lz4", "algorithm_combination",
       Plan::a048_planes,
       {"A048_TYPED_PLANES", "ZSTD_NATIVE_LEVEL3", "LZ4FRAME_NATIVE_LEVEL0"},
       {kActionA048Planes, kActionZstd3, kActionLz4}, {kZstd3, kLz4},
       {"int64", "float64"}},
      {22, "split_zstd3_lz4", "algorithm_combination", Plan::split_halves,
       {"NGF1_SPLIT_SEGMENTS", "ZSTD_NATIVE_LEVEL3", "LZ4FRAME_NATIVE_LEVEL0"},
       {kActionZstd3, kActionLz4}, {kZstd3, kLz4},
       {"int32", "int64", "float32", "float64"}},
      {23, "split_lz4_zstd3", "algorithm_combination", Plan::split_halves,
       {"NGF1_SPLIT_SEGMENTS", "LZ4FRAME_NATIVE_LEVEL0", "ZSTD_NATIVE_LEVEL3"},
       {kActionLz4, kActionZstd3}, {kLz4, kZstd3},
       {"int32", "int64", "float32", "float64"}},
  };
  return value;
}

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

const Candidate* candidate(std::string_view id) {
  const auto& all = candidates();
  const auto it = std::find_if(all.begin(), all.end(),
                               [&](const Candidate& value) { return id == value.id; });
  return it == all.end() ? nullptr : &*it;
}

const Candidate& candidate(std::uint16_t ordinal) {
  const auto& all = candidates();
  require(ordinal < all.size() && all[ordinal].ordinal == ordinal,
          "unknown recommend135 candidate ordinal");
  return all[ordinal];
}

std::uint32_t dtype_width(std::string_view dtype) {
  if (dtype == "int32" || dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "float64") return 8;
  fail("unsupported source dtype: " + std::string(dtype));
}

std::uint8_t dtype_code(std::string_view dtype) {
  if (dtype == "int32") return 1;
  if (dtype == "int64") return 2;
  if (dtype == "float32") return 3;
  if (dtype == "float64") return 4;
  fail("unsupported source dtype: " + std::string(dtype));
}

std::string dtype_from_code(std::uint8_t code) {
  switch (code) {
    case 1: return "int32";
    case 2: return "int64";
    case 3: return "float32";
    case 4: return "float64";
    default: fail("unknown recommend135 dtype code");
  }
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char* message) {
  require(left == 0 || right <= std::numeric_limits<std::uint64_t>::max() / left,
          message);
  return left * right;
}

std::uint64_t valid_count(const Block& block) {
  const auto expected_missing = block.row_count / 8 + (block.row_count % 8 != 0);
  require(block.missing.size() == expected_missing,
          "missing bitmap length does not match row_count");
  if (block.row_count % 8 != 0 && !block.missing.empty()) {
    const auto used = static_cast<unsigned>(block.row_count % 8);
    const auto unused_mask = static_cast<std::uint8_t>(0xffU << used);
    require((block.missing.back() & unused_mask) == 0,
            "missing bitmap has set bits beyond row_count");
  }
  std::uint64_t missing = 0;
  for (const auto byte : block.missing) missing += std::popcount(byte);
  require(missing <= block.row_count, "missing count exceeds row_count");
  return block.row_count - missing;
}

std::uint64_t validate_block(const Block& block) {
  const auto width = dtype_width(block.dtype);
  const auto count = valid_count(block);
  require(checked_multiply(count, width, "source value byte count overflow") ==
              block.values.size(),
          "compact value bytes do not match valid_count and source dtype");
  require(block.values.size() <= kMaxStreamBytes,
          "source value stream exceeds the 256 MiB block bound");
  require(block.missing.size() <= kMaxStreamBytes,
          "missing stream exceeds the 256 MiB block bound");
  return count;
}

bool supports(const Candidate& selected, std::string_view dtype) {
  return std::any_of(selected.dtypes.begin(), selected.dtypes.end(),
                     [&](const char* value) { return dtype == value; });
}

bool eligible_impl(const Candidate& selected, const Block& block) {
  const auto count = validate_block(block);
  if (!supports(selected, block.dtype)) return false;
  if (selected.requires_nonempty && count == 0) return false;
  return true;
}

void put(Bytes& output, std::uint64_t value, unsigned width) {
  for (unsigned index = 0; index < width; ++index)
    output.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
}

std::uint64_t get(std::span<const std::uint8_t> input, std::size_t& position,
                  unsigned width) {
  require(position <= input.size() && width <= input.size() - position,
          "truncated recommend135 metadata");
  std::uint64_t value = 0;
  for (unsigned index = 0; index < width; ++index)
    value |= std::uint64_t(input[position++]) << (8 * index);
  return value;
}

struct Binding {
  const Candidate* selected = nullptr;
  std::string dtype;
  std::uint64_t row_count = 0;
  std::uint64_t value_count = 0;
  std::uint64_t value_bytes = 0;
};

Bytes binding_bytes(const Candidate& selected, const Block& block,
                    std::uint64_t count) {
  Bytes output{'R', '1', 'B', '1'};
  put(output, 1, 1);
  put(output, dtype_code(block.dtype), 1);
  put(output, selected.ordinal, 2);
  put(output, block.row_count, 8);
  put(output, count, 8);
  put(output, block.values.size(), 8);
  require(output.size() == kBindingBytes, "internal binding size mismatch");
  return output;
}

Binding parse_binding(std::span<const std::uint8_t> bytes) {
  require(bytes.size() == kBindingBytes &&
              std::equal(bytes.begin(), bytes.begin() + 4, "R1B1"),
          "invalid recommend135 binding magic or size");
  std::size_t position = 4;
  require(get(bytes, position, 1) == 1, "unsupported recommend135 binding version");
  Binding result;
  result.dtype = dtype_from_code(static_cast<std::uint8_t>(get(bytes, position, 1)));
  result.selected = &candidate(static_cast<std::uint16_t>(get(bytes, position, 2)));
  result.row_count = get(bytes, position, 8);
  result.value_count = get(bytes, position, 8);
  result.value_bytes = get(bytes, position, 8);
  require(position == bytes.size(), "trailing recommend135 binding bytes");
  require(checked_multiply(result.value_count, dtype_width(result.dtype),
                           "bound source byte count overflow") == result.value_bytes,
          "recommend135 binding value count differs from dtype width");
  return result;
}

bool is_terminal(std::string_view action) {
  return action == kZstd1 || action == kZstd3 || action == kZstd9 ||
         action == kLz4 || action == kZlib6;
}

int zstd_level(std::string_view action) {
  if (action == kZstd1) return 1;
  if (action == kZstd3) return 3;
  if (action == kZstd9) return 9;
  fail("not a recommend135 Zstd action");
}

std::uint8_t codec_tag(std::string_view action) {
  if (action == kZstd1) return 1;
  if (action == kZstd3) return 2;
  if (action == kZstd9) return 3;
  if (action == kLz4) return 4;
  if (action == kZlib6) return 5;
  fail("unknown recommend135 terminal codec action");
}

std::int64_t action_id(std::string_view action) {
  if (action == kZstd1) return kActionZstd1;
  if (action == kZstd3) return kActionZstd3;
  if (action == kZstd9) return kActionZstd9;
  if (action == kLz4) return kActionLz4;
  if (action == kZlib6) return kActionZlib6;
  fail("unknown recommend135 action ID");
}

Bytes terminal_metadata(std::string_view action, const Stream& input) {
  require(input.width <= UINT16_MAX, "terminal input width is too large");
  Bytes result{'R', 'T', 'C', '1'};
  put(result, codec_tag(action), 1);
  put(result, static_cast<std::uint8_t>(input.kind), 1);
  put(result, input.width, 2);
  put(result, input.data.size(), 8);
  require(result.size() == 16, "internal terminal metadata size mismatch");
  return result;
}

struct TerminalMetadata {
  Kind kind = Kind::Serial;
  std::uint32_t width = 1;
  std::uint64_t bytes = 0;
};

TerminalMetadata parse_terminal_metadata(std::string_view action,
                                         std::span<const std::uint8_t> metadata) {
  require(metadata.size() == 16 &&
              std::equal(metadata.begin(), metadata.begin() + 4, "RTC1"),
          "invalid recommend135 terminal metadata");
  std::size_t position = 4;
  require(get(metadata, position, 1) == codec_tag(action),
          "terminal metadata codec differs from action");
  TerminalMetadata result;
  result.kind = static_cast<Kind>(get(metadata, position, 1));
  result.width = static_cast<std::uint32_t>(get(metadata, position, 2));
  result.bytes = get(metadata, position, 8);
  require(position == metadata.size() && result.width != 0 &&
              result.bytes <= kMaxStreamBytes && result.bytes % result.width == 0,
          "invalid terminal source kind, width, or length");
  if (result.kind == Kind::Serial) require(result.width == 1, "serial terminal width must be one");
  else if (result.kind == Kind::Numeric)
    require(result.width == 1 || result.width == 2 || result.width == 4 ||
                result.width == 8,
            "invalid numeric terminal width");
  else require(result.kind == Kind::Struct, "invalid terminal source kind");
  return result;
}

Bytes zstd_compress(std::span<const std::uint8_t> input, int level) {
  const auto bound = ZSTD_compressBound(input.size());
  require(!ZSTD_isError(bound) && bound <= kMaxStreamBytes,
          "Zstd compressed bound exceeds the block limit");
  Bytes output(bound);
  const std::uint8_t empty = 0;
  const auto size = ZSTD_compress(output.data(), output.size(),
                                  input.empty() ? &empty : input.data(), input.size(), level);
  require(!ZSTD_isError(size),
          std::string("Zstd encode failed: ") + ZSTD_getErrorName(size));
  output.resize(size);
  return output;
}

Bytes zstd_decompress(std::span<const std::uint8_t> input, std::uint64_t expected) {
  require(expected <= kMaxStreamBytes, "Zstd decoded size exceeds the block limit");
  const auto frame_size = ZSTD_findFrameCompressedSize(input.data(), input.size());
  require(!ZSTD_isError(frame_size) && frame_size == input.size(),
          "invalid, concatenated, or trailing Zstd frame");
  const auto content = ZSTD_getFrameContentSize(input.data(), input.size());
  require(content != ZSTD_CONTENTSIZE_ERROR && content != ZSTD_CONTENTSIZE_UNKNOWN &&
              content == expected,
          "Zstd content size differs from inverse metadata");
  Bytes output(static_cast<std::size_t>(expected));
  std::uint8_t empty = 0;
  const auto size = ZSTD_decompress(output.empty() ? &empty : output.data(), output.size(),
                                    input.data(), input.size());
  require(!ZSTD_isError(size) && size == expected,
          ZSTD_isError(size) ? std::string("Zstd decode failed: ") +
                                   ZSTD_getErrorName(size)
                             : "Zstd decoded size mismatch");
  return output;
}

Bytes lz4_compress(std::span<const std::uint8_t> input) {
  LZ4F_preferences_t preferences = LZ4F_INIT_PREFERENCES;
  preferences.frameInfo.blockSizeID = LZ4F_max64KB;
  preferences.frameInfo.blockMode = LZ4F_blockLinked;
  preferences.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
  preferences.frameInfo.blockChecksumFlag = LZ4F_noBlockChecksum;
  preferences.frameInfo.contentSize = input.size();
  preferences.compressionLevel = 0;
  const auto bound = LZ4F_compressFrameBound(input.size(), &preferences);
  require(!LZ4F_isError(bound) && bound <= kMaxStreamBytes,
          "LZ4 frame bound exceeds the block limit");
  Bytes output(bound);
  const std::uint8_t empty = 0;
  const auto size = LZ4F_compressFrame(output.data(), output.size(),
                                       input.empty() ? &empty : input.data(),
                                       input.size(), &preferences);
  require(!LZ4F_isError(size),
          std::string("LZ4 frame encode failed: ") + LZ4F_getErrorName(size));
  output.resize(size);
  return output;
}

Bytes lz4_decompress(std::span<const std::uint8_t> input, std::uint64_t expected) {
  require(expected <= kMaxStreamBytes, "LZ4 decoded size exceeds the block limit");
  LZ4F_dctx* context = nullptr;
  const auto created = LZ4F_createDecompressionContext(&context, LZ4F_VERSION);
  require(!LZ4F_isError(created) && context != nullptr,
          "LZ4 decompression context creation failed");
  struct Release {
    LZ4F_dctx* value;
    ~Release() { if (value) LZ4F_freeDecompressionContext(value); }
  } release{context};
  Bytes storage(static_cast<std::size_t>(expected) + 1);
  std::size_t source_position = 0;
  std::size_t output_position = 0;
  std::size_t hint = 1;
  while (hint != 0) {
    require(source_position < input.size(), "truncated LZ4 frame");
    auto source_size = input.size() - source_position;
    auto output_size = storage.size() - output_position;
    const auto old_source = source_position;
    const auto old_output = output_position;
    hint = LZ4F_decompress(context, storage.data() + output_position, &output_size,
                           input.data() + source_position, &source_size, nullptr);
    require(!LZ4F_isError(hint),
            std::string("LZ4 frame decode failed: ") + LZ4F_getErrorName(hint));
    source_position += source_size;
    output_position += output_size;
    require(output_position <= expected, "LZ4 frame decoded more bytes than declared");
    require(source_position != old_source || output_position != old_output || hint == 0,
            "LZ4 frame decoder made no progress");
  }
  require(source_position == input.size() && output_position == expected,
          "LZ4 frame has trailing bytes or a decoded size mismatch");
  storage.resize(output_position);
  return storage;
}

Bytes zlib_compress(std::span<const std::uint8_t> input) {
  require(input.size() <= std::numeric_limits<uLong>::max(),
          "zlib input exceeds uLong");
  uLongf capacity = compressBound(static_cast<uLong>(input.size()));
  require(capacity <= kMaxStreamBytes, "zlib compressed bound exceeds the block limit");
  Bytes output(static_cast<std::size_t>(capacity));
  const std::uint8_t empty = 0;
  const auto status = compress2(output.data(), &capacity,
                                input.empty() ? &empty : input.data(),
                                static_cast<uLong>(input.size()), 6);
  require(status == Z_OK, "zlib level-6 encode failed");
  output.resize(static_cast<std::size_t>(capacity));
  return output;
}

Bytes zlib_decompress(std::span<const std::uint8_t> input, std::uint64_t expected) {
  require(expected <= kMaxStreamBytes && input.size() <= UINT_MAX,
          "zlib decode bound exceeded");
  Bytes storage(static_cast<std::size_t>(expected) + 1);
  z_stream stream{};
  require(inflateInit(&stream) == Z_OK, "zlib inflate initialization failed");
  struct End {
    z_stream* value;
    ~End() { inflateEnd(value); }
  } end{&stream};
  stream.next_in = const_cast<Bytef*>(input.data());
  stream.avail_in = static_cast<uInt>(input.size());
  stream.next_out = storage.data();
  stream.avail_out = static_cast<uInt>(storage.size());
  const auto status = inflate(&stream, Z_FINISH);
  require(status == Z_STREAM_END && stream.avail_in == 0 &&
              stream.total_out == expected,
          "invalid, truncated, trailing, or wrong-sized zlib stream");
  storage.resize(static_cast<std::size_t>(expected));
  return storage;
}

Bytes terminal_forward(std::string_view action, std::span<const std::uint8_t> input) {
  if (action == kZstd1 || action == kZstd3 || action == kZstd9)
    return zstd_compress(input, zstd_level(action));
  if (action == kLz4) return lz4_compress(input);
  if (action == kZlib6) return zlib_compress(input);
  fail("terminal action is not allowlisted");
}

Bytes terminal_inverse(std::string_view action, std::span<const std::uint8_t> input,
                       std::uint64_t expected) {
  if (action == kZstd1 || action == kZstd3 || action == kZstd9)
    return zstd_decompress(input, expected);
  if (action == kLz4) return lz4_decompress(input, expected);
  if (action == kZlib6) return zlib_decompress(input, expected);
  fail("terminal action is not allowlisted");
}

const std::set<std::string_view>& allowed_actions() {
  static const std::set<std::string_view> value{
      kZstd1, kZstd3, kZstd9, kLz4, kZlib6, kHuffman, kRans, kColumnize,
      kXor, kFor, kBitpack, kFloatDeconstruct, kA048Planes};
  return value;
}

class CandidateDispatcher final : public native_graph_v1::StageDispatcher {
 public:
  explicit CandidateDispatcher(bool audit_fallbacks = false)
      : inner_(audit_fallbacks) {}

  std::string identity() const override {
    return "operator_recommend135_v1.dispatcher.terminal-codecs-v1+" +
           inner_.identity();
  }

  StageResult forward(const StageRequest& request,
                      const std::vector<Stream>& inputs) const override {
    enforce(request);
    if (!is_terminal(request.action)) return inner_.forward(request, inputs);
    require(request.profile == "bytes" && request.parameters.empty(),
            "terminal codec requires the frozen bytes profile without parameters");
    require(inputs.size() == 1, "terminal codec requires one input stream");
    const auto& input = inputs.front();
    return {{{Kind::Serial, 1, terminal_forward(request.action, input.data)}},
            terminal_metadata(request.action, input), {"bytes"}};
  }

  std::vector<Stream> inverse(const StageRequest& request,
                              const std::vector<Stream>& outputs,
                              std::span<const std::uint8_t> metadata) const override {
    enforce(request);
    if (!is_terminal(request.action)) return inner_.inverse(request, outputs, metadata);
    require(request.profile == "bytes" && request.parameters.empty(),
            "terminal codec requires the frozen bytes profile without parameters");
    require(outputs.size() == 1 && outputs.front().kind == Kind::Serial &&
                outputs.front().width == 1,
            "terminal codec requires one serial output stream");
    const auto source = parse_terminal_metadata(request.action, metadata);
    return {{source.kind, source.width,
             terminal_inverse(request.action, outputs.front().data, source.bytes)}};
  }

  const operator_harness_v2::FallbackAudit& fallback_audit() const {
    return inner_.fallback_audit();
  }

 private:
  static void enforce(const StageRequest& request) {
    require(allowed_actions().contains(request.action),
            "candidate action is not in the recommend135 allowlist: " + request.action);
    require(!operator_harness_v2::excluded_action_reason(request.action).has_value(),
            "candidate action belongs to a disabled family");
  }

  operator_harness_v2::PolicyDispatcher inner_;
};

native_graph_v1::Limits limits() {
  native_graph_v1::Limits result;
  result.max_nodes = 32;
  result.max_active_streams = 16;
  result.max_ordinary_depth = 8;
  result.max_stream_bytes = kMaxStreamBytes;
  result.max_working_bytes = 512ULL << 20;
  result.max_frame_bytes = 1ULL << 30;
  result.max_metadata_bytes = 1ULL << 20;
  return result;
}

Kind root_kind(const Candidate& selected) {
  return selected.plan == Plan::columnize ? Kind::Struct : Kind::Numeric;
}

std::int64_t xor_action_id(std::string_view dtype) {
  require(dtype == "int64" || dtype == "float64", "adjacent XOR requires 64-bit dtype");
  return dtype == "int64" ? kActionXorInt64 : kActionXorFloat64;
}

StageRequest terminal_request(std::string_view action) {
  return {std::string(action), "bytes", {}};
}

StageRequest columnize_request(std::uint32_t width) {
  return {std::string(kColumnize), "bytes",
          {{"mode", "transpose"}, {"width", std::to_string(width)}}};
}

StageRequest bitpack_request() {
  return {std::string(kBitpack), "bytes",
          {{"alignment_bytes", "1"}, {"bit_order", "msb"},
           {"minimum_width", "0"}, {"wide_cutoff", "64"}}};
}

StageRequest request_for_primary(const Candidate& selected,
                                 std::string_view dtype,
                                 std::uint32_t width) {
  switch (selected.plan) {
    case Plan::operator_only:
      return {selected.primary_action, "bytes", {}};
    case Plan::columnize:
      return columnize_request(width);
    case Plan::adjacent_xor:
      return {std::string(kXor), std::string(dtype), {}};
    case Plan::for_bitpack:
      return {std::string(kFor), "int64", {}};
    case Plan::float_deconstruct:
      return {std::string(kFloatDeconstruct), "float32", {{"format", "fp32"}}};
    case Plan::a048_planes:
      return {std::string(kA048Planes), "uint64", {{"bit_width", "64"}}};
    default:
      fail("candidate has no primary stage request");
  }
}

std::int64_t primary_action_id(const Candidate& selected, std::string_view dtype) {
  switch (selected.plan) {
    case Plan::operator_only:
      return selected.primary_action == kHuffman ? kActionHuffman : kActionRans;
    case Plan::columnize: return kActionColumnize;
    case Plan::adjacent_xor: return xor_action_id(dtype);
    case Plan::for_bitpack: return kActionForInt64;
    case Plan::float_deconstruct: return kActionFloat32;
    case Plan::a048_planes: return kActionA048Planes;
    default: fail("candidate has no primary action ID");
  }
}

std::string_view terminal_for(const Candidate& selected, std::size_t port) {
  require(!selected.terminal_actions.empty(), "candidate has no terminal codec");
  return selected.terminal_actions[port % selected.terminal_actions.size()];
}

std::vector<RootInput> roots(const Candidate& selected, const Block& block,
                             std::uint64_t count) {
  return {{{root_kind(selected), dtype_width(block.dtype), block.values}, true,
           block.dtype},
          {{Kind::Serial, 1, block.missing}, false, std::string(kMissingView)},
          {{Kind::Serial, 1, binding_bytes(selected, block, count)}, false,
           std::string(kBindingView)}};
}

std::vector<StreamId> apply_stage(native_graph_v1::GraphSession& graph,
                                  std::vector<StreamId> inputs,
                                  StageRequest request,
                                  std::int64_t policy_action_id) {
  Step step;
  step.kind = StepKind::Stage;
  step.inputs = std::move(inputs);
  step.stage = std::move(request);
  step.policy_action_id = policy_action_id;
  return graph.apply(step);
}

void apply_terminals(native_graph_v1::GraphSession& graph,
                     const Candidate& selected,
                     const std::vector<StreamId>& inputs) {
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto action = terminal_for(selected, index);
    (void)apply_stage(graph, {inputs[index]}, terminal_request(action), action_id(action));
  }
}

Bytes encode_with_dispatcher(const Candidate& selected, const Block& block,
                             CandidateDispatcher& dispatcher) {
  const auto count = validate_block(block);
  require(eligible_impl(selected, block),
          "candidate is not eligible for this dtype or shape: " +
              std::string(selected.id));
  native_graph_v1::GraphSession graph(roots(selected, block, count), dispatcher,
                                      limits(), std::string(kRegistry));
  if (selected.plan == Plan::raw) return graph.stop();
  if (selected.plan == Plan::codec) {
    apply_terminals(graph, selected, {0});
    return graph.stop();
  }
  if (selected.plan == Plan::split_halves) {
    Step split;
    split.kind = StepKind::SplitSegments;
    split.inputs = {0};
    split.segment_sizes = {count / 2, count - count / 2};
    split.policy_action_id = -1;
    apply_terminals(graph, selected, graph.apply(split));
    return graph.stop();
  }
  auto outputs = apply_stage(
      graph, {0}, request_for_primary(selected, block.dtype, dtype_width(block.dtype)),
      primary_action_id(selected, block.dtype));
  if (selected.plan == Plan::for_bitpack) {
    require(outputs.size() == 2, "FOR candidate expected two physical ports");
    std::vector<StreamId> packed;
    for (const auto output : outputs) {
      auto next = apply_stage(graph, {output}, bitpack_request(), kActionBitpackBytes);
      require(next.size() == 1, "bitpack candidate expected one physical port");
      packed.push_back(next.front());
    }
    outputs = std::move(packed);
  }
  if (!selected.terminal_actions.empty()) apply_terminals(graph, selected, outputs);
  return graph.stop();
}

struct ExpectedNode {
  StepKind kind;
  std::vector<StreamId> inputs;
  StageRequest request;
  std::vector<std::uint64_t> segments;
  std::int64_t policy_action_id;
  std::size_t output_count;
};

void expect_node(const native_graph_v1::detail::Node& actual,
                 const ExpectedNode& expected, StreamId& next) {
  require(actual.step.kind == expected.kind && actual.step.inputs == expected.inputs &&
              actual.step.stage == expected.request &&
              actual.step.segment_sizes == expected.segments &&
              actual.step.policy_action_id == expected.policy_action_id,
          "frame node differs from the candidate's frozen recipe");
  require(actual.outputs.size() == expected.output_count,
          "frame node output count differs from frozen recipe");
  for (const auto output : actual.outputs)
    require(output == next++, "frame node output IDs are not canonical");
}

void validate_recipe(const Candidate& selected, const Binding& binding,
                     const native_graph_v1::detail::Frame& frame) {
  std::vector<ExpectedNode> expected;
  StreamId next = 3;
  auto terminal = [&](StreamId input, std::string_view action) {
    expected.push_back({StepKind::Stage, {input}, terminal_request(action), {},
                        action_id(action), 1});
    return next++;
  };
  if (selected.plan == Plan::raw) {
    // No nodes.
  } else if (selected.plan == Plan::codec) {
    (void)terminal(0, terminal_for(selected, 0));
  } else if (selected.plan == Plan::split_halves) {
    expected.push_back({StepKind::SplitSegments, {0}, {},
                        {binding.value_count / 2,
                         binding.value_count - binding.value_count / 2},
                        -1, 2});
    const auto first = next++;
    const auto second = next++;
    (void)terminal(first, terminal_for(selected, 0));
    (void)terminal(second, terminal_for(selected, 1));
  } else {
    const auto width = dtype_width(binding.dtype);
    const auto request = request_for_primary(selected, binding.dtype, width);
    std::size_t ports = 1;
    if (selected.plan == Plan::columnize) ports = width;
    else if (selected.plan == Plan::for_bitpack ||
             selected.plan == Plan::float_deconstruct)
      ports = 2;
    else if (selected.plan == Plan::a048_planes)
      ports = 8;
    expected.push_back({StepKind::Stage, {0}, request, {},
                        primary_action_id(selected, binding.dtype), ports});
    std::vector<StreamId> outputs;
    for (std::size_t port = 0; port < ports; ++port) outputs.push_back(next++);
    if (selected.plan == Plan::for_bitpack) {
      std::vector<StreamId> packed;
      for (const auto output : outputs) {
        expected.push_back({StepKind::Stage, {output}, bitpack_request(), {},
                            kActionBitpackBytes, 1});
        packed.push_back(next++);
      }
      outputs = std::move(packed);
    }
    if (!selected.terminal_actions.empty())
      for (std::size_t port = 0; port < outputs.size(); ++port)
        (void)terminal(outputs[port], terminal_for(selected, port));
  }
  require(frame.nodes.size() == expected.size(),
          "frame node count differs from the candidate's frozen recipe");
  next = 3;
  for (std::size_t index = 0; index < expected.size(); ++index)
    expect_node(frame.nodes[index], expected[index], next);
}

void preflight_frame(const native_graph_v1::detail::Frame& frame,
                     const Binding& binding,
                     const CandidateDispatcher& dispatcher) {
  require(frame.dispatcher == dispatcher.identity(),
          "recommend135 dispatcher identity mismatch");
  require(frame.policy_registry == kRegistry,
          "recommend135 policy registry mismatch");
  require(frame.roots.size() == 3, "recommend135 frame must have exactly three roots");
  const auto& selected = *binding.selected;
  const auto width = dtype_width(binding.dtype);
  require(frame.roots[0].spec.kind == root_kind(selected) &&
              frame.roots[0].spec.width == width &&
              frame.roots[0].spec.bytes == binding.value_bytes &&
              frame.roots[0].numeric && frame.roots[0].logical_view == binding.dtype,
          "recommend135 value-root binding mismatch");
  const auto missing_bytes = binding.row_count / 8 + (binding.row_count % 8 != 0);
  require(frame.roots[1].spec.kind == Kind::Serial && frame.roots[1].spec.width == 1 &&
              frame.roots[1].spec.bytes == missing_bytes && !frame.roots[1].numeric &&
              frame.roots[1].logical_view == kMissingView,
          "recommend135 missing-root binding mismatch");
  require(frame.roots[2].spec.kind == Kind::Serial && frame.roots[2].spec.width == 1 &&
              frame.roots[2].spec.bytes == kBindingBytes && !frame.roots[2].numeric &&
              frame.roots[2].logical_view == kBindingView,
          "recommend135 schema-root binding mismatch");
  require(frame.numeric_bytes == binding.value_bytes,
          "recommend135 numeric denominator mismatch");
  require(frame.leaves.contains(1) && frame.leaves.contains(2) &&
              frame.leaves.at(2).data.size() == kBindingBytes,
          "recommend135 auxiliary roots must remain terminal leaves");
  for (const auto& node : frame.nodes) {
    require(node.step.policy_action_id < 6225 || node.step.policy_action_id > 6229,
            "A106-A110 action ID is disabled by recommend135");
    if (node.step.kind == StepKind::Stage) {
      require(allowed_actions().contains(node.step.stage.action),
              "frame stage is outside the recommend135 action allowlist");
      require(!operator_harness_v2::excluded_action_reason(node.step.stage.action),
              "frame stage belongs to a disabled family");
    }
    require(std::find(node.step.inputs.begin(), node.step.inputs.end(), 1) ==
                node.step.inputs.end() &&
                std::find(node.step.inputs.begin(), node.step.inputs.end(), 2) ==
                    node.step.inputs.end(),
            "candidate recipe consumed an auxiliary root");
  }
  validate_recipe(selected, binding, frame);
}

Block decode_with_dispatcher(std::span<const std::uint8_t> bytes,
                             CandidateDispatcher& dispatcher) {
  const auto frame = native_graph_v1::detail::parse_frame(bytes, limits());
  require(frame.leaves.contains(2), "recommend135 binding root is not a leaf");
  const auto binding = parse_binding(frame.leaves.at(2).data);
  preflight_frame(frame, binding, dispatcher);
  auto decoded = native_graph_v1::decode(bytes, dispatcher, limits());
  require(decoded.roots.size() == 3 && decoded.original_numeric_bytes == binding.value_bytes,
          "recommend135 decoded root count or denominator mismatch");
  require(decoded.roots[0].stream.data.size() == binding.value_bytes &&
              decoded.roots[1].stream.data.size() ==
                  binding.row_count / 8 + (binding.row_count % 8 != 0) &&
              decoded.roots[2].stream.data == frame.leaves.at(2).data,
          "recommend135 decoded roots differ from their binding");
  Block result{binding.dtype, binding.row_count,
               std::move(decoded.roots[0].stream.data),
               std::move(decoded.roots[1].stream.data)};
  require(validate_block(result) == binding.value_count,
          "decoded valid count differs from recommend135 binding");
  require(eligible_impl(*binding.selected, result),
          "decoded block violates its candidate's dtype/shape eligibility");
  return result;
}

std::string hex(std::span<const std::uint8_t> bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

std::string terminal_label(std::string_view action) {
  if (action == kZstd1) return "Zstd-1";
  if (action == kZstd3) return "Zstd-3";
  if (action == kZstd9) return "Zstd-9";
  if (action == kLz4) return "LZ4-frame-0";
  if (action == kZlib6) return "zlib-6";
  fail("unknown terminal label");
}

Json recipe_json(const Candidate& selected) {
  Json result{{"policy_registry", kRegistry},
              {"root_count", 3},
              {"binding_schema", "R1B1"},
              {"plan", ""},
              {"terminal_assignment", nullptr},
              {"steps", Json::array()}};
  auto stage = [&](std::string_view action, Json parameters = Json::object()) {
    result["steps"].push_back({{"kind", "stage"}, {"action", action},
                                {"parameters", std::move(parameters)}});
  };
  switch (selected.plan) {
    case Plan::raw:
      result["plan"] = "raw_three_roots";
      break;
    case Plan::codec:
      result["plan"] = "whole_values_terminal_codec";
      stage(selected.terminal_actions.front());
      break;
    case Plan::operator_only:
      result["plan"] = "single_operator";
      stage(selected.primary_action);
      break;
    case Plan::columnize:
      result["plan"] = "byte_columnize_then_terminal_per_plane";
      stage(kColumnize, {{"mode", "transpose"}, {"width", "source_width"}});
      break;
    case Plan::adjacent_xor:
      result["plan"] = "native_width8_adjacent_xor_then_terminal";
      stage(kXor, {{"profile", "source_dtype"}});
      break;
    case Plan::for_bitpack:
      result["plan"] = "int64_for_then_bitpack_each_port_then_terminal";
      stage(kFor, {{"profile", "int64"}});
      stage(kBitpack, {{"alignment_bytes", 1}, {"bit_order", "msb"},
                       {"minimum_width", 0}, {"wide_cutoff", 64}});
      break;
    case Plan::float_deconstruct:
      result["plan"] = "float32_original_bits_deconstruct_then_terminal_per_port";
      stage(kFloatDeconstruct, {{"format", "fp32"}});
      break;
    case Plan::a048_planes:
      result["plan"] = "typed_uint64_a048_planes_then_terminal_per_plane";
      stage(kA048Planes, {{"bit_width", 64}});
      break;
    case Plan::split_halves:
      result["plan"] = "split_valid_values_in_halves_then_distinct_terminals";
      result["steps"].push_back({{"kind", "split_segments"},
                                  {"segments", Json::array({"floor(n/2)", "remainder"})}});
      break;
  }
  if (selected.plan != Plan::raw && selected.plan != Plan::codec &&
      selected.plan != Plan::operator_only)
    for (const auto action : selected.terminal_actions) stage(action);
  if (!selected.terminal_actions.empty()) {
    result["terminal_assignment"] =
        selected.terminal_actions.size() == 1
            ? "same_codec_for_each_selected_output"
            : "codec_order_by_output_index_modulo_list_size";
  }
  if (selected.plan == Plan::adjacent_xor)
    result["action_id_selection"] = "6166_for_int64_6165_for_float64";
  result["steps"].push_back({{"kind", "stop"}});
  return result;
}

Json catalog_entry(const Candidate& selected) {
  std::vector<std::string> terminals;
  for (const auto action : selected.terminal_actions)
    if (std::find(terminals.begin(), terminals.end(), terminal_label(action)) == terminals.end())
      terminals.push_back(terminal_label(action));
  return {{"candidate_id", selected.id},
          {"candidate_ordinal", selected.ordinal},
          {"kind", selected.kind},
          {"component_ids", selected.component_ids},
          {"action_ids", selected.action_ids},
          {"terminal_codecs", terminals},
          {"supported_dtypes", selected.dtypes},
          {"requires_nonempty_values", selected.requires_nonempty},
          {"source_width_preserved", true},
          {"allows_harness_raw_fallback", false},
          {"recipe", recipe_json(selected)}};
}

}  // namespace

struct PreparedRegistry::Impl {
  struct Entry {
    std::string id;
    std::uint8_t dtype_mask = 0;
    bool requires_nonempty = false;
  };

  Json catalog;
  std::vector<Entry> entries;

  Impl() {
    Json catalog_entries = Json::array();
    entries.reserve(candidates().size());
    for (const auto& selected : candidates()) {
      Entry entry;
      entry.id = selected.id;
      entry.requires_nonempty = selected.requires_nonempty;
      for (const auto* dtype : selected.dtypes) {
        const std::string_view value(dtype);
        if (value == "int32") entry.dtype_mask |= UINT8_C(1) << 0;
        else if (value == "int64") entry.dtype_mask |= UINT8_C(1) << 1;
        else if (value == "float32") entry.dtype_mask |= UINT8_C(1) << 2;
        else if (value == "float64") entry.dtype_mask |= UINT8_C(1) << 3;
        else fail("candidate has unsupported prepared dtype: " +
                  std::string(value));
      }
      entries.push_back(std::move(entry));
      catalog_entries.push_back(catalog_entry(selected));
    }
    catalog = {{"schema", "operator-recommend135-candidate-catalog-v1"},
               {"policy_registry", kRegistry},
               {"candidates", std::move(catalog_entries)}};
  }
};

PreparedRegistry::PreparedRegistry() {
  static const auto shared = std::make_shared<const Impl>();
  impl_ = shared;
}

const Json& PreparedRegistry::catalog() const noexcept { return impl_->catalog; }

PreparedEligibility PreparedRegistry::prepare(const Block& block) const {
  const auto count = validate_block(block);
  const auto width = dtype_width(block.dtype);
  std::uint8_t dtype_mask = 0;
  if (block.dtype == "int32") dtype_mask = UINT8_C(1) << 0;
  else if (block.dtype == "int64") dtype_mask = UINT8_C(1) << 1;
  else if (block.dtype == "float32") dtype_mask = UINT8_C(1) << 2;
  else if (block.dtype == "float64") dtype_mask = UINT8_C(1) << 3;
  else fail("unsupported source dtype: " + block.dtype);

  PreparedEligibility result;
  result.block = {block.dtype, width, block.row_count, count,
                  block.values.size(), block.missing.size()};
  result.validation_passes = 1;
  result.candidate_ids.reserve(impl_->entries.size());
  for (const auto& entry : impl_->entries) {
    if ((entry.dtype_mask & dtype_mask) == 0) continue;
    if (entry.requires_nonempty && count == 0) continue;
    result.candidate_ids.push_back(entry.id);
  }
  return result;
}

std::vector<std::string> PreparedRegistry::eligible_candidates(
    const Block& block) const {
  return prepare(block).candidate_ids;
}

Json candidate_catalog() {
  Json entries = Json::array();
  for (const auto& selected : candidates()) entries.push_back(catalog_entry(selected));
  return {{"schema", "operator-recommend135-candidate-catalog-v1"},
          {"policy_registry", kRegistry},
          {"candidates", std::move(entries)}};
}

bool candidate_eligible(std::string_view id, const Block& block) {
  const auto* selected = candidate(id);
  if (selected == nullptr) return false;
  try {
    return eligible_impl(*selected, block);
  } catch (const std::exception&) {
    return false;
  }
}

Bytes encode_candidate(std::string_view id, const Block& block) {
  const auto* selected = candidate(id);
  require(selected != nullptr, "unknown recommend135 candidate: " + std::string(id));
  CandidateDispatcher dispatcher(false);
  return encode_with_dispatcher(*selected, block, dispatcher);
}

Block decode_candidate(std::span<const std::uint8_t> frame) {
  CandidateDispatcher dispatcher(false);
  return decode_with_dispatcher(frame, dispatcher);
}

Json inspect_candidate_run(std::string_view id, const Block& block, bool strict) {
  Json result{{"schema", "operator-recommend135-candidate-run-v1"},
              {"candidate_id", id},
              {"strict_requested", strict},
              {"eligible", false},
              {"execution_status", "failed"},
              {"fallback_used", false},
              {"fallback_reason", nullptr},
              {"unified_internal_fallback_audit", false},
              {"allowlist", {{"encode_passed", false},
                              {"decode_passed", false},
                              {"forbidden_hits", Json::array()}}},
              {"frame", nullptr},
              {"roundtrip", nullptr},
              {"error", nullptr}};
  const auto* selected = candidate(id);
  if (selected == nullptr) {
    result["error"] = {{"type", "unknown_candidate"},
                       {"message", "unknown recommend135 candidate"}};
    return result;
  }
  result["candidate_kind"] = selected->kind;
  result["recipe"] = catalog_entry(*selected).at("recipe");
  result["recipe"]["component_ids"] = selected->component_ids;
  result["recipe"]["action_ids"] = selected->action_ids;
  result["recipe"]["terminal_codecs"] =
      catalog_entry(*selected).at("terminal_codecs");
  try {
    const auto count = validate_block(block);
    result["source"] = {{"dtype", block.dtype},
                         {"row_count", block.row_count},
                         {"valid_count", count},
                         {"values_bytes", block.values.size()},
                         {"missing_bytes", block.missing.size()}};
    if (!eligible_impl(*selected, block)) {
      result["execution_status"] = "domain_rejected";
      result["error"] = {{"type", "domain_rejected"},
                          {"message", "candidate dtype/shape rule rejected the block"}};
      return result;
    }
    result["eligible"] = true;
    CandidateDispatcher dispatcher(true);
    const auto frame = encode_with_dispatcher(*selected, block, dispatcher);
    result["allowlist"]["encode_passed"] = true;
    const auto decoded = decode_with_dispatcher(frame, dispatcher);
    result["allowlist"]["decode_passed"] = true;
    const auto& fallback = dispatcher.fallback_audit();
    result["fallback_used"] = fallback.raw_fallback_stage_count != 0;
    result["fallback_reason"] = fallback.reasons.empty() ? Json(nullptr)
                                                          : Json(fallback.reasons);
    result["partial_wrapper_fallback_audit"] =
        {{"wrapped_stage_count", fallback.wrapped_stage_count},
         {"raw_fallback_stage_count", fallback.raw_fallback_stage_count},
         {"reason_counts", fallback.reason_counts}};
    require(!result.at("fallback_used").get<bool>(),
            "strict recommend135 candidate used a harness raw fallback");
    const auto parsed = native_graph_v1::detail::parse_frame(frame, limits());
    const auto digest = native_graph_v1::detail::sha256(frame);
    result["frame"] = {{"bytes", frame.size()},
                        {"ngf_version", 1},
                        {"root_count", parsed.roots.size()},
                        {"node_count", parsed.nodes.size()},
                        {"sha256", hex(digest)}};
    result["roundtrip"] =
        {{"decoded_from_frame_only", true},
         {"process_isolation_claimed", false},
         {"values_bit_exact", decoded.values == block.values},
         {"missing_bit_exact", decoded.missing == block.missing},
         {"dtype_exact", decoded.dtype == block.dtype},
         {"row_count_exact", decoded.row_count == block.row_count}};
    require(decoded == block, "recommend135 audit roundtrip mismatch");
    result["execution_status"] = "strict_ready";
    return result;
  } catch (const operator_harness_v2::StrictInputDomainError& error) {
    result["execution_status"] = "domain_rejected";
    result["error"] = {{"type", error.code()}, {"message", error.what()}};
  } catch (const std::exception& error) {
    result["error"] = {{"type", "execution_error"}, {"message", error.what()}};
  }
  return result;
}

}  // namespace operator_recommend135_v1
