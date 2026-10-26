#include "runtime.hpp"

#include "graph_internal.hpp"
#include "policy_dispatch.hpp"

#include "../../external/compression/lz4/lib/lz4frame.h"
#include "../../external/compression/zstd/lib/zstd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace harness_combo_pilot_v1 {
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
              "combo pilot typed values require a little-endian host");

constexpr std::string_view kRegistry = "harness-combo-pilot-catalog-v1";
constexpr std::string_view kBindingView = "harness_combo_pilot_binding_v1";
constexpr std::string_view kTreeRegistry = "ucc-operator-tree-v1";
constexpr std::string_view kTreeBindingView = "ucc_operator_tree_binding_v1";
constexpr std::string_view kMissingView = "missing_bitmap_lsb1";
constexpr std::uint64_t kMaxStreamBytes = 256ULL << 20;
constexpr std::size_t kMaxTreeJsonBytes = 64U << 10;
constexpr std::size_t kMaxTreeNodes = 48;
constexpr std::size_t kMaxTreeNodeIdBytes = 48;
constexpr std::size_t kMaxTreeJsonNesting = 16;

constexpr std::string_view kZstd3 =
    "harness_combo_pilot_v1.codec.zstd.level3.v1";
constexpr std::string_view kLz4 =
    "harness_combo_pilot_v1.codec.lz4frame.level0.v1";
constexpr std::string_view kColumnize = "columnize";
constexpr std::string_view kXorCarrier = "op.xor.previous_word.cpp_v2";
constexpr std::string_view kXorPorts =
    "op.xor.previous_word.semantic_ports.cpp_v1";
constexpr std::string_view kFor =
    "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1";
constexpr std::string_view kDictionary =
    "op.dictionary.project_words64.cpp_v1";
constexpr std::string_view kBitpack =
    "op.bitpack_int.max_value_width.cpp_v1";
constexpr std::string_view kFloatDeconstruct = "float_deconstruct";
constexpr std::string_view kA048Planes = "harness.a048.byte_planes.v1";

constexpr std::int64_t kActionZstd3 = 235003;
constexpr std::int64_t kActionLz4 = 235010;
constexpr std::int64_t kActionColumnize = 135020;
constexpr std::int64_t kActionA048Planes = 135048;
constexpr std::int64_t kActionForInt64 = 6137;
constexpr std::int64_t kActionBitpackBytes = 6145;
constexpr std::int64_t kActionXorCarrierFloat64 = 6165;
constexpr std::int64_t kActionXorCarrierInt64 = 6166;
constexpr std::int64_t kActionXorPortsFloat64 = 6167;
constexpr std::int64_t kActionXorPortsInt64 = 6168;
constexpr std::int64_t kActionDictionaryFloat64 = 6183;
constexpr std::int64_t kActionDictionaryInt64 = 6184;
constexpr std::int64_t kActionFloat32 = 6219;

constexpr std::uint8_t kInt32 = UINT8_C(1) << 0;
constexpr std::uint8_t kInt64 = UINT8_C(1) << 1;
constexpr std::uint8_t kFloat32 = UINT8_C(1) << 2;
constexpr std::uint8_t kFloat64 = UINT8_C(1) << 3;
constexpr std::uint8_t kAllDtypes = kInt32 | kInt64 | kFloat32 | kFloat64;
constexpr std::uint8_t kWord64Dtypes = kInt64 | kFloat64;

enum class Plan {
  codec,
  columnize,
  xor_carrier,
  for_bitpack,
  a048_planes,
  float_deconstruct,
  xor_semantic,
  xor_semantic_bitpack,
  xor_semantic_a048,
  dictionary,
  dictionary_bitpack,
};

enum class TerminalMode {
  all_zstd3,
  all_lz4,
  alternating_zstd3_lz4,
  alternating_lz4_zstd3,
  first_raw_rest_zstd3,
  first_raw_rest_lz4,
};

struct Candidate {
  std::uint16_t ordinal;
  const char* id;
  const char* kind;
  bool new_combination;
  Plan plan;
  TerminalMode terminals;
  std::uint8_t dtype_mask;
  bool requires_nonempty;
  std::vector<const char*> component_ids;
};

const std::vector<Candidate>& candidates() {
  static const std::vector<Candidate> value{
      {0, "zstd3", "algorithm", false, Plan::codec,
       TerminalMode::all_zstd3, kAllDtypes, false, {"ZSTD_NATIVE_LEVEL3"}},
      {1, "lz4", "algorithm", false, Plan::codec,
       TerminalMode::all_lz4, kAllDtypes, false, {"LZ4FRAME_NATIVE_LEVEL0"}},
      {2, "combo_v1:columnize_zstd3", "operator_tree", false, Plan::columnize,
       TerminalMode::all_zstd3, kAllDtypes, false,
       {"OPENZL_COLUMNIZE", "ZSTD_NATIVE_LEVEL3"}},
      {3, "combo_v1:columnize_lz4", "operator_tree", false, Plan::columnize,
       TerminalMode::all_lz4, kAllDtypes, false,
       {"OPENZL_COLUMNIZE", "LZ4FRAME_NATIVE_LEVEL0"}},
      {4, "combo_v1:xor_zstd3", "operator_tree", false, Plan::xor_carrier,
       TerminalMode::all_zstd3, kWord64Dtypes, false,
       {"A031", "ZSTD_NATIVE_LEVEL3"}},
      {5, "combo_v1:xor_lz4", "operator_tree", false, Plan::xor_carrier,
       TerminalMode::all_lz4, kWord64Dtypes, false,
       {"A031", "LZ4FRAME_NATIVE_LEVEL0"}},
      {6, "combo_v1:for_bitpack_zstd3", "operator_tree", false, Plan::for_bitpack,
       TerminalMode::all_zstd3, kInt64, true,
       {"A013", "A020", "ZSTD_NATIVE_LEVEL3"}},
      {7, "combo_v1:for_bitpack_lz4", "operator_tree", false, Plan::for_bitpack,
       TerminalMode::all_lz4, kInt64, true,
       {"A013", "A020", "LZ4FRAME_NATIVE_LEVEL0"}},
      {8, "combo_v1:a048_planes_zstd3", "operator_tree", false, Plan::a048_planes,
       TerminalMode::all_zstd3, kWord64Dtypes, false,
       {"A048_TYPED_PLANES", "ZSTD_NATIVE_LEVEL3"}},
      {9, "combo_v1:a048_planes_lz4", "operator_tree", false, Plan::a048_planes,
       TerminalMode::all_lz4, kWord64Dtypes, false,
       {"A048_TYPED_PLANES", "LZ4FRAME_NATIVE_LEVEL0"}},
      {10, "combo_v1:float_deconstruct_zstd3", "operator_tree", false,
       Plan::float_deconstruct, TerminalMode::all_zstd3, kFloat32, true,
       {"A103", "ZSTD_NATIVE_LEVEL3"}},
      {11, "combo_v1:float_deconstruct_lz4", "operator_tree", false,
       Plan::float_deconstruct, TerminalMode::all_lz4, kFloat32, true,
       {"A103", "LZ4FRAME_NATIVE_LEVEL0"}},
      {12, "combo_v1:xor_semantic_raw_zstd3", "operator_tree", true,
       Plan::xor_semantic, TerminalMode::first_raw_rest_zstd3, kWord64Dtypes,
       true, {"A031_SEMANTIC_PORTS", "RAW_FIRST_WORD", "ZSTD_NATIVE_LEVEL3"}},
      {13, "combo_v1:xor_semantic_raw_lz4", "operator_tree", true,
       Plan::xor_semantic, TerminalMode::first_raw_rest_lz4, kWord64Dtypes,
       true, {"A031_SEMANTIC_PORTS", "RAW_FIRST_WORD", "LZ4FRAME_NATIVE_LEVEL0"}},
      {14, "combo_v1:xor_semantic_bitpack_zstd3", "operator_tree", true,
       Plan::xor_semantic_bitpack, TerminalMode::all_zstd3, kWord64Dtypes,
       true, {"A031_SEMANTIC_PORTS", "A020", "ZSTD_NATIVE_LEVEL3"}},
      {15, "combo_v1:xor_semantic_bitpack_lz4", "operator_tree", true,
       Plan::xor_semantic_bitpack, TerminalMode::all_lz4, kWord64Dtypes,
       true, {"A031_SEMANTIC_PORTS", "A020", "LZ4FRAME_NATIVE_LEVEL0"}},
      {16, "combo_v1:xor_semantic_a048_zstd3", "operator_tree", true,
       Plan::xor_semantic_a048, TerminalMode::all_zstd3, kWord64Dtypes, true,
       {"A031_SEMANTIC_PORTS", "A048_TYPED_PLANES", "ZSTD_NATIVE_LEVEL3"}},
      {17, "combo_v1:xor_semantic_a048_lz4", "operator_tree", true,
       Plan::xor_semantic_a048, TerminalMode::all_lz4, kWord64Dtypes, true,
       {"A031_SEMANTIC_PORTS", "A048_TYPED_PLANES", "LZ4FRAME_NATIVE_LEVEL0"}},
      {18, "combo_v1:dictionary_bitpack_zstd3", "operator_tree", true,
       Plan::dictionary_bitpack, TerminalMode::all_zstd3, kWord64Dtypes, true,
       {"A042", "A020", "ZSTD_NATIVE_LEVEL3"}},
      {19, "combo_v1:dictionary_bitpack_lz4", "operator_tree", true,
       Plan::dictionary_bitpack, TerminalMode::all_lz4, kWord64Dtypes, true,
       {"A042", "A020", "LZ4FRAME_NATIVE_LEVEL0"}},
      {20, "combo_v1:dictionary_raw_dictionary_zstd3", "operator_tree", true,
       Plan::dictionary, TerminalMode::first_raw_rest_zstd3, kWord64Dtypes,
       true, {"A042", "RAW_DICTIONARY", "ZSTD_NATIVE_LEVEL3"}},
      {21, "combo_v1:dictionary_raw_dictionary_lz4", "operator_tree", true,
       Plan::dictionary, TerminalMode::first_raw_rest_lz4, kWord64Dtypes,
       true, {"A042", "RAW_DICTIONARY", "LZ4FRAME_NATIVE_LEVEL0"}},
      {22, "combo_v1:for_bitpack_raw_base_zstd3", "operator_tree", true,
       Plan::for_bitpack, TerminalMode::first_raw_rest_zstd3, kInt64, true,
       {"A013", "A020", "RAW_BASE", "ZSTD_NATIVE_LEVEL3"}},
      {23, "combo_v1:for_bitpack_raw_base_lz4", "operator_tree", true,
       Plan::for_bitpack, TerminalMode::first_raw_rest_lz4, kInt64, true,
       {"A013", "A020", "RAW_BASE", "LZ4FRAME_NATIVE_LEVEL0"}},
      {24, "combo_v1:columnize_zstd3_lz4", "algorithm_combination", false,
       Plan::columnize, TerminalMode::alternating_zstd3_lz4, kAllDtypes,
       false, {"OPENZL_COLUMNIZE", "ZSTD_NATIVE_LEVEL3", "LZ4FRAME_NATIVE_LEVEL0"}},
      {25, "combo_v1:columnize_lz4_zstd3", "algorithm_combination", false,
       Plan::columnize, TerminalMode::alternating_lz4_zstd3, kAllDtypes,
       false, {"OPENZL_COLUMNIZE", "LZ4FRAME_NATIVE_LEVEL0", "ZSTD_NATIVE_LEVEL3"}},
      {26, "combo_v1:a048_planes_zstd3_lz4", "algorithm_combination", false,
       Plan::a048_planes, TerminalMode::alternating_zstd3_lz4,
       kWord64Dtypes, false,
       {"A048_TYPED_PLANES", "ZSTD_NATIVE_LEVEL3", "LZ4FRAME_NATIVE_LEVEL0"}},
      {27, "combo_v1:a048_planes_lz4_zstd3", "algorithm_combination", false,
       Plan::a048_planes, TerminalMode::alternating_lz4_zstd3,
       kWord64Dtypes, false,
       {"A048_TYPED_PLANES", "LZ4FRAME_NATIVE_LEVEL0", "ZSTD_NATIVE_LEVEL3"}},
      {28, "combo_v1:float_deconstruct_zstd3_lz4", "algorithm_combination", false,
       Plan::float_deconstruct, TerminalMode::alternating_zstd3_lz4,
       kFloat32, true,
       {"A103", "ZSTD_NATIVE_LEVEL3", "LZ4FRAME_NATIVE_LEVEL0"}},
      {29, "combo_v1:float_deconstruct_lz4_zstd3", "algorithm_combination", false,
       Plan::float_deconstruct, TerminalMode::alternating_lz4_zstd3,
       kFloat32, true,
       {"A103", "LZ4FRAME_NATIVE_LEVEL0", "ZSTD_NATIVE_LEVEL3"}},
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
  const auto found = std::find_if(
      all.begin(), all.end(), [&](const Candidate& item) { return id == item.id; });
  return found == all.end() ? nullptr : &*found;
}

const Candidate& candidate(std::uint16_t ordinal) {
  const auto& all = candidates();
  require(ordinal < all.size() && all[ordinal].ordinal == ordinal,
          "unknown combo pilot candidate ordinal");
  return all[ordinal];
}

std::uint32_t dtype_width(std::string_view dtype) {
  if (dtype == "int32" || dtype == "float32") return 4;
  if (dtype == "int64" || dtype == "float64") return 8;
  fail("unsupported source dtype: " + std::string(dtype));
}

std::uint8_t dtype_mask(std::string_view dtype) {
  if (dtype == "int32") return kInt32;
  if (dtype == "int64") return kInt64;
  if (dtype == "float32") return kFloat32;
  if (dtype == "float64") return kFloat64;
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
    default: fail("unknown combo pilot dtype code");
  }
}

std::string hex(std::span<const std::uint8_t> bytes);

void require_fields(const Json& value,
                    std::initializer_list<std::string_view> expected,
                    std::string_view context) {
  require(value.is_object(), std::string(context) + " must be an object");
  require(value.size() == expected.size(),
          std::string(context) + " has missing or unknown fields");
  for (const auto field : expected)
    require(value.contains(std::string(field)),
            std::string(context) + " is missing field: " +
                std::string(field));
}

bool valid_node_id(std::string_view id) {
  if (id.empty() || id.size() > kMaxTreeNodeIdBytes || id == "values")
    return false;
  return std::all_of(id.begin(), id.end(), [](unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
  });
}

Json frozen_parameters(std::string_view op) {
  if (op == "columnize") return {{"mode", "transpose"}};
  if (op == "bitpack")
    return {{"alignment_bytes", 1},
            {"bit_order", "msb"},
            {"minimum_width", 0},
            {"wide_cutoff", 64}};
  if (op == "a048_planes") return {{"bit_width", 64}};
  if (op == "float_deconstruct") return {{"format", "fp32"}};
  return Json::object();
}

const std::set<std::string_view>& semantic_ops() {
  static const std::set<std::string_view> value{
      "zstd3",       "lz4",          "columnize", "xor_carrier",
      "xor_semantic", "for",          "bitpack",   "dictionary",
      "a048_planes", "float_deconstruct",
  };
  return value;
}

std::size_t semantic_output_bound(std::string_view op) {
  if (op == "columnize" || op == "a048_planes") return 8;
  if (op == "xor_semantic" || op == "for" || op == "dictionary" ||
      op == "float_deconstruct")
    return 2;
  return 1;
}

struct SemanticNode {
  std::string id;
  std::string op;
  std::string input;
  Json parameters;
};

struct TreeSpec {
  Kind root_kind = Kind::Numeric;
  std::vector<SemanticNode> nodes;
  Json canonical;
  std::string canonical_bytes;
  native_graph_v1::detail::Digest digest{};
  std::string identifier;
};

std::pair<std::string, std::size_t> parse_reference(std::string_view reference) {
  if (reference == "values") return {"values", 0};
  const auto separator = reference.find(':');
  require(separator != std::string_view::npos && separator != 0 &&
              separator + 1 < reference.size() &&
              reference.find(':', separator + 1) == std::string_view::npos,
          "operator tree input reference must be values or node_id:port");
  const auto id = reference.substr(0, separator);
  require(valid_node_id(id), "operator tree input reference has invalid node ID");
  const auto port_text = reference.substr(separator + 1);
  require(port_text.size() == 1 || port_text.front() != '0',
          "operator tree input reference port must use canonical decimal form");
  std::size_t port = 0;
  for (const auto ch : port_text) {
    require(ch >= '0' && ch <= '9',
            "operator tree input reference port must be decimal");
    require(port <= 1000, "operator tree input reference port is too large");
    port = port * 10 + static_cast<unsigned>(ch - '0');
  }
  require(port < 32, "operator tree input reference port is outside the bound");
  return {std::string(id), port};
}

void validate_json_nesting(std::span<const std::uint8_t> bytes) {
  std::size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (const auto byte : bytes) {
    const auto ch = static_cast<char>(byte);
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == '{' || ch == '[') {
      require(++depth <= kMaxTreeJsonNesting,
              "operator tree JSON exceeds the nesting-depth bound");
    } else if (ch == '}' || ch == ']') {
      require(depth != 0, "operator tree JSON has invalid nesting");
      --depth;
    }
  }
  require(!in_string && depth == 0,
          "operator tree JSON has unterminated strings or containers");
}

TreeSpec parse_tree(const Json& tree) {
  std::string supplied;
  try {
    supplied = tree.dump();
  } catch (const std::exception&) {
    fail("operator tree is not serializable JSON");
  }
  require(supplied.size() <= kMaxTreeJsonBytes,
          "operator tree JSON exceeds the 64 KiB bound");
  require_fields(tree, {"schema", "source", "nodes"}, "operator tree");
  require(tree.at("schema").is_string() &&
              tree.at("schema").get<std::string>() == kTreeRegistry,
          "operator tree schema must be ucc-operator-tree-v1");

  const auto& source = tree.at("source");
  require_fields(source, {"root_kind"}, "operator tree source");
  require(source.at("root_kind").is_string(),
          "operator tree source root_kind must be a string");
  const auto source_kind = source.at("root_kind").get<std::string>();
  require(source_kind == "numeric" || source_kind == "struct",
          "operator tree source root_kind must be numeric or struct");

  const auto& nodes = tree.at("nodes");
  require(nodes.is_array() && !nodes.empty() && nodes.size() <= kMaxTreeNodes,
          "operator tree nodes must contain between 1 and 48 entries");

  TreeSpec result;
  result.root_kind = source_kind == "numeric" ? Kind::Numeric : Kind::Struct;
  std::map<std::string, std::size_t> known_outputs;
  known_outputs.emplace("values", 1);
  std::set<std::string> consumed;
  Json canonical_nodes = Json::array();
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const auto& node = nodes.at(index);
    const auto context = "operator tree node " + std::to_string(index);
    require_fields(node, {"id", "op", "inputs", "parameters"}, context);
    require(node.at("id").is_string(), context + " id must be a string");
    require(node.at("op").is_string(), context + " op must be a string");
    const auto id = node.at("id").get<std::string>();
    const auto op = node.at("op").get<std::string>();
    require(valid_node_id(id), context + " id is invalid");
    require(!known_outputs.contains(id), "operator tree node IDs must be unique");
    require(semantic_ops().contains(op),
            "operator tree op is outside the semantic allowlist: " + op);

    const auto& inputs = node.at("inputs");
    require(inputs.is_array() && inputs.size() == 1 && inputs.at(0).is_string(),
            context + " must have exactly one string input reference");
    const auto input = inputs.at(0).get<std::string>();
    const auto [input_id, input_port] = parse_reference(input);
    const auto referenced = known_outputs.find(input_id);
    require(referenced != known_outputs.end(),
            "operator tree input must reference values or an earlier node");
    require(input_port < referenced->second,
            "operator tree input selects an unavailable output port");
    require(consumed.insert(input).second,
            "operator tree input stream is consumed more than once: " + input);

    const auto expected_parameters = frozen_parameters(op);
    require(node.at("parameters").is_object() &&
                node.at("parameters") == expected_parameters,
            context + " parameters differ from the frozen semantic contract");
    result.nodes.push_back({id, op, input, expected_parameters});
    known_outputs.emplace(id, semantic_output_bound(op));
    canonical_nodes.push_back({{"id", id},
                               {"inputs", Json::array({input})},
                               {"op", op},
                               {"parameters", expected_parameters}});
  }
  result.canonical = {{"nodes", std::move(canonical_nodes)},
                      {"schema", kTreeRegistry},
                      {"source", {{"root_kind", source_kind}}}};
  result.canonical_bytes = result.canonical.dump();
  require(result.canonical_bytes.size() <= kMaxTreeJsonBytes,
          "canonical operator tree JSON exceeds the 64 KiB bound");
  result.digest = native_graph_v1::detail::sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(result.canonical_bytes.data()),
      result.canonical_bytes.size()));
  result.identifier = "tree_v1:" + hex(result.digest);
  return result;
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
  std::uint64_t missing_count = 0;
  for (const auto byte : block.missing) missing_count += std::popcount(byte);
  require(missing_count <= block.row_count, "missing count exceeds row_count");
  return block.row_count - missing_count;
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

bool eligible_after_validation(const Candidate& selected, const Block& block,
                               std::uint64_t count) {
  return (selected.dtype_mask & dtype_mask(block.dtype)) != 0 &&
         (!selected.requires_nonempty || count != 0);
}

void put(Bytes& output, std::uint64_t value, unsigned width) {
  for (unsigned index = 0; index < width; ++index)
    output.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
}

std::uint64_t get(std::span<const std::uint8_t> input, std::size_t& position,
                  unsigned width) {
  require(position <= input.size() && width <= input.size() - position,
          "truncated combo pilot metadata");
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
  std::uint64_t missing_bytes = 0;
  native_graph_v1::detail::Digest missing_digest{};
};

constexpr std::size_t kBindingBytes = 72;

Bytes binding_bytes(const Candidate& selected, const Block& block,
                    std::uint64_t count) {
  Bytes output{'H', 'C', 'B', '1'};
  put(output, 1, 1);
  put(output, dtype_code(block.dtype), 1);
  put(output, selected.ordinal, 2);
  put(output, block.row_count, 8);
  put(output, count, 8);
  put(output, block.values.size(), 8);
  put(output, block.missing.size(), 8);
  const auto digest = native_graph_v1::detail::sha256(block.missing);
  output.insert(output.end(), digest.begin(), digest.end());
  require(output.size() == kBindingBytes, "internal combo binding size mismatch");
  return output;
}

Binding parse_binding(std::span<const std::uint8_t> bytes) {
  require(bytes.size() == kBindingBytes &&
              std::equal(bytes.begin(), bytes.begin() + 4, "HCB1"),
          "invalid combo pilot binding magic or size");
  std::size_t position = 4;
  require(get(bytes, position, 1) == 1,
          "unsupported combo pilot binding version");
  Binding result;
  result.dtype = dtype_from_code(static_cast<std::uint8_t>(get(bytes, position, 1)));
  result.selected = &candidate(static_cast<std::uint16_t>(get(bytes, position, 2)));
  result.row_count = get(bytes, position, 8);
  result.value_count = get(bytes, position, 8);
  result.value_bytes = get(bytes, position, 8);
  result.missing_bytes = get(bytes, position, 8);
  std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(position),
              result.missing_digest.size(), result.missing_digest.begin());
  position += result.missing_digest.size();
  require(position == bytes.size(), "trailing combo pilot binding bytes");
  require(checked_multiply(result.value_count, dtype_width(result.dtype),
                           "bound source byte count overflow") == result.value_bytes,
          "combo binding value count differs from dtype width");
  require(result.missing_bytes ==
              result.row_count / 8 + (result.row_count % 8 != 0),
          "combo binding bitmap length differs from row_count");
  return result;
}

std::uint8_t root_kind_code(Kind kind) {
  if (kind == Kind::Numeric) return 1;
  if (kind == Kind::Struct) return 2;
  fail("explicit operator tree root kind must be numeric or struct");
}

Kind root_kind_from_code(std::uint8_t code) {
  if (code == 1) return Kind::Numeric;
  if (code == 2) return Kind::Struct;
  fail("unknown explicit operator tree root kind code");
}

struct TreeBinding {
  TreeSpec tree;
  std::string dtype;
  std::uint64_t row_count = 0;
  std::uint64_t value_count = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t missing_bytes = 0;
  native_graph_v1::detail::Digest missing_digest{};
};

constexpr std::size_t kTreeBindingFixedBytes = 108;

Bytes tree_binding_bytes(const TreeSpec& tree, const Block& block,
                         std::uint64_t count) {
  require(tree.canonical_bytes.size() <= UINT32_MAX,
          "canonical operator tree is too large for its binding");
  Bytes output{'H', 'C', 'B', '2'};
  put(output, 1, 1);
  put(output, dtype_code(block.dtype), 1);
  put(output, root_kind_code(tree.root_kind), 1);
  put(output, 0, 1);
  put(output, block.row_count, 8);
  put(output, count, 8);
  put(output, block.values.size(), 8);
  put(output, block.missing.size(), 8);
  const auto missing_digest = native_graph_v1::detail::sha256(block.missing);
  output.insert(output.end(), missing_digest.begin(), missing_digest.end());
  output.insert(output.end(), tree.digest.begin(), tree.digest.end());
  put(output, tree.canonical_bytes.size(), 4);
  require(output.size() == kTreeBindingFixedBytes,
          "internal explicit-tree binding size mismatch");
  output.insert(output.end(), tree.canonical_bytes.begin(),
                tree.canonical_bytes.end());
  return output;
}

TreeBinding parse_tree_binding(std::span<const std::uint8_t> bytes) {
  require(bytes.size() >= kTreeBindingFixedBytes &&
              std::equal(bytes.begin(), bytes.begin() + 4, "HCB2"),
          "invalid explicit operator tree binding magic or size");
  std::size_t position = 4;
  require(get(bytes, position, 1) == 1,
          "unsupported explicit operator tree binding version");
  TreeBinding result;
  result.dtype = dtype_from_code(static_cast<std::uint8_t>(get(bytes, position, 1)));
  const auto bound_root_kind =
      root_kind_from_code(static_cast<std::uint8_t>(get(bytes, position, 1)));
  require(get(bytes, position, 1) == 0,
          "explicit operator tree binding reserved byte is nonzero");
  result.row_count = get(bytes, position, 8);
  result.value_count = get(bytes, position, 8);
  result.value_bytes = get(bytes, position, 8);
  result.missing_bytes = get(bytes, position, 8);
  std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(position),
              result.missing_digest.size(), result.missing_digest.begin());
  position += result.missing_digest.size();
  native_graph_v1::detail::Digest stored_tree_digest{};
  std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(position),
              stored_tree_digest.size(), stored_tree_digest.begin());
  position += stored_tree_digest.size();
  const auto tree_size = get(bytes, position, 4);
  require(tree_size <= kMaxTreeJsonBytes && position <= bytes.size() &&
              tree_size == bytes.size() - position,
          "explicit operator tree binding has an invalid tree length");
  validate_json_nesting(bytes.subspan(position, tree_size));
  Json parsed;
  try {
    parsed = Json::parse(bytes.begin() + static_cast<std::ptrdiff_t>(position),
                         bytes.end());
  } catch (const std::exception&) {
    fail("explicit operator tree binding contains invalid JSON");
  }
  result.tree = parse_tree(parsed);
  require(result.tree.root_kind == bound_root_kind,
          "explicit operator tree root kind differs from its binding");
  require(result.tree.digest == stored_tree_digest,
          "explicit operator tree digest differs from its binding");
  const std::string stored_json(
      reinterpret_cast<const char*>(bytes.data() + position), tree_size);
  require(stored_json == result.tree.canonical_bytes,
          "explicit operator tree binding is not canonical");
  require(checked_multiply(result.value_count, dtype_width(result.dtype),
                           "bound explicit-tree source byte count overflow") ==
              result.value_bytes,
          "explicit operator tree value count differs from dtype width");
  require(result.missing_bytes ==
              result.row_count / 8 + (result.row_count % 8 != 0),
          "explicit operator tree bitmap length differs from row_count");
  return result;
}

bool is_terminal(std::string_view action) {
  return action == kZstd3 || action == kLz4;
}

std::uint8_t codec_tag(std::string_view action) {
  if (action == kZstd3) return 1;
  if (action == kLz4) return 2;
  fail("unknown combo terminal codec action");
}

std::int64_t terminal_action_id(std::string_view action) {
  if (action == kZstd3) return kActionZstd3;
  if (action == kLz4) return kActionLz4;
  fail("unknown combo terminal action ID");
}

Bytes terminal_metadata(std::string_view action, const Stream& input) {
  require(input.width <= UINT16_MAX, "terminal input width is too large");
  Bytes result{'H', 'C', 'T', '1'};
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

TerminalMetadata parse_terminal_metadata(
    std::string_view action, std::span<const std::uint8_t> metadata) {
  require(metadata.size() == 16 &&
              std::equal(metadata.begin(), metadata.begin() + 4, "HCT1"),
          "invalid combo terminal metadata");
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
  if (result.kind == Kind::Serial) {
    require(result.width == 1, "serial terminal width must be one");
  } else if (result.kind == Kind::Numeric) {
    require(result.width == 1 || result.width == 2 || result.width == 4 ||
                result.width == 8,
            "invalid numeric terminal width");
  } else {
    require(result.kind == Kind::Struct, "invalid terminal source kind");
  }
  return result;
}

Bytes zstd_compress(std::span<const std::uint8_t> input) {
  const auto bound = ZSTD_compressBound(input.size());
  require(!ZSTD_isError(bound) && bound <= kMaxStreamBytes,
          "Zstd compressed bound exceeds the block limit");
  Bytes output(bound);
  const std::uint8_t empty = 0;
  const auto size = ZSTD_compress(output.data(), output.size(),
                                  input.empty() ? &empty : input.data(),
                                  input.size(), 3);
  require(!ZSTD_isError(size),
          std::string("Zstd encode failed: ") + ZSTD_getErrorName(size));
  output.resize(size);
  return output;
}

Bytes zstd_decompress(std::span<const std::uint8_t> input,
                      std::uint64_t expected) {
  require(expected <= kMaxStreamBytes, "Zstd decoded size exceeds block limit");
  const auto frame_size = ZSTD_findFrameCompressedSize(input.data(), input.size());
  require(!ZSTD_isError(frame_size) && frame_size == input.size(),
          "invalid, concatenated, or trailing Zstd frame");
  const auto content = ZSTD_getFrameContentSize(input.data(), input.size());
  require(content != ZSTD_CONTENTSIZE_ERROR && content != ZSTD_CONTENTSIZE_UNKNOWN &&
              content == expected,
          "Zstd content size differs from inverse metadata");
  Bytes output(static_cast<std::size_t>(expected));
  std::uint8_t empty = 0;
  const auto size = ZSTD_decompress(output.empty() ? &empty : output.data(),
                                    output.size(), input.data(), input.size());
  require(!ZSTD_isError(size) && size == expected,
          ZSTD_isError(size)
              ? std::string("Zstd decode failed: ") + ZSTD_getErrorName(size)
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
          "LZ4 frame bound exceeds block limit");
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

Bytes lz4_decompress(std::span<const std::uint8_t> input,
                     std::uint64_t expected) {
  require(expected <= kMaxStreamBytes, "LZ4 decoded size exceeds block limit");
  LZ4F_dctx* context = nullptr;
  const auto created = LZ4F_createDecompressionContext(&context, LZ4F_VERSION);
  require(!LZ4F_isError(created) && context != nullptr,
          "LZ4 decompression context creation failed");
  struct Release {
    LZ4F_dctx* value;
    ~Release() {
      if (value != nullptr) LZ4F_freeDecompressionContext(value);
    }
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
    require(output_position <= expected, "LZ4 decoded more bytes than declared");
    require(source_position != old_source || output_position != old_output || hint == 0,
            "LZ4 decoder made no progress");
  }
  require(source_position == input.size() && output_position == expected,
          "LZ4 frame has trailing bytes or a decoded size mismatch");
  storage.resize(output_position);
  return storage;
}

Bytes terminal_forward(std::string_view action,
                       std::span<const std::uint8_t> input) {
  if (action == kZstd3) return zstd_compress(input);
  if (action == kLz4) return lz4_compress(input);
  fail("terminal action is not allowlisted");
}

Bytes terminal_inverse(std::string_view action,
                       std::span<const std::uint8_t> input,
                       std::uint64_t expected) {
  if (action == kZstd3) return zstd_decompress(input, expected);
  if (action == kLz4) return lz4_decompress(input, expected);
  fail("terminal action is not allowlisted");
}

const std::set<std::string_view>& allowed_actions() {
  static const std::set<std::string_view> value{
      kZstd3,       kLz4,       kColumnize, kXorCarrier, kXorPorts,
      kFor,         kDictionary, kBitpack,   kFloatDeconstruct,
      kA048Planes,
  };
  return value;
}

class ComboDispatcher final : public native_graph_v1::StageDispatcher {
 public:
  explicit ComboDispatcher(bool audit_fallbacks) : inner_(audit_fallbacks) {}

  std::string identity() const override {
    return "harness_combo_pilot_v1.dispatcher.terminal-codecs-v1+" +
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

  std::vector<Stream> inverse(
      const StageRequest& request, const std::vector<Stream>& outputs,
      std::span<const std::uint8_t> metadata) const override {
    enforce(request);
    if (!is_terminal(request.action))
      return inner_.inverse(request, outputs, metadata);
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
            "stage action is outside the combo pilot allowlist: " + request.action);
    require(!operator_harness_v2::excluded_action_reason(request.action).has_value(),
            "stage action belongs to a disabled family");
  }

  operator_harness_v2::PolicyDispatcher inner_;
};

native_graph_v1::Limits limits() {
  native_graph_v1::Limits result;
  result.max_nodes = 64;
  result.max_active_streams = 32;
  result.max_ordinary_depth = 8;
  result.max_stream_bytes = kMaxStreamBytes;
  result.max_working_bytes = 512ULL << 20;
  result.max_frame_bytes = 1ULL << 30;
  result.max_metadata_bytes = 4ULL << 20;
  return result;
}

Kind root_kind(const Candidate& selected) {
  return selected.plan == Plan::columnize ? Kind::Struct : Kind::Numeric;
}

std::int64_t typed_action_id(std::string_view dtype, std::int64_t int_id,
                             std::int64_t float_id) {
  require(dtype == "int64" || dtype == "float64",
          "typed 64-bit stage requires int64 or float64");
  return dtype == "int64" ? int_id : float_id;
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

StageRequest a048_request() {
  return {std::string(kA048Planes), "uint64", {{"bit_width", "64"}}};
}

std::optional<std::string_view> terminal_for(TerminalMode mode,
                                             std::size_t port) {
  switch (mode) {
    case TerminalMode::all_zstd3: return kZstd3;
    case TerminalMode::all_lz4: return kLz4;
    case TerminalMode::alternating_zstd3_lz4:
      return port % 2 == 0 ? kZstd3 : kLz4;
    case TerminalMode::alternating_lz4_zstd3:
      return port % 2 == 0 ? kLz4 : kZstd3;
    case TerminalMode::first_raw_rest_zstd3:
      return port == 0 ? std::nullopt
                       : std::optional<std::string_view>(kZstd3);
    case TerminalMode::first_raw_rest_lz4:
      return port == 0 ? std::nullopt : std::optional<std::string_view>(kLz4);
  }
  fail("unknown combo terminal mode");
}

struct PlannedNode {
  Step step;
  std::vector<StreamId> outputs;
};

struct CompiledPlan {
  std::vector<PlannedNode> nodes;
};

class PlanBuilder {
 public:
  std::vector<StreamId> stage(std::vector<StreamId> inputs,
                              StageRequest request,
                              std::int64_t policy_action_id,
                              std::size_t output_count) {
    Step step;
    step.kind = StepKind::Stage;
    step.inputs = std::move(inputs);
    step.stage = std::move(request);
    step.policy_action_id = policy_action_id;
    std::vector<StreamId> outputs;
    outputs.reserve(output_count);
    for (std::size_t index = 0; index < output_count; ++index)
      outputs.push_back(next_++);
    nodes_.push_back({std::move(step), outputs});
    return outputs;
  }

  void terminals(const Candidate& selected,
                 const std::vector<StreamId>& inputs) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      const auto action = terminal_for(selected.terminals, index);
      if (!action.has_value()) continue;
      (void)stage({inputs[index]}, terminal_request(*action),
                  terminal_action_id(*action), 1);
    }
  }

  CompiledPlan finish() { return {std::move(nodes_)}; }

 private:
  StreamId next_ = 3;
  std::vector<PlannedNode> nodes_;
};

struct CompiledSemanticOp {
  StageRequest request;
  std::int64_t policy_action_id = 0;
  std::size_t output_count = 0;
  std::uint8_t dtype_mask = 0;
  bool requires_nonempty = false;
};

CompiledSemanticOp compile_semantic_op(const SemanticNode& node,
                                       std::string_view dtype,
                                       std::uint32_t width) {
  if (node.op == "zstd3")
    return {terminal_request(kZstd3), kActionZstd3, 1, kAllDtypes, false};
  if (node.op == "lz4")
    return {terminal_request(kLz4), kActionLz4, 1, kAllDtypes, false};
  if (node.op == "columnize")
    return {columnize_request(width), kActionColumnize, width, kAllDtypes, false};
  if (node.op == "xor_carrier")
    return {{std::string(kXorCarrier), std::string(dtype), {}},
            typed_action_id(dtype, kActionXorCarrierInt64,
                            kActionXorCarrierFloat64),
            1, kWord64Dtypes, false};
  if (node.op == "xor_semantic")
    return {{std::string(kXorPorts), std::string(dtype), {}},
            typed_action_id(dtype, kActionXorPortsInt64,
                            kActionXorPortsFloat64),
            2, kWord64Dtypes, true};
  if (node.op == "for")
    return {{std::string(kFor), "int64", {}}, kActionForInt64, 2, kInt64,
            true};
  if (node.op == "bitpack")
    return {bitpack_request(), kActionBitpackBytes, 1, kWord64Dtypes, false};
  if (node.op == "dictionary")
    return {{std::string(kDictionary), std::string(dtype), {}},
            typed_action_id(dtype, kActionDictionaryInt64,
                            kActionDictionaryFloat64),
            2, kWord64Dtypes, true};
  if (node.op == "a048_planes")
    return {a048_request(), kActionA048Planes, 8, kWord64Dtypes, false};
  if (node.op == "float_deconstruct")
    return {{std::string(kFloatDeconstruct), "float32", {{"format", "fp32"}}},
            kActionFloat32, 2, kFloat32, true};
  fail("operator tree op is outside the semantic allowlist: " + node.op);
}

CompiledPlan compile_tree_plan(const TreeSpec& tree, std::string_view dtype,
                               std::uint64_t value_count) {
  PlanBuilder builder;
  std::map<std::string, std::vector<StreamId>> outputs;
  outputs.emplace("values", std::vector<StreamId>{0});
  const auto mask = dtype_mask(dtype);
  const auto width = dtype_width(dtype);
  for (const auto& node : tree.nodes) {
    const auto [input_id, input_port] = parse_reference(node.input);
    const auto found = outputs.find(input_id);
    require(found != outputs.end() && input_port < found->second.size(),
            "operator tree input reference selects an unavailable output port: " +
                node.input);
    require(node.op != "bitpack" || input_id != "values",
            "operator tree bitpack requires an unsigned 64-bit transform port");
    const auto compiled = compile_semantic_op(node, dtype, width);
    require((compiled.dtype_mask & mask) != 0,
            "operator tree op is not eligible for source dtype: " + node.op);
    require(!compiled.requires_nonempty || value_count != 0,
            "operator tree op requires at least one valid value: " + node.op);
    auto node_outputs = builder.stage({found->second[input_port]},
                                      compiled.request,
                                      compiled.policy_action_id,
                                      compiled.output_count);
    require(outputs.emplace(node.id, std::move(node_outputs)).second,
            "operator tree contains a duplicate compiled node ID");
  }
  return builder.finish();
}

CompiledPlan compile_plan(const Candidate& selected, std::string_view dtype,
                          std::uint32_t width) {
  PlanBuilder builder;
  std::vector<StreamId> leaves;
  switch (selected.plan) {
    case Plan::codec:
      leaves = {0};
      break;
    case Plan::columnize:
      leaves = builder.stage({0}, columnize_request(width), kActionColumnize,
                             width);
      break;
    case Plan::xor_carrier:
      leaves = builder.stage(
          {0}, {std::string(kXorCarrier), std::string(dtype), {}},
          typed_action_id(dtype, kActionXorCarrierInt64,
                          kActionXorCarrierFloat64),
          1);
      break;
    case Plan::for_bitpack: {
      const auto ports = builder.stage(
          {0}, {std::string(kFor), "int64", {}}, kActionForInt64, 2);
      for (const auto port : ports) {
        const auto packed = builder.stage({port}, bitpack_request(),
                                          kActionBitpackBytes, 1);
        leaves.push_back(packed.front());
      }
      break;
    }
    case Plan::a048_planes:
      leaves = builder.stage({0}, a048_request(), kActionA048Planes, 8);
      break;
    case Plan::float_deconstruct:
      leaves = builder.stage(
          {0}, {std::string(kFloatDeconstruct), "float32", {{"format", "fp32"}}},
          kActionFloat32, 2);
      break;
    case Plan::xor_semantic:
      leaves = builder.stage(
          {0}, {std::string(kXorPorts), std::string(dtype), {}},
          typed_action_id(dtype, kActionXorPortsInt64, kActionXorPortsFloat64),
          2);
      break;
    case Plan::xor_semantic_bitpack: {
      const auto ports = builder.stage(
          {0}, {std::string(kXorPorts), std::string(dtype), {}},
          typed_action_id(dtype, kActionXorPortsInt64, kActionXorPortsFloat64),
          2);
      for (const auto port : ports) {
        const auto packed = builder.stage({port}, bitpack_request(),
                                          kActionBitpackBytes, 1);
        leaves.push_back(packed.front());
      }
      break;
    }
    case Plan::xor_semantic_a048: {
      const auto ports = builder.stage(
          {0}, {std::string(kXorPorts), std::string(dtype), {}},
          typed_action_id(dtype, kActionXorPortsInt64, kActionXorPortsFloat64),
          2);
      for (const auto port : ports) {
        auto planes = builder.stage({port}, a048_request(), kActionA048Planes, 8);
        leaves.insert(leaves.end(), planes.begin(), planes.end());
      }
      break;
    }
    case Plan::dictionary:
      leaves = builder.stage(
          {0}, {std::string(kDictionary), std::string(dtype), {}},
          typed_action_id(dtype, kActionDictionaryInt64,
                          kActionDictionaryFloat64),
          2);
      break;
    case Plan::dictionary_bitpack: {
      const auto ports = builder.stage(
          {0}, {std::string(kDictionary), std::string(dtype), {}},
          typed_action_id(dtype, kActionDictionaryInt64,
                          kActionDictionaryFloat64),
          2);
      for (const auto port : ports) {
        const auto packed = builder.stage({port}, bitpack_request(),
                                          kActionBitpackBytes, 1);
        leaves.push_back(packed.front());
      }
      break;
    }
  }
  builder.terminals(selected, leaves);
  return builder.finish();
}

std::vector<RootInput> roots(const Candidate& selected, const Block& block,
                             std::uint64_t count) {
  return {{{root_kind(selected), dtype_width(block.dtype), block.values}, true,
           block.dtype},
          {{Kind::Serial, 1, block.missing}, false, std::string(kMissingView)},
          {{Kind::Serial, 1, binding_bytes(selected, block, count)}, false,
           std::string(kBindingView)}};
}

Bytes execute_encode(const Candidate& selected, const Block& block,
                     std::uint64_t count, ComboDispatcher& dispatcher) {
  native_graph_v1::GraphSession graph(roots(selected, block, count), dispatcher,
                                      limits(), std::string(kRegistry));
  const auto plan = compile_plan(selected, block.dtype, dtype_width(block.dtype));
  for (const auto& node : plan.nodes) {
    const auto actual = graph.apply(node.step);
    require(actual == node.outputs,
            "graph assigned non-canonical output IDs for compiled combo recipe");
  }
  return graph.stop();
}

std::vector<RootInput> tree_roots(const TreeSpec& tree, const Block& block,
                                  std::uint64_t count) {
  return {{{tree.root_kind, dtype_width(block.dtype), block.values}, true,
           block.dtype},
          {{Kind::Serial, 1, block.missing}, false, std::string(kMissingView)},
          {{Kind::Serial, 1, tree_binding_bytes(tree, block, count)}, false,
           std::string(kTreeBindingView)}};
}

Bytes execute_encode_tree(const TreeSpec& tree, const Block& block,
                          std::uint64_t count, ComboDispatcher& dispatcher) {
  native_graph_v1::GraphSession graph(tree_roots(tree, block, count), dispatcher,
                                      limits(), std::string(kTreeRegistry));
  const auto plan = compile_tree_plan(tree, block.dtype, count);
  for (const auto& node : plan.nodes) {
    const auto actual = graph.apply(node.step);
    require(actual == node.outputs,
            "graph assigned non-canonical output IDs for explicit operator tree");
  }
  return graph.stop();
}

void validate_recipe(const Candidate& selected, const Binding& binding,
                     const native_graph_v1::detail::Frame& frame) {
  const auto plan =
      compile_plan(selected, binding.dtype, dtype_width(binding.dtype));
  require(frame.nodes.size() == plan.nodes.size(),
          "frame node count differs from the bound combo recipe");
  for (std::size_t index = 0; index < plan.nodes.size(); ++index) {
    const auto& actual = frame.nodes[index];
    const auto& expected = plan.nodes[index];
    require(actual.step.kind == expected.step.kind &&
                actual.step.inputs == expected.step.inputs &&
                actual.step.stage == expected.step.stage &&
                actual.step.segment_sizes == expected.step.segment_sizes &&
                actual.step.policy_action_id == expected.step.policy_action_id,
            "frame node differs from the bound combo recipe");
    require(actual.outputs == expected.outputs,
            "frame output IDs differ from the bound combo recipe");
  }
}

void validate_tree_recipe(const TreeBinding& binding,
                          const native_graph_v1::detail::Frame& frame) {
  const auto plan =
      compile_tree_plan(binding.tree, binding.dtype, binding.value_count);
  require(frame.nodes.size() == plan.nodes.size(),
          "frame node count differs from the bound explicit operator tree");
  for (std::size_t index = 0; index < plan.nodes.size(); ++index) {
    const auto& actual = frame.nodes[index];
    const auto& expected = plan.nodes[index];
    require(actual.step.kind == expected.step.kind &&
                actual.step.inputs == expected.step.inputs &&
                actual.step.stage == expected.step.stage &&
                actual.step.segment_sizes == expected.step.segment_sizes &&
                actual.step.policy_action_id == expected.step.policy_action_id,
            "frame node differs from the bound explicit operator tree recipe");
    require(actual.outputs == expected.outputs,
            "frame output IDs differ from the bound explicit operator tree recipe");
  }
}

void preflight_frame(const native_graph_v1::detail::Frame& frame,
                     const Binding& binding,
                     const ComboDispatcher& dispatcher) {
  require(frame.dispatcher == dispatcher.identity(),
          "combo pilot dispatcher identity mismatch");
  require(frame.policy_registry == kRegistry,
          "combo pilot policy registry mismatch");
  require(frame.roots.size() == 3,
          "combo pilot frame must have exactly three roots");
  const auto& selected = *binding.selected;
  const auto width = dtype_width(binding.dtype);
  require(frame.roots[0].spec.kind == root_kind(selected) &&
              frame.roots[0].spec.width == width &&
              frame.roots[0].spec.bytes == binding.value_bytes &&
              frame.roots[0].numeric &&
              frame.roots[0].logical_view == binding.dtype,
          "combo pilot value-root binding mismatch");
  require(frame.roots[1].spec.kind == Kind::Serial &&
              frame.roots[1].spec.width == 1 &&
              frame.roots[1].spec.bytes == binding.missing_bytes &&
              !frame.roots[1].numeric &&
              frame.roots[1].logical_view == kMissingView,
          "combo pilot missing-root binding mismatch");
  require(frame.roots[2].spec.kind == Kind::Serial &&
              frame.roots[2].spec.width == 1 &&
              frame.roots[2].spec.bytes == kBindingBytes &&
              !frame.roots[2].numeric &&
              frame.roots[2].logical_view == kBindingView,
          "combo pilot schema-root binding mismatch");
  require(frame.numeric_bytes == binding.value_bytes,
          "combo pilot numeric denominator mismatch");
  require(frame.leaves.contains(1) && frame.leaves.contains(2),
          "combo pilot auxiliary roots must remain terminal leaves");
  require(frame.leaves.at(1).kind == Kind::Serial &&
              frame.leaves.at(1).width == 1 &&
              frame.leaves.at(1).data.size() == binding.missing_bytes &&
              native_graph_v1::detail::sha256(frame.leaves.at(1).data) ==
                  binding.missing_digest,
          "combo pilot missing bitmap differs from its binding");
  require(frame.leaves.at(2).kind == Kind::Serial &&
              frame.leaves.at(2).width == 1 &&
              frame.leaves.at(2).data.size() == kBindingBytes,
          "combo pilot binding leaf is malformed");
  for (const auto& node : frame.nodes) {
    require(node.step.policy_action_id < 6225 ||
                node.step.policy_action_id > 6229,
            "A106-A110 action ID is disabled by the combo pilot");
    require(node.step.kind == StepKind::Stage,
            "combo pilot frame contains a non-stage node");
    require(allowed_actions().contains(node.step.stage.action),
            "frame stage is outside the combo pilot allowlist");
    require(!operator_harness_v2::excluded_action_reason(node.step.stage.action),
            "frame stage belongs to a disabled family");
    require(std::find(node.step.inputs.begin(), node.step.inputs.end(), 1) ==
                    node.step.inputs.end() &&
                std::find(node.step.inputs.begin(), node.step.inputs.end(), 2) ==
                    node.step.inputs.end(),
            "combo recipe consumed an auxiliary root");
  }
  validate_recipe(selected, binding, frame);
}

void preflight_tree_frame(const native_graph_v1::detail::Frame& frame,
                          const TreeBinding& binding,
                          const ComboDispatcher& dispatcher) {
  require(frame.dispatcher == dispatcher.identity(),
          "explicit operator tree dispatcher identity mismatch");
  require(frame.policy_registry == kTreeRegistry,
          "explicit operator tree policy registry mismatch");
  require(frame.roots.size() == 3,
          "explicit operator tree frame must have exactly three roots");
  const auto width = dtype_width(binding.dtype);
  require(frame.roots[0].spec.kind == binding.tree.root_kind &&
              frame.roots[0].spec.width == width &&
              frame.roots[0].spec.bytes == binding.value_bytes &&
              frame.roots[0].numeric &&
              frame.roots[0].logical_view == binding.dtype,
          "explicit operator tree value-root binding mismatch");
  require(frame.roots[1].spec.kind == Kind::Serial &&
              frame.roots[1].spec.width == 1 &&
              frame.roots[1].spec.bytes == binding.missing_bytes &&
              !frame.roots[1].numeric &&
              frame.roots[1].logical_view == kMissingView,
          "explicit operator tree missing-root binding mismatch");
  const auto expected_binding_bytes =
      kTreeBindingFixedBytes + binding.tree.canonical_bytes.size();
  require(frame.roots[2].spec.kind == Kind::Serial &&
              frame.roots[2].spec.width == 1 &&
              frame.roots[2].spec.bytes == expected_binding_bytes &&
              !frame.roots[2].numeric &&
              frame.roots[2].logical_view == kTreeBindingView,
          "explicit operator tree schema-root binding mismatch");
  require(frame.numeric_bytes == binding.value_bytes,
          "explicit operator tree numeric denominator mismatch");
  require(frame.leaves.contains(1) && frame.leaves.contains(2),
          "explicit operator tree auxiliary roots must remain terminal leaves");
  require(frame.leaves.at(1).kind == Kind::Serial &&
              frame.leaves.at(1).width == 1 &&
              frame.leaves.at(1).data.size() == binding.missing_bytes &&
              native_graph_v1::detail::sha256(frame.leaves.at(1).data) ==
                  binding.missing_digest,
          "explicit operator tree missing bitmap differs from its binding");
  require(frame.leaves.at(2).kind == Kind::Serial &&
              frame.leaves.at(2).width == 1 &&
              frame.leaves.at(2).data.size() == expected_binding_bytes,
          "explicit operator tree binding leaf is malformed");
  for (const auto& node : frame.nodes) {
    require(node.step.policy_action_id < 6225 ||
                node.step.policy_action_id > 6229,
            "A106-A110 action ID is disabled by the explicit operator tree");
    require(node.step.kind == StepKind::Stage,
            "explicit operator tree frame contains a non-stage node");
    require(allowed_actions().contains(node.step.stage.action),
            "explicit operator tree frame stage is outside the action allowlist");
    require(!operator_harness_v2::excluded_action_reason(node.step.stage.action),
            "explicit operator tree frame stage belongs to a disabled family");
    require(std::find(node.step.inputs.begin(), node.step.inputs.end(), 1) ==
                    node.step.inputs.end() &&
                std::find(node.step.inputs.begin(), node.step.inputs.end(), 2) ==
                    node.step.inputs.end(),
            "explicit operator tree recipe consumed an auxiliary root");
  }
  validate_tree_recipe(binding, frame);
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

const char* plan_name(Plan plan) {
  switch (plan) {
    case Plan::codec: return "whole_values_terminal_codec";
    case Plan::columnize: return "columnize_then_terminal_per_plane";
    case Plan::xor_carrier: return "adjacent_xor_carrier_then_terminal";
    case Plan::for_bitpack: return "for_then_bitpack_each_port_then_terminal";
    case Plan::a048_planes: return "a048_typed_planes_then_terminal_per_plane";
    case Plan::float_deconstruct:
      return "float_deconstruct_then_terminal_per_port";
    case Plan::xor_semantic:
      return "xor_semantic_ports_then_branch_terminals";
    case Plan::xor_semantic_bitpack:
      return "xor_semantic_ports_then_bitpack_each_then_terminals";
    case Plan::xor_semantic_a048:
      return "xor_semantic_ports_then_a048_each_then_terminals";
    case Plan::dictionary: return "dictionary_ports_then_branch_terminals";
    case Plan::dictionary_bitpack:
      return "dictionary_ports_then_bitpack_each_then_terminals";
  }
  fail("unknown combo plan");
}

const char* terminal_mode_name(TerminalMode mode) {
  switch (mode) {
    case TerminalMode::all_zstd3: return "zstd3_on_every_output";
    case TerminalMode::all_lz4: return "lz4_on_every_output";
    case TerminalMode::alternating_zstd3_lz4:
      return "zstd3_lz4_by_output_index";
    case TerminalMode::alternating_lz4_zstd3:
      return "lz4_zstd3_by_output_index";
    case TerminalMode::first_raw_rest_zstd3:
      return "first_output_raw_zstd3_on_remaining_outputs";
    case TerminalMode::first_raw_rest_lz4:
      return "first_output_raw_lz4_on_remaining_outputs";
  }
  fail("unknown combo terminal mode");
}

std::vector<std::string> dtype_names(std::uint8_t mask) {
  std::vector<std::string> values;
  if ((mask & kInt32) != 0) values.emplace_back("int32");
  if ((mask & kInt64) != 0) values.emplace_back("int64");
  if ((mask & kFloat32) != 0) values.emplace_back("float32");
  if ((mask & kFloat64) != 0) values.emplace_back("float64");
  return values;
}

Json transform_steps(Plan plan) {
  Json steps = Json::array();
  auto stage = [&](std::string_view action, std::string profile,
                   Json parameters = Json::object(),
                   std::string branch = "current") {
    steps.push_back({{"kind", "stage"},
                     {"action", action},
                     {"profile", std::move(profile)},
                     {"parameters", std::move(parameters)},
                     {"branch", std::move(branch)}});
  };
  switch (plan) {
    case Plan::codec:
      break;
    case Plan::columnize:
      stage(kColumnize, "bytes",
            {{"mode", "transpose"}, {"width", "source_width"}});
      break;
    case Plan::xor_carrier:
      stage(kXorCarrier, "source_dtype");
      break;
    case Plan::for_bitpack:
      stage(kFor, "int64");
      stage(kBitpack, "bytes",
            {{"alignment_bytes", 1}, {"bit_order", "msb"},
             {"minimum_width", 0}, {"wide_cutoff", 64}},
            "each_FOR_port");
      break;
    case Plan::a048_planes:
      stage(kA048Planes, "uint64", {{"bit_width", 64}});
      break;
    case Plan::float_deconstruct:
      stage(kFloatDeconstruct, "float32", {{"format", "fp32"}});
      break;
    case Plan::xor_semantic:
      stage(kXorPorts, "source_dtype");
      break;
    case Plan::xor_semantic_bitpack:
      stage(kXorPorts, "source_dtype");
      stage(kBitpack, "bytes",
            {{"alignment_bytes", 1}, {"bit_order", "msb"},
             {"minimum_width", 0}, {"wide_cutoff", 64}},
            "each_XOR_semantic_port");
      break;
    case Plan::xor_semantic_a048:
      stage(kXorPorts, "source_dtype");
      stage(kA048Planes, "uint64", {{"bit_width", 64}},
            "each_XOR_semantic_port");
      break;
    case Plan::dictionary:
      stage(kDictionary, "source_dtype");
      break;
    case Plan::dictionary_bitpack:
      stage(kDictionary, "source_dtype");
      stage(kBitpack, "bytes",
            {{"alignment_bytes", 1}, {"bit_order", "msb"},
             {"minimum_width", 0}, {"wide_cutoff", 64}},
            "each_dictionary_port");
      break;
  }
  return steps;
}

Json catalog_entry(const Candidate& selected) {
  return {{"id", selected.id},
          {"candidate_id", selected.id},
          {"candidate_ordinal", selected.ordinal},
          {"kind", selected.kind},
          {"new_combination", selected.new_combination},
          {"component_ids", selected.component_ids},
          {"dtypes", dtype_names(selected.dtype_mask)},
          {"supported_dtypes", dtype_names(selected.dtype_mask)},
          {"requires_nonempty_values", selected.requires_nonempty},
          {"strict_only", true},
          {"allows_harness_raw_fallback", false},
          {"description", plan_name(selected.plan)},
          {"recipe",
           {{"policy_registry", kRegistry},
            {"root_count", 3},
            {"binding_schema", "HCB1"},
            {"plan", plan_name(selected.plan)},
            {"transform_steps", transform_steps(selected.plan)},
            {"terminal_assignment", terminal_mode_name(selected.terminals)},
            {"stop", true}}}};
}

}  // namespace

Json catalog() {
  Json entries = Json::array();
  for (const auto& selected : candidates())
    entries.push_back(catalog_entry(selected));
  return {{"schema", "harness-combo-pilot-catalog-v1"},
          {"policy_registry", kRegistry},
          {"scope",
           {{"candidate_count", entries.size()},
            {"frozen_operator_catalog_modified", false},
            {"claims_full_operator_cartesian_product", false},
            {"excluded_families",
             Json::array({"A106-A110", "PPMd", "DMC", "PAQ8", "CMIX",
                          "LSTM", "NNCP"})}}},
          {"candidates", std::move(entries)}};
}

std::vector<std::string> eligible(const Block& block) {
  const auto count = validate_block(block);
  std::vector<std::string> result;
  result.reserve(candidates().size());
  for (const auto& selected : candidates())
    if (eligible_after_validation(selected, block, count))
      result.emplace_back(selected.id);
  return result;
}

Encoded encode(const std::string& id, const Block& block) {
  const auto* selected = candidate(id);
  require(selected != nullptr, "unknown combo pilot candidate: " + id);
  const auto count = validate_block(block);
  require(eligible_after_validation(*selected, block, count),
          "candidate is not eligible for this dtype or shape: " + id);

  ComboDispatcher dispatcher(true);
  auto frame_bytes = execute_encode(*selected, block, count, dispatcher);
  const auto frame = native_graph_v1::detail::parse_frame(frame_bytes, limits());
  require(frame.leaves.contains(2), "combo pilot binding root is not a leaf");
  const auto binding = parse_binding(frame.leaves.at(2).data);
  require(binding.selected == selected,
          "encoded frame selected a different combo candidate");
  preflight_frame(frame, binding, dispatcher);

  const auto& fallback = dispatcher.fallback_audit();
  require(fallback.raw_fallback_stage_count == 0,
          "strict combo encode used a harness raw fallback");
  const auto digest = native_graph_v1::detail::sha256(frame_bytes);
  Json audit{{"schema", "harness-combo-pilot-encode-audit-v1"},
             {"candidate_id", selected->id},
             {"execution_status", "strict_ready"},
             {"strict", true},
             {"strict_requested", true},
             {"fallback_used", false},
             {"raw_fallback_stage_count", fallback.raw_fallback_stage_count},
             {"wrapped_stage_count", fallback.wrapped_stage_count},
             {"reasons", fallback.reasons},
             {"source",
              {{"dtype", block.dtype},
               {"row_count", block.row_count},
               {"valid_count", count},
               {"values_bytes", block.values.size()},
               {"missing_bytes", block.missing.size()}}},
             {"allowlist",
              {{"forward_passed", true},
               {"disabled_families_enforced", true},
               {"forbidden_hits", Json::array()}}},
             {"fallback_audit",
              {{"wrapped_stage_count", fallback.wrapped_stage_count},
               {"raw_fallback_stage_count", fallback.raw_fallback_stage_count},
               {"reasons", fallback.reasons},
               {"reason_counts", fallback.reason_counts}}},
             {"frame",
              {{"bytes", frame_bytes.size()},
               {"ngf_version", 1},
               {"root_count", frame.roots.size()},
               {"node_count", frame.nodes.size()},
               {"sha256", hex(digest)}}},
             {"timing_contract",
              {{"audit_included_in_encode_call", true},
               {"includes",
                Json::array({"input_validation", "graph_execution", "NGF_framing",
                             "fallback_audit", "frame_preflight"})}}}};
  return {std::move(frame_bytes), std::move(audit)};
}

Block decode(std::span<const std::uint8_t> bytes,
             std::string_view expected_candidate_id) {
  ComboDispatcher dispatcher(true);
  const auto frame = native_graph_v1::detail::parse_frame(bytes, limits());
  require(frame.leaves.contains(2), "combo pilot binding root is not a leaf");
  const auto binding = parse_binding(frame.leaves.at(2).data);
  require(expected_candidate_id.empty() ||
              expected_candidate_id == binding.selected->id,
          "outer method name differs from the bound combo candidate");
  preflight_frame(frame, binding, dispatcher);

  auto decoded = native_graph_v1::decode(bytes, dispatcher, limits());
  require(decoded.roots.size() == 3 &&
              decoded.original_numeric_bytes == binding.value_bytes,
          "combo pilot decoded root count or denominator mismatch");
  require(decoded.roots[0].stream.kind == root_kind(*binding.selected) &&
              decoded.roots[0].stream.width == dtype_width(binding.dtype) &&
              decoded.roots[0].stream.data.size() == binding.value_bytes &&
              decoded.roots[1].stream.kind == Kind::Serial &&
              decoded.roots[1].stream.width == 1 &&
              decoded.roots[1].stream.data.size() == binding.missing_bytes &&
              native_graph_v1::detail::sha256(decoded.roots[1].stream.data) ==
                  binding.missing_digest &&
              decoded.roots[2].stream.data == frame.leaves.at(2).data,
          "combo pilot decoded roots differ from their binding");

  const auto& fallback = dispatcher.fallback_audit();
  require(fallback.raw_fallback_stage_count == 0,
          "strict combo decode used a harness raw fallback");
  Block result{binding.dtype, binding.row_count,
               std::move(decoded.roots[0].stream.data),
               std::move(decoded.roots[1].stream.data)};
  const auto count = validate_block(result);
  require(count == binding.value_count,
          "decoded valid count differs from combo pilot binding");
  require(eligible_after_validation(*binding.selected, result, count),
          "decoded block violates the bound candidate domain");
  return result;
}

std::string tree_identifier(const Json& tree) {
  return parse_tree(tree).identifier;
}

Json tree_contract() {
  return {
      {"schema", kTreeRegistry},
      {"method_identifier", "tree_v1:<sha256(canonical_tree_json)>"},
      {"limits",
       {{"canonical_json_bytes", kMaxTreeJsonBytes},
        {"json_nesting_depth", kMaxTreeJsonNesting},
        {"nodes", kMaxTreeNodes},
        {"node_id_bytes", kMaxTreeNodeIdBytes},
        {"inputs_per_node", 1},
        {"reference_port_upper_bound_exclusive", 32},
        {"active_streams", 32},
        {"ordinary_depth", 8},
        {"stream_bytes", kMaxStreamBytes},
        {"working_bytes", 512ULL << 20},
        {"metadata_bytes", 4ULL << 20},
        {"frame_bytes", 1ULL << 30}}},
      {"shape",
       {{"top_level_fields", Json::array({"schema", "source", "nodes"})},
        {"source_fields", Json::array({"root_kind"})},
        {"root_kinds", Json::array({"numeric", "struct"})},
        {"node_fields", Json::array({"id", "op", "inputs", "parameters"})},
        {"input_reference", "values or an earlier node_id:port"},
        {"single_consumer_streams", true},
        {"implicit_stop_keeps_all_unconsumed_leaves", true},
        {"unknown_fields_rejected", true}}},
      {"operators",
       Json::array(
           {{{"op", "zstd3"}, {"parameters", Json::object()}},
            {{"op", "lz4"}, {"parameters", Json::object()}},
            {{"op", "columnize"},
             {"parameters", {{"mode", "transpose"}}}},
            {{"op", "xor_carrier"}, {"parameters", Json::object()}},
            {{"op", "xor_semantic"}, {"parameters", Json::object()}},
            {{"op", "for"}, {"parameters", Json::object()}},
            {{"op", "bitpack"},
             {"parameters",
              {{"alignment_bytes", 1},
               {"bit_order", "msb"},
               {"minimum_width", 0},
               {"wide_cutoff", 64}}}},
            {{"op", "dictionary"}, {"parameters", Json::object()}},
            {{"op", "a048_planes"},
             {"parameters", {{"bit_width", 64}}}},
            {{"op", "float_deconstruct"},
             {"parameters", {{"format", "fp32"}}}}})},
      {"security",
       {{"runtime_actions_profiles_and_policy_ids_are_internal", true},
        {"strict_raw_fallback_forbidden", true},
        {"excluded_families",
         Json::array({"A106-A110", "PPMd", "DMC", "PAQ8", "CMIX", "LSTM",
                      "NNCP"})}}},
      {"binding",
       {{"schema", "HCB2"},
        {"contains_complete_canonical_tree", true},
        {"contains_tree_sha256", true},
        {"contains_source_dtype_shape_and_missing_sha256", true},
        {"validated_before_inverse", true}}}};
}

Encoded encode_tree(const Json& tree, const Block& block) {
  const auto parsed_tree = parse_tree(tree);
  const auto count = validate_block(block);
  ComboDispatcher dispatcher(true);
  auto frame_bytes =
      execute_encode_tree(parsed_tree, block, count, dispatcher);
  const auto frame = native_graph_v1::detail::parse_frame(frame_bytes, limits());
  require(frame.leaves.contains(2),
          "explicit operator tree binding root is not a leaf");
  const auto binding = parse_tree_binding(frame.leaves.at(2).data);
  require(binding.tree.identifier == parsed_tree.identifier &&
              binding.tree.canonical == parsed_tree.canonical,
          "encoded frame bound a different explicit operator tree");
  preflight_tree_frame(frame, binding, dispatcher);

  const auto& fallback = dispatcher.fallback_audit();
  require(fallback.raw_fallback_stage_count == 0,
          "strict explicit operator tree encode used a harness raw fallback");
  const auto digest = native_graph_v1::detail::sha256(frame_bytes);
  Json audit{{"schema", "ucc-operator-tree-encode-audit-v1"},
             {"method", parsed_tree.identifier},
             {"tree_sha256", hex(parsed_tree.digest)},
             {"execution_status", "strict_ready"},
             {"strict", true},
             {"strict_requested", true},
             {"fallback_used", false},
             {"raw_fallback_stage_count", fallback.raw_fallback_stage_count},
             {"wrapped_stage_count", fallback.wrapped_stage_count},
             {"reasons", fallback.reasons},
             {"source",
              {{"dtype", block.dtype},
               {"root_kind",
                parsed_tree.root_kind == Kind::Numeric ? "numeric" : "struct"},
               {"row_count", block.row_count},
               {"valid_count", count},
               {"values_bytes", block.values.size()},
               {"missing_bytes", block.missing.size()}}},
             {"allowlist",
              {{"semantic_dsl_passed", true},
               {"forward_passed", true},
               {"disabled_families_enforced", true},
               {"forbidden_hits", Json::array()}}},
             {"fallback_audit",
              {{"wrapped_stage_count", fallback.wrapped_stage_count},
               {"raw_fallback_stage_count", fallback.raw_fallback_stage_count},
               {"reasons", fallback.reasons},
               {"reason_counts", fallback.reason_counts}}},
             {"frame",
              {{"bytes", frame_bytes.size()},
               {"ngf_version", 1},
               {"binding_schema", "HCB2"},
               {"root_count", frame.roots.size()},
               {"node_count", frame.nodes.size()},
               {"sha256", hex(digest)}}},
             {"timing_contract",
              {{"audit_included_in_encode_call", true},
               {"includes",
                Json::array({"tree_validation", "input_validation",
                             "graph_execution", "NGF_framing", "fallback_audit",
                             "frame_preflight"})}}}};
  return {std::move(frame_bytes), std::move(audit)};
}

Block decode_tree(std::span<const std::uint8_t> bytes,
                  std::string_view expected_method) {
  ComboDispatcher dispatcher(true);
  const auto frame = native_graph_v1::detail::parse_frame(bytes, limits());
  require(frame.leaves.contains(2),
          "explicit operator tree binding root is not a leaf");
  const auto binding = parse_tree_binding(frame.leaves.at(2).data);
  require(expected_method.empty() || expected_method == binding.tree.identifier,
          "outer method name differs from the bound explicit operator tree");
  preflight_tree_frame(frame, binding, dispatcher);

  auto decoded = native_graph_v1::decode(bytes, dispatcher, limits());
  require(decoded.roots.size() == 3 &&
              decoded.original_numeric_bytes == binding.value_bytes,
          "explicit operator tree decoded root count or denominator mismatch");
  require(decoded.roots[0].stream.kind == binding.tree.root_kind &&
              decoded.roots[0].stream.width == dtype_width(binding.dtype) &&
              decoded.roots[0].stream.data.size() == binding.value_bytes &&
              decoded.roots[1].stream.kind == Kind::Serial &&
              decoded.roots[1].stream.width == 1 &&
              decoded.roots[1].stream.data.size() == binding.missing_bytes &&
              native_graph_v1::detail::sha256(decoded.roots[1].stream.data) ==
                  binding.missing_digest &&
              decoded.roots[2].stream.data == frame.leaves.at(2).data,
          "explicit operator tree decoded roots differ from their binding");

  const auto& fallback = dispatcher.fallback_audit();
  require(fallback.raw_fallback_stage_count == 0,
          "strict explicit operator tree decode used a harness raw fallback");
  Block result{binding.dtype, binding.row_count,
               std::move(decoded.roots[0].stream.data),
               std::move(decoded.roots[1].stream.data)};
  const auto count = validate_block(result);
  require(count == binding.value_count,
          "decoded valid count differs from explicit operator tree binding");
  return result;
}

}  // namespace harness_combo_pilot_v1
