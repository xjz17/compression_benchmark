#include "api.hpp"
#include "selection.hpp"
#include "bos_bfe_v2/bos2_refined.hpp"
#include "bos_bfe_v2/ucc_bos_bfe_bridge.hpp"
#include "../native_graph_v1/graph_internal.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace uc = unified_compression_v1;
namespace bridge = unified_compression_v1::bos_bfe_bridge;
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <class F>
void rejected(F&& operation, std::string_view expected_code = {}) {
  try {
    operation();
  } catch (const uc::Error& error) {
    if (!expected_code.empty() && error.code() != expected_code)
      throw std::runtime_error("unexpected unified error code: expected " +
                               std::string(expected_code) + ", got " +
                               error.code() + " (" + error.what() + ")");
    return;
  } catch (const std::exception&) {
    require(expected_code.empty(), "rejection bypassed the unified error API");
    return;
  }
  throw std::runtime_error("invalid BOS request unexpectedly succeeded");
}

void put(uc::Bytes& output, std::uint64_t value, unsigned bytes) {
  for (unsigned index = 0; index != bytes; ++index) {
    output.push_back(static_cast<std::uint8_t>(value));
    value >>= 8;
  }
}

std::uint64_t get(std::span<const std::uint8_t> input, std::size_t offset,
                  unsigned bytes) {
  require(offset + bytes <= input.size(), "truncated UCC1 test fixture");
  std::uint64_t result = 0;
  for (unsigned index = 0; index != bytes; ++index)
    result |= static_cast<std::uint64_t>(input[offset + index]) << (8 * index);
  return result;
}

void append_digest(uc::Bytes& frame) {
  std::array<unsigned char, 32> digest{};
  unsigned size = 0;
  require(EVP_Digest(frame.data(), frame.size(), digest.data(), &size,
                     EVP_sha256(), nullptr) == 1 &&
              size == digest.size(),
          "test SHA-256 failed");
  frame.insert(frame.end(), digest.begin(), digest.end());
}

void refresh_digest(uc::Bytes& frame) {
  require(frame.size() >= 32, "truncated UCC1 digest fixture");
  std::array<unsigned char, 32> digest{};
  unsigned size = 0;
  require(EVP_Digest(frame.data(), frame.size() - digest.size(), digest.data(),
                     &size, EVP_sha256(), nullptr) == 1 &&
              size == digest.size(),
          "test SHA-256 refresh failed");
  std::copy(digest.begin(), digest.end(), frame.end() - digest.size());
}

uc::Bytes wrap_algorithm(std::string_view name,
                         std::span<const std::uint8_t> payload) {
  uc::Bytes frame{'U', 'C', 'C', '1', 1, 1, 0, 0};
  put(frame, name.size(), 4);
  put(frame, payload.size(), 8);
  frame.insert(frame.end(), name.begin(), name.end());
  frame.insert(frame.end(), payload.begin(), payload.end());
  append_digest(frame);
  return frame;
}

struct EnvelopeView {
  std::string method;
  std::size_t payload_offset = 0;
  std::size_t payload_size = 0;
};

EnvelopeView envelope(std::span<const std::uint8_t> frame) {
  require(frame.size() >= 52 &&
              std::equal(frame.begin(), frame.begin() + 4, "UCC1") &&
              frame[4] == 1 && frame[5] == 1,
          "result is not an algorithm UCC1 frame");
  const auto name_size = static_cast<std::size_t>(get(frame, 8, 4));
  const auto payload_size = static_cast<std::size_t>(get(frame, 12, 8));
  require(20 + name_size + payload_size + 32 == frame.size(),
          "UCC1 test view found invalid lengths");
  return {std::string(reinterpret_cast<const char*>(frame.data() + 20),
                      name_size),
          20 + name_size, payload_size};
}

std::span<const std::uint8_t> inner(const uc::Bytes& frame) {
  const auto view = envelope(frame);
  return std::span<const std::uint8_t>(frame).subspan(view.payload_offset,
                                                     view.payload_size);
}

bool contains_ascii(std::span<const std::uint8_t> input,
                    std::string_view token) {
  return std::search(input.begin(), input.end(), token.begin(), token.end()) !=
         input.end();
}

std::string hex(std::span<const std::uint8_t> input) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(input.size() * 2);
  for (const auto byte : input) {
    output.push_back(digits[byte >> 4]);
    output.push_back(digits[byte & 15]);
  }
  return output;
}

template <class UInt>
void append_word(uc::Bytes& output, UInt value) {
  for (std::size_t byte = 0; byte != sizeof(UInt); ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

template <class UInt>
uc::Column patterned_column(std::string dtype, std::uint64_t rows,
                            std::span<const UInt> words,
                            std::span<const std::uint64_t> missing_rows = {}) {
  require(!words.empty(), "word pattern cannot be empty");
  uc::Column result{std::move(dtype), rows, {}, uc::Bytes((rows + 7) / 8, 0)};
  for (const auto row : missing_rows) {
    require(row < rows, "missing-row fixture is out of range");
    result.missing[row / 8] |= static_cast<std::uint8_t>(1U << (row % 8));
  }
  std::size_t value_index = 0;
  for (std::uint64_t row = 0; row != rows; ++row) {
    if ((result.missing[row / 8] & (1U << (row % 8))) != 0) continue;
    append_word(result.values, words[value_index++ % words.size()]);
  }
  uc::validate_column(result);
  return result;
}

uc::Column edge_f32() {
  static constexpr std::array<std::uint32_t, 14> words{
      0x00000000U, 0x80000000U, 0x00000001U, 0x007fffffU,
      0x00800000U, 0x7f7fffffU, 0xff7fffffU, 0x7f800000U,
      0xff800000U, 0x7fc01234U, 0x7fa12345U, 0xffc05678U,
      std::bit_cast<std::uint32_t>(1.25F),
      std::bit_cast<std::uint32_t>(-91.0625F)};
  static constexpr std::array<std::uint64_t, 3> missing{2, 11, 17};
  return patterned_column("float32", 21,
                          std::span<const std::uint32_t>(words),
                          std::span<const std::uint64_t>(missing));
}

uc::Column edge_f64() {
  static constexpr std::array<std::uint64_t, 14> words{
      0x0000000000000000ULL, 0x8000000000000000ULL,
      0x0000000000000001ULL, 0x000fffffffffffffULL,
      0x0010000000000000ULL, 0x7fefffffffffffffULL,
      0xffefffffffffffffULL, 0x7ff0000000000000ULL,
      0xfff0000000000000ULL, 0x7ff8000000001234ULL,
      0x7ff0123456789abcULL, 0xfff8000000005678ULL,
      std::bit_cast<std::uint64_t>(1.25),
      std::bit_cast<std::uint64_t>(-91.0625)};
  static constexpr std::array<std::uint64_t, 3> missing{0, 9, 20};
  return patterned_column("float64", 23,
                          std::span<const std::uint64_t>(words),
                          std::span<const std::uint64_t>(missing));
}

uc::Column strict_f64() {
  static constexpr std::array<std::uint64_t, 4> words{
      std::bit_cast<std::uint64_t>(1.25),
      std::bit_cast<std::uint64_t>(1.50),
      std::bit_cast<std::uint64_t>(1.75),
      std::bit_cast<std::uint64_t>(2.00)};
  return patterned_column("float64", 1024,
                          std::span<const std::uint64_t>(words));
}

uc::Column multi_f64() {
  static constexpr std::array<std::uint64_t, 8> words{
      std::bit_cast<std::uint64_t>(-5.125),
      std::bit_cast<std::uint64_t>(0.0),
      std::bit_cast<std::uint64_t>(12.375),
      std::bit_cast<std::uint64_t>(999.001),
      0x8000000000000000ULL, 0x0000000000000001ULL,
      0x7ff8000000004321ULL, 0x7ff0000000000000ULL};
  static constexpr std::array<std::uint64_t, 5> missing{
      0, 9999, 10000, 20003, 25012};
  return patterned_column("float64", 25013,
                          std::span<const std::uint64_t>(words),
                          std::span<const std::uint64_t>(missing));
}

uc::Column random_f64(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 random(seed);
  std::vector<std::uint64_t> words(count);
  for (auto& word : words) word = random();
  return patterned_column("float64", count,
                          std::span<const std::uint64_t>(words));
}

uc::Column integer_boundaries() {
  static constexpr std::array<std::uint64_t, 8> words{
      std::bit_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::min()),
      std::bit_cast<std::uint64_t>(std::int64_t{-1}), 0, 1, 2, 17,
      std::bit_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
      UINT64_C(0x8000000000000001)};
  static constexpr std::array<std::uint64_t, 2> missing{1, 9};
  return patterned_column("int64", 18,
                          std::span<const std::uint64_t>(words),
                          std::span<const std::uint64_t>(missing));
}

Json request(const uc::Column& column) {
  return {{"command", "compress"},
          {"method", "algorithm:BOS"},
          {"dtype", column.dtype},
          {"row_count", column.row_count},
          {"values_le_hex", hex(column.values)},
          {"missing_bitmap_hex", hex(column.missing)},
          {"warmups", 0},
          {"repeats", 1}};
}

void write_fixtures(const fs::path& directory) {
  fs::create_directories(directory);
  const std::vector<std::pair<std::string, uc::Column>> fixtures{
      {"edge_float32", edge_f32()},
      {"edge_float64", edge_f64()},
      {"multi_float64_25013", multi_f64()},
      {"empty_float32", uc::Column{"float32", 0, {}, {}}},
      {"empty_float64", uc::Column{"float64", 0, {}, {}}},
      {"all_null_float32", uc::Column{"float32", 17, {}, {0xff, 0xff, 0x01}}},
      {"all_null_float64", uc::Column{"float64", 17, {}, {0xff, 0xff, 0x01}}},
      {"integer_boundaries", integer_boundaries()},
      {"empty_int64", uc::Column{"int64", 0, {}, {}}},
      {"all_null_int64", uc::Column{"int64", 17, {}, {0xff, 0xff, 0x01}}},
  };
  Json manifest{{"schema", "bos-harness-cli-fixtures-v1"},
                {"method", "algorithm:BOS"},
                {"fixture_count", fixtures.size()},
                {"files", Json::array()}};
  for (const auto& [name, column] : fixtures) {
    uc::validate_column(column);
    const auto filename = name + ".json";
    std::ofstream output(directory / filename, std::ios::trunc);
    require(static_cast<bool>(output), "cannot create BOS CLI fixture");
    output << request(column).dump(2) << '\n';
    require(static_cast<bool>(output), "cannot write BOS CLI fixture");
    manifest["files"].push_back(
        {{"name", name}, {"path", filename}, {"dtype", column.dtype},
         {"row_count", column.row_count},
         {"valid_numeric_bytes", column.values.size()}});
  }
  std::ofstream output(directory / "MANIFEST.json", std::ios::trunc);
  require(static_cast<bool>(output), "cannot create BOS fixture manifest");
  output << manifest.dump(2) << '\n';
  require(static_cast<bool>(output), "cannot write BOS fixture manifest");
}

void require_roundtrip(const uc::Registry& registry,
                       const uc::CompressionResult& result,
                       const uc::Column& source) {
  require(result.method == "algorithm:BOS", "BOS public route changed");
  require(result.bit_exact && registry.decompress(result.compressed_data) == source,
          "BOS UCC1 frame is not bit exact or independently decodable");
  require(result.compressed_bytes == result.compressed_data.size() &&
              result.original_numeric_bytes == source.values.size(),
          "BOS public accounting excludes bytes");
  require(result.execution_audit.is_object() &&
              result.execution_audit.contains("strict") &&
              result.execution_audit.at("strict").is_boolean() &&
              result.execution_audit.contains("fallback_used") &&
              result.execution_audit.at("fallback_used").is_boolean() &&
              result.execution_audit.contains("raw_fallback_stage_count") &&
              result.execution_audit.at("raw_fallback_stage_count").is_number_unsigned(),
          "BOS public execution audit lacks exact fallback facts");
  const bool fallback = result.execution_audit.at("fallback_used").get<bool>();
  const auto fallback_count =
      result.execution_audit.at("raw_fallback_stage_count").get<std::uint64_t>();
  const auto payload = inner(result.compressed_data);
  require(bridge::is_frame(payload), "BOS result has an unrecognized inner frame");
  const auto info = bridge::inspect(payload);
  require((fallback_count != 0) == fallback &&
              result.execution_audit.at("strict").get<bool>() == !fallback &&
              fallback_count == info.raw_block_count,
          "BOS strict/fallback audit fields disagree");
}

void test_inventory_and_contract(const uc::Registry& registry) {
  require(registry.methods().size() == 218,
          "BOS replacement changed the 86+102+30 registry size");
  std::size_t combinations = 0;
  for (const auto& method : registry.methods())
    combinations += method.kind == "combo";
  require(combinations == 30, "BOS replacement changed the fixed30 catalog");

  const auto& bos = registry.resolve("algorithm:BOS");
  require(bos.kind == "algorithm" && bos.enabled && bos.available,
          "BOS is not an enabled algorithm route");
  const std::set<std::string> dtypes(bos.dtypes.begin(), bos.dtypes.end());
  require(dtypes == std::set<std::string>({"float32", "float64", "int64"}),
          "BOS does not advertise exactly int64/float32/float64");
  require(bos.contract.value("integer_route_id", "") ==
              "project_bos2_partitioned" &&
              bos.contract.value("float_route_id", "") == bridge::kRouteId &&
              bos.contract.value("legacy_integer_decode", "") ==
                  "project_bos64_adapter",
          "BOS public contract does not bind new and legacy implementations");
  require(bos.contract.value("timed_float_boundary", "") ==
              "whole-column precision scan + both complete NGF candidates + "
              "BHC1/UCC1 construction",
          "BOS public contract omits the complete float timing boundary");
}

void test_integer_replacement_and_legacy_decode(const uc::Registry& registry) {
  const std::vector<std::int64_t> values{
      std::numeric_limits<std::int64_t>::min(), -91, -7, 0, 1, 1, 2, 3,
      5, 8, 13, 21, 34, 55, 89, std::numeric_limits<std::int64_t>::max()};
  const auto source = uc::make_column(std::span(values));

  const std::vector<uc::Column> current_cases{
      source, uc::Column{"int64", 0, {}, {}},
      uc::Column{"int64", 17, {}, {0xff, 0xff, 0x01}}};
  for (const auto& current_source : current_cases) {
    const auto current =
        registry.compress(current_source, "algorithm:BOS", {1, 2});
    require_roundtrip(registry, current, current_source);
    const auto current_inner = inner(current.compressed_data);
    require(bridge::is_frame(current_inner),
            "new int64 BOS payload is outside the versioned BOS bridge");
    const auto info = bridge::inspect(current_inner);
    require(info.dtype == "int64" &&
                info.row_count == current_source.row_count &&
                info.selected_frontend == "bos2" && info.block_count == 1 &&
                info.bos2_block_count == 1,
            "new int64 BOS payload is not the project BOS2 route");

    std::vector<std::int64_t> compact;
    compact.reserve(current_source.values.size() / 8);
    for (std::size_t offset = 0; offset < current_source.values.size();
         offset += 8) {
      std::uint64_t word = 0;
      for (unsigned byte = 0; byte != 8; ++byte)
        word |= static_cast<std::uint64_t>(current_source.values[offset + byte])
                << (byte * 8);
      compact.push_back(std::bit_cast<std::int64_t>(word));
    }
    // The verified validated BOS-v2 deployment's public baseline adapter short-circuits a column
    // with zero valid values to a zero-byte BOS2 payload.  The isolated core
    // function retains its own non-wire empty representation, so compare it
    // only when at least one value is present.
    const auto expected = compact.empty()
                              ? uc::Bytes{}
                              : uc::bos_bfe_v2::integer::optimized_compose(compact);
    const auto parsed = native_graph_v1::detail::parse_frame(current_inner, {});
    require(parsed.nodes.size() == 1 &&
                parsed.nodes.front().outputs.size() == 2,
            "BOS2 reference comparison found an unexpected graph shape");
    const auto payload_leaf =
        parsed.leaves.find(parsed.nodes.front().outputs.front());
    const auto bitmap_leaf =
        parsed.leaves.find(parsed.nodes.front().outputs.back());
    require(payload_leaf != parsed.leaves.end() &&
                payload_leaf->second.data == expected,
            compact.empty()
                ? "zero-valid harness BOS2 leaf differs from the pinned "
                  "validated BOS-v2 deployment public wire"
                : "harness BOS2 leaf differs from the pinned Project-150 kernel");
    require(bitmap_leaf != parsed.leaves.end() &&
                bitmap_leaf->second.data == current_source.missing,
            "harness BOS2 leaf changed the source NULL bitmap");
    if (!current_source.values.empty())
      require(contains_ascii(current_inner, "BOS2"),
              "non-empty int64 BOS frame lacks the BOS2 wire");
  }

  const auto current = registry.compress(source, "algorithm:BOS");
  const auto current_inner = inner(current.compressed_data);

  // The source-tree baseline is deliberately retained as the predecessor
  // encoder.  It creates an authentic old project_bos64_adapter NGF frame;
  // only the UCC1 envelope is assembled by this test.
  const auto predecessor_inner = native_baseline86_v1::encode("BOS", source);
  const auto predecessor_ucc = wrap_algorithm("algorithm:BOS", predecessor_inner);
  require(registry.decompress(predecessor_ucc) == source,
          "new BOS decoder lost old integer UCC1/NGF compatibility");
  require(current_inner.size() != predecessor_inner.size() ||
              !std::equal(current_inner.begin(), current_inner.end(),
                          predecessor_inner.begin()),
          "new integer BOS silently kept the predecessor BOS64 encoder");
}

void test_float_shapes_and_edges(const uc::Registry& registry) {
  const std::vector<uc::Column> cases{
      edge_f32(), edge_f64(), uc::Column{"float32", 0, {}, {}},
      uc::Column{"float64", 17, {}, {0xff, 0xff, 0x01}}};
  for (const auto& source : cases) {
    const auto result = registry.compress(source, "algorithm:BOS", {1, 2});
    require_roundtrip(registry, result, source);
    const auto payload = inner(result.compressed_data);
    require(bridge::is_frame(payload), "float BOS inner frame is unrecognized");
    const auto info = bridge::inspect(payload);
    require(info.dtype == source.dtype && info.row_count == source.row_count &&
                info.block_count == 1 &&
                (info.selected_frontend == "bfe1" ||
                 info.selected_frontend == "bfe1_raw" ||
                 info.selected_frontend == "dsc3"),
            "single-block float BOS inspection differs from its input");
  }

  const auto source = multi_f64();
  const auto result = registry.compress(source, "algorithm:BOS");
  require_roundtrip(registry, result, source);
  const auto payload = inner(result.compressed_data);
  static constexpr std::array<std::uint8_t, 8> magic{
      'B', 'O', 'S', 'B', 'H', 'C', '1', 0};
  require(payload.size() >= magic.size() &&
              std::equal(magic.begin(), magic.end(), payload.begin()),
          "25013-row BOS column did not use BHC1");
  const auto info = bridge::inspect(payload);
  require(info.dtype == "float64" && info.row_count == 25013 &&
              info.selected_frontend == "chunked" && info.block_count == 3,
          "BHC1 did not preserve its three-block column contract");
}

void test_bindings_and_corruption(const uc::Registry& registry) {
  const auto source = multi_f64();
  const auto encoded = registry.compress(source, "algorithm:BOS");
  const auto view = envelope(encoded.compressed_data);

  auto corrupt_outer = encoded.compressed_data;
  corrupt_outer.back() ^= 1;
  rejected([&] { registry.decompress(corrupt_outer); }, "invalid_frame");

  auto corrupt_inner = encoded.compressed_data;
  corrupt_inner[view.payload_offset + view.payload_size / 2] ^= 0x40;
  refresh_digest(corrupt_inner);
  rejected([&] { registry.decompress(corrupt_inner); }, "invalid_frame");

  auto forged_route = encoded.compressed_data;
  require(view.method == "algorithm:BOS" &&
              view.method.size() == std::string_view("algorithm:LZ4").size(),
          "route-forgery fixture width changed");
  std::copy_n("algorithm:LZ4", view.method.size(),
              forged_route.begin() + 20);
  refresh_digest(forged_route);
  rejected([&] { registry.decompress(forged_route); }, "invalid_frame");

  auto forged_kind = encoded.compressed_data;
  forged_kind[5] = 2;
  refresh_digest(forged_kind);
  rejected([&] { registry.decompress(forged_kind); }, "invalid_frame");
}

void test_selection_and_raw_audit(const uc::Registry& registry) {
  const auto strict_source = strict_f64();
  uc::SelectionOptions options;
  options.candidate_scope = "bos-plus-known-combo";
  options.budget.max_trials = 2;
  options.candidates = {
      uc::Candidate::method_candidate("new-bos", "algorithm:BOS"),
      uc::Candidate::method_candidate("known-combo", "combo_v1:zstd3")};
  const auto selection = uc::select(registry, strict_source, options);
  require(selection.feedback.size() == 2,
          "budgeted selector did not measure BOS and the known combo");
  const auto bos_trial = std::find_if(
      selection.feedback.begin(), selection.feedback.end(),
      [](const auto& item) { return item.candidate.id == "new-bos"; });
  require(bos_trial != selection.feedback.end() && bos_trial->status == "ok" &&
              bos_trial->strict && bos_trial->bit_exact && bos_trial->feasible,
          "strict float BOS was not eligible for budgeted selection");
  require(selection.selected_result &&
              registry.decompress(selection.selected_result->compressed_data) ==
                  strict_source,
          "BOS/known-combo selector did not return a reusable measured winner");

  // Random IEEE words make BFE's complete raw mode plausible.  Discovering
  // such a public result is intentionally based on the implementation audit,
  // rather than a duplicated BFE cost model in the test.  If it is selected,
  // it must remain bit exact but cannot enter the ranked set.
  std::optional<uc::Column> raw_source;
  for (const auto count : {std::size_t{17}, std::size_t{257},
                           std::size_t{4096}, std::size_t{10000}}) {
    for (std::uint64_t seed = 1; seed <= 4 && !raw_source; ++seed) {
      auto candidate = random_f64(0x424f530000000000ULL + seed, count);
      const auto measured = registry.compress(candidate, "algorithm:BOS");
      require_roundtrip(registry, measured, candidate);
      if (measured.execution_audit.at("fallback_used").get<bool>())
        raw_source = std::move(candidate);
    }
  }
  require(raw_source.has_value(),
          "deterministic random IEEE fixtures did not exercise BFE raw mode");
  if (raw_source) {
    options = {};
    options.budget.max_trials = 1;
    options.candidates = {
        uc::Candidate::method_candidate("raw-bfe", "algorithm:BOS")};
    const auto raw_selection = uc::select(registry, *raw_source, options);
    require(raw_selection.feedback.size() == 1 &&
                raw_selection.feedback.front().status == "non_strict" &&
                !raw_selection.feedback.front().strict &&
                !raw_selection.selected_candidate &&
                !raw_selection.selected_result,
            "BFE raw mode entered the selector's ranked set");
  }
}

Json one_node_tree(std::string_view op) {
  return {{"schema", "ucc-operator-tree-v1"},
          {"source", {{"root_kind", "numeric"}}},
          {"nodes", Json::array({{{"id", "forbidden"},
                                   {"op", op},
                                   {"inputs", Json::array({"values"})},
                                   {"parameters", Json::object()}}})}};
}

void test_exclusions(const uc::Registry& registry) {
  const std::vector<std::int64_t> values{1, 2, 3, 5, 8, 13};
  const auto source = uc::make_column(std::span(values));
  const auto contract = registry.tree_contract();
  std::set<std::string> excluded;
  for (const auto& family : contract.at("security").at("excluded_families"))
    excluded.insert(family.get<std::string>());
  for (const auto* family : {"NNCP", "PPMd", "DMC", "PAQ8", "CMIX", "LSTM"}) {
    require(excluded.contains(family), "tree contract lost an excluded family");
    rejected([&] { registry.compress(source, family); }, "method_disabled");
    auto lower = std::string(family);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](char ch) {
      return ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch + ('a' - 'A')) : ch;
    });
    rejected([&] { registry.compress_tree(source, one_node_tree(lower)); });
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--write-fixtures") {
      write_fixtures(fs::path(argv[2]));
      std::cout << "wrote BOS CLI fixtures to " << argv[2] << '\n';
      return 0;
    }
    require(argc == 1,
            "usage: bos_harness_integration_test [--write-fixtures DIRECTORY]");
    uc::Registry registry;
    test_inventory_and_contract(registry);
    test_integer_replacement_and_legacy_decode(registry);
    test_float_shapes_and_edges(registry);
    test_bindings_and_corruption(registry);
    test_selection_and_raw_audit(registry);
    test_exclusions(registry);

    const auto& old = native_baseline86_v1::method("BOS");
    const bool old_float =
        std::find(old.dtypes.begin(), old.dtypes.end(), "float32") != old.dtypes.end() ||
        std::find(old.dtypes.begin(), old.dtypes.end(), "float64") != old.dtypes.end();
    require(!old_float,
            "legacy baseline unexpectedly advertises an undocumented float BOS frame");
    std::cout << "BOS harness integration passed: BOS2 replacement, BFE/DSC3 "
                 "float route, BHC1, legacy integer cold decode, binding, audit, "
                 "selection, fixed30 and exclusions; no predecessor float frame "
                 "exists in the retained baseline\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
