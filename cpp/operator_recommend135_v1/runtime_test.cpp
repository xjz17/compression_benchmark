#include "runtime.hpp"

#include "graph_internal.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using operator_recommend135_v1::Block;
using operator_recommend135_v1::PreparedRegistry;
using native_graph_v1::Bytes;

template <class T>
Bytes raw(std::initializer_list<T> values) {
  Bytes result(values.size() * sizeof(T));
  if (!result.empty()) std::memcpy(result.data(), values.begin(), result.size());
  return result;
}

void rejected(const std::function<void()>& operation) {
  bool failed = false;
  try {
    operation();
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed);
}

Block int64_fixture() {
  return {"int64", 7,
          raw<std::int64_t>({std::numeric_limits<std::int64_t>::min(), -1, 0,
                             1, std::numeric_limits<std::int64_t>::max()}),
          {0b00100100}};
}

Block float64_fixture() {
  return {"float64", 5,
          raw<std::uint64_t>({UINT64_C(0x7ff80000000000a5),
                              UINT64_C(0x8000000000000000),
                              UINT64_C(0x3ff0000000000000),
                              UINT64_C(0xfff0000000000000)}),
          {0b00000010}};
}

Block float32_fixture() {
  return {"float32", 6,
          raw<std::uint32_t>({UINT32_C(0x7fc01235), UINT32_C(0x80000000),
                              UINT32_C(0x3f800000), UINT32_C(0xff800000)}),
          {0b00010010}};
}

Block int32_fixture() {
  return {"int32", 5,
          raw<std::int32_t>({std::numeric_limits<std::int32_t>::min(), -7, 0,
                             std::numeric_limits<std::int32_t>::max()}),
          {0b00000100}};
}

void roundtrip(const std::string& id, const Block& block) {
  assert(operator_recommend135_v1::candidate_eligible(id, block));
  const auto frame = operator_recommend135_v1::encode_candidate(id, block);
  assert(frame.size() >= 48);
  assert(std::equal(frame.begin(), frame.begin() + 4, "NGF1"));
  assert(operator_recommend135_v1::decode_candidate(frame) == block);
}

const Block& fixture_for(const std::vector<std::string>& dtypes,
                         const Block& i32, const Block& i64,
                         const Block& f32, const Block& f64) {
  if (std::find(dtypes.begin(), dtypes.end(), "int64") != dtypes.end()) return i64;
  if (std::find(dtypes.begin(), dtypes.end(), "float32") != dtypes.end()) return f32;
  if (std::find(dtypes.begin(), dtypes.end(), "float64") != dtypes.end()) return f64;
  return i32;
}

std::vector<std::string> reference_eligible(const nlohmann::json& catalog,
                                            const Block& block) {
  std::vector<std::string> result;
  for (const auto& candidate : catalog.at("candidates")) {
    const auto id = candidate.at("candidate_id").get<std::string>();
    if (operator_recommend135_v1::candidate_eligible(id, block))
      result.push_back(id);
  }
  return result;
}

}  // namespace

int main() {
  static_assert(std::endian::native == std::endian::little);
  const auto catalog = operator_recommend135_v1::candidate_catalog();
  assert(catalog.at("schema") == "operator-recommend135-candidate-catalog-v1");
  assert(catalog.at("candidates").is_array());
  assert(catalog.at("candidates").size() >= 20);
  assert(catalog.at("candidates").size() <= 30);

  std::set<std::string> ids, kinds;
  for (const auto& candidate : catalog.at("candidates")) {
    for (const auto* field : {"candidate_id", "kind", "component_ids",
                              "action_ids", "terminal_codecs",
                              "supported_dtypes", "recipe"})
      assert(candidate.contains(field));
    assert(ids.emplace(candidate.at("candidate_id").get<std::string>()).second);
    kinds.emplace(candidate.at("kind").get<std::string>());
    const auto dumped = candidate.dump();
    for (const auto* denied : {"A106", "A107", "A108", "A109", "A110",
                               "nncp", "lstm", "ppmd", "dmc", "paq", "cmix"})
      assert(dumped.find(denied) == std::string::npos);
  }
  assert(kinds == std::set<std::string>({"algorithm", "algorithm_combination",
                                        "operator", "operator_tree", "raw"}));

  // PreparedRegistry must preserve the frozen catalog and produce exactly the
  // old per-candidate eligibility result after one validation of the block.
  // This catches rebuilding/consulting JSON metadata in the hot eligibility
  // loop and accidental dtype or empty-domain changes.
  const PreparedRegistry prepared_registry;
  assert(prepared_registry.catalog() == catalog);

  const Block empty{"int64", 0, {}, {}};
  for (const auto* id : {"raw", "zstd1", "zstd3", "zstd9", "lz4frame", "zlib6"})
    roundtrip(id, empty);

  const auto i32 = int32_fixture();
  const auto i64 = int64_fixture();
  const auto f32 = float32_fixture();
  const auto f64 = float64_fixture();
  const std::vector<Block> prepared_fixtures{
      i32,
      i64,
      f32,
      f64,
      Block{"int32", 0, {}, {}},
      Block{"int64", 0, {}, {}},
      Block{"float32", 0, {}, {}},
      Block{"float64", 0, {}, {}},
  };
  for (const auto& block : prepared_fixtures) {
    const auto prepared = prepared_registry.prepare(block);
    assert(prepared.validation_passes == 1);
    assert(prepared.block.dtype == block.dtype);
    assert(prepared.block.row_count == block.row_count);
    assert(prepared.block.value_count * prepared.block.dtype_width ==
           block.values.size());
    assert(prepared.block.value_bytes == block.values.size());
    assert(prepared.block.missing_bytes == block.missing.size());
    assert(prepared.candidate_ids == reference_eligible(catalog, block));
  }
  for (const auto& candidate : catalog.at("candidates")) {
    const auto id = candidate.at("candidate_id").get<std::string>();
    const auto dtypes = candidate.at("supported_dtypes").get<std::vector<std::string>>();
    const auto& block = fixture_for(dtypes, i32, i64, f32, f64);
    roundtrip(id, block);
  }

  // The float32 path preserves the four-byte source words, including NaN
  // payloads and signed zero; no double conversion is permitted.
  for (const auto* id : {"raw", "zstd3", "columnize_zstd3",
                         "float_deconstruct_zstd3", "split_zstd3_lz4"})
    roundtrip(id, f32);
  for (const auto* id : {"raw", "zstd3", "xor_zstd3", "a048_planes_zstd3_lz4"})
    roundtrip(id, f64);

  assert(!operator_recommend135_v1::candidate_eligible("for_bitpack_zstd3", f32));
  assert(!operator_recommend135_v1::candidate_eligible("float_deconstruct_zstd3", i64));
  assert(!operator_recommend135_v1::candidate_eligible("huffman_a077", empty));
  assert(!operator_recommend135_v1::candidate_eligible("does-not-exist", i64));

  Block malformed = i64;
  malformed.values.pop_back();
  assert(!operator_recommend135_v1::candidate_eligible("raw", malformed));
  rejected([&] { (void)operator_recommend135_v1::encode_candidate("raw", malformed); });
  malformed = i64;
  malformed.missing.front() |= 0b10000000;  // unused high row bit
  assert(!operator_recommend135_v1::candidate_eligible("raw", malformed));
  malformed = i64;
  malformed.dtype = "float";
  assert(!operator_recommend135_v1::candidate_eligible("raw", malformed));
  rejected([&] { (void)prepared_registry.eligible_candidates(malformed); });
  malformed = i64;
  malformed.missing.clear();
  rejected([&] { (void)prepared_registry.eligible_candidates(malformed); });
  malformed = i64;
  malformed.missing.front() |= 0b10000000;
  rejected([&] { (void)prepared_registry.eligible_candidates(malformed); });
  malformed = i64;
  malformed.values.pop_back();
  rejected([&] { (void)prepared_registry.eligible_candidates(malformed); });
  for (const auto* id : {"A106", "A107", "A108", "A109", "A110", "nncp"})
    rejected([&] { (void)operator_recommend135_v1::encode_candidate(id, i64); });

  auto corrupt = operator_recommend135_v1::encode_candidate("zstd3", i64);
  corrupt[corrupt.size() / 2] ^= 0x80;
  rejected([&] { (void)operator_recommend135_v1::decode_candidate(corrupt); });

  // Recomputing the outer NGF1 checksum does not make corrupted codec bytes
  // acceptable: the terminal wire and restored root digest are also checked.
  auto inner_corrupt = operator_recommend135_v1::encode_candidate("zstd3", i64);
  auto parsed = native_graph_v1::detail::parse_frame(inner_corrupt, {});
  assert(parsed.leaves.contains(3) && !parsed.leaves.at(3).data.empty());
  parsed.leaves.at(3).data.back() ^= 0x01;
  inner_corrupt = native_graph_v1::detail::serialize_frame(parsed, {});
  rejected([&] { (void)operator_recommend135_v1::decode_candidate(inner_corrupt); });

  // A validly checksummed frame still fails if its candidate binding and
  // actual action recipe disagree.
  auto mismatched = operator_recommend135_v1::encode_candidate("zstd1", i64);
  parsed = native_graph_v1::detail::parse_frame(mismatched, {});
  assert(parsed.nodes.size() == 1);
  parsed.nodes[0].step.stage.action = "recommend135.codec.zstd.level3.v1";
  mismatched = native_graph_v1::detail::serialize_frame(parsed, {});
  rejected([&] { (void)operator_recommend135_v1::decode_candidate(mismatched); });

  // Denied families/action IDs are rejected at the cold-decode preflight,
  // before their inverse can be dispatched.
  const std::vector<std::pair<const char*, std::int64_t>> forbidden{
      {"op.prediction_probability.ppm.provider_v1", 6225},
      {"op.prediction_probability.dmc.provider_v1", 6226},
      {"op.prediction_probability.paq.provider_v1", 6227},
      {"op.prediction_probability.cmix.provider_v1", 6228},
      {"op.prediction_probability.nncp.provider_v1", 6229}};
  for (const auto& [action, action_id] : forbidden) {
    auto denied = operator_recommend135_v1::encode_candidate("zstd3", i64);
    parsed = native_graph_v1::detail::parse_frame(denied, {});
    parsed.nodes[0].step.stage.action = action;
    parsed.nodes[0].step.policy_action_id = action_id;
    denied = native_graph_v1::detail::serialize_frame(parsed, {});
    rejected([&] { (void)operator_recommend135_v1::decode_candidate(denied); });
  }

  const auto audit =
      operator_recommend135_v1::inspect_candidate_run("split_zstd3_lz4", f32);
  assert(audit.at("schema") == "operator-recommend135-candidate-run-v1");
  assert(audit.at("execution_status") == "strict_ready");
  assert(audit.at("fallback_used") == false);
  assert(audit.at("unified_internal_fallback_audit") == false);
  assert(audit.at("roundtrip").at("decoded_from_frame_only") == true);
  assert(audit.at("roundtrip").at("values_bit_exact") == true);
  assert(audit.at("roundtrip").at("missing_bit_exact") == true);
  assert(audit.at("roundtrip").at("dtype_exact") == true);
  assert(audit.at("roundtrip").at("row_count_exact") == true);

  std::cout << "operator_recommend135_v1 runtime tests passed\n";
}
