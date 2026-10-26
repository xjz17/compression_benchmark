#include "dictionary_recode.hpp"
#include "max_value_bitpack.hpp"
#include "native_lz_action_stages.hpp"
#include "openzl_layout_stages.hpp"
#include "operator_layout_primitives.hpp"
#include "reference_operator_actions.hpp"
#include "spdp_exact_stages.hpp"

#include <array>
#include <bit>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using Bytes = std::vector<std::uint8_t>;
std::size_t checks = 0;
void check(bool condition, const char* label) {
  ++checks;
  if (!condition) throw std::runtime_error(label);
}
void rejects(const std::function<void()>& fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  check(rejected, "malformed input accepted");
}
void append_word(Bytes& out, std::uint64_t word, unsigned width = 8) {
  for (unsigned byte = 0; byte < width; ++byte) out.push_back(word >> (byte * 8));
}
std::uint64_t read_word(const Bytes& bytes, std::size_t pos, unsigned width = 8) {
  std::uint64_t word = 0;
  for (unsigned byte = 0; byte < width; ++byte) word |= std::uint64_t(bytes.at(pos + byte)) << (byte * 8);
  return word;
}

void split_recipes(const std::vector<std::uint64_t>& words) {
  Bytes raw, expected_high, expected_low, expected_hi32, expected_lo32;
  std::vector<std::uint64_t> low_values;
  for (const auto word : words) {
    append_word(raw, word);
    append_word(expected_high, std::bit_cast<std::uint64_t>(std::bit_cast<std::int64_t>(word) >> 4));
    append_word(expected_low, word & 15, 1);
    append_word(expected_lo32, word & UINT32_MAX, 4);
    append_word(expected_hi32, word >> 32, 4);
    low_values.push_back(word & 15);
  }
  const openzl_layout::Stream input{openzl_layout::Kind::Numeric, 8, raw};
  const std::uint8_t low4_widths[]{4,60};
  const auto split = openzl_layout::bitsplit(input, low4_widths);
  check(split.streams[0].data == expected_low, "A104 low4 profile");
  check(openzl_layout::inverse(split)[0].data == raw, "A104 low4 inverse");
  for (std::size_t i = 0; i < words.size(); ++i) {
    const auto high = exact_actions::operator_layout::sign_extend(read_word(split.streams[1].data, i*8), 60);
    check(std::bit_cast<std::uint64_t>(high) == read_word(expected_high, i*8), "signed high profile");
  }
  const std::uint8_t limb_widths[]{32,32};
  const auto limbs = openzl_layout::bitsplit(input, limb_widths);
  check(limbs.streams[0].data == expected_lo32 && limbs.streams[1].data == expected_hi32, "A051 limb profile");
  check(openzl_layout::inverse(limbs)[0].data == raw, "A051 limb inverse");

  const reference_operator_actions::Names eraser{"op.eraser.reference_v1"};
  check(reference_operator_actions::final_values(raw, eraser) == expected_high, "A001 signed high");
  check(reference_operator_actions::inverse_serialized(reference_operator_actions::serialized(raw, eraser), eraser) == raw,
        "A001 legacy metadata inverse");
  Bytes nibble_values;
  for (std::size_t i = 0; i < words.size(); i += 2)
    append_word(nibble_values, ((words[i] & 15) << 4) | (i+1 < words.size() ? words[i+1] & 15 : 0));
  for (const auto* action : {"op.match_token_format.reference_v1", "op.bitstream_pack.reference_v1"}) {
    const reference_operator_actions::Names actions{action};
    check(reference_operator_actions::final_values(raw, actions) == nibble_values, "A066 low4 MSB packing");
    check(reference_operator_actions::inverse_serialized(reference_operator_actions::serialized(raw, actions), actions) == raw,
          "A066 signed high and odd tail inverse");
  }
}

void dictionary_profiles() {
  using namespace exact_actions::dictionary_recode;
  const std::vector<std::int64_t> input{5,-1,5,3,-1,2};
  const std::array orders{Order::value, Order::first_seen, Order::frequency};
  const std::array<std::vector<std::int64_t>,3> expected{{{-1,2,3,5},{5,-1,3,2},{-1,5,2,3}}};
  for (std::size_t i = 0; i < orders.size(); ++i) {
    const auto encoded = encode<std::int64_t>(input, orders[i]);
    check(encoded.dictionary == expected[i], "dictionary ordering and frequency ties");
    check(decode<std::int64_t>(encoded.dictionary, encoded.ids) == input, "dictionary inverse");
  }
  const auto empty = encode<std::uint64_t>({}, Order::first_seen);
  check(empty.dictionary.empty() && empty.ids.empty(), "empty dictionary");
  rejects([&] { decode<std::int64_t>(input, std::array<std::uint64_t,1>{input.size()}); });
  const std::vector<std::uint64_t> words{0,UINT64_C(0x8000000000000000),UINT64_C(0x7ff8000000001234),0,UINT64_C(0x7ff8000000005678)};
  const auto encoded = encode<std::uint64_t>(words, Order::first_seen);
  const auto packed = exact_actions::max_value_bitpack::pack_fixed(encoded.ids, 2);
  check(decode<std::uint64_t>(encoded.dictionary,
            exact_actions::max_value_bitpack::unpack(packed.payload, words.size(), 2)) == words,
        "A043 dictionary plus bitpack preserves floating bit patterns");
}

void spdp_recipe() {
  using namespace exact_actions::operator_layout;
  Bytes raw;
  for (unsigned i = 1; i <= 16; ++i) raw.push_back(i);
  const Bytes expected{1,8,249,8,249,8,249,8,249,8,249,8,249,8,249,8};
  check(column_delta(raw, 8) == expected, "SPDP state crosses column boundaries");
  check(inverse_column_delta(expected, 8) == raw, "SPDP recipe inverse");
  Bytes packet{'S','2','D','1'}; append_word(packet, raw.size()); packet.insert(packet.end(),raw.begin(),raw.end());
  const auto transformed = exact_actions::spdp::block_transform_forward(packet);
  check(Bytes(transformed.begin()+12,transformed.end()) == expected, "A047 invokes column Delta recipe");
  check(exact_actions::spdp::block_transform_inverse(transformed) == packet, "A047 carrier inverse");
  const auto split = openzl_layout::transpose({openzl_layout::Kind::Struct,8,raw});
  Bytes columns;
  for (const auto& stream : split.streams) columns.insert(columns.end(),stream.data.begin(),stream.data.end());
  check(columns == columnize(raw,8), "A049 shares column core");
  check(openzl_layout::inverse(split)[0].data == raw, "A049 inverse");
  rejects([&] { columnize(raw,3); });
}

void configured_lz() {
  Bytes raw(257);
  for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = i % 13;
  using namespace native_lz_action;
  for (const auto window : {3072U,65535U}) {
    const SearchConfiguration configuration{window,3,258};
    const auto w = attach_window(raw, window);
    const auto m = attach_match_limits(w,3,258);
    const auto p = forward("op.match_parse_policy.native_lz.aligned8_greedy",m);
    check(p == configured_parse_packet(raw,configuration), "A057/A065 configuration-only recipe");
    check(w == forward("op.match_window_model.native_lz.window"+std::to_string(window),raw), "A065 legacy adapter");
    check(m == forward("op.lz_match.native_lz.aligned8.min3.max258",w), "A057 legacy adapter");
    const auto tokens = parse(raw,configuration);
    check(expand(tokens,raw.size()) == raw, "configured LZ inverse");
  }
  rejects([&] { configured_parse_packet(raw,{0,3,258}); });
  rejects([&] { attach_match_limits(raw,3,258); });
}
}  // namespace

int main() try {
  std::mt19937_64 random(20260917);
  split_recipes({});
  split_recipes({0,UINT64_MAX,UINT64_C(0x8000000000000000),UINT64_C(0x7fffffffffffffff),17});
  for (std::size_t size = 1; size <= 65; ++size) {
    std::vector<std::uint64_t> words(size);
    for (auto& word : words) word = random();
    split_recipes(words);
  }
  dictionary_profiles(); spdp_recipe(); configured_lz();
  rejects([] { exact_actions::operator_layout::BitFields invalid{0,64}; });
  rejects([] { exact_actions::operator_layout::BitFields invalid{32,33}; });
  rejects([] { exact_actions::operator_layout::sign_extend(16,4); });
  std::cout << "operator_consolidation checks=" << checks << " status=PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
