#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace exact_actions::dictionary_recode {
enum class Order { first_seen, value, frequency };

template <std::integral Word>
struct Encoded {
  std::vector<Word> dictionary;
  std::vector<std::uint64_t> ids;
};

template <std::integral Word>
Encoded<Word> encode(std::span<const Word> values, Order order) {
  if (order != Order::first_seen && order != Order::value && order != Order::frequency)
    throw std::invalid_argument("invalid dictionary order");
  Encoded<Word> out;
  std::unordered_map<Word, std::uint64_t> counts;
  for (const auto value : values) {
    auto [entry, inserted] = counts.emplace(value, 0);
    ++entry->second;
    if (inserted) out.dictionary.push_back(value);
  }
  if (order != Order::first_seen) {
    std::sort(out.dictionary.begin(), out.dictionary.end());
    if (order == Order::frequency)
      std::stable_sort(out.dictionary.begin(), out.dictionary.end(),
          [&](Word a, Word b) { return counts.at(a) > counts.at(b); });
  }
  std::unordered_map<Word, std::uint64_t> ids;
  for (std::size_t i = 0; i < out.dictionary.size(); ++i) ids.emplace(out.dictionary[i], i);
  out.ids.reserve(values.size());
  for (const auto value : values) out.ids.push_back(ids.at(value));
  return out;
}

template <std::integral Word>
std::vector<Word> decode(std::span<const Word> dictionary,
                         std::span<const std::uint64_t> ids) {
  std::vector<Word> out;
  out.reserve(ids.size());
  for (const auto id : ids) {
    if (id >= dictionary.size()) throw std::invalid_argument("dictionary id out of range");
    out.push_back(dictionary[id]);
  }
  return out;
}
}  // namespace exact_actions::dictionary_recode
