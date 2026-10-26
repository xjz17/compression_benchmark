#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace grouping_codec {

inline constexpr const char* kRowName = "grouping";

struct ColumnStream {
  std::vector<std::uint64_t> timestamps;
  std::vector<std::uint64_t> values;
};

ColumnStream intColumnStream(const std::vector<std::string>& cells, std::uint64_t row_offset);

ColumnStream doubleColumnStream(const std::vector<std::string>& cells, std::uint64_t row_offset);

void benchStreams(const std::vector<ColumnStream>& columns, std::vector<long long>& result);

}  // namespace grouping_codec
