#include "grouping_codec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace grouping_codec {
namespace {

constexpr std::uint64_t kTimeBytes = 8;
constexpr std::uint64_t kValueBytes = 8;

struct GroupedLayout {
  std::vector<std::uint64_t> timestamps;
  std::vector<std::uint64_t> bitmap;
  std::vector<std::vector<std::uint64_t>> values;
};

std::string trimCell(std::string value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.erase(value.begin());
  }
  while (!value.empty() &&
         (value.back() == ' ' || value.back() == '\t' || value.back() == '\r')) {
    value.pop_back();
  }
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

GroupedLayout groupLayout(const std::vector<ColumnStream>& columns) {
  GroupedLayout out;
  out.values.resize(columns.size());
  if (columns.empty()) {
    return out;
  }
  if (columns.size() == 1) {
    out.timestamps = columns[0].timestamps;
    out.values[0] = columns[0].values;
    return out;
  }

  std::vector<std::size_t> pos(columns.size(), 0);
  while (true) {
    std::uint64_t next = std::numeric_limits<std::uint64_t>::max();
    bool found = false;
    for (std::size_t i = 0; i < columns.size(); ++i) {
      if (pos[i] < columns[i].timestamps.size()) {
        next = std::min(next, columns[i].timestamps[pos[i]]);
        found = true;
      }
    }
    if (!found) {
      break;
    }

    const std::size_t row = out.timestamps.size();
    out.timestamps.push_back(next);
    const std::size_t need_words = (out.timestamps.size() * columns.size() + 63) / 64;
    while (out.bitmap.size() < need_words) {
      out.bitmap.push_back(0);
    }
    for (std::size_t col = 0; col < columns.size(); ++col) {
      if (pos[col] >= columns[col].timestamps.size() ||
          columns[col].timestamps[pos[col]] != next) {
        continue;
      }
      const std::size_t bit = row * columns.size() + col;
      out.bitmap[bit / 64] |= (std::uint64_t{1} << (bit % 64));
      out.values[col].push_back(columns[col].values[pos[col]]);
      pos[col]++;
    }
  }
  return out;
}

std::vector<ColumnStream> recoverLayout(const GroupedLayout& layout) {
  std::vector<ColumnStream> out(layout.values.size());
  if (layout.values.empty()) {
    return out;
  }
  if (layout.values.size() == 1) {
    out[0].timestamps = layout.timestamps;
    out[0].values = layout.values[0];
    return out;
  }

  std::vector<std::size_t> pos(layout.values.size(), 0);
  for (std::size_t row = 0; row < layout.timestamps.size(); ++row) {
    const std::uint64_t ts = layout.timestamps[row];
    for (std::size_t col = 0; col < layout.values.size(); ++col) {
      const std::size_t bit = row * layout.values.size() + col;
      if (((layout.bitmap[bit / 64] >> (bit % 64)) & 1U) == 0) {
        continue;
      }
      const std::size_t value_pos = pos[col]++;
      out[col].timestamps.push_back(ts);
      out[col].values.push_back(layout.values[col][value_pos]);
    }
  }
  return out;
}

std::uint64_t originalSize(const std::vector<ColumnStream>& columns) {
  std::uint64_t points = 0;
  for (const auto& col : columns) {
    points += col.values.size();
  }
  return points * (kTimeBytes + kValueBytes);
}

std::uint64_t groupedSize(const GroupedLayout& layout) {
  std::uint64_t values = 0;
  for (const auto& col : layout.values) {
    values += col.size();
  }
  const std::uint64_t bitmap_bytes =
      layout.values.size() <= 1
          ? 0
          : (layout.timestamps.size() * static_cast<std::uint64_t>(layout.values.size()) + 7) / 8;
  return layout.timestamps.size() * kTimeBytes + values * kValueBytes + bitmap_bytes;
}

void verifyRoundTrip(const std::vector<ColumnStream>& want,
                     const std::vector<ColumnStream>& got) {
  if (want.size() != got.size()) {
    throw std::runtime_error("grouping column count mismatch");
  }
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i].timestamps != got[i].timestamps || want[i].values != got[i].values) {
      throw std::runtime_error("grouping lossless check failed");
    }
  }
}

}  // namespace

ColumnStream intColumnStream(const std::vector<std::string>& cells, std::uint64_t row_offset) {
  ColumnStream out;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    const std::string t = trimCell(cells[i]);
    if (t.empty()) {
      continue;
    }
    try {
      std::int64_t value = std::stoll(t);
      out.timestamps.push_back(row_offset + static_cast<std::uint64_t>(i));
      out.values.push_back(static_cast<std::uint64_t>(value));
    } catch (...) {
    }
  }
  return out;
}

ColumnStream doubleColumnStream(const std::vector<std::string>& cells, std::uint64_t row_offset) {
  ColumnStream out;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    const std::string t = trimCell(cells[i]);
    if (t.empty()) {
      continue;
    }
    try {
      const double value = std::stod(t);
      if (!std::isfinite(value)) {
        continue;
      }
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      out.timestamps.push_back(row_offset + static_cast<std::uint64_t>(i));
      out.values.push_back(bits);
    } catch (...) {
    }
  }
  return out;
}

void benchStreams(const std::vector<ColumnStream>& raw_columns, std::vector<long long>& result) {
  result.assign(4, 0);
  std::vector<ColumnStream> columns;
  columns.reserve(raw_columns.size());
  for (const auto& col : raw_columns) {
    if (!col.values.empty()) {
      columns.push_back(col);
    }
  }
  const std::uint64_t original = originalSize(columns);
  if (original == 0) {
    return;
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  const GroupedLayout layout = groupLayout(columns);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const std::vector<ColumnStream> recovered = recoverLayout(layout);
  const auto t2 = std::chrono::high_resolution_clock::now();
  verifyRoundTrip(columns, recovered);

  result[0] = static_cast<long long>(original);
  result[1] = static_cast<long long>(groupedSize(layout));
  result[2] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  result[3] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

}  // namespace grouping_codec
