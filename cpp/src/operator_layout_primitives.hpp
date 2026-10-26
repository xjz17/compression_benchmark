#pragma once

#include "adjacent_residual.hpp"

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace exact_actions::operator_layout {
using Bytes = std::vector<std::uint8_t>;

inline std::uint64_t low_mask(unsigned width) {
  if (width == 0 || width > 64) throw std::invalid_argument("invalid bit field width");
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

// Fields are least-significant first. Carrier type and signed views are separate.
class BitFields {
 public:
  explicit BitFields(std::span<const std::uint8_t> widths, unsigned word_bits = 64)
      : widths_(widths.begin(), widths.end()) {
    if (widths.empty() || widths.size() > 64 || word_bits == 0 || word_bits > 64)
      throw std::invalid_argument("invalid bit field layout");
    for (auto width : widths) {
      masks_.push_back(low_mask(width));
      offsets_.push_back(total_);
      total_ += width;
      if (total_ > word_bits) throw std::invalid_argument("bit fields exceed word width");
    }
  }
  BitFields(std::initializer_list<std::uint8_t> widths, unsigned word_bits = 64)
      : BitFields(std::span<const std::uint8_t>(widths.begin(), widths.size()), word_bits) {}

  unsigned total_width() const { return total_; }
  std::uint64_t field(std::uint64_t word, std::size_t index) const {
    return (word >> offsets_.at(index)) & masks_.at(index);
  }
  std::uint64_t join(std::span<const std::uint64_t> fields) const {
    if (fields.size() != widths_.size()) throw std::invalid_argument("bit field count mismatch");
    std::uint64_t word = 0;
    for (std::size_t i = 0; i < fields.size(); ++i) {
      if (fields[i] > masks_[i]) throw std::invalid_argument("bit field value exceeds width");
      word |= fields[i] << offsets_[i];
    }
    return word;
  }

 private:
  std::vector<std::uint8_t> widths_;
  std::vector<unsigned> offsets_;
  std::vector<std::uint64_t> masks_;
  unsigned total_ = 0;
};

inline std::int64_t sign_extend(std::uint64_t value, unsigned width) {
  const auto mask = low_mask(width);
  if (value > mask) throw std::invalid_argument("signed field value exceeds width");
  if (width < 64 && (value & (std::uint64_t{1} << (width - 1)))) value |= ~mask;
  return std::bit_cast<std::int64_t>(value);
}

inline Bytes columnize(std::span<const std::uint8_t> records, std::size_t width) {
  if (width == 0 || records.size() % width != 0)
    throw std::invalid_argument("columnize requires complete fixed-width records");
  const auto rows = records.size() / width;
  Bytes columns(records.size());
  for (std::size_t column = 0; column < width; ++column)
    for (std::size_t row = 0; row < rows; ++row)
      columns[column * rows + row] = records[row * width + column];
  return columns;
}

inline Bytes uncolumnize(std::span<const std::uint8_t> columns, std::size_t width) {
  if (width == 0 || columns.size() % width != 0)
    throw std::invalid_argument("uncolumnize requires complete fixed-width columns");
  const auto rows = columns.size() / width;
  Bytes records(columns.size());
  for (std::size_t column = 0; column < width; ++column)
    for (std::size_t row = 0; row < rows; ++row)
      records[row * width + column] = columns[column * rows + row];
  return records;
}

// SPDP concatenates columns before Delta; state deliberately crosses columns.
inline Bytes column_delta(std::span<const std::uint8_t> records, std::size_t width) {
  auto columns = columnize(records, width);
  std::uint8_t previous = 0;
  for (auto& byte : columns) {
    const auto current = byte;
    byte = adjacent_residual::delta_word(current, previous);
    previous = current;
  }
  return columns;
}

inline Bytes inverse_column_delta(std::span<const std::uint8_t> residuals,
                                  std::size_t width) {
  Bytes columns(residuals.begin(), residuals.end());
  std::uint8_t previous = 0;
  for (auto& byte : columns) byte = previous = static_cast<std::uint8_t>(byte + previous);
  return uncolumnize(columns, width);
}
}  // namespace exact_actions::operator_layout
