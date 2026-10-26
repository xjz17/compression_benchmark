#include "bos_operator_ablation.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bos_operator_ablation {
namespace {

constexpr std::uint8_t kMagic[] = {'B', 'O', 'S', 'A'};
constexpr std::uint8_t kDeltaFlag = 1U << 0;
constexpr std::uint8_t kForFlag = 1U << 1;
constexpr std::uint8_t kOutlierFlag = 1U << 2;
constexpr std::uint8_t kBitpackFlag = 1U << 3;

std::uint64_t to_u64(std::int64_t value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::int64_t to_i64(std::uint64_t value) {
  std::int64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t zigzag_encode(std::int64_t value) {
  return (to_u64(value) << 1) ^
         (value < 0 ? std::numeric_limits<std::uint64_t>::max() : 0U);
}

std::int64_t zigzag_decode(std::uint64_t value) {
  return to_i64((value >> 1) ^ (std::uint64_t{0} - (value & 1U)));
}

int bit_width(std::uint64_t value) {
  return value == 0 ? 0 : 64 - __builtin_clzll(value);
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

std::uint32_t get_u32(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 4 > in.size()) throw std::runtime_error("BOS ablation truncated uint32");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(in[pos++]) << shift;
  }
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> in, std::size_t& pos) {
  if (pos + 8 > in.size()) throw std::runtime_error("BOS ablation truncated uint64");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(in[pos++]) << shift;
  }
  return value;
}

class BitWriter {
 public:
  void write(std::uint64_t value, int width) {
    for (int bit = width - 1; bit >= 0; --bit) {
      current_ = static_cast<std::uint8_t>((current_ << 1) | ((value >> bit) & 1U));
      if (++used_ == 8) {
        bytes_.push_back(current_);
        current_ = 0;
        used_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> finish() {
    if (used_ != 0) bytes_.push_back(static_cast<std::uint8_t>(current_ << (8 - used_)));
    return std::move(bytes_);
  }

 private:
  std::vector<std::uint8_t> bytes_;
  std::uint8_t current_ = 0;
  int used_ = 0;
};

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  std::uint64_t read(int width) {
    std::uint64_t value = 0;
    for (int bit = 0; bit < width; ++bit) {
      if (position_ >= bytes_.size() * 8) {
        throw std::runtime_error("BOS ablation packed body truncated");
      }
      value = (value << 1) |
              ((bytes_[position_ / 8] >> (7 - position_ % 8)) & 1U);
      ++position_;
    }
    return value;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t position_ = 0;
};

struct NormalRange {
  std::uint64_t base = 0;
  int width = 0;
  std::size_t count = 0;
};

NormalRange select_normal_range(const std::vector<std::uint64_t>& codes,
                                bool use_for, bool use_outliers,
                                bool use_bitpacking) {
  if (codes.empty()) return {};
  const auto [minimum, maximum] = std::minmax_element(codes.begin(), codes.end());
  NormalRange best{use_for ? *minimum : 0U,
                   bit_width(*maximum - (use_for ? *minimum : 0U)), codes.size()};
  if (!use_outliers || !use_bitpacking) return best;

  const std::size_t dense_bits = best.count * static_cast<std::size_t>(best.width);
  std::size_t best_bits = dense_bits;
  std::vector<std::uint64_t> sorted = codes;
  std::sort(sorted.begin(), sorted.end());
  for (int width = 0; width < 64; ++width) {
    const std::uint64_t limit = width == 0 ? 0U : (std::uint64_t{1} << width) - 1U;
    if (!use_for) {
      const auto end = std::upper_bound(sorted.begin(), sorted.end(), limit);
      const std::size_t normal_count = static_cast<std::size_t>(end - sorted.begin());
      const std::size_t outlier_count = sorted.size() - normal_count;
      const std::size_t bits = normal_count * static_cast<std::size_t>(width) +
                               outlier_count * 96U;
      if (normal_count != 0 && bits < best_bits) {
        best = {0U, width, normal_count};
        best_bits = bits;
      }
      continue;
    }

    std::size_t left = 0;
    for (std::size_t right = 0; right < sorted.size(); ++right) {
      while (sorted[right] - sorted[left] > limit) ++left;
      const std::size_t normal_count = right - left + 1;
      const std::size_t outlier_count = sorted.size() - normal_count;
      const std::size_t bits = normal_count * static_cast<std::size_t>(width) +
                               outlier_count * 96U;
      if (bits < best_bits) {
        best = {sorted[left], width, normal_count};
        best_bits = bits;
      }
    }
  }
  return best;
}

bool in_normal_range(std::uint64_t code, const NormalRange& range) {
  if (code < range.base) return false;
  if (range.width == 64) return true;
  const std::uint64_t limit = range.width == 0
                                  ? 0U
                                  : (std::uint64_t{1} << range.width) - 1U;
  return code - range.base <= limit;
}

}  // namespace

std::vector<std::uint8_t> encode(std::span<const std::int64_t> values,
                                 std::string_view omitted_operator) {
  const bool use_delta = omitted_operator != "delta";
  const bool use_for = omitted_operator != "for";
  const bool use_outliers = omitted_operator != "outlier_separation";
  const bool use_bitpacking = omitted_operator != "bit-packing";
  if (!omitted_operator.empty() && omitted_operator != "delta" &&
      omitted_operator != "for" && omitted_operator != "outlier_separation" &&
      omitted_operator != "bit-packing") {
    throw std::invalid_argument("unsupported BOS ablation operator");
  }
  if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("BOS ablation column exceeds uint32 row count");
  }

  std::vector<std::int64_t> source;
  source.reserve(use_delta && !values.empty() ? values.size() - 1 : values.size());
  if (use_delta) {
    for (std::size_t i = 1; i < values.size(); ++i) {
      source.push_back(to_i64(to_u64(values[i]) - to_u64(values[i - 1])));
    }
  } else {
    source.assign(values.begin(), values.end());
  }

  const std::int64_t source_base = use_for && !source.empty()
      ? *std::min_element(source.begin(), source.end())
      : 0;
  std::vector<std::uint64_t> codes;
  codes.reserve(source.size());
  for (std::int64_t value : source) {
    codes.push_back(use_for ? to_u64(value) - to_u64(source_base) : zigzag_encode(value));
  }
  const NormalRange range =
      select_normal_range(codes, use_for, use_outliers, use_bitpacking);

  std::vector<std::pair<std::uint32_t, std::uint64_t>> outliers;
  std::vector<std::uint64_t> normal;
  normal.reserve(range.count);
  for (std::size_t i = 0; i < codes.size(); ++i) {
    if (in_normal_range(codes[i], range)) {
      normal.push_back(codes[i] - range.base);
    } else {
      outliers.emplace_back(static_cast<std::uint32_t>(i), codes[i]);
    }
  }

  std::vector<std::uint8_t> body;
  if (use_bitpacking) {
    BitWriter writer;
    for (std::uint64_t value : normal) writer.write(value, range.width);
    body = writer.finish();
  } else {
    body.reserve(normal.size() * sizeof(std::uint64_t));
    for (std::uint64_t value : normal) put_u64(body, value);
  }

  std::vector<std::uint8_t> out(std::begin(kMagic), std::end(kMagic));
  put_u32(out, static_cast<std::uint32_t>(values.size()));
  std::uint8_t flags = 0;
  if (use_delta) flags |= kDeltaFlag;
  if (use_for) flags |= kForFlag;
  if (use_outliers) flags |= kOutlierFlag;
  if (use_bitpacking) flags |= kBitpackFlag;
  out.push_back(flags);
  put_u64(out, values.empty() ? 0U : to_u64(values.front()));
  put_u64(out, to_u64(source_base));
  put_u64(out, range.base);
  out.push_back(static_cast<std::uint8_t>(range.width));
  put_u32(out, static_cast<std::uint32_t>(outliers.size()));
  put_u32(out, static_cast<std::uint32_t>(body.size()));
  for (const auto& [position, value] : outliers) {
    put_u32(out, position);
    put_u64(out, value);
  }
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::int64_t> decode(std::span<const std::uint8_t> encoded) {
  if (encoded.size() < sizeof(kMagic) ||
      !std::equal(std::begin(kMagic), std::end(kMagic), encoded.begin())) {
    throw std::runtime_error("BOS ablation invalid magic");
  }
  std::size_t pos = sizeof(kMagic);
  const std::uint32_t count = get_u32(encoded, pos);
  if (pos >= encoded.size()) throw std::runtime_error("BOS ablation flags truncated");
  const std::uint8_t flags = encoded[pos++];
  const bool use_delta = (flags & kDeltaFlag) != 0;
  const bool use_for = (flags & kForFlag) != 0;
  const bool use_bitpacking = (flags & kBitpackFlag) != 0;
  const std::int64_t first = to_i64(get_u64(encoded, pos));
  const std::int64_t source_base = to_i64(get_u64(encoded, pos));
  const std::uint64_t normal_base = get_u64(encoded, pos);
  if (pos >= encoded.size()) throw std::runtime_error("BOS ablation width truncated");
  const int width = encoded[pos++];
  if (width < 0 || width > 64) throw std::runtime_error("BOS ablation invalid width");
  const std::uint32_t outlier_count = get_u32(encoded, pos);
  const std::uint32_t body_size = get_u32(encoded, pos);
  const std::size_t source_count = use_delta ? (count == 0 ? 0 : count - 1) : count;
  if (outlier_count > source_count) throw std::runtime_error("BOS ablation invalid outlier count");

  std::vector<std::pair<std::uint32_t, std::uint64_t>> outliers;
  outliers.reserve(outlier_count);
  for (std::uint32_t i = 0; i < outlier_count; ++i) {
    const std::uint32_t index = get_u32(encoded, pos);
    const std::uint64_t value = get_u64(encoded, pos);
    if (index >= source_count || (!outliers.empty() && index <= outliers.back().first)) {
      throw std::runtime_error("BOS ablation invalid outlier position");
    }
    outliers.emplace_back(index, value);
  }
  if (pos + body_size != encoded.size()) throw std::runtime_error("BOS ablation body size mismatch");
  const auto body = encoded.subspan(pos, body_size);
  const std::size_t normal_count = source_count - outliers.size();
  if (!use_bitpacking && body.size() != normal_count * sizeof(std::uint64_t)) {
    throw std::runtime_error("BOS ablation raw body size mismatch");
  }
  if (use_bitpacking && body.size() != (normal_count * static_cast<std::size_t>(width) + 7) / 8) {
    throw std::runtime_error("BOS ablation packed body size mismatch");
  }

  BitReader reader(body);
  std::size_t raw_pos = 0;
  std::size_t outlier_pos = 0;
  std::vector<std::int64_t> source;
  source.reserve(source_count);
  for (std::size_t i = 0; i < source_count; ++i) {
    std::uint64_t code = 0;
    if (outlier_pos < outliers.size() && outliers[outlier_pos].first == i) {
      code = outliers[outlier_pos++].second;
    } else if (use_bitpacking) {
      code = normal_base + reader.read(width);
    } else {
      code = normal_base + get_u64(body, raw_pos);
    }
    source.push_back(use_for ? to_i64(to_u64(source_base) + code) : zigzag_decode(code));
  }

  std::vector<std::int64_t> values;
  values.reserve(count);
  if (use_delta) {
    if (count != 0) values.push_back(first);
    for (std::int64_t residual : source) {
      values.push_back(to_i64(to_u64(values.back()) + to_u64(residual)));
    }
  } else {
    values = std::move(source);
  }
  if (values.size() != count) throw std::runtime_error("BOS ablation decoded count mismatch");
  return values;
}

}  // namespace bos_operator_ablation
