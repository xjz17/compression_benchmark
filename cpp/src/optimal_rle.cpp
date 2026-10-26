#include "optimal_rle.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace unified_rle {
namespace {
using Bytes = std::vector<std::uint8_t>;
using U64 = std::uint64_t;
constexpr U64 kInf = std::numeric_limits<U64>::max();
constexpr std::uint8_t kRaw = 0, kConstant = 1, kBitpack = 2, kHybrid = 3;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::size_t add_size(std::size_t a, std::size_t b) {
  require(b <= std::numeric_limits<std::size_t>::max() - a,
          "URL1 size overflow");
  return a + b;
}

U64 add_cost(U64 a, U64 b) {
  require(b <= kInf - a, "URL1 cost overflow");
  return a + b;
}

unsigned varint_bytes(U64 value) {
  unsigned result = 1;
  while (value >= 128) { value >>= 7; ++result; }
  return result;
}

void put_varint(Bytes& out, U64 value) {
  do {
    const auto next = static_cast<std::uint8_t>(value & 127);
    value >>= 7;
    out.push_back(next | (value ? 128 : 0));
  } while (value);
}

struct Cursor {
  std::span<const std::uint8_t> input;
  std::size_t pos = 0;

  std::uint8_t byte() {
    require(pos < input.size(), "URL1 truncated byte");
    return input[pos++];
  }
  U64 varint() {
    U64 value = 0;
    for (unsigned i = 0; i != 10; ++i) {
      const auto b = byte();
      require(i != 9 || b <= 1, "URL1 overflowing varint");
      value |= U64(b & 127) << (7 * i);
      if (!(b & 128)) {
        require(i == 0 || (b & 127) != 0, "URL1 noncanonical varint");
        return value;
      }
    }
    throw std::invalid_argument("URL1 overflowing varint");
  }
  std::span<const std::uint8_t> take(std::size_t size) {
    require(size <= input.size() - pos, "URL1 truncated body");
    const auto part = input.subspan(pos, size);
    pos += size;
    return part;
  }
};

struct Header { std::uint8_t width, mode; U64 count; };

Header header(Cursor& cursor) {
  require(is_frame(cursor.input), "URL1 bad magic");
  cursor.pos = 4;
  const auto width = cursor.byte();
  const auto mode = cursor.byte();
  require(width >= 1 && width <= 8, "URL1 symbol width must be 1..8");
  require(mode <= kHybrid, "URL1 invalid mode");
  const auto count = cursor.varint();
  require(count <= std::numeric_limits<std::size_t>::max() / width,
          "URL1 decoded size overflow");
  require(count != 0 || mode == kRaw, "URL1 empty stream requires raw mode");
  return {width, mode, count};
}

U64 read_symbol(std::span<const std::uint8_t> raw) {
  U64 value = 0;
  for (std::size_t i = 0; i < raw.size(); ++i)
    value |= U64(raw[i]) << (8 * i);
  return value;
}

void append_symbol(Bytes& out, U64 value, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

unsigned bits(U64 value) { return std::bit_width(value); }

std::size_t packed_size(std::size_t count, unsigned width) {
  // Avoid count * width and +7 overflow, even for an attacker-supplied count.
  const auto full = count / 8;
  require(width == 0 || full <= std::numeric_limits<std::size_t>::max() / width,
          "URL1 packed size overflow");
  return add_size(full * width, ((count % 8) * width + 7) / 8);
}

void put_packed(Bytes& out, std::span<const U64> values, unsigned width) {
  const auto start = out.size();
  out.resize(add_size(start, packed_size(values.size(), width)), 0);
  std::size_t byte_pos = start;
  unsigned offset = 0;
  for (U64 value : values) {
    unsigned remaining = width;
    while (remaining) {
      const unsigned take = std::min(remaining, 8 - offset);
      const auto mask = (1u << take) - 1;
      out[byte_pos] |= static_cast<std::uint8_t>((value & mask) << offset);
      value >>= take;
      remaining -= take;
      offset += take;
      if (offset == 8) { offset = 0; ++byte_pos; }
    }
  }
}

// Reads exactly count packed symbols, verifies zero high padding and the
// canonical minimum bit width, and appends full little-endian symbols.
void get_packed(Cursor& cursor, Bytes& out, std::size_t count,
                unsigned width, unsigned symbol_width) {
  require(width <= symbol_width * 8, "URL1 invalid packed bit width");
  const auto body = cursor.take(packed_size(count, width));
  const unsigned used_last = static_cast<unsigned>((count % 8) * width % 8);
  if (used_last && !body.empty())
    require((body.back() >> used_last) == 0, "URL1 nonzero packed padding");
  std::size_t byte_pos = 0;
  unsigned offset = 0;
  U64 aggregate = 0;
  for (std::size_t i = 0; i < count; ++i) {
    U64 value = 0;
    unsigned remaining = width, shift = 0;
    while (remaining) {
      const unsigned take = std::min(remaining, 8 - offset);
      const auto mask = (1u << take) - 1;
      value |= U64((body[byte_pos] >> offset) & mask) << shift;
      remaining -= take;
      shift += take;
      offset += take;
      if (offset == 8) { offset = 0; ++byte_pos; }
    }
    aggregate |= value;
    append_symbol(out, value, symbol_width);
  }
  require(bits(aggregate) == width, "URL1 nonminimal packed bit width");
}

struct Minimum { U64 cost; std::size_t index; };
Minimum better(Minimum a, Minimum b) {
  return a.cost < b.cost || (a.cost == b.cost && a.index < b.index) ? a : b;
}

// Incremental RMQ on dp[start], with deterministic earliest-index ties.
class RangeMinimum {
 public:
  explicit RangeMinimum(std::size_t count) {
    while (base_ < count) {
      require(base_ <= std::numeric_limits<std::size_t>::max() / 2,
              "URL1 DP tree overflow");
      base_ *= 2;
    }
    require(base_ <= std::vector<Minimum>().max_size() / 2,
            "URL1 DP tree too large");
    tree_.assign(base_ * 2, {kInf, std::numeric_limits<std::size_t>::max()});
  }
  void put(std::size_t index, U64 cost) {
    auto p = base_ + index;
    tree_[p] = {cost, index};
    while (p > 1) { p /= 2; tree_[p] = better(tree_[p * 2], tree_[p * 2 + 1]); }
  }
  Minimum query(std::size_t first, std::size_t last) const {
    Minimum result{kInf, std::numeric_limits<std::size_t>::max()};
    for (first += base_, last += base_ + 1; first < last; first /= 2, last /= 2) {
      if (first & 1) result = better(result, tree_[first++]);
      if (last & 1) result = better(result, tree_[--last]);
    }
    return result;
  }
 private:
  std::size_t base_ = 1;
  std::vector<Minimum> tree_;
};

struct Parent { std::size_t start = 0; bool repeat = false; };
struct Hybrid { U64 cost; std::vector<Parent> parents; };

// Frozen URL1 encoder used only as a differential oracle.  Keep this routine
// structurally unchanged when optimizing the production path: byte-for-byte
// compatibility, including segmentation ties, is part of the wire contract.
Hybrid optimal_segments_reference(std::span<const U64> values) {
  const std::size_t n = values.size();
  std::vector<U64> dp(add_size(n, 1), kInf);
  std::vector<Parent> parents(add_size(n, 1));
  RangeMinimum rmq(add_size(n, 1));
  dp[0] = 0;
  rmq.put(0, 0);
  std::size_t run_start = 0;
  for (std::size_t end = 1; end <= n; ++end) {
    if (end == 1 || values[end - 1] != values[end - 2]) run_start = end - 1;
    const auto consider = [&](U64 cost, std::size_t start, bool repeat) {
      // Equal cost: prefer repeat, then earlier start (longer final segment).
      const auto previous = parents[end];
      if (cost < dp[end] || (cost == dp[end] &&
          (repeat > previous.repeat || (repeat == previous.repeat && start < previous.start)))) {
        dp[end] = cost;
        parents[end] = {start, repeat};
      }
    };
    U64 aggregate = 0;
    for (std::size_t length = 1; length <= std::min<std::size_t>(128, end); ++length) {
      aggregate |= values[end - length];
      const U64 token = (length - 1) << 1;
      const U64 bytes = varint_bytes(token) + 1 + packed_size(length, bits(aggregate));
      consider(add_cost(dp[end - length], bytes), end - length, false);
    }
    const std::size_t run_length = end - run_start;
    std::size_t min_length = 2;
    // token = 2*length-3; b bytes cover lengths up to 2^(7*b-1)+1.
    for (unsigned b = 1; min_length <= run_length && b <= 10; ++b) {
      const U64 limit = b == 10 ? (U64(1) << 63) + 1 : (U64(1) << (7 * b - 1)) + 1;
      const std::size_t max_length = static_cast<std::size_t>(std::min<U64>(run_length, limit));
      const auto m = rmq.query(end - max_length, end - min_length);
      const U64 bytes = b + 1 + packed_size(1, bits(values[end - 1]));
      consider(add_cost(m.cost, bytes), m.index, true);
      if (max_length == run_length) break;
      min_length = max_length + 1;
    }
    rmq.put(end, dp[end]);
  }
  return {dp[n], std::move(parents)};
}

struct AffineItem {
  __int128 key;
  std::size_t index;
};

// Literal bands cover at most 64 consecutive starts.  A fixed circular
// buffer avoids std::deque block allocation and pointer chasing in the
// per-symbol hot loop.
class FixedAffineQueue {
 public:
  bool empty() const { return size_ == 0; }
  const AffineItem& front() const { return items_[begin_]; }
  const AffineItem& back() const {
    return items_[(begin_ + size_ - 1) % items_.size()];
  }
  void pop_front() {
    begin_ = (begin_ + 1) % items_.size();
    --size_;
  }
  void pop_back() { --size_; }
  void push_back(AffineItem item) {
    require(size_ < items_.size(), "URL1 literal queue overflow");
    items_[(begin_ + size_) % items_.size()] = item;
    ++size_;
  }

 private:
  std::array<AffineItem, 65> items_{};
  std::size_t begin_ = 0;
  std::size_t size_ = 0;
};

// For one literal bit width, maintain minima of
//   dp[start] + ceil((end-start)*width/8)
// over a monotonically advancing [first,last] range.  Splitting start by the
// period 8/gcd(width,8) makes the packed-size term affine.  Each start enters
// and leaves each deque once.
class LiteralBandMinimum {
 public:
  explicit LiteralBandMinimum(unsigned width)
      : width_(width), divisor_(std::gcd(width, 8u)),
        period_(8 / divisor_), slope_(width / divisor_) {}

  Minimum query(const std::vector<U64>& dp, std::size_t end,
                std::size_t first, std::size_t last, U64 overhead) {
    require(first <= last && last < end, "URL1 invalid literal query");
    require(last + 1 >= next_, "URL1 literal query upper bound regressed");
    // `first` is monotone for each band (the fixed-length window and the
    // latest higher-width symbol both only move forward).  Starts already
    // left of it can never be queried, so do not enqueue a large skipped
    // prefix when a bit-width band first becomes active late in the stream.
    if (next_ < first) next_ = first;
    for (std::size_t residue = 0; residue < period_; ++residue) {
      auto& queue = queues_[residue];
      while (!queue.empty() && queue.front().index < first) queue.pop_front();
    }
    while (next_ <= last) {
      const auto residue = next_ % period_;
      const auto quotient = next_ / period_;
      const __int128 key = static_cast<__int128>(dp[next_]) -
                           static_cast<__int128>(quotient) * slope_;
      auto& queue = queues_[residue];
      // Equal keys retain the earlier index, matching the frozen encoder's
      // longer-final-literal tie break.
      while (!queue.empty() && key < queue.back().key) queue.pop_back();
      queue.push_back({key, next_});
      ++next_;
    }
    Minimum answer{kInf, std::numeric_limits<std::size_t>::max()};
    for (std::size_t residue = 0; residue < period_; ++residue) {
      auto& queue = queues_[residue];
      if (queue.empty()) continue;
      const auto distance = end - residue;
      const __int128 packed =
          (static_cast<__int128>(distance) * slope_ + period_ - 1) / period_;
      const __int128 total = queue.front().key + packed + overhead;
      require(total >= 0 && total <= kInf, "URL1 literal cost overflow");
      answer = better(answer, {static_cast<U64>(total), queue.front().index});
    }
    return answer;
  }

 private:
  unsigned width_, divisor_, period_, slope_;
  std::size_t next_ = 0;
  std::array<FixedAffineQueue, 8> queues_;
};

struct ByteLiteralMinimum {
  std::array<std::size_t, 9> last{};
  std::array<LiteralBandMinimum, 9> short_band{
      LiteralBandMinimum(0), LiteralBandMinimum(1), LiteralBandMinimum(2),
      LiteralBandMinimum(3), LiteralBandMinimum(4), LiteralBandMinimum(5),
      LiteralBandMinimum(6), LiteralBandMinimum(7), LiteralBandMinimum(8)};
  std::array<LiteralBandMinimum, 9> long_band{
      LiteralBandMinimum(0), LiteralBandMinimum(1), LiteralBandMinimum(2),
      LiteralBandMinimum(3), LiteralBandMinimum(4), LiteralBandMinimum(5),
      LiteralBandMinimum(6), LiteralBandMinimum(7), LiteralBandMinimum(8)};

  ByteLiteralMinimum() {
    last.fill(std::numeric_limits<std::size_t>::max());
  }

  Minimum query(const std::vector<U64>& dp, std::span<const U64> values,
                std::size_t end) {
    require(end >= 1 && end <= values.size(), "URL1 invalid byte literal end");
    const auto current_width = bits(values[end - 1]);
    require(current_width <= 8, "URL1 byte literal exceeds 8 bits");
    last[current_width] = end - 1;
    Minimum answer{kInf, std::numeric_limits<std::size_t>::max()};
    std::size_t latest_greater = std::numeric_limits<std::size_t>::max();
    for (int width = 8; width >= 0; --width) {
      const auto latest = last[static_cast<std::size_t>(width)];
      if (latest != std::numeric_limits<std::size_t>::max() &&
          (latest_greater == std::numeric_limits<std::size_t>::max() ||
           latest > latest_greater)) {
        const auto qualified_first =
            latest_greater == std::numeric_limits<std::size_t>::max()
                ? 0 : latest_greater + 1;
        const auto short_first = std::max(
            qualified_first, end > 64 ? end - 64 : std::size_t(0));
        if (short_first <= latest)
          answer = better(answer, short_band[width].query(
              dp, end, short_first, latest, 2));
        if (end >= 65) {
          const auto long_first = std::max(
              qualified_first, end > 128 ? end - 128 : std::size_t(0));
          const auto long_last = std::min(latest, end - 65);
          if (long_first <= long_last)
            answer = better(answer, long_band[width].query(
                dp, end, long_first, long_last, 3));
        }
      }
      if (latest != std::numeric_limits<std::size_t>::max() &&
          (latest_greater == std::numeric_limits<std::size_t>::max() ||
           latest > latest_greater))
        latest_greater = latest;
    }
    require(answer.cost != kInf, "URL1 byte literal minimum is absent");
    return answer;
  }
};

struct RepeatQueue {
  std::deque<Minimum> values;

  void put(std::size_t index, U64 cost) {
    while (!values.empty() && cost < values.back().cost) values.pop_back();
    values.push_back({cost, index});
  }
  Minimum query(std::size_t first) {
    while (!values.empty() && values.front().index < first) values.pop_front();
    require(!values.empty(), "URL1 repeat minimum is absent");
    return values.front();
  }
  void clear() { values.clear(); }
};

Hybrid optimal_segments_fast(std::span<const U64> values,
                             unsigned symbol_width) {
  const std::size_t n = values.size();
  std::vector<U64> dp(add_size(n, 1), kInf);
  std::vector<Parent> parents(add_size(n, 1));
  std::vector<unsigned> value_widths;
  value_widths.reserve(n);
  for (const auto value : values) value_widths.push_back(bits(value));
  dp[0] = 0;
  ByteLiteralMinimum byte_literals;
  std::array<RepeatQueue, 10> repeat_queues;
  std::size_t run_start = 0;
  for (std::size_t end = 1; end <= n; ++end) {
    const bool new_run = end == 1 || values[end - 1] != values[end - 2];
    if (new_run) {
      run_start = end - 1;
      for (auto& queue : repeat_queues) queue.clear();
    }
    const auto consider = [&](U64 cost, std::size_t start, bool repeat) {
      const auto previous = parents[end];
      if (cost < dp[end] || (cost == dp[end] &&
          (repeat > previous.repeat ||
           (repeat == previous.repeat && start < previous.start)))) {
        dp[end] = cost;
        parents[end] = {start, repeat};
      }
    };

    if (symbol_width == 1) {
      const auto literal = byte_literals.query(dp, values, end);
      consider(literal.cost, literal.index, false);
    } else {
      // The cold typed-word paths retain the frozen bounded scan.  Precomputed
      // symbol widths remove repeated OR/bit_width work without changing a
      // single candidate or tie.
      unsigned aggregate_width = 0;
      for (std::size_t length = 1;
           length <= std::min<std::size_t>(128, end); ++length) {
        aggregate_width = std::max(aggregate_width,
                                   value_widths[end - length]);
        const U64 token_bytes = length <= 64 ? 1 : 2;
        const U64 bytes = token_bytes + 1 +
                          packed_size(length, aggregate_width);
        consider(add_cost(dp[end - length], bytes), end - length, false);
      }
    }

    const std::size_t run_length = end - run_start;
    std::size_t min_length = 2;
    for (unsigned b = 1; min_length <= run_length && b <= 10; ++b) {
      const U64 limit = b == 10 ? (U64(1) << 63) + 1
                                : (U64(1) << (7 * b - 1)) + 1;
      const auto max_length = static_cast<std::size_t>(
          std::min<U64>(run_length, limit));
      const auto insert = end - min_length;
      repeat_queues[b - 1].put(insert, dp[insert]);
      const auto first = end - max_length;
      const auto minimum = repeat_queues[b - 1].query(first);
      const U64 bytes = b + 1 + packed_size(1, value_widths[end - 1]);
      consider(add_cost(minimum.cost, bytes), minimum.index, true);
      if (max_length == run_length) break;
      min_length = max_length + 1;
    }
  }
  return {dp[n], std::move(parents)};
}

void put_hybrid(Bytes& out, std::span<const U64> values, const Hybrid& hybrid) {
  std::vector<std::pair<std::size_t, Parent>> segments;
  for (std::size_t end = values.size(); end != 0;) {
    const auto parent = hybrid.parents[end];
    require(parent.start < end, "URL1 invalid internal DP parent");
    segments.emplace_back(end, parent);
    end = parent.start;
  }
  std::reverse(segments.begin(), segments.end());
  for (const auto& [end, parent] : segments) {
    const auto length = end - parent.start;
    const auto segment = values.subspan(parent.start, parent.repeat ? 1 : length);
    U64 aggregate = 0;
    for (U64 value : segment) aggregate |= value;
    const auto width = bits(aggregate);
    const U64 token = parent.repeat ? ((U64(length) - 2) << 1) | 1 : (U64(length) - 1) << 1;
    put_varint(out, token);
    out.push_back(static_cast<std::uint8_t>(width));
    put_packed(out, segment, width);
  }
}

Bytes encode_impl(std::span<const std::uint8_t> raw, std::uint8_t width,
                  bool frozen_reference) {
  require(width >= 1 && width <= 8, "URL1 symbol width must be 1..8");
  require(raw.size() % width == 0, "URL1 incomplete symbol");
  const auto count = raw.size() / width;
  require(count <= (U64(1) << 63), "URL1 count too large for hybrid tokens");
  std::vector<U64> values;
  values.reserve(count);
  U64 aggregate = 0;
  bool constant = true;
  for (std::size_t i = 0; i < count; ++i) {
    const auto value = read_symbol(raw.subspan(i * width, width));
    if (i && value != values.front()) constant = false;
    values.push_back(value);
    aggregate |= value;
  }
  std::uint8_t mode = kRaw;
  U64 best = raw.size();
  const auto take = [&](std::uint8_t candidate, U64 cost) {
    if (cost < best) { mode = candidate; best = cost; }
  };
  const auto packed_width = bits(aggregate);
  if (count && constant) take(kConstant, width);
  if (count) take(kBitpack, add_cost(1, packed_size(count, packed_width)));
  Hybrid hybrid;
  if (count && best > 2) {
    hybrid = frozen_reference ? optimal_segments_reference(values)
                              : optimal_segments_fast(values, width);
    take(kHybrid, hybrid.cost);
  }
  Bytes out{'U', 'R', 'L', '1', width, mode};
  put_varint(out, count);
  const auto header_size = out.size();
  require(best <= out.max_size() - header_size, "URL1 output too large");
  out.reserve(header_size + static_cast<std::size_t>(best));
  if (mode == kRaw) out.insert(out.end(), raw.begin(), raw.end());
  else if (mode == kConstant)
    out.insert(out.end(), raw.begin(), raw.begin() + width);
  else if (mode == kBitpack) {
    out.push_back(static_cast<std::uint8_t>(packed_width));
    put_packed(out, values, packed_width);
  } else put_hybrid(out, values, hybrid);
  require(out.size() == header_size + best, "URL1 internal size mismatch");
  return out;
}
}  // namespace

bool is_frame(std::span<const std::uint8_t> frame) {
  return frame.size() >= 4 && frame[0] == 'U' && frame[1] == 'R' &&
         frame[2] == 'L' && frame[3] == '1';
}

std::uint8_t symbol_width(std::span<const std::uint8_t> frame) {
  Cursor cursor{frame};
  return header(cursor).width;
}

Bytes encode(std::span<const std::uint8_t> raw, std::uint8_t width) {
  return encode_impl(raw, width, false);
}

namespace testing {
Bytes encode_frozen_reference(std::span<const std::uint8_t> raw,
                              std::uint8_t width) {
  return encode_impl(raw, width, true);
}
}  // namespace testing

Bytes decode(std::span<const std::uint8_t> frame) {
  return decode(frame, kDefaultMaxDecodedBytes);
}

Bytes decode(std::span<const std::uint8_t> frame, std::size_t max_decoded_bytes) {
  Cursor cursor{frame};
  const auto h = header(cursor);
  const auto count = static_cast<std::size_t>(h.count);
  const auto decoded_size = count * h.width;
  require(decoded_size <= max_decoded_bytes, "URL1 decoded size exceeds configured limit");
  Bytes out;
  require(decoded_size <= out.max_size(), "URL1 decoded size exceeds vector limit");
  if (h.mode == kRaw) {
    const auto body = cursor.take(decoded_size);
    out.assign(body.begin(), body.end());
  } else if (h.mode == kConstant) {
    const auto value = read_symbol(cursor.take(h.width));
    out.reserve(decoded_size);
    for (std::size_t i = 0; i < count; ++i) append_symbol(out, value, h.width);
  } else if (h.mode == kBitpack) {
    const auto width = cursor.byte();
    // Validate the body length before allocation, especially for truncated input.
    require(width <= h.width * 8, "URL1 invalid packed bit width");
    require(packed_size(count, width) <= frame.size() - cursor.pos, "URL1 truncated body");
    out.reserve(decoded_size);
    get_packed(cursor, out, count, width, h.width);
  } else {
    std::size_t produced = 0;
    while (produced < count) {
      const U64 token = cursor.varint();
      const bool repeat = token & 1;
      const U64 length64 = (token >> 1) + (repeat ? 2 : 1);
      require(length64 <= count - produced, "URL1 segment exceeds symbol count");
      require(repeat || length64 <= 128, "URL1 literal segment exceeds 128");
      const auto length = static_cast<std::size_t>(length64);
      const auto width = cursor.byte();
      if (repeat) {
        Bytes value;
        get_packed(cursor, value, 1, width, h.width);
        // Grow only after validating this segment's complete encoded payload.
        const auto scalar = read_symbol(value);
        for (std::size_t i = 0; i < length; ++i) append_symbol(out, scalar, h.width);
      } else get_packed(cursor, out, length, width, h.width);
      produced += length;
    }
  }
  require(cursor.pos == frame.size(), "URL1 trailing garbage");
  require(out.size() == decoded_size, "URL1 decoded size mismatch");
  return out;
}
}  // namespace unified_rle
