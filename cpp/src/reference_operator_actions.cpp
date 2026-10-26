#include "reference_operator_actions.hpp"
#include "minimum_for.hpp"
#include "unsigned_leb128.hpp"
#include "zigzag.hpp"
#include "dictionary_recode.hpp"
#include "max_value_bitpack.hpp"
#include "operator_layout_primitives.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <zlib.h>

namespace reference_operator_actions {
namespace {
using I = std::int64_t;
using U = std::uint64_t;
using Values = std::vector<I>;
constexpr I kMin = std::numeric_limits<I>::min();
constexpr I kMax = std::numeric_limits<I>::max();
// OPA1 action IDs are an immutable wire contract.  ID 1 is retired from new
// forward pipelines, but its name, tag and inverse remain available so saved
// OPA1/OPC1 streams still decode without renumbering IDs 2..28.
constexpr std::uint8_t kRetiredScalingWireId = 1;
constexpr std::uint8_t kLegacyZigzagVarintRecipeWireId = 19;
constexpr std::array<std::string_view, 29> kFamilies = {
    "eraser", "scaling", "frame_of_reference", "value_reorder",
    "regression_predictor", "outlier_separation", "bitpack_int", "delta",
    "xor", "trailing_zero_pack", "dictionary", "zigzag", "block_transform",
    "symbol_recode", "run_length", "lz_match", "match_parse_policy",
    "match_window_model", "match_token_format", "integer_code", "bitstream_pack",
    "entropy_huffman", "entropy_range", "entropy_fse", "entropy_special",
    "literal_entropy", "block_state_control", "container_frame", "xor"};
constexpr std::array<std::string_view, 29> kTags = {
    "eraser", "scaling", "for", "reorder", "regression", "outlier", "bitpack",
    "delta", "xor", "trailing0", "dictionary", "zigzag", "block", "symbol",
    "runlength", "lzmatch", "policy", "window", "tokens", "intcode", "bitstream",
    "entropy_huff", "entropy_rang", "entropy_fse", "entropy_spec", "literal_entr",
    "window", "container", "xorchimp"};

const Names& wire_actions() {
  static const Names result = [] {
    Names names;
    for (std::size_t i = 0; i < 28; ++i)
      names.push_back("op." + std::string(kFamilies[i]) + ".reference_v1");
    names.push_back("op.xor.chimp_history128.reference_v1");
    return names;
  }();
  return result;
}

I signed_bits(U value) { return std::bit_cast<I>(value); }
U unsigned_bits(I value) { return std::bit_cast<U>(value); }
I add(I a, I b) { return signed_bits(unsigned_bits(a) + unsigned_bits(b)); }
I subtract(I a, I b) { return signed_bits(unsigned_bits(a) - unsigned_bits(b)); }
I zigzag(I value) {
  return signed_bits(compression_operators::zigzag::encode64(value));
}
I unzigzag(U value) { return compression_operators::zigzag::decode64(value); }
[[noreturn]] void bad(const char* message) { throw std::runtime_error(message); }
void require(bool condition, const char* message) { if (!condition) bad(message); }
void put(Bytes& out, U value, unsigned width) {
  for (unsigned byte = 0; byte < width; ++byte)
    out.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}
void append(Bytes& out, std::span<const std::uint8_t> bytes) {
  out.insert(out.end(), bytes.begin(), bytes.end());
}
void append_text(Bytes& out, std::string_view text) {
  out.insert(out.end(), text.begin(), text.end());
}
struct Reader {
  std::span<const std::uint8_t> raw;
  std::size_t pos = 0;
  void need(std::size_t n) const { require(pos <= raw.size() && n <= raw.size() - pos, "truncated reference stream"); }
  U get(unsigned width) {
    need(width);
    U result = 0;
    for (unsigned i = 0; i < width; ++i) result |= U{raw[pos++]} << (8 * i);
    return result;
  }
  std::span<const std::uint8_t> take(std::size_t count) {
    need(count); const auto result = raw.subspan(pos, count); pos += count; return result;
  }
};
Bytes value_bytes(std::span<const I> values) {
  Bytes out; out.reserve(values.size() * 8);
  for (I value : values) put(out, unsigned_bits(value), 8);
  return out;
}
Values read_values(std::span<const std::uint8_t> raw) {
  require(raw.size() % 8 == 0, "reference actions require complete int64 lanes");
  Reader in{raw}; Values out; out.reserve(raw.size() / 8);
  while (in.pos < raw.size()) out.push_back(signed_bits(in.get(8)));
  return out;
}

struct Record {
  std::string tag;
  std::vector<Values> arrays;
  Bytes extra;
};
struct State { Values values; std::vector<Record> records; };

Bytes meta_bytes(const Record& record) {
  require(record.arrays.size() <= 65535 && record.extra.size() <= UINT32_MAX, "reference metadata too large");
  Bytes out(12, 0);
  std::copy_n(record.tag.begin(), std::min<std::size_t>(record.tag.size(), 12), out.begin());
  put(out, record.arrays.size(), 2); put(out, record.extra.size(), 4); append(out, record.extra);
  for (const auto& array : record.arrays) {
    require(array.size() <= UINT32_MAX / 8, "reference metadata array too large");
    put(out, array.size(), 4); put(out, array.size() * 8, 4); append(out, value_bytes(array));
  }
  return out;
}
Record parse_meta(Reader& in) {
  const auto tag_raw = in.take(12);
  std::string tag(tag_raw.begin(), tag_raw.end());
  while (!tag.empty() && tag.back() == '\0') tag.pop_back();
  const auto arrays = in.get(2), extra_size = in.get(4);
  auto extra = in.take(extra_size);
  Record record{tag, {}, Bytes(extra.begin(), extra.end())};
  for (U i = 0; i < arrays; ++i) {
    const auto count = in.get(4), size = in.get(4);
    require(size == count * 8, "reference metadata array size mismatch");
    record.arrays.push_back(read_values(in.take(size)));
  }
  return record;
}
Bytes compact(const Values& values) {
  require(values.size() <= UINT32_MAX, "reference compact count exceeds uint32");
  std::string_view tag = "I0"; unsigned width = 0;
  if (!values.empty()) {
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    if (*minimum >= 0 && *maximum <= 255) { tag = "U1"; width = 1; }
    else if (*minimum >= -128 && *maximum <= 127) { tag = "I1"; width = 1; }
    else if (*minimum >= -32768 && *maximum <= 32767) { tag = "I2"; width = 2; }
    else if (*minimum >= INT32_MIN && *maximum <= INT32_MAX) { tag = "I4"; width = 4; }
    else { tag = "I8"; width = 8; }
  }
  Bytes out; append_text(out, tag); put(out, values.size(), 4);
  for (I value : values) put(out, unsigned_bits(value), width);
  return out;
}
Values parse_compact(Reader& in) {
  const auto tag_raw = in.take(2);
  std::string tag(tag_raw.begin(), tag_raw.end());
  const auto count = in.get(4);
  unsigned width = 0;
  if (tag == "I0") require(count == 0, "I0 payload must be empty");
  else if (tag == "U1" || tag == "I1") width = 1;
  else if (tag == "I2") width = 2;
  else if (tag == "I4") width = 4;
  else if (tag == "I8") width = 8;
  else bad("unknown reference compact tag");
  require(count * width == in.raw.size() - in.pos, "reference compact payload length mismatch");
  Values out; out.reserve(count);
  for (U i = 0; i < count; ++i) {
    U word = in.get(width);
    if (width < 8 && tag != "U1" && (word & (U{1} << (width * 8 - 1))))
      word |= ~((U{1} << (width * 8)) - 1);
    out.push_back(signed_bits(word));
  }
  return out;
}

// CPython math.fsum's partials algorithm, including final half-even fixup.
// Input here is finite (the source lanes are int64, including raw float bits).
double precise_sum(const std::vector<double>& values) {
  std::vector<double> partials;
  for (double x : values) {
    std::size_t kept = 0;
    for (double y : partials) {
      if (std::abs(x) < std::abs(y)) std::swap(x, y);
      const double hi = x + y;
      const double yr = hi - x;
      const double lo = y - yr;
      if (lo != 0.0) partials[kept++] = lo;
      x = hi;
    }
    partials.resize(kept);
    if (x != 0.0) partials.push_back(x);
  }
  double hi = 0.0, lo = 0.0;
  std::size_t n = partials.size();
  if (n) {
    hi = partials[--n];
    while (n) {
      const double x = hi, y = partials[--n];
      hi = x + y;
      const double yr = hi - x;
      lo = y - yr;
      if (lo != 0.0) break;
    }
    if (n && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0))) {
      const double y = lo * 2.0;
      const double x = hi + y;
      if (y == x - hi) hi = x;
    }
  }
  return hi;
}
I numpy_cast(double value) {
  if (!std::isfinite(value) || value < -0x1p63 || value >= 0x1p63) return kMin;
  return static_cast<I>(value);
}
double percentile(const Values& values, double fraction) {
  std::vector<double> sorted; sorted.reserve(values.size());
  for (I value : values) sorted.push_back(static_cast<double>(value));
  std::sort(sorted.begin(), sorted.end());
  const double index = (sorted.size() - 1) * fraction;
  const auto low = static_cast<std::size_t>(index);
  const auto high = std::min(low + 1, sorted.size() - 1);
  const double weight = index - low, a = sorted[low], b = sorted[high];
  const double difference = b - a;
  return weight >= 0.5 ? b - difference * (1.0 - weight) : a + difference * weight;
}
Values stable_order(std::span<const I> values) {
  Values order(values.size()); std::iota(order.begin(), order.end(), I{0});
  std::stable_sort(order.begin(), order.end(), [&](I a, I b) { return values[a] < values[b]; });
  return order;
}

void transform(State& state, std::uint8_t id) {
  require(id < kTags.size(), "unknown reference action ID");
  require(id != kRetiredScalingWireId,
          "op.scaling.reference_v1 is retired from forward execution");
  const auto& values = state.values;
  Record record{std::string(kTags[id]), {}, {}};
  Values out;
  switch (id) {
  case 0: {
    const exact_actions::operator_layout::BitFields fields{4, 60};
    Values low; low.reserve(values.size()); out.reserve(values.size());
    for (I value : values) {
      low.push_back(fields.field(unsigned_bits(value), 0));
      out.push_back(exact_actions::operator_layout::sign_extend(fields.field(unsigned_bits(value), 1), 60));
    }
    record.arrays.push_back(std::move(low)); break;
  }
  case 2: case 6: {
    const auto encoded = minimum_for::encode<I>(values);
    for (const U residual : encoded.offsets) out.push_back(signed_bits(residual));
    record.arrays.push_back({encoded.minimum}); break;
  }
  case 3: {
    auto order = stable_order(values);
    for (I index : order) out.push_back(values[index]);
    record.arrays.push_back(std::move(order)); break;
  }
  case 4: {
    double theta0 = 0.0, theta1 = 0.0;
    Values first;
    if (!values.empty()) first.push_back(values.front());
    if (values.size() > 1) {
      std::vector<double> x, y, xx, xy;
      for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        const double a = static_cast<double>(values[i]);
        const double b = static_cast<double>(values[i + 1]);
        x.push_back(a); y.push_back(b); xx.push_back(a * a); xy.push_back(a * b);
      }
      const double sx = precise_sum(x), sy = precise_sum(y), sxx = precise_sum(xx), sxy = precise_sum(xy);
      const double n = static_cast<double>(x.size());
      const double denom = n * sxx - sx * sx;
      const bool degenerate = std::abs(denom) <= 1e-12;
      theta1 = degenerate ? 0.0 : (n * sxy - sx * sy) / denom;
      theta0 = degenerate ? sy / n : (sy - theta1 * sx) / n;
      for (std::size_t i = 0; i < x.size(); ++i)
        out.push_back(subtract(values[i + 1], numpy_cast(std::nearbyint(theta0 + theta1 * x[i]))));
    }
    record.arrays.push_back(std::move(first));
    put(record.extra, std::bit_cast<U>(theta0), 8); put(record.extra, std::bit_cast<U>(theta1), 8); break;
  }
  case 5: {
    out = values;
    if (values.empty()) break;
    const double q1 = percentile(values, 0.25), q3 = percentile(values, 0.75);
    const double iqr = std::max(q3 - q1, 1.0);
    const double lower = std::floor(q1 - 1.5 * iqr), upper = std::ceil(q3 + 1.5 * iqr);
    require(std::isfinite(lower) && std::isfinite(upper) && lower >= -0x1p63 && lower < 0x1p63 && upper >= -0x1p63 && upper < 0x1p63,
            "reference outlier threshold cannot be stored as int64 (legacy Python also rejects)");
    const I low = static_cast<I>(lower), high = static_cast<I>(upper);
    Values positions, originals;
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (values[i] < low || values[i] > high) { positions.push_back(i); originals.push_back(values[i]); }
      if (values[i] < low) out[i] = low;
      if (values[i] > high) out[i] = high;
    }
    record.arrays = {std::move(positions), std::move(originals)}; break;
  }
  case 7: case 8: {
    Values first;
    if (!values.empty()) first.push_back(values.front());
    for (std::size_t i = 1; i < values.size(); ++i)
      out.push_back(id == 7 ? subtract(values[i], values[i - 1]) : values[i] ^ values[i - 1]);
    record.arrays.push_back(std::move(first)); break;
  }
  case 9: {
    int shift = 64; bool any = false;
    for (I value : values) if (value != 0) { any = true; shift = std::min(shift, std::countr_zero(unsigned_bits(value))); }
    shift = any ? std::min(shift, 16) : 0;
    for (I value : values) out.push_back(value >> shift);
    record.arrays.push_back({shift}); break;
  }
  case 10: case 13: case 21: case 22: case 23: case 24: case 25: {
    using exact_actions::dictionary_recode::Order;
    auto encoded = exact_actions::dictionary_recode::encode<I>(values, id == 10 ? Order::value : Order::frequency);
    for (const auto code : encoded.ids) out.push_back(static_cast<I>(code));
    record.arrays.push_back(std::move(encoded.dictionary)); break;
  }
  case 11: for (I value : values) out.push_back(zigzag(value)); break;
  case 12: {
    Values sizes, orders;
    for (std::size_t start = 0; start < values.size(); start += 64) {
      const auto count = std::min<std::size_t>(64, values.size() - start);
      const auto order = stable_order(std::span(values).subspan(start, count));
      for (I index : order) out.push_back(values[start + index]);
      orders.insert(orders.end(), order.begin(), order.end()); sizes.push_back(count);
    }
    record.arrays = {std::move(sizes), std::move(orders)}; break;
  }
  case 14: case 15: {
    for (std::size_t start = 0; start < values.size();) {
      std::size_t end = start + 1;
      while (end < values.size() && values[end] == values[start]) ++end;
      const I run = end - start;
      if (id == 14) { out.push_back(run); out.push_back(values[start]); }
      else if (run >= 4) { out.push_back(1); out.push_back(run); out.push_back(values[start]); }
      else { out.push_back(0); out.push_back(run); out.insert(out.end(), values.begin() + start, values.begin() + end); }
      start = end;
    }
    break;
  }
  case 16: for (std::size_t i = 0; i < values.size(); ++i) out.push_back(signed_bits(unsigned_bits(values[i]) * 3 + i % 251)); break;
  case 17: case 26: for (std::size_t i = 0; i < values.size(); ++i) out.push_back(signed_bits(unsigned_bits(values[i]) ^ (U{i} * 131))); break;
  case 18: case 20: {
    const exact_actions::operator_layout::BitFields fields{4, 60};
    Values highs; std::vector<U> lows;
    for (I value : values) {
      lows.push_back(fields.field(unsigned_bits(value), 0));
      highs.push_back(exact_actions::operator_layout::sign_extend(fields.field(unsigned_bits(value), 1), 60));
    }
    const auto packed = exact_actions::max_value_bitpack::pack_fixed(lows, 4);
    for (const auto byte : packed.payload) out.push_back(byte);
    record.arrays = {Values{static_cast<I>(values.size())}, std::move(highs)}; break;
  }
  case 19: {
    // Frozen OPA1 wire ID 19 is the legacy A046 -> A023 recipe, not a
    // separately selectable current implementation. Keep its wire unchanged.
    Values lengths;
    for (I value : values) {
      const U code = compression_operators::zigzag::encode64(value);
      Bytes encoded;
      compression_operators::unsigned_leb128::append(encoded, code);
      for (auto byte : encoded) out.push_back(byte);
      lengths.push_back(static_cast<I>(encoded.size()));
    }
    record.arrays.push_back(std::move(lengths)); break;
  }
  case 27: {
    out = {31, 139, static_cast<I>(values.size() & 65535)};
    out.insert(out.end(), values.begin(), values.end());
    U total = 0; for (I value : values) total += unsigned_bits(value);
    out.push_back(total & 65535); break;
  }
  case 28: {
    Values first, references;
    if (!values.empty()) first.push_back(values.front());
    std::array<U, 128> stored{};
    std::array<std::size_t, 16384> indices{};
    if (!values.empty()) { stored[0] = unsigned_bits(values.front()); indices[stored[0] & 16383] = 0; }
    std::size_t index = 0, current = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
      const U value = unsigned_bits(values[i]); const auto key = value & 16383;
      const auto candidate_index = indices[key];
      std::size_t reference = index % 128;
      if (index - candidate_index < 128) {
        const auto slot = candidate_index % 128;
        if (std::countr_zero(value ^ stored[slot]) > 13) reference = slot;
      }
      references.push_back(reference); out.push_back(signed_bits(stored[reference] ^ value));
      current = (current + 1) % 128; stored[current] = value; ++index; indices[key] = index;
    }
    record.arrays = {std::move(first), std::move(references)}; break;
  }
  default: bad("unknown reference transform");
  }
  state.values = std::move(out); state.records.push_back(std::move(record));
}

const Values& array_at(const Record& record, std::size_t index) {
  require(index < record.arrays.size(), "missing reference metadata array"); return record.arrays[index];
}
I scalar_at(const Record& record, std::size_t index = 0) {
  const auto& values = array_at(record, index); require(values.size() == 1, "reference scalar metadata length mismatch"); return values.front();
}
std::size_t checked_index(I value, std::size_t bound) {
  require(value >= 0 && static_cast<U>(value) < bound, "reference metadata index out of range"); return value;
}
Values invert(Values current, const Record& record, std::uint8_t id) {
  Values out;
  switch (id) {
  case 0: {
    const auto& low = array_at(record, 0); require(low.size() == current.size(), "eraser metadata length mismatch");
    const exact_actions::operator_layout::BitFields layout{4, 60};
    for (std::size_t i = 0; i < current.size(); ++i) {
      const U fields[]{unsigned_bits(low[i]), unsigned_bits(current[i]) & exact_actions::operator_layout::low_mask(60)};
      out.push_back(signed_bits(layout.join(fields)));
    }
    break;
  }
  case 1: return current;
  case 2: case 6: { const auto base = scalar_at(record); for (I value : current) out.push_back(minimum_for::restore(base, unsigned_bits(value))); break; }
  case 3: {
    const auto& order = array_at(record, 0); require(order.size() == current.size(), "reorder metadata size mismatch"); out.resize(current.size());
    for (std::size_t i = 0; i < current.size(); ++i) out[checked_index(order[i], out.size())] = current[i]; break;
  }
  case 4: {
    const auto& first = array_at(record, 0); if (first.empty()) return {};
    require(first.size() == 1 && record.extra.size() == 16, "invalid regression metadata");
    Reader coefficients{record.extra}; const double b0 = std::bit_cast<double>(coefficients.get(8)), b1 = std::bit_cast<double>(coefficients.get(8));
    out.push_back(first.front());
    for (I residual : current) {
      const double rounded = std::nearbyint(b0 + b1 * static_cast<double>(out.back()));
      require(std::isfinite(rounded) && std::abs(rounded) < 0x1p126, "invalid regression prediction");
      const __int128 value = static_cast<__int128>(rounded) + residual;
      require(value >= kMin && value <= kMax, "reference regression inverse exceeds int64 (legacy Python also rejects)");
      out.push_back(static_cast<I>(value));
    }
    break;
  }
  case 5: {
    if (record.arrays.empty()) return current;
    const auto& positions = array_at(record, 0); const auto& originals = array_at(record, 1);
    require(positions.size() == originals.size(), "outlier metadata size mismatch");
    for (std::size_t i = 0; i < positions.size(); ++i) current[checked_index(positions[i], current.size())] = originals[i]; return current;
  }
  case 7: case 8: {
    const auto& first = array_at(record, 0); if (first.empty()) return {};
    require(first.size() == 1, "invalid first-value metadata"); out.push_back(first.front());
    for (I value : current) out.push_back(id == 7 ? add(out.back(), value) : out.back() ^ value); break;
  }
  case 9: {
    const I shift = scalar_at(record); require(shift >= 0 && shift <= 63, "invalid trailing-zero shift");
    for (I value : current) out.push_back(signed_bits(unsigned_bits(value) << shift)); break;
  }
  case 10: case 13: case 21: case 22: case 23: case 24: case 25: {
    const auto& dictionary = array_at(record, 0);
    std::vector<U> ids;
    for (I code : current) ids.push_back(unsigned_bits(code));
    out = exact_actions::dictionary_recode::decode<I>(dictionary, ids); break;
  }
  case 11: for (I value : current) out.push_back(unzigzag(unsigned_bits(value))); break;
  case 12: {
    const auto& sizes = array_at(record, 0); const auto& orders = array_at(record, 1);
    require(orders.size() == current.size(), "block ordering metadata size mismatch"); out.resize(current.size()); std::size_t pos = 0;
    for (I size : sizes) {
      require(size >= 0 && static_cast<U>(size) <= current.size() - pos, "invalid block extent");
      for (I i = 0; i < size; ++i) out[pos + checked_index(orders[pos + i], size)] = current[pos + i];
      pos += size;
    }
    require(pos == current.size(), "block extent sum mismatch"); break;
  }
  case 14: {
    require(current.size() % 2 == 0, "invalid RLE pair count");
    for (std::size_t i = 0; i < current.size(); i += 2) {
      require(current[i] >= 0 && static_cast<U>(current[i]) <= UINT32_MAX, "invalid RLE count");
      out.insert(out.end(), current[i], current[i + 1]);
    }
    break;
  }
  case 15: {
    for (std::size_t pos = 0; pos < current.size();) {
      require(current.size() - pos >= 2, "truncated LZ reference token");
      const I kind = current[pos++], count = current[pos++]; require(count >= 0 && static_cast<U>(count) <= UINT32_MAX, "invalid LZ reference count");
      if (kind == 1) { require(pos < current.size(), "missing LZ repeated value"); out.insert(out.end(), count, current[pos++]); }
      else { require(static_cast<U>(count) <= current.size() - pos, "truncated LZ literals"); out.insert(out.end(), current.begin() + pos, current.begin() + pos + count); pos += count; }
    }
    break;
  }
  case 16: for (std::size_t i = 0; i < current.size(); ++i) out.push_back(signed_bits((unsigned_bits(current[i]) - i % 251) * U{0xaaaaaaaaaaaaaaab})); break;
  case 17: case 26: for (std::size_t i = 0; i < current.size(); ++i) out.push_back(signed_bits(unsigned_bits(current[i]) ^ (U{i} * 131))); break;
  case 18: case 20: {
    const I size = scalar_at(record); const auto& highs = array_at(record, 1);
    require(size >= 0 && static_cast<U>(size) == highs.size() && (static_cast<U>(size) + 1) / 2 == current.size(), "nibble metadata size mismatch");
    Bytes payload;
    for (const I byte : current) { require(byte >= 0 && byte <= 255, "invalid packed nibble byte"); payload.push_back(byte); }
    const auto lows = exact_actions::max_value_bitpack::unpack(payload, highs.size(), 4);
    const exact_actions::operator_layout::BitFields layout{4, 60};
    for (std::size_t i = 0; i < highs.size(); ++i) {
      const U fields[]{lows[i], unsigned_bits(highs[i]) & exact_actions::operator_layout::low_mask(60)};
      out.push_back(signed_bits(layout.join(fields)));
    }
    break;
  }
  case 19: {
    const auto& lengths = array_at(record, 0); std::size_t pos = 0;
    for (I length : lengths) {
      require(length > 0 && length <= 10 && static_cast<U>(length) <= current.size() - pos, "invalid integer-code length");
      U value = 0;
      for (I i = 0; i < length; ++i) value |= (unsigned_bits(current[pos++]) & 127) << (7 * i);
      out.push_back(unzigzag(value));
    }
    require(pos == current.size(), "trailing integer-code values"); break;
  }
  case 27: require(current.size() >= 4, "truncated reference container"); return Values(current.begin() + 3, current.end() - 1);
  case 28: {
    const auto& first = array_at(record, 0); const auto& references = array_at(record, 1);
    if (first.empty()) return {};
    require(first.size() == 1 && references.size() == current.size(), "Chimp metadata length mismatch");
    std::array<U, 128> stored{}; stored[0] = unsigned_bits(first.front()); out.push_back(first.front()); std::size_t slot = 0;
    for (std::size_t i = 0; i < current.size(); ++i) {
      const U value = stored[checked_index(references[i], 128)] ^ unsigned_bits(current[i]);
      out.push_back(signed_bits(value)); slot = (slot + 1) % 128; stored[slot] = value;
    }
    break;
  }
  default: bad("unknown reference inverse");
  }
  return out;
}

std::vector<std::uint8_t> ids_for(const Names& names,
                                  bool allow_retired = false) {
  require(names.size() <= 255, "reference sequence exceeds 255 actions");
  const auto& all = wire_actions(); std::vector<std::uint8_t> ids;
  for (const auto& name : names) {
    const auto found = std::find(all.begin(), all.end(), name);
    require(found != all.end(), "unknown reference action name");
    const auto id = static_cast<std::uint8_t>(found - all.begin());
    require(allow_retired || id != kRetiredScalingWireId,
            "op.scaling.reference_v1 is retired from forward execution");
    require(allow_retired || id != kLegacyZigzagVarintRecipeWireId,
            "op.integer_code.reference_v1 is retired from forward execution; use ZigZag then unsigned ULEB128");
    ids.push_back(id);
  }
  return ids;
}
State run(std::span<const std::uint8_t> raw, const Names& names) {
  State state{read_values(raw), {}};
  for (auto id : ids_for(names)) transform(state, id);
  return state;
}
Bytes state_bytes(const State& state) {
  Bytes out; for (const auto& record : state.records) append(out, meta_bytes(record)); append(out, compact(state.values)); return out;
}
Bytes compress(std::span<const std::uint8_t> raw) {
  uLongf size = compressBound(raw.size()); Bytes out(size);
  require(compress2(out.data(), &size, raw.data(), raw.size(), 6) == Z_OK, "reference zlib compression failed"); out.resize(size); return out;
}
Bytes decompress(std::span<const std::uint8_t> wire) {
  require(wire.size() <= UINT_MAX, "reference compressed input too large");
  z_stream stream{};
  require(inflateInit(&stream) == Z_OK, "reference zlib initialization failed");
  stream.next_in = const_cast<Bytef*>(wire.data()); stream.avail_in = wire.size();
  Bytes out; std::array<std::uint8_t, 32768> block{};
  int status = Z_OK;
  while (status == Z_OK) {
    stream.next_out = block.data(); stream.avail_out = block.size();
    status = inflate(&stream, Z_NO_FLUSH);
    out.insert(out.end(), block.begin(), block.begin() + block.size() - stream.avail_out);
  }
  inflateEnd(&stream); require(status == Z_STREAM_END, "invalid reference zlib stream"); return out;
}
std::pair<std::vector<std::uint8_t>, std::size_t> read_header(std::span<const std::uint8_t> raw, std::string_view expected_magic = "OPA1") {
  Reader in{raw}; const auto magic = in.take(4);
  require(std::string_view(reinterpret_cast<const char*>(magic.data()), 4) == expected_magic, "invalid reference codec header");
  require(in.get(1) == 1, "invalid reference version"); const auto count = in.get(1);
  const auto bytes = in.take(count); std::vector<std::uint8_t> ids(bytes.begin(), bytes.end());
  for (auto id : ids) require(id < 29, "invalid reference action ID");
  return {std::move(ids), in.pos};
}
Bytes inverse_state(std::span<const std::uint8_t> raw, std::span<const std::uint8_t> ids) {
  Reader in{raw}; std::vector<Record> records;
  for (auto id : ids) {
    auto record = parse_meta(in); require(record.tag == kTags[id], "reference action/tag mismatch"); records.push_back(std::move(record));
  }
  auto current = parse_compact(in);
  for (std::size_t i = ids.size(); i-- > 0;) current = invert(std::move(current), records[i], ids[i]);
  return value_bytes(current);
}
}  // namespace

const Names& all_actions() {
  static const Names result = [] {
    Names names;
    const auto& wire = wire_actions();
    for (std::size_t id = 0; id < wire.size(); ++id)
      if (id != kRetiredScalingWireId && id != kLegacyZigzagVarintRecipeWireId)
        names.push_back(wire[id]);
    return names;
  }();
  return result;
}
Bytes serialized(std::span<const std::uint8_t> raw, const Names& names) { return state_bytes(run(raw, names)); }
Bytes final_values(std::span<const std::uint8_t> raw, const Names& names) { return value_bytes(run(raw, names).values); }
Bytes inverse_serialized(std::span<const std::uint8_t> state, const Names& names) {
  return inverse_state(state, ids_for(names, true));
}
Bytes compact_encode(std::span<const std::uint8_t> raw) { return compact(read_values(raw)); }
Bytes compact_decode(std::span<const std::uint8_t> raw) { Reader in{raw}; return value_bytes(parse_compact(in)); }
Bytes inverse_step(std::span<const std::uint8_t> state, const std::string& tag) {
  const auto found = std::find(kTags.begin(), kTags.end(), tag);
  require(found != kTags.end(), "unsupported inverse metadata tag");
  const std::array<std::uint8_t, 1> ids{static_cast<std::uint8_t>(found - kTags.begin())};
  return inverse_state(state, ids);
}
Bytes inspect(std::span<const std::uint8_t> raw, const Names& names) {
  const auto state = run(raw, names); Bytes metadata;
  for (const auto& record : state.records) append(metadata, meta_bytes(record));
  const auto values = value_bytes(state.values); Bytes out;
  put(out, metadata.size(), 8); put(out, values.size(), 8);
  append(out, metadata); append(out, values); append(out, compact(state.values)); return out;
}
Bytes raw_encode(std::span<const std::uint8_t> raw, const Names& names) {
  const auto ids = ids_for(names); Bytes out{'O', 'P', 'A', '1', 1, static_cast<std::uint8_t>(ids.size())};
  append(out, ids); append(out, serialized(raw, names)); return out;
}
Bytes encode(std::span<const std::uint8_t> raw, const Names& names) { return compress(raw_encode(raw, names)); }
Bytes legacy_encode(std::span<const std::uint8_t> raw, const Names& names) {
  require(!names.empty(), "invalid legacy pipeline depth");
  const auto ids = ids_for(names);
  for (auto id : ids) require(id < 28, "unknown legacy operator");
  auto out = raw_encode(raw, names); out[2] = 'C'; return compress(out);
}
Bytes legacy_decode(std::span<const std::uint8_t> wire) {
  const auto raw = decompress(wire); const auto [ids, offset] = read_header(raw, "OPC1");
  for (auto id : ids) require(id < 28, "invalid legacy operator ID");
  return inverse_state(std::span(raw).subspan(offset), ids);
}
Bytes decode(std::span<const std::uint8_t> wire) {
  const auto raw = decompress(wire); const auto [ids, offset] = read_header(raw);
  return inverse_state(std::span(raw).subspan(offset), ids);
}
Names actions(std::span<const std::uint8_t> wire) {
  const auto raw = decompress(wire); const auto [ids, offset] = read_header(raw); Names names;
  for (auto id : ids) names.push_back(wire_actions()[id]); return names;
}
}  // namespace reference_operator_actions
