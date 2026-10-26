#include "features.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace wc::features_v1 {
namespace {
void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(message);
}
void validate(const FlowView& flow) {
  require(flow.kind == "serial" || flow.kind == "numeric" || flow.kind == "struct",
          "unknown physical flow kind");
  require(flow.width > 0 && flow.width <= 65536 && flow.bytes.size() % flow.width == 0,
          "invalid physical width or partial element");
  require(flow.kind != "serial" || flow.width == 1, "serial width must be one");
  require(flow.kind != "numeric" || flow.width == 1 || flow.width == 2 ||
              flow.width == 4 || flow.width == 8, "invalid numeric width");
  const std::array<std::pair<const char*, unsigned>, 13> views{{
      {"int8", 1}, {"uint8", 1}, {"int16", 2}, {"uint16", 2},
      {"int32", 4}, {"uint32", 4}, {"int64", 8}, {"uint64", 8},
      {"float16", 2}, {"float32", 4}, {"float64", 8},
      {"xor_residual32", 4}, {"xor_residual64", 8}}};
  if (flow.logical_view == "bytes") return;
  for (const auto& [name, width] : views) {
    if (flow.logical_view == name) {
      require(flow.kind == "numeric" && flow.width == width,
              "logical view does not match the physical flow");
      return;
    }
  }
  throw std::invalid_argument("unknown logical view");
}
std::uint64_t word(std::span<const std::uint8_t> bytes) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
  return value;
}
double half(std::uint16_t value) {
  const int exponent = (value >> 10) & 31;
  const int fraction = value & 1023;
  double out;
  if (exponent == 31) out = fraction ? std::numeric_limits<double>::quiet_NaN()
                                    : std::numeric_limits<double>::infinity();
  else if (!exponent) out = std::ldexp(double(fraction), -24);
  else out = std::ldexp(double(1024 + fraction), exponent - 25);
  return value & 0x8000 ? -out : out;
}
enum class NumberKind { f64, f32, f16, i64, i32, i16, i8, unsigned_word };
NumberKind number_kind(const std::string& view) {
  if (view == "float64") return NumberKind::f64;
  if (view == "float32") return NumberKind::f32;
  if (view == "float16") return NumberKind::f16;
  if (view == "int64") return NumberKind::i64;
  if (view == "int32") return NumberKind::i32;
  if (view == "int16") return NumberKind::i16;
  if (view == "int8") return NumberKind::i8;
  return NumberKind::unsigned_word;
}
template <typename T> T little_word(const std::uint8_t* data) {
  if constexpr (std::endian::native == std::endian::little) {
    T value; std::memcpy(&value, data, sizeof(T)); return value;
  } else {
    return static_cast<T>(word({data, sizeof(T)}));
  }
}
double number(const FlowView& flow, std::size_t index, NumberKind kind) {
  const auto* data = flow.bytes.data() + index * flow.width;
  switch (kind) {
    case NumberKind::f64: return std::bit_cast<double>(little_word<std::uint64_t>(data));
    case NumberKind::f32: return std::bit_cast<float>(little_word<std::uint32_t>(data));
    case NumberKind::f16: return half(little_word<std::uint16_t>(data));
    case NumberKind::i64: return double(std::bit_cast<std::int64_t>(little_word<std::uint64_t>(data)));
    case NumberKind::i32: return double(std::bit_cast<std::int32_t>(little_word<std::uint32_t>(data)));
    case NumberKind::i16: return double(std::bit_cast<std::int16_t>(little_word<std::uint16_t>(data)));
    case NumberKind::i8: return double(std::bit_cast<std::int8_t>(*data));
    case NumberKind::unsigned_word: return double(word({data, flow.width}));
  }
  throw std::invalid_argument("unknown numeric feature kind");
}
std::size_t finite_unique_count(std::span<const double> values) {
  require(values.size() <= 8192, "unique-feature sample bound exceeded");
  if (values.empty()) return 0;
  // Numeric equality on finite binary64 values is bit identity except +/-0.
  // A flat bounded set avoids sorting and per-value node allocations. This
  // changes neither the sampled values nor the original feature definition.
  std::size_t capacity = 2;
  while (capacity < values.size() * 2) capacity *= 2;
  std::vector<std::uint64_t> keys(capacity);
  std::vector<std::uint8_t> occupied(capacity, 0);
  std::size_t count = 0;
  for (const double value : values) {
    const std::uint64_t key = value == 0.0 ? 0 : std::bit_cast<std::uint64_t>(value);
    auto hash = key;
    hash = (hash ^ (hash >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    hash = (hash ^ (hash >> 27)) * UINT64_C(0x94d049bb133111eb);
    hash ^= hash >> 31;
    auto slot = static_cast<std::size_t>(hash) & (capacity - 1);
    while (occupied[slot] && keys[slot] != key) slot = (slot + 1) & (capacity - 1);
    if (!occupied[slot]) { occupied[slot] = 1; keys[slot] = key; ++count; }
  }
  return count;
}
double median(std::vector<double> values) {
  require(!values.empty(), "empty median");
  const auto middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2) return upper;
  const double lower = *std::max_element(values.begin(), values.begin() + middle);
  // Deliberately match NumPy's binary64 even-length mean, including overflow.
  return (lower + upper) / 2.0;
}
template <std::size_t N> void finite(const std::array<double, N>& values) {
  for (const double value : values) require(std::isfinite(value), "non-finite feature");
}
}  // namespace

std::array<double, task_width> task(std::span<const FlowView> flows) {
  require(!flows.empty() && flows.size() <= 32, "invalid root flow count");
  std::array<std::uint64_t, 256> histogram{};
  std::uint64_t byte_count = 0, equal_bytes = 0, total = 0, width_sum = 0;
  unsigned maximum_width = 0, previous = 0;
  std::uint32_t serial = 0, numeric_kind = 0, structs = 0;
  std::array<unsigned, 4> views{};
  std::size_t numeric_sample_capacity = 0;
  for (const auto& flow : flows) {
    if (flow.width && flow.logical_view != "bytes" &&
        !flow.logical_view.starts_with("xor_residual"))
      numeric_sample_capacity += std::min<std::size_t>(flow.bytes.size() / flow.width, 8192);
  }
  // Stream the finite subset directly into its final storage.  This preserves
  // sample order and every arithmetic operation while avoiding a second
  // sample-sized allocation and filtering pass.
  std::vector<double> values;
  values.reserve(numeric_sample_capacity);
  std::size_t sampled_values = 0, sampled_zero_values = 0;
  for (const auto& flow : flows) {
    validate(flow);
    require(total <= UINT64_MAX - flow.bytes.size(), "root byte total overflow");
    total += flow.bytes.size(); width_sum += flow.width;
    maximum_width = std::max(maximum_width, flow.width);
    serial += flow.kind == "serial"; numeric_kind += flow.kind == "numeric";
    structs += flow.kind == "struct";
    views[0] += flow.logical_view == "int64"; views[1] += flow.logical_view == "float64";
    views[2] += flow.logical_view == "float32"; views[3] += flow.logical_view == "uint32";
    for (std::size_t i = 0; i < flow.bytes.size() && byte_count < 65536; ++i) {
      const auto value = flow.bytes[i];
      equal_bytes += byte_count > 0 && value == previous;
      previous = value; ++histogram[value]; ++byte_count;
    }
    if (flow.logical_view == "bytes" || flow.logical_view.starts_with("xor_residual")) continue;
    const auto count = flow.bytes.size() / flow.width;
    const auto sample_count = std::min<std::size_t>(count, 8192);
    const auto kind = number_kind(flow.logical_view);
    const double step = count > 8192 ? double(count - 1) / 8191.0 : 1.0;
    for (std::size_t i = 0; i < sample_count; ++i) {
      const auto index = count > 8192 ? (i == 8191 ? count - 1 : std::size_t(double(i) * step)) : i;
      const auto value = number(flow, index, kind);
      ++sampled_values;
      if (std::isfinite(value)) {
        values.push_back(value);
        sampled_zero_values += value == 0.0;
      }
    }
  }
  double entropy = 0;
  if (byte_count) for (const auto count : histogram) if (count) {
    const double p = double(count) / double(byte_count);
    entropy -= p * std::log2(p) / 8.0;
  }
  double zero_values = 0, unique_fraction = 1, smoothness = 0;
  if (!values.empty()) {
    zero_values = double(sampled_zero_values) / double(values.size());
    const std::span<const double> unique(values.data(), std::min<std::size_t>(8192, values.size()));
    unique_fraction = double(finite_unique_count(unique)) / double(unique.size());
    smoothness = 1;
    if (values.size() > 1) {
      std::vector<double> absolute, differences;
      absolute.reserve(values.size()); differences.reserve(values.size() - 1);
      for (std::size_t i = 0; i < values.size(); ++i) {
        absolute.push_back(std::abs(values[i]));
        if (i) differences.push_back(std::abs(values[i] - values[i - 1]));
      }
      smoothness = 1.0 / (1.0 + median(std::move(differences)) /
                                      (median(std::move(absolute)) + 1.0e-12));
    }
  }
  const auto count = double(flows.size());
  std::array<double, task_width> result{
      std::log2(double(total) + 1.0) / 32.0, count / 32.0,
      double(width_sum) / count / 8.0, double(maximum_width) / 8.0,
      serial / count, numeric_kind / count, structs / count,
      views[0] / count, views[1] / count, views[2] / count, views[3] / count,
      entropy, byte_count ? double(histogram[0]) / byte_count : 0.0,
      byte_count > 1 ? double(equal_bytes) / double(byte_count - 1) : 0.0,
      zero_values, 1.0 - unique_fraction, smoothness,
      sampled_values == 0 ? 0.0 :
          double(sampled_values - values.size()) / double(sampled_values)};
  finite(result); return result;
}

std::array<double, state_width> state(const State& input) {
  require(!input.flows.empty() && input.flows.size() <= input.max_active,
          "invalid active flow count");
  require(input.original_numeric_bytes && input.max_nodes && input.max_depth &&
              input.max_active && input.max_policy_edits && input.max_recipe_edits &&
              input.candidate_limit && input.action_count, "zero state denominator");
  require(input.max_nodes <= 64 && input.max_depth <= 10 && input.max_active <= 32,
          "graph limit exceeds frozen feature contract");
  double maximum_depth = 0, depth_sum = 0, byte_sum = 0, width_sum = 0;
  unsigned serial = 0, numeric = 0;
  std::array<double, 6> parents{}, sizes{};
  const auto count = double(input.flows.size());
  const auto raw = double(input.original_numeric_bytes);
  for (std::size_t i = 0; i < input.flows.size(); ++i) {
    const auto& active = input.flows[i]; validate(active.flow);
    require(active.parent_action >= -1 && active.parent_action < input.action_count,
            "unknown parent action");
    require(active.depth <= input.max_depth, "flow depth exceeds graph limit");
    maximum_depth = std::max(maximum_depth, double(active.depth));
    depth_sum += active.depth; byte_sum += active.flow.bytes.size(); width_sum += active.flow.width;
    serial += active.flow.kind == "serial"; numeric += active.flow.kind == "numeric";
    parents[i % 6] += (double(active.parent_action + 1) / input.action_count) / count;
    sizes[i % 6] += std::log1p(double(active.flow.bytes.size()) / raw);
  }
  double pending = 0;
  if (input.pending_edit_kind == "insert") pending = .25;
  else if (input.pending_edit_kind == "delete") pending = .5;
  else if (input.pending_edit_kind == "replace") pending = .75;
  else if (input.pending_edit_kind == "parameter_change") pending = 1;
  else require(input.pending_edit_kind.empty(), "unknown edit kind");
  require(!input.pending_edit_index || *input.pending_edit_index <= 64, "edit index outside graph");
  double objective = -1;
  if (input.current_objective) {
    require(std::isfinite(*input.current_objective) && *input.current_objective >= 0,
            "invalid current objective");
    objective = *input.current_objective / (1 + *input.current_objective);
  }
  const double edit_limit = input.continuation_edit_mode ? input.max_recipe_edits : input.max_policy_edits;
  std::array<double, state_width> result{
      double(input.node_count) / input.max_nodes, maximum_depth / input.max_depth,
      depth_sum / count / input.max_depth, count / input.max_active, byte_sum / raw,
      serial / count, numeric / count, width_sum / count / 8.0,
      input.policy_edits / edit_limit, double(std::min(input.candidate_used, input.candidate_limit)) / input.candidate_limit,
      double(input.replayed_nodes) / input.max_nodes, double(input.continuation_edit_mode),
      double(input.edit_count) / input.max_recipe_edits, pending,
      input.pending_edit_index ? double(*input.pending_edit_index + 1) / 65.0 : 0.0,
      objective};
  std::copy(parents.begin(), parents.end(), result.begin() + 16);
  std::copy(sizes.begin(), sizes.end(), result.begin() + 22);
  finite(result); return result;
}

std::array<double, observation_width> observation(
    std::span<const FlowView> roots, const State& input, const std::array<double, 3>& weights) {
  double sum = 0;
  for (const double weight : weights) {
    require(std::isfinite(weight) && weight >= 0, "invalid objective weight"); sum += weight;
  }
  require(std::abs(sum - 1.0) <= 1.0e-8, "objective weights must sum to one");
  const auto a = task(roots);
  const auto b = state(input);
  std::array<double, observation_width> output{};
  std::copy(a.begin(), a.end(), output.begin());
  std::copy(b.begin(), b.end(), output.begin() + task_width);
  std::copy(weights.begin(), weights.end(), output.begin() + task_width + state_width);
  return output;
}
}  // namespace wc::features_v1
