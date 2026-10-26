#include "composite.hpp"

#include <openssl/evp.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace native_composite_candidate_v1 {
namespace {

constexpr std::size_t kDigestBytes = 32;
constexpr std::size_t kFixedHeaderBytes = 84;
constexpr std::size_t kTrailerBytes = kDigestBytes;
constexpr std::uint64_t kMaxFrameBytes = 1ULL << 30;
constexpr std::uint32_t kMaxSelectorIdentityBytes = 512;

void require(bool good, const char* message) {
  if (!good) throw std::invalid_argument(message);
}

std::array<std::uint8_t, kDigestBytes> sha256(
    std::span<const std::uint8_t> input) {
  std::array<std::uint8_t, kDigestBytes> result{};
  unsigned size = 0;
  if (EVP_Digest(input.data(), input.size(), result.data(), &size,
                 EVP_sha256(), nullptr) != 1 || size != result.size())
    throw std::runtime_error("OpenSSL SHA-256 failed");
  return result;
}

std::string hex(std::span<const std::uint8_t> input) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(input.size() * 2);
  for (const auto byte : input) {
    output.push_back(digits[byte >> 4]);
    output.push_back(digits[byte & 15]);
  }
  return output;
}

struct Writer {
  Bytes bytes;
  void number(std::uint64_t value, unsigned width) {
    require(width <= 8, "invalid integer width");
    require(bytes.size() <= kMaxFrameBytes - width, "NCC1 frame exceeds limit");
    for (unsigned index = 0; index < width; ++index)
      bytes.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
  }
  void append(std::span<const std::uint8_t> value) {
    require(value.size() <= kMaxFrameBytes - bytes.size(),
            "NCC1 frame exceeds limit");
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  void text(const std::string& value) {
    require(value.size() <= kMaxSelectorIdentityBytes,
            "selector identity exceeds limit");
    number(value.size(), 4);
    append({reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
  }
};

struct Reader {
  std::span<const std::uint8_t> bytes;
  std::size_t position = 0;
  std::span<const std::uint8_t> take(std::uint64_t count) {
    require(position <= bytes.size() && count <= bytes.size() - position,
            "truncated NCC1 frame");
    auto result = bytes.subspan(position, static_cast<std::size_t>(count));
    position += static_cast<std::size_t>(count);
    return result;
  }
  std::uint64_t number(unsigned width) {
    auto value = take(width);
    std::uint64_t result = 0;
    for (unsigned index = 0; index < width; ++index)
      result |= std::uint64_t(value[index]) << (8 * index);
    return result;
  }
  std::string text() {
    const auto count = number(4);
    require(count <= kMaxSelectorIdentityBytes, "selector identity exceeds limit");
    auto value = take(count);
    return {reinterpret_cast<const char*>(value.data()), value.size()};
  }
};

struct InnerIdentity {
  std::string action;
  std::string profile;
  std::string configuration;
  std::uint64_t row_count = 0;
  std::uint64_t original_numeric_bytes = 0;
};

// Read only the authenticated NGF1 identity prefix. The outer NCC1 checksum
// already covers every inner byte; baseline86::decode subsequently performs
// the complete NGF1 checksum, structural and inverse-dispatch validation.
InnerIdentity inspect_inner_identity(std::span<const std::uint8_t> frame) {
  require(frame.size() >= 48, "truncated inner NGF1 frame");
  Reader header{frame};
  const auto magic = header.take(4);
  require(std::memcmp(magic.data(), "NGF1", 4) == 0,
          "NCC1 inner payload is not NGF1");
  require(header.number(2) == 1 && header.number(2) == 0,
          "unsupported inner NGF1 version or flags");
  const auto body_size = header.number(8);
  require(body_size == frame.size() - 48, "inner NGF1 length mismatch");
  Reader input{frame.subspan(16, static_cast<std::size_t>(body_size))};
  const auto dispatcher = input.text();
  const auto registry = input.text();
  require(dispatcher == "native-baseline86-full-frame-dispatch-v1",
          "inner NGF1 dispatcher is outside baseline86");
  require(registry == native_baseline86_v1::profile_version,
          "inner baseline86 profile identity mismatch");
  InnerIdentity result;
  result.original_numeric_bytes = input.number(8);
  const auto roots = input.number(4);
  require(roots == 2, "inner baseline86 root count mismatch");
  for (std::uint64_t index = 0; index < roots; ++index) {
    (void)input.take(13);  // kind(1), width(4), byte count(8)
    (void)input.take(1);   // original-numeric flag
    (void)input.text();
    (void)input.take(kDigestBytes);
  }
  require(input.number(4) == 1, "inner baseline86 node count mismatch");
  require(input.number(1) == 1, "inner baseline86 node is not a stage");
  require(input.number(8) == 0, "inner baseline86 policy action ID mismatch");
  result.action = input.text();
  result.profile = input.text();
  const auto parameter_count = input.number(4);
  require(parameter_count == 2, "inner baseline86 parameter count mismatch");
  for (std::uint64_t index = 0; index < parameter_count; ++index) {
    const auto key = input.text();
    const auto value = input.text();
    if (key == "configuration") result.configuration = value;
    else if (key == "row_count") {
      std::size_t used = 0;
      result.row_count = std::stoull(value, &used);
      require(used == value.size(), "invalid inner baseline86 row count");
    } else {
      throw std::invalid_argument("unknown inner baseline86 parameter");
    }
  }
  require(!result.configuration.empty(),
          "inner baseline86 configuration is absent");
  return result;
}

std::string checked_selector_identity(const Selector& selector) {
  auto identity = selector.identity();
  require(!identity.empty() && identity.size() <= kMaxSelectorIdentityBytes &&
              identity.find('\0') == std::string::npos,
          "invalid selector identity");
  return identity;
}

void validate_weights(const Weights& weights) {
  double sum = 0;
  for (const auto value : weights) {
    require(std::isfinite(value) && value >= 0.0 && value <= 1.0,
            "invalid objective weight");
    sum += value;
  }
  require(std::abs(sum - 1.0) <= 1e-12, "objective weights must sum to one");
}

CandidateRecord make_record(Backend backend, std::string label,
                            std::string thread_contract) {
  const auto& source = native_baseline86_v1::method(label);
  return {backend,
          std::move(label),
          source.implementation,
          source.source,
          source.configuration,
          source.implementation_class,
          std::move(thread_contract),
          source.linked};
}

std::string registry_material() {
  std::string result = registry_version;
  result.push_back('\n');
  result += native_baseline86_v1::profile_version;
  result.push_back('\n');
  for (const auto& item : inventory()) {
    result += std::to_string(static_cast<std::uint16_t>(item.backend));
    for (const auto* value : {&item.label, &item.implementation, &item.source,
                              &item.configuration, &item.implementation_class,
                              &item.thread_contract}) {
      result.push_back('\0');
      result += *value;
    }
    result.push_back('\0');
    result += item.linked ? "linked" : "unlinked";
    result.push_back('\n');
  }
  return result;
}

std::array<std::uint8_t, kDigestBytes> registry_digest() {
  const auto material = registry_material();
  return sha256({reinterpret_cast<const std::uint8_t*>(material.data()),
                 material.size()});
}

Backend parse_backend(std::uint64_t value) {
  require(value >= static_cast<std::uint16_t>(Backend::Xpack) &&
              value <= static_cast<std::uint16_t>(Backend::Zstd),
          "unknown NCC1 backend ID");
  return static_cast<Backend>(value);
}

std::uint64_t raw_word(std::span<const std::uint8_t> bytes) {
  std::uint64_t result = 0;
  for (std::size_t index = 0; index < bytes.size(); ++index)
    result |= std::uint64_t(bytes[index]) << (8 * index);
  return result;
}

double numeric_value(const Block& block, std::size_t index) {
  const auto width = native_baseline86_v1::width(block.dtype);
  const auto word = raw_word({block.values.data() + index * width, width});
  if (block.dtype == "float64") return std::bit_cast<double>(word);
  if (block.dtype == "float32")
    return std::bit_cast<float>(static_cast<std::uint32_t>(word));
  if (block.dtype == "int64") return static_cast<double>(std::bit_cast<std::int64_t>(word));
  if (block.dtype == "int32") return static_cast<double>(std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(word)));
  if (block.dtype == "int16") return static_cast<double>(std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(word)));
  if (block.dtype == "int8") return static_cast<double>(std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(word)));
  return static_cast<double>(word);
}

double median(std::vector<double> values) {
  if (values.empty()) return 0;
  const auto middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const auto upper = values[middle];
  if (values.size() % 2) return upper;
  const auto lower = *std::max_element(values.begin(), values.begin() + middle);
  return (lower + upper) / 2.0;
}

double stable_dot(std::span<const double> left, std::span<const double> right) {
  require(left.size() == right.size(), "dot-product shape mismatch");
  double sum = 0, correction = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto value = left[index] * right[index];
    const auto next = sum + value;
    if (std::abs(sum) >= std::abs(value)) correction += (sum - next) + value;
    else correction += (value - next) + sum;
    sum = next;
  }
  return sum + correction;
}

template <std::size_t Size>
std::array<double, Size> json_array(const nlohmann::json& value,
                                   const char* description) {
  require(value.is_array() && value.size() == Size, description);
  std::array<double, Size> result{};
  for (std::size_t index = 0; index < Size; ++index) {
    result[index] = value[index].get<double>();
    require(std::isfinite(result[index]), description);
  }
  return result;
}

void require_sha256_text(const std::string& value, const char* description) {
  require(value.size() == 64 &&
              std::all_of(value.begin(), value.end(), [](unsigned char byte) {
                return (byte >= '0' && byte <= '9') ||
                       (byte >= 'a' && byte <= 'f');
              }),
          description);
}

}  // namespace

const std::vector<CandidateRecord>& inventory() {
  static const auto records = std::vector<CandidateRecord>{
      make_record(Backend::Xpack, "XPACK",
                  "single calling thread; one upstream context; no worker pool"),
      make_record(Backend::Blosc2, "Blosc2",
                  "cparams.nthreads=1 and dparams.nthreads=1"),
      make_record(Backend::Blosc, "Blosc",
                  "blosc_set_nthreads(1) for encode and decode"),
      make_record(Backend::Libdeflate, "Libdeflate",
                  "single calling thread; raw deflate buffer API has no worker pool"),
      make_record(Backend::Zstd, "Zstd",
                  "single calling thread; ZSTD_compress/ZSTD_decompress buffer API"),
  };
  return records;
}

const CandidateRecord& candidate(Backend backend) {
  for (const auto& item : inventory())
    if (item.backend == backend) return item;
  throw std::invalid_argument("unknown composite candidate backend");
}

Backend backend_from_label(std::string_view label) {
  for (const auto& item : inventory())
    if (item.label == label) return item.backend;
  throw std::invalid_argument("label is outside composite candidate registry");
}

std::string registry_sha256() {
  const auto digest = registry_digest();
  return hex(digest);
}

TaskFeatures task_features(const Block& block) {
  native_baseline86_v1::validate(block);
  const auto width = native_baseline86_v1::width(block.dtype);
  const auto value_count = block.values.size() / width;
  const auto sample_limit = std::max<std::size_t>(1, 3072 / width);
  const auto sample_count = std::min(value_count, sample_limit);
  std::array<std::uint64_t, 256> histogram{};
  std::vector<std::uint64_t> raw_values;
  std::vector<double> finite_values;
  raw_values.reserve(sample_count);
  finite_values.reserve(sample_count);
  std::uint64_t sampled_bytes = 0, zero_bytes = 0, ff_bytes = 0, equal_bytes = 0;
  std::uint8_t previous_byte = 0;
  std::size_t numeric_zero = 0, numeric_negative = 0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const auto index = value_count <= sample_limit
                           ? sample
                           : (sample + 1 == sample_count
                                  ? value_count - 1
                                  : (sample * (value_count - 1)) / (sample_count - 1));
    const auto bytes = std::span<const std::uint8_t>(
        block.values.data() + index * width, width);
    raw_values.push_back(raw_word(bytes));
    for (const auto byte : bytes) {
      ++histogram[byte];
      zero_bytes += byte == 0;
      ff_bytes += byte == 255;
      equal_bytes += sampled_bytes && byte == previous_byte;
      previous_byte = byte;
      ++sampled_bytes;
    }
    const auto value = numeric_value(block, index);
    numeric_zero += value == 0.0;
    numeric_negative += std::isfinite(value) && std::signbit(value) && value != 0.0;
    if (std::isfinite(value)) finite_values.push_back(value);
  }
  double entropy = 0;
  std::size_t unique_bytes = 0;
  if (sampled_bytes) {
    for (const auto count : histogram) {
      unique_bytes += count != 0;
      if (!count) continue;
      const auto probability = static_cast<double>(count) / sampled_bytes;
      entropy -= probability * std::log2(probability) / 8.0;
    }
  }
  std::sort(raw_values.begin(), raw_values.end());
  const auto unique_numeric =
      std::distance(raw_values.begin(),
                    std::unique(raw_values.begin(), raw_values.end()));
  std::vector<double> magnitudes, deltas;
  magnitudes.reserve(finite_values.size());
  if (finite_values.size() > 1) deltas.reserve(finite_values.size() - 1);
  std::size_t zero_deltas = 0, monotonic_deltas = 0;
  for (std::size_t index = 0; index < finite_values.size(); ++index) {
    magnitudes.push_back(std::abs(finite_values[index]));
    if (!index) continue;
    const auto delta = finite_values[index] - finite_values[index - 1];
    if (!std::isfinite(delta)) continue;
    deltas.push_back(std::abs(delta));
    zero_deltas += delta == 0.0;
    monotonic_deltas += delta >= 0.0;
  }
  const auto valid_values = static_cast<double>(value_count);
  const auto samples = static_cast<double>(sample_count);
  const auto finite_count = static_cast<double>(finite_values.size());
  const auto delta_count = static_cast<double>(deltas.size());
  const auto missing = block.row_count - value_count;
  const auto smoothness = deltas.empty()
      ? (finite_values.empty() ? 0.0 : 1.0)
      : 1.0 / (1.0 + median(std::move(deltas)) /
                         (median(std::move(magnitudes)) + 1e-12));
  TaskFeatures result{
      std::log2(static_cast<double>(block.values.size()) + 1.0) / 32.0,
      std::log2(static_cast<double>(block.row_count) + 1.0) / 32.0,
      static_cast<double>(width) / 8.0,
      block.row_count ? valid_values / block.row_count : 0.0,
      block.row_count ? static_cast<double>(missing) / block.row_count : 0.0,
      entropy,
      sampled_bytes ? static_cast<double>(zero_bytes) / sampled_bytes : 0.0,
      sampled_bytes ? static_cast<double>(ff_bytes) / sampled_bytes : 0.0,
      sampled_bytes > 1 ? static_cast<double>(equal_bytes) / (sampled_bytes - 1) : 0.0,
      static_cast<double>(unique_bytes) / 256.0,
      sample_count ? static_cast<double>(numeric_zero) / samples : 0.0,
      sample_count ? finite_count / samples : 0.0,
      sample_count ? static_cast<double>(numeric_negative) / samples : 0.0,
      sample_count ? static_cast<double>(unique_numeric) / samples : 0.0,
      delta_count ? static_cast<double>(zero_deltas) / delta_count : 0.0,
      smoothness,
      delta_count ? static_cast<double>(monotonic_deltas) / delta_count : 0.0,
      value_count ? samples / valid_values : 0.0};
  for (const auto value : result)
    require(std::isfinite(value), "non-finite composite task feature");
  return result;
}

std::string FixedSelector::identity() const {
  return "fixed-composite-candidate/" + candidate(backend_).label;
}

ObjectiveLinearSelector::ObjectiveLinearSelector(std::string identity,
                                                 std::vector<Arm> arms)
    : identity_(std::move(identity)), arms_(std::move(arms)) {
  require(!identity_.empty() && identity_.size() <= kMaxSelectorIdentityBytes &&
              identity_.find('\0') == std::string::npos,
          "invalid objective selector identity");
  require(!arms_.empty() && arms_.size() <= inventory().size(),
          "invalid objective selector arm count");
  std::vector<Backend> seen;
  for (const auto& arm : arms_) {
    (void)candidate(arm.backend);
    require(std::find(seen.begin(), seen.end(), arm.backend) == seen.end(),
            "duplicate objective selector arm");
    seen.push_back(arm.backend);
    for (const auto value : arm.coefficients)
      require(std::isfinite(value) && value >= 0.0,
              "invalid objective selector coefficient");
  }
}

Backend ObjectiveLinearSelector::select(const TaskFeatures&,
                                        const Weights& weights) const {
  validate_weights(weights);
  const auto score = [&](const Arm& arm) {
    return std::inner_product(weights.begin(), weights.end(),
                              arm.coefficients.begin(), 0.0);
  };
  return std::min_element(arms_.begin(), arms_.end(),
                          [&](const Arm& left, const Arm& right) {
                            const auto a = score(left), b = score(right);
                            return a == b
                                ? static_cast<std::uint16_t>(left.backend) <
                                      static_cast<std::uint16_t>(right.backend)
                                : a < b;
                          })->backend;
}

ObjectiveLinearSelector train_only_objective_linear_v1() {
  using Arm = ObjectiveLinearSelector::Arm;
  return ObjectiveLinearSelector(
      "train-only-direct-codec-linear-selector-v1/"
      "c76f74104ec651705f3e772137280e45d1f9746a44c2714c5f5df50c422cca5b",
      {Arm{Backend::Blosc2,
           {0.3403064291717435, 0.14338201512257506, 0.15690049941381037}},
       Arm{Backend::Blosc,
           {0.3415070231101138, 0.14372975034602783, 0.15483851678947194}},
       Arm{Backend::Xpack,
           {0.3073575774391887, 0.49212542252832064, 0.2553066455646457}},
       Arm{Backend::Libdeflate,
           {0.33824905802259136, 0.4733200531786582, 0.22947738292191802}},
       Arm{Backend::Zstd,
           {0.34237686937779227, 0.2740297532701376, 0.2400913626869967}}});
}

DataAwareLinearSelector DataAwareLinearSelector::load(
    const std::string& path, const std::string& expected_file_sha256) {
  require_sha256_text(expected_file_sha256, "invalid expected model SHA-256");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::invalid_argument("cannot open data-aware selector model");
  const std::string payload{std::istreambuf_iterator<char>(input), {}};
  const auto actual_sha = hex(sha256({
      reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()}));
  require(actual_sha == expected_file_sha256,
          "data-aware selector model SHA-256 mismatch");
  const auto root = nlohmann::json::parse(payload);
  require(root.value("schema", "") ==
              "train-only-direct-codec-data-aware-selector-v1",
          "unknown data-aware selector schema");
  const auto& feature = root.at("feature_contract");
  require(feature.value("schema", "") == feature_schema &&
              feature.value("maximum_sample_bytes", 0) == 3072 &&
              feature.value("byte_order", "") == "little",
          "data-aware selector feature contract mismatch");
  static const std::array<const char*, 18> feature_order{
      "log2_value_bytes_plus_1_over_32",
      "log2_row_count_plus_1_over_32",
      "value_width_over_8", "valid_value_fraction", "missing_fraction",
      "sample_byte_entropy_over_8", "sample_zero_byte_fraction",
      "sample_ff_byte_fraction", "sample_adjacent_equal_byte_fraction",
      "sample_unique_byte_count_over_256", "sample_numeric_zero_fraction",
      "sample_finite_fraction", "sample_finite_negative_fraction",
      "sample_unique_raw_value_fraction",
      "sample_filtered_finite_zero_delta_fraction",
      "sample_filtered_finite_smoothness",
      "sample_filtered_finite_nondecreasing_delta_fraction",
      "sample_value_fraction"};
  require(feature.at("feature_order").is_array() &&
              feature.at("feature_order").size() == feature_order.size(),
          "data-aware selector feature order width mismatch");
  for (std::size_t index = 0; index < feature_order.size(); ++index)
    require(feature.at("feature_order")[index].get<std::string>() ==
                feature_order[index],
            "data-aware selector feature order mismatch");
  const auto& training = root.at("training_identity");
  require(training.value("sample_count", 0) == 40 &&
              training.value("validation_used_for_fit", true) == false &&
              training.value("web20_used_for_fit", true) == false &&
              training.value("independent_test_used_for_fit", true) == false,
          "data-aware selector is not bound to isolated train-only fitting");

  DataAwareLinearSelector result;
  const auto parameter_sha = root.at("model_parameter_sha256").get<std::string>();
  require_sha256_text(parameter_sha, "invalid model-parameter SHA-256");
  result.identity_ = "train-only-direct-codec-data-aware-selector-v1/" +
                     actual_sha + "/parameters/" + parameter_sha;
  const auto& parameters = root.at("parameters");
  result.mean_ = json_array<18>(parameters.at("feature_mean"),
                                "invalid feature mean");
  result.scale_ = json_array<18>(parameters.at("feature_scale"),
                                 "invalid feature scale");
  for (const auto value : result.scale_)
    require(value > 0.0, "nonpositive feature scale");
  const auto& clamp = parameters.at("prediction_clamp");
  for (std::size_t metric = 0; metric < 3; ++metric) {
    const auto* key = metric == 0 ? "R" :
                      metric == 1 ? "E_over_E_plus_E0" : "D_over_D_plus_D0";
    const auto bounds = json_array<2>(clamp.at(key), "invalid prediction clamp");
    require(bounds[0] >= 0.0 && bounds[0] < bounds[1],
            "invalid prediction clamp order");
    result.clamp_low_[metric] = bounds[0];
    result.clamp_high_[metric] = bounds[1];
  }
  const auto& order = parameters.at("candidate_order");
  require(order.is_array() && order.size() == 5,
          "invalid data-aware candidate order");
  const auto& priors = parameters.at("prior_coefficients");
  const auto& residuals = parameters.at("residual_coefficients");
  for (const auto& label_value : order) {
    const auto label = label_value.get<std::string>();
    Arm arm;
    arm.backend = backend_from_label(label);
    require(std::none_of(result.arms_.begin(), result.arms_.end(),
                         [&](const auto& existing) {
                           return existing.backend == arm.backend;
                         }),
            "duplicate data-aware candidate");
    arm.prior = json_array<3>(priors.at(label), "invalid candidate prior");
    const auto& matrix = residuals.at(label);
    require(matrix.is_array() && matrix.size() == 3,
            "invalid candidate residual matrix");
    for (std::size_t metric = 0; metric < 3; ++metric)
      arm.residual[metric] =
          json_array<18>(matrix[metric], "invalid residual coefficient row");
    result.arms_.push_back(std::move(arm));
  }
  require(result.arms_.size() == inventory().size(),
          "data-aware model omits a composite candidate");
  return result;
}

std::vector<DataAwareLinearSelector::Prediction>
DataAwareLinearSelector::predict(const TaskFeatures& task,
                                 const Weights& weights) const {
  validate_weights(weights);
  std::array<double, 18> standardized{};
  for (std::size_t index = 0; index < standardized.size(); ++index) {
    require(std::isfinite(task[index]), "non-finite selector task feature");
    standardized[index] = (task[index] - mean_[index]) / scale_[index];
  }
  std::vector<Prediction> output;
  output.reserve(arms_.size());
  for (const auto& arm : arms_) {
    std::array<double, 3> predicted{};
    for (std::size_t metric = 0; metric < predicted.size(); ++metric) {
      predicted[metric] = std::clamp(
          arm.prior[metric] + stable_dot(arm.residual[metric], standardized),
          clamp_low_[metric], clamp_high_[metric]);
    }
    const auto score = stable_dot(weights, predicted);
    require(std::isfinite(score), "non-finite data-aware selector score");
    output.push_back({arm.backend, predicted, score});
  }
  require(!output.empty(), "data-aware selector produced no predictions");
  return output;
}

Backend DataAwareLinearSelector::select(const TaskFeatures& task,
                                        const Weights& weights) const {
  const auto predictions = predict(task, weights);
  // Candidate order from the authenticated model breaks exact ties.
  return std::min_element(predictions.begin(), predictions.end(),
                          [](const auto& left, const auto& right) {
                            return left.weighted_score < right.weighted_score;
                          })->backend;
}

Bytes encode(const Selector& selector, const Weights& weights, const Block& block,
             FrameMode mode) {
  validate_weights(weights);
  TaskFeatures features{};
  if (selector.uses_task_features()) features = task_features(block);
  const auto backend = selector.select(features, weights);
  const auto& record = candidate(backend);
  if (!record.linked)
    throw native_baseline86_v1::Unsupported(record.label +
                                            ": candidate backend is not linked");
  const auto inner = native_baseline86_v1::encode(record.label, block);
  if (mode == FrameMode::BaselineNgf1) return inner;
  require(mode == FrameMode::ProvenanceNcc1, "unknown composite frame mode");
  const auto selector_identity = checked_selector_identity(selector);

  Writer output;
  output.append({reinterpret_cast<const std::uint8_t*>("NCC1"), 4});
  output.number(1, 2);
  output.number(static_cast<std::uint16_t>(backend), 2);
  output.number(block.values.size(), 8);
  for (const auto weight : weights)
    output.number(std::bit_cast<std::uint64_t>(weight), 8);
  output.append(registry_digest());
  output.text(selector_identity);
  output.number(inner.size(), 8);
  output.append(inner);
  output.append(sha256(output.bytes));
  return output.bytes;
}

Decoded decode(std::span<const std::uint8_t> frame) {
  if (frame.size() >= 4 && std::memcmp(frame.data(), "NGF1", 4) == 0) {
    const auto identity = inspect_inner_identity(frame);
    constexpr std::string_view prefix = "baseline86/";
    require(identity.action.starts_with(prefix),
            "NGF1 stage is outside composite candidate registry");
    const auto backend = backend_from_label(identity.action.substr(prefix.size()));
    const auto& record = candidate(backend);
    require(record.linked && identity.configuration == record.configuration,
            "NGF1 method configuration differs from composite candidate registry");
    auto block = native_baseline86_v1::decode(frame);
    require(identity.profile == block.dtype && identity.row_count == block.row_count &&
                identity.original_numeric_bytes == block.values.size(),
            "NGF1 method identity differs from decoded block");
    return {backend, {}, {}, std::move(block),
            identity.original_numeric_bytes, false};
  }
  require(frame.size() >= kFixedHeaderBytes + kTrailerBytes,
          "truncated NCC1 frame");
  require(frame.size() <= kMaxFrameBytes, "NCC1 frame exceeds limit");
  const auto body = frame.first(frame.size() - kTrailerBytes);
  const auto expected_digest = sha256(body);
  require(std::equal(expected_digest.begin(), expected_digest.end(),
                     frame.end() - kTrailerBytes),
          "NCC1 checksum mismatch");

  Reader input{body};
  const auto magic = input.take(4);
  require(std::memcmp(magic.data(), "NCC1", 4) == 0, "invalid NCC1 magic");
  require(input.number(2) == 1, "unsupported NCC1 wire version");
  const auto backend = parse_backend(input.number(2));
  const auto original_numeric_bytes = input.number(8);
  Weights weights{};
  for (auto& weight : weights)
    weight = std::bit_cast<double>(input.number(8));
  validate_weights(weights);
  const auto encoded_registry = input.take(kDigestBytes);
  const auto expected_registry = registry_digest();
  require(std::equal(expected_registry.begin(), expected_registry.end(),
                     encoded_registry.begin()),
          "NCC1 candidate registry identity mismatch");
  const auto selector_identity = input.text();
  require(!selector_identity.empty() &&
              selector_identity.find('\0') == std::string::npos,
          "invalid selector identity");
  const auto inner_size = input.number(8);
  require(inner_size > 0 && inner_size == body.size() - input.position,
          "invalid NCC1 inner-frame length");
  const auto inner = input.take(inner_size);
  const auto inner_identity = inspect_inner_identity(inner);
  const auto& record = candidate(backend);
  require(inner_identity.action == "baseline86/" + record.label,
          "NCC1 backend differs from inner NGF1 method");
  require(inner_identity.configuration == record.configuration,
          "NCC1 backend configuration differs from inner NGF1 method");
  auto block = native_baseline86_v1::decode(inner);
  require(input.position == body.size(), "trailing NCC1 body data");
  require(block.values.size() == original_numeric_bytes,
          "NCC1 original numeric denominator mismatch");
  require(inner_identity.profile == block.dtype &&
              inner_identity.row_count == block.row_count &&
              inner_identity.original_numeric_bytes == original_numeric_bytes,
          "NCC1 inner method profile differs from decoded block");
  return {backend, weights, selector_identity, std::move(block),
          original_numeric_bytes, true};
}

Measurement measure(const Selector& selector, const Weights& weights,
                    const Block& block, unsigned warmups, unsigned repeats,
                    FrameMode mode) {
  require(warmups <= 100 && repeats > 0 && repeats <= 1000,
          "invalid measurement repeat count");
  Measurement result;
  result.original_numeric_bytes = block.values.size();
  result.selector_identity = checked_selector_identity(selector);
  result.frame_mode = mode;
  for (unsigned index = 0; index < warmups; ++index) {
    const auto frame = encode(selector, weights, block, mode);
    require(decode(frame).block == block, "warmup bit-exact roundtrip failed");
  }
  for (unsigned index = 0; index < repeats; ++index) {
    const auto start = std::chrono::steady_clock::now();
    auto frame = encode(selector, weights, block, mode);
    const auto finish = std::chrono::steady_clock::now();
    result.compression_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
            .count());
    if (index == 0) {
      const auto parsed = decode(frame);
      result.backend = parsed.backend;
      result.frame = std::move(frame);
    } else {
      require(frame == result.frame,
              "selector or complete encoding is nondeterministic");
    }
  }
  for (unsigned index = 0; index < repeats; ++index) {
    const auto start = std::chrono::steady_clock::now();
    auto restored = decode(result.frame);
    const auto finish = std::chrono::steady_clock::now();
    result.decompression_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
            .count());
    require(restored.block == block, "measured bit-exact roundtrip failed");
    require(restored.backend == result.backend,
            "measured backend identity changed");
  }
  return result;
}

Measurement measure_direct_backend(Backend backend, const Block& block,
                                   unsigned warmups, unsigned repeats) {
  require(warmups <= 100 && repeats > 0 && repeats <= 1000,
          "invalid measurement repeat count");
  const auto& record = candidate(backend);
  require(record.linked, "direct candidate backend is not linked");
  Measurement result;
  result.backend = backend;
  result.original_numeric_bytes = block.values.size();
  result.selector_identity = "direct-baseline86/" + record.label;
  result.frame_mode = FrameMode::BaselineNgf1;
  for (unsigned index = 0; index < warmups; ++index) {
    const auto frame = native_baseline86_v1::encode(record.label, block);
    require(native_baseline86_v1::decode(frame) == block,
            "direct baseline warmup roundtrip failed");
  }
  for (unsigned index = 0; index < repeats; ++index) {
    const auto start = std::chrono::steady_clock::now();
    auto frame = native_baseline86_v1::encode(record.label, block);
    const auto finish = std::chrono::steady_clock::now();
    result.compression_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
            .count());
    if (index == 0) result.frame = std::move(frame);
    else require(frame == result.frame,
                 "direct baseline complete encoding is nondeterministic");
  }
  for (unsigned index = 0; index < repeats; ++index) {
    const auto start = std::chrono::steady_clock::now();
    auto restored = native_baseline86_v1::decode(result.frame);
    const auto finish = std::chrono::steady_clock::now();
    result.decompression_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
            .count());
    require(restored == block, "direct baseline measured roundtrip failed");
  }
  return result;
}

}  // namespace native_composite_candidate_v1
