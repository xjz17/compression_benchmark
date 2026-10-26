#include "native_layout_bridge_v1.h"
#include "openzl_layout_stages.hpp"

#include <charconv>
#include <chrono>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <string_view>
#include <vector>

struct wc_layout_result_v1 {
  openzl_layout::Bytes data;
  std::string error;
};

namespace {
using namespace openzl_layout;
using Arguments = std::vector<std::string_view>;

void require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}

std::uint64_t number(std::string_view text) {
  std::uint64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  require(!text.empty() && error == std::errc{} && end == text.data() + text.size(),
          "invalid unsigned argument");
  return value;
}

std::uint32_t width(std::string_view text) {
  const auto value = number(text);
  require(value <= UINT32_MAX, "width overflows uint32");
  return static_cast<std::uint32_t>(value);
}

template <typename T> std::vector<T> list(std::string_view text) {
  std::vector<T> values;
  if (text.empty() || text == "-") return values;
  for (std::size_t start = 0;;) {
    const auto comma = text.find(',', start);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto value = number(text.substr(start, end - start));
    require(value <= std::numeric_limits<T>::max(), "list item out of range");
    values.push_back(static_cast<T>(value));
    if (comma == std::string_view::npos) return values;
    start = comma + 1;
  }
}

Kind kind(std::string_view text) {
  if (text == "serial") return Kind::Serial;
  if (text == "struct") return Kind::Struct;
  if (text == "numeric") return Kind::Numeric;
  throw std::invalid_argument("invalid data flow kind");
}

Arguments arguments(std::size_t argc, const char* const* argv) {
  require(argc > 0 && argc <= 5 && argv, "invalid argument count or pointer");
  Arguments args;
  for (std::size_t i = 0; i < argc; ++i) {
    require(argv[i], "null argument");
    args.emplace_back(argv[i]);
  }
  return args;
}

Bytes execute(const Arguments& args, const Bytes& raw) {
  const auto arity = [&](std::size_t count) {
    require(args.size() == count, "wrong number of layout arguments");
  };
  const auto mode = args[0];
  if (mode == "decode" || mode == "decode-bundle") {
    arity(1);
    auto restored = inverse(deserialize(raw));
    if (mode == "decode-bundle") return serialize(raw_bundle(std::move(restored)));
    Bytes output;
    for (const auto& flow : restored)
      output.insert(output.end(), flow.data.begin(), flow.data.end());
    return output;
  }
  require(mode == "encode" && args.size() >= 2, "expected encode or decode");
  const auto action = args[1];
  SplitResult output;
  if (action == "transpose") {
    arity(3); output = transpose({Kind::Struct, width(args[2]), raw});
  } else if (action == "split-struct") {
    arity(3); output = split_struct(raw, list<std::uint32_t>(args[2]));
  } else if (action == "float-deconstruct") {
    arity(3);
    output = float_deconstruct({Kind::Numeric, args[2] == "fp32" ? 4u : 2u, raw},
                               std::string(args[2]));
  } else if (action == "bitsplit") {
    arity(4);
    output = bitsplit({Kind::Numeric, width(args[2]), raw}, list<std::uint8_t>(args[3]));
  } else if (action == "bitsplit-top8") {
    arity(3); output = bitsplit_top8({Kind::Numeric, width(args[2]), raw});
  } else if (action == "bitsplit-fp") {
    arity(3); output = bitsplit_fp({Kind::Numeric, width(args[2]), raw});
  } else if (action == "bitsplit-bf16") {
    arity(2); output = bitsplit_bf16({Kind::Numeric, 2, raw});
  } else if (action == "bitunpack") {
    arity(3); output = bitunpack(raw, width(args[2]));
  } else if (action == "concat") {
    arity(2);
    const auto inputs = deserialize(raw);
    require(inputs.operation == Operation::Raw, "concat requires an OLS1 raw bundle");
    output = concat(inverse(inputs));
  } else if (action == "split-segments") {
    arity(5);
    output = split_segments({kind(args[2]), width(args[3]), raw}, list<std::uint64_t>(args[4]));
  } else if (action == "split-by-range") {
    require(args.size() == 3 || args.size() == 4, "invalid split-by-range arity");
    const auto minimum = args.size() == 4 ? number(args[3]) : 0;
    require(args.size() == 3 || minimum > 0, "explicit minimum must be positive");
    output = split_by_range({Kind::Numeric, width(args[2]), raw}, minimum);
  } else {
    throw std::invalid_argument("unknown layout action");
  }
  return serialize(output);
}

template <typename Function> wc_layout_result_v1* protect(Function&& function) noexcept {
  try {
    auto result = std::make_unique<wc_layout_result_v1>();
    try {
      result->data = function();
    } catch (const std::exception& error) {
      result->error = error.what();
    } catch (...) {
      result->error = "unknown native layout failure";
    }
    return result.release();
  } catch (...) {
    return nullptr;
  }
}

Bytes input(const std::uint8_t* data, std::size_t size) {
  require(size == 0 || data, "null input pointer");
  if (size == 0) return {};
  return Bytes(data, data + size);
}

std::uint64_t elapsed(std::chrono::steady_clock::time_point started) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started).count());
}

void samples(std::ostream& out, const std::vector<std::uint64_t>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}
}  // namespace

extern "C" {
std::uint32_t wc_layout_abi_version_v1() { return 1; }

wc_layout_result_v1* wc_layout_run_v1(std::size_t argc, const char* const* argv,
                                    const std::uint8_t* data, std::size_t size) {
  return protect([&] { return execute(arguments(argc, argv), input(data, size)); });
}

wc_layout_result_v1* wc_layout_benchmark_v1(
    std::size_t argc, const char* const* argv, const std::uint8_t* data,
    std::size_t size, std::uint32_t warmups, std::uint32_t repeats) {
  return protect([&] {
    require(warmups <= 31 && repeats >= 1 && repeats <= 1001, "invalid timing repetitions");
    const auto args = arguments(argc, argv);
    require(args[0] == "encode", "benchmark requires encode arguments");
    const auto raw = input(data, size);
    const Arguments decode_args{"decode"};
    Bytes expected_raw = raw;
    if (args.size() == 2 && args[1] == "concat") {
      expected_raw.clear();
      for (const auto& flow : inverse(deserialize(raw)))
        expected_raw.insert(expected_raw.end(), flow.data.begin(), flow.data.end());
    }
    Bytes frame;
    std::vector<std::uint64_t> encode_samples, decode_samples;
    for (std::uint32_t i = 0; i < warmups + repeats; ++i) {
      auto started = std::chrono::steady_clock::now();
      auto encoded = execute(args, raw);
      const auto encode_ns = elapsed(started);
      started = std::chrono::steady_clock::now();
      const auto restored = execute(decode_args, encoded);
      const auto decode_ns = elapsed(started);
      require(restored == expected_raw, "native layout bit-exact verification failed");
      if (!frame.empty()) require(frame == encoded, "nondeterministic layout frame");
      frame = std::move(encoded);
      if (i >= warmups) {
        encode_samples.push_back(encode_ns);
        decode_samples.push_back(decode_ns);
      }
    }
    std::ostringstream report;
    report << "{\"schema\":\"native-layout-stage-timing-v1\","
           << "\"whole_graph_measurement\":false,\"formal_eligible\":false,"
           << "\"stage_payload_bytes\":" << expected_raw.size()
           << ",\"complete_ols1_bytes\":" << frame.size()
           << ",\"warmups\":" << warmups << ",\"repeats\":" << repeats
           << ",\"compression_samples_ns\":";
    samples(report, encode_samples);
    report << ",\"decompression_samples_ns\":";
    samples(report, decode_samples);
    report << ",\"timing_boundary\":\"single in-process C++ layout stage with OLS1 framing; "
           << "excludes ABI transfer, graph scheduling, TGF1 framing, verification and process startup\"}";
    const auto value = report.str();
    return Bytes(value.begin(), value.end());
  });
}

const std::uint8_t* wc_layout_result_data_v1(const wc_layout_result_v1* result) {
  return result ? result->data.data() : nullptr;
}
std::size_t wc_layout_result_size_v1(const wc_layout_result_v1* result) {
  return result ? result->data.size() : 0;
}
const char* wc_layout_result_error_v1(const wc_layout_result_v1* result) {
  return result ? result->error.c_str() : "native layout result allocation failed";
}
void wc_layout_result_free_v1(wc_layout_result_v1* result) { delete result; }
}
