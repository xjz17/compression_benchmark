#include "operator102_special_bridges.hpp"
#include "operator102_measure.hpp"

#include "stage_dispatch.hpp"
#include "deflate_container_stages.hpp"
#include "general_codec_reference_stages.hpp"
#include "lz4_exact_stages.hpp"
#include "native_lz_action_stages.hpp"
#include "parameterized_deflate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <utility>

// These two frozen source files predate public headers.  Their functions have
// external linkage.  The operator102 CLI links the translation units with the
// CLI main symbols renamed, exactly as stage_dispatch does for older sources.
namespace reger_legacy_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes core_forward(const Bytes&);
Bytes core_inverse(const Bytes&);
Bytes frame_forward(const Bytes&);
Bytes frame_inverse(const Bytes&);
}  // namespace reger_legacy_cpp

namespace blosc2_binding_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes core_forward(const Bytes&);
Bytes core_inverse(const Bytes&);
Bytes frame_forward(const Bytes&);
Bytes frame_inverse(const Bytes&);
}  // namespace blosc2_binding_cpp

namespace native_graph_v1::operator102 {
namespace {

using RequestAndMetadata = std::pair<StageRequest, Bytes>;
using Clock = std::chrono::steady_clock;

struct IterationTiming {
  std::uint64_t encode_ns = 0;
  std::uint64_t decode_ns = 0;
  std::uint64_t full_encode_ns = 0;
  std::uint64_t full_decode_ns = 0;
  const std::vector<Bytes>* auxiliaries = nullptr;
  const Bytes* adapter = nullptr;
  Clock::time_point full_decode_begin{};
  Clock::time_point kernel_decode_begin{};
  std::vector<Bytes> parsed_auxiliaries;
  std::optional<Bytes> parsed_adapter;
};

struct DecodedRepresentation {
  std::vector<Stream> outputs;
  std::vector<Bytes> metadata;
};

void require(bool condition, const std::string& message);

std::uint64_t nanoseconds(Clock::time_point begin, Clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

DecodedRepresentation cross_o2r1_boundary(
    const std::vector<Stream>& outputs, const std::vector<Bytes>& metadata,
    IterationTiming& timing, Clock::time_point full_encode_begin,
    Clock::time_point kernel_encode_begin, Clock::time_point kernel_encode_end) {
  static const std::vector<Bytes> empty_auxiliaries;
  const auto& auxiliaries = timing.auxiliaries ? *timing.auxiliaries : empty_auxiliaries;
  auto packed = operator102_measure::pack_o2r1(
      outputs, metadata, auxiliaries, timing.adapter);
  const auto full_encode_end = Clock::now();
  timing.encode_ns = nanoseconds(kernel_encode_begin, kernel_encode_end);
  timing.full_encode_ns = nanoseconds(full_encode_begin, full_encode_end);
  timing.full_decode_begin = full_encode_end;
  auto parsed = operator102_measure::parse_o2r1(packed);
  timing.parsed_auxiliaries = std::move(parsed.auxiliaries);
  timing.parsed_adapter = std::move(parsed.adapter);
  timing.kernel_decode_begin = Clock::now();
  return {std::move(parsed.outputs), std::move(parsed.inverse_metadata)};
}

void finish_timing(IterationTiming* timing, Clock::time_point decode_end) {
  if (!timing) return;
  timing->decode_ns = nanoseconds(timing->kernel_decode_begin, decode_end);
  timing->full_decode_ns = nanoseconds(timing->full_decode_begin, decode_end);
  static const std::vector<Bytes> empty_auxiliaries;
  const auto& auxiliaries = timing->auxiliaries
      ? *timing->auxiliaries : empty_auxiliaries;
  require(timing->parsed_auxiliaries == auxiliaries &&
              timing->parsed_adapter.has_value() == (timing->adapter != nullptr) &&
              (!timing->adapter || timing->parsed_adapter.value() == *timing->adapter),
          "special bridge O2R1 common representation mismatch");
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Bytes copy(std::span<const std::uint8_t> value) {
  return {value.begin(), value.end()};
}

SpecialBridgeResult direct_result(
    std::string_view id, Bytes before, Bytes after, Bytes target_inverse,
    Bytes fully_restored, std::string contract) {
  require(target_inverse == before,
          std::string(id) + " focal inverse did not reproduce target input");
  return {std::string(id), std::move(before), "serial", 1,
          {std::move(after)}, {}, {}, {"bytes"}, {"serial"}, {1},
          std::move(target_inverse), std::move(fully_restored), std::move(contract)};
}

std::string kind_name(Kind kind) {
  if (kind == Kind::Serial) return "serial";
  if (kind == Kind::Numeric) return "numeric";
  if (kind == Kind::Struct) return "struct";
  throw std::runtime_error("unknown native graph stream kind");
}

SpecialBridgeResult native_lz_window(std::span<const std::uint8_t> raw,
                                     IterationTiming* timing = nullptr) {
  auto before = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto after = native_lz_action::attach_window(before, 32768);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A065 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  const auto inverse = native_lz_action::inverse(
      "op.match_window_model.native_lz.window32768", decode_after);
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A065", std::move(before), std::move(after), inverse, inverse,
      "target=attach_window(32768); NLS1 bytes are charged; no match search");
}

SpecialBridgeResult native_lz_limits(std::span<const std::uint8_t> raw,
                                     IterationTiming* timing = nullptr) {
  const auto original = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto before = native_lz_action::attach_window(original, 32768);
  auto after = native_lz_action::attach_match_limits(before, 5, 175);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A057 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  const auto inverse = native_lz_action::inverse(
      "op.lz_match.native_lz.aligned8.min5.max175", decode_after);
  const auto restored = native_lz_action::inverse(
      "op.match_window_model.native_lz.window32768", inverse);
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A057", std::move(before), std::move(after), inverse, restored,
      "prefix=attach_window(32768); target=attach_match_limits(5,175); no match search");
}

SpecialBridgeResult magic_length(std::span<const std::uint8_t> raw,
                                 IterationTiming* timing = nullptr) {
  auto before = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto after = reference_actions::general::frame_forward(before, false);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A086 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  auto inverse = reference_actions::general::frame_inverse(decode_after, false);
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A086", std::move(before), std::move(after), inverse, inverse,
      "target=B1F1 magic plus uint64LE payload length envelope");
}

SpecialBridgeResult reger_core(std::span<const std::uint8_t> raw,
                               IterationTiming* timing = nullptr) {
  auto before = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto after = reger_legacy_cpp::core_forward(before);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A094 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  auto inverse = reger_legacy_cpp::core_inverse(decode_after);
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A094", std::move(before), std::move(after), inverse, inverse,
      "target=legacy REGER RGP1 block core; block and segment parameters remain frozen");
}

SpecialBridgeResult lz4_frame(std::span<const std::uint8_t> raw,
                              IterationTiming* timing = nullptr) {
  const auto original = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto match = exact_actions::lz4::lz_match_forward(original);
  auto token = exact_actions::lz4::token_format_forward(match);
  auto before = exact_actions::lz4::integer_code_forward(token);
  auto after = exact_actions::lz4::container_forward(before);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A096 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  auto inverse = exact_actions::lz4::container_inverse(decode_after);
  auto restored = exact_actions::lz4::lz_match_inverse(
      exact_actions::lz4::token_format_inverse(
          exact_actions::lz4::integer_code_inverse(inverse)));
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A096", std::move(before), std::move(after), inverse, restored,
      "prefix=LZ4 match/token/integer stages; target=LZ4 container only");
}

SpecialBridgeResult gzip_stored_parts(std::span<const std::uint8_t> raw,
                                      IterationTiming* timing = nullptr) {
  const auto original = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto lz = reference_actions::general::lz_forward(original, 32768, 3, 258, true);
  auto huffman = reference_actions::general::huffman_forward(lz);
  // This is the exact source-level body of general_residual::gzip_forward,
  // expressed here from its public component APIs so the bridge does not pull
  // unrelated Brotli/Zstd/Bzip2/Blosc residual backends into the driver.
  auto restored_raw = reference_actions::general::lz_inverse(
      reference_actions::general::huffman_inverse(huffman));
  auto before = exact_actions::deflate_container::gzip_inverse(
      exact_actions::parameterized_deflate::encode(
          restored_raw, 6, exact_actions::parameterized_deflate::Wrapper::gzip));
  auto after = exact_actions::deflate_container::gzip_forward(before);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A097 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  auto inverse = exact_actions::deflate_container::gzip_inverse(decode_after);
  auto restored_wire = exact_actions::deflate_container::gzip_forward(inverse);
  auto restored_prefix = reference_actions::general::huffman_forward(
      reference_actions::general::lz_forward(
          exact_actions::parameterized_deflate::decode(
              restored_wire, exact_actions::parameterized_deflate::Wrapper::gzip),
          32768, 3, 258, true));
  auto restored = reference_actions::general::lz_inverse(
      reference_actions::general::huffman_inverse(restored_prefix));
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A097", std::move(before), std::move(after), inverse, restored,
      "prefix=DEFLATE reference LZ/Huffman plus exact gzip residual; target=stored gzip header/core/trailer assembly");
}

SpecialBridgeResult reger_frame(std::span<const std::uint8_t> raw,
                                IterationTiming* timing = nullptr) {
  const auto original = copy(raw);
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto before = reger_legacy_cpp::core_forward(original);
  auto after = reger_legacy_cpp::frame_forward(before);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A100 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  auto inverse = reger_legacy_cpp::frame_inverse(decode_after);
  auto restored = reger_legacy_cpp::core_inverse(inverse);
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A100", std::move(before), std::move(after), inverse, restored,
      "prefix=legacy REGER RGP1 block core; target=REGER3 single-column frame");
}

SpecialBridgeResult blosc2_frame(std::span<const std::uint8_t> raw,
                                 IterationTiming* timing = nullptr) {
  const auto original = copy(raw);
  // core_forward returns the normalized B2P1 packet.  A102 owns the inverse
  // physical->logical normalization, so the ablation input must be the real
  // physical chunk reconstructed by frame_forward, not the B2P1 packet.
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  auto core_packet = blosc2_binding_cpp::core_forward(original);
  auto before = blosc2_binding_cpp::frame_forward(core_packet);
  auto after = blosc2_binding_cpp::frame_inverse(before);
  const auto encode_end = Clock::now();
  auto decode_after = after;
  if (timing) {
    auto parsed = cross_o2r1_boundary({Stream{Kind::Serial, 1, after}}, {Bytes{}},
                                     *timing, full_encode_begin, encode_begin,
                                     encode_end);
    require(parsed.outputs.size() == 1 && parsed.metadata.size() == 1,
            "A102 O2R1 state shape mismatch");
    decode_after = std::move(parsed.outputs.front().data);
  }
  auto inverse = blosc2_binding_cpp::frame_forward(decode_after);
  auto restored = blosc2_binding_cpp::core_inverse(decode_after);
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  return direct_result(
      "A102", std::move(before), std::move(after), inverse, restored,
      "prefix=real C-Blosc2 3.3.3 physical chunk; target=physical-to-B2P1 logical block normalization");
}

SpecialBridgeResult legacy_container(
    std::string_view id, std::span<const std::uint8_t> raw,
    IterationTiming* timing = nullptr) {
  const bool alp = id == "A099";
  NativeStageDispatcher dispatcher;
  Stream current{Kind::Numeric, 8, copy(raw)};
  std::vector<RequestAndMetadata> prefix;
  const std::vector<StageRequest> requests = alp
      ? std::vector<StageRequest>{
            {"op.scaling.alp_legacy_fe_exceptions.cpp_v1", "float64", {}},
            {"op.block_transform.int64_lo_hi32.cpp_v1", "", {}},
            {"op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1", "", {}},
            {"op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1", "", {}}}
      : std::vector<StageRequest>{
            {"op.scaling.legacy_exact_decimal.cpp_v1", "int64", {{"max_point", "0"}}},
            {"op.block_transform.int64_lo_hi32.cpp_v1", "", {}},
            {"op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1", "", {}},
            {"op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1", "", {}}};
  const auto full_encode_begin = Clock::now();
  const auto encode_begin = full_encode_begin;
  for (const auto& request : requests) {
    auto result = dispatcher.forward(request, {current});
    require(result.outputs.size() == 1, std::string(id) + " prefix unexpectedly split");
    prefix.emplace_back(request, std::move(result.metadata));
    current = std::move(result.outputs.front());
  }
  const auto before_stream = current;
  StageRequest target{
      alp ? "op.container_frame.alp2_legacy.cpp_v1"
          : "op.container_frame.legacy_numeric_limbs.cpp_v1",
      "", {}};
  auto target_result = dispatcher.forward(target, {current});
  const auto encode_end = Clock::now();
  require(target_result.outputs.size() == 1,
          std::string(id) + " target unexpectedly split");
  std::vector<Stream> decode_outputs = target_result.outputs;
  Bytes decode_target_metadata = target_result.metadata;
  if (timing) {
    std::vector<Bytes> all_metadata;
    all_metadata.reserve(prefix.size() + 1);
    for (const auto& item : prefix) all_metadata.push_back(item.second);
    all_metadata.push_back(target_result.metadata);
    auto parsed = cross_o2r1_boundary(target_result.outputs, all_metadata, *timing,
                                     full_encode_begin, encode_begin, encode_end);
    require(parsed.outputs.size() == target_result.outputs.size() &&
                parsed.metadata.size() == prefix.size() + 1,
            std::string(id) + " O2R1 state shape mismatch");
    decode_outputs = std::move(parsed.outputs);
    for (std::size_t index = 0; index < prefix.size(); ++index)
      prefix[index].second = std::move(parsed.metadata[index]);
    decode_target_metadata = std::move(parsed.metadata.back());
  }
  auto target_inverse_streams = dispatcher.inverse(
      target, decode_outputs, decode_target_metadata);
  require(target_inverse_streams.size() == 1,
          std::string(id) + " target inverse unexpectedly split");
  current = target_inverse_streams.front();
  require(current.kind == before_stream.kind && current.width == before_stream.width &&
              current.data == before_stream.data,
          std::string(id) + " focal inverse changed target input");
  for (std::size_t i = requests.size(); i-- > 0;) {
    auto restored = dispatcher.inverse(prefix[i].first, {current}, prefix[i].second);
    require(restored.size() == 1, std::string(id) + " prefix inverse unexpectedly split");
    current = std::move(restored.front());
  }
  const auto decode_end = Clock::now();
  finish_timing(timing, decode_end);
  std::vector<Bytes> ports;
  std::vector<std::string> output_kinds;
  std::vector<std::uint32_t> output_widths;
  for (auto& output : target_result.outputs) {
    output_kinds.push_back(kind_name(output.kind));
    output_widths.push_back(output.width);
    ports.push_back(std::move(output.data));
  }
  std::vector<Bytes> prefix_metadata;
  for (auto& item : prefix) prefix_metadata.push_back(std::move(item.second));
  return {std::string(id), before_stream.data,
          kind_name(before_stream.kind), before_stream.width, std::move(ports),
          std::move(prefix_metadata),
          std::move(target_result.metadata),
          std::move(target_result.output_logical_views),
          std::move(output_kinds), std::move(output_widths),
          target_inverse_streams.front().data, std::move(current.data),
          alp ? "prefix=ALP legacy scaling/limbs/native-LZ/token; target=ALP2 legacy frame"
              : "prefix=int64 legacy scaling/limbs/native-LZ/token; target=legacy numeric limb frame"};
}

SpecialBridgeResult run_special_bridge_once(
    std::string_view id, std::span<const std::uint8_t> raw,
    IterationTiming* timing = nullptr) {
  if (id == "A057") return native_lz_limits(raw, timing);
  if (id == "A065") return native_lz_window(raw, timing);
  if (id == "A086") return magic_length(raw, timing);
  if (id == "A094") return reger_core(raw, timing);
  if (id == "A096") return lz4_frame(raw, timing);
  if (id == "A097") return gzip_stored_parts(raw, timing);
  if (id == "A098" || id == "A099") return legacy_container(id, raw, timing);
  if (id == "A100") return reger_frame(raw, timing);
  if (id == "A102") return blosc2_frame(raw, timing);
  throw std::invalid_argument("unsupported operator102 special bridge: " +
                              std::string(id));
}

}  // namespace

bool supports_special_bridge(std::string_view id) {
  constexpr std::array<std::string_view, 10> supported{
      "A057", "A065", "A086", "A094", "A096",
      "A097", "A098", "A099", "A100", "A102"};
  return std::find(supported.begin(), supported.end(), id) != supported.end();
}

SpecialBridgeResult run_special_bridge(
    std::string_view id, std::span<const std::uint8_t> raw) {
  auto result = run_special_bridge_once(id, raw);
  require(result.target_inverse == result.target_before,
          std::string(id) + " target inverse mismatch");
  require(result.fully_restored == copy(raw),
          std::string(id) + " full bridge round-trip mismatch");
  return result;
}

SpecialBridgeTiming benchmark_special_bridge(
    std::string_view id, std::span<const std::uint8_t> raw,
    const std::vector<Bytes>& auxiliaries, const Bytes* adapter,
    std::size_t warmups, std::size_t repeats) {
  if (repeats == 0) throw std::invalid_argument("timing repeats must be positive");
  SpecialBridgeTiming timing;
  timing.encode_ns.reserve(repeats);
  timing.decode_ns.reserve(repeats);
  timing.full_encode_ns.reserve(repeats);
  timing.full_decode_ns.reserve(repeats);
  for (std::size_t iteration = 0; iteration < warmups + repeats; ++iteration) {
    IterationTiming sample;
    sample.auxiliaries = &auxiliaries;
    sample.adapter = adapter;
    const auto result = run_special_bridge_once(id, raw, &sample);
    require(result.target_inverse == result.target_before,
            std::string(id) + " timed target inverse mismatch");
    require(result.fully_restored == copy(raw),
            std::string(id) + " timed full bridge round-trip mismatch");
    if (iteration >= warmups) {
      timing.encode_ns.push_back(sample.encode_ns);
      timing.decode_ns.push_back(sample.decode_ns);
      timing.full_encode_ns.push_back(sample.full_encode_ns);
      timing.full_decode_ns.push_back(sample.full_decode_ns);
    }
  }
  timing.bitexact = timing.encode_ns.size() == repeats &&
                    timing.decode_ns.size() == repeats &&
                    timing.full_encode_ns.size() == repeats &&
                    timing.full_decode_ns.size() == repeats;
  return timing;
}

}  // namespace native_graph_v1::operator102
