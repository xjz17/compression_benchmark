#include "operator102_measure.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

namespace native_graph_v1::operator102_measure {
namespace {
double entropy_bits(const auto& counts, std::uint64_t total) {
  if (total == 0) return 0.0;
  double bits = 0.0;
  for (const auto& item : counts) {
    const auto count = [&]() -> std::uint64_t {
      if constexpr (requires { item.second; }) return item.second;
      else return item;
    }();
    if (count != 0) bits += static_cast<double>(count) *
                           std::log2(static_cast<double>(total) / count);
  }
  return bits;
}

Bytes descriptor(std::uint8_t role, Kind kind, std::uint32_t width,
                 std::uint32_t ordinal, std::uint64_t length) {
  Bytes output;
  output.reserve(18);
  output.push_back(role);
  output.push_back(static_cast<std::uint8_t>(kind));
  for (unsigned shift = 0; shift < 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(width >> shift));
  for (unsigned shift = 0; shift < 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(ordinal >> shift));
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.push_back(static_cast<std::uint8_t>(length >> shift));
  return output;
}

void append_u32(Bytes& output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(Bytes& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t read_u32(std::span<const std::uint8_t> bytes, std::size_t& offset) {
  if (offset + 4 > bytes.size()) throw std::runtime_error("truncated O2R1 uint32");
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(bytes[offset++]) << shift;
  return value;
}

std::uint64_t read_u64(std::span<const std::uint8_t> bytes, std::size_t& offset) {
  if (offset + 8 > bytes.size()) throw std::runtime_error("truncated O2R1 uint64");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
  return value;
}

void append_o2r1_record(Bytes& output, std::uint8_t role, Kind kind,
                        std::uint32_t width, std::uint32_t ordinal,
                        std::span<const std::uint8_t> payload) {
  const auto header = descriptor(role, kind, width, ordinal, payload.size());
  output.insert(output.end(), header.begin(), header.end());
  output.insert(output.end(), payload.begin(), payload.end());
}
}  // namespace

EntropySummary entropy_one(std::span<const std::uint8_t> bytes) {
  EntropySummary result;
  result.bytes = bytes.size();
  result.streams = 1;
  std::array<std::uint64_t, 256> symbols{};
  for (const auto value : bytes) ++symbols[value];
  result.h0_bits = entropy_bits(symbols, bytes.size());

  const auto prefix = std::min<std::size_t>(3, bytes.size());
  std::array<std::uint64_t, 256> prefix_symbols{};
  for (std::size_t index = 0; index < prefix; ++index) ++prefix_symbols[bytes[index]];
  result.markov3_prefix_h0_bits = entropy_bits(prefix_symbols, prefix);

  // A hash entry per observed transition has prohibitive allocator overhead
  // for the largest columns.  Partition by the oldest context byte and sort
  // compact uint32 keys containing the remaining 16 context bits + symbol.
  // This is exact, requires four bytes per conditioned byte plus vector
  // capacity, and never forms contexts across the caller's stream boundary.
  std::array<std::vector<std::uint32_t>, 256> transitions;
  if (bytes.size() > 3) {
    std::array<std::size_t, 256> sizes{};
    for (std::size_t index = 3; index < bytes.size(); ++index) ++sizes[bytes[index - 3]];
    for (std::size_t bucket = 0; bucket < sizes.size(); ++bucket)
      transitions[bucket].reserve(sizes[bucket]);
    for (std::size_t index = 3; index < bytes.size(); ++index) {
      const auto key = (static_cast<std::uint32_t>(bytes[index - 2]) << 16U) |
                       (static_cast<std::uint32_t>(bytes[index - 1]) << 8U) |
                       static_cast<std::uint32_t>(bytes[index]);
      transitions[bytes[index - 3]].push_back(key);
    }
  }
  result.markov3_conditioned_bytes = bytes.size() - prefix;
  for (auto& bucket : transitions) {
    std::sort(bucket.begin(), bucket.end());
    std::size_t begin = 0;
    while (begin < bucket.size()) {
      const auto context = bucket[begin] >> 8U;
      std::uint64_t context_total = 0;
      double count_log_count = 0.0;
      while (begin < bucket.size() && (bucket[begin] >> 8U) == context) {
        const auto key = bucket[begin];
        auto end = begin + 1;
        while (end < bucket.size() && bucket[end] == key) ++end;
        const auto count = static_cast<std::uint64_t>(end - begin);
        context_total += count;
        count_log_count += static_cast<double>(count) * std::log2(static_cast<double>(count));
        begin = end;
      }
      result.markov3_conditional_bits +=
          static_cast<double>(context_total) * std::log2(static_cast<double>(context_total)) -
          count_log_count;
    }
  }
  result.markov3_modeled_bits = result.markov3_prefix_h0_bits +
                                result.markov3_conditional_bits;
  return result;
}

EntropySummary& operator+=(EntropySummary& left, const EntropySummary& right) {
  left.bytes += right.bytes;
  left.streams += right.streams;
  left.markov3_conditioned_bytes += right.markov3_conditioned_bytes;
  left.h0_bits += right.h0_bits;
  left.markov3_conditional_bits += right.markov3_conditional_bits;
  left.markov3_prefix_h0_bits += right.markov3_prefix_h0_bits;
  left.markov3_modeled_bits += right.markov3_modeled_bits;
  return left;
}

EntropySummary entropy_streams(const std::vector<std::span<const std::uint8_t>>& streams) {
  EntropySummary result;
  for (const auto stream : streams) result += entropy_one(stream);
  return result;
}

EntropySummary representation_entropy(const std::vector<Stream>& outputs,
                                      const std::vector<Bytes>& inverse_metadata) {
  EntropySummary result;
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto header = descriptor(1, outputs[index].kind, outputs[index].width,
                                   static_cast<std::uint32_t>(index),
                                   outputs[index].data.size());
    result += entropy_one(header);
    result += entropy_one(outputs[index].data);
  }
  for (std::size_t index = 0; index < inverse_metadata.size(); ++index) {
    const auto header = descriptor(2, Kind::Serial, 1,
                                   static_cast<std::uint32_t>(index),
                                   inverse_metadata[index].size());
    result += entropy_one(header);
    result += entropy_one(inverse_metadata[index]);
  }
  return result;
}

Bytes pack_o2r1(const std::vector<Stream>& outputs,
                const std::vector<Bytes>& inverse_metadata,
                const std::vector<Bytes>& auxiliaries,
                const Bytes* adapter) {
  const auto count = outputs.size() + inverse_metadata.size() + auxiliaries.size() +
                     (adapter ? 1U : 0U);
  if (count > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("O2R1 record count overflow");
  Bytes packed{'O', '2', 'R', '1'};
  append_u32(packed, static_cast<std::uint32_t>(count));
  for (std::size_t index = 0; index < outputs.size(); ++index)
    append_o2r1_record(packed, 1, outputs[index].kind, outputs[index].width,
                       static_cast<std::uint32_t>(index), outputs[index].data);
  for (std::size_t index = 0; index < inverse_metadata.size(); ++index)
    append_o2r1_record(packed, 2, Kind::Serial, 1,
                       static_cast<std::uint32_t>(index), inverse_metadata[index]);
  for (std::size_t index = 0; index < auxiliaries.size(); ++index)
    append_o2r1_record(packed, 3, Kind::Serial, 1,
                       static_cast<std::uint32_t>(index), auxiliaries[index]);
  if (adapter) append_o2r1_record(packed, 4, Kind::Struct, 1, 0, *adapter);
  return packed;
}

O2R1State parse_o2r1(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < 8 || bytes[0] != 'O' || bytes[1] != '2' ||
      bytes[2] != 'R' || bytes[3] != '1')
    throw std::runtime_error("invalid O2R1 header");
  std::size_t offset = 4;
  const auto count = read_u32(bytes, offset);
  O2R1State result;
  std::uint8_t prior_role = 0;
  for (std::uint32_t record = 0; record < count; ++record) {
    if (offset + 2 > bytes.size()) throw std::runtime_error("truncated O2R1 descriptor");
    const auto role = bytes[offset++];
    const auto raw_kind = bytes[offset++];
    if (role < 1 || role > 4 || role < prior_role)
      throw std::runtime_error("invalid O2R1 role ordering");
    prior_role = role;
    if (raw_kind != static_cast<std::uint8_t>(Kind::Serial) &&
        raw_kind != static_cast<std::uint8_t>(Kind::Struct) &&
        raw_kind != static_cast<std::uint8_t>(Kind::Numeric))
      throw std::runtime_error("invalid O2R1 stream kind");
    const auto kind = static_cast<Kind>(raw_kind);
    const auto width = read_u32(bytes, offset);
    const auto ordinal = read_u32(bytes, offset);
    const auto length = read_u64(bytes, offset);
    if (length > bytes.size() - offset)
      throw std::runtime_error("truncated O2R1 payload");
    Bytes payload(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                  bytes.begin() + static_cast<std::ptrdiff_t>(offset + length));
    offset += static_cast<std::size_t>(length);
    if (role == 1) {
      if (ordinal != result.outputs.size() || width == 0 || payload.size() % width)
        throw std::runtime_error("invalid O2R1 output descriptor");
      result.outputs.push_back({kind, width, std::move(payload)});
    } else if (role == 2) {
      if (kind != Kind::Serial || width != 1 ||
          ordinal != result.inverse_metadata.size())
        throw std::runtime_error("invalid O2R1 metadata descriptor");
      result.inverse_metadata.push_back(std::move(payload));
    } else if (role == 3) {
      if (kind != Kind::Serial || width != 1 || ordinal != result.auxiliaries.size())
        throw std::runtime_error("invalid O2R1 auxiliary descriptor");
      result.auxiliaries.push_back(std::move(payload));
    } else {
      if (kind != Kind::Struct || width != 1 || ordinal != 0 || result.adapter)
        throw std::runtime_error("invalid O2R1 adapter descriptor");
      result.adapter = std::move(payload);
    }
  }
  if (offset != bytes.size()) throw std::runtime_error("trailing bytes after O2R1 bundle");
  return result;
}

PipelineMeasurement measure_pipeline(const StageDispatcher& dispatcher,
                                     const Stream& input,
                                     const std::vector<PipelineStep>& steps,
                                     std::size_t focal_step,
                                     const EntropySummary* input_entropy) {
  if (!steps.empty() && focal_step >= steps.size())
    throw std::invalid_argument("focal_step is outside pipeline");

  std::vector<Stream> current{input};
  std::vector<Bytes> metadata;
  metadata.reserve(steps.size());
  PipelineMeasurement result;

  for (std::size_t index = 0; index < steps.size(); ++index) {
    if (index == focal_step) {
      if (index == 0 && input_entropy) {
        result.focal_before = *input_entropy;
        result.focal_before += entropy_one(descriptor(
            1, input.kind, input.width, 0, input.data.size()));
      }
      else {
        result.focal_before = representation_entropy(current, metadata);
      }
    }
    auto transformed = dispatcher.forward(steps[index].request, current);
    if (transformed.outputs.empty())
      throw std::runtime_error("native stage returned no output streams");
    metadata.push_back(std::move(transformed.metadata));
    current = std::move(transformed.outputs);
    if (index == focal_step) {
      result.focal_after = representation_entropy(current, metadata);
    }
  }

  for (const auto& packet : metadata) {
    result.metadata_bytes += packet.size();
  }
  for (const auto& stream : current) result.output_payload_bytes += stream.data.size();
  result.final_outputs = current;
  result.inverse_metadata = metadata;
  if (!steps.empty() && focal_step + 1 == steps.size()) result.final_representation = result.focal_after;
  else {
    result.final_representation = representation_entropy(current, metadata);
  }

  for (std::size_t reverse = steps.size(); reverse > 0; --reverse)
    current = dispatcher.inverse(steps[reverse - 1].request, current, metadata[reverse - 1]);
  result.bitexact = current.size() == 1 && current.front() == input;
  if (!result.bitexact) throw std::runtime_error("pipeline inverse was not bit-exact");

  if (steps.empty()) {
    if (input_entropy) {
      result.focal_before = *input_entropy;
      result.focal_before += entropy_one(descriptor(
          1, input.kind, input.width, 0, input.data.size()));
    } else {
      result.focal_before = representation_entropy(current, {});
    }
    result.focal_after = result.focal_before;
    result.output_payload_bytes = input.data.size();
    result.final_representation = result.focal_after;
    result.final_outputs = current;
    result.inverse_metadata.clear();
    result.bitexact = true;
  }
  return result;
}

PipelineTiming benchmark_pipeline(const StageDispatcher& dispatcher,
                                  const Stream& input,
                                  const std::vector<PipelineStep>& steps,
                                  const std::vector<Bytes>& auxiliaries,
                                  const Bytes* adapter,
                                  std::size_t warmups,
                                  std::size_t repeats) {
  if (repeats == 0) throw std::invalid_argument("timing repeats must be positive");
  using Clock = std::chrono::steady_clock;
  PipelineTiming timing;
  timing.encode_ns.reserve(repeats);
  timing.decode_ns.reserve(repeats);
  timing.full_encode_ns.reserve(repeats);
  timing.full_decode_ns.reserve(repeats);
  for (std::size_t iteration = 0; iteration < warmups + repeats; ++iteration) {
    // Preparing the immutable input view is outside the operator timing
    // boundary.  The forward stages own every output and metadata allocation.
    std::vector<Stream> current{input};
    std::vector<Bytes> metadata;
    metadata.reserve(steps.size());
    const auto full_encode_begin = Clock::now();
    const auto encode_begin = full_encode_begin;
    for (const auto& step : steps) {
      auto transformed = dispatcher.forward(step.request, current);
      if (transformed.outputs.empty())
        throw std::runtime_error("timed native stage returned no output streams");
      metadata.push_back(std::move(transformed.metadata));
      current = std::move(transformed.outputs);
    }
    const auto encode_end = Clock::now();
    auto bundle = pack_o2r1(current, metadata, auxiliaries, adapter);
    const auto full_encode_end = Clock::now();
    const auto full_decode_begin = full_encode_end;
    auto parsed = parse_o2r1(bundle);
    current = std::move(parsed.outputs);
    metadata = std::move(parsed.inverse_metadata);
    if (metadata.size() != steps.size())
      throw std::runtime_error("O2R1 inverse metadata count mismatch");
    const auto decode_begin = Clock::now();
    for (std::size_t reverse = steps.size(); reverse > 0; --reverse)
      current = dispatcher.inverse(steps[reverse - 1].request, current,
                                   metadata[reverse - 1]);
    const auto decode_end = Clock::now();
    const auto full_decode_end = decode_end;
    if (parsed.auxiliaries != auxiliaries ||
        parsed.adapter.has_value() != (adapter != nullptr) ||
        (adapter && parsed.adapter.value() != *adapter))
      throw std::runtime_error("O2R1 common representation round-trip mismatch");
    if (current.size() != 1 || current.front() != input)
      throw std::runtime_error("timed pipeline inverse was not bit-exact");
    if (iteration >= warmups) {
      timing.encode_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              encode_end - encode_begin).count()));
      timing.decode_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              decode_end - decode_begin).count()));
      timing.full_encode_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              full_encode_end - full_encode_begin).count()));
      timing.full_decode_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              full_decode_end - full_decode_begin).count()));
    }
  }
  timing.bitexact = timing.encode_ns.size() == repeats &&
                    timing.decode_ns.size() == repeats &&
                    timing.full_encode_ns.size() == repeats &&
                    timing.full_decode_ns.size() == repeats;
  return timing;
}

}  // namespace native_graph_v1::operator102_measure
