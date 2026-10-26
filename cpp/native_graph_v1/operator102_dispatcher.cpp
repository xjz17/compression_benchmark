#include "operator102_dispatcher.hpp"

#include "../src/learning_codec_exact_stages.hpp"
#include "../src/p04_exact_stages.hpp"
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
#include "../src/general_codec_reference_stages.hpp"
#include "../src/general_codec_residual_stages.hpp"
#endif

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <limits>

namespace native_graph_v1::operator102_measure {
namespace {
using Function = Bytes (*)(std::span<const std::uint8_t>);

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

Bytes shape_metadata(const Stream& input) {
  return {'O', 'S', 'H', '1', static_cast<std::uint8_t>(input.kind),
          static_cast<std::uint8_t>(input.width),
          static_cast<std::uint8_t>(input.width >> 8U),
          static_cast<std::uint8_t>(input.width >> 16U),
          static_cast<std::uint8_t>(input.width >> 24U)};
}

Stream restore(std::span<const std::uint8_t> metadata, Bytes bytes) {
  require(metadata.size() == 9 && metadata[0] == 'O' && metadata[1] == 'S' &&
              metadata[2] == 'H' && metadata[3] == '1',
          "invalid operator102 shape metadata");
  const auto kind = static_cast<Kind>(metadata[4]);
  const auto width = static_cast<std::uint32_t>(metadata[5]) |
                     (static_cast<std::uint32_t>(metadata[6]) << 8U) |
                     (static_cast<std::uint32_t>(metadata[7]) << 16U) |
                     (static_cast<std::uint32_t>(metadata[8]) << 24U);
  require(width != 0 && bytes.size() % width == 0, "restored operator102 shape is invalid");
  return {kind, width, std::move(bytes)};
}

StageResult encoded(const Stream& input, Bytes bytes) {
  return {{{Kind::Serial, 1, std::move(bytes)}}, shape_metadata(input), {"bytes"}};
}

const Stream& single(const std::vector<Stream>& streams, const char* label) {
  require(streams.size() == 1, std::string(label) + " requires one physical stream");
  return streams.front();
}

Function simple_forward(const std::string& action) {
  using namespace exact_actions::learning;
  using namespace exact_actions::p04;
  if (action == "op.zigzag.sprintz.cpp_v1") return sprintz_zigzag_forward;
  if (action == "op.bitpack_int.sprintz.cpp_v1") return sprintz_bitpack_forward_current;
  if (action == "op.block_transform.buff_column_split.cpp_v1") return buff_column_split_forward;
  if (action == "op.bitpack_int.buff.cpp_v1") return buff_bitpack_forward;
  if (action == "op.dictionary.alp_rd.cpp_v1") return alp_dictionary_forward;
  if (action == "op.frame_of_reference.alp.cpp_v1") return alp_for_forward;
  if (action == "op.bitpack_int.alp.cpp_v1") return alp_bitpack_forward;
  if (action == "op.block_state_control.alp_residual_core.cpp_v1") return alp_residual_core_forward;
  if (action == "op.value_reorder.reger.cpp_v1") return reger_reorder_forward;
  if (action == "op.delta.reger_signed_residual.cpp_v2") return reger_delta_forward;
  if (action == "op.zigzag.reger_residual.cpp_v2") return reger_zigzag_forward;
  if (action == "op.frame_of_reference.reger.cpp_v1") return reger_for_forward;
  if (action == "op.bitpack_int.reger.cpp_v1") return reger_bitpack_forward;
  if (action == "op.block_state_control.reger_residual_core.cpp_v1") return reger_residual_core_forward;
  return nullptr;
}

Function simple_inverse(const std::string& action) {
  using namespace exact_actions::learning;
  using namespace exact_actions::p04;
  if (action == "op.zigzag.sprintz.cpp_v1") return sprintz_zigzag_inverse;
  if (action == "op.bitpack_int.sprintz.cpp_v1") return sprintz_bitpack_inverse_current;
  if (action == "op.block_transform.buff_column_split.cpp_v1") return buff_column_split_inverse;
  if (action == "op.bitpack_int.buff.cpp_v1") return buff_bitpack_inverse;
  if (action == "op.dictionary.alp_rd.cpp_v1") return alp_dictionary_inverse;
  if (action == "op.frame_of_reference.alp.cpp_v1") return alp_for_inverse;
  if (action == "op.bitpack_int.alp.cpp_v1") return alp_bitpack_inverse;
  if (action == "op.block_state_control.alp_residual_core.cpp_v1") return alp_residual_core_inverse;
  if (action == "op.value_reorder.reger.cpp_v1") return reger_reorder_inverse;
  if (action == "op.delta.reger_signed_residual.cpp_v2") return reger_delta_inverse;
  if (action == "op.zigzag.reger_residual.cpp_v2") return reger_zigzag_inverse;
  if (action == "op.frame_of_reference.reger.cpp_v1") return reger_for_inverse;
  if (action == "op.bitpack_int.reger.cpp_v1") return reger_bitpack_inverse;
  if (action == "op.block_state_control.reger_residual_core.cpp_v1") return reger_residual_core_inverse;
  return nullptr;
}

StageResult adjacent_xor_forward(const Stream& input) {
  require(input.kind == Kind::Numeric && (input.width == 4 || input.width == 8),
          "adjacent XOR residual builder requires numeric width 4 or 8");
  require(input.data.size() % input.width == 0,
          "adjacent XOR residual input is not width aligned");
  Bytes residual(input.data.size());
  if (!input.data.empty()) {
    std::copy_n(input.data.begin(), input.width, residual.begin());
    for (std::size_t offset = input.width; offset < input.data.size(); offset += input.width)
      for (std::size_t byte = 0; byte < input.width; ++byte)
        residual[offset + byte] = input.data[offset + byte] ^ input.data[offset - input.width + byte];
  }
  return {{{Kind::Numeric, input.width, std::move(residual)}}, shape_metadata(input),
          {input.width == 8 ? "uint64" : "uint32"}};
}

std::vector<Stream> adjacent_xor_inverse(const std::vector<Stream>& outputs,
                                         std::span<const std::uint8_t> metadata) {
  const auto& residual = single(outputs, "adjacent XOR residual inverse");
  require(residual.kind == Kind::Numeric && (residual.width == 4 || residual.width == 8),
          "adjacent XOR residual output shape is invalid");
  Bytes original(residual.data.size());
  if (!residual.data.empty()) {
    std::copy_n(residual.data.begin(), residual.width, original.begin());
    for (std::size_t offset = residual.width; offset < residual.data.size(); offset += residual.width)
      for (std::size_t byte = 0; byte < residual.width; ++byte)
        original[offset + byte] = residual.data[offset + byte] ^ original[offset - residual.width + byte];
  }
  return {restore(metadata, std::move(original))};
}

constexpr std::array<std::uint8_t, 4> kFallbackMagic{'L', 'F', 'W', '1'};
constexpr std::array<std::uint8_t, 4> kNncpChunkMagic{'N', 'N', 'C', '1'};
constexpr std::size_t kNncpChunkBytes = 4096;

void append_u16(Bytes& output, std::uint16_t value) {
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void append_u32(Bytes& output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(Bytes& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint16_t read_u16(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  require(pos + 2 <= bytes.size(), "truncated LFW1 uint16");
  const auto value = static_cast<std::uint16_t>(bytes[pos]) |
                     (static_cast<std::uint16_t>(bytes[pos + 1]) << 8U);
  pos += 2;
  return value;
}

std::uint32_t read_u32(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  require(pos + 4 <= bytes.size(), "truncated LFW1 uint32");
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) value |= bytes[pos++] << shift;
  return value;
}

std::uint64_t read_u64(std::span<const std::uint8_t> bytes, std::size_t& pos) {
  require(pos + 8 <= bytes.size(), "truncated LFW1 uint64");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(bytes[pos++]) << shift;
  return value;
}

struct FallbackPacket {
  bool raw = false;
  Kind kind = Kind::Serial;
  std::uint32_t width = 1;
  std::uint64_t original_bytes = 0;
  std::string reason;
  Bytes native_metadata;
};

Bytes fallback_metadata(const Stream& input, bool raw, std::string_view reason,
                        std::span<const std::uint8_t> native_metadata) {
  require(reason.size() <= std::numeric_limits<std::uint16_t>::max(),
          "LFW1 fallback reason is too long");
  Bytes output(kFallbackMagic.begin(), kFallbackMagic.end());
  output.push_back(raw ? 1 : 0);
  output.push_back(static_cast<std::uint8_t>(input.kind));
  append_u32(output, input.width);
  append_u64(output, input.data.size());
  append_u16(output, static_cast<std::uint16_t>(reason.size()));
  output.insert(output.end(), reason.begin(), reason.end());
  append_u64(output, native_metadata.size());
  output.insert(output.end(), native_metadata.begin(), native_metadata.end());
  return output;
}

FallbackPacket parse_fallback_metadata(std::span<const std::uint8_t> bytes) {
  require(bytes.size() >= kFallbackMagic.size() &&
              std::equal(kFallbackMagic.begin(), kFallbackMagic.end(), bytes.begin()),
          "invalid LFW1 exact raw-fallback metadata");
  std::size_t pos = kFallbackMagic.size();
  require(pos + 2 <= bytes.size(), "truncated LFW1 mode/shape");
  const auto mode = bytes[pos++];
  require(mode <= 1, "invalid LFW1 mode");
  FallbackPacket packet;
  packet.raw = mode == 1;
  packet.kind = static_cast<Kind>(bytes[pos++]);
  require(packet.kind == Kind::Serial || packet.kind == Kind::Numeric ||
              packet.kind == Kind::Struct,
          "invalid LFW1 stream kind");
  packet.width = read_u32(bytes, pos);
  packet.original_bytes = read_u64(bytes, pos);
  const auto reason_size = read_u16(bytes, pos);
  require(pos + reason_size <= bytes.size(), "truncated LFW1 reason");
  packet.reason.assign(reinterpret_cast<const char*>(bytes.data() + pos), reason_size);
  pos += reason_size;
  const auto metadata_size = read_u64(bytes, pos);
  require(metadata_size <= bytes.size() - pos, "truncated LFW1 native metadata");
  packet.native_metadata.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos),
                                bytes.end());
  require(packet.native_metadata.size() == metadata_size,
          "trailing LFW1 native metadata bytes");
  require(packet.width != 0 && packet.original_bytes % packet.width == 0,
          "invalid LFW1 original shape");
  require(packet.raw ? packet.native_metadata.empty() : packet.reason.empty(),
          "inconsistent LFW1 selector fields");
  return packet;
}

struct NncpChunkPacket {
  std::uint32_t raw_bytes = 0;
  Kind output_kind = Kind::Serial;
  std::uint32_t output_width = 1;
  std::uint64_t output_bytes = 0;
  Bytes native_metadata;
};

struct NncpPacket {
  Kind original_kind = Kind::Serial;
  std::uint32_t original_width = 1;
  std::uint64_t original_bytes = 0;
  std::vector<NncpChunkPacket> chunks;
};

Bytes nncp_chunk_metadata(const Stream& input,
                          const std::vector<NncpChunkPacket>& chunks) {
  require(chunks.size() <= std::numeric_limits<std::uint32_t>::max(),
          "NNC1 has too many chunks");
  Bytes output(kNncpChunkMagic.begin(), kNncpChunkMagic.end());
  output.push_back(static_cast<std::uint8_t>(input.kind));
  append_u32(output, input.width);
  append_u64(output, input.data.size());
  append_u32(output, static_cast<std::uint32_t>(kNncpChunkBytes));
  append_u32(output, static_cast<std::uint32_t>(chunks.size()));
  for (const auto& chunk : chunks) {
    append_u32(output, chunk.raw_bytes);
    output.push_back(static_cast<std::uint8_t>(chunk.output_kind));
    append_u32(output, chunk.output_width);
    append_u64(output, chunk.output_bytes);
    append_u64(output, chunk.native_metadata.size());
    output.insert(output.end(), chunk.native_metadata.begin(), chunk.native_metadata.end());
  }
  return output;
}

NncpPacket parse_nncp_chunk_metadata(std::span<const std::uint8_t> bytes) {
  require(bytes.size() >= kNncpChunkMagic.size() &&
              std::equal(kNncpChunkMagic.begin(), kNncpChunkMagic.end(), bytes.begin()),
          "invalid NNC1 NNCP chunk metadata");
  std::size_t pos = kNncpChunkMagic.size();
  require(pos < bytes.size(), "truncated NNC1 original kind");
  NncpPacket packet;
  packet.original_kind = static_cast<Kind>(bytes[pos++]);
  require(packet.original_kind == Kind::Serial || packet.original_kind == Kind::Numeric ||
              packet.original_kind == Kind::Struct,
          "invalid NNC1 original kind");
  packet.original_width = read_u32(bytes, pos);
  packet.original_bytes = read_u64(bytes, pos);
  const auto chunk_bytes = read_u32(bytes, pos);
  const auto chunk_count = read_u32(bytes, pos);
  require(packet.original_width != 0 &&
              packet.original_bytes % packet.original_width == 0,
          "invalid NNC1 original shape");
  require(chunk_bytes == kNncpChunkBytes, "unexpected NNC1 chunk size");
  const auto expected_chunks = packet.original_bytes == 0
                                   ? std::uint64_t{1}
                                   : 1 + (packet.original_bytes - 1) / kNncpChunkBytes;
  require(chunk_count == expected_chunks, "invalid NNC1 chunk count");
  packet.chunks.reserve(chunk_count);
  std::uint64_t raw_sum = 0;
  for (std::uint32_t index = 0; index < chunk_count; ++index) {
    NncpChunkPacket chunk;
    chunk.raw_bytes = read_u32(bytes, pos);
    require(chunk.raw_bytes <= kNncpChunkBytes,
            "NNC1 raw chunk exceeds the frozen NNCP limit");
    if (packet.original_bytes != 0)
      require(chunk.raw_bytes != 0, "NNC1 contains an empty nonterminal chunk");
    require(pos < bytes.size(), "truncated NNC1 output kind");
    chunk.output_kind = static_cast<Kind>(bytes[pos++]);
    require(chunk.output_kind == Kind::Serial || chunk.output_kind == Kind::Numeric ||
                chunk.output_kind == Kind::Struct,
            "invalid NNC1 output kind");
    chunk.output_width = read_u32(bytes, pos);
    chunk.output_bytes = read_u64(bytes, pos);
    const auto metadata_size = read_u64(bytes, pos);
    require(chunk.output_width != 0 && chunk.output_bytes % chunk.output_width == 0,
            "invalid NNC1 output shape");
    require(metadata_size <= bytes.size() - pos, "truncated NNC1 native metadata");
    chunk.native_metadata.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos),
                                 bytes.begin() + static_cast<std::ptrdiff_t>(pos + metadata_size));
    pos += metadata_size;
    raw_sum += chunk.raw_bytes;
    packet.chunks.push_back(std::move(chunk));
  }
  require(pos == bytes.size(), "trailing NNC1 metadata bytes");
  require(raw_sum == packet.original_bytes, "NNC1 raw chunk lengths do not cover input");
  return packet;
}

bool exact_raw_fallback_requested(const StageRequest& request) {
  const auto it = request.parameters.find("exact_raw_fallback");
  if (it == request.parameters.end()) return false;
  require(it->second == "true", "exact_raw_fallback must be true when supplied");
  return true;
}

bool allowed_domain_rejection(const std::string& message) {
  static constexpr std::array<std::string_view, 11> allowed{
      "Elf32 exact action requires finite float64 values",
      "Elf32 exact action requires a bitwise lossless float32 cast",
      "RLE scaling requires finite float64 values",
      "RLE decimal scaling is not bitwise reversible",
      "TS_2DIFF scaling requires finite values",
      "TS_2DIFF scaled value exceeds int64",
      "TS_2DIFF decimal scaling is not bitwise reversible",
      "DBP scaling requires finite values",
      "DBP scaled value exceeds int64",
      "DBP decimal scaling exceeds int64",
      "DBP decimal scaling is not bitwise reversible"};
  return std::find(allowed.begin(), allowed.end(), message) != allowed.end();
}
}  // namespace

ExactRawFallbackInfo inspect_exact_raw_fallback_metadata(
    std::span<const std::uint8_t> metadata) {
  ExactRawFallbackInfo result;
  if (metadata.size() < kFallbackMagic.size() ||
      !std::equal(kFallbackMagic.begin(), kFallbackMagic.end(), metadata.begin()))
    return result;
  const auto packet = parse_fallback_metadata(metadata);
  result.wrapper = true;
  result.used_raw_fallback = packet.raw;
  result.original_bytes = packet.original_bytes;
  result.reason = packet.reason;
  return result;
}

std::string Operator102Dispatcher::identity() const {
  return "operator102.dispatcher.exact-bridge-v3.lfw1.nnc1-4096+" + native_.identity();
}

StageResult Operator102Dispatcher::forward(const StageRequest& request,
                                           const std::vector<Stream>& inputs) const {
  if (exact_raw_fallback_requested(request)) {
    const auto& input = single(inputs, request.action.c_str());
    auto strict = request;
    strict.parameters.erase("exact_raw_fallback");
    try {
      auto transformed = native_.forward(strict, inputs);
      transformed.metadata = fallback_metadata(input, false, {}, transformed.metadata);
      return transformed;
    } catch (const std::exception& error) {
      if (!allowed_domain_rejection(error.what())) throw;
      return {{{input}}, fallback_metadata(input, true, error.what(), {}), {"raw_fallback"}};
    }
  }
  if (request.action == "op.prediction_probability.nncp.provider_v1") {
    const auto& input = single(inputs, request.action.c_str());
    require(input.width != 0 && input.data.size() % input.width == 0,
            "NNCP chunk adapter input is not width aligned");
    StageResult result;
    std::vector<NncpChunkPacket> chunks;
    const auto chunk_count = input.data.empty()
                                 ? std::size_t{1}
                                 : 1 + (input.data.size() - 1) / kNncpChunkBytes;
    result.outputs.reserve(chunk_count);
    result.output_logical_views.reserve(chunk_count);
    chunks.reserve(chunk_count);
    for (std::size_t index = 0; index < chunk_count; ++index) {
      const auto begin = std::min(input.data.size(), index * kNncpChunkBytes);
      const auto end = std::min(input.data.size(), begin + kNncpChunkBytes);
      Stream chunk_input{input.kind, input.width,
                         Bytes(input.data.begin() + static_cast<std::ptrdiff_t>(begin),
                               input.data.begin() + static_cast<std::ptrdiff_t>(end))};
      auto transformed = native_.forward(request, {chunk_input});
      require(transformed.outputs.size() == 1 &&
                  transformed.output_logical_views.size() == 1,
              "NNCP native stage returned an unexpected number of ports");
      const auto& output = transformed.outputs.front();
      chunks.push_back({static_cast<std::uint32_t>(end - begin), output.kind,
                        output.width, output.data.size(), std::move(transformed.metadata)});
      result.outputs.push_back(std::move(transformed.outputs.front()));
      result.output_logical_views.push_back("nncp_chunk_" + std::to_string(index));
    }
    result.metadata = nncp_chunk_metadata(input, chunks);
    return result;
  }
  if (request.action == "build.adjacent_xor_residual_words")
    return adjacent_xor_forward(single(inputs, request.action.c_str()));
  if (const auto function = simple_forward(request.action)) {
    const auto& input = single(inputs, request.action.c_str());
    return encoded(input, function(input.data));
  }
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  {
    using namespace reference_actions::general;
    if (request.action == "op.literal_entropy.zstd_huffman.cpp_reference_v1") {
      const auto& input = single(inputs, request.action.c_str());
      return encoded(input, huffman_forward(input.data));
    }
    if (request.action == "op.run_length.bzip2_uint8.cpp_reference_v1") {
      const auto& input = single(inputs, request.action.c_str());
      return encoded(input, rle_forward(input.data));
    }
  }
  {
    using namespace exact_actions::general_residual;
    const auto residual = request.action == "op.block_state_control.brotli_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.zstd_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.blosc1_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc1_residual_core.cpp_v2" ||
        request.action == "op.block_state_control.blosc2_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc2_residual_core.cpp_v2" ||
        request.action == "op.block_state_control.bzip2_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.zlib_deflate_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.deflate_residual_core.cpp_v2" ||
        request.action == "op.block_state_control.xz_lzma2_residual_core.cpp_v1";
    if (!residual) return native_.forward(request, inputs);
    const auto& input = single(inputs, request.action.c_str());
    if (request.action == "op.block_state_control.brotli_residual_core.cpp_v1")
      return encoded(input, brotli_forward(input.data));
    if (request.action == "op.block_state_control.zstd_residual_core.cpp_v1")
      return encoded(input, zstd_forward(input.data));
    if (request.action == "op.block_state_control.blosc1_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc1_residual_core.cpp_v2")
      return encoded(input, blosc1_forward(input.data, 8));
    if (request.action == "op.block_state_control.blosc2_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc2_residual_core.cpp_v2")
      return encoded(input, blosc2_forward(input.data, 8));
    if (request.action == "op.block_state_control.bzip2_residual_core.cpp_v1")
      return encoded(input, bzip2_forward(input.data));
    if (request.action == "op.block_state_control.zlib_deflate_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.deflate_residual_core.cpp_v2")
      return encoded(input, zlib_forward(input.data, 9));
    if (request.action == "op.block_state_control.xz_lzma2_residual_core.cpp_v1")
      return encoded(input, xz_forward(input.data));
  }
#endif
  return native_.forward(request, inputs);
}

std::vector<Stream> Operator102Dispatcher::inverse(const StageRequest& request,
                                                   const std::vector<Stream>& outputs,
                                                   std::span<const std::uint8_t> metadata) const {
  if (exact_raw_fallback_requested(request)) {
    const auto packet = parse_fallback_metadata(metadata);
    if (packet.raw) {
      const auto& output = single(outputs, request.action.c_str());
      require(output.kind == packet.kind && output.width == packet.width &&
                  output.data.size() == packet.original_bytes,
              "raw LFW1 output does not match recorded original shape");
      return {output};
    }
    auto strict = request;
    strict.parameters.erase("exact_raw_fallback");
    auto restored = native_.inverse(strict, outputs, packet.native_metadata);
    const auto& output = single(restored, request.action.c_str());
    require(output.kind == packet.kind && output.width == packet.width &&
                output.data.size() == packet.original_bytes,
            "strict LFW1 inverse does not match recorded original shape");
    return restored;
  }
  if (request.action == "op.prediction_probability.nncp.provider_v1") {
    const auto packet = parse_nncp_chunk_metadata(metadata);
    require(outputs.size() == packet.chunks.size(),
            "NNC1 physical output count does not match metadata");
    Bytes original;
    original.reserve(packet.original_bytes);
    for (std::size_t index = 0; index < packet.chunks.size(); ++index) {
      const auto& descriptor = packet.chunks[index];
      const auto& output = outputs[index];
      require(output.kind == descriptor.output_kind &&
                  output.width == descriptor.output_width &&
                  output.data.size() == descriptor.output_bytes,
              "NNC1 output descriptor does not match physical chunk");
      auto restored = native_.inverse(request, {output}, descriptor.native_metadata);
      const auto& chunk = single(restored, "NNC1 NNCP inverse");
      require(chunk.kind == packet.original_kind &&
                  chunk.width == packet.original_width &&
                  chunk.data.size() == descriptor.raw_bytes,
              "NNC1 restored chunk shape is invalid");
      original.insert(original.end(), chunk.data.begin(), chunk.data.end());
    }
    require(original.size() == packet.original_bytes,
            "NNC1 restored bytes do not match original length");
    return {{packet.original_kind, packet.original_width, std::move(original)}};
  }
  if (request.action == "build.adjacent_xor_residual_words")
    return adjacent_xor_inverse(outputs, metadata);
  if (const auto function = simple_inverse(request.action)) {
    const auto& output = single(outputs, request.action.c_str());
    return {restore(metadata, function(output.data))};
  }
#if defined(OPERATOR102_ENABLE_SUPPORT_DEPS)
  {
    using namespace reference_actions::general;
    if (request.action == "op.literal_entropy.zstd_huffman.cpp_reference_v1") {
      const auto& output = single(outputs, request.action.c_str());
      return {restore(metadata, huffman_inverse(output.data))};
    }
    if (request.action == "op.run_length.bzip2_uint8.cpp_reference_v1") {
      const auto& output = single(outputs, request.action.c_str());
      return {restore(metadata, rle_inverse(output.data))};
    }
  }
  {
    using namespace exact_actions::general_residual;
    const auto residual = request.action == "op.block_state_control.brotli_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.zstd_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.blosc1_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc1_residual_core.cpp_v2" ||
        request.action == "op.block_state_control.blosc2_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc2_residual_core.cpp_v2" ||
        request.action == "op.block_state_control.bzip2_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.zlib_deflate_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.deflate_residual_core.cpp_v2" ||
        request.action == "op.block_state_control.xz_lzma2_residual_core.cpp_v1";
    if (!residual) return native_.inverse(request, outputs, metadata);
    const auto& output = single(outputs, request.action.c_str());
    if (request.action == "op.block_state_control.brotli_residual_core.cpp_v1")
      return {restore(metadata, brotli_inverse(output.data))};
    if (request.action == "op.block_state_control.zstd_residual_core.cpp_v1")
      return {restore(metadata, zstd_inverse(output.data))};
    if (request.action == "op.block_state_control.blosc1_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc1_residual_core.cpp_v2")
      return {restore(metadata, blosc1_inverse(output.data, 8))};
    if (request.action == "op.block_state_control.blosc2_residual_core_t8.cpp_v1" ||
        request.action == "op.block_state_control.blosc2_residual_core.cpp_v2")
      return {restore(metadata, blosc2_inverse(output.data, 8))};
    if (request.action == "op.block_state_control.bzip2_residual_core.cpp_v1")
      return {restore(metadata, bzip2_inverse(output.data))};
    if (request.action == "op.block_state_control.zlib_deflate_residual_core.cpp_v1" ||
        request.action == "op.block_state_control.deflate_residual_core.cpp_v2")
      return {restore(metadata, zlib_inverse(output.data))};
    if (request.action == "op.block_state_control.xz_lzma2_residual_core.cpp_v1")
      return {restore(metadata, xz_inverse(output.data))};
  }
#endif
  return native_.inverse(request, outputs, metadata);
}

}  // namespace native_graph_v1::operator102_measure
