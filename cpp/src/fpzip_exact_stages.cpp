#include "fpzip_exact_stages.hpp"

#include "codec.h"
#include "pccodec.h"
#include "rcdecoder.h"
#include "rcencoder.h"
#include "rcqsmodel.h"
#include "types.h"
#include "write.h"
#include <fpzip.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace exact_actions::fpzip {
namespace {

constexpr std::uint8_t kMagic[] = {'F', 'P', 'R', '1'};
constexpr uint kBits = 64;
constexpr uint kBias = kBits;
constexpr uint kSymbols = 2 * kBits + 1;

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 4 > input.size()) throw std::runtime_error("truncated FPR1 count");
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8)
    value |= static_cast<std::uint32_t>(input[pos++]) << shift;
  return value;
}
std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 8 > input.size()) throw std::runtime_error("truncated FPR1 residual");
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(input[pos++]) << shift;
  return value;
}

std::uint64_t bits_of(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
double double_of(std::uint64_t bits) {
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
std::uint64_t map_forward(double value) {
  std::uint64_t bits = ~bits_of(value);
  bits ^= (std::uint64_t{0} - (bits >> 63)) >> 1;
  return bits;
}
double map_inverse(std::uint64_t value) {
  value ^= (std::uint64_t{0} - (value >> 63)) >> 1;
  value = ~value;
  return double_of(value);
}

double predict_1d(double previous) {
  // This keeps the same left-associative floating expression as FPZIP's
  // 3D Lorenzo stencil after all off-axis neighbors collapse to +0.0.  It is
  // observably different from returning `previous` for signed zero.
  return previous - 0.0 + 0.0 - 0.0 + 0.0 - 0.0 + 0.0;
}

struct PredictorPacket {
  std::uint32_t count = 0;
  std::vector<std::uint64_t> residuals;
};

PredictorPacket parse_packet(std::span<const std::uint8_t> packet) {
  if (packet.size() < 8 || std::memcmp(packet.data(), kMagic, 4) != 0)
    throw std::runtime_error("bad FPR1 packet");
  std::size_t pos = 4;
  PredictorPacket parsed;
  parsed.count = get_u32(packet, pos);
  if (packet.size() != 8 + static_cast<std::uint64_t>(parsed.count) * 8)
    throw std::runtime_error("FPR1 packet size mismatch");
  parsed.residuals.reserve(parsed.count);
  for (std::uint32_t i = 0; i < parsed.count; ++i)
    parsed.residuals.push_back(get_u64(packet, pos));
  return parsed;
}

std::vector<std::uint8_t> make_packet(
    std::span<const std::uint64_t> residuals) {
  if (residuals.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("FPZIP 1D column exceeds uint32 count");
  std::vector<std::uint8_t> out(std::begin(kMagic), std::end(kMagic));
  put_u32(out, static_cast<std::uint32_t>(residuals.size()));
  for (const auto value : residuals) put_u64(out, value);
  return out;
}

class BoundedDecoder final : public RCdecoder {
 public:
  explicit BoundedDecoder(std::span<const std::uint8_t> input)
      : input_(input) {}
  uint getbyte() override {
    if (pos_ >= input_.size()) {
      error = true;
      return 0;
    }
    return input_[pos_++];
  }
  std::size_t bytes() const override { return pos_; }

 private:
  std::span<const std::uint8_t> input_;
  std::size_t pos_ = 0;
};

void encode_header(RCencoder& encoder, std::uint32_t count) {
  encoder.encode<uint>('f', 8);
  encoder.encode<uint>('p', 8);
  encoder.encode<uint>('z', 8);
  encoder.encode<uint>('\0', 8);
  encoder.encode<uint>(FPZ_MAJ_VERSION, 16);
  encoder.encode<uint>(FPZ_MIN_VERSION, 8);
  encoder.encode<uint>(FPZIP_TYPE_DOUBLE, 1);
  encoder.encode<uint>(0, 7);
  encoder.encode<uint>(count, 32);
  encoder.encode<uint>(1, 32);
  encoder.encode<uint>(1, 32);
  encoder.encode<uint>(1, 32);
}

std::uint32_t decode_header(RCdecoder& decoder) {
  if (decoder.decode<uint>(8) != 'f' || decoder.decode<uint>(8) != 'p' ||
      decoder.decode<uint>(8) != 'z' || decoder.decode<uint>(8) != '\0')
    throw std::runtime_error("bad FPZIP magic");
  if (decoder.decode<uint>(16) != FPZ_MAJ_VERSION ||
      decoder.decode<uint>(8) != FPZ_MIN_VERSION)
    throw std::runtime_error("unsupported FPZIP version");
  const auto type = decoder.decode<uint>(1);
  const auto precision = decoder.decode<uint>(7);
  const auto nx = decoder.decode<uint>(32);
  const auto ny = decoder.decode<uint>(32);
  const auto nz = decoder.decode<uint>(32);
  const auto nf = decoder.decode<uint>(32);
  if (type != FPZIP_TYPE_DOUBLE || precision != 0 || ny != 1 || nz != 1 || nf != 1)
    throw std::runtime_error("FPZIP wire is not the lossless 1D double path");
  return nx;
}

}  // namespace

std::vector<std::uint8_t> predictor_forward(std::span<const std::uint8_t> raw_f64) {
  if (raw_f64.size() % 8 != 0) throw std::runtime_error("FPZIP action requires f64 bytes");
  const std::size_t count = raw_f64.size() / 8;
  if (count > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("FPZIP 1D column exceeds uint32 count");
  std::vector<std::uint64_t> residuals;
  residuals.reserve(count);
  double previous = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, raw_f64.data() + i * 8, 8);
    const std::uint64_t current = map_forward(double_of(bits));
    residuals.push_back(current - map_forward(predict_1d(previous)));
    previous = map_inverse(current);
  }
  return make_packet(residuals);
}

std::vector<std::uint8_t> predictor_inverse(std::span<const std::uint8_t> packet) {
  const auto parsed = parse_packet(packet);
  std::vector<std::uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(parsed.count) * 8);
  double previous = 0.0;
  for (const auto residual : parsed.residuals) {
    const std::uint64_t current = map_forward(predict_1d(previous)) + residual;
    const std::uint64_t bits = bits_of(map_inverse(current));
    put_u64(raw, bits);
    previous = map_inverse(current);
  }
  return raw;
}

std::vector<std::uint8_t> range_forward(std::span<const std::uint8_t> packet) {
  const auto parsed = parse_packet(packet);
  std::vector<std::uint8_t> wire(64 + static_cast<std::size_t>(parsed.count) * 16);
  RCmemencoder encoder(wire.data(), wire.size());
  encode_header(encoder, parsed.count);
  RCqsmodel model(true, kSymbols);
  double previous = 0.0;
  for (const auto residual : parsed.residuals) {
    const std::uint64_t prediction = map_forward(predict_1d(previous));
    const std::uint64_t current = prediction + residual;
    if (prediction < current) {
      const std::uint64_t difference = current - prediction;
      const uint width = PC::bsr(difference);
      encoder.encode(kBias + 1 + width, &model);
      encoder.encode<std::uint64_t>(difference - (std::uint64_t{1} << width), width);
    } else if (prediction > current) {
      const std::uint64_t difference = prediction - current;
      const uint width = PC::bsr(difference);
      encoder.encode(kBias - 1 - width, &model);
      encoder.encode<std::uint64_t>(difference - (std::uint64_t{1} << width), width);
    } else {
      encoder.encode(kBias, &model);
    }
    previous = map_inverse(current);
  }
  encoder.finish();
  if (encoder.error) throw std::runtime_error("FPZIP range output buffer overflow");
  wire.resize(encoder.bytes());
  return wire;
}

std::vector<std::uint8_t> range_inverse(std::span<const std::uint8_t> wire) {
  if (wire.size() < 4) throw std::runtime_error("truncated FPZIP wire");
  BoundedDecoder decoder(wire);
  decoder.init();
  const std::uint32_t count = decode_header(decoder);
  RCqsmodel model(false, kSymbols);
  std::vector<std::uint64_t> residuals;
  residuals.reserve(count);
  double previous = 0.0;
  for (std::uint32_t i = 0; i < count; ++i) {
    const uint symbol = decoder.decode(&model);
    const std::uint64_t prediction = map_forward(predict_1d(previous));
    std::uint64_t current = prediction;
    if (symbol > kBias) {
      const uint width = symbol - kBias - 1;
      const std::uint64_t difference = (std::uint64_t{1} << width) +
          decoder.decode<std::uint64_t>(width);
      current = prediction + difference;
    } else if (symbol < kBias) {
      const uint width = kBias - 1 - symbol;
      const std::uint64_t difference = (std::uint64_t{1} << width) +
          decoder.decode<std::uint64_t>(width);
      current = prediction - difference;
    }
    residuals.push_back(current - prediction);
    previous = map_inverse(current);
  }
  if (decoder.error) throw std::runtime_error("truncated FPZIP range stream");
  return make_packet(residuals);
}

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw_f64) {
  return range_forward(predictor_forward(raw_f64));
}
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire) {
  return predictor_inverse(range_inverse(wire));
}

}  // namespace exact_actions::fpzip
