#include "snappy_exact_stages.hpp"
#include "unsigned_leb128.hpp"

#include <snappy.h>

#include <cstring>
#include <limits>
#include <stdexcept>

namespace exact_actions::snappy {
namespace {

constexpr std::uint8_t kMatchMagic[] = {'S', 'N', 'M', '1'};
constexpr std::uint8_t kTokenMagic[] = {'S', 'N', 'T', '1'};

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 4 > input.size()) throw std::runtime_error("Snappy stage truncated u32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i)
    value |= static_cast<std::uint32_t>(input[pos++]) << (8 * i);
  return value;
}

bool has_magic(std::span<const std::uint8_t> input, const std::uint8_t (&magic)[4]) {
  return input.size() >= 4 && std::memcmp(input.data(), magic, 4) == 0;
}

void put_varint(std::vector<std::uint8_t>& out, std::uint32_t value) {
  compression_operators::unsigned_leb128::append(out, value);
}

std::uint32_t get_varint(std::span<const std::uint8_t> input, std::size_t& pos) {
  return static_cast<std::uint32_t>(
      compression_operators::unsigned_leb128::read(input, pos, 32));
}

struct Record {
  bool literal = false;
  std::vector<std::uint8_t> bytes;
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
};

std::vector<Record> parse_tag_body(std::span<const std::uint8_t> body,
                                   std::uint32_t expected_size) {
  std::vector<Record> records;
  std::size_t pos = 0;
  std::size_t produced = 0;
  while (pos < body.size()) {
    const std::uint8_t tag = body[pos++];
    const int type = tag & 3;
    if (type == 0) {
      std::uint32_t length = tag >> 2;
      if (length < 60) {
        ++length;
      } else {
        const std::uint32_t count = length - 59;
        if (count == 0 || count > 4 || pos + count > body.size())
          throw std::runtime_error("Snappy malformed literal length");
        length = 0;
        for (std::uint32_t i = 0; i < count; ++i)
          length |= static_cast<std::uint32_t>(body[pos++]) << (8 * i);
        ++length;
      }
      if (pos + length > body.size()) throw std::runtime_error("Snappy truncated literal");
      Record record;
      record.literal = true;
      record.bytes.assign(body.begin() + static_cast<std::ptrdiff_t>(pos),
                          body.begin() + static_cast<std::ptrdiff_t>(pos + length));
      record.length = length;
      records.push_back(std::move(record));
      pos += length;
      produced += length;
      continue;
    }

    Record record;
    record.literal = false;
    if (type == 1) {
      if (pos >= body.size()) throw std::runtime_error("Snappy truncated COPY_1");
      record.length = 4 + ((tag >> 2) & 7);
      record.offset = (static_cast<std::uint32_t>(tag & 0xe0) << 3) | body[pos++];
    } else if (type == 2) {
      if (pos + 2 > body.size()) throw std::runtime_error("Snappy truncated COPY_2");
      record.length = 1 + (tag >> 2);
      record.offset = static_cast<std::uint32_t>(body[pos]) |
                      (static_cast<std::uint32_t>(body[pos + 1]) << 8);
      pos += 2;
    } else {
      if (pos + 4 > body.size()) throw std::runtime_error("Snappy truncated COPY_4");
      record.length = 1 + (tag >> 2);
      record.offset = static_cast<std::uint32_t>(body[pos]) |
                      (static_cast<std::uint32_t>(body[pos + 1]) << 8) |
                      (static_cast<std::uint32_t>(body[pos + 2]) << 16) |
                      (static_cast<std::uint32_t>(body[pos + 3]) << 24);
      pos += 4;
    }
    if (record.offset == 0 || record.offset > produced)
      throw std::runtime_error("Snappy invalid copy offset");
    produced += record.length;
    records.push_back(std::move(record));
  }
  if (produced != expected_size) throw std::runtime_error("Snappy decoded size mismatch");
  return records;
}

std::vector<std::uint8_t> make_match_packet(std::uint32_t raw_size,
                                            const std::vector<Record>& records) {
  std::vector<std::uint8_t> out(std::begin(kMatchMagic), std::end(kMatchMagic));
  put_u32(out, raw_size);
  put_u32(out, static_cast<std::uint32_t>(records.size()));
  for (const Record& record : records) {
    out.push_back(record.literal ? 0 : 1);
    if (record.literal) {
      put_u32(out, static_cast<std::uint32_t>(record.bytes.size()));
      out.insert(out.end(), record.bytes.begin(), record.bytes.end());
    } else {
      put_u32(out, record.offset);
      put_u32(out, record.length);
    }
  }
  return out;
}

std::pair<std::uint32_t, std::vector<Record>> read_match_packet(
    std::span<const std::uint8_t> packet) {
  if (!has_magic(packet, kMatchMagic)) throw std::runtime_error("bad SNM1 packet");
  std::size_t pos = 4;
  const auto raw_size = get_u32(packet, pos);
  const auto count = get_u32(packet, pos);
  std::vector<Record> records;
  records.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (pos >= packet.size()) throw std::runtime_error("truncated SNM1 record");
    const std::uint8_t type = packet[pos++];
    Record record;
    if (type == 0) {
      record.literal = true;
      const auto length = get_u32(packet, pos);
      if (pos + length > packet.size()) throw std::runtime_error("truncated SNM1 literal");
      record.bytes.assign(packet.begin() + static_cast<std::ptrdiff_t>(pos),
                          packet.begin() + static_cast<std::ptrdiff_t>(pos + length));
      record.length = length;
      pos += length;
    } else if (type == 1) {
      record.offset = get_u32(packet, pos);
      record.length = get_u32(packet, pos);
      if (record.offset == 0 || record.length < 4)
        throw std::runtime_error("invalid SNM1 match");
    } else {
      throw std::runtime_error("unknown SNM1 record type");
    }
    records.push_back(std::move(record));
  }
  if (pos != packet.size()) throw std::runtime_error("SNM1 trailing bytes");
  return {raw_size, std::move(records)};
}

void emit_literal(std::vector<std::uint8_t>& out, std::span<const std::uint8_t> bytes) {
  if (bytes.empty()) throw std::runtime_error("empty Snappy literal record");
  const std::uint32_t n = static_cast<std::uint32_t>(bytes.size() - 1);
  if (n < 60) {
    out.push_back(static_cast<std::uint8_t>(n << 2));
  } else {
    int count = 1;
    while (count < 4 && (n >> (8 * count)) != 0) ++count;
    out.push_back(static_cast<std::uint8_t>((59 + count) << 2));
    for (int i = 0; i < count; ++i)
      out.push_back(static_cast<std::uint8_t>(n >> (8 * i)));
  }
  out.insert(out.end(), bytes.begin(), bytes.end());
}

void emit_copy(std::vector<std::uint8_t>& out, std::uint32_t offset,
               std::uint32_t length) {
  if (length < 4 || length > 64 || offset == 0 || offset >= 65536)
    throw std::runtime_error("unsupported Snappy semantic copy");
  if (length < 12 && offset < 2048) {
    out.push_back(static_cast<std::uint8_t>(1 + ((length - 4) << 2) +
                                            ((offset >> 8) << 5)));
    out.push_back(static_cast<std::uint8_t>(offset));
  } else {
    out.push_back(static_cast<std::uint8_t>(2 + ((length - 1) << 2)));
    out.push_back(static_cast<std::uint8_t>(offset));
    out.push_back(static_cast<std::uint8_t>(offset >> 8));
  }
}

}  // namespace

std::vector<std::uint8_t> original_encode(
    std::span<const std::uint8_t> input) {
  std::vector<char> wire(::snappy::MaxCompressedLength(input.size()));
  std::size_t wire_size = 0;
  ::snappy::RawCompress(reinterpret_cast<const char*>(input.data()), input.size(),
                        wire.data(), &wire_size);
  return {reinterpret_cast<const std::uint8_t*>(wire.data()),
          reinterpret_cast<const std::uint8_t*>(wire.data()) + wire_size};
}

// Port of snap 1.1.1. Copyright 2011, The Snappy-Rust Authors. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
std::vector<std::uint8_t> snap111_match_forward(std::span<const std::uint8_t> input) {
  if (input.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Snappy input exceeds uint32 size");
  std::vector<Record> records;
  for (std::size_t base = 0; base < input.size(); base += 65536) {
    const auto raw = input.subspan(base, std::min<std::size_t>(65536, input.size() - base));
    const auto size = raw.size();
    auto literal = [&](std::size_t start, std::size_t end) {
      if (start == end) return;
      Record r; r.literal = true; r.length = end - start;
      r.bytes.assign(raw.begin() + start, raw.begin() + end);
      records.push_back(std::move(r));
    };
    auto copy = [&](std::size_t offset, std::size_t length) {
      Record r; r.offset = offset; r.length = length; records.push_back(std::move(r));
    };
    if (size < 17) { literal(0, size); continue; }
    std::size_t table_size = 256; unsigned shift = 24;
    while (table_size < 16384 && table_size < size) { table_size *= 2; --shift; }
    std::vector<std::size_t> table(table_size, 0);
    auto word = [&](std::size_t pos) {
      return std::uint32_t(raw[pos]) | (std::uint32_t(raw[pos + 1]) << 8) |
             (std::uint32_t(raw[pos + 2]) << 16) | (std::uint32_t(raw[pos + 3]) << 24);
    };
    auto hash_at = [&](std::size_t pos) { return (word(pos) * std::uint32_t(0x1e35a7bd)) >> shift; };
    std::size_t pos = 1, next_emit = 0, limit = size - 15;
    auto next_hash = hash_at(pos); bool done = false;
    while (!done) {
      std::size_t skip = 32, next_pos = pos, candidate = 0;
      for (;;) {
        pos = next_pos; const auto stride = skip >> 5;
        next_pos = pos + stride; skip += stride;
        if (next_pos > limit) { literal(next_emit, size); done = true; break; }
        candidate = table[next_hash]; table[next_hash] = pos; next_hash = hash_at(next_pos);
        if (word(pos) == word(candidate)) break;
      }
      if (done) break;
      literal(next_emit, pos);
      for (;;) {
        const auto start = pos; pos += 4; auto match_pos = candidate + 4;
        while (pos < size && raw[pos] == raw[match_pos]) { ++pos; ++match_pos; }
        const auto offset = start - candidate; auto length = pos - start;
        while (length >= 68) { copy(offset, 64); length -= 64; }
        if (length > 64) { copy(offset, 60); length -= 60; }
        copy(offset, length); next_emit = pos;
        if (pos >= limit) { literal(next_emit, size); done = true; break; }
        table[hash_at(pos - 1)] = pos - 1;
        const auto current_hash = hash_at(pos); candidate = table[current_hash]; table[current_hash] = pos;
        if (word(pos) != word(candidate)) { next_hash = hash_at(pos + 1); ++pos; break; }
      }
    }
  }
  return make_match_packet(static_cast<std::uint32_t>(input.size()), records);
}

std::vector<std::uint8_t> lz_match_forward(std::span<const std::uint8_t> input) {
  if (input.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Snappy input exceeds uint32 size");
  const auto wire = original_encode(input);
  const auto span = std::span<const std::uint8_t>(wire);
  std::size_t pos = 0;
  const auto raw_size = get_varint(span, pos);
  if (raw_size != input.size()) throw std::runtime_error("Snappy raw size mismatch");
  return make_match_packet(raw_size, parse_tag_body(span.subspan(pos), raw_size));
}

std::vector<std::uint8_t> lz_match_inverse(std::span<const std::uint8_t> packet) {
  auto [raw_size, records] = read_match_packet(packet);
  std::vector<std::uint8_t> out;
  out.reserve(raw_size);
  for (const Record& record : records) {
    if (record.literal) {
      out.insert(out.end(), record.bytes.begin(), record.bytes.end());
    } else {
      if (record.offset > out.size()) throw std::runtime_error("SNM1 copy before start");
      for (std::uint32_t i = 0; i < record.length; ++i)
        out.push_back(out[out.size() - record.offset]);
    }
  }
  if (out.size() != raw_size) throw std::runtime_error("SNM1 expansion size mismatch");
  return out;
}

std::vector<std::uint8_t> token_format_forward(std::span<const std::uint8_t> packet) {
  auto [raw_size, records] = read_match_packet(packet);
  std::vector<std::uint8_t> body;
  for (const Record& record : records) {
    if (record.literal) emit_literal(body, record.bytes);
    else emit_copy(body, record.offset, record.length);
  }
  std::vector<std::uint8_t> out(std::begin(kTokenMagic), std::end(kTokenMagic));
  put_u32(out, raw_size);
  put_u32(out, static_cast<std::uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> token_format_inverse(std::span<const std::uint8_t> packet) {
  if (!has_magic(packet, kTokenMagic)) throw std::runtime_error("bad SNT1 packet");
  std::size_t pos = 4;
  const auto raw_size = get_u32(packet, pos);
  const auto body_size = get_u32(packet, pos);
  if (pos + body_size != packet.size()) throw std::runtime_error("SNT1 body mismatch");
  return make_match_packet(raw_size, parse_tag_body(packet.subspan(pos), raw_size));
}

std::vector<std::uint8_t> integer_code_forward(std::span<const std::uint8_t> packet) {
  if (!has_magic(packet, kTokenMagic)) throw std::runtime_error("bad SNT1 packet");
  std::size_t pos = 4;
  const auto raw_size = get_u32(packet, pos);
  const auto body_size = get_u32(packet, pos);
  if (pos + body_size != packet.size()) throw std::runtime_error("SNT1 body mismatch");
  std::vector<std::uint8_t> out;
  put_varint(out, raw_size);
  out.insert(out.end(), packet.begin() + static_cast<std::ptrdiff_t>(pos), packet.end());
  return out;
}

std::vector<std::uint8_t> integer_code_inverse(std::span<const std::uint8_t> wire) {
  std::size_t pos = 0;
  const auto raw_size = get_varint(wire, pos);
  std::vector<std::uint8_t> out(std::begin(kTokenMagic), std::end(kTokenMagic));
  put_u32(out, raw_size);
  put_u32(out, static_cast<std::uint32_t>(wire.size() - pos));
  out.insert(out.end(), wire.begin() + static_cast<std::ptrdiff_t>(pos), wire.end());
  return out;
}

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> input) {
  return integer_code_forward(token_format_forward(lz_match_forward(input)));
}

std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire) {
  return lz_match_inverse(token_format_inverse(integer_code_inverse(wire)));
}

}  // namespace exact_actions::snappy
