#include "exception_separation.hpp"

#include <array>
#include <iostream>
#include <random>

namespace es = exact_actions::exception_separation;

namespace {
template<class F>
void rejects(F function, const char* message) {
  try { function(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}

void verify(std::span<const std::uint32_t> values, unsigned width) {
  const auto parts = es::separate(values, width);
  es::require(parts.low.size() == values.size(), "low count mismatch");
  std::size_t exception = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint64_t factor = std::uint64_t{1} << width;
    es::require(parts.low[i] == values[i] % factor, "low field differs from remainder");
    const auto high = values[i] / factor;
    if (high != 0) {
      es::require(parts.positions.at(exception) == i && parts.high.at(exception) == high,
                  "exception field differs from quotient");
      ++exception;
    }
  }
  es::require(exception == parts.positions.size(), "extra exception");
  const es::Words expected(values.begin(), values.end());
  es::require(es::restore(parts) == expected, "API roundtrip mismatch");
  const auto packet = es::serialize(parts);
  es::require(es::serialize(es::parse(packet)) == packet, "packet roundtrip mismatch");
  const auto packed = es::pack(parts);
  es::require(es::serialize(es::unpack(packed)) == packet, "packed stream roundtrip mismatch");
  es::require(es::restore(es::unpack(packed)) == expected, "packed values mismatch");
}
}  // namespace

int main() {
  std::mt19937 random(0x45485331);
  std::size_t roundtrips = 0;
  for (unsigned width = 0; width <= 32; ++width) {
    for (std::size_t count : {0U, 1U, 2U, 7U, 8U, 9U, 127U, 128U, 255U, 256U, 1025U}) {
      es::Words values(count);
      for (auto& value : values) value = random();
      if (count > 0) values[0] = 0;
      if (count > 1) values[1] = UINT32_MAX;
      if (count > 2) values[2] = width == 32 ? UINT32_MAX : std::uint32_t{1} << width;
      verify(values, width);
      verify(es::Words(count, 0), width);
      roundtrips += 2;
    }
  }

  const es::Words source{0, 7, 8, 255, UINT32_MAX};
  const auto parts = es::separate(source, 3);
  es::require(parts.low == es::Words({0, 7, 0, 7, 7}), "known low stream");
  es::require(parts.positions == es::Words({2, 3, 4}), "known position stream");
  es::require(parts.high == es::Words({1, 31, 536870911}), "known high stream");
  const auto zero = es::separate(source, 0);
  es::require(zero.low == es::Words(5, 0) && zero.positions == es::Words({1, 2, 3, 4}),
              "zero width must retain nonzero originals as high exceptions");
  const auto full = es::separate(source, 32);
  es::require(full.low == source && full.positions.empty() && full.high.empty(),
              "full width must have no exceptions");
  const es::Words golden_source{1, 6};
  const es::Bytes golden_wire{'E','H','P','1',2, 2,0,0,0, 1,0,0,0, 1,1,
                               0x60, 0x80, 0x80};
  es::require(es::pack(es::separate(golden_source, 2)) == golden_wire,
              "A020 MSB recipe known wire mismatch");
  es::require(es::restore(es::unpack(golden_wire)) == golden_source,
              "A020 recipe known wire decode mismatch");

  rejects([&] { es::separate(source, 33); }, "accepted width 33");
  rejects([&] { es::separate(source, UINT32_MAX); }, "accepted huge width");
  auto bad = parts;
  bad.high.pop_back();
  rejects([&] { es::restore(bad); }, "accepted missing high stream value");
  bad = parts; bad.positions.pop_back();
  rejects([&] { es::restore(bad); }, "accepted missing position value");
  bad = parts; bad.positions[1] = bad.positions[0];
  rejects([&] { es::restore(bad); }, "accepted duplicate position");
  bad = parts; std::swap(bad.positions[0], bad.positions[1]);
  rejects([&] { es::restore(bad); }, "accepted unsorted positions");
  bad = parts; bad.positions.back() = static_cast<std::uint32_t>(source.size());
  rejects([&] { es::restore(bad); }, "accepted out of range position");
  bad = parts; bad.high[0] = 0;
  rejects([&] { es::restore(bad); }, "accepted zero exception");
  bad = parts; bad.high[0] = std::uint32_t{1} << 29;
  rejects([&] { es::restore(bad); }, "accepted high overflow");
  bad = parts; bad.low[0] = 8;
  rejects([&] { es::restore(bad); }, "accepted low overflow");
  bad = full; bad.positions = {0}; bad.high = {1};
  rejects([&] { es::restore(bad); }, "accepted full-width exception");
  bad = zero; bad.low[0] = 1;
  rejects([&] { es::restore(bad); }, "accepted nonzero zero-width low");
  rejects([&] { es::words(std::array<std::uint8_t, 1>{0}); }, "accepted unaligned input");

  const auto wire = es::serialize(parts);
  for (std::size_t size = 0; size < wire.size(); ++size)
    rejects([&] { es::parse(std::span(wire).first(size)); }, "accepted truncated EHS1");
  auto corrupt = wire; corrupt.push_back(0);
  rejects([&] { es::parse(corrupt); }, "accepted trailing EHS1 byte");
  corrupt = wire; corrupt[0] = 'X';
  rejects([&] { es::parse(corrupt); }, "accepted wrong EHS1 magic");
  corrupt = wire; corrupt[4] = 33;
  rejects([&] { es::parse(corrupt); }, "accepted EHS1 width 33");
  corrupt = wire; corrupt[9] = 6;
  rejects([&] { es::parse(corrupt); }, "accepted EHS1 exception count greater than N");
  corrupt = wire; std::fill(corrupt.begin() + 5, corrupt.begin() + 9, 0xff);
  rejects([&] { es::parse(corrupt); }, "accepted EHS1 corrupt count");

  const auto packed = es::pack(parts);
  for (std::size_t size = 0; size < packed.size(); ++size)
    rejects([&] { es::unpack(std::span(packed).first(size)); }, "accepted truncated EHP1");
  corrupt = packed; corrupt.push_back(0);
  rejects([&] { es::unpack(corrupt); }, "accepted trailing EHP1 byte");
  corrupt = packed; corrupt[0] = 'X';
  rejects([&] { es::unpack(corrupt); }, "accepted wrong EHP1 magic");
  corrupt = packed; corrupt[4] = 33;
  rejects([&] { es::unpack(corrupt); }, "accepted EHP1 width 33");
  corrupt = packed; corrupt[13] = 33;
  rejects([&] { es::unpack(corrupt); }, "accepted EHP1 position width 33");
  corrupt = packed; corrupt[14] = 30;
  rejects([&] { es::unpack(corrupt); }, "accepted EHP1 high width overflow");
  corrupt = golden_wire; corrupt.back() |= 1;
  rejects([&] { es::unpack(corrupt); }, "accepted packed padding corruption");
  corrupt = golden_wire; corrupt.back() = 0;
  rejects([&] { es::unpack(corrupt); }, "accepted zero/nonminimal high");
  corrupt = es::pack(es::separate({}, 3)); corrupt[13] = 1;
  rejects([&] { es::unpack(corrupt); }, "accepted nonminimal empty position width");
  std::cout << "exception_separation=PASS widths=0..32 roundtrips=" << roundtrips
            << " invalid_streams=PASS A020_recipe=PASS\n";
}
