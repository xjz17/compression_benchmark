#include "gorilla_exact_stages.hpp"
#include "graph.hpp"
#include "stage_dispatch.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace native_graph_v1;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T>
Bytes raw(const std::vector<T>& values) {
  Bytes output(values.size() * sizeof(T));
  if (!output.empty()) std::memcpy(output.data(), values.data(), output.size());
  return output;
}

std::vector<std::int64_t> integer_values() {
  std::vector<std::int64_t> values;
  values.reserve(4096);
  for (std::int64_t index = 0; index < 4096; ++index)
    values.push_back(1'700'000'000'000LL + (index / 16) * 100 + index % 4);
  return values;
}

std::vector<std::int64_t> floating_words() {
  const std::array<std::uint64_t, 12> special = {
      0x0000000000000000ULL, 0x8000000000000000ULL,
      0x0000000000000001ULL, 0x8000000000000001ULL,
      0x3ff0000000000000ULL, 0xbff0000000000000ULL,
      0x7ff0000000000000ULL, 0xfff0000000000000ULL,
      0x7ff8000000000001ULL, 0x7ff0000000000001ULL,
      0xfff8000000000001ULL, 0x400921fb54442d18ULL,
  };
  std::vector<std::int64_t> values;
  values.reserve(4096 + special.size());
  for (const auto bits : special) values.push_back(static_cast<std::int64_t>(bits));
  for (std::uint64_t index = 0; index < 4096; ++index) {
    const double value = 20.0 + static_cast<double>(index / 8) * 0.01 +
                         static_cast<double>(index % 3) * 0.001;
    values.push_back(static_cast<std::int64_t>(std::bit_cast<std::uint64_t>(value)));
  }
  return values;
}

Step stage(StreamId input, std::string action, std::string profile) {
  return {StepKind::Stage, {input}, {std::move(action), std::move(profile), {}},
          {}, -1};
}

Recipe recipe(std::string profile) {
  Recipe result;
  result.policy_registry = "gorilla-source-equivalent-two-stage-v1";
  result.steps = {
      stage(0, "op.xor.gorilla.cpp_v1", profile),
      stage(1, "op.trailing_zero_pack.gorilla.cpp_v1", "bytes"),
      {StepKind::Stop},
  };
  return result;
}

RootInput root(std::string profile, const std::vector<std::int64_t>& values) {
  return {{Kind::Numeric, 8, raw(values)}, true, std::move(profile)};
}

bool same_root(const RootInput& left, const RootInput& right) {
  return left.stream == right.stream &&
         left.original_numeric_payload == right.original_numeric_payload &&
         left.logical_view == right.logical_view;
}

void reject_reserved(const StageDispatcher& dispatcher, std::string profile,
                     std::uint64_t marker) {
  const std::vector<std::int64_t> values{1, static_cast<std::int64_t>(marker), 2};
  bool rejected = false;
  try {
    (void)dispatcher.forward(
        {"op.xor.gorilla.cpp_v1", std::move(profile), {}},
        {root("reserved-test", values).stream});
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "Gorilla graph accepted its reserved end marker");
}

Bytes verify_profile(const StageDispatcher& dispatcher, std::string profile,
                     const std::vector<std::int64_t>& values) {
  const auto source = root(profile, values);
  const auto xor_result = dispatcher.forward(
      {"op.xor.gorilla.cpp_v1", profile, {}}, {source.stream});
  require(xor_result.outputs.size() == 1, "Gorilla XOR output count drift");
  const auto expected_xor = profile == "int64"
                                ? exact_actions::gorilla::xor_forward_int64(values)
                                : exact_actions::gorilla::xor_forward_double_bits(values);
  require(xor_result.outputs[0].data == expected_xor,
          "Gorilla graph XOR differs from complete-method source");

  const auto pack_result = dispatcher.forward(
      {"op.trailing_zero_pack.gorilla.cpp_v1", "bytes", {}},
      xor_result.outputs);
  require(pack_result.outputs.size() == 1, "Gorilla pack output count drift");
  const auto expected_wire = profile == "int64"
                                 ? exact_actions::gorilla::compose_int64(values)
                                 : exact_actions::gorilla::compose_double_bits(values);
  require(pack_result.outputs[0].data == expected_wire,
          "Gorilla two-stage wire differs from complete-method source");

  const auto unpaced = dispatcher.inverse(
      {"op.trailing_zero_pack.gorilla.cpp_v1", "bytes", {}},
      pack_result.outputs, pack_result.metadata);
  const auto restored = dispatcher.inverse(
      {"op.xor.gorilla.cpp_v1", profile, {}}, unpaced, xor_result.metadata);
  require(restored.size() == 1 && restored[0] == source.stream,
          "Gorilla source-equivalent stage inverse mismatch");

  const std::vector<RootInput> roots{source};
  const auto measured = benchmark(roots, recipe(profile), dispatcher, 1, 2);
  require(measured.original_numeric_bytes == source.stream.data.size() &&
              measured.encode_nanoseconds.size() == 2 &&
              measured.decode_nanoseconds.size() == 2,
          "Gorilla complete-frame timing contract drift");
  const auto decoded = decode(measured.frame, dispatcher);
  require(decoded.roots.size() == 1 && same_root(decoded.roots[0], roots[0]),
          "Gorilla source-equivalent complete-frame roundtrip mismatch");
  return measured.frame;
}

void put_u64(std::ofstream& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.put(static_cast<char>(value >> shift));
}

std::uint64_t get_u64(std::ifstream& input) {
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    const auto byte = input.get();
    require(byte != std::char_traits<char>::eof(), "truncated Gorilla bundle");
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(byte)) << shift;
  }
  return value;
}

std::vector<Bytes> frames() {
  const auto dispatcher = make_default_dispatcher();
  reject_reserved(*dispatcher, "int64", 0x8000000000000000ULL);
  reject_reserved(*dispatcher, "float64", 0x7ff8000000000000ULL);
  return {verify_profile(*dispatcher, "int64", integer_values()),
          verify_profile(*dispatcher, "float64", floating_words())};
}

void run_contract() {
  const auto results = frames();
  require(results.size() == 2, "Gorilla frame count drift");
  std::cout << "gorilla_source_equivalent=PASS int64_frame_bytes="
            << results[0].size() << " float64_frame_bytes=" << results[1].size()
            << "\n";
}

void write_fixture(const std::string& path) {
  const auto results = frames();
  std::ofstream output(path, std::ios::binary);
  require(bool(output), "failed to create Gorilla frame bundle");
  put_u64(output, results.size());
  for (const auto& frame : results) {
    put_u64(output, frame.size());
    output.write(reinterpret_cast<const char*>(frame.data()),
                 static_cast<std::streamsize>(frame.size()));
  }
  require(bool(output), "failed to write Gorilla frame bundle");
}

void decode_fixture(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "failed to open Gorilla frame bundle");
  require(get_u64(input) == 2, "Gorilla frame bundle count drift");
  const auto dispatcher = make_default_dispatcher();
  const std::array<std::vector<std::int64_t>, 2> expected{
      integer_values(), floating_words()};
  const std::array<std::string, 2> profiles{"int64", "float64"};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto size = get_u64(input);
    require(size > 0 && size < (1ULL << 30), "invalid Gorilla frame size");
    Bytes frame(size);
    input.read(reinterpret_cast<char*>(frame.data()),
               static_cast<std::streamsize>(frame.size()));
    require(bool(input), "truncated Gorilla frame");
    const auto decoded = decode(frame, *dispatcher);
    const std::vector<RootInput> roots{root(profiles[index], expected[index])};
    require(decoded.roots.size() == 1 &&
                same_root(decoded.roots[0], roots[0]),
            "independent Gorilla complete-frame decode mismatch");
  }
  require(input.get() == std::char_traits<char>::eof(),
          "trailing Gorilla bundle bytes");
  std::cout << "gorilla_source_equivalent_independent_cold_decode=PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      run_contract();
    } else if (argc == 3 && std::string(argv[1]) == "encode") {
      write_fixture(argv[2]);
    } else if (argc == 3 && std::string(argv[1]) == "decode") {
      decode_fixture(argv[2]);
    } else {
      throw std::runtime_error(
          "usage: gorilla_source_equivalent_contract_test [encode|decode FILE]");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
