#include "graph_internal.hpp"
#include "stage_dispatch.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace native_graph_v1;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<RootInput> roots() {
  Bytes values;
  values.reserve(8192 * sizeof(std::uint64_t));
  for (std::uint64_t index = 0; index < 8192; ++index) {
    // Smooth low bytes and repeated high bytes are representative of numeric
    // columns where byte shuffling before LZ4 is useful.  The words are still
    // treated as opaque bits, so the test also covers exact uint64 recovery.
    const std::uint64_t word = 1'700'000'000'000ULL + (index / 4) * 1000 + index % 4;
    for (unsigned byte = 0; byte < 8; ++byte)
      values.push_back(static_cast<std::uint8_t>(word >> (8 * byte)));
  }
  return {{{Kind::Numeric, 8, std::move(values)}, true, "int64"}};
}

Step stage(StreamId input, std::string action) {
  return {StepKind::Stage, {input}, {std::move(action), "bytes", {}}, {}, -1};
}

Recipe recipe() {
  Recipe result;
  result.policy_registry = "existing-6341-actions-strong-path-contract-v1";
  result.steps = {
      stage(0, "op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1"),
      stage(1, "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1"),
      stage(2, "op.match_token_format.lz4.cpp_v1"),
      stage(3, "op.integer_code.lz4.cpp_v1"),
      {StepKind::Stop},
  };
  return result;
}

void verify(const Decoded& decoded) {
  const auto expected = roots();
  require(decoded.original_numeric_bytes == expected[0].stream.data.size(),
          "strong path changed B_raw");
  require(decoded.roots.size() == 1, "strong path changed root count");
  require(decoded.roots[0].stream == expected[0].stream,
          "strong path changed numeric bits");
  require(decoded.roots[0].logical_view == "int64",
          "strong path changed logical view");
  require(decoded.roots[0].original_numeric_payload,
          "strong path changed numeric-root flag");
}

void run_contract() {
  const auto dispatcher = make_default_dispatcher();
  const auto input = roots();
  const auto plan = recipe();
  const auto frame = encode(input, plan, *dispatcher);
  verify(decode(frame, *dispatcher));

  const auto parsed = detail::parse_frame(frame, {});
  require(parsed.nodes.size() == 4, "strong path node count drift");
  const std::vector<std::string> expected_actions = {
      "op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1",
      "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
      "op.match_token_format.lz4.cpp_v1",
      "op.integer_code.lz4.cpp_v1",
  };
  for (std::size_t index = 0; index < expected_actions.size(); ++index)
    require(parsed.nodes[index].step.stage.action == expected_actions[index],
            "strong path action order drift");

  const auto measured = benchmark(input, plan, *dispatcher, 1, 3);
  require(measured.original_numeric_bytes == input[0].stream.data.size(),
          "strong path timing denominator drift");
  require(measured.encode_nanoseconds.size() == 3 &&
              measured.decode_nanoseconds.size() == 3,
          "strong path timing sample count drift");
  verify(decode(measured.frame, *dispatcher));

  // The complete NGF1 frame checksum must reject a leaf-payload mutation.
  auto corrupted = frame;
  require(corrupted.size() > 64, "strong path frame unexpectedly small");
  corrupted[corrupted.size() / 2] ^= 1;
  bool rejected = false;
  try {
    (void)decode(corrupted, *dispatcher);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "strong path accepted corrupted complete frame");

  std::cout << "strong_existing_path=PASS raw_bytes="
            << measured.original_numeric_bytes << " frame_bytes="
            << measured.frame.size() << " nodes=" << parsed.nodes.size() << "\n";
}

void write_fixture(const std::string& path) {
  const auto dispatcher = make_default_dispatcher();
  const auto frame = encode(roots(), recipe(), *dispatcher);
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(frame.data()),
               static_cast<std::streamsize>(frame.size()));
  require(bool(output), "failed to write strong-path fixture");
}

void decode_fixture(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "failed to open strong-path fixture");
  Bytes frame((std::istreambuf_iterator<char>(input)), {});
  const auto dispatcher = make_default_dispatcher();
  verify(decode(frame, *dispatcher));
  std::cout << "strong_existing_path_independent_cold_decode=PASS\n";
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
      throw std::runtime_error("usage: strong_existing_paths_contract_test [encode|decode FILE]");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
