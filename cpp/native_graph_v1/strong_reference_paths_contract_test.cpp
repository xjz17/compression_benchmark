#include "graph_internal.hpp"
#include "stage_dispatch.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace native_graph_v1;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<RootInput> roots() {
  Bytes values;
  values.reserve(4096 * sizeof(std::uint64_t));
  for (std::uint64_t index = 0; index < 4096; ++index) {
    const std::uint64_t word = 1'700'000'000'000ULL + (index / 16) * 100 + index % 4;
    for (unsigned byte = 0; byte < 8; ++byte)
      values.push_back(static_cast<std::uint8_t>(word >> (8 * byte)));
  }
  return {{{Kind::Numeric, 8, std::move(values)}, true, "int64"}};
}

Step stage(StreamId input, std::string action, std::string profile = "bytes") {
  return {StepKind::Stage, {input},
          {std::move(action), std::move(profile), {}}, {}, -1};
}

Recipe recipe(std::initializer_list<const char*> actions,
              std::string first_profile = "bytes") {
  Recipe result;
  result.policy_registry = "existing-6341-actions-reference-chain-contract-v1";
  StreamId input = 0;
  for (const auto* action : actions) {
    const auto profile = input == 0 ? first_profile : "bytes";
    result.steps.push_back(stage(input++, action, profile));
  }
  result.steps.push_back({StepKind::Stop});
  return result;
}

Recipe two_leaf_lz4_recipe(std::string action) {
  Recipe result;
  result.policy_registry = "existing-6341-actions-two-leaf-lz4-contract-v1";
  result.steps = {
      stage(0, std::move(action), "int64"),
      stage(1, "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1"),
      stage(3, "op.match_token_format.lz4.cpp_v1"),
      stage(4, "op.integer_code.lz4.cpp_v1"),
      stage(2, "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1"),
      stage(6, "op.match_token_format.lz4.cpp_v1"),
      stage(7, "op.integer_code.lz4.cpp_v1"),
      {StepKind::Stop},
  };
  return result;
}

const std::vector<Recipe>& recipes() {
  static const std::vector<Recipe> value = {
      recipe({"op.delta.ts_2diff.cpp_v1",
              "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
              "op.match_token_format.lz4.cpp_v1",
              "op.integer_code.lz4.cpp_v1"}, "int64"),
      recipe({"op.delta.stream_vbyte.cpp_v1",
              "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
              "op.match_token_format.lz4.cpp_v1",
              "op.integer_code.lz4.cpp_v1"}, "int64"),
      recipe({"op.xor.previous_word.cpp_v2",
              "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
              "op.match_token_format.lz4.cpp_v1",
              "op.integer_code.lz4.cpp_v1"}, "int64"),
      recipe({"op.xor.gorilla.cpp_v1",
              "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",
              "op.match_token_format.lz4.cpp_v1",
              "op.integer_code.lz4.cpp_v1"}, "int64"),
      two_leaf_lz4_recipe("op.dictionary.project_words64.cpp_v1"),
      two_leaf_lz4_recipe(
          "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1"),
      two_leaf_lz4_recipe("op.xor.previous_word.semantic_ports.cpp_v1"),
      recipe({"op.block_transform.bzip2_bwt.cpp_reference_v1",
              "op.symbol_recode.bzip2_mtf.cpp_reference_v1",
              "op.run_length.byte_optimal.cpp_v1",
              "op.entropy_huffman.project_bytes.cpp_v1"}),
  };
  return value;
}

void verify(const Decoded& decoded) {
  const auto expected = roots();
  require(decoded.original_numeric_bytes == expected[0].stream.data.size(),
          "reference chain changed B_raw");
  require(decoded.roots.size() == 1 &&
              decoded.roots[0].stream == expected[0].stream,
          "reference chain changed numeric bits");
  require(decoded.roots[0].logical_view == "int64" &&
              decoded.roots[0].original_numeric_payload,
          "reference chain changed root metadata");
}

void run_contract() {
  const auto dispatcher = make_default_dispatcher();
  for (std::size_t index = 0; index < recipes().size(); ++index) {
    const auto measured = benchmark(roots(), recipes()[index], *dispatcher, 1, 2);
    require(measured.original_numeric_bytes == roots()[0].stream.data.size(),
            "reference-chain timing denominator drift");
    require(measured.encode_nanoseconds.size() == 2 &&
                measured.decode_nanoseconds.size() == 2,
            "reference-chain timing sample count drift");
    verify(decode(measured.frame, *dispatcher));
    auto corrupted = measured.frame;
    require(corrupted.size() > 64, "reference-chain frame unexpectedly small");
    corrupted[corrupted.size() / 2] ^= 1;
    bool rejected = false;
    try {
      (void)decode(corrupted, *dispatcher);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "reference-chain accepted a corrupted complete frame");
    std::cout << "strong_reference_path=" << index << " raw_bytes="
              << measured.original_numeric_bytes << " frame_bytes="
              << measured.frame.size() << " PASS\n";
  }
}

void put_u64(std::ofstream& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.put(static_cast<char>(value >> shift));
}

std::uint64_t get_u64(std::ifstream& input) {
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    const auto byte = input.get();
    require(byte != std::char_traits<char>::eof(),
            "truncated reference bundle length");
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(byte)) << shift;
  }
  return value;
}

void write_fixture(const std::string& path) {
  const auto dispatcher = make_default_dispatcher();
  std::ofstream output(path, std::ios::binary);
  require(bool(output), "failed to create reference-chain fixture");
  put_u64(output, recipes().size());
  for (const auto& plan : recipes()) {
    const auto frame = encode(roots(), plan, *dispatcher);
    put_u64(output, frame.size());
    output.write(reinterpret_cast<const char*>(frame.data()),
                 static_cast<std::streamsize>(frame.size()));
  }
  require(bool(output), "failed to write reference-chain fixture");
}

void decode_fixture(const std::string& path) {
  const auto dispatcher = make_default_dispatcher();
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "failed to open reference-chain fixture");
  require(get_u64(input) == recipes().size(), "reference-chain fixture count drift");
  for (std::size_t index = 0; index < recipes().size(); ++index) {
    const auto size = get_u64(input);
    require(size > 0 && size < (1ULL << 32), "invalid reference-chain frame size");
    Bytes frame(size);
    input.read(reinterpret_cast<char*>(frame.data()),
               static_cast<std::streamsize>(frame.size()));
    require(bool(input), "truncated reference-chain frame");
    verify(decode(frame, *dispatcher));
  }
  require(input.get() == std::char_traits<char>::eof(),
          "trailing reference-chain fixture bytes");
  std::cout << "strong_reference_paths_independent_cold_decode=PASS\n";
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
          "usage: strong_reference_paths_contract_test [encode|decode FILE]");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
