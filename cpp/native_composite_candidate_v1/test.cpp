#include "composite.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace native_composite_candidate_v1;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!output) throw std::runtime_error("fixture write failed");
}
Bytes read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("fixture read failed");
  return {std::istreambuf_iterator<char>(input), {}};
}
Block fixture() {
  Block block{"float64", 1030, {}, Bytes(129, 0)};
  block.missing[0] = 0b00100100;
  block.missing.back() = 0b00100000;
  std::vector<std::uint64_t> words;
  words.reserve(1027);
  const std::uint64_t specials[] = {
      0x0000000000000000ULL, 0x8000000000000000ULL,
      0x7ff0000000000000ULL, 0xfff0000000000000ULL,
      0x7ff8000000000042ULL, 0x0000000000000001ULL,
      0x0010000000000000ULL, 0x7fefffffffffffffULL};
  for (std::size_t index = 0; index < 1027; ++index)
    words.push_back(index < std::size(specials)
                        ? specials[index]
                        : 0x3ff0000000000000ULL + ((index / 32) & 7));
  const auto* begin = reinterpret_cast<const std::uint8_t*>(words.data());
  block.values.assign(begin, begin + words.size() * sizeof(words[0]));
  native_baseline86_v1::validate(block);
  return block;
}
std::vector<Backend> linked_backends() {
  std::vector<Backend> result;
  for (const auto& item : inventory())
    if (item.linked) result.push_back(item.backend);
  return result;
}
}  // namespace

int main(int argc, char** argv) try {
  const auto block = fixture();
  const Weights weights{0.5, 0.0, 0.5};
  const auto linked = linked_backends();
  require(linked.size() >= 2, "fewer than two composite candidates are linked");
  require(!registry_sha256().empty() && registry_sha256().size() == 64,
          "invalid registry digest");

  if (argc == 1) {
    for (const auto backend : linked) {
      const auto& item = candidate(backend);
      require(item.implementation_class == "direct_current",
              "composite candidate is not a direct current implementation");
      require(!item.source.empty() && !item.configuration.empty(),
              "candidate lineage is incomplete");
      FixedSelector selector(backend);
      const auto frame = encode(selector, weights, block);
      require(frame == native_baseline86_v1::encode(item.label, block),
              "default composite wire differs from baseline complete NGF1 frame");
      const auto restored = decode(frame);
      require(restored.backend == backend && restored.block == block &&
                  restored.original_numeric_bytes == block.values.size() &&
                  !restored.embedded_selection_provenance,
              "composite frame roundtrip failed");
      auto measured = measure(selector, weights, block, 0, 2);
      require(measured.original_numeric_bytes == block.values.size() &&
                  measured.backend == backend &&
                  measured.compression_ns.size() == 2 &&
                  measured.decompression_ns.size() == 2,
              "measurement contract failed");
      auto direct = measure_direct_backend(backend, block, 0, 2);
      require(direct.frame == frame && direct.backend == backend &&
                  direct.original_numeric_bytes == block.values.size() &&
                  direct.selector_identity == "direct-baseline86/" + item.label,
              "direct baseline measurement differs from complete NGF1 wire");
      auto corrupt = frame;
      corrupt[corrupt.size() / 2] ^= 1;
      bool rejected = false;
      try { (void)decode(corrupt); } catch (const std::exception&) { rejected = true; }
      require(rejected, "corrupt NCC1 frame was accepted");
      corrupt.assign(frame.begin(), frame.end() - 1);
      rejected = false;
      try { (void)decode(corrupt); } catch (const std::exception&) { rejected = true; }
      require(rejected, "truncated NCC1 frame was accepted");
      const auto provenance_frame =
          encode(selector, weights, block, FrameMode::ProvenanceNcc1);
      const auto provenance = decode(provenance_frame);
      require(provenance.backend == backend && provenance.weights == weights &&
                  provenance.block == block &&
                  provenance.embedded_selection_provenance &&
                  provenance.selector_identity == selector.identity(),
              "NCC1 audit-frame roundtrip failed");
    }
    const auto features = task_features(block);
    require(features.size() == 18, "task feature width drifted");
    for (const auto value : features)
      require(std::isfinite(value), "task feature is not finite");
    auto policy = train_only_objective_linear_v1();
    const std::array<std::pair<Weights, Backend>, 6> expected{{
        {{1, 0, 0}, Backend::Xpack}, {{0, 1, 0}, Backend::Blosc2},
        {{0, 0, 1}, Backend::Blosc}, {{0.5, 0.5, 0}, Backend::Blosc2},
        {{0.5, 0, 0.5}, Backend::Blosc}, {{0.4, 0.3, 0.3}, Backend::Blosc}}};
    for (const auto& [objective, backend] : expected) {
      const auto frame = encode(policy, objective, block);
      require(decode(frame).backend == backend,
              "train-only objective selector choice drifted");
    }
    const Block all_missing{"int64", 3, {}, {0x07}};
    FixedSelector empty_selector(Backend::Blosc2);
    const auto empty_frame = encode(empty_selector, weights, all_missing);
    const auto empty_decoded = decode(empty_frame);
    require(empty_decoded.block == all_missing &&
                empty_decoded.original_numeric_bytes == 0,
            "zero-denominator complete-frame roundtrip failed");
    const auto empty_measurement =
        measure(empty_selector, weights, all_missing, 0, 1);
    require(empty_measurement.original_numeric_bytes == 0,
            "zero denominator was not preserved");
    std::cout << "native composite candidate checks passed, linked="
              << linked.size() << '\n';
    return 0;
  }
  require(argc == 3, "expected write-fixture or check-fixture DIRECTORY");
  const fs::path directory = argv[2];
  if (std::string(argv[1]) == "write-fixture") {
    fs::create_directories(directory);
    write(directory / "values.bin", block.values);
    write(directory / "missing.bitmap", block.missing);
    write(directory / "empty.values.bin", {});
    write(directory / "all-missing.bitmap", Bytes{0x07});
    return 0;
  }
  if (std::string(argv[1]) == "check-fixture") {
    require(read(directory / "decoded.values.bin") == block.values,
            "cold decoded values mismatch");
    require(read(directory / "decoded.missing.bitmap") == block.missing,
            "cold decoded bitmap mismatch");
    return 0;
  }
  throw std::invalid_argument("unknown test command");
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
