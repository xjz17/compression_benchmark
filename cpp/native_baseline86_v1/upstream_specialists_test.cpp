#include "upstream_specialists.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>

namespace fs = std::filesystem;
using native_baseline86_v1::upstream_specialists::Admission;
using native_baseline86_v1::upstream_specialists::Bytes;

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Bytes generic_fixture() {
  Bytes out;
  for (unsigned repeat = 0; repeat != 8; ++repeat) {
    out.insert(out.end(), repeat * 13 + 1, static_cast<std::uint8_t>(repeat));
    for (unsigned value = 0; value != 256; ++value)
      out.push_back(static_cast<std::uint8_t>(value));
  }
  return out;
}

Bytes float_fixture(std::size_t count = 4097) {
  const std::uint64_t seeds[] = {
      0x0000000000000000ULL, 0x8000000000000000ULL,
      0x3ff0000000000000ULL, 0xc004000000000000ULL,
      0x7ff0000000000000ULL, 0xfff0000000000000ULL,
      0x7ff8000000000042ULL, 0x0000000000000001ULL,
      0x0010000000000000ULL, 0x7fefffffffffffffULL};
  std::vector<std::uint64_t> words(count);
  for (std::size_t index = 0; index != count; ++index)
    words[index] = index % 19 ? seeds[index % std::size(seeds)]
                              : (index * 6364136223846793005ULL +
                                 1442695040888963407ULL);
  Bytes raw(words.size() * 8);
  std::memcpy(raw.data(), words.data(), raw.size());
  return raw;
}

Bytes constant_fixture() {
  const std::uint64_t word = 0x8000000000000000ULL;
  std::vector<std::uint64_t> words(257, word);
  Bytes raw(words.size() * 8);
  std::memcpy(raw.data(), words.data(), raw.size());
  return raw;
}

Bytes buff_fixture() {
  const double values[] = {0.5, 1.0, 1.25, 2.5, -1.5, -0.25, 64.0, 64.5};
  Bytes raw(sizeof(values));
  std::memcpy(raw.data(), values, sizeof(values));
  return raw;
}

Bytes fixture(const std::string& label) {
  if (label == "Chimp" || label == "FPCmp-R" || label == "FPCmp-S")
    return float_fixture();
  if (label == "CORAD") return constant_fixture();
  if (label == "BUFF") return buff_fixture();
  return generic_fixture();
}

std::string filename(std::string label) {
  for (auto& value : label)
    if (value == '.' || value == '-') value = '_';
  return label;
}

void write(const fs::path& path, const Bytes& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!output) throw std::runtime_error("specialist fixture write failed");
}

Bytes read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("specialist fixture read failed");
  return {std::istreambuf_iterator<char>(input), {}};
}

void roundtrip(const std::string& label, const Bytes& raw) {
  using namespace native_baseline86_v1::upstream_specialists;
  const auto first = encode(label, raw);
  const auto second = encode(label, raw);
  check(first == second, "specialist encoding is nondeterministic");
  check(decode(label, first, raw.size()) == raw,
        "specialist bit-exact roundtrip failed");
  auto corrupt = first;
  corrupt.back() ^= 1;
  bool rejected = false;
  try { (void)decode(label, corrupt, raw.size()); }
  catch (const std::exception&) { rejected = true; }
  check(rejected, "specialist accepted corrupted payload");
  corrupt = first;
  corrupt.pop_back();
  rejected = false;
  try { (void)decode(label, corrupt, raw.size()); }
  catch (const std::exception&) { rejected = true; }
  check(rejected, "specialist accepted truncated payload");
  corrupt = first;
  corrupt.push_back(0);
  rejected = false;
  try { (void)decode(label, corrupt, raw.size()); }
  catch (const std::exception&) { rejected = true; }
  check(rejected, "specialist accepted trailing payload");
}
}  // namespace

int main(int argc, char** argv) try {
  using namespace native_baseline86_v1::upstream_specialists;
  const auto& all = profiles();
  check(all.size() == 10, "specialist profile count drifted");
  std::set<std::string> labels;
  for (const auto& item : all) labels.insert(item.label);
  check(labels.size() == all.size(), "duplicate specialist label");
  check(std::count_if(all.begin(), all.end(), [](const auto& item) {
          return item.admission == Admission::direct_upstream;
        }) == 3, "direct upstream profile count drifted");
  check(std::count_if(all.begin(), all.end(), [](const auto& item) {
          return item.admission == Admission::direct_project_generic;
        }) == 5, "direct generic profile count drifted");
  check(std::count_if(all.begin(), all.end(), [](const auto& item) {
          return item.admission == Admission::direct_exact_subset;
        }) == 2, "direct exact-subset profile count drifted");

  if (argc == 1) {
    for (const auto& item : all) roundtrip(item.label, fixture(item.label));
    for (const auto* label : {"RLE", "Dict.", "ANS", "Huffman", "Arithmetic",
                              "Chimp", "FPCmp-R", "FPCmp-S"})
      roundtrip(label, {});

    bool rejected = false;
    try { (void)encode("CORAD", buff_fixture()); }
    catch (const NotApplicable&) { rejected = true; }
    check(rejected, "CORAD nonconstant input was not rejected");

    rejected = false;
    try { (void)encode("CORAD", {}); }
    catch (const NotApplicable&) { rejected = true; }
    check(rejected, "CORAD empty input was not rejected");

    const double infinity = std::numeric_limits<double>::infinity();
    Bytes nonfinite(sizeof(infinity) * 2);
    std::memcpy(nonfinite.data(), &infinity, sizeof(infinity));
    std::memcpy(nonfinite.data() + sizeof(infinity), &infinity,
                sizeof(infinity));
    rejected = false;
    try { (void)encode("CORAD", nonfinite); }
    catch (const NotApplicable&) { rejected = true; }
    check(rejected, "CORAD nonfinite constant input was not rejected");

    const double lossy_value = 0.1;
    Bytes lossy(sizeof(lossy_value));
    std::memcpy(lossy.data(), &lossy_value, sizeof(lossy_value));
    rejected = false;
    try { (void)encode("BUFF", lossy); }
    catch (const NotApplicable&) { rejected = true; }
    check(rejected, "BUFF non-bit-exact input was not rejected");

    rejected = false;
    try { (void)encode("BUFF", {}); }
    catch (const NotApplicable&) { rejected = true; }
    check(rejected, "BUFF empty input was not rejected");

    const double wide_values[] = {-1.0e9, 1.0e9};
    Bytes wide(sizeof(wide_values));
    std::memcpy(wide.data(), wide_values, sizeof(wide_values));
    rejected = false;
    try { (void)encode("BUFF", wide); }
    catch (const NotApplicable&) { rejected = true; }
    check(rejected, "BUFF input above the vendored four-slice limit was not rejected");

    auto chimp = encode("Chimp", float_fixture(64));
    rejected = false;
    try { (void)decode("FPCmp-S", chimp, 64 * 8); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "specialist method mismatch was accepted");
    std::cout << "upstream specialist in-process checks passed\n";
    return 0;
  }

  check(argc == 3, "expected write-fixtures/decode-fixtures DIRECTORY");
  const fs::path directory = argv[2];
  if (std::string(argv[1]) == "write-fixtures") {
    fs::create_directories(directory);
    std::ofstream manifest(directory / "manifest.tsv", std::ios::trunc);
    for (const auto& item : all) {
      const auto raw = fixture(item.label);
      const auto frame = encode(item.label, raw);
      const auto stem = filename(item.label);
      write(directory / (stem + ".usp"), frame);
      write(directory / (stem + ".raw"), raw);
      manifest << item.label << '\t' << stem << '\n';
    }
    check(static_cast<bool>(manifest), "specialist manifest write failed");
    return 0;
  }
  if (std::string(argv[1]) == "decode-fixtures") {
    std::ifstream manifest(directory / "manifest.tsv");
    check(static_cast<bool>(manifest), "specialist manifest read failed");
    std::string label, stem;
    std::size_t count = 0;
    while (manifest >> label >> stem) {
      const auto raw = read(directory / (stem + ".raw"));
      const auto frame = read(directory / (stem + ".usp"));
      check(decode(label, frame, raw.size()) == raw,
            "specialist cold decode mismatch");
      ++count;
    }
    check(count == all.size(), "specialist cold fixture count mismatch");
    std::cout << "upstream specialist cold frames decoded=" << count << '\n';
    return 0;
  }
  throw std::runtime_error("unknown specialist test command");
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
