#include "baseline.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;
using native_baseline86_v1::Block;
using native_baseline86_v1::Bytes;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void write(const fs::path& path, const Bytes& data) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!data.empty())
    output.write(reinterpret_cast<const char*>(data.data()), data.size());
  if (!output) throw std::runtime_error("fixture write failed");
}
Bytes read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}
Block fixture() {
  Block block{"int64", 13, {}, {0b00100100, 0}};
  const std::int64_t values[] = {0, -1, 2, 2, 9, 9, 10, -999, 1024, 42, 42};
  const auto* begin = reinterpret_cast<const std::uint8_t*>(values);
  block.values.assign(begin, begin + sizeof(values));
  return block;
}
Block float_fixture() {
  Block block{"float64", 12, {}, {0b00000110, 0}};
  const std::uint64_t words[] = {
      0x0000000000000000ULL, 0x8000000000000000ULL,
      0x7ff0000000000000ULL, 0xfff0000000000000ULL,
      0x7ff8000000000042ULL, 0x0000000000000001ULL,
      0x3ff0000000000000ULL, 0xc004000000000000ULL,
      0x7fefffffffffffffULL, 0x0010000000000000ULL};
  const auto* begin = reinterpret_cast<const std::uint8_t*>(words);
  block.values.assign(begin, begin + sizeof(words));
  return block;
}
Block finite_float_fixture() {
  Block block{"float64", 6, {}, {0b00000100}};
  const double values[] = {0.0, -0.0, 1.25, -10.5, 42.0};
  const auto* begin = reinterpret_cast<const std::uint8_t*>(values);
  block.values.assign(begin, begin + sizeof(values));
  return block;
}
bool vendored_lightweight_label(const std::string& label) {
  static const std::vector<std::string> labels = {
      "Libdeflate", "FastLZ", "LZF",   "LZFSE", "LZVN",
      "BriefLZ",    "LZAV",   "LZG",   "LZJB",  "TAMP", "Bzip3",
      "FSE", "Huff0", "LZMAT", "LZRW", "WFLZ", "Yappy",
      "YALZ77", "QuickLZ", "SLZ", "Density", "Gipfeli", "GLZA",
      "LZO1", "LZO2A", "LZSSE", "Pithy", "Shrinker", "NRV", "XPACK",
      "BSC", "Crush", "CSC", "FastLZMA2", "Kanzi", "Lizard", "LZHAM",
      "Lzlib", "PPMD8", "Tornado", "Zlib-ng", "Zling", "ZXC"};
  return std::find(labels.begin(), labels.end(), label) != labels.end();
}
Block vendored_bulk_fixture(const std::string& dtype) {
  Block block{dtype, 1024, {}, Bytes(128, 0)};
  std::vector<std::uint64_t> words(1024);
  for (std::size_t i = 0; i < words.size(); ++i) {
    if (dtype == "int64")
      words[i] = static_cast<std::uint64_t>((i / 64) & 3);
    else
      words[i] = 0x3ff0000000000000ULL + ((i / 64) & 3);
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(words.data());
  block.values.assign(begin, begin + words.size() * sizeof(words[0]));
  return block;
}
Block method_fixture(const std::string& label,const std::string& dtype) {
  if(label=="CORAD") {
    auto b=vendored_bulk_fixture("float64");
    for(std::size_t i=8;i<b.values.size();++i)b.values[i]=b.values[i%8];
    return b;
  }
  if(label=="BUFF") {
    const double values[]={0.5,1.0,1.25,2.5,-1.5,-0.25,64.0,64.5};
    const auto* p=reinterpret_cast<const std::uint8_t*>(values);
    return Block{"float64",8,Bytes(p,p+sizeof(values)),{0}};
  }
  if(vendored_lightweight_label(label))return vendored_bulk_fixture(dtype);
  if(dtype=="int64")return fixture();
  if(label=="Camel"||label=="Elf"||label=="Elf+")return finite_float_fixture();
  return float_fixture();
}
void roundtrip(const std::string& method, const Block& block) {
  try {
    const auto frame = native_baseline86_v1::encode(method, block);
    require(native_baseline86_v1::decode(frame) == block, "roundtrip failed");
  } catch (const std::exception& error) {
    throw std::runtime_error(method + ": " + error.what());
  }
}
}  // namespace

int main(int argc, char** argv) try {
  const auto& records = native_baseline86_v1::inventory();
  require(records.size() == 86, "inventory is not 86 labels");
  const auto expected_linked=86-!native_baseline86_v1::method("LZSSE").linked
                               -!native_baseline86_v1::method("GLZA").linked;
  require(std::count_if(records.begin(), records.end(),
                        [](const auto& record) { return record.linked; }) == expected_linked,
          "linked v3 inventory count drifted");
  require(std::count_if(records.begin(), records.end(), [](const auto& record) {
            return record.primary_comparison;
          }) == expected_linked,
          "primary v3 inventory count drifted");
  if (argc == 1) {
    const auto block = fixture();
    const auto float_block = float_fixture();
    native_baseline86_v1::validate(block);
    native_baseline86_v1::validate(float_block);
    for (const auto& record : native_baseline86_v1::inventory()) {
      if (!record.linked) continue;
      for (const auto& dtype : record.dtypes)
        roundtrip(record.label, Block{dtype, 3, {}, {0x07}});
      if (std::find(record.dtypes.begin(), record.dtypes.end(), block.dtype) !=
          record.dtypes.end())
        roundtrip(record.label, method_fixture(record.label,"int64"));
      if (std::find(record.dtypes.begin(), record.dtypes.end(), float_block.dtype) !=
          record.dtypes.end()) {
        roundtrip(record.label, method_fixture(record.label,"float64"));
      }
    }
    Block reserved{"int64", 1, {}, {0}};
    const std::int64_t reserved_value = INT64_MIN;
    const auto* reserved_begin = reinterpret_cast<const std::uint8_t*>(&reserved_value);
    reserved.values.assign(reserved_begin, reserved_begin + sizeof(reserved_value));
    for (const auto* label : {"Chimp128", "Gorilla"}) {
      bool rejected_reserved = false;
      try { (void)native_baseline86_v1::encode(label, reserved); }
      catch (const native_baseline86_v1::Unsupported&) { rejected_reserved = true; }
      require(rejected_reserved, "reserved end marker was not rejected");
    }
    auto corrupt = native_baseline86_v1::encode("Zlib", block);
    corrupt[corrupt.size() / 2] ^= 1;
    bool rejected = false;
    try { (void)native_baseline86_v1::decode(corrupt); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "corrupt complete frame was accepted");
    corrupt.pop_back();
    rejected = false;
    try { (void)native_baseline86_v1::decode(corrupt); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "truncated complete frame was accepted");
    bool unsupported = false;
    try { (void)native_baseline86_v1::encode("not-a-catalog-label", block); }
    catch (const native_baseline86_v1::Unsupported&) { unsupported = true; }
    require(unsupported, "unknown label did not fail closed");
    for(const auto* label:{"CORAD","BUFF"}) {
      bool inapplicable=false;
      try{(void)native_baseline86_v1::encode(label,float_fixture());}
      catch(const native_baseline86_v1::Unsupported&){inapplicable=true;}
      require(inapplicable,"lossy original profile was silently repaired");
    }
    std::cout << "native baseline86 in-process roundtrip checks passed\n";
    return 0;
  }
  require(argc == 3,
          "expected write-fixture/check-fixture/encode-matrix/decode-matrix DIRECTORY");
  const fs::path directory = argv[2];
  if (std::string(argv[1]) == "write-fixture") {
    fs::create_directories(directory);
    const auto block = fixture();
    write(directory / "values.bin", block.values);
    write(directory / "missing.bitmap", block.missing);
    write(directory / "empty.values.bin", {});
    write(directory / "all-missing.bitmap", Bytes{0x07});
    return 0;
  }
  if (std::string(argv[1]) == "check-fixture") {
    const auto expected = fixture();
    require(read(directory / "decoded.values.bin") == expected.values,
            "cold-decoded values mismatch");
    require(read(directory / "decoded.missing.bitmap") == expected.missing,
            "cold-decoded bitmap mismatch");
    return 0;
  }
  if (std::string(argv[1]) == "encode-matrix") {
    fs::create_directories(directory);
    std::ofstream manifest(directory / "matrix.tsv", std::ios::trunc);
    if (!manifest) throw std::runtime_error("cannot write cold matrix manifest");
    std::size_t index = 0;
    for (const auto& record : native_baseline86_v1::inventory()) {
      if (!record.linked) continue;
      for (const auto* dtype : {"int64", "float64"}) {
        if (std::find(record.dtypes.begin(), record.dtypes.end(), dtype) ==
            record.dtypes.end())
          continue;
        const auto block = method_fixture(record.label,dtype);
        const auto filename = "matrix-" + std::to_string(index++) + ".ngf";
        write(directory / filename, native_baseline86_v1::encode(record.label, block));
        write(directory / (filename + ".raw"), block.values);
        write(directory / (filename + ".missing"), block.missing);
        manifest << record.label << '\t' << dtype << '\t' << filename << '\n';
      }
    }
    if (!manifest || index == 0) throw std::runtime_error("empty cold matrix");
    std::cout << "cold matrix encoded frames=" << index << '\n';
    return 0;
  }
  if (std::string(argv[1]) == "decode-matrix") {
    std::ifstream manifest(directory / "matrix.tsv");
    if (!manifest) throw std::runtime_error("cannot read cold matrix manifest");
    std::size_t count = 0;
    std::string line;
    while (std::getline(manifest, line)) {
      std::istringstream fields(line);
      std::string label, dtype, filename;
      if (!std::getline(fields, label, '\t') || !std::getline(fields, dtype, '\t') ||
          !std::getline(fields, filename) || label.empty() || filename.empty())
        throw std::runtime_error("invalid cold matrix manifest row");
      const auto expected = method_fixture(label,dtype);
      if (native_baseline86_v1::decode(read(directory / filename)) != expected)
        throw std::runtime_error(label + ": cold matrix roundtrip mismatch");
      ++count;
    }
    if (count == 0) throw std::runtime_error("empty cold matrix decode");
    std::cout << "cold matrix decoded frames=" << count << '\n';
    return 0;
  }
  throw std::runtime_error("unknown test command");
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
