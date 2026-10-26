#include "input_builder.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
using J = nlohmann::json;
namespace fs = std::filesystem;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::string digest(const std::string& bytes) {
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  check(context != nullptr, "cannot allocate digest context");
  std::array<unsigned char, EVP_MAX_MD_SIZE> output{};
  unsigned output_size = 0;
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1 &&
                  EVP_DigestFinal_ex(context, output.data(), &output_size) == 1;
  EVP_MD_CTX_free(context);
  check(ok && output_size == 32, "SHA256 failed");
  static constexpr char hex[] = "0123456789abcdef";
  std::string result(64, '0');
  for (unsigned i = 0; i < output_size; ++i) {
    result[2 * i] = hex[output[i] >> 4];
    result[2 * i + 1] = hex[output[i] & 15];
  }
  return result;
}

void write(const fs::path& path, const std::string& bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  check(bool(output), "cannot create fixture " + path.string());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  check(bool(output), "cannot write fixture " + path.string());
}

std::string read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  check(bool(input), "cannot read " + path.string());
  const auto length = input.tellg();
  check(length >= 0, "cannot size " + path.string());
  input.seekg(0);
  std::string bytes(static_cast<std::size_t>(length), '\0');
  if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  check(bool(input) || bytes.empty(), "short read " + path.string());
  return bytes;
}

template <typename UInt>
void append_le(std::string& output, UInt value) {
  for (std::size_t i = 0; i < sizeof(UInt); ++i)
    output.push_back(static_cast<char>((value >> (8U * i)) & 0xffU));
}

std::string bitmap(std::uint64_t rows, const std::vector<std::uint64_t>& missing) {
  std::string result(static_cast<std::size_t>((rows + 7) / 8), '\0');
  for (const auto row : missing) {
    check(row < rows, "fixture missing row outside bitmap");
    result[static_cast<std::size_t>(row / 8)] |=
        static_cast<char>(1U << (row % 8));
  }
  return result;
}

struct Fixture {
  fs::path root;
  fs::path plan;
  fs::path float32_values_path;
  std::string float32_values;

  Fixture() {
    root = fs::temp_directory_path() /
           ("operator-recommend135-input-builder-test-" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
  }

  ~Fixture() { fs::remove_all(root); }
};

J identity(const std::string& dataset, const std::string& column,
           const std::string& dtype) {
  return {{"column", column},
          {"dataset_id", dataset},
          {"dtype", dtype},
          {"role", "value"},
          {"source_column", column},
          {"table_name", "measurements"}};
}

J add_block(const fs::path& root, const std::string& dataset,
            const std::string& column, const std::string& dtype,
            const std::string& entity_id, std::uint64_t entity_ordinal,
            std::uint64_t block_index, std::uint64_t row_count,
            const std::vector<std::uint64_t>& missing_rows,
            const std::string& values) {
  const std::size_t width = dtype == "float32" ? 4 : 8;
  const auto valid_count = row_count - missing_rows.size();
  check(values.size() == valid_count * width, "fixture value count mismatch");
  const auto mask = bitmap(row_count, missing_rows);
  const auto shard = root / "payloads" /
                     (column + "-" + entity_id + "-" + std::to_string(block_index) +
                      ".shard");
  const std::string prefix = "PAD";
  const std::string separator = "xy";
  write(shard, prefix + values + separator + mask + "END");
  return {{"block_id", dataset + ":" + column + ":e" +
                           std::to_string(entity_ordinal) + ":b" +
                           std::to_string(block_index)},
          {"block_index", block_index},
          {"column", column},
          {"dataset_id", dataset},
          {"dense_row_slot_bytes", row_count * width},
          {"dtype", dtype},
          {"entity_id", entity_id},
          {"entity_ordinal", entity_ordinal},
          {"expected_sequence_sha256", digest("sequence-" + column + entity_id +
                                                std::to_string(block_index))},
          {"missing_bitmap_length", mask.size()},
          {"missing_bitmap_offset", prefix.size() + values.size() + separator.size()},
          {"missing_bitmap_path", fs::absolute(shard).string()},
          {"missing_bitmap_sha256", digest(mask)},
          {"missing_count", missing_rows.size()},
          {"original_numeric_bytes", values.size()},
          {"role", "value"},
          {"row_count", row_count},
          {"source_column", column},
          {"source_original_numeric_bytes", values.size()},
          {"table_name", "measurements"},
          {"valid_count", valid_count},
          {"values_length", values.size()},
          {"values_offset", prefix.size()},
          {"values_path", fs::absolute(shard).string()},
          {"values_sha256", digest(values)}};
}

std::string float_values(std::size_t count, std::uint32_t seed) {
  static constexpr std::array<std::uint32_t, 8> special = {
      0x7fc00001U, 0x80000000U, 0x7f800000U, 0xff800000U,
      0x00000001U, 0x3f800000U, 0xbf800000U, 0xffc00001U};
  std::string values;
  values.reserve(count * 4);
  for (std::size_t i = 0; i < count; ++i) {
    const auto value = i < special.size() ? special[i]
                                          : seed + static_cast<std::uint32_t>(i * 17U);
    append_le(values, value);
  }
  return values;
}

std::string int64_values(std::size_t count, std::uint64_t seed) {
  std::string values;
  values.reserve(count * 8);
  for (std::size_t i = 0; i < count; ++i)
    append_le(values, seed + static_cast<std::uint64_t>(i));
  return values;
}

fs::path add_manifest(Fixture& fixture, const std::string& column,
                      const std::string& dtype, const J& blocks, J& plan,
                      std::size_t ordinal) {
  const std::string dataset = "dataset-main";
  const auto path = fixture.root / "manifests" / (column + ".json");
  J manifest{{"column_identity", identity(dataset, column, dtype)},
             {"column_key", identity(dataset, column, dtype).dump()},
             {"dataset_id", dataset},
             {"schema", "native-baseline86-scaled-column-segment-manifest-v1"},
             {"source_catalog_path", "/fixture/blocks.jsonl"},
             {"source_catalog_sha256", digest("catalog-" + column)},
             {"source_dataset_manifest", "/fixture/dataset_manifest.json"},
             {"source_dataset_manifest_sha256", digest("dataset-manifest")},
             {"status", "rendered_from_accepted_offset_catalog"},
             {"blocks", blocks}};
  const auto bytes = manifest.dump();
  write(path, bytes);
  const auto absolute = fs::absolute(path).string();
  plan["jobs"].push_back({{"column_id", digest("column-" + column)},
                           {"dataset_id", dataset},
                           {"job_id", "job-" + std::to_string(ordinal)},
                           {"manifests", J::array({absolute})},
                           {"methods", J::array({"Zstd"})},
                           {"mode", "native"},
                           {"scheduling_original_numeric_bytes", 0}});
  plan["manifest_sha256"][absolute] = digest(bytes);
  return path;
}

Fixture make_fixture() {
  Fixture fixture;
  J plan{{"schema", "cpp135-valid48-column-plan-v1"},
         {"jobs", J::array()},
         {"manifest_sha256", J::object()},
         {"counts", {{"datasets", 1}, {"value_columns", 3}}}};

  J f32_blocks = J::array();
  const auto f32_a0 = float_values(67, 0x10000000U);
  const auto f32_a1 = float_values(66, 0x20000000U);
  fixture.float32_values = f32_a0 + f32_a1;
  f32_blocks.push_back(add_block(fixture.root, "dataset-main", "temperature",
                                 "float32", "station-a", 0, 0, 70,
                                 {0, 5, 69}, f32_a0));
  f32_blocks.push_back(add_block(fixture.root, "dataset-main", "temperature",
                                 "float32", "station-a", 0, 1, 70,
                                 {1, 2, 68, 69}, f32_a1));
  f32_blocks.push_back(add_block(fixture.root, "dataset-main", "temperature",
                                 "float32", "station-b", 1, 0, 66, {},
                                 float_values(66, 0x30000000U)));
  f32_blocks.push_back(add_block(fixture.root, "dataset-main", "temperature",
                                 "float32", "station-b", 1, 1, 66, {},
                                 float_values(66, 0x40000000U)));
  const auto f32_manifest =
      add_manifest(fixture, "temperature", "float32", f32_blocks, plan, 0);
  fixture.float32_values_path = f32_blocks[0]["values_path"].get<std::string>();

  J i64_blocks = J::array();
  i64_blocks.push_back(add_block(fixture.root, "dataset-main", "counter", "int64",
                                 "station-a", 0, 0, 128, {},
                                 int64_values(128, 0x1122334455660000ULL)));
  add_manifest(fixture, "counter", "int64", i64_blocks, plan, 1);

  J f64_blocks = J::array();
  f64_blocks.push_back(add_block(fixture.root, "dataset-main", "pressure", "float64",
                                 "station-a", 0, 0, 70, {},
                                 int64_values(70, 0x3ff0000000000000ULL)));
  f64_blocks.push_back(add_block(fixture.root, "dataset-main", "pressure", "float64",
                                 "station-b", 1, 0, 70, {},
                                 int64_values(70, 0x4000000000000000ULL)));
  add_manifest(fixture, "pressure", "float64", f64_blocks, plan, 2);

  fixture.plan = fixture.root / "PLAN_COLUMNS.json";
  write(fixture.plan, plan.dump());
  return fixture;
}

const J& sample(const J& result, const std::string& dtype, const std::string& scale) {
  for (const auto& item : result.at("samples"))
    if (item.at("dtype") == dtype && item.at("scale") == scale) return item;
  throw std::runtime_error("missing sample " + dtype + "/" + scale);
}

void test_windows_preserve_bits_masks_and_entity_boundaries() {
  auto fixture = make_fixture();
  const auto output = fixture.root / "output";
  const auto result = operator_recommend135_v1::build_inputs(fixture.plan, output, 80);

  check(result.at("schema") == "operator-recommend135-inputs-v1", "wrong schema");
  check(result.at("samples").size() == 3, "expected f32 u64/primary and one deduped i64");
  check(result.at("coverage").at("covered_dataset_count") == 1,
        "dataset coverage missing");
  check(result.at("coverage").at("missing_dataset_ids").empty(),
        "fixture dataset should be covered");

  const auto& primary = sample(result, "float32", "primary");
  check(primary.at("element_count") == 66, "primary N must be floor(L/2)");
  check(primary.at("series_valid_count") == 133,
        "longest entity must win without station-b concatenation");
  check(primary.at("entity_identity").at("entity_id") == "station-a",
        "wrong entity selected");
  check(primary.at("head").at("row_start") == 1 &&
            primary.at("head").at("row_end_exclusive") == 68 &&
            primary.at("head").at("row_count") == 67,
        "head row range is not exact");
  check(primary.at("tail").at("row_start") == 70 &&
            primary.at("tail").at("row_end_exclusive") == 138 &&
            primary.at("tail").at("row_count") == 68,
        "tail row range is not exact");
  check(primary.at("head").at("row_end_exclusive").get<std::uint64_t>() <=
            primary.at("tail").at("row_start").get<std::uint64_t>(),
        "head and tail overlap");

  const auto head_values = read(primary.at("head").at("values_path"));
  const auto tail_values = read(primary.at("tail").at("values_path"));
  check(head_values == fixture.float32_values.substr(0, 66 * 4),
        "float32 head bits changed");
  check(tail_values == fixture.float32_values.substr(67 * 4, 66 * 4),
        "float32 tail bits changed");
  check(static_cast<unsigned char>(head_values[0]) == 0x01 &&
            static_cast<unsigned char>(head_values[1]) == 0x00 &&
            static_cast<unsigned char>(head_values[2]) == 0xc0 &&
            static_cast<unsigned char>(head_values[3]) == 0x7f,
        "NaN payload bits were not preserved");
  check(read(primary.at("head").at("missing_path")) == bitmap(67, {4}),
        "head missing bitmap was not rebased exactly");
  check(read(primary.at("tail").at("missing_path")) == bitmap(68, {1, 2}),
        "tail missing bitmap was not preserved exactly");
  check(primary.at("head").at("values_sha256") == digest(head_values) &&
            primary.at("tail").at("values_sha256") == digest(tail_values),
        "output value digest mismatch");
  check(primary.at("head").at("source_blocks").size() == 1 &&
            primary.at("tail").at("source_blocks").size() == 1,
        "unexpected source evidence span");
  check(primary.at("head").at("source_blocks")[0].at("full_values_region_verified") &&
            primary.at("head").at("source_blocks")[0].at("full_missing_region_verified"),
        "full source regions were not recorded as verified");
  check(primary.at("head").at("source_blocks")[0].at("source_row_start") == 1 &&
            primary.at("head").at("source_blocks")[0].at("source_row_end_exclusive") == 68,
        "source row slice evidence is imprecise");

  const auto& u64 = sample(result, "float32", "u64");
  check(u64.at("element_count") == 64, "u64 layer must contain 64 valid values");
  check(u64.at("head").at("row_count") == 65 && u64.at("tail").at("row_count") == 64,
        "u64 missing-aware row counts are wrong");

  const auto& i64 = sample(result, "int64", "primary");
  check(i64.at("element_count") == 64, "128-value series must use N=64");
  check(i64.at("equivalent_scales") == J::array({"u64"}),
        "N=64 u64 layer should be explicitly deduplicated");

  bool saw_float64_gap = false;
  for (const auto& gap : result.at("gaps"))
    if (gap.at("dataset_id") == "dataset-main" && gap.at("dtype") == "float64") {
      saw_float64_gap = true;
      check(gap.at("reason") == "no_same_entity_series_with_128_valid_values",
            "cross-entity gap reason is unclear");
      check(gap.at("longest_same_entity_valid_count") == 70,
            "float64 entities were incorrectly concatenated");
    }
  check(saw_float64_gap, "missing float64 coverage gap");
  check(J::parse(read(output / "INPUTS.json")) == result,
        "returned JSON and persisted INPUTS.json differ");
}

void test_changed_source_region_is_rejected() {
  auto fixture = make_fixture();
  auto bytes = read(fixture.float32_values_path);
  bytes[3] ^= 0x01;
  write(fixture.float32_values_path, bytes);
  try {
    (void)operator_recommend135_v1::build_inputs(fixture.plan,
                                                 fixture.root / "changed-source", 80);
    throw std::runtime_error("changed source region was accepted");
  } catch (const std::exception& error) {
    check(std::string(error.what()).find("values SHA256 mismatch") != std::string::npos,
          std::string("unexpected source mutation error: ") + error.what());
  }
}

void test_changed_manifest_is_rejected() {
  auto fixture = make_fixture();
  auto plan = J::parse(read(fixture.plan));
  const auto manifest_path = plan.at("jobs")[0].at("manifests")[0].get<std::string>();
  write(manifest_path, read(manifest_path) + "\n");
  try {
    (void)operator_recommend135_v1::build_inputs(fixture.plan,
                                                 fixture.root / "changed-manifest", 80);
    throw std::runtime_error("changed manifest was accepted");
  } catch (const std::exception& error) {
    check(std::string(error.what()).find("manifest SHA256 mismatch") != std::string::npos,
          std::string("unexpected manifest mutation error: ") + error.what());
  }
}

}  // namespace

int main() {
  try {
    test_windows_preserve_bits_masks_and_entity_boundaries();
    test_changed_source_region_is_rejected();
    test_changed_manifest_is_rejected();
    std::cout << "input_builder_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "input_builder_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
