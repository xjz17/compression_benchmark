#include "baseline.hpp"
#include "decimal_scaling.hpp"
#include "column_scales.hpp"
#include "typed_segments.hpp"
#include "packed_frames.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string_view>
#include <set>
#include <array>
#include <chrono>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;
using J = nlohmann::json;
using native_baseline86_v1::Block;
using native_baseline86_v1::Bytes;
namespace typed_segments=native_baseline86_v1::typed_segments;

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

std::uint64_t number(const char* text, const char* field) {
  if (!text || !*text) fail(std::string("missing ") + field);
  std::string value(text);
  std::size_t used = 0;
  const auto parsed = std::stoull(value, &used);
  if (used != value.size()) fail(std::string("invalid ") + field);
  return parsed;
}

Bytes read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open input file: " + path.string());
  input.seekg(0, std::ios::end);
  const auto length = input.tellg();
  if (length < 0) fail("cannot size input file: " + path.string());
  input.seekg(0);
  Bytes result(static_cast<std::size_t>(length));
  if (!result.empty()) input.read(reinterpret_cast<char*>(result.data()), length);
  if (!input) fail("cannot read input file: " + path.string());
  return result;
}

void write(const fs::path& path, std::span<const std::uint8_t> data) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) fail("cannot open output file: " + path.string());
  if (!data.empty())
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
  if (!output) fail("cannot write output file: " + path.string());
}

std::string sha256(std::span<const std::uint8_t> data) {
  auto* context = EVP_MD_CTX_new();
  if (!context) fail("cannot allocate SHA-256 context");
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned length = 0;
  const auto* ptr = data.empty() ? reinterpret_cast<const unsigned char*>("")
                                 : data.data();
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, ptr, data.size()) == 1 &&
                  EVP_DigestFinal_ex(context, digest, &length) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || length != 32) fail("SHA-256 failed");
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned i = 0; i < length; ++i) out << std::setw(2) << unsigned(digest[i]);
  return out.str();
}

std::string file_sha256(const fs::path&path){
  std::ifstream file(path,std::ios::binary);if(!file)fail("cannot hash file: "+path.string());
  auto*context=EVP_MD_CTX_new();if(!context)fail("cannot allocate file digest");
  bool ok=EVP_DigestInit_ex(context,EVP_sha256(),nullptr)==1;
  std::array<char,1<<20>buffer{};
  while(file){file.read(buffer.data(),buffer.size());if(file.gcount())ok=ok&&EVP_DigestUpdate(context,buffer.data(),file.gcount())==1;}
  unsigned char digest[EVP_MAX_MD_SIZE];unsigned length=0;
  ok=ok&&file.eof()&&EVP_DigestFinal_ex(context,digest,&length)==1;
  EVP_MD_CTX_free(context);if(!ok||length!=32)fail("file SHA-256 failed");
  std::ostringstream out;out<<std::hex<<std::setfill('0');for(unsigned i=0;i<length;++i)out<<std::setw(2)<<unsigned(digest[i]);return out.str();
}

std::string self_binary_sha256() {
#if defined(__linux__)
  const auto path=fs::canonical("/proc/self/exe");
#elif defined(__APPLE__)
  std::uint32_t size=4096;
  std::vector<char> buffer(size);
  if (_NSGetExecutablePath(buffer.data(),&size)!=0) {
    buffer.resize(size);
    if (_NSGetExecutablePath(buffer.data(),&size)!=0) fail("cannot identify running CLI");
  }
  const auto path=fs::canonical(buffer.data());
#else
  fail("running binary qualification requires Linux or macOS");
#endif
#if defined(__linux__) || defined(__APPLE__)
  return sha256(read(path));
#endif
}

std::string store_complete_frame(const fs::path& root,
                                 std::span<const std::uint8_t> frame,
                                 const std::string& digest) {
  if (digest.size() != 64) fail("invalid complete-frame SHA-256");
  const fs::path relative = fs::path(digest.substr(0, 2)) / (digest + ".ngf1");
  const fs::path target = root / relative;
  fs::create_directories(target.parent_path());
  if (fs::exists(target)) {
    const auto existing = read(target);
    if (sha256(existing) == digest &&
        std::equal(existing.begin(), existing.end(), frame.begin(), frame.end()))
      return relative.generic_string();
  }
  auto temporary = target;
  temporary += ".partial";
  write(temporary, frame);
  std::error_code error;
  fs::rename(temporary, target, error);
  if (error) {
    if (!fs::exists(target))
      fail("cannot publish content-addressed complete frame: " + error.message());
    const auto existing = read(target);
    if (sha256(existing) != digest ||
        !std::equal(existing.begin(), existing.end(), frame.begin(), frame.end()))
      fail("concurrent complete-frame publication differs");
    fs::remove(temporary, error);
  }
  return relative.generic_string();
}

J inventory_json(bool scaled = false) {
  J methods = J::array();
  std::size_t linked = 0, primary = 0, proxy = 0;
  const auto records = scaled ? native_baseline86_v1::scaling_inventory()
                              : native_baseline86_v1::inventory();
  for (const auto& record : records) {
    methods.push_back({
        {"label", record.label}, {"linked", record.linked},
        {"implementation_class", record.implementation_class},
        {"primary_comparison", record.primary_comparison},
        {"implementation", record.implementation}, {"source", record.source},
        {"configuration", record.configuration}, {"dtypes", record.dtypes},
        {"limitations", record.limitations},
        {"shared_or_project", record.shared_or_project},
    });
    if (scaled) {
      methods.back()["base_configuration"] = native_baseline86_v1::method(record.label).configuration;
      methods.back()["scaling_configuration"] = native_baseline86_v1::scaling_configuration;
    }
    linked += record.linked;
    primary += record.linked && record.primary_comparison;
    proxy += record.linked && !record.primary_comparison;
  }
  return {{"schema", scaled ? "native-baseline86-scaled-inventory-v1" : "native-baseline86-inventory-v1"},
          {"profile_version", scaled ? native_baseline86_v1::scaled_profile_version : native_baseline86_v1::profile_version},
          {"catalog_labels", methods.size()}, {"linked", linked},
          {"primary_comparison", primary}, {"proxy_or_alias", proxy},
          {"unsupported", methods.size() - linked}, {"methods", std::move(methods)},
          {"claim_boundary",
           "Only linked methods with primary_comparison=true are admitted to the "
           "same-label primary table. Proxy/alias and unsupported rows remain visible."}};
}

double median(std::vector<std::uint64_t> values) {
  if (values.empty()) fail("cannot summarize empty timing sample");
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2) return static_cast<double>(values[middle]);
  return (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}

void add_scaling_stats(J& out, const native_baseline86_v1::ScalingStats& stats) {
  out["scaled_value_count"] = stats.scaled_value_count;
  out["exception_value_count"] = stats.exception_value_count;
  out["scaling_metadata_bytes"] = stats.scaling_metadata_bytes;
  out["raw_exception_bytes"] = stats.raw_exception_bytes;
  const auto count = stats.scaled_value_count + stats.exception_value_count;
  out["exception_fraction"] = count ? J(static_cast<double>(stats.exception_value_count)/count) : J(nullptr);
  out["scaling_adapter_source"] = "cpp/native_baseline86_v1/decimal_scaling.cpp";
}

J receipt(const std::string& label, const Block& block,
          native_baseline86_v1::Measurement result,
          std::optional<std::uint16_t> decimal_places = std::nullopt) {
  const auto& record = native_baseline86_v1::method(label);
  J out{{"status", result.original_numeric_bytes ? "ok" : "N/A"},
        {"method", label}, {"implementation_class", record.implementation_class},
        {"primary_comparison", record.primary_comparison},
        {"configuration", record.configuration}, {"dtype", block.dtype},
        {"row_count", block.row_count},
        {"original_numeric_bytes", result.original_numeric_bytes},
        {"complete_frame_bytes", result.frame.size()},
        {"compression_ns_samples", result.compression_ns},
        {"decompression_ns_samples", result.decompression_ns},
        {"complete_frame_sha256", sha256(result.frame)},
        {"timing_scope",
         "C++ in-process complete NGF1 encode/decode from resident values and missing bitmap; source I/O, model selection, and correctness comparison excluded"},
        {"thread_policy", "caller must bind one process to one CPU; each admitted backend uses its fixed single-thread profile"}};
  if (decimal_places) {
    if (!result.scaling_stats) fail("scaled measurement lacks diagnostic counts");
    add_scaling_stats(out, *result.scaling_stats);
    out["profile_version"] = native_baseline86_v1::scaled_profile_version;
    out["base_configuration"] = record.configuration;
    out["scaling_configuration"] = native_baseline86_v1::scaling_configuration;
    out["configuration"] = std::string(native_baseline86_v1::scaling_configuration) + ";base=" + record.configuration;
    out["decimal_places"] = *decimal_places;
    out["precision_semantics"] = "typed_shortest_roundtrip_decimal_v1";
    out["precision_selection"] = "explicit frozen parameter; whole-column derivation cost must be added by the campaign";
    out["precision_derivation_ns_samples"] = nullptr;
    out["timing_scope"] = "C++ complete NGF1 encode/decode including per-block precision checks, exact decimal conversion, integer codec, raw exceptions, missing bitmap and all framing; frozen whole-column precision derivation and source I/O excluded here";
  }
  const auto compression = median(result.compression_ns);
  const auto decompression = median(result.decompression_ns);
  out["compression_ns"] = compression;
  out["decompression_ns"] = decompression;
  if (result.original_numeric_bytes) {
    const auto denominator = static_cast<double>(result.original_numeric_bytes);
    out["compression_ratio"] = result.frame.size() / denominator;
    out["compression_ns_per_original_numeric_byte"] = compression / denominator;
    out["decompression_ns_per_original_numeric_byte"] = decompression / denominator;
  } else {
    out["compression_ratio"] = nullptr;
    out["compression_ns_per_original_numeric_byte"] = nullptr;
    out["decompression_ns_per_original_numeric_byte"] = nullptr;
  }
  return out;
}

std::uint16_t precision(const char* text) {
  const auto p = number(text, "decimal places");
  if (p > native_baseline86_v1::decimal_scaling::maximum_decimal_places)
    fail("decimal precision exceeds 324");
  return static_cast<std::uint16_t>(p);
}

J execution_context() {
  char host[256]{};
  if (gethostname(host, sizeof(host)-1) != 0) fail("cannot identify timing host");
  J context{{"host", host}, {"cpu_affinity", nullptr}};
#ifdef __linux__
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  if (sched_getaffinity(0, sizeof(cpus), &cpus) != 0) fail("cannot read timing CPU affinity");
  context["cpu_affinity"] = J::array();
  for (int cpu=0; cpu<CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu,&cpus)) context["cpu_affinity"].push_back(cpu);
#endif
  return context;
}

void charge_precision(J& result, const J& column, bool first) {
  const auto encode = result.at("compression_ns_samples").get<std::vector<std::uint64_t>>();
  const auto calibration = column.at("precision_derivation_ns_samples").get<std::vector<std::uint64_t>>();
  if (encode.size()!=calibration.size()) fail("precision and codec timing repeat counts differ");
  std::vector<std::uint64_t> charged(encode.size(),0), total=encode;
  if (first) for (std::size_t i=0;i<encode.size();++i) {
    charged[i]=calibration[i];
    if (charged[i]>std::numeric_limits<std::uint64_t>::max()-total[i]) fail("precision timing addition overflow");
    total[i]+=charged[i];
  }
  result["compression_encode_only_ns_samples"]=encode;
  result["precision_derivation_ns_samples"]=charged;
  result["column_precision_derivation_ns_samples"]=calibration;
  result["precision_charge_once"]=first;
  result["column_key"]=column.at("column_key");
  result["compression_ns_samples"]=total;
  result["compression_ns"]=median(total);
  const auto bytes=result.at("original_numeric_bytes").get<std::uint64_t>();
  result["compression_ns_per_original_numeric_byte"]=bytes ? J(median(total)/bytes):J(nullptr);
  result["precision_selection"]="validated frozen whole-column maximum; measured selection cost charged once per method and column on first successful block";
  result["timing_scope"]="complete per-block scaling/int64-codec/NGF1 encode plus one whole-column precision derivation per method; source I/O and independent correctness audits excluded; decode includes complete inverse and exceptions";
}

Block block_from_args(const char* dtype, const char* rows, const char* values,
                      const char* missing) {
  Block block{dtype, number(rows, "row count"), read(values), read(missing)};
  native_baseline86_v1::validate(block);
  return block;
}

void verify_digest(const Bytes& bytes, const J& row, const char* field) {
  const std::string key = std::string(field) + "_sha256";
  if (row.contains(key) && row.at(key).is_string() &&
      sha256(bytes) != row.at(key).get<std::string>())
    fail(std::string(field) + " SHA-256 mismatch");
}

int benchmark_manifest(int argc, char** argv, bool scaled = false) {
  if (argc < 6)
    fail("benchmark-manifest MANIFEST JSONL WARMUPS REPEATS [--frame-dir DIR] [METHOD ...]");
  const fs::path manifest_path = argv[2];
  if (scaled && fs::file_size(manifest_path)>(256ULL<<20)) fail("scaled manifest exceeds 256 MiB");
  std::ifstream source(manifest_path);
  if (!source) fail("cannot open manifest: " + manifest_path.string());
  const auto manifest_bytes=scaled ? read(manifest_path) : Bytes{};
  const auto manifest_digest=scaled ? sha256(manifest_bytes) : std::string{};
  J manifest = scaled ? J::parse(manifest_bytes) : J::parse(source);
  if (!manifest.contains("blocks") || !manifest.at("blocks").is_array())
    fail("manifest lacks a blocks array");
  const auto warmups64 = number(argv[4], "warmup count");
  const auto repeats64 = number(argv[5], "repeat count");
  if (scaled && (warmups64>100 || repeats64<1 || repeats64>1000)) fail("invalid measurement repeat count");
  const unsigned warmups = static_cast<unsigned>(warmups64);
  const unsigned repeats = static_cast<unsigned>(repeats64);
  if (scaled && fs::exists(argv[3])) fail("scaled output already exists; use a new isolated attempt");
  std::optional<fs::path> frame_dir;
  std::optional<fs::path> frame_pack_path;
  bool compound_frame=false;
  std::optional<fs::path> column_scales_path;
  std::vector<std::string> methods;
  for (int i = 6; i < argc; ++i) {
    if (scaled && std::string_view(argv[i]) == "--compound-frame") {
      if(compound_frame)fail("duplicate compound frame option");compound_frame=true;
    } else if (std::string_view(argv[i]) == "--frame-dir") {
      if (frame_dir || ++i >= argc) fail("--frame-dir requires one unique directory");
      frame_dir = fs::absolute(argv[i]);
    } else if (scaled && std::string_view(argv[i]) == "--column-scales") {
      if (column_scales_path || ++i>=argc) fail("--column-scales requires one unique receipt");
      column_scales_path = fs::absolute(argv[i]);
    } else if (scaled && std::string_view(argv[i]) == "--frame-pack") {
      if(frame_pack_path||++i>=argc)fail("--frame-pack requires one unique custody file");
      frame_pack_path=fs::absolute(argv[i]);
    } else {
      methods.emplace_back(argv[i]);
    }
  }
  if(frame_dir&&frame_pack_path)fail("choose one frame custody format");
  if (methods.empty())
    for (const auto& item : scaled ? native_baseline86_v1::scaling_inventory() : native_baseline86_v1::inventory()) methods.push_back(item.label);
  for (const auto& name : methods) {
    (void)native_baseline86_v1::method(name);
    if (scaled && !native_baseline86_v1::supports_decimal_scaling(name)) fail("method has no decimal scaling adapter: "+name);
  }
  std::map<std::string,J> scales;
  std::set<std::pair<std::string,std::string>> precision_charged;
  std::string scale_receipt_sha;
  std::string producer_cli_sha;
  if (scaled) {
    if (!column_scales_path) fail("benchmark-manifest-scaled requires --column-scales");
    if (fs::file_size(*column_scales_path)>(256ULL<<20)) fail("precision receipt exceeds 256 MiB");
    const auto scale_bytes=read(*column_scales_path);
    const auto scale_receipt=J::parse(scale_bytes);
    if (!scale_receipt.contains("execution_context") || scale_receipt.at("execution_context")!=execution_context())
      fail("precision receipt timing host/CPU affinity differs from benchmark");
    if (scale_receipt.value("timing_qualification","")!="requires independent campaign resource/source/runtime/input qualification before scientific publication")
      fail("precision timing qualification boundary differs");
    auto precision_evidence=scale_receipt;
    precision_evidence.erase("execution_context");
    precision_evidence.erase("timing_qualification");
    const auto verified=native_baseline86_v1::column_scales::validate(manifest_path,precision_evidence,self_binary_sha256());
    if (verified.at("manifest_sha256")!=manifest_digest) fail("manifest changed during precision validation");
    if (verified.at("warmups")!=warmups || verified.at("repeats")!=repeats) fail("precision and codec warmups/repeats differ");
    for (const auto& column:verified.at("columns")) scales.emplace(column.at("column_key").get<std::string>(),column);
    scale_receipt_sha=sha256(scale_bytes);
    producer_cli_sha=verified.at("scanner_cli_sha256").get<std::string>();
  }
  std::optional<native_baseline86_v1::packed_frames::Writer> frame_pack;
  if(compound_frame){
    if(!scaled||!frame_pack_path||methods.size()!=1||manifest.at("blocks").empty())fail("compound framing requires one scaled method and a nonempty whole-column manifest");
    const auto first_key=native_baseline86_v1::column_scales::column_key(manifest,manifest.at("blocks").at(0));
    for(const auto&b:manifest.at("blocks"))if(native_baseline86_v1::column_scales::column_key(manifest,b)!=first_key)fail("compound framing requires one whole logical column");
  }
  if(frame_pack_path)frame_pack.emplace(*frame_pack_path,compound_frame);
  std::ofstream output(argv[3], std::ios::trunc);
  if (!output) fail("cannot open JSONL output");
  for (const auto& row : manifest.at("blocks")) {
    J base{{"schema", "native-baseline86-result-v1"},
           {"profile_version", native_baseline86_v1::profile_version},
           {"manifest", fs::absolute(manifest_path).string()},
           {"block_id", row.value("block_id", "")},
           {"dataset_id", manifest.value("dataset_id", "")}};
    for (const auto* field : {"domain", "role", "split", "entity_id", "column",
                              "source_column", "block_index", "time_column",
                              "time_precision", "byte_order", "valid_count",
                              "missing_count"}) {
      if (row.contains(field)) base[field] = row.at(field);
      else if (manifest.contains(field)) base[field] = manifest.at(field);
    }
    if (row.contains("dtype")) base["dtype"] = row.at("dtype");
    if (row.contains("row_count")) base["row_count"] = row.at("row_count");
    if (row.contains("original_numeric_bytes"))
      base["declared_original_numeric_bytes"] = row.at("original_numeric_bytes");
    auto decorate = [](J& result, const native_baseline86_v1::Record& record) {
      result["method"] = record.label;
      result["linked"] = record.linked;
      result["implementation_class"] = record.implementation_class;
      result["primary_comparison"] = record.primary_comparison;
      result["implementation"] = record.implementation;
      result["source"] = record.source;
      result["configuration"] = record.configuration;
      result["admitted_dtypes"] = record.dtypes;
      result["limitations"] = record.limitations;
      result["timing_scope"] =
          "C++ in-process complete NGF1 encode/decode from resident values and missing bitmap; source I/O, model selection, and correctness comparison excluded";
      result["thread_policy"] =
          "caller must bind one process to one CPU; each admitted backend uses its fixed single-thread profile";
    };
    try {
      const auto values_path = manifest_path.parent_path() /
                               row.at("values_path").get<std::string>();
      const auto missing_path = manifest_path.parent_path() /
                                row.at("missing_bitmap_path").get<std::string>();
      Block block{row.at("dtype").get<std::string>(),
                  row.at("row_count").get<std::uint64_t>(),
                  typed_segments::read(values_path,row,"values"),
                  typed_segments::read(missing_path,row,"missing_bitmap")};
      verify_digest(block.values, row, "values");
      verify_digest(block.missing, row, "missing_bitmap");
      native_baseline86_v1::validate(block);
      if (row.contains("original_numeric_bytes") &&
          row.at("original_numeric_bytes").get<std::uint64_t>() != block.values.size())
        fail("manifest original_numeric_bytes mismatch");
      base["dtype"] = block.dtype;
      base["row_count"] = block.row_count;
      base["original_numeric_bytes"] = block.values.size();
      for (const auto& label : methods) {
        auto result = base;
        const auto& record = native_baseline86_v1::method(label);
        decorate(result, record);
        try {
          if (!record.linked) throw native_baseline86_v1::Unsupported(record.limitations);
          const bool scaled_float=scaled&&(block.dtype=="float32"||block.dtype=="float64");
          if (!scaled_float && std::find(record.dtypes.begin(), record.dtypes.end(), block.dtype) ==
              record.dtypes.end())
            throw native_baseline86_v1::Unsupported("dtype is not admitted");
          std::optional<std::uint16_t> p;
          std::string key;
          if (scaled_float) {
            key=native_baseline86_v1::column_scales::column_key(manifest,row);
            p=scales.at(key).at("decimal_places").get<std::uint16_t>();
            result["admitted_dtypes"]={"float32","float64"};
            result["profile_version"]=native_baseline86_v1::scaled_profile_version;
            result["base_configuration"]=record.configuration;
            result["scaling_configuration"]=native_baseline86_v1::scaling_configuration;
            result["configuration"]=std::string(native_baseline86_v1::scaling_configuration)+";base="+record.configuration;
            result["implementation"]="exact_decimal_scaling("+record.implementation+")";
            result["source"]=record.source+"; cpp/native_baseline86_v1/decimal_scaling.cpp";
            result["decimal_places"]=*p;
            result["column_key"]=key;
          }
          auto measured = p ? native_baseline86_v1::measure_scaled(label,block,*p,warmups,repeats)
                            : native_baseline86_v1::measure(label, block, warmups, repeats);
          std::uint64_t prefix_encode_ns=0,prefix_decode_ns=0;
          if(compound_frame){
            const auto begin=std::chrono::steady_clock::now();std::array<std::uint8_t,8> prefix{};
            const auto n=static_cast<std::uint64_t>(measured.frame.size());
            for(unsigned i=0;i<8;++i)prefix[i]=static_cast<std::uint8_t>(n>>(8*i));
            const auto encoded=std::chrono::steady_clock::now();
            const auto decode_begin=std::chrono::steady_clock::now();std::uint64_t parsed=0;
            for(unsigned i=0;i<8;++i)parsed|=std::uint64_t(prefix[i])<<(8*i);
            const auto decoded=std::chrono::steady_clock::now();if(parsed!=n)fail("compound length-prefix self-check failed");
            prefix_encode_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(encoded-begin).count();
            prefix_decode_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(decoded-decode_begin).count();
            result["ngf1_compression_ns_samples"]=measured.compression_ns;result["ngf1_decompression_ns_samples"]=measured.decompression_ns;
            for(auto&ns:measured.compression_ns){if(ns>std::numeric_limits<std::uint64_t>::max()-prefix_encode_ns)fail("compound encode time overflow");ns+=prefix_encode_ns;}
            for(auto&ns:measured.decompression_ns){if(ns>std::numeric_limits<std::uint64_t>::max()-prefix_decode_ns)fail("compound decode time overflow");ns+=prefix_decode_ns;}
          }
          const auto frame_digest = sha256(measured.frame);
          if (frame_dir) {
            result["complete_frame_relative_path"] =
                store_complete_frame(*frame_dir, measured.frame, frame_digest);
            result["complete_frame_storage"] = "content_addressed_ngf1";
          }
          if(frame_pack){
            const auto offset=frame_pack->append(measured.frame);
            result["complete_frame_storage"]=compound_frame?"P172B861_scientific_compound_v1":"length_prefixed_ngf1_custody_pack_v1";
            result["complete_frame_pack_record_offset"]=offset;
            result["complete_frame_pack_length"]=measured.frame.size();
          }
          const auto ngf1_bytes=measured.frame.size();
          result.update(receipt(label, block, std::move(measured),p));
          if(compound_frame){
            const bool first=result.at("complete_frame_pack_record_offset")==16;
            const std::uint64_t scientific_bytes=ngf1_bytes+8+(first?16:0);
            result["complete_ngf1_frame_bytes"]=ngf1_bytes;
            result["compound_length_prefix_bytes"]=8;result["compound_header_bytes_charged"]=first?16:0;
            result["compound_header_charge_once"]=first;result["compound_frame_schema"]="P172B861";
            result["compound_prefix_encode_ns"]=prefix_encode_ns;result["compound_prefix_decode_ns"]=prefix_decode_ns;
            result["complete_frame_bytes"]=scientific_bytes;
            result["complete_frame_sha256_semantics"]="exact child NGF1; full compound SHA is bound by cold pack receipt";
            if(!block.values.empty())result["compression_ratio"]=static_cast<double>(scientific_bytes)/block.values.size();
          }
          if (p) {
            const bool first=!block.values.empty()&&!precision_charged.contains({label,key});
            charge_precision(result,scales.at(key),first);
            if (first) precision_charged.emplace(label,key);
            if(compound_frame)result["timing_scope"]="resident NGF1 decimal-scaled encode/decode plus in-memory compound length-prefix construction/parsing; whole-column precision derivation charged once; source and compound file I/O and fixed global header I/O excluded";
          }
        } catch (const native_baseline86_v1::Unsupported& error) {
          result["status"] = "unsupported";
          result["reason"] = error.what();
        } catch (const std::exception& error) {
          result["status"] = "failed";
          result["reason"] = error.what();
        }
        if (scaled) {
          result["campaign_profile_version"]=native_baseline86_v1::scaled_profile_version;
          result["column_scales_receipt_sha256"]=scale_receipt_sha;
          result["producer_cli_sha256"]=producer_cli_sha;
          result["manifest_sha256"]=manifest_digest;
          result["values_sha256"]=row.at("values_sha256");
          result["missing_bitmap_sha256"]=row.at("missing_bitmap_sha256");
          result["execution_context"]=execution_context();
          result["timing_qualification"]="requires independent campaign resource/source/runtime/input qualification before scientific publication";
        }
        output << result.dump() << '\n';
        if(!frame_pack)output.flush();
      }
    } catch (const std::exception& error) {
      base["original_numeric_bytes"] = nullptr;
      for (const auto& label : methods) {
        auto result = base;
        decorate(result, native_baseline86_v1::method(label));
        result["status"] = "invalid_input";
        result["reason"] = error.what();
        output << result.dump() << '\n';
        output.flush();
      }
    }
  }
  if(frame_pack)frame_pack->finish();
  if(scaled){output.flush();if(!output)fail("cannot persist complete scaled JSONL");}
  return 0;
}

int verify_scaled_pack(int argc,char**argv){
  if(argc<8)fail("verify-scaled-pack MANIFEST JSONL PACK SCALES RECEIPT METHOD [METHOD ...]");
  const fs::path manifest_path=argv[2],results_path=argv[3],pack_path=argv[4],scales_path=argv[5],receipt_path=argv[6];
  if(fs::exists(receipt_path))fail("cold verification receipt already exists");
  if(fs::file_size(manifest_path)>(256ULL<<20)||fs::file_size(scales_path)>(256ULL<<20)||fs::file_size(results_path)>(512ULL<<20))fail("cold verification metadata exceeds bound");
  const auto manifest_bytes=read(manifest_path),scale_bytes=read(scales_path);
  const auto manifest=J::parse(manifest_bytes),scale_receipt=J::parse(scale_bytes);
  const auto binary_sha=self_binary_sha256(),manifest_sha=sha256(manifest_bytes),scale_sha=sha256(scale_bytes),results_sha=file_sha256(results_path),pack_sha=file_sha256(pack_path);
  auto evidence=scale_receipt;evidence.erase("execution_context");evidence.erase("timing_qualification");
  // The decoder receives only each stored frame. Validate the precision receipt
  // against source payloads after all frame decodes, not before the first one.
  std::map<std::string,J>columns,blocks;
  for(const auto&column:evidence.at("columns"))if(!columns.emplace(column.at("column_key").get<std::string>(),column).second)fail("duplicate cold precision column");
  for(const auto&block:manifest.at("blocks")){
    if(!blocks.emplace(block.at("block_id").get<std::string>(),block).second)fail("duplicate cold manifest block");
  }
  std::set<std::string>methods;
  for(int i=7;i<argc;++i){if(!native_baseline86_v1::supports_decimal_scaling(argv[i])||!methods.emplace(argv[i]).second)fail("invalid/duplicate cold method");}
  std::set<std::pair<std::string,std::string>>seen,charged;
  native_baseline86_v1::packed_frames::Reader pack(pack_path);
  std::ifstream input(results_path);if(!input)fail("cannot open cold results");
  std::uint64_t rows=0,frame_bytes=0,scientific_frame_bytes=0,numeric=0,not_applicable=0,unsupported=0,original_bytes=0;
  std::string line;
  while(std::getline(input,line)){
    if(line.empty()||line.size()>(8ULL<<20))fail("invalid cold JSONL record size");
    const auto row=J::parse(line);const auto id=row.at("block_id").get<std::string>(),method=row.at("method").get<std::string>();
    if(!blocks.contains(id)||!methods.contains(method)||!seen.emplace(id,method).second)fail("cold result key is extra/duplicate");
    const auto&source=blocks.at(id);const auto dtype=source.at("dtype").get<std::string>();
    const bool is_float=dtype=="float32"||dtype=="float64";
    const auto key=is_float?native_baseline86_v1::column_scales::column_key(manifest,source):std::string{};
    const auto p=is_float?columns.at(key).at("decimal_places").get<std::uint16_t>():0;
    if(row.at("dataset_id")!=manifest.at("dataset_id")||row.at("dtype")!=dtype||row.at("row_count")!=source.at("row_count")||row.at("manifest_sha256")!=manifest_sha||row.at("column_scales_receipt_sha256")!=scale_sha||row.at("producer_cli_sha256")!=binary_sha||row.at("execution_context")!=scale_receipt.at("execution_context")||row.at("values_sha256")!=source.at("values_sha256")||row.at("missing_bitmap_sha256")!=source.at("missing_bitmap_sha256"))fail("cold result identity/source mismatch");
    if(is_float&&(row.at("decimal_places")!=p||row.at("profile_version")!=native_baseline86_v1::scaled_profile_version))fail("cold result precision/profile mismatch");
    auto load_source=[&](){
      Block b{dtype,source.at("row_count").get<std::uint64_t>(),
        typed_segments::read(manifest_path.parent_path()/source.at("values_path").get<std::string>(),source,"values"),
        typed_segments::read(manifest_path.parent_path()/source.at("missing_bitmap_path").get<std::string>(),source,"missing_bitmap")};
      verify_digest(b.values,source,"values");verify_digest(b.missing,source,"missing_bitmap");native_baseline86_v1::validate(b);
      if(b.values.size()!=source.at("original_numeric_bytes").get<std::uint64_t>())fail("cold source byte count mismatch");return b;
    };
    const auto status=row.at("status").get<std::string>();
    if(status=="unsupported"){
      auto b=load_source();bool reproduced=false;
      try{if(is_float)(void)native_baseline86_v1::encode_scaled(method,b,p);else(void)native_baseline86_v1::encode(method,b);}
      catch(const native_baseline86_v1::Unsupported&error){reproduced=row.at("reason")==error.what();}
      if(!reproduced||row.contains("complete_frame_pack_record_offset")||row.contains("complete_frame_bytes")||row.contains("compression_ratio")||row.at("original_numeric_bytes")!=b.values.size())fail("unsupported boundary not independently reproduced");++unsupported;++rows;continue;
    }
    if(status!="ok"&&status!="N/A")fail("failed/nonterminal result cannot pass cold verification");
    const auto length=row.at("complete_frame_pack_length").get<std::uint64_t>();
    const auto scientific_bytes=row.at("complete_frame_bytes").get<std::uint64_t>();
    if(pack.is_compound()){
      const auto first=frame_bytes==0;const auto expected_bytes=length+8+(first?16:0);
      if(methods.size()!=1||columns.size()!=1||row.at("complete_frame_storage")!="P172B861_scientific_compound_v1"||row.at("compound_frame_schema")!="P172B861"||scientific_bytes!=expected_bytes||row.at("complete_ngf1_frame_bytes")!=length||row.at("compound_length_prefix_bytes")!=8||row.at("compound_header_bytes_charged")!=(first?16:0)||row.at("compound_header_charge_once")!=first)fail("cold scientific compound boundary mismatch");
    }else if(row.at("complete_frame_storage")!="length_prefixed_ngf1_custody_pack_v1"||scientific_bytes!=length)fail("cold custody format mismatch");
    auto frame=pack.next(row.at("complete_frame_pack_record_offset").get<std::uint64_t>(),length);
    if(row.at("complete_frame_sha256")!=sha256(frame))fail("cold complete frame SHA mismatch");
    auto restored=native_baseline86_v1::decode(frame); // No source values are read before the decoder.
    const auto expected=load_source();if(restored!=expected)fail("cold original bits/NULL bitmap differ");
    if(row.at("original_numeric_bytes")!=expected.values.size()||(expected.values.empty()!=(status=="N/A")))fail("cold denominator/status mismatch");
    if(is_float){
      const auto info=native_baseline86_v1::inspect_scaled(frame);
      if(info.method!=method||info.dtype!=dtype||info.row_count!=expected.row_count||info.decimal_places!=p)fail("actual frame method/precision differs from result");
      J actual;add_scaling_stats(actual,info.stats);
      for(const auto&[name,value]:actual.items())if(row.at(name)!=value)fail("cold scaling diagnostics differ");
      const auto calibration=columns.at(key).at("precision_derivation_ns_samples").get<std::vector<std::uint64_t>>();
      auto samples=[&](const char*name){const auto&array=row.at(name);if(!array.is_array()||array.size()!=calibration.size())fail("cold timing sample count differs");for(const auto&x:array)if(!x.is_number_unsigned())fail("cold timing sample is not unsigned ns");return array.get<std::vector<std::uint64_t>>();};
      const auto encode=samples("compression_encode_only_ns_samples"),cost=samples("precision_derivation_ns_samples"),total=samples("compression_ns_samples"),decompression=samples("decompression_ns_samples");
      if(pack.is_compound()){
        const auto ne=samples("ngf1_compression_ns_samples"),nd=samples("ngf1_decompression_ns_samples");
        if(!row.at("compound_prefix_encode_ns").is_number_unsigned()||!row.at("compound_prefix_decode_ns").is_number_unsigned())fail("compound prefix timing must be unsigned");
        const auto pe=row.at("compound_prefix_encode_ns").get<std::uint64_t>(),pd=row.at("compound_prefix_decode_ns").get<std::uint64_t>();
        for(std::size_t i=0;i<ne.size();++i)if(ne[i]>std::numeric_limits<std::uint64_t>::max()-pe||nd[i]>std::numeric_limits<std::uint64_t>::max()-pd||encode[i]!=ne[i]+pe||decompression[i]!=nd[i]+pd)fail("compound prefix timing arithmetic differs");
      }
      const bool first=!expected.values.empty()&&!charged.contains({method,key});
      if(row.at("precision_charge_once")!=first||row.at("column_precision_derivation_ns_samples")!=calibration||encode.size()!=calibration.size()||cost.size()!=encode.size()||total.size()!=encode.size())fail("cold precision charge shape differs");
      for(std::size_t i=0;i<encode.size();++i){const auto extra=first?calibration[i]:0;if(cost[i]!=extra||extra>std::numeric_limits<std::uint64_t>::max()-encode[i]||total[i]!=encode[i]+extra)fail("cold precision charge arithmetic differs");}
      if(row.at("compression_ns")!=median(total)||row.at("decompression_ns")!=median(decompression))fail("cold timing median differs");
      const auto denominator=expected.values.size();
      if(denominator){if(row.at("compression_ns_per_original_numeric_byte")!=median(total)/denominator||row.at("decompression_ns_per_original_numeric_byte")!=median(decompression)/denominator)fail("cold normalized timing differs");}
      else if(!row.at("compression_ns_per_original_numeric_byte").is_null()||!row.at("decompression_ns_per_original_numeric_byte").is_null()||!row.at("compression_ratio").is_null())fail("cold N/A metrics are not null");
      if(first)charged.emplace(method,key);
    }else if(native_baseline86_v1::encode(method,expected)!=frame)fail("cold integer frame profile differs");
    if(expected.values.size()&&row.at("compression_ratio")!=static_cast<double>(scientific_bytes)/expected.values.size())fail("cold full-frame ratio differs");
    original_bytes+=expected.values.size();frame_bytes+=frame.size();scientific_frame_bytes+=scientific_bytes;++rows;if(status=="ok")++numeric;else++not_applicable;
  }
  if(!input.eof()||seen.size()!=blocks.size()*methods.size())fail("cold result union incomplete");
  pack.finish();if(pack.is_compound()&&scientific_frame_bytes&&scientific_frame_bytes!=fs::file_size(pack_path))fail("scientific compound byte union differs from actual pack");if(results_sha!=file_sha256(results_path)||manifest_sha!=file_sha256(manifest_path)||scale_sha!=file_sha256(scales_path)||pack_sha!=file_sha256(pack_path))fail("cold artifact changed while verifying");
  const auto verified=native_baseline86_v1::column_scales::validate(manifest_path,evidence,binary_sha);
  if(verified.at("manifest_sha256")!=manifest_sha)fail("cold verification manifest changed");
  if(results_sha!=file_sha256(results_path)||manifest_sha!=file_sha256(manifest_path)||scale_sha!=file_sha256(scales_path)||pack_sha!=file_sha256(pack_path))fail("cold artifact changed during final precision audit");
  J result{{"schema","scaled-packed-cold-verification-v1"},{"status","verified"},{"dataset_id",manifest.at("dataset_id")},{"manifest_sha256",manifest_sha},{"results_sha256",results_sha},{"pack_sha256",pack_sha},{"precision_receipt_sha256",scale_sha},{"cli_sha256",binary_sha},{"methods",methods},{"expected_rows",blocks.size()*methods.size()},{"verified_rows",rows},{"numeric_rows",numeric},{"not_applicable_rows",not_applicable},{"unsupported_rows",unsupported},{"numeric_original_bytes",original_bytes},{"complete_ngf1_bytes",frame_bytes},{"custody_pack_bytes",fs::file_size(pack_path)},{"scientific_compound_frame_bytes",pack.is_compound()?J(fs::file_size(pack_path)):J(nullptr)},{"compound_frame_schema",pack.is_compound()?J("P172B861"):J(nullptr)},{"source_cache_used_by_decoder",false}};
  std::ofstream output(receipt_path);if(!output)fail("cannot create cold receipt");output<<result.dump(2)<<'\n';output.close();if(!output)fail("cannot persist cold receipt");std::cout<<result.dump()<<'\n';return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) fail("expected inventory, encode, decode, measure, or benchmark-manifest");
  const std::string command = argv[1];
  if (command == "inventory" || command == "inventory-scaled") {
    if (argc != 2) fail("inventory takes no arguments");
    std::cout << inventory_json(command == "inventory-scaled").dump(2) << '\n';
  } else if (command == "encode-scaled") {
    if (argc != 9) fail("encode-scaled METHOD DTYPE ROWS VALUES MISSING P FRAME");
    const auto block = block_from_args(argv[3], argv[4], argv[5], argv[6]);
    const auto p = precision(argv[7]);
    native_baseline86_v1::ScalingStats stats;
    const auto frame = native_baseline86_v1::encode_scaled(argv[2], block, p, &stats);
    write(argv[8], frame);
    const auto& record = native_baseline86_v1::method(argv[2]);
    J out{{"status", block.values.empty() ? "N/A" : "ok"}, {"method", argv[2]},
                   {"dtype", block.dtype}, {"row_count", block.row_count},
                   {"profile_version", native_baseline86_v1::scaled_profile_version},
                   {"base_configuration", record.configuration},
                   {"scaling_configuration", native_baseline86_v1::scaling_configuration},
                   {"decimal_places", p}, {"original_numeric_bytes", block.values.size()},
                   {"complete_frame_bytes", frame.size()},
                   {"complete_frame_sha256", sha256(frame)}};
    add_scaling_stats(out, stats);
    std::cout << out.dump() << '\n';
  } else if (command == "encode") {
    if (argc != 8) fail("encode METHOD DTYPE ROWS VALUES MISSING FRAME");
    const auto block = block_from_args(argv[3], argv[4], argv[5], argv[6]);
    const auto frame = native_baseline86_v1::encode(argv[2], block);
    write(argv[7], frame);
    std::cout << J{{"status", "ok"}, {"method", argv[2]},
                   {"original_numeric_bytes", block.values.size()},
                   {"complete_frame_bytes", frame.size()},
                   {"complete_frame_sha256", sha256(frame)}}.dump() << '\n';
  } else if (command == "decode") {
    if (argc != 5) fail("decode FRAME VALUES_OUT MISSING_OUT");
    const auto frame = read(argv[2]);
    const auto block = native_baseline86_v1::decode(frame);
    write(argv[3], block.values);
    write(argv[4], block.missing);
    std::cout << J{{"status", "ok"}, {"dtype", block.dtype},
                   {"row_count", block.row_count},
                   {"original_numeric_bytes", block.values.size()},
                   {"values_sha256", sha256(block.values)},
                   {"missing_bitmap_sha256", sha256(block.missing)}}.dump() << '\n';
  } else if (command == "measure") {
    if (argc != 10)
      fail("measure METHOD DTYPE ROWS VALUES MISSING WARMUPS REPEATS FRAME");
    const auto block = block_from_args(argv[3], argv[4], argv[5], argv[6]);
    auto measured = native_baseline86_v1::measure(
        argv[2], block, number(argv[7], "warmup count"),
        number(argv[8], "repeat count"));
    write(argv[9], measured.frame);
    std::cout << receipt(argv[2], block, std::move(measured)).dump() << '\n';
  } else if (command == "measure-scaled") {
    if (argc != 11) fail("measure-scaled METHOD DTYPE ROWS VALUES MISSING P WARMUPS REPEATS FRAME");
    const auto block = block_from_args(argv[3], argv[4], argv[5], argv[6]);
    const auto p = precision(argv[7]);
    const auto warmups = number(argv[8], "warmup count");
    const auto repeats = number(argv[9], "repeat count");
    if (warmups > 100 || repeats < 1 || repeats > 1000) fail("invalid measurement repeat count");
    auto measured = native_baseline86_v1::measure_scaled(argv[2], block, p, warmups, repeats);
    write(argv[10], measured.frame);
    std::cout << receipt(argv[2], block, std::move(measured), p).dump() << '\n';
  } else if (command == "scan-column-scales") {
    if (argc!=4 && argc!=6) fail("scan-column-scales MANIFEST OUT [WARMUPS REPEATS]");
    if (fs::exists(argv[3])) fail("precision receipt already exists; use a new isolated attempt");
    const auto warmups=argc==6?number(argv[4],"warmup count"):1;
    const auto repeats=argc==6?number(argv[5],"repeat count"):3;
    if (warmups>100 || repeats<1 || repeats>1000) fail("invalid measurement repeat count");
    auto out=native_baseline86_v1::column_scales::scan(argv[2],warmups,repeats,self_binary_sha256());
    out["execution_context"]=execution_context();
    out["timing_qualification"]="requires independent campaign resource/source/runtime/input qualification before scientific publication";
    std::ofstream file(argv[3]);
    if (!file) fail("cannot write precision receipt");
    file<<out.dump(2)<<'\n';file.close();
    if (!file) fail("cannot persist precision receipt");
    std::cout<<out.dump()<<'\n';
  } else if (command == "benchmark-manifest") {
    return benchmark_manifest(argc, argv);
  } else if (command == "benchmark-manifest-scaled") {
    return benchmark_manifest(argc, argv, true);
  } else if (command == "verify-scaled-pack") {
    return verify_scaled_pack(argc,argv);
  } else {
    fail("unknown command: " + command);
  }
  return 0;
} catch (const native_baseline86_v1::Unsupported& error) {
  std::cerr << "unsupported: " << error.what() << '\n';
  return 3;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}
