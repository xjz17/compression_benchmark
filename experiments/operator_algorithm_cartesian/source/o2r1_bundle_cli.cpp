// Isolated production producer for one (column, operator) physical bundle.
//
// This translation unit deliberately reuses the exact request loader, input
// adapter, catalog-row decoder and special bridges of operator102_measure_cli.
// Renaming the included main keeps this additive: no accepted executable or
// result is modified.
#define main operator102_measure_cli_embedded_main
#include "../../../cpp/native_graph_v1/operator102_measure_cli.cpp"
#undef main

#include "../../../cpp/src/deflate_container_stages.hpp"
#include "../../../cpp/src/general_codec_reference_stages.hpp"
#include "../../../cpp/src/lz4_exact_stages.hpp"
#include "../../../cpp/src/native_lz_action_stages.hpp"
#include "../../../cpp/src/parameterized_deflate.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace reger_legacy_cpp {
Bytes core_inverse(const Bytes&);
Bytes frame_inverse(const Bytes&);
}
namespace blosc2_binding_cpp {
Bytes core_inverse(const Bytes&);
}

namespace joint_o2r1 {
namespace fs2 = std::filesystem;

constexpr std::size_t kWarmups = 1;
constexpr std::size_t kRepeats = 3;

void write_bytes(const fs2::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create output: " + path.string());
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  output.flush();
  if (!output) throw std::runtime_error("cannot finish output: " + path.string());
}

void write_json(const fs2::path& path, const J& value) {
  const auto encoded = value.dump(2) + "\n";
  write_bytes(path, std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(encoded.data()), encoded.size()));
}

Bytes as_bytes(const std::string& value) {
  return Bytes(value.begin(), value.end());
}

std::string canonical_sha(const J& value) {
  return sha256(as_bytes(value.dump()));
}

J source_identity(const std::vector<Block>& blocks, const J& manifest) {
  // Paths are transport details and differ between mirrors.  Bind the
  // manifest bytes and semantic identity, never the absolute host path.
  J manifest_binding = manifest;
  manifest_binding.erase("path");
  J rows = J::array();
  for (const auto& block : blocks) {
    J auxiliaries = J::array();
    for (const auto& auxiliary : block.auxiliaries)
      auxiliaries.push_back({{"bytes", auxiliary.size()},
                             {"sha256", sha256(auxiliary)}});
    rows.push_back({{"block_id", block.block_id},
                    {"dtype", block.dtype},
                    {"width", block.width},
                    {"valid_count", block.valid_count},
                    {"values_bytes", block.values.size()},
                    {"values_sha256", sha256(block.values)},
                    {"auxiliaries", std::move(auxiliaries)}});
  }
  return {{"schema", "source-column-physical-identity-v1"},
          {"manifest", std::move(manifest_binding)}, {"blocks", std::move(rows)}};
}

void add_samples(std::array<std::uint64_t, kRepeats>& total,
                 const std::vector<std::uint64_t>& sample,
                 const char* label) {
  if (sample.size() != kRepeats)
    throw std::runtime_error(std::string(label) + " sample count mismatch");
  for (std::size_t index = 0; index < kRepeats; ++index) total[index] += sample[index];
}

J samples_json(const std::array<std::uint64_t, kRepeats>& samples) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  return {{"samples_ns", samples}, {"median_ns", sorted[1]}};
}

Bytes inverse_special_from_parsed(std::string_view id,
                                  const O2R1State& parsed) {
  require(parsed.outputs.size() == 1,
          std::string(id) + " cold O2R1 output count mismatch");
  const auto& after = parsed.outputs.front().data;
  if (id == "A065") {
    require(parsed.inverse_metadata.size() == 1,
            "A065 cold metadata count mismatch");
    return native_lz_action::inverse(
        "op.match_window_model.native_lz.window32768", after);
  }
  if (id == "A057") {
    require(parsed.inverse_metadata.size() == 1,
            "A057 cold metadata count mismatch");
    auto before = native_lz_action::inverse(
        "op.lz_match.native_lz.aligned8.min5.max175", after);
    return native_lz_action::inverse(
        "op.match_window_model.native_lz.window32768", before);
  }
  if (id == "A086") {
    require(parsed.inverse_metadata.size() == 1,
            "A086 cold metadata count mismatch");
    return reference_actions::general::frame_inverse(after, false);
  }
  if (id == "A094") {
    require(parsed.inverse_metadata.size() == 1,
            "A094 cold metadata count mismatch");
    return reger_legacy_cpp::core_inverse(after);
  }
  if (id == "A096") {
    require(parsed.inverse_metadata.size() == 1,
            "A096 cold metadata count mismatch");
    return exact_actions::lz4::lz_match_inverse(
        exact_actions::lz4::token_format_inverse(
            exact_actions::lz4::integer_code_inverse(
                exact_actions::lz4::container_inverse(after))));
  }
  if (id == "A097") {
    require(parsed.inverse_metadata.size() == 1,
            "A097 cold metadata count mismatch");
    auto wire = exact_actions::deflate_container::gzip_forward(
        exact_actions::deflate_container::gzip_inverse(after));
    auto raw = exact_actions::parameterized_deflate::decode(
        wire, exact_actions::parameterized_deflate::Wrapper::gzip);
    auto prefix = reference_actions::general::huffman_forward(
        reference_actions::general::lz_forward(raw, 32768, 3, 258, true));
    return reference_actions::general::lz_inverse(
        reference_actions::general::huffman_inverse(prefix));
  }
  if (id == "A100") {
    require(parsed.inverse_metadata.size() == 1,
            "A100 cold metadata count mismatch");
    return reger_legacy_cpp::core_inverse(
        reger_legacy_cpp::frame_inverse(after));
  }
  if (id == "A102") {
    require(parsed.inverse_metadata.size() == 1,
            "A102 cold metadata count mismatch");
    return blosc2_binding_cpp::core_inverse(after);
  }
  if (id == "A098" || id == "A099") {
    require(parsed.inverse_metadata.size() == 5,
            std::string(id) + " cold metadata count mismatch");
    const bool alp = id == "A099";
    const std::vector<StageRequest> prefix = alp
        ? std::vector<StageRequest>{
              {"op.scaling.alp_legacy_fe_exceptions.cpp_v1", "float64", {}},
              {"op.block_transform.int64_lo_hi32.cpp_v1", "", {}},
              {"op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1", "", {}},
              {"op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1", "", {}}}
        : std::vector<StageRequest>{
              {"op.scaling.legacy_exact_decimal.cpp_v1", "int64", {{"max_point", "0"}}},
              {"op.block_transform.int64_lo_hi32.cpp_v1", "", {}},
              {"op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1", "", {}},
              {"op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1", "", {}}};
    StageRequest target{
        alp ? "op.container_frame.alp2_legacy.cpp_v1"
            : "op.container_frame.legacy_numeric_limbs.cpp_v1", "", {}};
    NativeStageDispatcher native;
    auto current = native.inverse(target, parsed.outputs,
                                  parsed.inverse_metadata.back());
    require(current.size() == 1, std::string(id) + " cold target inverse split");
    for (std::size_t reverse = prefix.size(); reverse > 0; --reverse) {
      current = native.inverse(prefix[reverse - 1], current,
                               parsed.inverse_metadata[reverse - 1]);
      require(current.size() == 1,
              std::string(id) + " cold prefix inverse split");
    }
    return current.front().data;
  }
  throw std::runtime_error("unsupported cold special inverse: " + std::string(id));
}

J produce(const J& request) {
  require(request.value("schema", std::string{}) ==
              "operator-algorithm-o2r1-column-request-v1",
          "wrong bundle request schema");
  require(request.contains("implementation") &&
              request.at("implementation").is_object(),
          "request requires one resolved implementation");
  const auto& implementation = request.at("implementation");
  const auto implementation_id =
      implementation.at("implementation_id").get<std::string>();
  J manifest_identity;
  const auto blocks = load_blocks(request, manifest_identity);
  const auto physical_identity = source_identity(blocks, manifest_identity);
  const auto source_column_sha = canonical_sha(physical_identity);

  J base = {{"schema", "operator-algorithm-o2r1-column-response-v1"},
            {"request_id", request.value("request_id", std::string{})},
            {"dataset_id", request.at("dataset_id")},
            {"column_id", request.at("column_id")},
            {"operator_id", implementation_id},
            {"map_key", implementation.value("map_key", std::string{})},
            {"manifest", manifest_identity},
            {"source_column_sha256", source_column_sha}};
  if (request.contains("provenance")) base["provenance"] = request.at("provenance");
  if (!implementation.value("applicable", true)) {
    base["status"] = "not_applicable";
    base["applicable"] = false;
    base["strict"] = false;
    base["fallback_used"] = false;
    base["bit_exact"] = false;
    base["ranking_eligible"] = false;
    base["reason"] = implementation.value(
        "reason", "mapper-declared not applicable");
    return base;
  }

  const auto output = fs2::absolute(request.at("output_dir").get<std::string>());
  const auto temporary = fs2::path(output.string() + ".tmp." +
                                   std::to_string(::getpid()));
  require(!fs2::exists(output) && !fs2::exists(temporary),
          "bundle output already exists");
  fs2::create_directories(temporary / "blocks");

  Operator102Dispatcher dispatcher;
  const auto present = decode_pipeline(implementation, "present_pipeline");
  require(!present.empty(), "present pipeline is empty");
  const auto focal = implementation.value("focal_step_index", present.size() - 1);
  require(focal < present.size(), "focal step index exceeds present pipeline");
  std::array<std::uint64_t, kRepeats> kernel_encode{}, kernel_decode{};
  std::array<std::uint64_t, kRepeats> full_encode{}, full_decode{};
  J block_receipts = J::array();
  bool any_fallback = false;
  std::uint64_t valid_count = 0;
  std::uint64_t source_numeric_bytes = 0;
  std::uint64_t o2r1_bytes = 0;

  try {
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      const auto& block = blocks[index];
      const auto kind = decode_kind(implementation.value(
          "initial_kind", block.dtype == "bytes" ? std::string("serial")
                                                   : std::string("numeric")));
      const auto width = implementation.value(
          "initial_width", kind == Kind::Serial ? 1U : block.width);
      std::string adapter_operation;
      auto adapted = adapt_values(block, width, adapter_operation);
      Stream input{kind, width, adapted};
      Bytes adapter_packet;
      const Bytes* adapter = nullptr;
      if (implementation.value("charge_input_adapter_metadata", false)) {
        adapter_packet = adapter_metadata(block, implementation, width,
                                          adapter_operation);
        adapter = &adapter_packet;
      }

      PipelineMeasurement measured;
      PipelineTiming timed;
      bool special = false;
#if defined(OPERATOR102_ENABLE_SPECIAL_BRIDGES)
      special = native_graph_v1::operator102::supports_special_bridge(
          implementation_id);
      if (special) {
        std::string ignored_contract;
        measured = measure_special(implementation_id, input.data,
                                   ignored_contract).first;
        const auto sample = native_graph_v1::operator102::benchmark_special_bridge(
            implementation_id, input.data, block.auxiliaries, adapter,
            kWarmups, kRepeats);
        timed.bitexact = sample.bitexact;
        timed.encode_ns = sample.encode_ns;
        timed.decode_ns = sample.decode_ns;
        timed.full_encode_ns = sample.full_encode_ns;
        timed.full_decode_ns = sample.full_decode_ns;
      } else
#endif
      {
        measured = measure_pipeline(dispatcher, input, present, focal);
        timed = benchmark_pipeline(dispatcher, input, present, block.auxiliaries,
                                  adapter, kWarmups, kRepeats);
      }
      require(measured.bitexact && timed.bitexact,
              "operator forward/inverse was not bit-exact");
      bool block_fallback = false;
      std::string fallback_reason;
      if (implementation.contains("domain_fallback_contract")) {
        require(focal < measured.inverse_metadata.size(),
                "domain wrapper metadata missing");
        const auto info = inspect_exact_raw_fallback_metadata(
            measured.inverse_metadata[focal]);
        require(info.wrapper, "domain fallback contract did not emit selector");
        block_fallback = info.used_raw_fallback;
        fallback_reason = info.reason;
        any_fallback = any_fallback || block_fallback;
      }

      auto bundle = pack_o2r1(measured.final_outputs,
                              measured.inverse_metadata,
                              block.auxiliaries, adapter);
      const auto block_root = temporary / "blocks" /
          (std::string(8 - std::to_string(index).size(), '0') +
           std::to_string(index));
      fs2::create_directories(block_root);
      const auto o2r1_path = block_root / "representation.o2r1";
      write_bytes(o2r1_path, bundle);
      const auto cold_bundle = read_file(o2r1_path);
      require(cold_bundle == bundle, "cold O2R1 file read changed bytes");
      const auto parsed = parse_o2r1(cold_bundle);
      require(parsed.outputs == measured.final_outputs &&
                  parsed.inverse_metadata == measured.inverse_metadata &&
                  parsed.auxiliaries == block.auxiliaries &&
                  parsed.adapter.has_value() == (adapter != nullptr) &&
                  (!adapter || parsed.adapter.value() == *adapter),
              "serialized O2R1 state differs after parse");
      if (special) {
        const auto restored = inverse_special_from_parsed(implementation_id, parsed);
        require(restored == input.data,
                "cold parsed special bridge inverse changed source bytes");
      } else {
        auto current = parsed.outputs;
        require(parsed.inverse_metadata.size() == present.size(),
                "parsed inverse metadata count differs from pipeline");
        for (std::size_t reverse = present.size(); reverse > 0; --reverse)
          current = dispatcher.inverse(present[reverse - 1].request, current,
                                       parsed.inverse_metadata[reverse - 1]);
        require(current.size() == 1 && current.front() == input,
                "parsed O2R1 inverse changed adapted source bytes");
      }

      J inverse_receipt = {
          {"schema", "operator-o2r1-inverse-receipt-v1"},
          {"status", "passed"}, {"bit_exact", true},
          {"o2r1_reloaded_from_file", true},
          {"inverse_executed_from_parsed_o2r1", true},
          {"source_values_and_auxiliaries_bound", true},
          {"applicable", true}, {"strict", !block_fallback},
          {"fallback_used", block_fallback},
          {"fallback_reason", block_fallback ? J(fallback_reason) : J(nullptr)},
          {"operator_id", implementation_id},
          {"source_column_sha256", source_column_sha},
          {"block_index", index}, {"block_id", block.block_id},
          {"o2r1_sha256", sha256(bundle)}, {"o2r1_bytes", bundle.size()},
          {"source_values_sha256", sha256(block.values)},
          {"source_values_bytes", block.values.size()},
          {"source_auxiliary_count", block.auxiliaries.size()}};
      write_json(block_root / "OPERATOR_INVERSE.json", inverse_receipt);
      block_receipts.push_back({
          {"block_index", index}, {"block_id", block.block_id},
          {"o2r1_path", fs2::relative(o2r1_path, temporary).string()},
          {"operator_inverse_receipt_path",
           fs2::relative(block_root / "OPERATOR_INVERSE.json", temporary).string()},
          {"o2r1_sha256", sha256(bundle)}, {"o2r1_bytes", bundle.size()},
          {"source_numeric_bytes", block.values.size()},
          {"valid_count", block.valid_count},
          {"strict", !block_fallback}, {"fallback_used", block_fallback}});
      valid_count += block.valid_count;
      source_numeric_bytes += block.values.size();
      o2r1_bytes += bundle.size();
      add_samples(kernel_encode, timed.encode_ns, "operator encode");
      add_samples(kernel_decode, timed.decode_ns, "operator decode");
      add_samples(full_encode, timed.full_encode_ns, "full operator encode");
      add_samples(full_decode, timed.full_decode_ns, "full operator decode");
    }

    J receipt = base;
    receipt.update({
        {"status", any_fallback ? "operator_fallback" : "passed"},
        {"applicable", true}, {"strict", !any_fallback},
        {"fallback_used", any_fallback}, {"bit_exact", true},
        {"cold_o2r1_inverse_each_block", true},
        {"ranking_eligible", !any_fallback},
        {"source_physical_identity", physical_identity},
        {"blocks", block_receipts}, {"block_count", blocks.size()},
        {"valid_count", valid_count},
        {"source_numeric_bytes", source_numeric_bytes},
        {"o2r1_complete_bytes", o2r1_bytes},
        {"timing", {
            {"clock", "std::chrono::steady_clock"},
            {"warmups_per_block", kWarmups}, {"repeats", kRepeats},
            {"threads", 1},
            {"kernel_encode", samples_json(kernel_encode)},
            {"kernel_decode", samples_json(kernel_decode)},
            {"full_encode_including_o2r1_pack", samples_json(full_encode)},
            {"full_decode_including_o2r1_parse", samples_json(full_decode)},
            {"aggregation", "sum matching repeat index across entity blocks, then median3"},
            {"file_io_and_process_startup_excluded", true}}}});
    write_json(temporary / "BUNDLE.json", receipt);
    fs2::rename(temporary, output);
    receipt["output_dir"] = output.string();
    receipt["bundle_receipt_sha256"] = sha256(read_file(output / "BUNDLE.json"));
    return receipt;
  } catch (...) {
    // This is a new attempt directory only.  Old and accepted artifacts are
    // never touched; the queue writes the failure receipt outside this path.
    std::error_code ignored;
    fs2::remove_all(temporary, ignored);
    throw;
  }
}
}  // namespace joint_o2r1

int main(int argc, char** argv) {
  if (argc != 1 && !(argc == 2 && std::string(argv[1]) == "--jsonl")) {
    std::cerr << "usage: o2r1_bundle_cli [--jsonl] < requests.jsonl > responses.jsonl\n";
    return 64;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    try {
      std::cout << joint_o2r1::produce(J::parse(line)).dump() << '\n'
                << std::flush;
    } catch (const std::exception& error) {
      J response = {{"schema", "operator-algorithm-o2r1-column-response-v1"},
                    {"status", "failed"}, {"applicable", nullptr},
                    {"strict", false}, {"fallback_used", nullptr},
                    {"bit_exact", false}, {"ranking_eligible", false},
                    {"error", {{"class", "bundle_producer_error"},
                               {"message", error.what()}}}};
      try {
        const auto request = J::parse(line);
        for (const auto key : {"request_id", "dataset_id", "column_id"})
          if (request.contains(key)) response[key] = request.at(key);
        if (request.contains("implementation")) {
          const auto implementation_id =
              request.at("implementation").value("implementation_id", "");
          response["operator_id"] = implementation_id;
          if (implementation_id == "A004" &&
              std::string(error.what()) ==
                  "Camel integer is not exactly representable as float64") {
            response["status"] = "not_applicable";
            response["applicable"] = false;
            response["strict"] = false;
            response["fallback_used"] = false;
            response["reason"] =
                "A004 Camel exact int64-to-float64 input-domain boundary";
            response["error"] = {
                {"class", "native_input_domain_rejection"},
                {"message", error.what()},
                {"narrow_rule", "A004_exact_int64_float64_roundtrip_only"}};
          }
        }
      } catch (...) {}
      std::cout << response.dump() << '\n' << std::flush;
    }
  }
  return 0;
}
