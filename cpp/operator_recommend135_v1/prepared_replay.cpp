#include "campaign.hpp"
#include "common.hpp"
#include "model.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace operator_recommend135_v1 {
namespace {

namespace d = detail;
using Json = nlohmann::json;
using Path = std::filesystem::path;

constexpr std::array<const char*, 4> kTargets = {"space", "encode",
                                                 "decode", "balanced"};
constexpr std::size_t kRepeats = 5;

struct SegmentTiming {
  std::uint64_t feature_cpu_ns = 0;
  std::uint64_t feature_wall_ns = 0;
  std::uint64_t eligibility_cpu_ns = 0;
  std::uint64_t eligibility_wall_ns = 0;
  std::uint64_t model_cpu_ns = 0;
  std::uint64_t model_wall_ns = 0;
  std::uint64_t total_cpu_ns = 0;
  std::uint64_t total_wall_ns = 0;
};

struct SelectionRun {
  std::vector<double> features;
  std::vector<std::string> full_eligible;
  std::vector<std::string> compression_eligible;
  Json prediction;
  Json top_ids;
  SegmentTiming timing;
  std::uint64_t validation_passes = 0;
};

struct E2ERun {
  std::string candidate_id;
  native_graph_v1::Bytes frame;
  std::uint64_t selection_cpu_ns = 0;
  std::uint64_t selection_wall_ns = 0;
  std::uint64_t encode_cpu_ns = 0;
  std::uint64_t encode_wall_ns = 0;
  std::uint64_t total_cpu_ns = 0;
  std::uint64_t total_wall_ns = 0;
  std::uint64_t decode_cpu_ns = 0;
  std::uint64_t decode_wall_ns = 0;
};

std::uint64_t median(std::vector<std::uint64_t> values) {
  d::require(!values.empty(), "cannot take median of empty timings");
  std::sort(values.begin(), values.end());
  return values.at(values.size() / 2);
}

std::vector<std::string> catalog_ids(const Json& catalog) {
  d::require(catalog.contains("candidates") &&
                 catalog.at("candidates").is_array(),
             "candidate catalog is missing candidates");
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto& candidate : catalog.at("candidates")) {
    const auto id = candidate.at("candidate_id").get<std::string>();
    d::require(!id.empty() && seen.insert(id).second,
               "candidate catalog contains an empty or duplicate ID");
    result.push_back(id);
  }
  return result;
}

std::vector<std::string> without_raw(
    const std::vector<std::string>& candidates) {
  std::vector<std::string> result;
  result.reserve(candidates.size());
  std::copy_if(candidates.begin(), candidates.end(),
               std::back_inserter(result),
               [](const std::string& id) { return id != "raw"; });
  return result;
}

Json extract_top_ids(const Json& prediction) {
  Json result = Json::object();
  for (const char* target : kTargets) {
    const auto& target_prediction = prediction.at("targets").at(target);
    result[target] = {{"top1", target_prediction.at("top1")},
                      {"top3", target_prediction.at("top3")},
                      {"top5", target_prediction.at("top5")}};
  }
  return result;
}

Json semantic_prediction(Json prediction) {
  prediction.erase("online_cost");
  return prediction;
}

Json timing_json(const SegmentTiming& timing) {
  return {{"feature_cpu_ns", timing.feature_cpu_ns},
          {"feature_wall_ns", timing.feature_wall_ns},
          {"eligibility_cpu_ns", timing.eligibility_cpu_ns},
          {"eligibility_wall_ns", timing.eligibility_wall_ns},
          {"model_cpu_ns", timing.model_cpu_ns},
          {"model_wall_ns", timing.model_wall_ns},
          {"total_cpu_ns", timing.total_cpu_ns},
          {"total_wall_ns", timing.total_wall_ns}};
}

SelectionRun run_reference(const Json& model, const Block& head) {
  SelectionRun result;
  const auto total_cpu_start = d::cpu_ns();
  const auto total_wall_start = d::wall_ns();

  const auto feature_cpu_start = d::cpu_ns();
  const auto feature_wall_start = d::wall_ns();
  result.features =
      extract_features(head.values, head.dtype, head.missing, head.row_count);
  result.timing.feature_wall_ns = d::wall_ns() - feature_wall_start;
  result.timing.feature_cpu_ns = d::cpu_ns() - feature_cpu_start;

  const auto eligibility_cpu_start = d::cpu_ns();
  const auto eligibility_wall_start = d::wall_ns();
  const Json catalog = candidate_catalog();
  for (const auto& candidate : catalog.at("candidates")) {
    const auto id = candidate.at("candidate_id").get<std::string>();
    if (candidate_eligible(id, head)) {
      result.full_eligible.push_back(id);
    }
  }
  result.compression_eligible = without_raw(result.full_eligible);
  d::require(!result.compression_eligible.empty(),
             "reference path found no compression candidate");
  result.timing.eligibility_wall_ns = d::wall_ns() - eligibility_wall_start;
  result.timing.eligibility_cpu_ns = d::cpu_ns() - eligibility_cpu_start;

  const auto model_cpu_start = d::cpu_ns();
  const auto model_wall_start = d::wall_ns();
  result.prediction =
      predict(model, result.features, result.compression_eligible);
  result.top_ids = extract_top_ids(result.prediction);
  result.timing.model_wall_ns = d::wall_ns() - model_wall_start;
  result.timing.model_cpu_ns = d::cpu_ns() - model_cpu_start;

  result.timing.total_wall_ns = d::wall_ns() - total_wall_start;
  result.timing.total_cpu_ns = d::cpu_ns() - total_cpu_start;
  return result;
}

SelectionRun run_prepared(const PreparedModel& model,
                          const PreparedRegistry& registry,
                          const Block& head) {
  SelectionRun result;
  const auto total_cpu_start = d::cpu_ns();
  const auto total_wall_start = d::wall_ns();

  const auto feature_cpu_start = d::cpu_ns();
  const auto feature_wall_start = d::wall_ns();
  result.features =
      extract_features(head.values, head.dtype, head.missing, head.row_count);
  result.timing.feature_wall_ns = d::wall_ns() - feature_wall_start;
  result.timing.feature_cpu_ns = d::cpu_ns() - feature_cpu_start;

  const auto eligibility_cpu_start = d::cpu_ns();
  const auto eligibility_wall_start = d::wall_ns();
  auto prepared = registry.prepare(head);
  result.full_eligible = std::move(prepared.candidate_ids);
  result.validation_passes = prepared.validation_passes;
  result.compression_eligible = without_raw(result.full_eligible);
  d::require(result.validation_passes == 1,
             "prepared registry did not validate the input exactly once");
  d::require(!result.compression_eligible.empty(),
             "prepared path found no compression candidate");
  result.timing.eligibility_wall_ns = d::wall_ns() - eligibility_wall_start;
  result.timing.eligibility_cpu_ns = d::cpu_ns() - eligibility_cpu_start;

  const auto model_cpu_start = d::cpu_ns();
  const auto model_wall_start = d::wall_ns();
  result.prediction =
      model.predict(result.features, result.compression_eligible);
  result.top_ids = extract_top_ids(result.prediction);
  result.timing.model_wall_ns = d::wall_ns() - model_wall_start;
  result.timing.model_cpu_ns = d::cpu_ns() - model_cpu_start;

  result.timing.total_wall_ns = d::wall_ns() - total_wall_start;
  result.timing.total_cpu_ns = d::cpu_ns() - total_cpu_start;
  return result;
}

void require_selection_parity(const SelectionRun& reference,
                              const SelectionRun& prepared,
                              std::string_view sample_id) {
  const std::string prefix = "selection parity failure for " +
                             std::string(sample_id) + ": ";
  d::require(reference.features == prepared.features,
             prefix + "feature vectors differ");
  d::require(reference.full_eligible == prepared.full_eligible,
             prefix + "full eligibility differs");
  d::require(reference.compression_eligible == prepared.compression_eligible,
             prefix + "compression eligibility differs");
  d::require(reference.top_ids == prepared.top_ids,
             prefix + "Top1/3/5 IDs differ");
  d::require(semantic_prediction(reference.prediction) ==
                 semantic_prediction(prepared.prediction),
             prefix + "complete ranking, scores, or serving differ");
}

Json median_selection_timings(const std::vector<SelectionRun>& runs) {
  d::require(runs.size() == kRepeats,
             "selection timing does not contain five repeats");
  std::vector<std::uint64_t> feature_cpu, feature_wall, eligibility_cpu,
      eligibility_wall, model_cpu, model_wall, total_cpu, total_wall;
  for (const auto& run : runs) {
    feature_cpu.push_back(run.timing.feature_cpu_ns);
    feature_wall.push_back(run.timing.feature_wall_ns);
    eligibility_cpu.push_back(run.timing.eligibility_cpu_ns);
    eligibility_wall.push_back(run.timing.eligibility_wall_ns);
    model_cpu.push_back(run.timing.model_cpu_ns);
    model_wall.push_back(run.timing.model_wall_ns);
    total_cpu.push_back(run.timing.total_cpu_ns);
    total_wall.push_back(run.timing.total_wall_ns);
  }
  return {{"feature_cpu_ns", median(feature_cpu)},
          {"feature_wall_ns", median(feature_wall)},
          {"eligibility_cpu_ns", median(eligibility_cpu)},
          {"eligibility_wall_ns", median(eligibility_wall)},
          {"model_cpu_ns", median(model_cpu)},
          {"model_wall_ns", median(model_wall)},
          {"total_cpu_ns", median(total_cpu)},
          {"total_wall_ns", median(total_wall)}};
}

Json raw_selection_timings(const std::vector<SelectionRun>& runs) {
  Json result = Json::array();
  for (std::size_t i = 0; i < runs.size(); ++i) {
    Json item = timing_json(runs[i].timing);
    item["repeat"] = i;
    item["internal_model_online_cost"] =
        runs[i].prediction.at("online_cost");
    result.push_back(std::move(item));
  }
  return result;
}

template <class Selector>
E2ERun run_e2e(Selector&& select, const Block& tail,
               std::string_view target) {
  const auto total_cpu_start = d::cpu_ns();
  const auto total_wall_start = d::wall_ns();
  const auto selection_cpu_start = d::cpu_ns();
  const auto selection_wall_start = d::wall_ns();
  SelectionRun selection = select();
  const auto selection_wall_ns = d::wall_ns() - selection_wall_start;
  const auto selection_cpu_ns = d::cpu_ns() - selection_cpu_start;
  const auto& top1 = selection.prediction.at("targets")
                         .at(std::string(target))
                         .at("top1");
  d::require(!top1.empty(), "selected target has no Top1 candidate");
  const auto candidate = top1.at(0).get<std::string>();

  const auto encode_cpu_start = d::cpu_ns();
  const auto encode_wall_start = d::wall_ns();
  auto frame = encode_candidate(candidate, tail);
  const auto encode_wall_ns = d::wall_ns() - encode_wall_start;
  const auto encode_cpu_ns = d::cpu_ns() - encode_cpu_start;
  const auto total_wall_ns = d::wall_ns() - total_wall_start;
  const auto total_cpu_ns = d::cpu_ns() - total_cpu_start;

  const auto decode_cpu_start = d::cpu_ns();
  const auto decode_wall_start = d::wall_ns();
  const Block decoded = decode_candidate(frame);
  const auto decode_wall_ns = d::wall_ns() - decode_wall_start;
  const auto decode_cpu_ns = d::cpu_ns() - decode_cpu_start;
  d::require(decoded == tail, "in-process selected frame is not bit exact");

  return {candidate,
          std::move(frame),
          selection_cpu_ns,
          selection_wall_ns,
          encode_cpu_ns,
          encode_wall_ns,
          total_cpu_ns,
          total_wall_ns,
          decode_cpu_ns,
          decode_wall_ns};
}

Json e2e_timings(const std::vector<E2ERun>& runs) {
  d::require(runs.size() == kRepeats,
             "end-to-end timing does not contain five repeats");
  std::vector<std::uint64_t> selection_cpu, selection_wall, encode_cpu,
      encode_wall, total_cpu, total_wall, decode_cpu, decode_wall;
  Json raw = Json::array();
  for (std::size_t i = 0; i < runs.size(); ++i) {
    const auto& run = runs[i];
    d::require(run.candidate_id == runs.front().candidate_id &&
                   run.frame == runs.front().frame,
               "end-to-end repeats produced different selected frames");
    selection_cpu.push_back(run.selection_cpu_ns);
    selection_wall.push_back(run.selection_wall_ns);
    encode_cpu.push_back(run.encode_cpu_ns);
    encode_wall.push_back(run.encode_wall_ns);
    total_cpu.push_back(run.total_cpu_ns);
    total_wall.push_back(run.total_wall_ns);
    decode_cpu.push_back(run.decode_cpu_ns);
    decode_wall.push_back(run.decode_wall_ns);
    raw.push_back({{"repeat", i},
                   {"selection_cpu_ns", run.selection_cpu_ns},
                   {"selection_wall_ns", run.selection_wall_ns},
                   {"encode_cpu_ns", run.encode_cpu_ns},
                   {"encode_wall_ns", run.encode_wall_ns},
                   {"total_cpu_ns", run.total_cpu_ns},
                   {"total_wall_ns", run.total_wall_ns},
                   {"decode_cpu_ns", run.decode_cpu_ns},
                   {"decode_wall_ns", run.decode_wall_ns}});
  }
  return {{"candidate_id", runs.front().candidate_id},
          {"frame_bytes", runs.front().frame.size()},
          {"frame_sha256", d::sha(runs.front().frame)},
          {"bit_exact_all_repeats", true},
          {"median",
           {{"selection_cpu_ns", median(selection_cpu)},
            {"selection_wall_ns", median(selection_wall)},
            {"encode_cpu_ns", median(encode_cpu)},
            {"encode_wall_ns", median(encode_wall)},
            {"total_cpu_ns", median(total_cpu)},
            {"total_wall_ns", median(total_wall)},
            {"decode_cpu_ns", median(decode_cpu)},
            {"decode_wall_ns", median(decode_wall)}}},
          {"raw_timings", std::move(raw)}};
}

Json child_process(const Path& executable,
                   const std::vector<std::string>& arguments,
                   const Path& log_path, int wall_seconds = 180) {
  const auto wall_start = d::wall_ns();
  const pid_t pid = ::fork();
  d::require(pid >= 0, "fork failed");
  if (pid == 0) {
    const rlimit memory{2ULL << 30, 2ULL << 30};
    const rlimit core{0, 0};
    const rlimit cpu{179, 180};
    if (::setpgid(0, 0) != 0 || ::setrlimit(RLIMIT_AS, &memory) != 0 ||
        ::setrlimit(RLIMIT_CORE, &core) != 0 ||
        ::setrlimit(RLIMIT_CPU, &cpu) != 0) {
      _exit(125);
    }
    const int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) _exit(124);
    ::dup2(fd, STDOUT_FILENO);
    ::dup2(fd, STDERR_FILENO);
    ::close(fd);
    std::vector<std::string> owned{executable.string()};
    owned.insert(owned.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(owned.size() + 1);
    for (auto& value : owned) argv.push_back(value.data());
    argv.push_back(nullptr);
    ::execv(executable.c_str(), argv.data());
    _exit(127);
  }

  ::setpgid(pid, pid);
  int status = 0;
  bool timeout = false;
  while (true) {
    const pid_t waited = ::waitpid(pid, &status, WNOHANG);
    if (waited == pid) break;
    if (waited < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("waitpid failed");
    }
    if (d::wall_ns() - wall_start >
        static_cast<std::uint64_t>(wall_seconds) * 1000000000ULL) {
      timeout = true;
      d::require(::kill(-pid, SIGKILL) == 0 || errno == ESRCH,
                 "failed to terminate timed-out child process group");
      while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
          throw std::runtime_error("failed to reap timed-out child");
        }
      }
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return {{"pid", pid},
          {"parent_pid", ::getpid()},
          {"exit_code", WIFEXITED(status) ? WEXITSTATUS(status) : -1},
          {"signal", WIFSIGNALED(status) ? WTERMSIG(status) : 0},
          {"timeout", timeout},
          {"wall_ns", d::wall_ns() - wall_start},
          {"argv", arguments},
          {"log_path", log_path.string()}};
}

bool child_succeeded(const Json& process) {
  return process.at("exit_code") == 0 &&
         !process.at("timeout").get<bool>();
}

Json fresh_decode_witness(
    const Path& cli, const Json& sample,
    const std::map<std::string, Json>& selected_expected_frames,
                          const Path& sample_root) {
  const Path witness_root = sample_root / "fresh_decode";
  std::filesystem::create_directories(witness_root);
  const Path sample_path = witness_root / "SAMPLE.json";
  d::write_json(sample_path, sample);
  Json cases = Json::array();
  std::vector<std::string> selected_candidates;
  selected_candidates.reserve(selected_expected_frames.size());
  for (const auto& [candidate, expected_frame] : selected_expected_frames) {
    selected_candidates.push_back(candidate);
    const Path case_root = witness_root / candidate;
    std::filesystem::create_directories(case_root);
    const Path frame_path = case_root / "frame.ngf";
    const Json encode_process = child_process(
        cli, {"encode", candidate, sample_path.string(), frame_path.string()},
        case_root / "encode.log");
    d::require(child_succeeded(encode_process),
               "fresh witness encode failed for " + candidate);
    const Json encode_receipt = d::read_json(frame_path.string() + ".json");
    d::require(encode_receipt.at("candidate_id") == candidate &&
                   encode_receipt.at("frame_sha256") ==
                       d::file_sha(frame_path),
               "fresh witness encode receipt mismatch");
    d::require(encode_receipt.at("frame_sha256") ==
                       expected_frame.at("frame_sha256") &&
                   encode_receipt.at("frame_bytes") ==
                       expected_frame.at("frame_bytes"),
               "fresh witness frame differs from the timed in-process frame");

    const Path decoded_root = case_root / "decoded";
    const Json decode_process =
        child_process(cli, {"decode", frame_path.string(),
                            decoded_root.string()},
                      case_root / "decode.log");
    d::require(child_succeeded(decode_process),
               "fresh witness decode failed for " + candidate);
    const Json verification = verify_decoded(sample, decoded_root);
    d::require(verification.at("verified").get<bool>(),
               "fresh witness decode is not bit exact");
    d::require(encode_process.at("pid") != decode_process.at("pid") &&
                   verification.at("decode").at("decode_pid") ==
                       decode_process.at("pid"),
               "fresh witness decode did not run in its recorded process");
    cases.push_back({{"candidate_id", candidate},
                     {"frame_bytes", encode_receipt.at("frame_bytes")},
                     {"frame_sha256", encode_receipt.at("frame_sha256")},
                     {"matches_timed_frame", true},
                     {"encode_process", encode_process},
                     {"decode_process", decode_process},
                     {"verification", verification}});
  }
  return {{"case_count", cases.size()},
          {"unique_selected_candidates", selected_candidates},
          {"cases", std::move(cases)}};
}

void add_aggregate(std::map<std::string, std::vector<std::uint64_t>>& target,
                   const Json& timing) {
  for (const auto* key : {"feature_cpu_ns", "feature_wall_ns",
                          "eligibility_cpu_ns", "eligibility_wall_ns",
                          "model_cpu_ns", "model_wall_ns", "total_cpu_ns",
                          "total_wall_ns"}) {
    target[key].push_back(timing.at(key).get<std::uint64_t>());
  }
}

using TimingAggregate =
    std::map<std::string, std::vector<std::uint64_t>>;
using DatasetTimingAggregate = std::map<std::string, TimingAggregate>;

void add_dataset_aggregate(DatasetTimingAggregate& target,
                           const std::string& dataset_id,
                           const Json& timing) {
  add_aggregate(target[dataset_id], timing);
}

Json aggregate_medians(
    const std::map<std::string, std::vector<std::uint64_t>>& values) {
  Json result = Json::object();
  for (const auto& [key, observations] : values) {
    result[key] = median(observations);
  }
  return result;
}

Json aggregate_dataset_medians(const DatasetTimingAggregate& values) {
  TimingAggregate dataset_medians;
  for (const auto& [dataset_id, observations] : values) {
    (void)dataset_id;
    const Json per_dataset = aggregate_medians(observations);
    for (const auto& [key, value] : per_dataset.items()) {
      dataset_medians[key].push_back(value.get<std::uint64_t>());
    }
  }
  return aggregate_medians(dataset_medians);
}

Json replay(const Path& frozen_root, const Path& output_root) {
  const Path inputs_path = frozen_root / "inputs" / "INPUTS.json";
  const Path rows_path = frozen_root / "campaign" / "ROWS.json";
  const Path model_path = frozen_root / "model" / "MODEL.json";
  const Path cli_path = frozen_root / "build" / "operator_recommend135_cli";
  d::require(std::filesystem::is_regular_file(inputs_path),
             "missing frozen inputs/INPUTS.json");
  d::require(std::filesystem::is_regular_file(rows_path),
             "missing frozen campaign/ROWS.json");
  d::require(std::filesystem::is_regular_file(model_path),
             "missing frozen model/MODEL.json");
  d::require(std::filesystem::is_regular_file(cli_path) &&
                 ::access(cli_path.c_str(), X_OK) == 0,
             "frozen CLI is missing or not executable");

  const auto model_load_cpu_start = d::cpu_ns();
  const auto model_load_wall_start = d::wall_ns();
  const Json model = d::read_json(model_path);
  const auto model_load_wall_ns = d::wall_ns() - model_load_wall_start;
  const auto model_load_cpu_ns = d::cpu_ns() - model_load_cpu_start;

  const auto registry_cpu_start = d::cpu_ns();
  const auto registry_wall_start = d::wall_ns();
  const PreparedRegistry registry;
  const auto registry_wall_ns = d::wall_ns() - registry_wall_start;
  const auto registry_cpu_ns = d::cpu_ns() - registry_cpu_start;
  const Json& catalog = registry.catalog();
  const auto ids = catalog_ids(catalog);
  d::require(ids.size() == 24,
             "prepared replay requires the frozen 24-candidate catalog");
  const auto catalog_sha256 = d::sha(catalog.dump());
  d::require(model.at("candidate_catalog_sha256").is_string() &&
                 model.at("candidate_catalog_sha256").get<std::string>() ==
                     catalog_sha256,
             "model/runtime catalog SHA mismatch");
  d::require(model.at("candidate_ids").get<std::vector<std::string>>() == ids,
             "model candidate scope differs from runtime catalog");
  d::require(model.at("feature_names") == Json(feature_names()),
             "model feature contract differs from runtime");

  const auto preparation_cpu_start = d::cpu_ns();
  const auto preparation_wall_start = d::wall_ns();
  const PreparedModel prepared_model(model);
  const auto preparation_wall_ns = d::wall_ns() - preparation_wall_start;
  const auto preparation_cpu_ns = d::cpu_ns() - preparation_cpu_start;
  d::require(prepared_model.candidate_catalog_sha256() == catalog_sha256,
             "prepared model catalog binding mismatch");

  const Json inputs = d::read_json(inputs_path);
  const Json rows = d::read_json(rows_path);
  d::require(inputs.value("schema", std::string()) ==
                 "operator-recommend135-inputs-v1" &&
                 inputs.contains("samples") && inputs.at("samples").is_array(),
             "unsupported frozen input schema");
  d::require(rows.is_array(), "frozen campaign rows must be an array");
  d::require(inputs.at("samples").size() == 360 && rows.size() == 360,
             "prepared replay requires exactly 360 frozen samples and rows");

  std::map<std::string, const Json*> rows_by_sample;
  std::set<std::string> datasets;
  std::set<std::string> source_groups;
  for (const auto& row : rows) {
    const auto id = row.at("sample_id").get<std::string>();
    d::require(rows_by_sample.emplace(id, &row).second,
               "duplicate sample ID in frozen rows");
    datasets.insert(row.at("dataset_id").get<std::string>());
    source_groups.insert(row.at("group_id").get<std::string>());
    d::require(row.at("catalog_sha256") == catalog_sha256,
               "frozen row catalog binding mismatch");
  }
  d::require(datasets.size() == 135,
             "frozen replay does not cover exactly 135 datasets");

  std::map<std::string, std::vector<std::uint64_t>> reference_aggregate;
  std::map<std::string, std::vector<std::uint64_t>> prepared_aggregate;
  std::map<std::string, std::vector<std::uint64_t>> e2e_reference_aggregate;
  std::map<std::string, std::vector<std::uint64_t>> e2e_prepared_aggregate;
  DatasetTimingAggregate reference_dataset_aggregate;
  DatasetTimingAggregate prepared_dataset_aggregate;
  DatasetTimingAggregate e2e_reference_dataset_aggregate;
  DatasetTimingAggregate e2e_prepared_dataset_aggregate;
  Json receipt_index = Json::array();
  std::size_t fresh_decode_cases = 0;
  std::size_t sample_number = 0;

  for (const auto& sample : inputs.at("samples")) {
    const auto sample_id = sample.at("sample_id").get<std::string>();
    const auto row_it = rows_by_sample.find(sample_id);
    d::require(row_it != rows_by_sample.end(),
               "input sample is missing from frozen rows: " + sample_id);
    const Json& row = *row_it->second;
    d::require(sample.at("dataset_id") == row.at("dataset_id") &&
                   sample.at("dtype") == row.at("dtype") &&
                   sample.at("scale") == row.at("scale"),
               "input/row metadata mismatch for " + sample_id);
    const Block head = load_input_part(sample, "head");
    const Block tail = load_input_part(sample, "tail");

    const SelectionRun reference_warmup = run_reference(model, head);
    const SelectionRun prepared_warmup =
        run_prepared(prepared_model, registry, head);
    require_selection_parity(reference_warmup, prepared_warmup, sample_id);
    d::require(Json(reference_warmup.features) == row.at("features"),
               "extracted head features differ from frozen row: " +
                   sample_id);
    d::require(Json(reference_warmup.full_eligible) ==
                   row.at("eligible_candidates"),
               "runtime eligibility differs from frozen row: " + sample_id);

    std::vector<SelectionRun> reference_runs;
    std::vector<SelectionRun> prepared_runs;
    reference_runs.reserve(kRepeats);
    prepared_runs.reserve(kRepeats);
    for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
      SelectionRun reference;
      SelectionRun prepared;
      if (repeat % 2 == 0) {
        reference = run_reference(model, head);
        prepared = run_prepared(prepared_model, registry, head);
      } else {
        prepared = run_prepared(prepared_model, registry, head);
        reference = run_reference(model, head);
      }
      require_selection_parity(reference, prepared, sample_id);
      d::require(Json(reference.features) == row.at("features"),
                 "repeat feature mismatch for " + sample_id);
      reference_runs.push_back(std::move(reference));
      prepared_runs.push_back(std::move(prepared));
    }

    const Json reference_median = median_selection_timings(reference_runs);
    const Json prepared_median = median_selection_timings(prepared_runs);
    const auto dataset_id = sample.at("dataset_id").get<std::string>();
    const auto reference_semantic_sha256 = d::sha(
        semantic_prediction(reference_runs.front().prediction).dump());
    const auto prepared_semantic_sha256 = d::sha(
        semantic_prediction(prepared_runs.front().prediction).dump());
    d::require(reference_semantic_sha256 == prepared_semantic_sha256,
               "semantic prediction hashes differ for " + sample_id);
    add_aggregate(reference_aggregate, reference_median);
    add_aggregate(prepared_aggregate, prepared_median);
    add_dataset_aggregate(reference_dataset_aggregate, dataset_id,
                          reference_median);
    add_dataset_aggregate(prepared_dataset_aggregate, dataset_id,
                          prepared_median);

    Json target_results = Json::object();
    std::set<std::string> selected_candidates;
    std::map<std::string, Json> selected_expected_frames;
    for (const char* target : kTargets) {
      auto reference_selector = [&]() { return run_reference(model, head); };
      auto prepared_selector = [&]() {
        return run_prepared(prepared_model, registry, head);
      };
      const E2ERun reference_e2e_warmup =
          run_e2e(reference_selector, tail, target);
      const E2ERun prepared_e2e_warmup =
          run_e2e(prepared_selector, tail, target);
      d::require(reference_e2e_warmup.candidate_id ==
                         prepared_e2e_warmup.candidate_id &&
                     reference_e2e_warmup.frame == prepared_e2e_warmup.frame,
                 "warmup end-to-end frame parity failed for " + sample_id +
                     "/" + target);

      std::vector<E2ERun> reference_e2e_runs;
      std::vector<E2ERun> prepared_e2e_runs;
      reference_e2e_runs.reserve(kRepeats);
      prepared_e2e_runs.reserve(kRepeats);
      for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
        E2ERun reference;
        E2ERun prepared;
        if (repeat % 2 == 0) {
          reference = run_e2e(reference_selector, tail, target);
          prepared = run_e2e(prepared_selector, tail, target);
        } else {
          prepared = run_e2e(prepared_selector, tail, target);
          reference = run_e2e(reference_selector, tail, target);
        }
        d::require(reference.candidate_id == prepared.candidate_id,
                   "end-to-end selected candidate differs for " + sample_id +
                       "/" + target);
        d::require(reference.frame == prepared.frame,
                   "end-to-end full frame bytes differ for " + sample_id +
                       "/" + target);
        reference_e2e_runs.push_back(std::move(reference));
        prepared_e2e_runs.push_back(std::move(prepared));
      }
      const Json reference_e2e = e2e_timings(reference_e2e_runs);
      const Json prepared_e2e = e2e_timings(prepared_e2e_runs);
      d::require(reference_e2e.at("candidate_id") ==
                         prepared_e2e.at("candidate_id") &&
                     reference_e2e.at("frame_sha256") ==
                         prepared_e2e.at("frame_sha256") &&
                     reference_e2e.at("frame_bytes") ==
                         prepared_e2e.at("frame_bytes"),
                 "end-to-end representative frame receipt differs");
      selected_candidates.insert(
          reference_e2e.at("candidate_id").get<std::string>());
      const auto selected_id =
          reference_e2e.at("candidate_id").get<std::string>();
      const Json expected_frame{{"frame_sha256",
                                 reference_e2e.at("frame_sha256")},
                                {"frame_bytes",
                                 reference_e2e.at("frame_bytes")}};
      const auto [expected_it, inserted] =
          selected_expected_frames.emplace(selected_id, expected_frame);
      d::require(inserted || expected_it->second == expected_frame,
                 "one selected candidate produced different timed frames across targets");
      for (const auto* key : {"selection_cpu_ns", "selection_wall_ns",
                              "encode_cpu_ns", "encode_wall_ns",
                              "total_cpu_ns", "total_wall_ns",
                              "decode_cpu_ns", "decode_wall_ns"}) {
        e2e_reference_aggregate[std::string(target) + "." + key].push_back(
            reference_e2e.at("median").at(key).get<std::uint64_t>());
        e2e_prepared_aggregate[std::string(target) + "." + key].push_back(
            prepared_e2e.at("median").at(key).get<std::uint64_t>());
        e2e_reference_dataset_aggregate[dataset_id]
                                       [std::string(target) + "." + key]
                                           .push_back(reference_e2e.at("median")
                                                          .at(key)
                                                          .get<std::uint64_t>());
        e2e_prepared_dataset_aggregate[dataset_id]
                                      [std::string(target) + "." + key]
                                          .push_back(prepared_e2e.at("median")
                                                         .at(key)
                                                         .get<std::uint64_t>());
      }
      target_results[target] = {
          {"selected_candidate_id", reference_e2e.at("candidate_id")},
          {"frames_identical", true},
          {"reference", reference_e2e},
          {"prepared", prepared_e2e}};
    }

    const Path sample_root = output_root / "samples" / sample_id;
    std::filesystem::create_directories(sample_root);
    d::require(selected_candidates.size() == selected_expected_frames.size(),
               "selected frame witness map is incomplete");
    const Json fresh = fresh_decode_witness(
        cli_path, sample, selected_expected_frames, sample_root);
    fresh_decode_cases += fresh.at("case_count").get<std::size_t>();
    const Json receipt{
        {"schema", "operator-recommend135-prepared-sample-v1"},
        {"sample_id", sample_id},
        {"dataset_id", sample.at("dataset_id")},
        {"group_id", row.at("group_id")},
        {"scale", sample.at("scale")},
        {"dtype", sample.at("dtype")},
        {"candidate_scope", "frozen_runtime_24_minus_raw_for_selection"},
        {"ranking_score_serving_parity", true},
        {"semantic_prediction_sha256", reference_semantic_sha256},
        {"feature_matches_frozen_row", true},
        {"eligibility_matches_frozen_row", true},
        {"prepared_validation_passes_per_selection", 1},
        {"selection",
         {{"reference",
           {{"median", reference_median},
            {"raw_timings", raw_selection_timings(reference_runs)}}},
          {"prepared",
           {{"median", prepared_median},
            {"raw_timings", raw_selection_timings(prepared_runs)}}},
          {"top_ids", reference_runs.front().top_ids}}},
        {"end_to_end", target_results},
        {"fresh_process_decode", fresh}};
    const Path receipt_path = sample_root / "RECEIPT.json";
    d::write_json(receipt_path, receipt);
    receipt_index.push_back(
        {{"sample_id", sample_id},
         {"path", receipt_path.string()},
         {"sha256", d::file_sha(receipt_path)},
         {"unique_selected_candidate_count", selected_candidates.size()}});

    ++sample_number;
    d::write_json(output_root / "PROGRESS.json",
                  {{"samples_done", sample_number},
                   {"samples_total", inputs.at("samples").size()},
                   {"parity_failures", 0},
                   {"fresh_decode_cases", fresh_decode_cases},
                   {"last_sample", sample_id}});
  }
  d::require(sample_number == 360 && receipt_index.size() == 360,
             "prepared replay did not complete every frozen sample");

  const auto compression_ids = without_raw(ids);
  return {
      {"schema", "operator-recommend135-prepared-replay-v1"},
      {"status", "completed"},
      {"mode", "frozen_model_semantic_and_performance_parity"},
      {"formal_evaluation_contract",
       {{"model_source", "frozen model/MODEL.json"},
        {"fit_or_refit_performed_by_this_runner", false},
        {"confirm_group_labels_used_by_this_runner_for_fitting_thresholds_or_model_selection",
         false},
        {"frozen_model_training_provenance",
         "not reinterpreted; MODEL.json is used only for same-input implementation parity and makes no source-group generalization claim"},
        {"legacy_oof_refit", false},
        {"historical_oof_parity_gap",
         "frozen per-fold model artifacts are absent; refitting them is forbidden, so historical OOF parity is not rerun"},
        {"source_group_split",
         "preserved as frozen row metadata; this runner performs no fitting"},
        {"quality_scope",
         "semantic/performance parity only; 3519-column oracle quality is a separate frozen-split evaluation"}}},
      {"candidate_scope",
       {{"catalog_candidate_count", ids.size()},
        {"catalog_candidate_ids", ids},
        {"selection_candidate_count", compression_ids.size()},
        {"selection_candidate_ids", compression_ids},
        {"raw_excluded_from_selection", true},
        {"disabled_or_absent_families",
         Json::array({"NNCP", "PPMd", "DMC", "PAQ8", "CMIX", "LSTM"})}}},
      {"bindings",
       {{"frozen_root", std::filesystem::absolute(frozen_root).string()},
        {"inputs_sha256", d::file_sha(inputs_path)},
        {"rows_sha256", d::file_sha(rows_path)},
        {"model_sha256", d::file_sha(model_path)},
        {"catalog_sha256", catalog_sha256},
        {"cli_sha256", d::file_sha(cli_path)}}},
      {"coverage",
       {{"dataset_count", datasets.size()},
        {"source_group_count", source_groups.size()},
        {"sample_count", sample_number},
        {"targets_per_sample", kTargets.size()},
        {"warmups_per_path", 1},
        {"measured_repeats_per_path", kRepeats}}},
      {"cold_preparation",
       {{"model_json_load_cpu_ns", model_load_cpu_ns},
        {"model_json_load_wall_ns", model_load_wall_ns},
        {"prepared_registry_cpu_ns", registry_cpu_ns},
        {"prepared_registry_wall_ns", registry_wall_ns},
        {"prepared_model_cpu_ns", preparation_cpu_ns},
        {"prepared_model_wall_ns", preparation_wall_ns}}},
      {"selection_timing_sample_equal_median_of_sample_medians",
       {{"reference", aggregate_medians(reference_aggregate)},
        {"prepared", aggregate_medians(prepared_aggregate)},
        {"contract",
         "loaded Block; feature extraction, eligibility, complete prediction JSON and Top1/3/5 extraction included"}}},
      {"selection_timing_dataset_equal_median_of_dataset_medians",
       {{"reference", aggregate_dataset_medians(reference_dataset_aggregate)},
        {"prepared", aggregate_dataset_medians(prepared_dataset_aggregate)},
        {"aggregation",
         "median of sample medians within each dataset, then median across 135 datasets"},
        {"contract",
         "loaded Block; feature extraction, eligibility, complete prediction JSON and Top1/3/5 extraction included"}}},
      {"end_to_end_sample_equal_median_of_sample_medians",
       {{"reference", aggregate_medians(e2e_reference_aggregate)},
        {"prepared", aggregate_medians(e2e_prepared_aggregate)},
        {"contract",
         "loaded head/tail Blocks; select on head then encode selected tail; decode timed separately and bit-exact"}}},
      {"end_to_end_dataset_equal_median_of_dataset_medians",
       {{"reference",
         aggregate_dataset_medians(e2e_reference_dataset_aggregate)},
        {"prepared",
         aggregate_dataset_medians(e2e_prepared_dataset_aggregate)},
        {"aggregation",
         "median of sample medians within each dataset, then median across 135 datasets"},
        {"contract",
         "loaded head/tail Blocks; select on head then encode selected tail; decode timed separately and bit-exact"}}},
      {"parity",
       {{"samples_checked", sample_number},
        {"complete_ranking_score_serving_mismatches", 0},
        {"top_id_mismatches", 0},
        {"eligibility_mismatches", 0},
        {"feature_mismatches", 0},
        {"full_frame_byte_mismatches", 0},
        {"in_process_decode_failures", 0},
        {"fresh_process_decode_failures", 0}}},
      {"fresh_process_decode",
       {{"case_count", fresh_decode_cases},
        {"scope", "one case per sample and unique Top1 candidate across four targets"},
        {"decoder", cli_path.string()},
        {"frame_contract", "old CLI encode followed by fresh old CLI decode"}}},
      {"receipt_index", receipt_index}};
}

}  // namespace
}  // namespace operator_recommend135_v1

int main(int argc, char** argv) {
  namespace ors = operator_recommend135_v1;
  namespace d = operator_recommend135_v1::detail;
  std::filesystem::path output;
  bool output_owned = false;
  try {
    d::require(argc == 3,
               "usage: prepared_replay FROZEN_ROOT OUTPUT");
    const std::filesystem::path frozen_root = argv[1];
    output = argv[2];
    if (std::filesystem::exists(output)) {
      d::require(std::filesystem::is_directory(output) &&
                     std::filesystem::is_empty(output),
                 "output must not exist or must be an empty directory");
    } else {
      std::filesystem::create_directories(output);
    }
    output_owned = true;
    d::write_json(output / "START.json",
                  {{"schema", "operator-recommend135-prepared-replay-start-v1"},
                   {"pid", ::getpid()},
                   {"frozen_root",
                    std::filesystem::absolute(frozen_root).string()},
                   {"policy", "frozen model only; no fit or refit"}});
    const nlohmann::json result = ors::replay(frozen_root, output);
    d::write_json(output / "REPLAY.json", result);
    std::cout << nlohmann::json({{"status", "completed"},
                                 {"samples", 360},
                                 {"output", output.string()}})
                     .dump()
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    const nlohmann::json failure{{"status", "failed"},
                                 {"error", error.what()},
                                 {"pid", ::getpid()}};
    if (output_owned) {
      try {
        d::write_json(output / "FAILURE.json", failure);
      } catch (...) {
      }
    }
    std::cerr << failure.dump() << '\n';
    return 1;
  }
}
