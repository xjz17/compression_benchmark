// Local-only C++ coordinator: each operator tree owns a separate worker,
// frame, fresh-process decoder and durable receipt. No remote process control.
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/utsname.h>
#if defined(__linux__)
#include <sched.h>
#endif
#include <unistd.h>

namespace {
using J = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}

std::string read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(bool(input), "cannot read " + path.string());
  const auto size = input.tellg();
  require(size >= 0 && size <= std::streamoff(64ULL << 20), "oversized coordinator file");
  input.seekg(0);
  std::string result(static_cast<std::size_t>(size), '\0');
  if (size) input.read(result.data(), size);
  require(bool(input) || !size, "short read");
  return result;
}

std::string sha(const std::string& value) {
  auto* context = EVP_MD_CTX_new();
  require(context != nullptr, "cannot allocate SHA256 context");
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned length = 0;
  const auto ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, value.data(), value.size()) == 1 &&
                  EVP_DigestFinal_ex(context, digest, &length) == 1;
  EVP_MD_CTX_free(context);
  require(ok && length == 32, "SHA256 failed");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned i = 0; i < length; ++i) output << std::setw(2) << unsigned(digest[i]);
  return output.str();
}

void atomic(const fs::path& path, const std::string& value) {
  fs::create_directories(path.parent_path());
  const auto temporary = path.string() + ".tmp-" + std::to_string(getpid());
  const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(fd >= 0, "cannot create receipt temporary file");
  std::size_t position = 0;
  while (position < value.size()) {
    const auto written = write(fd, value.data() + position, value.size() - position);
    if (written <= 0) { close(fd); throw std::runtime_error("receipt write failed"); }
    position += static_cast<std::size_t>(written);
  }
  require(fsync(fd) == 0, "receipt fsync failed");
  close(fd);
  require(rename(temporary.c_str(), path.c_str()) == 0, "receipt rename failed");
  const int directory = open(path.parent_path().c_str(), O_RDONLY);
  if (directory >= 0) { require(fsync(directory) == 0, "directory fsync failed"); close(directory); }
}

void atomic_json(const fs::path& path, const J& value) { atomic(path, value.dump(2) + "\n"); }

void durable_artifact(const fs::path& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  require(fd >= 0, "cannot open artifact for fsync");
  const auto flushed = fsync(fd);
  close(fd);
  require(flushed == 0, "artifact fsync failed");
  const int directory = open(path.parent_path().c_str(), O_RDONLY);
  require(directory >= 0, "cannot open artifact directory");
  const auto synced = fsync(directory);
  close(directory);
  require(synced == 0, "artifact directory fsync failed");
}

void immutable(const fs::path& path, const J& value) {
  if (fs::exists(path)) require(J::parse(read(path)) == value, "immutable run/receipt identity changed");
  else atomic_json(path, value);
}

void event(const fs::path& path, const J& value) {
  const auto text = value.dump() + "\n";
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
  require(fd >= 0, "event log open failed");
  std::size_t position = 0;
  while (position < text.size()) {
    const auto n = write(fd, text.data() + position, text.size() - position);
    if (n <= 0) { close(fd); throw std::runtime_error("event log write failed"); }
    position += static_cast<std::size_t>(n);
  }
  require(fsync(fd) == 0, "event log fsync failed");
  close(fd);
}

struct Child {
  int exit_code = -1;
  bool timed_out = false;
  pid_t pid = -1;
};

Child execute(const fs::path& binary, const fs::path& request,
              const fs::path& response, const fs::path& error,
              unsigned timeout_ms, unsigned max_rss_mib) {
  const auto pid = fork();
  require(pid >= 0, "fork failed");
  if (pid == 0) {
    if (setpgid(0, 0) != 0) _exit(125);
    const auto input = open(request.c_str(), O_RDONLY);
    const auto output = open(response.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    const auto errors = open(error.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (input < 0 || output < 0 || errors < 0) _exit(125);
    if (dup2(input, STDIN_FILENO) < 0 || dup2(output, STDOUT_FILENO) < 0 ||
        dup2(errors, STDERR_FILENO) < 0) _exit(125);
    close(input); close(output); close(errors);
#if defined(__linux__)
    struct rlimit memory {static_cast<rlim_t>(max_rss_mib) << 20,
                          static_cast<rlim_t>(max_rss_mib) << 20};
    if (setrlimit(RLIMIT_AS, &memory) != 0) _exit(125);
#else
    // Darwin does not provide Linux's usable address-space enforcement.
    // The run receipt states this explicitly; production 192 runs use Linux.
    (void)max_rss_mib;
#endif
    struct rlimit core {0, 0};
    if (setrlimit(RLIMIT_CORE, &core) != 0) _exit(125);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    execl(binary.c_str(), binary.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  // This pid was created above by this coordinator. Never address an existing
  // experiment, process name or externally supplied pid/process group.
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  int status = 0;
  while (true) {
    const auto result = waitpid(pid, &status, WNOHANG);
    if (result == pid) break;
    require(result >= 0, "waitpid failed");
    if (Clock::now() >= deadline) {
      // setpgid may not have happened yet under extreme scheduling delay.
      if (kill(-pid, SIGKILL) != 0) (void)kill(pid, SIGKILL);
      require(waitpid(pid, &status, 0) == pid, "timed-out child reap failed");
      return {-1, true, pid};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return {WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status), false, pid};
}

J response(const fs::path& file, const std::string& request_id, const std::string& phase) {
  const auto value = J::parse(read(file));
  require(value.is_object() && value.value("schema", "") == "operator-harness-response-v2",
          "worker response schema mismatch");
  require(value.value("request_id", "") == request_id && value.value("phase", "") == phase,
          "worker response identity mismatch");
  return value;
}

bool excluded(const std::string& id) {
  return id == "A106" || id == "A107" || id == "A108" || id == "A109" || id == "A110";
}

std::string text_field(const J& value, const char* field) {
  return value.contains(field) && value.at(field).is_string() ? value.at(field).get<std::string>() : "";
}

void samples(const J& values, unsigned repeats, const char* label) {
  require(values.is_array() && values.size() == repeats, std::string("invalid sample count: ") + label);
  for (const auto& value : values)
    require(value.is_number_unsigned() || (value.is_number_integer() && value.get<std::int64_t>() >= 0),
            std::string("invalid timer sample: ") + label);
}

void verified_receipt(const fs::path& path, const std::string& identity) {
  auto record = J::parse(read(path));
  const auto expected = record.at("receipt_sha256").get<std::string>();
  record.erase("receipt_sha256");
  require(sha(record.dump()) == expected && record.at("run_sha256") == identity,
          "committed receipt is changed or belongs to another run");
  if (record.at("status") == "ok") {
    require(sha(read(record.at("frame_path").get<std::string>())) ==
                record.at("frame_sha256").get<std::string>(),
            "committed frame changed");
    require(sha(read(record.at("output_path").get<std::string>())) ==
                record.at("input_sha256").get<std::string>(),
            "committed restored input changed");
  }
}

J scheduling_observation() {
  J result{{"nice", getpriority(PRIO_PROCESS, 0)}};
  struct utsname host {};
  require(uname(&host) == 0, "cannot identify execution host");
  result["hostname"] = host.nodename;
  result["os"] = host.sysname;
#if defined(__linux__)
  const auto policy = sched_getscheduler(0);
  require(policy >= 0, "cannot inspect scheduler");
  result["scheduler_policy"] = policy;
  result["sched_idle_observed"] = policy == SCHED_IDLE;
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  require(sched_getaffinity(0, sizeof(cpus), &cpus) == 0, "cannot inspect CPU affinity");
  result["cpu_affinity"] = J::array();
  for (int i = 0; i < CPU_SETSIZE; ++i)
    if (CPU_ISSET(i, &cpus)) result["cpu_affinity"].push_back(i);
#else
  result["scheduler_policy"] = "not_observed_on_this_platform";
#endif
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 6, "usage: operator_harness_v2_batch_runner DRIVER PLAN_JSON OUTPUT_DIR TIMEOUT_MS MAX_AS_MIB");
    const auto binary = fs::canonical(argv[1]);
    const auto plan = J::parse(read(argv[2]));
    const auto output = fs::absolute(argv[3]).lexically_normal();
    const auto timeout_ms = static_cast<unsigned>(std::stoul(argv[4]));
    const auto max_rss_mib = static_cast<unsigned>(std::stoul(argv[5]));
    require(timeout_ms > 0 && timeout_ms <= 120000 && max_rss_mib >= 256 && max_rss_mib <= 4096,
            "invalid child resource limits");
    require(plan.at("schema") == "operator-harness-batch-plan-v2" && plan.at("cases").is_array(),
            "wrong batch plan schema");
    require(!plan.at("cases").empty() && plan.at("cases").size() <= 4096, "invalid case count");
    if (fs::exists(output) && !fs::is_empty(output))
      require(fs::is_regular_file(output / "RUN.json"), "refusing a nonempty unrelated output directory");
    fs::create_directories(output);
    const int lock = open((output / ".lock").c_str(), O_CREAT | O_RDWR, 0600);
    require(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0, "another runner owns this output directory");
    std::set<std::string> case_ids;
    for (const auto& item : plan.at("cases")) {
      require(item.is_object() && item.contains("case_id") && item.at("case_id").is_string(), "case_id missing");
      require(case_ids.insert(item.at("case_id").get<std::string>()).second, "duplicate case_id");
    }
    J run{{"schema", "operator-harness-native-batch-run-v2"}, {"plan_sha256", sha(plan.dump())},
          {"driver_sha256", sha(read(binary))}, {"timeout_ms", timeout_ms}, {"max_address_space_mib", max_rss_mib},
          {"excluded_ids", {"A106", "A107", "A108", "A109", "A110"}},
          {"scope", "bounded operator-tree validation on shared host; no performance ranking"}};
#if defined(__linux__)
    run["address_space_limit_enforced"] = true;
#else
    run["address_space_limit_enforced"] = false;
#endif
    run["coordinator_scheduling_observed"] = scheduling_observation();
    const auto identity = sha(run.dump());
    run["run_sha256"] = identity;
    immutable(output / "RUN.json", run);
    immutable(output / "PLAN.json", plan);
    J counts = J::object(), outcomes = J::array();
    for (const auto& item : plan.at("cases")) {
      const auto case_id = item.at("case_id").get<std::string>();
      const auto key = sha(item.dump());
      const auto directory = output / "cases" / key;
      const auto receipt = directory / "RESULT.json";
      if (fs::exists(receipt)) {
        verified_receipt(receipt, identity);
        const auto record = J::parse(read(receipt));
        const auto state = record.at("status").get<std::string>();
        counts[state] = counts.value(state, 0) + 1;
        outcomes.push_back({{"case_id", case_id}, {"implementation_id", item.value("implementation_id", "")},
                            {"status", state}, {"receipt", receipt.string()}});
        continue;
      }
      fs::create_directories(directory);
      event(output / "events.jsonl", {{"event", "case_started"}, {"case_id", case_id}});
      J result{{"schema", "operator-harness-native-case-result-v2"}, {"case_id", case_id},
               {"run_sha256", identity}, {"case_sha256", key}, {"status", "failed"},
               {"implementation_id", text_field(item, "implementation_id")},
               {"batch_id", text_field(item, "batch_id")}, {"sample_id", text_field(item, "sample_id")},
               {"scope", "this exact sample, source and IO recipe; not full-domain verification"}};
      for (const auto* field : {"suite", "category", "case_class", "boundary_family",
                                 "expected_execution_path", "expected_outcome", "expected_reason",
                                 "expected_reason_code", "expected_reason_category", "planned_path", "provenance"})
        if (item.contains(field)) result[field] = item.at(field);
      try {
        const auto id = item.at("implementation_id").get<std::string>();
        require(id.size() == 4 && id[0] == 'A' && std::all_of(id.begin() + 1, id.end(), ::isdigit), "invalid implementation_id");
        if (excluded(id)) {
          result["status"] = "disabled";
          result["reason"] = "EXCLUDED_PROBABILITY_MODEL";
        } else {
          auto request = item.at("request");
          require(request.at("implementation_ids") == J::array({id}), "implementation binding mismatch");
          request["schema"] = "operator-harness-request-v2";
          request["command"] = "run";
          request["request_id"] = case_id;
          request["frame_path"] = (directory / "frame.ngf").string();
          atomic(directory / "REQUEST.json", request.dump() + "\n");
          const auto before = sha(read(request.at("input_path").get<std::string>()));
          require(before == item.at("sample_sha256").get<std::string>(),
                  "sample SHA256 changed before execution");
          result["input_sha256"] = before;
          result["sample_sha256"] = item.at("sample_sha256");
          const auto child = execute(binary, directory / "REQUEST.json", directory / "RUN_RESPONSE.json",
                                     directory / "run.stderr.log", timeout_ms, max_rss_mib);
          result["native_run_pid"] = child.pid;
          result["native_run_exit_code"] = child.exit_code;
          result["native_run_timed_out"] = child.timed_out;
          if (child.timed_out) { result["status"] = "timeout"; throw std::runtime_error("native run deadline exceeded"); }
          require(child.exit_code == 0, "native run exited " + std::to_string(child.exit_code));
          const auto measured = response(directory / "RUN_RESPONSE.json", case_id, "run");
          result["native_run"] = measured;
          if (measured.at("status") != "ok") {
            result["status"] = measured.value("status", "failed");
            result["reason"] = measured.value("reason", measured.value("error", J("native run rejected")));
            if (measured.contains("reason_code")) result["reason_code"] = measured.at("reason_code");
          } else {
            require(measured.at("in_process_bitexact") == true && measured.at("input_sha256") == before,
                    "native run exactness/input evidence mismatch");
            const auto frame = directory / "frame.ngf";
            const auto frame_hash = sha(read(frame));
            require(measured.at("frame_sha256") == frame_hash && measured.at("frame_bytes") == fs::file_size(frame),
                    "native frame byte accounting mismatch");
            const auto repetitions = request.value("repeats", 3U);
            samples(measured.at("encode_ns"), repetitions, "encode");
            samples(measured.at("decode_ns"), repetitions, "decode");
            J cold{{"schema", "operator-harness-request-v2"}, {"command", "decode"},
                   {"request_id", case_id}, {"implementation_ids", {id}}, {"frame_path", frame.string()},
                   {"catalog_sha256", request.at("catalog_sha256")},
                   {"output_path", (directory / "restored.bin").string()}, {"warmups", 0}, {"repeats", 1}};
            atomic(directory / "COLD_REQUEST.json", cold.dump() + "\n");
            const auto decoder = execute(binary, directory / "COLD_REQUEST.json", directory / "COLD_RESPONSE.json",
                                         directory / "decode.stderr.log", timeout_ms, max_rss_mib);
            result["native_decode_pid"] = decoder.pid;
            result["native_decode_exit_code"] = decoder.exit_code;
            result["native_decode_timed_out"] = decoder.timed_out;
            require(child.pid != decoder.pid, "fresh decoder process identity unexpectedly reused");
            if (decoder.timed_out) { result["status"] = "timeout"; throw std::runtime_error("native decode deadline exceeded"); }
            require(decoder.exit_code == 0, "fresh decoder exited " + std::to_string(decoder.exit_code));
            const auto decoded = response(directory / "COLD_RESPONSE.json", case_id, "decode");
            require(decoded.at("status") == "ok" && decoded.at("output_sha256") == before,
                    "fresh-process decoder mismatch");
            require(read(directory / "restored.bin") == read(request.at("input_path").get<std::string>()),
                    "independent original/restored byte comparison failed");
            samples(decoded.at("decode_ns"), 1, "fresh decode");
            durable_artifact(frame);
            durable_artifact(directory / "restored.bin");
            result.update({{"status", "ok"}, {"cold_decode_verified", true},
                           {"input_sha256", before}, {"frame_sha256", frame_hash},
                           {"frame_path", frame.string()}, {"output_path", (directory / "restored.bin").string()},
                           {"native_cold_decode", decoded}});
          }
        }
      } catch (const std::exception& error) { result["error"] = error.what(); }
      result["receipt_sha256"] = sha(result.dump());
      atomic_json(receipt, result);
      const auto id = result.value("implementation_id", "");
      if (id.size() == 4 && id[0] == 'A' && std::all_of(id.begin() + 1, id.end(), ::isdigit))
        immutable(output / "operators" / id / (key + ".json"),
                  {{"implementation_id", id}, {"case_id", case_id}, {"status", result.at("status")},
                   {"case_receipt", receipt.string()}, {"receipt_sha256", result.at("receipt_sha256")}});
      const auto state = result.at("status").get<std::string>();
      counts[state] = counts.value(state, 0) + 1;
      outcomes.push_back({{"case_id", case_id}, {"implementation_id", id}, {"status", state}, {"receipt", receipt.string()}});
      event(output / "events.jsonl", {{"event", "case_committed"}, {"case_id", case_id}, {"status", state}});
      std::cout << J{{"case_id", case_id}, {"status", state}}.dump() << '\n' << std::flush;
    }
    atomic_json(output / "SUMMARY.json", {{"schema", "operator-harness-native-batch-summary-v2"},
                {"run_sha256", identity}, {"status_counts", counts}, {"case_count", outcomes.size()}, {"cases", outcomes}});
    close(lock);
    return 0; // A complete result ledger may contain explicit gaps/failures.
  } catch (const std::exception& error) {
    std::cerr << "HARNESS_BATCH_FATAL: " << error.what() << '\n';
    return 1;
  }
}
