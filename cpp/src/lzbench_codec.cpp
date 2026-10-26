#include "lzbench_codec.hpp"

#include "algorithm_allow_list.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace wc_lzbench_bench {

namespace {

bool env_bench_on() {
  if (const char *env = std::getenv("WEB_COMPRESSION_LZBENCH_BENCH")) {
    return env[0] != '0';
  }
  // lzbench is an external subprocess path; keep it opt-in for native tri-language runs.
  return false;
}

std::string unique_tag() {
  static std::mt19937_64 rng{std::random_device{}()};
  std::ostringstream o;
  o << std::hex << rng();
  return o.str();
}

fs::path repo_root() {
  if (const char *r = std::getenv("WEB_COMPRESSION_REPO_ROOT"); r != nullptr && r[0] != '\0') {
    return fs::path(r);
  }
  return {};
}

fs::path resolve_patch_cli_script() {
  const fs::path root = repo_root();
  if (root.empty()) {
    return {};
  }
  const fs::path script = root / "scripts" / "lzbench_patch_cli.py";
  return fs::is_regular_file(script) ? script : fs::path{};
}

std::string resolve_python() {
  if (const char *py = std::getenv("WEB_COMPRESSION_PYTHON"); py != nullptr && py[0] != '\0') {
    return py;
  }
#ifdef _WIN32
  return "python";
#else
  return "python3";
#endif
}

bool write_file_bytes(const fs::path &path, const void *data, std::size_t nbytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  return static_cast<bool>(out.write(static_cast<const char *>(data),
                                     static_cast<std::streamsize>(nbytes)));
}

std::vector<std::string> split_csv_simple(const std::string &line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

bool run_patch_cli(const fs::path &script, const fs::path &payload_path,
                   std::vector<std::vector<std::string>> &rows_out) {
  const fs::path root = repo_root();
  if (root.empty()) {
    return false;
  }
  const fs::path tmp_dir = fs::temp_directory_path();
  const fs::path out_csv = tmp_dir / ("wc_lzbench_out_" + unique_tag() + ".csv");

  std::ostringstream cmd;
#ifdef _WIN32
  cmd << "cmd /c \"set WEB_COMPRESSION_REPO_ROOT=" << root.string()
      << "&& \"" << resolve_python() << "\" \"" << script.string() << "\" \""
      << payload_path.string() << "\" > \"" << out_csv.string() << "\" 2>nul\"";
#else
  cmd << "WEB_COMPRESSION_REPO_ROOT=\"" << root.string() << "\" "
      << resolve_python() << " \"" << script.string() << "\" \""
      << payload_path.string() << "\" > \"" << out_csv.string() << "\" 2>/dev/null";
#endif
  const int st = std::system(cmd.str().c_str());
  if (st != 0 && !fs::is_regular_file(out_csv)) {
    std::error_code ec;
    fs::remove(out_csv, ec);
    return false;
  }

  std::ifstream in(out_csv);
  if (!in) {
    std::error_code ec;
    fs::remove(out_csv, ec);
    return false;
  }
  rows_out.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto parts = split_csv_simple(line);
    if (parts.size() >= 5) {
      rows_out.push_back(std::move(parts));
    }
  }
  std::error_code ec;
  fs::remove(out_csv, ec);
  return !rows_out.empty();
}

}  // namespace

bool bench_enabled() { return false; }

void accumulatePatchBytes(const std::vector<char> &payload, BenchmarkTotals &totals,
                          const std::unordered_set<std::string> *skip_algorithms) {
  if (!bench_enabled() || payload.empty()) {
    return;
  }
  if (!algorithm_allow_list::allowlist_may_want_lzbench(skip_algorithms)) {
    return;
  }
  const fs::path script = resolve_patch_cli_script();
  if (script.empty()) {
    return;
  }
  const fs::path tmp_in =
      fs::temp_directory_path() / ("wc_lzbench_in_" + unique_tag() + ".bin");
  if (!write_file_bytes(tmp_in, payload.data(), payload.size())) {
    return;
  }
  std::vector<std::vector<std::string>> rows;
  if (!run_patch_cli(script, tmp_in, rows)) {
    std::error_code ec;
    fs::remove(tmp_in, ec);
    return;
  }
  std::error_code ec;
  fs::remove(tmp_in, ec);

  for (const auto &fields : rows) {
    if (fields.size() < 5) {
      continue;
    }
    if (algorithm_allow_list::should_skip(skip_algorithms, fields[0])) {
      continue;
    }
    try {
      const long long orig = std::stoll(fields[1]);
      const long long comp = std::stoll(fields[2]);
      const long long ct = std::stoll(fields[3]);
      const long long dt = std::stoll(fields[4]);
      if (orig <= 0 || dt <= 0) {
        continue;
      }
      totals.merge(fields[0], orig, comp, ct, dt, LossAccum{});
    } catch (...) {
      continue;
    }
  }
}

void accumulateByteColumns(const std::vector<std::vector<char>> &column_byte_data,
                           BenchmarkTotals &totals,
                           const std::unordered_set<std::string> *skip_algorithms) {
  if (!algorithm_allow_list::allowlist_may_want_lzbench(skip_algorithms)) {
    return;
  }
  for (const auto &col : column_byte_data) {
    accumulatePatchBytes(col, totals, skip_algorithms);
  }
}

}  // namespace wc_lzbench_bench
