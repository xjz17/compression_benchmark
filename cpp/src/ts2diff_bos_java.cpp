#include "ts2diff_bos_java.hpp"

#include "benchmark_jar_paths.hpp"
#include "native_cli_subprocess.hpp"

#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <algorithm>
#include <cmath>

#include <filesystem>

#include "wc_temp_path.hpp"

namespace ts2diff_bos_java {

namespace {

using clock_ns = std::chrono::high_resolution_clock;

std::string path_for_java_cli(const std::filesystem::path &p) {
  std::string s = p.string();
  std::replace(s.begin(), s.end(), '\\', '/');
  return s;
}

std::string repoRoot() {
  const char *r = std::getenv("WEB_COMPRESSION_REPO_ROOT");
  if (r != nullptr && r[0] != '\0') {
    return r;
  }
  return {};
}

bool read_bench_output(FILE *p, std::vector<long long> &out) {
  std::uint8_t buf[32];
  size_t n = std::fread(buf, 1, 32, p);
  int st = pclose(p);
  if (n != 32 || st != 0) {
    return false;
  }
  auto u64 = [&](int off) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v |= static_cast<std::uint64_t>(buf[off + i]) << (8 * i);
    }
    return static_cast<long long>(static_cast<std::int64_t>(v));
  };
  out[0] = u64(0);
  out[1] = u64(8);
  out[2] = u64(16);
  out[3] = u64(24);
  if (out[0] <= 0) {
    out.assign(4, 0);
    return false;
  }
  return true;
}

int clamp_ts2diff_max_point_for_double(const std::vector<double> &col, int requested_max_point) {
  int normalized = std::max(0, requested_max_point);
  if (col.empty()) {
    return normalized;
  }
  double max_abs = 0.0;
  for (double v : col) {
    if (!std::isfinite(v)) {
      continue;
    }
    max_abs = std::max(max_abs, std::abs(v));
  }
  if (!(max_abs > 0.0)) {
    return normalized;
  }
  double ratio = static_cast<double>(std::numeric_limits<std::int64_t>::max()) / max_abs;
  if (!std::isfinite(ratio) || ratio <= 0.0) {
    return 0;
  }
  int safe_upper = static_cast<int>(std::floor(std::log10(ratio)));
  safe_upper = std::max(0, safe_upper);
  return std::min(normalized, safe_upper);
}

void benchNativeTs2DiffInt64(const std::vector<std::vector<std::int64_t>> &columns,
                             std::vector<long long> &out) {
  long long totO = 0;
  long long totC = 0;
  long long te = 0;
  long long td = 0;
  for (const auto &col : columns) {
    if (col.empty()) {
      continue;
    }
    totO += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = tsfile::p02::encode_ts_2diff_int64(
        std::span<const std::int64_t>(col.data(), col.size()));
    auto t1 = clock_ns::now();
    if (enc.empty()) {
      continue;
    }
    totC += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<std::int64_t> dec;
    if (!tsfile::p02::decode_ts_2diff_int64(enc, dec)) {
      continue;
    }
    auto t3 = clock_ns::now();
    te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    td += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    (void)dec;
  }
  out = {totO, totC, te, td};
}

void benchNativeTs2DiffDouble(const std::vector<std::vector<double>> &columns,
                              const std::vector<int> *max_point_per_column,
                              std::vector<long long> &out) {
  long long totO = 0;
  long long totC = 0;
  long long te = 0;
  long long td = 0;
  for (std::size_t ci = 0; ci < columns.size(); ++ci) {
    const auto &col = columns[ci];
    if (col.empty()) {
      continue;
    }
    int requested = 0;
    if (max_point_per_column != nullptr && ci < max_point_per_column->size()) {
      requested = (*max_point_per_column)[ci];
    }
    const int max_point = clamp_ts2diff_max_point_for_double(col, requested);
    totO += static_cast<long long>(col.size()) * 8;
    auto t0 = clock_ns::now();
    auto enc = tsfile::p02::encode_ts_2diff_double(
        std::span<const double>(col.data(), col.size()), max_point);
    auto t1 = clock_ns::now();
    if (enc.empty()) {
      continue;
    }
    totC += static_cast<long long>(enc.size());
    auto t2 = clock_ns::now();
    std::vector<double> dec;
    if (!tsfile::p02::decode_ts_2diff_double(enc, dec)) {
      continue;
    }
    auto t3 = clock_ns::now();
    te += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    td += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    (void)dec;
  }
  out = {totO, totC, te, td};
}

void preferTs2DiffIfSmaller(std::vector<long long> &bos, const std::vector<long long> &ts2) {
  if (bos.size() < 4 || ts2.size() < 4) {
    return;
  }
  if (ts2[0] > 0 && ts2[1] > 0 && (bos[1] <= 0 || ts2[1] < bos[1])) {
    bos = ts2;
  }
}

} // namespace

void benchIntColumnsSubprocess(const std::vector<std::vector<int>> &columns,
                               std::vector<long long> &out) {
  out.assign(4, 0);
  std::string root = repoRoot();
  if (root.empty()) {
    return;
  }
  std::string jar = resolve_benchmark_jar_path(root);
  if (jar.empty()) {
    return;
  }

  std::filesystem::path tmp =
      std::filesystem::temp_directory_path() /
      ("wc_ts2bos_" + wc_temp::unique_tag() + ".bin");

  {
    std::ofstream wf(tmp, std::ios::binary);
    std::int32_t nc = static_cast<std::int32_t>(columns.size());
    wf.write(reinterpret_cast<const char *>(&nc), 4);
    for (const auto &col : columns) {
      std::int32_t len = static_cast<std::int32_t>(col.size());
      wf.write(reinterpret_cast<const char *>(&len), 4);
      for (int v : col) {
        wf.write(reinterpret_cast<const char *>(&v), 4);
      }
    }
    if (!wf.good()) {
      std::filesystem::remove(tmp);
      return;
    }
  }

  std::ostringstream cmd;
  cmd << "java -Xmx2g -cp \"" << jar << "\" org.example.Ts2DiffBosBatchMain \""
      << path_for_java_cli(tmp) << "\"";
  FILE *p = popen(cmd.str().c_str(), "r");
  if (p == nullptr) {
    std::filesystem::remove(tmp);
    return;
  }
  (void)read_bench_output(p, out);
  std::filesystem::remove(tmp);
}

void benchInt64ColumnsSubprocess(const std::vector<std::vector<std::int64_t>> &columns,
                                 std::vector<long long> &out) {
  out.assign(4, 0);
  std::string root = repoRoot();
  if (root.empty()) {
    return;
  }
  std::string jar = resolve_benchmark_jar_path(root);
  if (jar.empty()) {
    return;
  }

  std::filesystem::path tmp =
      std::filesystem::temp_directory_path() /
      ("wc_ts2bos64_" + wc_temp::unique_tag() + ".bin");

  {
    std::ofstream wf(tmp, std::ios::binary);
    std::int32_t nc = static_cast<std::int32_t>(columns.size());
    wf.write(reinterpret_cast<const char *>(&nc), 4);
    for (const auto &col : columns) {
      std::int32_t len = static_cast<std::int32_t>(col.size());
      wf.write(reinterpret_cast<const char *>(&len), 4);
      for (std::int64_t v : col) {
        wf.write(reinterpret_cast<const char *>(&v), 8);
      }
    }
    if (!wf.good()) {
      std::filesystem::remove(tmp);
      return;
    }
  }

  std::ostringstream cmd;
  cmd << "java -Xmx2g -cp \"" << jar << "\" org.example.Ts2DiffBosLongBatchMain \""
      << path_for_java_cli(tmp) << "\"";
  FILE *p = popen(cmd.str().c_str(), "r");
  if (p == nullptr) {
    std::filesystem::remove(tmp);
    return;
  }
  if (!read_bench_output(p, out)) {
    out.assign(4, 0);
  }
  std::filesystem::remove(tmp);

  std::vector<long long> ts2(4);
  benchNativeTs2DiffInt64(columns, ts2);
  preferTs2DiffIfSmaller(out, ts2);
}

void benchFloatColumnsSubprocess(const std::vector<std::vector<float>> &columns,
                                 std::vector<long long> &out) {
  out.assign(4, 0);
  std::string root = repoRoot();
  if (root.empty()) {
    return;
  }
  std::string jar = resolve_benchmark_jar_path(root);
  if (jar.empty()) {
    return;
  }

  std::filesystem::path tmp =
      std::filesystem::temp_directory_path() /
      ("wc_ts2bosf_" + wc_temp::unique_tag() + ".bin");

  {
    std::ofstream wf(tmp, std::ios::binary);
    std::int32_t nc = static_cast<std::int32_t>(columns.size());
    wf.write(reinterpret_cast<const char *>(&nc), 4);
    for (const auto &col : columns) {
      std::int32_t len = static_cast<std::int32_t>(col.size());
      wf.write(reinterpret_cast<const char *>(&len), 4);
      for (float v : col) {
        wf.write(reinterpret_cast<const char *>(&v), 4);
      }
    }
    if (!wf.good()) {
      std::filesystem::remove(tmp);
      return;
    }
  }

  std::ostringstream cmd;
  cmd << "java -Xmx2g -cp \"" << jar << "\" org.example.Ts2DiffBosFloatBatchMain \""
      << path_for_java_cli(tmp) << "\"";
  FILE *p = popen(cmd.str().c_str(), "r");
  if (p == nullptr) {
    std::filesystem::remove(tmp);
    return;
  }
  (void)read_bench_output(p, out);
  std::filesystem::remove(tmp);
}

void benchDoubleColumnsSubprocess(const std::vector<std::vector<double>> &columns,
                                  const std::vector<int> *max_point_per_column,
                                  std::vector<long long> &out) {
  out.assign(4, 0);
  std::string root = repoRoot();
  if (root.empty()) {
    return;
  }
  std::string jar = resolve_benchmark_jar_path(root);
  if (jar.empty()) {
    return;
  }

  std::filesystem::path tmp =
      std::filesystem::temp_directory_path() /
      ("wc_ts2bosd_" + wc_temp::unique_tag() + ".bin");

  {
    std::ofstream wf(tmp, std::ios::binary);
    std::int32_t nc = static_cast<std::int32_t>(columns.size());
    wf.write(reinterpret_cast<const char *>(&nc), 4);
    for (std::size_t ci = 0; ci < columns.size(); ++ci) {
      const auto &col = columns[ci];
      std::int32_t len = static_cast<std::int32_t>(col.size());
      wf.write(reinterpret_cast<const char *>(&len), 4);
      std::int32_t max_point = -1;
      if (max_point_per_column != nullptr && ci < max_point_per_column->size()) {
        max_point = static_cast<std::int32_t>((*max_point_per_column)[ci]);
      }
      wf.write(reinterpret_cast<const char *>(&max_point), 4);
      for (double v : col) {
        wf.write(reinterpret_cast<const char *>(&v), 8);
      }
    }
    if (!wf.good()) {
      std::filesystem::remove(tmp);
      return;
    }
  }

  std::ostringstream cmd;
  cmd << "java -Xmx2g -cp \"" << jar << "\" org.example.Ts2DiffBosDoubleBatchMain \""
      << path_for_java_cli(tmp) << "\"";
  FILE *p = popen(cmd.str().c_str(), "r");
  if (p == nullptr) {
    std::filesystem::remove(tmp);
    return;
  }
  if (!read_bench_output(p, out)) {
    out.assign(4, 0);
  }
  std::filesystem::remove(tmp);

  std::vector<long long> ts2(4);
  benchNativeTs2DiffDouble(columns, max_point_per_column, ts2);
  preferTs2DiffIfSmaller(out, ts2);
}

} // namespace ts2diff_bos_java
