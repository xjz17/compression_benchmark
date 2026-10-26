#include "elf_double_java.hpp"

#include "benchmark_jar_paths.hpp"
#include "native_cli_subprocess.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <filesystem>

#include "wc_temp_path.hpp"

namespace elf_double_java {

namespace {

std::string repoRoot() {
  const char *r = std::getenv("WEB_COMPRESSION_REPO_ROOT");
  if (r != nullptr && r[0] != '\0') {
    return r;
  }
  return ".";
}

std::string elfJar(const std::string &root) {
  const char *override = std::getenv("WEB_COMPRESSION_ELF_JAR");
  if (override != nullptr && override[0] != '\0') {
    if (std::ifstream(override).good()) {
      return override;
    }
  }
  const char *candidates[] = {
      "/external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar",
      "/backend/lib/start-compress-1.0-SNAPSHOT.jar",
  };
  for (const char *rel : candidates) {
    std::string p = root + rel;
    if (std::ifstream(p).good()) {
      return p;
    }
  }
  return {};
}

std::string benchmarkOverlayJar(const std::string &root) {
  const std::string path =
      root + "/backend/lib/benchmark-java-chimp-overlay-20260724.jar";
  return std::ifstream(path).good() ? path : std::string{};
}

} // namespace

void benchDoubleColumnsSubprocess(const std::vector<std::vector<double>> &columns,
                                  const char *mode,
                                  std::vector<long long> &out) {
  out.assign(4, 0);
  const std::string root = repoRoot();
  if (mode == nullptr || mode[0] == '\0') {
    return;
  }
  const std::string bench = resolve_benchmark_jar_path(root);
  const std::string elf = elfJar(root);
  if (bench.empty() || elf.empty()) {
    return;
  }

  std::filesystem::path tmp =
      std::filesystem::temp_directory_path() /
      ("wc_elf64_" + wc_temp::unique_tag() + ".bin");

  {
    std::ofstream wf(tmp, std::ios::binary);
    std::int32_t nc = static_cast<std::int32_t>(columns.size());
    wf.write(reinterpret_cast<const char *>(&nc), 4);
    for (const auto &col : columns) {
      std::int32_t len = static_cast<std::int32_t>(col.size());
      wf.write(reinterpret_cast<const char *>(&len), 4);
      for (double v : col) {
        wf.write(reinterpret_cast<const char *>(&v), 8);
      }
    }
    if (!wf.good()) {
      std::filesystem::remove(tmp);
      return;
    }
  }

  const char cp_sep =
#ifdef _WIN32
      ';';
#else
      ':';
#endif
  std::ostringstream cp;
  const std::string overlay = benchmarkOverlayJar(root);
  if (!overlay.empty()) {
    cp << overlay << cp_sep;
  }
  cp << bench << cp_sep << elf;
  const std::filesystem::path depDir =
      std::filesystem::path(elf).parent_path() / "dependency";
  if (std::filesystem::is_directory(depDir)) {
    if (std::strcmp(mode, "chimp") == 0) {
      for (const char *name : {"dsiutils-2.7.0.jar", "fastutil-8.5.8.jar"}) {
        const auto dependency = depDir / name;
        if (std::filesystem::is_regular_file(dependency)) {
          cp << cp_sep << dependency.string();
        }
      }
    } else {
      for (const auto &ent : std::filesystem::directory_iterator(depDir)) {
        if (ent.path().extension() == ".jar") {
          cp << cp_sep << ent.path().string();
        }
      }
    }
  }

  std::string tmp_arg = tmp.string();
  std::replace(tmp_arg.begin(), tmp_arg.end(), '\\', '/');

  std::vector<std::string> jargs = {"-cp", cp.str(), "org.example.ElfDoubleBatchMain", tmp_arg,
                                  mode};
  const char *dbg = std::getenv("WEB_COMPRESSION_ELF_DEBUG");
  if (dbg != nullptr && dbg[0] == '1') {
    std::fprintf(stderr, "elf_double_java cmd: %s\n",
                 native_cli_subprocess::build_java_popen_command(jargs).c_str());
  }

  std::array<std::uint8_t, 32> buf{};
  int st = -1;
  const bool ok_io = native_cli_subprocess::run_java_capture_32(jargs, buf, st);
  std::filesystem::remove(tmp);
  if (!ok_io) {
    if (dbg != nullptr && dbg[0] == '1') {
      std::fprintf(stderr, "elf_double_java: java failed exit=%d mode=%s\n", st, mode);
    }
    return;
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
  }
}

} // namespace elf_double_java
