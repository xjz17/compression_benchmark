#include "algorithm_allow_list.hpp"

#include "active_method_policy.hpp"
#include "native_lz_family.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace algorithm_allow_list {
namespace {

bool is_known_non_lzbench_row(const std::string& algo) {
  const auto& native_algorithms = native_lz_family_algorithms();
  if (std::find(native_algorithms.begin(), native_algorithms.end(), algo) !=
      native_algorithms.end()) {
    return true;
  }
  static const char* kExact[] = {
      "ts_2diff_bos",
      "TS_2DIFF", "RLE", "GORILLA", "CHIMP", "CHIMP128", "Pongo", "SPRINTZ", "RLBE",
      "GZIP", "SNAPPY", "LZ4", "BROTLI", "brotli", "ans", "huffman", "arithmetic",
      "fpzip", "fastpfor", "fpc", "pfpc", "most_minimal", "corad_tricklet",
      "streamvbyte_delta1", "simple8b", "leco", "alp", "elf_on_chimp", "elf_plus",
      "camel", "blosc_blosclz", "blosc2_blosclz", "ext_cmix", "buff", "buff_simd",
      "buff_kernel_native", "subcolumn", "grouping",
      "reger", "DELTA_BINARY_PACKED", "gz", "snappy-framed",
      "lz4-framed",
      "density", "gipfeli", "glza", "lzmat", "lzo1", "lzo1a", "lzo1b",
      "lzo1c", "lzo1f", "lzo1y", "lzo1z", "lzo2a", "lzrw", "lzsse2",
      "lzsse4", "lzsse4fast", "lzsse8", "lzsse8fast", "lzvn", "pithy",
      "shrinker", "ucl_nrv2d", "ucl_nrv2e", "wflz", "xpack", "yalz77",
      "yappy",
  };
  for (const char* name : kExact) {
    if (algo == name) {
      return true;
    }
  }
  if (algo.rfind("col_", 0) == 0 || algo.rfind("delta", 0) == 0 || algo.rfind("sz_", 0) == 0 ||
      algo.rfind("sz3_", 0) == 0 || algo.rfind("serf_", 0) == 0 || algo.rfind("spdp_", 0) == 0 ||
      algo.rfind("ext_", 0) == 0 || algo.rfind("machete_", 0) == 0 ||
      algo.rfind("simpiece_", 0) == 0 || algo.rfind("fpcompress_", 0) == 0 ||
      algo.rfind("ext_zfp_", 0) == 0 || algo.rfind("xz_level_", 0) == 0 ||
      algo.rfind("lzma_level_", 0) == 0 ||
      algo.rfind("zstd_level_", 0) == 0) {
    return true;
  }
  return false;
}

bool might_be_lzbench_catalog_row(const std::string& algo) {
  if (algo.empty()) {
    return false;
  }
  if (algo.rfind("lzbench_", 0) == 0) {
    return !is_known_non_lzbench_row(algo.substr(std::string("lzbench_").size()));
  }
  if (is_known_non_lzbench_row(algo)) {
    return false;
  }
  if (algo == "lzfse" || algo == "lzjb") {
    return true;
  }
  if (algo.size() == 4 && algo[0] == 'b' && algo[1] == 's' && algo[2] == 'c' &&
      algo[3] >= '0' && algo[3] <= '2') {
    return true;
  }
  bool has_us = false;
  for (char c : algo) {
    if (c == '_') {
      has_us = true;
      break;
    }
  }
  return has_us;
}

}  // namespace

std::vector<std::string> parse_csv_list(const char* raw) {
  std::vector<std::string> out;
  if (raw == nullptr) {
    return out;
  }
  std::string text(raw);
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, ',')) {
    while (!part.empty() && (part.front() == ' ' || part.front() == '\t')) {
      part.erase(part.begin());
    }
    while (!part.empty() && (part.back() == ' ' || part.back() == '\t')) {
      part.pop_back();
    }
    if (!part.empty()) {
      out.push_back(part);
    }
  }
  return out;
}

std::unordered_set<std::string> from_allowlist_env() {
  const char* raw = std::getenv(kAllowlistEnvKey);
  std::unordered_set<std::string> out;
  for (const auto& part : parse_csv_list(raw)) {
    out.insert(part);
    const std::string lzma_prefix = "lzma_level_";
    if (part.rfind(lzma_prefix, 0) == 0) {
      out.insert("xz_level_" + part.substr(lzma_prefix.size()));
    }
  }
  return out;
}

std::unordered_set<std::string> from_incremental_skip_env() {
  const char* raw = std::getenv(kIncrementalSkipEnvKey);
  std::unordered_set<std::string> out;
  for (const auto& part : parse_csv_list(raw)) {
    out.insert(part);
  }
  return out;
}

bool has_allowlist() { return !from_allowlist_env().empty(); }

bool should_skip(const std::unordered_set<std::string>* incremental_skip,
                 const std::string& algorithm) {
  if (!active_method_policy::is_active_algorithm_key(algorithm)) {
    return true;
  }
  const auto allowlist = from_allowlist_env();
  if (!allowlist.empty()) {
    return algorithm.empty() || allowlist.count(algorithm) == 0;
  }
  return incremental_skip != nullptr && incremental_skip->count(algorithm) > 0;
}

bool should_run(const std::unordered_set<std::string>* incremental_skip, const char* algorithm) {
  if (algorithm == nullptr) {
    return false;
  }
  return !should_skip(incremental_skip, std::string(algorithm));
}

bool should_run(const std::unordered_set<std::string>* incremental_skip,
                const std::string& algorithm) {
  return !should_skip(incremental_skip, algorithm);
}

bool allowlist_may_want_lzbench(const std::unordered_set<std::string>* incremental_skip) {
  if (!has_allowlist()) {
    return true;
  }
  for (const auto& algo : from_allowlist_env()) {
    if (should_run(incremental_skip, algo) && might_be_lzbench_catalog_row(algo)) {
      return true;
    }
  }
  return false;
}

void set_allowlist_env(const std::string& csv_list) {
#ifdef _WIN32
  _putenv_s(kAllowlistEnvKey, csv_list.c_str());
#else
  setenv(kAllowlistEnvKey, csv_list.c_str(), 1);
#endif
}

}  // namespace algorithm_allow_list
