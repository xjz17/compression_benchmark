#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace algorithm_allow_list {

inline constexpr const char* kAllowlistEnvKey = "COMPRESSION_BENCHMARK_RUN_ALGORITHMS";
inline constexpr const char* kIncrementalSkipEnvKey = "COMPRESSION_BENCHMARK_SKIP_ALGORITHMS";

std::vector<std::string> parse_csv_list(const char* raw);

std::unordered_set<std::string> from_allowlist_env();

std::unordered_set<std::string> from_incremental_skip_env();

bool has_allowlist();

/** True when the algorithm should be skipped (not benchmarked / not flushed). */
bool should_skip(const std::unordered_set<std::string>* incremental_skip, const std::string& algorithm);

bool should_run(const std::unordered_set<std::string>* incremental_skip, const char* algorithm);

bool should_run(const std::unordered_set<std::string>* incremental_skip, const std::string& algorithm);

bool allowlist_may_want_lzbench(const std::unordered_set<std::string>* incremental_skip);

void set_allowlist_env(const std::string& csv_list);

}  // namespace algorithm_allow_list
