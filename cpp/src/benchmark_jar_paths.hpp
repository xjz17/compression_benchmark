#pragma once

#include <cstdlib>
#include <fstream>
#include <string>

/** Resolve shaded benchmark JAR: env override, then java/target, then backend/lib. */
inline std::string resolve_benchmark_jar_path(const std::string &repo_root) {
  const char *override_jar = std::getenv("WEB_COMPRESSION_BENCHMARK_JAR");
  if (override_jar != nullptr && override_jar[0] != '\0') {
    if (std::ifstream(override_jar).good()) {
      return std::string(override_jar);
    }
  }
  const std::string target_jar = repo_root + "/java/target/benchmark-java-1.0-SNAPSHOT.jar";
  if (std::ifstream(target_jar).good()) {
    return target_jar;
  }
  const std::string backend_jar = repo_root + "/backend/lib/benchmark-java-1.0-SNAPSHOT.jar";
  if (std::ifstream(backend_jar).good()) {
    return backend_jar;
  }
  return {};
}
