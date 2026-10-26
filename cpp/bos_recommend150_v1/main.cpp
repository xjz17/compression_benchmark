#include "recommend150.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using bos_recommend150_v1::FinalRequest;
using bos_recommend150_v1::TuneRequest;

namespace {

[[noreturn]] void usage() {
  throw std::runtime_error(
      "usage:\n"
      "  bos_recommend150 tune --features F --split S --cohorts C "
      "--candidates K --development-feed D "
      "--out DIR [--expected-samples 3433] [--expected-methods 85]\n"
      "  bos_recommend150 final-eval --features F --split S --cohorts C "
      "--candidates K --feed D "
      "--models DIR --partition confirm|external-final --out REPORT "
      "[--expected-samples N] [--expected-methods 85] [--timing-repeats 1000]");
}

std::map<std::string, std::string> flags(int argc, char** argv, int begin) {
  std::map<std::string, std::string> result;
  for (int i = begin; i < argc; i += 2) {
    if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0) usage();
    if (!result.emplace(argv[i] + 2, argv[i + 1]).second)
      throw std::runtime_error("duplicate flag " + std::string(argv[i]));
  }
  return result;
}

std::string take(const std::map<std::string, std::string>& values,
                 const std::string& key) {
  const auto it = values.find(key);
  if (it == values.end() || it->second.empty())
    throw std::runtime_error("missing --" + key);
  return it->second;
}

std::size_t count(const std::map<std::string, std::string>& values,
                  const std::string& key, std::size_t fallback) {
  const auto it = values.find(key);
  if (it == values.end()) return fallback;
  std::size_t used = 0;
  const auto value = std::stoull(it->second, &used);
  if (used != it->second.size()) throw std::runtime_error("invalid --" + key);
  return value;
}

void reject_unknown(const std::map<std::string, std::string>& values,
                    const std::initializer_list<const char*>& allowed) {
  for (const auto& [key, unused] : values) {
    (void)unused;
    bool found = false;
    for (const char* item : allowed) found = found || key == item;
    if (!found) throw std::runtime_error("unknown flag --" + key);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) usage();
    const std::string command(argv[1]);
    const auto f = flags(argc, argv, 2);
    if (command == "tune") {
      reject_unknown(f, {"features", "split", "cohorts", "candidates",
                         "development-feed", "out",
                         "expected-samples", "expected-methods"});
      TuneRequest request;
      request.features_jsonl = take(f, "features");
      request.split_json = take(f, "split");
      request.cohorts_json = take(f, "cohorts");
      request.candidates_json = take(f, "candidates");
      request.development_feed_jsonl = take(f, "development-feed");
      request.output_directory = take(f, "out");
      request.expected_samples = count(f, "expected-samples", 3433);
      request.expected_methods = count(f, "expected-methods", 85);
      std::cout << bos_recommend150_v1::tune(request).dump(2) << '\n';
      return EXIT_SUCCESS;
    }
    if (command == "final-eval") {
      reject_unknown(f, {"features", "split", "cohorts", "candidates",
                         "feed", "models", "partition",
                         "out", "expected-samples", "expected-methods",
                         "timing-repeats"});
      FinalRequest request;
      request.features_jsonl = take(f, "features");
      request.split_json = take(f, "split");
      request.cohorts_json = take(f, "cohorts");
      request.candidates_json = take(f, "candidates");
      request.feed_jsonl = take(f, "feed");
      request.model_directory = take(f, "models");
      request.output_json = take(f, "out");
      request.partition = take(f, "partition");
      request.expected_samples = count(
          f, "expected-samples", request.partition == "confirm" ? 86 : 125);
      request.expected_methods = count(f, "expected-methods", 85);
      request.timing_repeats = count(f, "timing-repeats", 1000);
      std::cout << bos_recommend150_v1::final_evaluate(request).dump(2)
                << '\n';
      return EXIT_SUCCESS;
    }
    usage();
  } catch (const std::exception& error) {
    std::cerr << "bos_recommend150_v1: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
