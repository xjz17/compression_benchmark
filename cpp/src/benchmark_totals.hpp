#pragma once

#include "benchmark_loss.hpp"

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

/** Accumulates originalSize, compressedSize, compressionTime, decompressionTime, and loss across patches. */
class BenchmarkTotals {
 public:
  void merge(const std::string& algorithm, const std::vector<long long>& result) {
    merge(algorithm, result, 0.0, 0.0);
  }

  void merge(const std::string& algorithm, const std::vector<long long>& result, double loss_err,
             double loss_ref) {
    if (algorithm.empty() || result.size() < 4 || result[0] <= 0) {
      return;
    }
    merge(algorithm, result[0], result[1], result[2], result[3], loss_err, loss_ref);
  }

  void merge(const std::string& algorithm, long long original, long long compressed,
             long long comp_time, long long decomp_time, double loss_err = 0.0,
             double loss_ref = 0.0) {
    if (original <= 0) {
      return;
    }
    auto& acc = by_algorithm_[algorithm];
    if (acc.size() < 4) {
      acc.assign(4, 0);
    }
    acc[0] += original;
    acc[1] += compressed;
    acc[2] += comp_time;
    acc[3] += decomp_time;
    if (loss_err != 0.0 || loss_ref != 0.0) {
      auto& loss = loss_by_algorithm_[algorithm];
      loss.err += loss_err;
      loss.ref += loss_ref;
    }
  }

  void merge(const std::string& algorithm, const std::vector<long long>& result,
             const LossAccum& loss) {
    merge(algorithm, result, loss.err, loss.ref);
  }

  void merge(const std::string& algorithm, long long original, long long compressed,
             long long comp_time, long long decomp_time, const LossAccum& loss) {
    merge(algorithm, original, compressed, comp_time, decomp_time, loss.err, loss.ref);
  }

  const std::unordered_map<std::string, std::vector<long long>>& snapshot() const {
    return by_algorithm_;
  }

  int is_lossy(const std::string& algorithm) const {
    const auto it = loss_by_algorithm_.find(algorithm);
    const double sum_err = it == loss_by_algorithm_.end() ? 0.0 : it->second.err;
    const double sum_ref = it == loss_by_algorithm_.end() ? 0.0 : it->second.ref;
    return benchmark_loss::is_lossy_flag(algorithm, sum_err, sum_ref);
  }

  double loss_rate(const std::string& algorithm) const {
    const auto it = loss_by_algorithm_.find(algorithm);
    if (it == loss_by_algorithm_.end()) {
      return 0.0;
    }
    return benchmark_loss::loss_rate(it->second.err, it->second.ref);
  }

 private:
  std::unordered_map<std::string, std::vector<long long>> by_algorithm_;
  std::unordered_map<std::string, LossAccum> loss_by_algorithm_;
};

inline void merge_report_stream(BenchmarkTotals& totals, std::istream& in) {
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    if (line.rfind("algorithm", 0) == 0) {
      continue;
    }
    const std::size_t c1 = line.find(',');
    if (c1 == std::string::npos || c1 == 0) {
      continue;
    }
    const std::string algo = line.substr(0, c1);
    std::vector<long long> vals;
    std::size_t pos = c1 + 1;
    for (int i = 0; i < 4; ++i) {
      const std::size_t c = line.find(',', pos);
      const std::string field =
          (c == std::string::npos) ? line.substr(pos) : line.substr(pos, c - pos);
      try {
        vals.push_back(std::stoll(field));
      } catch (...) {
        vals.clear();
        break;
      }
      if (c == std::string::npos) {
        break;
      }
      pos = c + 1;
    }
    if (vals.size() >= 4) {
      double loss_err = 0.0;
      double loss_ref = 0.0;
      const std::size_t c5 = line.find(',', pos);
      if (c5 != std::string::npos) {
        const std::size_t c6 = line.find(',', c5 + 1);
        const std::string err_field = line.substr(pos, c5 - pos);
        const std::string ref_field =
            (c6 == std::string::npos) ? line.substr(c5 + 1) : line.substr(c5 + 1, c6 - c5 - 1);
        try {
          loss_err = std::stod(err_field);
          if (!ref_field.empty()) {
            loss_ref = std::stod(ref_field);
          }
        } catch (...) {
          loss_err = 0.0;
          loss_ref = 0.0;
        }
      }
      totals.merge(algo, vals, loss_err, loss_ref);
    }
  }
}
