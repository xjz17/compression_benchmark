#pragma once

#include <memory>
#include <string>
#include <valarray>

namespace prediction_probability {

struct Options {
  std::string provider;
  unsigned ppmd_order = 6;
  unsigned ppmd_memory_mib = 16;
  unsigned paq_level = 0;
  bool allow_large_memory = false;
};

void validate_options(const Options& options);

class ProbabilityModel {
 public:
  explicit ProbabilityModel(const Options& options);
  ~ProbabilityModel();
  ProbabilityModel(const ProbabilityModel&) = delete;
  ProbabilityModel& operator=(const ProbabilityModel&) = delete;

  double predict();
  void observe(unsigned bit);
  const std::valarray<float>& byte_distribution();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace prediction_probability
