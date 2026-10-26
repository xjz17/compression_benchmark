// Shared adapters to the actual cmix v21 model implementations.
// This is a GPL-3.0 combined work; preserve external/compression/cmix/COPYING
// and all upstream notices.
#include "prediction_probability_model.hpp"

#include "../../external/compression/cmix/src/models/paq8.cpp"
#include "../../external/compression/cmix/src/models/ppmd.cpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

// Same no-dictionary setting as upstream runner.cpp:17; required by FXCM.
char* dictionary_path = nullptr;

namespace prediction_probability {

void validate_options(const Options& o) {
  if (o.provider != "ppmd" && o.provider != "dmc" &&
      o.provider != "paq" && o.provider != "cmix")
    throw std::runtime_error("unknown/unavailable provider: " + o.provider);
  if (o.ppmd_order < 2 || o.ppmd_order > 64)
    throw std::runtime_error("PPMd order must be 2..64");
  if (o.ppmd_memory_mib < 4 || o.ppmd_memory_mib > 256)
    throw std::runtime_error("PPMd memory must be 4..256 MiB");
  if (o.paq_level > 3)
    throw std::runtime_error("bounded PAQ adapter permits levels 0..3 only");
  if (o.provider == "cmix" && !o.allow_large_memory)
    throw std::runtime_error(
        "cmix full predictor requires --allow-large-memory; upstream recommends at least 32 GB RAM");
}

struct ProbabilityModel::Impl {
  explicit Impl(const Options& o) : provider(o.provider), vocab(256, true) {
    validate_options(o);
    if (provider == "ppmd") {
      ppmd.reset(new PPMD::PPMD(o.ppmd_order, o.ppmd_memory_mib, byte, vocab));
    } else if (provider == "dmc") {
      // Exact upstream dmcModel count branch, not the ten-model DMC forest.
      dmc.reset(new paq8::dmcModel(65536, 1024));
    } else if (provider == "paq") {
      paq8::level = static_cast<int>(o.paq_level);
      paq8::buf.setsize(paq8::MEM() * 8);
      paq.reset(new paq8::Predictor());
    } else {
      cmix.reset(new Predictor(vocab));
    }
  }

  double predict() {
    if (has_prediction) return cached_prediction;
    if (ppmd) cached_prediction = ppmd->Predict()[0];
    else if (dmc) cached_prediction = dmc->pr1() / 4096.0;
    else if (paq) cached_prediction = paq->p() / 4096.0;
    else cached_prediction = cmix->Predict();
    if (!std::isfinite(cached_prediction) || cached_prediction < 0 || cached_prediction > 1)
      throw std::runtime_error("source predictor returned invalid probability");
    has_prediction = true;
    return cached_prediction;
  }

  void observe(unsigned bit) {
    if (bit > 1) throw std::runtime_error("observation must be 0 or 1");
    if (!has_prediction) throw std::runtime_error("predict must precede observe");
    if (ppmd) {
      ppmd->Perceive(bit);
      byte = (byte << 1) | bit;
      if (++bits_in_byte == 8) {
        ppmd->ByteUpdate();
        byte = 0;
        bits_in_byte = 0;
      }
    } else if (dmc) {
      paq8::y = bit;
      dmc->update();
    } else if (paq) {
      paq8::y = bit;
      paq->update();
    } else {
      cmix->Perceive(bit);
    }
    has_prediction = false;
  }

  const std::valarray<float>& byte_distribution() {
    if (!ppmd) throw std::runtime_error("byte distribution exists only for ppmd");
    if (bits_in_byte != 0) throw std::runtime_error("byte distribution requires a byte boundary");
    return ppmd->BytePredict();
  }

  std::string provider;
  std::vector<bool> vocab;
  unsigned byte = 0;
  unsigned bits_in_byte = 0;
  bool has_prediction = false;
  double cached_prediction = 0.5;
  std::unique_ptr<PPMD::PPMD> ppmd;
  std::unique_ptr<paq8::dmcModel> dmc;
  std::unique_ptr<paq8::Predictor> paq;
  std::unique_ptr<Predictor> cmix;
};

ProbabilityModel::ProbabilityModel(const Options& options)
    : impl_(std::make_unique<Impl>(options)) {}
ProbabilityModel::~ProbabilityModel() = default;
double ProbabilityModel::predict() { return impl_->predict(); }
void ProbabilityModel::observe(unsigned bit) { impl_->observe(bit); }
const std::valarray<float>& ProbabilityModel::byte_distribution() {
  return impl_->byte_distribution();
}

}  // namespace prediction_probability
