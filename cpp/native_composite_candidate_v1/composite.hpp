#pragma once

#include "baseline.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace native_composite_candidate_v1 {

using Bytes = native_baseline86_v1::Bytes;
using Block = native_baseline86_v1::Block;
using Weights = std::array<double, 3>;
using TaskFeatures = std::array<double, 18>;

inline constexpr const char* registry_version =
    "native-composite-candidate-registry-20260923-v1";
inline constexpr const char* feature_schema =
    "native-composite-task-features-18-sample3072-v1";

// These are coarse, complete compression-algorithm backends.  They are not
// atomic graph operators and deliberately have no canonical graph action ID.
enum class Backend : std::uint16_t {
  Xpack = 1,
  Blosc2 = 2,
  Blosc = 3,
  Libdeflate = 4,
  Zstd = 5,
};

struct CandidateRecord {
  Backend backend;
  std::string label;
  std::string implementation;
  std::string source;
  std::string configuration;
  std::string implementation_class;
  std::string thread_contract;
  bool linked = false;
};

const std::vector<CandidateRecord>& inventory();
const CandidateRecord& candidate(Backend backend);
Backend backend_from_label(std::string_view label);
std::string registry_sha256();
TaskFeatures task_features(const Block& block);

// A frozen selector plugs in here.  Feature extraction and model inference
// belong inside select(); measure() invokes select() inside every timed encode.
// The decode path never needs the selector or its model.
class Selector {
 public:
  virtual ~Selector() = default;
  virtual std::string identity() const = 0;
  // A selector that returns true receives the versioned 18-dimensional task
  // vector.  Extraction then happens inside encode() and its measured time.
  virtual bool uses_task_features() const { return true; }
  virtual Backend select(const TaskFeatures& task_features,
                         const Weights& weights) const = 0;
};

class FixedSelector final : public Selector {
 public:
  explicit FixedSelector(Backend backend) : backend_(backend) {}
  std::string identity() const override;
  bool uses_task_features() const override { return false; }
  Backend select(const TaskFeatures&, const Weights&) const override { return backend_; }
 private:
  Backend backend_;
};

// A tiny frozen policy: each arm predicts its weighted objective as the dot
// product between three train-only coefficients and the requested weights.
// Model parameters are loaded/constructed outside the per-block timer; its
// forward pass and choice remain inside encode().  This policy does not use
// data features, while the same Selector API admits a future 18+3 model.
class ObjectiveLinearSelector final : public Selector {
 public:
  struct Arm { Backend backend; std::array<double, 3> coefficients; };
  ObjectiveLinearSelector(std::string identity, std::vector<Arm> arms);
  std::string identity() const override { return identity_; }
  bool uses_task_features() const override { return false; }
  Backend select(const TaskFeatures&, const Weights& weights) const override;
 private:
  std::string identity_;
  std::vector<Arm> arms_;
};

// Exact coefficient transcription of policy.json SHA-256
// c76f74104ec651705f3e772137280e45d1f9746a44c2714c5f5df50c422cca5b.
ObjectiveLinearSelector train_only_objective_linear_v1();

class DataAwareLinearSelector final : public Selector {
 public:
  struct Prediction {
    Backend backend;
    std::array<double, 3> costs{};
    double weighted_score = 0.0;
  };
  // Loading, JSON validation and SHA-256 verification happen outside timed
  // encode. expected_file_sha256 is mandatory and binds the exact model file.
  static DataAwareLinearSelector load(const std::string& path,
                                      const std::string& expected_file_sha256);
  std::string identity() const override { return identity_; }
  bool uses_task_features() const override { return true; }
  std::vector<Prediction> predict(const TaskFeatures& task_features,
                                  const Weights& weights) const;
  Backend select(const TaskFeatures& task_features,
                 const Weights& weights) const override;
 private:
  struct Arm {
    Backend backend;
    std::array<double, 3> prior{};
    std::array<std::array<double, 18>, 3> residual{};
  };
  std::string identity_;
  std::array<double, 18> mean_{}, scale_{};
  std::array<double, 3> clamp_low_{}, clamp_high_{};
  std::vector<Arm> arms_;
};

struct Decoded {
  Backend backend;
  Weights weights{};
  std::string selector_identity;
  Block block;
  std::uint64_t original_numeric_bytes = 0;
  bool embedded_selection_provenance = false;
};

enum class FrameMode { BaselineNgf1, ProvenanceNcc1 };

// BaselineNgf1 is the performance/default wire.  NGF1 already records the
// selected backend, dtype, row count, bitmap, configuration, payload, graph and
// checksum, so independent decode needs neither the selector nor an extra
// envelope.  Selector/model provenance belongs in the benchmark receipt.
// ProvenanceNcc1 is an optional audit wire that embeds weights, selector and
// candidate-registry identity and therefore pays extra bytes and a second hash.
Bytes encode(const Selector& selector, const Weights& weights, const Block& block,
             FrameMode mode = FrameMode::BaselineNgf1);
Decoded decode(std::span<const std::uint8_t> frame);

struct Measurement {
  Bytes frame;
  Backend backend{};
  std::uint64_t original_numeric_bytes = 0;
  std::vector<std::uint64_t> compression_ns;
  std::vector<std::uint64_t> decompression_ns;
  std::string selector_identity;
  FrameMode frame_mode = FrameMode::BaselineNgf1;
};

// Timer covers selector identity, feature/model work performed by select(),
// candidate dispatch, complete inner-frame encode/decode and NCC1 framing/hash.
// It excludes source I/O and the post-timing byte-for-byte correctness check.
Measurement measure(const Selector& selector, const Weights& weights,
                    const Block& block, unsigned warmups, unsigned repeats,
                    FrameMode mode = FrameMode::BaselineNgf1);

// Direct whole-codec baseline path. It bypasses selector/weight/composite
// dispatch work in both timers while producing the identical baseline86 NGF1
// frame. This is the conservative same-run comparator for a fixed codec.
Measurement measure_direct_backend(Backend backend, const Block& block,
                                   unsigned warmups, unsigned repeats);

}  // namespace native_composite_candidate_v1
