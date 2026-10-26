#pragma once

#include "stage_dispatch.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace native_graph_v1::operator102_measure {

// Entropy is evaluated independently for every physical stream.  In
// particular, callers must not concatenate entity blocks, output ports, or
// inverse metadata: doing so would create contexts that the representation
// does not contain.
struct EntropySummary {
  std::uint64_t bytes = 0;
  std::uint64_t streams = 0;
  std::uint64_t markov3_conditioned_bytes = 0;
  double h0_bits = 0.0;
  double markov3_conditional_bits = 0.0;
  double markov3_prefix_h0_bits = 0.0;
  double markov3_modeled_bits = 0.0;
};

EntropySummary entropy_one(std::span<const std::uint8_t> bytes);
EntropySummary entropy_streams(const std::vector<std::span<const std::uint8_t>>& streams);
EntropySummary& operator+=(EntropySummary& left, const EntropySummary& right);

// Charges the physical tuple descriptor (role/kind/width/ordinal/length) and
// payload of every port and inverse-metadata record.  Descriptors and payloads
// are separate entropy streams so Markov contexts never cross a record edge.
EntropySummary representation_entropy(const std::vector<Stream>& outputs,
                                      const std::vector<Bytes>& inverse_metadata);

struct PipelineStep {
  StageRequest request;
};

struct PipelineMeasurement {
  bool bitexact = false;
  std::uint64_t output_payload_bytes = 0;
  std::uint64_t metadata_bytes = 0;
  EntropySummary final_representation;
  EntropySummary focal_before;
  EntropySummary focal_after;
  // The exact physical state sent to the fixed downstream ablation sink.
  // Keeping typed output descriptors and every inverse metadata packet here
  // lets the caller build a self-describing, byte-reversible bundle without
  // rerunning the operator.
  std::vector<Stream> final_outputs;
  std::vector<Bytes> inverse_metadata;
};

// Wall-clock measurements for the reversible present pipeline.  Every sample
// is a complete forward pass followed by its matching inverse pass.  The
// input Stream and the decoded output comparison are deliberately outside the
// timed regions; all stage dispatch, allocation, metadata construction and
// inverse-stage work are inside.  A failed repeat throws instead of returning
// a timing sample.
struct PipelineTiming {
  bool bitexact = false;
  // Kernel-only stage dispatch, retained as a diagnostic view.
  std::vector<std::uint64_t> encode_ns;
  std::vector<std::uint64_t> decode_ns;
  // Publication/ranking boundary: kernel plus exact O2R1 pack/parse.
  std::vector<std::uint64_t> full_encode_ns;
  std::vector<std::uint64_t> full_decode_ns;
};

struct O2R1State {
  std::vector<Stream> outputs;
  std::vector<Bytes> inverse_metadata;
  std::vector<Bytes> auxiliaries;
  std::optional<Bytes> adapter;
};

// Exact physical representation shared with the production ablation sink.
Bytes pack_o2r1(const std::vector<Stream>& outputs,
                const std::vector<Bytes>& inverse_metadata,
                const std::vector<Bytes>& auxiliaries,
                const Bytes* adapter);
O2R1State parse_o2r1(std::span<const std::uint8_t> bytes);

// Runs each real native stage once, retains every inverse-metadata packet as
// part of the representation, then executes the exact inverse stages in
// reverse order and requires a bit-for-bit match with the original stream.
// focal_step selects the operator whose immediate entropy effect is reported.
PipelineMeasurement measure_pipeline(const StageDispatcher& dispatcher,
                                     const Stream& input,
                                     const std::vector<PipelineStep>& steps,
                                     std::size_t focal_step,
                                     const EntropySummary* input_entropy = nullptr);

PipelineTiming benchmark_pipeline(const StageDispatcher& dispatcher,
                                  const Stream& input,
                                  const std::vector<PipelineStep>& steps,
                                  const std::vector<Bytes>& auxiliaries,
                                  const Bytes* adapter,
                                  std::size_t warmups,
                                  std::size_t repeats);

}  // namespace native_graph_v1::operator102_measure
