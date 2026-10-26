#pragma once
#include "graph.hpp"
#include <array>
#include <stdexcept>

namespace native_graph_v1::detail {
using Digest = std::array<std::uint8_t, 32>;
struct Spec {
  Kind kind;
  std::uint32_t width;
  std::uint64_t bytes;
  bool operator==(const Spec&) const = default;
};
struct RootRecord {
  Spec spec;
  bool numeric;
  Digest digest;
  std::string logical_view;
};
struct Node {
  Step step;
  std::vector<Spec> input_specs;
  std::vector<StreamId> outputs;
  std::vector<Spec> output_specs;
  std::vector<std::string> output_logical_views;
  Bytes metadata;
};
struct Frame {
  std::string dispatcher;
  std::string policy_registry;
  std::uint64_t numeric_bytes = 0;
  std::vector<RootRecord> roots;
  std::vector<Node> nodes;
  std::map<StreamId, Stream> leaves;
};
inline void require(bool condition, const char* why) {
  if (!condition) throw std::invalid_argument(why);
}
std::uint64_t checked_add(std::uint64_t, std::uint64_t);
std::uint64_t checked_multiply(std::uint64_t, std::uint64_t);
void validate_limits(const Limits&);
void validate_spec(const Spec&, const Limits&);
void validate_step(const Step&, const Limits&);
Spec spec(const Stream&);
std::string physical_view(Kind);
void validate_view(const std::string&);
Digest sha256(std::span<const std::uint8_t>);
void validate_frame(const Frame&, const Limits&);
Bytes serialize_frame(const Frame&, const Limits&);
Frame parse_frame(std::span<const std::uint8_t>, const Limits&);
}  // namespace native_graph_v1::detail
