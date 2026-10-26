#include "graph_internal.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <limits>
#include <set>

namespace native_graph_v1::detail {
std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
  require(b <= UINT64_MAX - a, "byte count overflow");
  return a + b;
}
std::uint64_t checked_multiply(std::uint64_t a, std::uint64_t b) {
  require(a == 0 || b <= UINT64_MAX / a, "byte count overflow");
  return a * b;
}
Spec spec(const Stream& s) { return {s.kind, s.width, s.data.size()}; }
std::string physical_view(Kind kind) {
  if (kind == Kind::Serial) return "bytes";
  if (kind == Kind::Struct) return "struct";
  return "physical_word";
}
void validate_view(const std::string& view) {
  require(!view.empty() && view.size() <= 128 && view.find('\0') == std::string::npos,
          "invalid logical view");
}
void validate_limits(const Limits& l) {
  require(l.max_nodes > 0 && l.max_nodes <= 4096, "invalid node limit");
  require(l.max_active_streams > 0 && l.max_active_streams <= 1024, "invalid data flow limit");
  require(l.max_ordinary_depth > 0 && l.max_ordinary_depth <= 4096, "invalid depth limit");
  require(l.max_stream_bytes <= l.max_working_bytes && l.max_working_bytes <= SIZE_MAX,
          "invalid memory limits");
  require(l.max_frame_bytes >= 48 && l.max_frame_bytes <= SIZE_MAX &&
          l.max_metadata_bytes <= l.max_frame_bytes, "invalid frame limits");
}
void validate_spec(const Spec& s, const Limits& l) {
  require(s.width > 0 && s.width <= 65536, "invalid stream width");
  if (s.kind == Kind::Serial) require(s.width == 1, "serial stream width must be 1");
  else if (s.kind == Kind::Numeric)
    require(s.width == 1 || s.width == 2 || s.width == 4 || s.width == 8,
            "numeric width must be 1, 2, 4, or 8");
  else require(s.kind == Kind::Struct, "invalid stream kind");
  require(s.bytes <= l.max_stream_bytes && s.bytes % s.width == 0,
          "invalid or oversized stream byte count");
}
void validate_step(const Step& s, const Limits& l) {
  require(s.policy_action_id >= -1 && s.policy_action_id < INT64_MAX, "invalid policy action ID");
  require(s.inputs.size() <= l.max_active_streams, "too many node inputs");
  std::set<StreamId> ids(s.inputs.begin(), s.inputs.end());
  require(ids.size() == s.inputs.size(), "duplicate node input");
  if (s.kind == StepKind::Stage) {
    require(!s.inputs.empty() && !s.stage.action.empty() && s.stage.action.size() <= 512 &&
            s.stage.profile.size() <= 512 && s.stage.parameters.size() <= 64,
            "invalid stage request");
    require(s.segment_sizes.empty(), "stage cannot carry segment sizes");
    for (const auto& [key, value] : s.stage.parameters)
      require(!key.empty() && key.size() <= 512 && value.size() <= 4096 &&
              key.find('\0') == std::string::npos && value.find('\0') == std::string::npos,
              "invalid stage parameter");
    require(s.stage.action.find('\0') == std::string::npos &&
            s.stage.profile.find('\0') == std::string::npos, "invalid stage identity");
  } else {
    require(s.stage.action.empty() && s.stage.profile.empty() && s.stage.parameters.empty(),
            "control operation has stage parameters");
    if (s.kind == StepKind::SplitSegments)
      require(s.inputs.size() == 1 && !s.segment_sizes.empty() &&
              s.segment_sizes.size() <= l.max_active_streams, "invalid continuous Split");
    else if (s.kind == StepKind::Concat)
      require(s.inputs.size() >= 2 && s.segment_sizes.empty(), "invalid Concat");
    else if (s.kind == StepKind::Stop)
      require(s.inputs.empty() && s.segment_sizes.empty(), "STOP closes all active data flows");
    else require(false, "unknown node operation");
  }
}
Digest sha256(std::span<const std::uint8_t> data) {
  Digest out{};
  unsigned n = 0;
  if (EVP_Digest(data.data(), data.size(), out.data(), &n, EVP_sha256(), nullptr) != 1 ||
      n != out.size()) throw std::runtime_error("OpenSSL SHA-256 failed");
  return out;
}
namespace {
void working(const std::map<StreamId, Spec>& active, const Limits& l) {
  require(active.size() <= l.max_active_streams, "active data flow budget exceeded");
  std::uint64_t sum = 0;
  for (const auto& [id, s] : active) { (void)id; sum = checked_add(sum, s.bytes); }
  require(sum <= l.max_working_bytes, "active data flow byte budget exceeded");
}
struct Writer {
  Bytes data;
  std::uint64_t limit;
  void room(std::uint64_t n) {
    require(n <= limit && data.size() <= limit - n, "frame size limit exceeded");
  }
  void number(std::uint64_t v, unsigned n) {
    room(n);
    for (unsigned i = 0; i < n; ++i) data.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  void bytes(std::span<const std::uint8_t> b) {
    room(b.size()); data.insert(data.end(), b.begin(), b.end());
  }
  void text(const std::string& s) {
    require(s.size() <= UINT32_MAX, "string too long"); number(s.size(), 4);
    bytes({reinterpret_cast<const std::uint8_t*>(s.data()), s.size()});
  }
  void descriptor(const Spec& s) {
    number(static_cast<std::uint8_t>(s.kind), 1); number(s.width, 4); number(s.bytes, 8);
  }
};
struct Reader {
  std::span<const std::uint8_t> data;
  std::size_t position = 0;
  std::span<const std::uint8_t> bytes(std::uint64_t n) {
    require(position <= data.size() && n <= data.size() - position, "truncated NGF1 frame");
    auto result = data.subspan(position, static_cast<std::size_t>(n));
    position += static_cast<std::size_t>(n); return result;
  }
  std::uint64_t number(unsigned n) {
    auto b = bytes(n); std::uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= std::uint64_t(b[i]) << (8 * i);
    return value;
  }
  std::uint32_t count(std::uint32_t limit) {
    auto n = number(4); require(n <= limit, "oversized record count");
    return static_cast<std::uint32_t>(n);
  }
  std::string text(std::uint32_t limit) {
    auto b = bytes(count(limit));
    return {reinterpret_cast<const char*>(b.data()), b.size()};
  }
  Spec descriptor(const Limits& l) {
    Spec s{static_cast<Kind>(number(1)), static_cast<std::uint32_t>(number(4)), number(8)};
    validate_spec(s, l); return s;
  }
  void end() { require(position == data.size(), "trailing NGF1 data"); }
};
}  // namespace

void validate_frame(const Frame& f, const Limits& l) {
  validate_limits(l);
  require(!f.dispatcher.empty() && f.dispatcher.size() <= 512 &&
          f.dispatcher.find('\0') == std::string::npos, "invalid dispatcher identity");
  require(f.policy_registry.size() <= 128 && f.policy_registry.find('\0') == std::string::npos, "invalid policy registry");
  require(!f.roots.empty() && f.roots.size() <= l.max_active_streams, "invalid root count");
  require(f.nodes.size() <= l.max_nodes, "node budget exceeded");
  std::map<StreamId, Spec> active;
  std::map<StreamId, std::uint32_t> depths;
  std::map<StreamId, std::string> views;
  StreamId next = 0;
  std::uint64_t numeric_bytes = 0, metadata_bytes = 0;
  for (const auto& root : f.roots) {
    validate_spec(root.spec, l); validate_view(root.logical_view);
    views.emplace(next, root.logical_view);
    if (root.numeric) numeric_bytes = checked_add(numeric_bytes, root.spec.bytes);
    depths.emplace(next, 0); active.emplace(next++, root.spec);
  }
  require(numeric_bytes == f.numeric_bytes, "original numeric denominator mismatch");
  working(active, l);
  for (const auto& node : f.nodes) {
    validate_step(node.step, l);
    require(node.step.kind != StepKind::Stop, "STOP is terminal framing, not an interior node");
    require(node.input_specs.size() == node.step.inputs.size(), "node input descriptor count mismatch");
    require(!node.outputs.empty() && node.outputs.size() <= l.max_active_streams &&
            node.outputs.size() == node.output_specs.size(), "node output descriptor count mismatch");
    require(node.outputs.size() == node.output_logical_views.size(), "output logical view count mismatch");
    for (const auto& view : node.output_logical_views) validate_view(view);
    std::uint32_t depth = 0;
    for (std::size_t i = 0; i < node.step.inputs.size(); ++i) {
      const auto id = node.step.inputs[i];
      require(active.contains(id) && active.at(id) == node.input_specs[i],
              "node input missing, consumed twice, or descriptor mismatch");
      depth = std::max(depth, depths.at(id));
    }
    if (node.step.kind == StepKind::Stage) ++depth;
    require(depth <= l.max_ordinary_depth, "ordinary operator depth exceeded");
    if (node.step.kind == StepKind::SplitSegments) {
      require(node.outputs.size() == node.step.segment_sizes.size() && node.metadata.empty(),
              "invalid Split recovery metadata");
      const auto input = node.input_specs[0];
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < node.outputs.size(); ++i) {
        auto n = checked_multiply(node.step.segment_sizes[i], input.width);
        require(node.output_specs[i] == Spec{input.kind, input.width, n}, "Split output shape mismatch");
        require(node.output_logical_views[i] == views.at(node.step.inputs[0]), "Split logical view mismatch");
        sum = checked_add(sum, n);
      }
      require(sum == input.bytes, "Split does not cover original input exactly");
    } else if (node.step.kind == StepKind::Concat) {
      require(node.outputs.size() == 1 && node.metadata.empty(), "invalid Concat recovery metadata");
      const auto first = node.input_specs[0];
      std::uint64_t sum = 0;
      for (auto input : node.input_specs) {
        require(input.kind == first.kind && input.width == first.width, "Concat type mismatch");
        sum = checked_add(sum, input.bytes);
      }
      require(node.output_specs[0] == Spec{first.kind, first.width, sum}, "Concat output shape mismatch");
      auto output_view = views.at(node.step.inputs[0]);
      for (const auto id : node.step.inputs)
        if (views.at(id) != views.at(node.step.inputs[0])) output_view = "bytes";
      require(node.output_logical_views[0] == output_view, "Concat logical view mismatch");
    }
    metadata_bytes = checked_add(metadata_bytes, node.metadata.size());
    require(metadata_bytes <= l.max_metadata_bytes, "graph metadata budget exceeded");
    for (const auto id : node.step.inputs) { active.erase(id); depths.erase(id); views.erase(id); }
    for (std::size_t i = 0; i < node.outputs.size(); ++i) {
      require(node.outputs[i] == next++, "node output IDs must be fresh and sequential");
      validate_spec(node.output_specs[i], l);
      require(active.emplace(node.outputs[i], node.output_specs[i]).second, "duplicate node output");
      depths.emplace(node.outputs[i], depth);
      views.emplace(node.outputs[i], node.output_logical_views[i]);
    }
    working(active, l);
  }
  require(f.leaves.size() == active.size(), "missing or extra final leaf");
  for (const auto& [id, value] : f.leaves)
    require(active.contains(id) && active.at(id) == spec(value), "final leaf ID or descriptor mismatch");
}

Bytes serialize_frame(const Frame& f, const Limits& l) {
  validate_frame(f, l);
  Writer body{{}, l.max_frame_bytes - 48};
  body.text(f.dispatcher); body.text(f.policy_registry); body.number(f.numeric_bytes, 8); body.number(f.roots.size(), 4);
  for (const auto& root : f.roots) {
    body.descriptor(root.spec); body.number(root.numeric, 1); body.text(root.logical_view); body.bytes(root.digest);
  }
  body.number(f.nodes.size(), 4);
  for (const auto& node : f.nodes) {
    body.number(static_cast<std::uint8_t>(node.step.kind), 1);
    body.number(static_cast<std::uint64_t>(node.step.policy_action_id + 1), 8);
    body.text(node.step.stage.action); body.text(node.step.stage.profile);
    body.number(node.step.stage.parameters.size(), 4);
    for (const auto& [key, value] : node.step.stage.parameters) { body.text(key); body.text(value); }
    body.number(node.step.segment_sizes.size(), 4);
    for (const auto n : node.step.segment_sizes) body.number(n, 8);
    body.number(node.step.inputs.size(), 4);
    for (std::size_t i = 0; i < node.step.inputs.size(); ++i) {
      body.number(node.step.inputs[i], 4); body.descriptor(node.input_specs[i]);
    }
    body.number(node.outputs.size(), 4);
    for (std::size_t i = 0; i < node.outputs.size(); ++i) {
      body.number(node.outputs[i], 4); body.descriptor(node.output_specs[i]);
      body.text(node.output_logical_views[i]);
    }
    body.number(node.metadata.size(), 8); body.bytes(node.metadata);
  }
  body.number(f.leaves.size(), 4);
  for (const auto& [id, leaf] : f.leaves) {
    body.number(id, 4); body.descriptor(spec(leaf)); body.bytes(leaf.data);
  }
  Writer out{{}, l.max_frame_bytes};
  out.bytes(Bytes{'N','G','F','1'}); out.number(1, 2); out.number(0, 2);
  out.number(body.data.size(), 8); out.bytes(body.data);
  auto digest = sha256(out.data); out.bytes(digest);
  return std::move(out.data);
}

Frame parse_frame(std::span<const std::uint8_t> data, const Limits& l) {
  validate_limits(l);
  require(data.size() >= 48 && data.size() <= l.max_frame_bytes, "invalid NGF1 frame size");
  Reader header{data};
  auto magic = header.bytes(4);
  require(std::equal(magic.begin(), magic.end(), "NGF1"), "expected NGF1 frame (not TGF1)");
  require(header.number(2) == 1 && header.number(2) == 0, "unsupported NGF1 version or flags");
  const auto body_size = header.number(8);
  require(body_size == data.size() - 48, "NGF1 length mismatch");
  const auto digest = sha256(data.first(data.size() - 32));
  require(std::equal(digest.begin(), digest.end(), data.end() - 32), "NGF1 checksum mismatch");
  Reader r{data.subspan(16, static_cast<std::size_t>(body_size))};
  Frame f;
  f.dispatcher = r.text(512); f.policy_registry = r.text(128); f.numeric_bytes = r.number(8);
  const auto roots = r.count(l.max_active_streams);
  for (std::uint32_t i = 0; i < roots; ++i) {
    auto s = r.descriptor(l); auto numeric = r.number(1);
    require(numeric <= 1, "invalid numeric payload flag");
    auto view = r.text(128);
    Digest d{}; auto b = r.bytes(d.size()); std::copy(b.begin(), b.end(), d.begin());
    f.roots.push_back({s, numeric != 0, d, std::move(view)});
  }
  const auto nodes = r.count(l.max_nodes);
  std::uint64_t metadata_total = 0;
  for (std::uint32_t i = 0; i < nodes; ++i) {
    Node n;
    n.step.kind = static_cast<StepKind>(r.number(1));
    const auto policy_id = r.number(8);
    require(policy_id <= static_cast<std::uint64_t>(INT64_MAX), "policy ID overflow");
    n.step.policy_action_id = static_cast<std::int64_t>(policy_id) - 1;
    n.step.stage.action = r.text(512); n.step.stage.profile = r.text(512);
    const auto params = r.count(64);
    std::string previous;
    for (std::uint32_t k = 0; k < params; ++k) {
      auto key = r.text(512), value = r.text(4096);
      require(k == 0 || previous < key, "duplicate or noncanonical parameter key");
      previous = key; n.step.stage.parameters.emplace(std::move(key), std::move(value));
    }
    const auto segments = r.count(l.max_active_streams);
    for (std::uint32_t k = 0; k < segments; ++k) n.step.segment_sizes.push_back(r.number(8));
    const auto inputs = r.count(l.max_active_streams);
    for (std::uint32_t k = 0; k < inputs; ++k) {
      n.step.inputs.push_back(static_cast<StreamId>(r.number(4))); n.input_specs.push_back(r.descriptor(l));
    }
    const auto outputs = r.count(l.max_active_streams);
    for (std::uint32_t k = 0; k < outputs; ++k) {
      n.outputs.push_back(static_cast<StreamId>(r.number(4))); n.output_specs.push_back(r.descriptor(l));
      n.output_logical_views.push_back(r.text(128));
    }
    const auto metadata_bytes = r.number(8);
    metadata_total = checked_add(metadata_total, metadata_bytes);
    require(metadata_total <= l.max_metadata_bytes, "graph metadata budget exceeded");
    auto b = r.bytes(metadata_bytes); n.metadata.assign(b.begin(), b.end());
    f.nodes.push_back(std::move(n));
  }
  const auto leaves = r.count(l.max_active_streams);
  std::uint64_t leaf_bytes = 0;
  for (std::uint32_t i = 0; i < leaves; ++i) {
    const auto id = static_cast<StreamId>(r.number(4)); auto s = r.descriptor(l);
    leaf_bytes = checked_add(leaf_bytes, s.bytes);
    require(leaf_bytes <= l.max_working_bytes, "leaf allocation budget exceeded");
    require(!f.leaves.contains(id), "duplicate final leaf ID");
    auto b = r.bytes(s.bytes);
    f.leaves.emplace(id, Stream{s.kind, s.width, Bytes(b.begin(), b.end())});
  }
  r.end(); validate_frame(f, l);
  return f;
}
}  // namespace native_graph_v1::detail
