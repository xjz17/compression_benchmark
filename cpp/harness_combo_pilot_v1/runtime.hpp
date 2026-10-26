#pragma once

#include "graph.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace harness_combo_pilot_v1 {

// values contains only non-null values in source order and in the exact
// little-endian source dtype.  A set bit in missing denotes NULL; bits are
// assigned least-significant first.
struct Block {
  std::string dtype;
  std::uint64_t row_count = 0;
  native_graph_v1::Bytes values;
  native_graph_v1::Bytes missing;
  bool operator==(const Block&) const = default;
};

struct Encoded {
  native_graph_v1::Bytes frame;
  nlohmann::json audit;
};

// This is an isolated pilot catalog.  Candidate IDs and recipe descriptions
// are self-contained and intentionally do not claim to be entries in the
// frozen 102-operator catalog or the frozen recommendation oracle.
nlohmann::json catalog();

// Validates the block once, then returns the pilot candidates whose declared
// dtype and non-empty-domain requirements are satisfied.
std::vector<std::string> eligible(const Block& block);

// Runs the strict dispatcher with fallback auditing enabled.  The returned
// audit is part of the encode operation; a raw harness fallback is an error.
Encoded encode(const std::string& id, const Block& block);

// Decodes using only the complete NGF frame.  The bound candidate recipe,
// dtype, row count, value shape, and missing bitmap digest are checked before
// inverse execution.
// When expected_candidate_id is non-empty, the HCB1 binding is checked before
// any inverse stage runs.  This is used by outer containers to bind their
// public method name to the self-contained inner frame.
Block decode(std::span<const std::uint8_t> frame,
             std::string_view expected_candidate_id = {});

// Explicit, bounded operator-tree interface.  The tree uses the
// ucc-operator-tree-v1 semantic DSL returned by tree_contract(); callers never
// supply runtime action names, profiles, or policy action IDs.
Encoded encode_tree(const nlohmann::json& tree, const Block& block);

// If expected_method is non-empty it must equal tree_identifier() for the tree
// bound inside the HCB2 root.  The comparison happens before inverse execution.
Block decode_tree(std::span<const std::uint8_t> frame,
                  std::string_view expected_method = {});

// Validates and canonicalizes the tree, then returns
// tree_v1:<sha256(canonical-json)>.
std::string tree_identifier(const nlohmann::json& tree);

// Machine-readable description of the accepted explicit-tree DSL.
nlohmann::json tree_contract();

}  // namespace harness_combo_pilot_v1
