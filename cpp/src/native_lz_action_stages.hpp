#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Byte-for-byte port of the frozen native-LZ aligned-eight fast action path.
// There is one parameterized search kernel, not 42 different search functions.
// Original min/max stay in NLS1 context; token length bias is independent.
namespace native_lz_action {
using Bytes = std::vector<std::uint8_t>;
struct Token { int literal = -1; std::uint32_t offset = 0, length = 0; };
struct Context {
  std::uint8_t role = 0;
  std::uint32_t window = 0, minimum = 0, maximum = 0;
  std::uint64_t raw_size = 0;
};
struct SearchConfiguration {
  std::uint32_t window = 65535, minimum = 3, maximum = 258;
};
std::vector<Token> parse(const Bytes&, const SearchConfiguration&);
// Direct A060 entry: A065/A057 are configuration, not required search stages.
Bytes configured_parse_packet(const Bytes&, const SearchConfiguration&);
// Frozen NLS1 carrier adapters used by the historical stage protocol.
Bytes attach_window(const Bytes&, std::uint32_t window);
Bytes attach_match_limits(const Bytes&, std::uint32_t minimum, std::uint32_t maximum);
std::vector<Token> parse(const Bytes&, std::uint32_t window, std::uint32_t minimum,
                         std::uint32_t maximum);
// Compatibility entry points for backend callers below the fast-path threshold
// and for the explicitly selected literal mode. They are not new RL actions.
std::vector<Token> parse_chain(const Bytes&, std::uint32_t window, std::uint32_t minimum,
                               std::uint32_t maximum, std::uint32_t candidates, bool lazy);
Bytes literal(const Bytes&);
Bytes serialize_tokens(const std::vector<Token>&);
std::vector<Token> deserialize_tokens(const Bytes&);
Bytes expand(const std::vector<Token>&, std::uint64_t expected);
Bytes pack(const std::vector<Token>&, std::uint32_t minimum);
std::vector<Token> unpack(const Bytes&, std::uint32_t minimum, std::uint64_t expected);
Context context(const Bytes&);
Bytes forward(const std::string& action, const Bytes&);
Bytes inverse(const std::string& action, const Bytes&, const Context* = nullptr);
}  // namespace native_lz_action
