#include "native_lz_family.hpp"

#include "active_method_policy.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <unordered_map>

namespace {

struct Preset {
  int min_match;
  int max_match;
  int window;
  int max_candidates;
  bool lazy;
};

// Patch benchmarks normally feed 10,000 numeric values (about 80 KiB) at a time.
// Keep those patches on the bounded-candidate path; the full chain search is far
// too slow for large columns and previously caused ELF-family 10 minute timeouts.
constexpr std::size_t kDefaultFastPathBytes = 64 * 1024;
constexpr int kMatchChunkSize = 32;
constexpr int kMatchPrefixBytes = 8;

bool has_preset(const std::vector<std::pair<std::string, Preset>>& table,
                const std::string& algorithm) {
  return std::any_of(table.begin(), table.end(), [&](const auto& row) {
    return row.first == algorithm;
  });
}

int clamp_int(int value, int lo, int hi) { return std::max(lo, std::min(hi, value)); }

int algorithm_signature(const std::string& algorithm) {
  int sig = 0;
  for (std::size_t i = 0; i < algorithm.size(); ++i) {
    sig += static_cast<int>(i + 1) * static_cast<unsigned char>(algorithm[i]);
  }
  return sig;
}

bool all_digits(const std::string& text) {
  return !text.empty() &&
         std::all_of(text.begin(), text.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

int algorithm_level(const std::string& algorithm) {
  if (algorithm.rfind("bsc", 0) == 0 && algorithm.size() > 3) {
    const std::string tail = algorithm.substr(3);
    if (all_digits(tail)) {
      return std::stoi(tail);
    }
  }
  const std::size_t idx = algorithm.rfind('_');
  if (idx != std::string::npos && idx + 1 < algorithm.size()) {
    const std::string tail = algorithm.substr(idx + 1);
    if (all_digits(tail)) {
      return std::stoi(tail);
    }
  }
  if (algorithm == "brotli" || algorithm == "gz" || algorithm == "fpzip" ||
      algorithm == "lzfse") {
    return 6;
  }
  if (algorithm == "fpcompress_dpratio") {
    return 8;
  }
  if (algorithm == "fpcompress_dpspeed") {
    return 2;
  }
  if (algorithm == "lz4-framed" || algorithm == "snappy-framed") {
    return 1;
  }
  if (algorithm == "blosc_blosclz" || algorithm == "blosc2_blosclz") {
    return 5;
  }
  if (algorithm == "ext_zfp_zfpy") {
    return 7;
  }
  if (algorithm == "lzjb") {
    return 2;
  }
  return 5;
}

bool starts_with_any(const std::string& value, std::initializer_list<const char*> prefixes) {
  for (const char* prefix : prefixes) {
    if (value.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

std::string lower_string(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return s;
}

std::string algorithm_profile(const std::string& algorithm) {
  const std::string s = lower_string(algorithm);
  if (s == "fpcompress_dpspeed" || s == "lz4-framed" || s == "snappy-framed" || s == "lzfse" || s == "lzjb" ||
      starts_with_any(s, {"lz4fast_", "fastlz_", "quicklz_", "lzf_"})) {
    return "speed";
  }
  if (s == "gz" ||
      starts_with_any(s, {"zlib_", "zlib_ng_", "slz_deflate_", "slz_gzip_", "slz_zlib_",
                          "libdeflate_"})) {
    return "deflate";
  }
  if (s == "fpzip" || s == "ext_zfp_zfpy") {
    return "float";
  }
  if (s == "fpcompress_dpratio" || s == "brotli" ||
      starts_with_any(s, {"xz_level_", "zstd_level_", "fastlzma2_", "lzlib_", "csc_",
                          "bzip2_", "bzip3_", "bsc", "kanzi_", "zling_", "ppmd8_",
                          "lzg_", "lzham_", "tornado_"})) {
    return "ratio";
  }
  return "balanced";
}

Preset generated_preset(const std::string& algorithm) {
  const int level = algorithm_level(algorithm);
  const int sig = algorithm_signature(algorithm);
  const std::string profile = algorithm_profile(algorithm);
  if (profile == "speed") {
    const int min_match = 3 + (sig % 2);
    const int max_match = clamp_int(18 + std::min(80, level * 4) + (sig % 13), min_match, 258);
    const int window = clamp_int(1024 * (2 + std::min(30, std::max(1, level))), 1024, 65535);
    const int candidates = clamp_int(1 + (level / 4) + (sig % 3), 1, 8);
    return {min_match, max_match, window, candidates, false};
  }
  if (profile == "deflate") {
    const int min_match = 3;
    const int max_match = clamp_int(66 + std::min(120, level * 10) + (sig % 23), min_match, 258);
    const int window = clamp_int(8192 * (1 + std::min(7, std::max(1, level) / 2)), 4096, 65535);
    const int candidates = clamp_int(6 + level * 2 + (sig % 5), 4, 32);
    return {min_match, max_match, window, candidates, level >= 5 || sig % 4 == 0};
  }
  if (profile == "ratio") {
    const int min_match = 3 + (sig % 3);
    const int max_match = clamp_int(130 + std::min(100, level * 12) + (sig % 29), min_match, 258);
    const int window = level >= 4 ? 65535 : 32768;
    const int candidates = clamp_int(16 + level * 4 + (sig % 9), 8, 64);
    return {min_match, max_match, window, candidates, true};
  }
  if (profile == "float") {
    const int min_match = 4;
    const int max_match = clamp_int(96 + level * 9 + (sig % 41), min_match, 258);
    const int candidates = clamp_int(12 + level * 3 + (sig % 11), 8, 48);
    return {min_match, max_match, 32768, candidates, true};
  }
  const int min_match = 3 + (sig % 2);
  const int max_match = clamp_int(64 + std::min(120, level * 8) + (sig % 31), min_match, 258);
  const int window = clamp_int(4096 * (1 + std::min(15, std::max(1, level))), 4096, 65535);
  const int candidates = clamp_int(4 + level * 2 + (sig % 7), 2, 40);
  return {min_match, max_match, window, candidates, level >= 6};
}

void add_generated_preset(std::vector<std::pair<std::string, Preset>>& table,
                          const std::string& algorithm) {
  if (!has_preset(table, algorithm)) {
    table.push_back({algorithm, generated_preset(algorithm)});
  }
}

std::vector<int> sample_levels(int first, int last, int max_count = 10) {
  std::vector<int> out;
  if (first > last) {
    return out;
  }
  const int total = last - first + 1;
  if (total <= max_count) {
    for (int level = first; level <= last; ++level) {
      out.push_back(level);
    }
    return out;
  }
  if (max_count <= 1) {
    return {first};
  }
  for (int i = 0; i < max_count; ++i) {
    int level = i == max_count - 1
                    ? last
                    : static_cast<int>(std::round(first + ((last - first) * i) /
                                                              static_cast<double>(max_count - 1)));
    level = std::max(first, std::min(last, level));
    if (std::find(out.begin(), out.end(), level) == out.end()) {
      out.push_back(level);
    }
  }
  for (int level = first; level <= last && static_cast<int>(out.size()) < max_count; ++level) {
    if (std::find(out.begin(), out.end(), level) == out.end()) {
      out.push_back(level);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string algorithm_key_for_level(std::string codec_id, int level) {
  std::replace(codec_id.begin(), codec_id.end(), '-', '_');
  return codec_id + "_" + std::to_string(level);
}

void add_range(std::vector<std::pair<std::string, Preset>>& table,
               const std::string& codec_id, int first, int last) {
  for (int level : sample_levels(first, last)) {
    add_generated_preset(table, algorithm_key_for_level(codec_id, level));
  }
}

void add_preset_rows(std::vector<std::pair<std::string, Preset>>& table,
                     std::initializer_list<const char*> algorithms) {
  for (const char* algorithm : algorithms) {
    add_generated_preset(table, algorithm);
  }
}

void add_lzbench_catalog_rows(std::vector<std::pair<std::string, Preset>>& table) {
  add_range(table, "brieflz", 1, 9);
  add_preset_rows(table, {"bsc0", "bsc1", "bsc2", "bsc3", "bsc4", "bsc5", "bsc6"});
  add_range(table, "bzip2", 1, 9);
  add_range(table, "bzip3", 1, 10);
  add_range(table, "crush", 0, 2);
  add_range(table, "csc", 1, 5);
  add_range(table, "fastlz", 1, 2);
  add_range(table, "fastlzma2", 1, 10);
  add_range(table, "kanzi", 1, 9);
  add_range(table, "libdeflate", 1, 12);
  add_range(table, "lizard", 10, 49);
  for (int level : {1, 3, 9, 17}) {
    add_generated_preset(table, "lz4fast_" + std::to_string(level));
  }
  add_range(table, "lz4hc", 1, 12);
  add_range(table, "lzav", 1, 2);
  add_range(table, "lzf", 0, 1);
  add_preset_rows(table, {"lzfse"});
  add_range(table, "lzg", 1, 9);
  add_range(table, "lzham", 0, 4);
  add_preset_rows(table, {"lzjb"});
  add_range(table, "lzlib", 0, 9);
  add_range(table, "lzo1x", 1, 999);
  add_range(table, "ppmd8", 1, 9);
  add_range(table, "quicklz", 1, 3);
  add_range(table, "slz_deflate", 1, 3);
  add_range(table, "slz_gzip", 1, 3);
  add_range(table, "slz_zlib", 1, 3);
  add_range(table, "tornado", 1, 16);
  add_range(table, "ucl_nrv2b", 1, 9);
  add_range(table, "zlib", 1, 9);
  add_range(table, "zlib-ng", 1, 9);
  add_range(table, "zling", 0, 4);
  add_range(table, "zxc", 1, 6);
  add_range(table, "tamp", 8, 15);
}

void add_generated_presets(std::vector<std::pair<std::string, Preset>>& table) {
  add_preset_rows(table, {"brotli", "gz", "lz4-framed", "snappy-framed",
                          "blosc_blosclz", "blosc2_blosclz", "fpzip",
                          "fpcompress_dpratio", "fpcompress_dpspeed",
                          "elf_on_chimp", "elf_plus", "camel",
                          "ext_zfp_zfpy", "fse", "huff0"});
  for (int level = 0; level <= 9; ++level) {
    add_generated_preset(table, "xz_level_" + std::to_string(level));
  }
  for (int level = 1; level <= 9; ++level) {
    add_generated_preset(table, "zstd_level_" + std::to_string(level));
  }
  add_lzbench_catalog_rows(table);
}

const std::vector<std::pair<std::string, Preset>>& preset_table() {
  static const std::vector<std::pair<std::string, Preset>> table = [] {
    std::vector<std::pair<std::string, Preset>> out = {
      {"density", {4, 66, 4096, 8, false}},
      {"gipfeli", {4, 130, 32768, 16, true}},
      {"glza", {5, 255, 65535, 64, true}},
      {"lzmat", {3, 255, 8192, 16, false}},
      {"lzo1", {3, 33, 4096, 4, false}},
      {"lzo1a", {3, 64, 4096, 6, false}},
      {"lzo1b", {3, 96, 8192, 8, false}},
      {"lzo1c", {3, 128, 8192, 12, true}},
      {"lzo1f", {3, 32, 2048, 2, false}},
      {"lzo1y", {3, 160, 16384, 16, true}},
      {"lzo1z", {3, 255, 32768, 24, true}},
      {"lzo2a", {3, 255, 65535, 32, true}},
      {"lzrw", {3, 18, 4096, 1, false}},
      {"lzsse2", {4, 64, 8192, 4, false}},
      {"lzsse4", {4, 96, 16384, 8, false}},
      {"lzsse4fast", {4, 48, 8192, 2, false}},
      {"lzsse8", {4, 160, 32768, 16, true}},
      {"lzsse8fast", {4, 80, 16384, 4, false}},
      {"lzvn", {3, 258, 65535, 8, false}},
      {"pithy", {4, 128, 32768, 8, false}},
      {"shrinker", {3, 255, 4096, 32, true}},
      {"ucl_nrv2d", {3, 255, 8192, 16, true}},
      {"ucl_nrv2e", {3, 255, 16384, 24, true}},
      {"wflz", {3, 64, 4096, 4, false}},
      {"xpack", {4, 255, 65535, 64, true}},
      {"yalz77", {3, 255, 32768, 16, true}},
      {"yappy", {4, 32, 4096, 2, false}},
    };
    add_generated_presets(out);
    return out;
  }();
  return table;
}

const Preset& preset_for(const std::string& algorithm) {
  for (const auto& row : preset_table()) {
    if (row.first == algorithm) {
      return row.second;
    }
  }
  throw std::invalid_argument("unknown native LZ-family algorithm: " + algorithm);
}

int byte_at(const std::vector<char>& data, std::size_t pos) {
  return static_cast<unsigned char>(data[pos]);
}

int key_at(const std::vector<char>& data, std::size_t pos) {
  if (pos + 2 >= data.size()) {
    return -1;
  }
  return (byte_at(data, pos) << 16) | (byte_at(data, pos + 1) << 8) | byte_at(data, pos + 2);
}

void add_position(std::unordered_map<int, std::vector<int>>& chains,
                  const std::vector<char>& data, std::size_t pos, const Preset& preset) {
  int key = key_at(data, pos);
  if (key < 0) {
    return;
  }
  auto& bucket = chains[key];
  bucket.push_back(static_cast<int>(pos));
  const std::size_t cap = static_cast<std::size_t>(std::max(4, preset.max_candidates * 8));
  if (bucket.size() > cap) {
    bucket.erase(bucket.begin(), bucket.begin() + static_cast<long>(bucket.size() - cap));
  }
}

std::pair<int, int> find_match(const std::vector<char>& data, std::size_t pos,
                               const std::unordered_map<int, std::vector<int>>& chains,
                               const Preset& preset) {
  if (pos + static_cast<std::size_t>(preset.min_match) > data.size()) {
    return {0, 0};
  }
  auto it = chains.find(key_at(data, pos));
  if (it == chains.end()) {
    return {0, 0};
  }
  int best_len = 0;
  int best_offset = 0;
  int checked = 0;
  const int max_len = std::min<int>(preset.max_match, static_cast<int>(data.size() - pos));
  const auto& candidates = it->second;
  for (auto rit = candidates.rbegin(); rit != candidates.rend(); ++rit) {
    const int prev = *rit;
    const int offset = static_cast<int>(pos) - prev;
    if (offset <= 0 || offset > preset.window) {
      continue;
    }
    checked++;
    int length = 0;
    while (length < max_len &&
           data[static_cast<std::size_t>(prev + length)] == data[pos + static_cast<std::size_t>(length)]) {
      length++;
    }
    if (length >= preset.min_match &&
        (length > best_len || (length == best_len && (best_offset == 0 || offset < best_offset)))) {
      best_len = length;
      best_offset = offset;
      if (best_len == max_len) {
        break;
      }
    }
    if (checked >= preset.max_candidates) {
      break;
    }
  }
  return {best_len, best_offset};
}

std::pair<int, int> find_first_match(
    const std::vector<char>& data, std::size_t pos,
    const std::unordered_map<int, std::vector<int>>& chains, const Preset& preset) {
  if (pos + static_cast<std::size_t>(preset.min_match) > data.size()) {
    return {0, 0};
  }
  auto it = chains.find(key_at(data, pos));
  if (it == chains.end()) {
    return {0, 0};
  }
  const int max_len = std::min<int>(preset.max_match, static_cast<int>(data.size() - pos));
  for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit) {
    const int prev = *rit;
    const int offset = static_cast<int>(pos) - prev;
    if (offset <= 0 || offset > preset.window) {
      continue;
    }
    int length = 0;
    while (length < max_len &&
           data[static_cast<std::size_t>(prev + length)] ==
               data[pos + static_cast<std::size_t>(length)]) {
      ++length;
    }
    if (length >= preset.min_match) {
      return {length, offset};
    }
  }
  return {0, 0};
}

struct LzToken {
  bool match = false;
  unsigned char literal = 0;
  int offset = 0;
  int length = 0;
};

std::vector<LzToken> tokenize_ablation(const std::vector<char>& data, Preset preset,
                                       const std::string& omitted_operator) {
  if (omitted_operator == "match_window_model") {
    preset.window = 65535;
  }
  std::vector<LzToken> tokens;
  tokens.reserve(data.size());
  std::unordered_map<int, std::vector<int>> chains;
  std::size_t pos = 0;
  while (pos < data.size()) {
    std::pair<int, int> match{0, 0};
    if (omitted_operator != "lz_match") {
      match = omitted_operator == "match_parse_policy"
                  ? find_first_match(data, pos, chains, preset)
                  : find_match(data, pos, chains, preset);
    }
    int match_len = match.first;
    const int offset = match.second;
    if (omitted_operator != "match_parse_policy" && preset.lazy &&
        match_len >= preset.min_match && pos + 1 < data.size()) {
      add_position(chains, data, pos, preset);
      const auto next = find_match(data, pos + 1, chains, preset);
      if (next.first > match_len) {
        tokens.push_back({false, static_cast<unsigned char>(data[pos]), 0, 0});
        ++pos;
        continue;
      }
    }
    if (match_len >= preset.min_match && omitted_operator != "match_token_format") {
      tokens.push_back({true, 0, offset, match_len});
      for (std::size_t p = pos; p < pos + static_cast<std::size_t>(match_len); ++p) {
        add_position(chains, data, p, preset);
      }
      pos += static_cast<std::size_t>(match_len);
    } else {
      tokens.push_back({false, static_cast<unsigned char>(data[pos]), 0, 0});
      add_position(chains, data, pos, preset);
      ++pos;
    }
  }
  return tokens;
}

std::vector<char> serialize_tokens(const std::vector<LzToken>& tokens, const Preset& preset,
                                   bool packed_controls) {
  std::vector<char> out;
  if (!packed_controls) {
    out.reserve(tokens.size() * 2);
    for (const auto& token : tokens) {
      out.push_back(token.match ? 1 : 0);
      if (!token.match) {
        out.push_back(static_cast<char>(token.literal));
        continue;
      }
      out.push_back(static_cast<char>(token.offset & 0xFF));
      out.push_back(static_cast<char>((token.offset >> 8) & 0xFF));
      out.push_back(static_cast<char>(token.length - preset.min_match));
    }
    return out;
  }
  out.reserve(tokens.size());
  for (std::size_t begin = 0; begin < tokens.size(); begin += 8) {
    const std::size_t end = std::min(begin + 8, tokens.size());
    const std::size_t ctrl_index = out.size();
    out.push_back(0);
    unsigned char control = 0;
    for (std::size_t index = begin; index < end; ++index) {
      const auto& token = tokens[index];
      const int bit = static_cast<int>(index - begin);
      if (!token.match) {
        out.push_back(static_cast<char>(token.literal));
        continue;
      }
      control = static_cast<unsigned char>(control | (1u << bit));
      out.push_back(static_cast<char>(token.offset & 0xFF));
      out.push_back(static_cast<char>((token.offset >> 8) & 0xFF));
      out.push_back(static_cast<char>(token.length - preset.min_match));
    }
    out[ctrl_index] = static_cast<char>(control);
  }
  return out;
}

std::size_t fast_path_threshold() {
  const char* raw = std::getenv("COMPRESSION_BENCHMARK_NATIVE_LZ_FAST_BYTES");
  if (raw == nullptr || raw[0] == '\0') {
    return kDefaultFastPathBytes;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(raw, &end, 10);
  if (end == raw || parsed < 0) {
    return kDefaultFastPathBytes;
  }
  return static_cast<std::size_t>(parsed);
}

bool literal_fast_enabled() {
  const char* raw = std::getenv("COMPRESSION_BENCHMARK_NATIVE_LZ_LITERAL_FAST");
  if (raw == nullptr) {
    return false;
  }
  std::string value = lower_string(raw);
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::vector<char> compress_native_lz_family_literal(const std::vector<char>& data) {
  std::vector<char> out;
  out.reserve(data.size() + ((data.size() + 7) / 8));
  std::size_t pos = 0;
  while (pos < data.size()) {
    out.push_back(0);
    const std::size_t end = std::min(pos + 8, data.size());
    out.insert(out.end(), data.begin() + static_cast<std::ptrdiff_t>(pos),
               data.begin() + static_cast<std::ptrdiff_t>(end));
    pos = end;
  }
  return out;
}

int match_length_fast(const std::vector<char>& data, int left, std::size_t right, int max_len,
                      int start = 0) {
  int length = start;
  const int prefix_len = std::min(kMatchPrefixBytes, max_len);
  while (length < prefix_len) {
    if (data[static_cast<std::size_t>(left + length)] !=
        data[right + static_cast<std::size_t>(length)]) {
      return length;
    }
    length++;
  }
  while (length + kMatchChunkSize <= max_len) {
    bool equal = true;
    for (int i = 0; i < kMatchChunkSize; ++i) {
      if (data[static_cast<std::size_t>(left + length + i)] !=
          data[right + static_cast<std::size_t>(length + i)]) {
        equal = false;
        break;
      }
    }
    if (!equal) {
      break;
    }
    length += kMatchChunkSize;
  }
  while (length < max_len &&
         data[static_cast<std::size_t>(left + length)] ==
             data[right + static_cast<std::size_t>(length)]) {
    length++;
  }
  return length;
}

bool key64_at(const std::vector<char>& data, std::size_t pos, std::uint64_t& out) {
  if (pos + 8 > data.size()) {
    return false;
  }
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(
                 static_cast<unsigned char>(data[pos + static_cast<std::size_t>(i)]))
             << (8 * i);
  }
  out = value;
  return true;
}

void remember_aligned(std::unordered_map<std::uint64_t, int>& last_pos,
                      const std::vector<char>& data, std::size_t pos) {
  std::uint64_t key = 0;
  if (key64_at(data, pos, key)) {
    last_pos[key] = static_cast<int>(pos);
  }
}

void remember_match_aligned(std::unordered_map<std::uint64_t, int>& last_pos,
                            const std::vector<char>& data, std::size_t start, int length) {
  remember_aligned(last_pos, data, start);
  const std::size_t tail = start + static_cast<std::size_t>(length) - 8;
  if (tail > start) {
    remember_aligned(last_pos, data, tail);
  }
}

std::pair<int, int> find_match_fast(const std::vector<char>& data, std::size_t pos,
                                    const std::unordered_map<std::uint64_t, int>& last_pos,
                                    const Preset& preset) {
  std::uint64_t key = 0;
  if (!key64_at(data, pos, key)) {
    return {0, 0};
  }
  auto it = last_pos.find(key);
  if (it == last_pos.end()) {
    return {0, 0};
  }
  const int prev = it->second;
  const int offset = static_cast<int>(pos) - prev;
  if (offset <= 0 || offset > preset.window) {
    return {0, 0};
  }
  const int max_len = std::min<int>(preset.max_match, static_cast<int>(data.size() - pos));
  if (max_len < 8) {
    return {0, 0};
  }
  int length = match_length_fast(data, prev, pos, max_len, 8);
  length = (length / 8) * 8;
  if (length < 8 || length < preset.min_match) {
    return {0, 0};
  }
  return {length, offset};
}

std::vector<char> compress_native_lz_family_fast(const std::vector<char>& data,
                                                 const Preset& preset) {
  std::vector<char> out;
  out.reserve(data.size());
  std::unordered_map<std::uint64_t, int> last_pos;
  std::size_t pos = 0;
  while (pos < data.size()) {
    const std::size_t ctrl_index = out.size();
    out.push_back(0);
    unsigned char ctrl = 0;
    int bit = 0;
    while (bit < 8 && pos < data.size()) {
      auto [match_len, offset] = find_match_fast(data, pos, last_pos, preset);
      if (match_len >= preset.min_match) {
        ctrl = static_cast<unsigned char>(ctrl | (1u << bit));
        out.push_back(static_cast<char>(offset & 0xFF));
        out.push_back(static_cast<char>((offset >> 8) & 0xFF));
        out.push_back(static_cast<char>(match_len - preset.min_match));
        remember_match_aligned(last_pos, data, pos, match_len);
        pos += static_cast<std::size_t>(match_len);
        bit++;
      } else if (bit == 0 && pos + 8 <= data.size()) {
        remember_aligned(last_pos, data, pos);
        out.insert(out.end(), data.begin() + static_cast<std::ptrdiff_t>(pos),
                   data.begin() + static_cast<std::ptrdiff_t>(pos + 8));
        pos += 8;
        bit = 8;
      } else {
        out.push_back(data[pos]);
        remember_aligned(last_pos, data, pos);
        pos++;
        bit++;
      }
    }
    out[ctrl_index] = static_cast<char>(ctrl);
  }
  return out;
}

long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

const std::vector<std::string>& native_lz_family_algorithms() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> out;
    for (const auto& row : preset_table()) {
      if (active_method_policy::is_active_algorithm_key(row.first)) {
        out.push_back(row.first);
      }
    }
    return out;
  }();
  return names;
}

std::vector<char> compress_native_lz_family(const std::vector<char>& data,
                                            const std::string& algorithm) {
  if (!active_method_policy::is_active_algorithm_key(algorithm)) {
    throw std::invalid_argument(
        "algorithm '" + algorithm + "' is retired by active-method policy " +
        active_method_policy::kPolicyId);
  }
  const Preset& preset = preset_for(algorithm);
  if (data.empty()) {
    return {};
  }
  if (data.size() >= fast_path_threshold()) {
    if (literal_fast_enabled()) {
      return compress_native_lz_family_literal(data);
    }
    return compress_native_lz_family_fast(data, preset);
  }
  std::vector<char> out;
  out.reserve(data.size());
  std::unordered_map<int, std::vector<int>> chains;
  std::size_t pos = 0;
  while (pos < data.size()) {
    const std::size_t ctrl_index = out.size();
    out.push_back(0);
    unsigned char ctrl = 0;
    for (int bit = 0; bit < 8 && pos < data.size(); ++bit) {
      auto [match_len, offset] = find_match(data, pos, chains, preset);
      if (preset.lazy && match_len >= preset.min_match && pos + 1 < data.size()) {
        add_position(chains, data, pos, preset);
        auto [next_len, next_offset] = find_match(data, pos + 1, chains, preset);
        (void)next_offset;
        if (next_len > match_len) {
          out.push_back(data[pos]);
          pos++;
          continue;
        }
      }
      if (match_len >= preset.min_match) {
        ctrl = static_cast<unsigned char>(ctrl | (1u << bit));
        out.push_back(static_cast<char>(offset & 0xFF));
        out.push_back(static_cast<char>((offset >> 8) & 0xFF));
        out.push_back(static_cast<char>(match_len - preset.min_match));
        for (std::size_t p = pos; p < pos + static_cast<std::size_t>(match_len); ++p) {
          add_position(chains, data, p, preset);
        }
        pos += static_cast<std::size_t>(match_len);
      } else {
        out.push_back(data[pos]);
        add_position(chains, data, pos, preset);
        pos++;
      }
    }
    out[ctrl_index] = static_cast<char>(ctrl);
  }
  return out;
}

std::vector<char> compress_native_lz_family_ablation(
    const std::vector<char>& data, const std::string& algorithm,
    const std::string& omitted_operator) {
  if (!active_method_policy::is_active_algorithm_key(algorithm)) {
    throw std::invalid_argument(
        "algorithm '" + algorithm + "' is retired by active-method policy " +
        active_method_policy::kPolicyId);
  }
  if (omitted_operator.empty()) {
    return compress_native_lz_family(data, algorithm);
  }
  if (omitted_operator != "lz_match" && omitted_operator != "match_parse_policy" &&
      omitted_operator != "match_window_model" &&
      omitted_operator != "match_token_format" && omitted_operator != "bitstream_pack") {
    throw std::invalid_argument("unsupported native LZ operator ablation: " +
                                omitted_operator);
  }
  const Preset preset = preset_for(algorithm);
  const auto tokens = tokenize_ablation(data, preset, omitted_operator);
  return serialize_tokens(tokens, preset, omitted_operator != "bitstream_pack");
}

std::vector<char> decompress_native_lz_family(const std::vector<char>& data,
                                              const std::string& algorithm,
                                              std::size_t expected_size) {
  const Preset& preset = preset_for(algorithm);
  std::vector<char> out;
  out.reserve(expected_size);
  std::size_t pos = 0;
  while (out.size() < expected_size) {
    if (pos >= data.size()) {
      throw std::runtime_error("native LZ stream truncated before control byte");
    }
    const unsigned char ctrl = static_cast<unsigned char>(data[pos++]);
    for (int bit = 0; bit < 8 && out.size() < expected_size; ++bit) {
      if (ctrl & (1u << bit)) {
        if (pos + 3 > data.size()) {
          throw std::runtime_error("native LZ stream truncated inside match token");
        }
        const int offset = static_cast<unsigned char>(data[pos]) |
                           (static_cast<unsigned char>(data[pos + 1]) << 8);
        const int length = static_cast<unsigned char>(data[pos + 2]) + preset.min_match;
        pos += 3;
        if (offset <= 0 || static_cast<std::size_t>(offset) > out.size()) {
          throw std::runtime_error("native LZ stream has invalid match offset");
        }
        const std::size_t start = out.size() - static_cast<std::size_t>(offset);
        for (int i = 0; i < length; ++i) {
          out.push_back(out[start + static_cast<std::size_t>(i)]);
          if (out.size() > expected_size) {
            throw std::runtime_error("native LZ stream expands beyond expected size");
          }
        }
      } else {
        if (pos >= data.size()) {
          throw std::runtime_error("native LZ stream truncated inside literal");
        }
        out.push_back(data[pos++]);
      }
    }
  }
  return out;
}

std::vector<char> decompress_native_lz_family_ablation(
    const std::vector<char>& data, const std::string& algorithm,
    std::size_t expected_size, const std::string& omitted_operator) {
  if (omitted_operator != "bitstream_pack") {
    return decompress_native_lz_family(data, algorithm, expected_size);
  }
  const Preset& preset = preset_for(algorithm);
  std::vector<char> out;
  out.reserve(expected_size);
  std::size_t pos = 0;
  while (out.size() < expected_size) {
    if (pos >= data.size()) {
      throw std::runtime_error("native LZ unpacked stream truncated before token tag");
    }
    const bool match = data[pos++] != 0;
    if (!match) {
      if (pos >= data.size()) {
        throw std::runtime_error("native LZ unpacked stream truncated inside literal");
      }
      out.push_back(data[pos++]);
      continue;
    }
    if (pos + 3 > data.size()) {
      throw std::runtime_error("native LZ unpacked stream truncated inside match token");
    }
    const int offset = static_cast<unsigned char>(data[pos]) |
                       (static_cast<unsigned char>(data[pos + 1]) << 8);
    const int length = static_cast<unsigned char>(data[pos + 2]) + preset.min_match;
    pos += 3;
    if (offset <= 0 || static_cast<std::size_t>(offset) > out.size()) {
      throw std::runtime_error("native LZ unpacked stream has invalid match offset");
    }
    const std::size_t start = out.size() - static_cast<std::size_t>(offset);
    for (int i = 0; i < length; ++i) {
      out.push_back(out[start + static_cast<std::size_t>(i)]);
      if (out.size() > expected_size) {
        throw std::runtime_error("native LZ unpacked stream expands beyond expected size");
      }
    }
  }
  return out;
}

void bench_native_lz_family_bytes(const std::vector<std::vector<char>>& columns,
                                  const std::string& algorithm,
                                  std::vector<long long>& result,
                                  LossAccum* loss) {
  long long total_o = 0;
  long long total_c = 0;
  long long total_te = 0;
  long long total_td = 0;
  LossAccum local_loss;
  for (const auto& data : columns) {
    const long long start_c = now_ns();
    const auto compressed = compress_native_lz_family(data, algorithm);
    total_te += now_ns() - start_c;
    const long long start_d = now_ns();
    const auto restored = decompress_native_lz_family(compressed, algorithm, data.size());
    total_td += now_ns() - start_d;
    total_o += static_cast<long long>(data.size());
    total_c += static_cast<long long>(compressed.size());
    benchmark_loss::accumulate_bytes(data, restored, local_loss);
  }
  result.assign({total_o, total_c, total_te, total_td});
  if (loss != nullptr) {
    loss->err += local_loss.err;
    loss->ref += local_loss.ref;
  }
}
