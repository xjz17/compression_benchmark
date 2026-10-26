// Optional golden verifier: checks manifest-listed byte lengths and file presence.
// SHA-256 in manifest is produced by Java export; this tool does not re-hash.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(TSFILE_GOLDEN_DIR)
#define TSFILE_GOLDEN_DIR_STR TSFILE_GOLDEN_DIR
#else
#define TSFILE_GOLDEN_DIR_STR ""
#endif

static std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  size_t j = s.size();
  while (j > i && (s[j - 1] == ' ' || s[j - 1] == '\t')) --j;
  return s.substr(i, j - i);
}

static bool split_tab(const std::string& line, std::vector<std::string>& out) {
  out.clear();
  std::string cur;
  for (char c : line) {
    if (c == '\t') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return true;
}

static long long parse_ll(const char* s) {
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (end == s || *end != '\0') return -1;
  return v;
}

static std::uintmax_t file_size(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return static_cast<std::uintmax_t>(-1);
  return static_cast<std::uintmax_t>(f.tellg());
}

int main(int argc, char** argv) {
  std::string golden_root;
  if (argc >= 2) {
    golden_root = argv[1];
  } else {
    const char* env = std::getenv("TSFILE_GOLDEN_DIR");
    if (env && env[0]) golden_root = env;
  }
  if (golden_root.empty()) {
    std::string def(TSFILE_GOLDEN_DIR_STR);
    if (!def.empty()) golden_root = std::move(def);
  }

  if (golden_root.empty()) {
    std::cerr << "Usage: verify_tsfile_goldens <golden_dir>\n"
                 "   or: TSFILE_GOLDEN_DIR=<dir> verify_tsfile_goldens\n"
                 "   or: build with -DTSFILE_GOLDEN_DIR=... \n";
    return 2;
  }

  if (!golden_root.empty() && golden_root.back() == '/')
    golden_root.pop_back();

  std::string manifest_path = golden_root + "/manifest.tsv";
  std::ifstream mf(manifest_path);
  if (!mf) {
    std::cerr << "Cannot open " << manifest_path << '\n';
    return 1;
  }

  std::string line;
  if (!std::getline(mf, line)) {
    std::cerr << "Empty manifest\n";
    return 1;
  }
  // skip header

  int rows = 0;
  int errors = 0;
  while (std::getline(mf, line)) {
    line = trim(line);
    if (line.empty()) continue;
    std::vector<std::string> cols;
    split_tab(line, cols);
    if (cols.size() < 7) {
      std::cerr << "Bad manifest row (need 7 tab columns): " << line << '\n';
      ++errors;
      continue;
    }
    const std::string& encoding = cols[0];
    const std::string& dtype = cols[1];
    const std::string& values_len_s = cols[3];
    const std::string& bytes_len_s = cols[4];
    const std::string& filename = cols[6];

    long long values_len = parse_ll(values_len_s.c_str());
    long long bytes_len = parse_ll(bytes_len_s.c_str());
    if (values_len != 64) {
      std::cerr << "Unexpected values_len for " << encoding << " " << dtype << ": " << values_len << '\n';
      ++errors;
    }
    if (bytes_len < 0) {
      std::cerr << "Bad bytes_len for " << filename << '\n';
      ++errors;
      continue;
    }

    std::string bin_path = golden_root + "/" + filename;
    std::uintmax_t sz = file_size(bin_path);
    if (sz == static_cast<std::uintmax_t>(-1)) {
      std::cerr << "Missing or unreadable " << bin_path << '\n';
      ++errors;
      continue;
    }
    if (static_cast<long long>(sz) != bytes_len) {
      std::cerr << "Size mismatch " << bin_path << ": file " << sz << " manifest " << bytes_len << '\n';
      ++errors;
    }
    ++rows;
  }

  if (rows == 0) {
    std::cerr << "No manifest rows verified\n";
    return 1;
  }

  if (errors) {
    std::cerr << errors << " error(s); " << rows << " row(s) checked\n";
    return 1;
  }

  std::cout << "OK: " << rows << " golden file(s) match manifest sizes\n";
  return 0;
}
