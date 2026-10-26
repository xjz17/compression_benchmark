// Verifies `benchmark/reference/golden/column_encodings/manifest.tsv`: each listed file exists and its
// size matches `bytes_len`. Regenerate the manifest when adding goldens (sizes must match Java export).

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(COLUMN_ENCODINGS_GOLDEN_DIR)
#define COLUMN_ENCODINGS_GOLDEN_DIR_STR COLUMN_ENCODINGS_GOLDEN_DIR
#else
#define COLUMN_ENCODINGS_GOLDEN_DIR_STR ""
#endif

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) {
        s.pop_back();
    }
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
        ++i;
    }
    std::size_t j = s.size();
    while (j > i && (s[j - 1] == ' ' || s[j - 1] == '\t')) {
        --j;
    }
    return s.substr(i, j - i);
}

static void split_tab(const std::string &line, std::vector<std::string> &out) {
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
}

static long long parse_ll(const char *s) {
    char *end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        return -1;
    }
    return v;
}

static std::uintmax_t file_size(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return static_cast<std::uintmax_t>(-1);
    }
    return static_cast<std::uintmax_t>(f.tellg());
}

int main(int argc, char **argv) {
    std::string golden_root;
    if (argc >= 2) {
        golden_root = argv[1];
    } else if (const char *env = std::getenv("COLUMN_ENCODINGS_GOLDEN_DIR"); env && env[0]) {
        golden_root = env;
    }
    if (golden_root.empty()) {
        golden_root = COLUMN_ENCODINGS_GOLDEN_DIR_STR;
    }
    if (golden_root.empty()) {
        std::cerr << "Usage: verify_column_encodings_goldens <golden_dir>\n";
        return 2;
    }
    if (golden_root.back() == '/') {
        golden_root.pop_back();
    }

    std::ifstream mf(golden_root + "/manifest.tsv");
    if (!mf) {
        std::cerr << "Cannot open manifest.tsv under " << golden_root << '\n';
        return 1;
    }

    std::string line;
    if (!std::getline(mf, line)) {
        std::cerr << "Empty manifest\n";
        return 1;
    }

    int rows = 0;
    int errors = 0;
    while (std::getline(mf, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> cols;
        split_tab(line, cols);
        if (cols.size() < 3) {
            std::cerr << "Bad row: " << line << '\n';
            ++errors;
            continue;
        }
        const std::string &filename = cols[0];
        long long values_len = parse_ll(cols[1].c_str());
        long long bytes_len = parse_ll(cols[2].c_str());
        if (values_len < 0 || bytes_len < 0) {
            std::cerr << "Bad lengths: " << filename << '\n';
            ++errors;
            continue;
        }

        std::string path = golden_root + "/" + filename;
        std::uintmax_t sz = file_size(path);
        if (sz == static_cast<std::uintmax_t>(-1)) {
            std::cerr << "Missing " << path << '\n';
            ++errors;
            continue;
        }
        if (static_cast<long long>(sz) != bytes_len) {
            std::cerr << "Size mismatch " << path << ": got " << sz << " want " << bytes_len << '\n';
            ++errors;
        }
        ++rows;
    }

    if (rows == 0) {
        std::cerr << "No data rows in manifest\n";
        return 1;
    }
    if (errors) {
        return 1;
    }
    std::cout << "OK: " << rows << " column_encodings golden(s) match manifest sizes\n";
    return 0;
}
