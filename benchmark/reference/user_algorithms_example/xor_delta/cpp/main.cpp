/**
 * Reference lossless user codec (stdlib only). Entry: main.cpp
 * Wire: magic WCUC1, dtype tag, column count, per-column length + delta stream.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static constexpr uint32_t kMagic = 0x31435557u; /* WCUC1 little-endian */

struct ByteWriter {
    std::vector<uint8_t> buf;
    void put_u32(uint32_t v) {
        buf.push_back(static_cast<uint8_t>(v));
        buf.push_back(static_cast<uint8_t>(v >> 8));
        buf.push_back(static_cast<uint8_t>(v >> 16));
        buf.push_back(static_cast<uint8_t>(v >> 24));
    }
    void put_i64(int64_t v) {
        uint64_t u = static_cast<uint64_t>(v);
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>(u >> (8 * i)));
        }
    }
    void put_f64(double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>(bits >> (8 * i)));
        }
    }
};

struct ByteReader {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;
    bool read_u32(uint32_t& out) {
        if (pos + 4 > size) return false;
        out = static_cast<uint32_t>(data[pos]) | (static_cast<uint32_t>(data[pos + 1]) << 8) |
              (static_cast<uint32_t>(data[pos + 2]) << 16) |
              (static_cast<uint32_t>(data[pos + 3]) << 24);
        pos += 4;
        return true;
    }
    bool read_i64(int64_t& out) {
        if (pos + 8 > size) return false;
        uint64_t u = 0;
        for (int i = 0; i < 8; ++i) u |= static_cast<uint64_t>(data[pos + i]) << (8 * i);
        out = static_cast<int64_t>(u);
        pos += 8;
        return true;
    }
    bool read_f64(double& out) {
        if (pos + 8 > size) return false;
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) bits |= static_cast<uint64_t>(data[pos + i]) << (8 * i);
        std::memcpy(&out, &bits, sizeof(out));
        pos += 8;
        return true;
    }
};

static std::vector<std::vector<double>> read_columns(const std::string& path,
                                                    const std::string& dtype) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    std::stringstream hs(line);
    std::string cell;
    std::vector<std::vector<double>> cols;
    while (std::getline(hs, cell, ',')) {
        cols.emplace_back();
    }
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        for (size_t i = 0; i < cols.size(); ++i) {
            if (!std::getline(ls, cell, ',')) {
                break;
            }
            if (cell.empty()) {
                continue;
            }
            char* end = nullptr;
            double v = std::strtod(cell.c_str(), &end);
            if (end == cell.c_str()) {
                continue;
            }
            if (dtype == "int" && std::floor(v) != v) {
                continue;
            }
            cols[i].push_back(v);
        }
    }
    cols.erase(std::remove_if(cols.begin(), cols.end(),
                              [](const std::vector<double>& c) { return c.empty(); }),
               cols.end());
    return cols;
}

static std::vector<uint8_t> encode_columns(const std::vector<std::vector<double>>& cols,
                                           const std::string& dtype) {
    ByteWriter w;
    w.put_u32(kMagic);
    w.put_u32(dtype == "int" ? 1u : 2u);
    w.put_u32(static_cast<uint32_t>(cols.size()));
    for (const auto& col : cols) {
        w.put_u32(static_cast<uint32_t>(col.size()));
        if (col.empty()) {
            continue;
        }
        if (dtype == "int") {
            int64_t prev = static_cast<int64_t>(col[0]);
            w.put_i64(prev);
            for (size_t i = 1; i < col.size(); ++i) {
                int64_t cur = static_cast<int64_t>(col[i]);
                w.put_i64(cur - prev);
                prev = cur;
            }
        } else {
            double prev = col[0];
            w.put_f64(prev);
            for (size_t i = 1; i < col.size(); ++i) {
                double cur = col[i];
                w.put_f64(cur - prev);
                prev = cur;
            }
        }
    }
    return w.buf;
}

static bool decode_columns(const std::vector<uint8_t>& blob, const std::string& dtype,
                           std::vector<std::vector<double>>& cols) {
    ByteReader r{blob.data(), blob.size()};
    uint32_t magic = 0, tag = 0, ncol = 0;
    if (!r.read_u32(magic) || magic != kMagic || !r.read_u32(tag) || !r.read_u32(ncol)) {
        return false;
    }
    cols.clear();
    for (uint32_t c = 0; c < ncol; ++c) {
        uint32_t n = 0;
        if (!r.read_u32(n)) {
            return false;
        }
        std::vector<double> col;
        col.reserve(n);
        if (n == 0) {
            cols.push_back(std::move(col));
            continue;
        }
        if (tag == 1u) {
            int64_t v = 0, delta = 0;
            if (!r.read_i64(v)) {
                return false;
            }
            col.push_back(static_cast<double>(v));
            for (uint32_t i = 1; i < n; ++i) {
                if (!r.read_i64(delta)) {
                    return false;
                }
                v += delta;
                col.push_back(static_cast<double>(v));
            }
        } else {
            double v = 0.0, delta = 0.0;
            if (!r.read_f64(v)) {
                return false;
            }
            col.push_back(v);
            for (uint32_t i = 1; i < n; ++i) {
                if (!r.read_f64(delta)) {
                    return false;
                }
                v += delta;
                col.push_back(v);
            }
        }
        cols.push_back(std::move(col));
    }
    return true;
}

static int write_csv(const std::string& path, const std::vector<std::vector<double>>& cols,
                     const std::string& dtype) {
    std::ofstream out(path);
    if (!out) {
        return 2;
    }
    for (size_t i = 0; i < cols.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << "col" << i;
    }
    out << '\n';
    size_t nrows = 0;
    for (const auto& c : cols) {
        nrows = std::max(nrows, c.size());
    }
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t i = 0; i < cols.size(); ++i) {
            if (i) {
                out << ',';
            }
            if (r < cols[i].size()) {
                double v = cols[i][r];
                if (dtype == "int") {
                    out << static_cast<int64_t>(v);
                } else {
                    out << v;
                }
            }
        }
        out << '\n';
    }
    return 0;
}

static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    std::string mode = "compress";
    std::string input;
    std::string output;
    std::string dtype = "float";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (a == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (a == "--data-type" && i + 1 < argc) {
            dtype = argv[++i];
        }
    }
    if (input.empty() || output.empty()) {
        return 2;
    }
    if (mode == "compress") {
        auto cols = read_columns(input, dtype);
        if (cols.empty()) {
            return 2;
        }
        auto blob = encode_columns(cols, dtype);
        std::ofstream out(output, std::ios::binary);
        out.write(reinterpret_cast<const char*>(blob.data()),
                  static_cast<std::streamsize>(blob.size()));
        return out.good() ? 0 : 2;
    }
    if (mode == "decompress") {
        auto blob = read_file_bytes(input);
        std::vector<std::vector<double>> cols;
        if (!decode_columns(blob, dtype, cols)) {
            return 2;
        }
        return write_csv(output, cols, dtype);
    }
    return 2;
}
