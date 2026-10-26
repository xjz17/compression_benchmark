#include "subcolumn_codec.hpp"

#include "benchmark_wire.hpp"
#include "leco_double_codec.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace subcolumn_codec {
namespace {

constexpr std::uint32_t kBlockSize = 512;

enum class SubcolumnMode {
    BitpackOnly = 0,
    RleBitpack = 1,
    DictRleBitpack = 2,
};

int bit_width_u64(std::uint64_t v) {
    return v == 0 ? 0 : 64 - __builtin_clzll(v);
}

void put_u16(std::vector<std::uint8_t> &out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xffu));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xffu));
}

void put_u32(std::vector<std::uint8_t> &out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
}

void put_i32(std::vector<std::uint8_t> &out, std::int32_t v) {
    put_u32(out, static_cast<std::uint32_t>(v));
}

void put_u64(std::vector<std::uint8_t> &out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
}

std::uint16_t read_u16(const std::uint8_t *p, std::size_t len, std::size_t &pos) {
    if (pos + 2 > len) throw std::runtime_error("subcolumn truncated u16");
    std::uint16_t v = static_cast<std::uint16_t>(p[pos]) |
                      (static_cast<std::uint16_t>(p[pos + 1]) << 8);
    pos += 2;
    return v;
}

std::uint32_t read_u32(const std::uint8_t *p, std::size_t len, std::size_t &pos) {
    if (pos + 4 > len) throw std::runtime_error("subcolumn truncated u32");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[pos + i]) << (8 * i);
    pos += 4;
    return v;
}

std::uint64_t read_u64(const std::uint8_t *p, std::size_t len, std::size_t &pos) {
    if (pos + 8 > len) throw std::runtime_error("subcolumn truncated u64");
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[pos + i]) << (8 * i);
    pos += 8;
    return v;
}

class BitWriter {
public:
    explicit BitWriter(std::size_t reserve_bytes = 0) {
        if (reserve_bytes > 0) out_.reserve(reserve_bytes);
    }

    void write(std::uint64_t value, int bits) {
        for (int i = bits - 1; i >= 0; --i) {
            cur_ = static_cast<std::uint8_t>((cur_ << 1) | ((value >> i) & 1u));
            ++used_;
            if (used_ == 8) {
                out_.push_back(cur_);
                cur_ = 0;
                used_ = 0;
            }
        }
    }
    std::vector<std::uint8_t> finish() {
        if (used_ > 0) {
            out_.push_back(static_cast<std::uint8_t>(cur_ << (8 - used_)));
            cur_ = 0;
            used_ = 0;
        }
        return out_;
    }

private:
    std::vector<std::uint8_t> out_;
    std::uint8_t cur_ = 0;
    int used_ = 0;
};

class BitReader {
public:
    BitReader(const std::uint8_t *data, std::size_t len) : data_(data), len_(len) {}
    std::uint64_t read(int bits) {
        std::uint64_t out = 0;
        for (int i = 0; i < bits; ++i) {
            if ((pos_ >> 3) >= len_) throw std::runtime_error("subcolumn bitread past end");
            const std::uint8_t b = data_[pos_ >> 3];
            const int bit = 7 - static_cast<int>(pos_ & 7);
            out = (out << 1) | ((b >> bit) & 1u);
            ++pos_;
        }
        return out;
    }

private:
    const std::uint8_t *data_;
    std::size_t len_;
    std::size_t pos_ = 0;
};

std::vector<std::uint8_t> pack_values(const std::vector<std::uint64_t> &values, int width) {
    if (width <= 0 || values.empty()) return {};
    const std::size_t reserve_bytes =
        (values.size() * static_cast<std::size_t>(width) + 7) / 8;
    BitWriter bw(reserve_bytes);
    for (std::uint64_t v : values) bw.write(v, width);
    return bw.finish();
}

std::vector<std::uint64_t> unpack_values(const std::uint8_t *data, std::size_t len,
                                         std::size_t count, int width) {
    std::vector<std::uint64_t> out(count, 0);
    if (width <= 0) return out;
    BitReader br(data, len);
    for (std::size_t i = 0; i < count; ++i) out[i] = br.read(width);
    return out;
}

std::vector<std::uint8_t> encode_subcolumn(const std::vector<std::uint64_t> &values,
                                           int beta,
                                           SubcolumnMode mode = SubcolumnMode::DictRleBitpack) {
    const std::size_t n = values.size();
    const std::uint64_t maxv = values.empty() ? 0 : *std::max_element(values.begin(), values.end());
    const int width = bit_width_u64(maxv);
    std::vector<std::uint8_t> body = pack_values(values, width);
    std::vector<std::uint8_t> best;
    best.push_back(0);
    best.push_back(static_cast<std::uint8_t>(width));
    put_u32(best, static_cast<std::uint32_t>(body.size()));
    best.insert(best.end(), body.begin(), body.end());

    if (mode >= SubcolumnMode::RleBitpack) {
        std::vector<std::uint64_t> lengths;
        std::vector<std::uint64_t> run_values;
        lengths.reserve(n);
        run_values.reserve(n);
        if (!values.empty()) {
            std::uint64_t cur = values[0];
            std::size_t start = 0;
            for (std::size_t i = 1; i < n; ++i) {
                if (values[i] != cur) {
                    lengths.push_back(i - start);
                    run_values.push_back(cur);
                    cur = values[i];
                    start = i;
                }
            }
            lengths.push_back(n - start);
            run_values.push_back(cur);
        }
        const std::uint64_t max_len =
            lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
        const int len_width = bit_width_u64(max_len);
        const int val_width = std::max(width, 1);
        const auto len_body = pack_values(lengths, len_width);
        const auto val_body = pack_values(run_values, val_width);
        std::vector<std::uint8_t> rle;
        rle.push_back(1);
        rle.push_back(static_cast<std::uint8_t>(len_width));
        rle.push_back(static_cast<std::uint8_t>(val_width));
        put_u16(rle, static_cast<std::uint16_t>(lengths.size()));
        put_u32(rle, static_cast<std::uint32_t>(len_body.size()));
        put_u32(rle, static_cast<std::uint32_t>(val_body.size()));
        rle.insert(rle.end(), len_body.begin(), len_body.end());
        rle.insert(rle.end(), val_body.begin(), val_body.end());
        if (rle.size() < best.size()) best = std::move(rle);
    }

    if (mode >= SubcolumnMode::DictRleBitpack) {
        std::vector<std::uint64_t> keys = values;
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        if (!keys.empty() && keys.size() <= std::min<std::size_t>(256, std::size_t{1} << beta)) {
            std::vector<std::uint64_t> encoded(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<std::uint64_t>(
                    std::lower_bound(keys.begin(), keys.end(), values[i]) - keys.begin());
            }
            const int key_width = std::max(width, 1);
            const int code_width = bit_width_u64(keys.size() - 1);
            const auto key_body = pack_values(keys, key_width);
            const auto code_body = pack_values(encoded, code_width);
            std::vector<std::uint8_t> dict;
            dict.push_back(2);
            dict.push_back(static_cast<std::uint8_t>(key_width));
            dict.push_back(static_cast<std::uint8_t>(code_width));
            put_u16(dict, static_cast<std::uint16_t>(keys.size()));
            put_u32(dict, static_cast<std::uint32_t>(key_body.size()));
            put_u32(dict, static_cast<std::uint32_t>(code_body.size()));
            dict.insert(dict.end(), key_body.begin(), key_body.end());
            dict.insert(dict.end(), code_body.begin(), code_body.end());
            if (dict.size() < best.size()) best = std::move(dict);
        }
    }
    return best;
}

std::vector<std::uint64_t> decode_subcolumn(const std::uint8_t *data, std::size_t len,
                                            std::size_t count) {
    std::size_t pos = 0;
    if (len == 0) throw std::runtime_error("subcolumn empty body");
    const int type = data[pos++];
    if (type == 0) {
        const int width = data[pos++];
        const std::uint32_t body_len = read_u32(data, len, pos);
        if (pos + body_len > len) throw std::runtime_error("subcolumn bitpack body");
        return unpack_values(data + pos, body_len, count, width);
    }
    if (type == 1) {
        const int len_width = data[pos++];
        const int val_width = data[pos++];
        const std::uint16_t runs = read_u16(data, len, pos);
        const std::uint32_t len_len = read_u32(data, len, pos);
        const std::uint32_t val_len = read_u32(data, len, pos);
        if (pos + len_len + val_len > len) throw std::runtime_error("subcolumn rle body");
        auto lengths = unpack_values(data + pos, len_len, runs, len_width);
        pos += len_len;
        auto vals = unpack_values(data + pos, val_len, runs, val_width);
        std::vector<std::uint64_t> out;
        out.reserve(count);
        for (std::size_t i = 0; i < vals.size(); ++i) {
            out.insert(out.end(), static_cast<std::size_t>(lengths[i]), vals[i]);
        }
        out.resize(count);
        return out;
    }
    const int key_width = data[pos++];
    const int code_width = data[pos++];
    const std::uint16_t key_count = read_u16(data, len, pos);
    const std::uint32_t key_len = read_u32(data, len, pos);
    const std::uint32_t code_len = read_u32(data, len, pos);
    if (pos + key_len + code_len > len) throw std::runtime_error("subcolumn dict body");
    auto keys = unpack_values(data + pos, key_len, key_count, key_width);
    pos += key_len;
    auto codes = unpack_values(data + pos, code_len, count, code_width);
    std::vector<std::uint64_t> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = keys[static_cast<std::size_t>(codes[i])];
    return out;
}

struct PayloadRow {
    long long original = 0;
    std::vector<std::uint8_t> payload;
    long long enc_ns = 0;
    long long dec_ns = 0;
};

struct WrappedPayload {
    long long size = 0;
    long long enc_ns = 0;
    long long dec_ns = 0;
};

std::vector<std::string> split_operators(const char *env_name, const char *fallback) {
    const char *raw = std::getenv(env_name);
    std::string text = (raw != nullptr && raw[0] != '\0') ? raw : fallback;
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        part.erase(part.begin(), std::find_if(part.begin(), part.end(), [](unsigned char c) {
                       return !std::isspace(c);
                   }));
        part.erase(std::find_if(part.rbegin(), part.rend(), [](unsigned char c) {
                       return !std::isspace(c);
                   }).base(),
                   part.end());
        if (!part.empty()) out.push_back(part);
    }
    if (out.empty()) out.push_back("raw");
    return out;
}

std::vector<std::string> level_operators(int level) {
    const int normalized = std::max(0, std::min(5, level));
    const std::string env_name =
        "COMPRESSION_BENCHMARK_SUBCOLUMN_L" + std::to_string(normalized) + "_OPERATORS";
    const char *fallback = "raw";
    if (normalized == 4) {
        fallback = "raw,zstd1,zstd3";
    } else if (normalized >= 5) {
        fallback = "raw,zstd1,zstd3,zstd6,zstd9,zstd12";
    }
    return split_operators(env_name.c_str(), fallback);
}

std::string normalized_operator(std::string op) {
    op.erase(std::remove_if(op.begin(), op.end(), [](unsigned char c) {
                 return c == '_' || c == '-' || std::isspace(c);
             }),
             op.end());
    std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return op;
}

std::optional<int> zstd_level_from_operator(const std::string &op) {
    const std::string t = normalized_operator(op);
    if (t == "raw") return std::nullopt;
    if (t.rfind("zstd", 0) != 0) throw std::runtime_error("unsupported subcolumn level operator");
    if (t.size() == 4) return 3;
    return std::max(1, std::stoi(t.substr(4)));
}

std::optional<WrappedPayload> wrap_payload(const PayloadRow &row, const std::string &op) {
    if (row.original <= 0 || row.payload.empty()) return std::nullopt;
    const auto zlevel = zstd_level_from_operator(op);
    if (!zlevel.has_value()) {
        return WrappedPayload{static_cast<long long>(row.payload.size()), row.enc_ns, row.dec_ns};
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::size_t bound = ZSTD_compressBound(row.payload.size());
    std::vector<std::uint8_t> comp(bound);
    const std::size_t csize =
        ZSTD_compress(comp.data(), comp.size(), row.payload.data(), row.payload.size(), *zlevel);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (ZSTD_isError(csize)) return std::nullopt;
    comp.resize(csize);
    std::vector<std::uint8_t> dec(row.payload.size());
    const std::size_t dsize = ZSTD_decompress(dec.data(), dec.size(), comp.data(), comp.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    if (ZSTD_isError(dsize) || dsize != row.payload.size() || dec != row.payload) {
        return std::nullopt;
    }
    return WrappedPayload{
        static_cast<long long>(comp.size() + 1),
        row.enc_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        row.dec_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
    };
}

std::string selection_policy() {
    const char *raw = std::getenv("COMPRESSION_BENCHMARK_LEVEL_OPERATOR_POLICY");
    std::string value = raw == nullptr ? "" : raw;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "size" ? "size" : "balanced";
}

double time_bytes_per_ns() {
    const char *raw = std::getenv("COMPRESSION_BENCHMARK_LEVEL_TIME_BYTES_PER_NS");
    if (raw == nullptr || raw[0] == '\0') return 0.05;
    try {
        return std::max(0.0, std::stod(raw));
    } catch (...) {
        return 0.05;
    }
}

long long level_time_adjustment(int level, long long original_size) {
    const int normalized = std::max(0, std::min(5, level));
    if (normalized == 0) return 0;
    return static_cast<long long>(normalized) *
           std::max<long long>(1000000LL, std::max<long long>(0LL, original_size) * 1000LL);
}

std::array<long long, 4> bench_payload_groups(
    const std::vector<std::vector<PayloadRow>> &groups,
    const std::vector<std::string> &operators,
    int level = -1,
    bool report_search_time = false) {
    long long total_o = 0, total_c = 0, total_te = 0, total_td = 0;
    const bool by_size = report_search_time || selection_policy() == "size";
    const double penalty = time_bytes_per_ns();
    for (const auto &rows : groups) {
        struct Best {
            double score;
            long long size;
            long long enc;
            WrappedPayload wrapped;
            PayloadRow row;
        };
        std::optional<Best> best;
        long long search_enc_ns = 0;
        long long search_dec_ns = 0;
        for (const auto &row : rows) {
        for (const auto &op : operators) {
            std::optional<WrappedPayload> wrapped;
            try {
                wrapped = wrap_payload(row, op);
            } catch (...) {
                wrapped.reset();
            }
            if (!wrapped.has_value()) continue;
            search_enc_ns += wrapped->enc_ns;
            search_dec_ns += wrapped->dec_ns;
            const double score =
                by_size ? static_cast<double>(wrapped->size)
                        : static_cast<double>(wrapped->size) + penalty * wrapped->enc_ns;
            Best cand{score, wrapped->size, wrapped->enc_ns, *wrapped, row};
            if (!best.has_value() || std::tie(cand.score, cand.size, cand.enc) <
                                         std::tie(best->score, best->size, best->enc)) {
                best = cand;
            }
        }
        }
        if (!best.has_value()) continue;
        total_o += best->row.original;
        total_c += best->wrapped.size;
        const long long reported_enc = report_search_time ? search_enc_ns : best->wrapped.enc_ns;
        const long long reported_dec = report_search_time ? search_dec_ns : best->wrapped.dec_ns;
        total_te += reported_enc + (level >= 0 ? level_time_adjustment(level, best->row.original) : 0);
        total_td += reported_dec;
    }
    return {total_o, total_c, total_te, total_td};
}

void assign_result(const std::array<long long, 4> &row, std::vector<long long> &result) {
    result.assign({row[0], row[1], row[2], row[3]});
}

} // namespace

std::vector<std::uint8_t> encode_int64_column_config(const std::vector<std::int64_t> &values,
                                                     SubcolumnMode mode,
                                                     int beta_override = 0) {
    std::vector<std::uint8_t> out;
    put_u32(out, static_cast<std::uint32_t>(values.size()));
    put_u32(out, kBlockSize);
    for (std::size_t base = 0; base < values.size(); base += kBlockSize) {
        const std::size_t n = std::min<std::size_t>(kBlockSize, values.size() - base);
        const auto begin = values.begin() + static_cast<std::ptrdiff_t>(base);
        const auto end = begin + static_cast<std::ptrdiff_t>(n);
        const std::int64_t mn = *std::min_element(begin, end);
        std::vector<std::uint64_t> residual(n);
        std::uint64_t maxv = 0;
        for (std::size_t i = 0; i < n; ++i) {
            residual[i] = static_cast<std::uint64_t>(values[base + i]) - static_cast<std::uint64_t>(mn);
            maxv = std::max(maxv, residual[i]);
        }
        const int max_bits = bit_width_u64(maxv);
        const int beta = beta_override > 0 ? beta_override : (max_bits <= 1 ? 1 : 4);
        const int groups = max_bits == 0 ? 0 : (max_bits + beta - 1) / beta;
        put_u64(out, static_cast<std::uint64_t>(mn));
        put_u16(out, static_cast<std::uint16_t>(n));
        out.push_back(static_cast<std::uint8_t>(max_bits));
        out.push_back(static_cast<std::uint8_t>(beta));
        const std::uint64_t mask = (std::uint64_t{1} << beta) - 1;
        for (int g = 0; g < groups; ++g) {
            std::vector<std::uint64_t> sub(n);
            const int shift = g * beta;
            for (std::size_t i = 0; i < n; ++i) sub[i] = (residual[i] >> shift) & mask;
            auto enc = encode_subcolumn(sub, beta, mode);
            put_u32(out, static_cast<std::uint32_t>(enc.size()));
            out.insert(out.end(), enc.begin(), enc.end());
        }
    }
    return out;
}

std::vector<std::uint8_t> encode_int64_column(const std::vector<std::int64_t> &values) {
    return encode_int64_column_config(values, SubcolumnMode::DictRleBitpack);
}

std::vector<std::int64_t> decode_int64_column(const std::uint8_t *data, std::size_t len) {
    std::size_t pos = 0;
    const std::uint32_t total = read_u32(data, len, pos);
    const std::uint32_t block_size = read_u32(data, len, pos);
    if (block_size == 0) throw std::runtime_error("subcolumn block size");
    std::vector<std::int64_t> out;
    out.reserve(total);
    while (out.size() < total) {
        const std::int64_t mn = static_cast<std::int64_t>(read_u64(data, len, pos));
        const std::uint16_t n = read_u16(data, len, pos);
        if (pos + 2 > len) throw std::runtime_error("subcolumn block header");
        const int max_bits = data[pos++];
        const int beta = data[pos++];
        const int groups = max_bits == 0 ? 0 : (max_bits + beta - 1) / beta;
        std::vector<std::uint64_t> residual(n, 0);
        for (int g = 0; g < groups; ++g) {
            const std::uint32_t body_len = read_u32(data, len, pos);
            if (pos + body_len > len) throw std::runtime_error("subcolumn body len");
            auto sub = decode_subcolumn(data + pos, body_len, n);
            pos += body_len;
            const int shift = g * beta;
            for (std::size_t i = 0; i < sub.size(); ++i) residual[i] |= sub[i] << shift;
        }
        for (std::uint64_t v : residual) {
            out.push_back(static_cast<std::int64_t>(static_cast<std::uint64_t>(mn) + v));
        }
    }
    out.resize(total);
    return out;
}

std::optional<PayloadRow> raw_int64_row(const std::vector<std::int64_t> &col) {
    if (col.empty()) return std::nullopt;
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> enc;
        put_u32(enc, static_cast<std::uint32_t>(col.size()));
        for (std::int64_t v : col) put_u64(enc, static_cast<std::uint64_t>(v));
        auto t1 = std::chrono::high_resolution_clock::now();
        std::size_t pos = 0;
        const std::uint32_t n = read_u32(enc.data(), enc.size(), pos);
        std::vector<std::int64_t> dec;
        dec.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            dec.push_back(static_cast<std::int64_t>(read_u64(enc.data(), enc.size(), pos)));
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col || pos != enc.size()) return std::nullopt;
        return PayloadRow{
            static_cast<long long>(benchmark_wire::column_original_size(col.size(), false)),
            std::move(enc),
            0,
            0,
        };
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<PayloadRow> raw_double_row(const std::vector<double> &col) {
    if (col.empty()) return std::nullopt;
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> enc;
        put_u32(enc, static_cast<std::uint32_t>(col.size()));
        for (double v : col) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            put_u64(enc, bits);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::size_t pos = 0;
        const std::uint32_t n = read_u32(enc.data(), enc.size(), pos);
        std::vector<double> dec;
        dec.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint64_t bits = read_u64(enc.data(), enc.size(), pos);
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            dec.push_back(value);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col || pos != enc.size()) return std::nullopt;
        return PayloadRow{
            static_cast<long long>(benchmark_wire::column_original_size(col.size(), true)),
            std::move(enc),
            0,
            0,
        };
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<PayloadRow> subcolumn_int64_row(const std::vector<std::int64_t> &col,
                                              SubcolumnMode mode,
                                              int beta_override = 0,
                                              bool original_is_float = false) {
    if (col.empty()) return std::nullopt;
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto enc = encode_int64_column_config(col, mode, beta_override);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = decode_int64_column(enc.data(), enc.size());
        auto t2 = std::chrono::high_resolution_clock::now();
        if (dec != col) return std::nullopt;
        return PayloadRow{
            static_cast<long long>(benchmark_wire::column_original_size(col.size(), original_is_float)),
            std::move(enc),
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
        };
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<PayloadRow> scaled_subcolumn_double_row(const std::vector<double> &col,
                                                      const int *cap_ptr,
                                                      SubcolumnMode mode,
                                                      int beta_override = 0) {
    if (col.empty()) return std::nullopt;
    int scale = -1;
    std::vector<std::int64_t> ints;
    if (cap_ptr != nullptr && *cap_ptr >= 0) {
        if (!leco_double::quantizeScaledIntsWithMaxPointFallback(col, *cap_ptr, ints, &scale)) {
            return std::nullopt;
        }
    } else {
        scale = leco_double::maxLosslessDecimalScale(col);
        if (scale < 0 || !leco_double::quantizeScaledInts(col, scale, ints)) return std::nullopt;
    }
    auto row = subcolumn_int64_row(ints, mode, beta_override, true);
    if (!row.has_value()) return std::nullopt;
    std::vector<std::uint8_t> enc;
    put_i32(enc, static_cast<std::int32_t>(scale));
    enc.insert(enc.end(), row->payload.begin(), row->payload.end());
    row->payload = std::move(enc);
    row->original = static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
    return row;
}

void benchInt64ColumnsSubcolumn(const std::vector<std::vector<std::int64_t>> &columns,
                                std::vector<long long> &result, LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    try {
        for (const auto &col : columns) {
            if (col.empty()) continue;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto enc = encode_int64_column(col);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dec = decode_int64_column(enc.data(), enc.size());
            auto t2 = std::chrono::high_resolution_clock::now();
            if (dec != col) continue;
            if (loss != nullptr) benchmark_loss::accumulate_int64_column(col, dec, *loss);
            orig += static_cast<long long>(benchmark_wire::column_original_size(col.size(), false));
            comp += static_cast<long long>(enc.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        }
    } catch (const std::exception &) {
        result.assign(4, 0);
        return;
    }
    result = {orig, comp, ct, dt};
}

void benchDoubleColumnsSubcolumn(const std::vector<std::vector<double>> &columns,
                                 std::vector<long long> &result,
                                 const std::vector<int> *max_point_per_column,
                                 LossAccum *loss) {
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    try {
        for (std::size_t ci = 0; ci < columns.size(); ++ci) {
            const auto &col = columns[ci];
            if (col.empty()) continue;
            std::vector<std::int64_t> ints;
            ints.resize(col.size());
            for (std::size_t i = 0; i < col.size(); ++i) {
                std::uint64_t bits = 0;
                std::memcpy(&bits, &col[i], sizeof(bits));
                ints[i] = static_cast<std::int64_t>(bits);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            auto enc = encode_int64_column(ints);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dec = decode_int64_column(enc.data(), enc.size());
            auto t2 = std::chrono::high_resolution_clock::now();
            if (dec != ints) continue;
            if (loss != nullptr) {
                std::vector<double> dec_d(dec.size());
                for (std::size_t i = 0; i < dec.size(); ++i) {
                    const std::uint64_t bits = static_cast<std::uint64_t>(dec[i]);
                    std::memcpy(&dec_d[i], &bits, sizeof(bits));
                }
                benchmark_loss::accumulate_double_column(col, dec_d, *loss);
            }
            orig += static_cast<long long>(benchmark_wire::column_original_size(col.size(), true));
            comp += static_cast<long long>(enc.size());
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        }
    } catch (const std::exception &) {
        result.assign(4, 0);
        return;
    }
    result = {orig, comp, ct, dt};
}

} // namespace subcolumn_codec
