#include "elf_float_codec.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#if defined(ELF_PLUS_FLOAT_GOLDEN_DIR)
#define ELF_PLUS_FLOAT_GOLDEN_DIR_STR ELF_PLUS_FLOAT_GOLDEN_DIR
#else
#define ELF_PLUS_FLOAT_GOLDEN_DIR_STR ""
#endif

static constexpr int kChunk = 2048;

static bool read_file(const std::string &path, std::vector<std::uint8_t> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    out.resize(sz);
    if (sz) {
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(sz));
    }
    return true;
}

static std::vector<float> floats64() {
    std::vector<float> v(64);
    for (int i = 0; i < 64; i++) {
        v[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i) * 0.03125f;
    }
    return v;
}

static std::vector<float> floats512() {
    std::vector<float> v(512);
    for (int i = 0; i < 512; i++) {
        v[static_cast<std::size_t>(i)] = 1.25f + static_cast<float>(i) * 0.0625f;
    }
    return v;
}

static bool check_one(const std::string &root, const char *fn, const std::vector<float> &col,
                      const char *label) {
    std::vector<std::uint8_t> golden;
    if (!read_file(root + '/' + fn, golden) || golden.empty()) {
        std::cerr << label << ": missing golden\n";
        return false;
    }
    try {
        auto got_enc = elf_float::encodeElfPlusFloatColumn(col);
        if (got_enc.size() != golden.size() || std::memcmp(got_enc.data(), golden.data(), golden.size()) != 0) {
            std::cerr << label << ": memcmp != Java golden\n";
            return false;
        }
        std::size_t off = 0;
        const int n = static_cast<int>(col.size());
        for (int st = 0; st < n; st += kChunk) {
            int cn = std::min(kChunk, n - st);
            std::span<const float> chunk(col.data() + st, static_cast<std::size_t>(cn));
            auto ref_piece = elf_float::encodeElfPlusFloatColumn(chunk);
            if (off + ref_piece.size() > golden.size()) {
                std::cerr << label << ": chunk offset overflow\n";
                return false;
            }
            if (std::memcmp(golden.data() + off, ref_piece.data(), ref_piece.size()) != 0) {
                std::cerr << label << ": golden slice mismatch\n";
                return false;
            }
            std::vector<float> dec;
            if (!elf_float::decodeElfPlusFloatChunk(
                    std::span<const std::uint8_t>(golden.data() + off, ref_piece.size()), dec, cn)) {
                std::cerr << label << ": decode failed\n";
                return false;
            }
            if (static_cast<int>(dec.size()) != cn) {
                std::cerr << label << ": decode len\n";
                return false;
            }
            for (int i = 0; i < cn; i++) {
                std::uint32_t a = 0, b = 0;
                std::memcpy(&a, &dec[static_cast<std::size_t>(i)], 4);
                std::memcpy(&b, &col[static_cast<std::size_t>(st + i)], 4);
                if (a != b) {
                    std::cerr << label << ": round-trip bits differ at " << (st + i) << '\n';
                    return false;
                }
            }
            off += ref_piece.size();
        }
        if (off != golden.size()) {
            std::cerr << label << ": trailing golden bytes\n";
            return false;
        }
    } catch (const std::exception &e) {
        std::cerr << label << ": " << e.what() << '\n';
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    std::string root = ELF_PLUS_FLOAT_GOLDEN_DIR_STR;
    if (argc > 1 && argv[1][0] != '\0') {
        root = argv[1];
    }
    if (root.empty()) {
        std::cerr << "ELF_PLUS_FLOAT_GOLDEN_DIR not set\n";
        return 2;
    }
    bool ok = check_one(root, "elf_plus_float_n64.bin", floats64(), "elf_plus64");
    ok = check_one(root, "elf_plus_float_n512.bin", floats512(), "elf_plus512") && ok;
    return ok ? 0 : 1;
}
