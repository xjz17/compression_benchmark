// Column encodings vs Java `ColumnarEncodings`: memcmp on deterministic layouts; LZ4 framed payload
// may differ for non-empty data (Apache Java block encoder vs liblz4); see CPP_MIGRATION P1.2.

#include "column_encodings.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(COLUMN_ENCODINGS_GOLDEN_DIR)
#define COLUMN_ENCODINGS_GOLDEN_DIR_STR COLUMN_ENCODINGS_GOLDEN_DIR
#else
#define COLUMN_ENCODINGS_GOLDEN_DIR_STR ""
#endif

static bool read_whole_file(const std::string &path, std::vector<std::uint8_t> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    out.resize(sz);
    if (sz > 0) {
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(sz));
    }
    return true;
}

static std::vector<int> default_int_column() {
    std::vector<int> v;
    v.reserve(64);
    for (int i = 0; i < 64; i++) {
        v.push_back(10000 + i * 3);
    }
    return v;
}

static std::vector<float> default_float_column() {
    std::vector<float> v;
    v.reserve(64);
    for (int i = 0; i < 64; i++) {
        v.push_back(1.0f + static_cast<float>(i) * 0.03125f);
    }
    return v;
}

static std::vector<int> int_column_dupes() {
    return {10, 20, 10, 20, 30};
}

static std::vector<float> float_column_dupes() {
    return {1.0f, 1.0f, 2.0f};
}

static std::vector<int> int_column_runs() {
    return {42, 42, 42, 7, 7};
}

static std::vector<int> int_column_neg_delta() {
    return {-10, -13, -7, -20};
}

static bool ints_equal_report(const std::vector<int> &got, const std::vector<int> &want,
                              const char *name) {
    if (got.size() != want.size()) {
        std::cerr << name << ": length " << got.size() << " vs expected " << want.size() << '\n';
        return false;
    }
    if (!got.empty() &&
        std::memcmp(got.data(), want.data(), got.size() * sizeof(int)) != 0) {
        std::cerr << name << ": int column mismatch\n";
        return false;
    }
    return true;
}

static bool floats_equal_report(const std::vector<float> &got, const std::vector<float> &want,
                                const char *name) {
    if (got.size() != want.size()) {
        std::cerr << name << ": length " << got.size() << " vs expected " << want.size() << '\n';
        return false;
    }
    if (!got.empty() &&
        std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0) {
        std::cerr << name << ": float column mismatch\n";
        return false;
    }
    return true;
}

static bool memcmp_report(const std::vector<std::uint8_t> &got, const std::vector<std::uint8_t> &want,
                          const char *name) {
    if (got.size() != want.size()) {
        std::cerr << name << ": size " << got.size() << " vs golden " << want.size() << '\n';
        return false;
    }
    if (std::memcmp(got.data(), want.data(), got.size()) != 0) {
        std::cerr << name << ": memcmp mismatch\n";
        return false;
    }
    return true;
}

static bool check_pair(const std::string &root, const char *filename,
                       const std::vector<std::uint8_t> &encoded, const char *label) {
    std::vector<std::uint8_t> golden;
    std::string path = root + '/' + filename;
    if (!read_whole_file(path, golden)) {
        std::cerr << "missing golden: " << path << '\n';
        return false;
    }
    return memcmp_report(encoded, golden, label);
}

/** P1.5: for deterministic layouts, `encode(decode(blob))` must equal `blob`. */
template <typename DecFn, typename EncFn>
static bool roundtrip_deterministic_blob(const std::string &root, const char *filename,
                                         const char *label, DecFn &&dec, EncFn &&enc) {
    std::vector<std::uint8_t> blob;
    std::string path = root + '/' + filename;
    if (!read_whole_file(path, blob)) {
        std::cerr << "missing golden for roundtrip: " << path << '\n';
        return false;
    }
    try {
        auto values = dec(blob);
        return memcmp_report(enc(values), blob, label);
    } catch (const std::exception &e) {
        std::cerr << label << ": " << e.what() << '\n';
        return false;
    }
}

int main() {
    const std::string root = COLUMN_ENCODINGS_GOLDEN_DIR_STR;
    if (root.empty()) {
        std::cerr << "COLUMN_ENCODINGS_GOLDEN_DIR not defined\n";
        return 2;
    }

    bool ok = true;
    const std::vector<std::int64_t> int64_extremes = {
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max(),
        -1,
        0,
        1,
        std::numeric_limits<std::int64_t>::min() + 1,
    };
    const auto int64_payload =
        colenc::encodeDeltaBinaryPackedInt64Ablation(int64_extremes, "");
    if (colenc::decodeDeltaBinaryPackedInt64Ablation(int64_payload,
                                                     int64_extremes.size()) != int64_extremes) {
        std::cerr << "DELTA_BINARY_PACKED int64 overflow round trip failed\n";
        ok = false;
    }

    ok &= check_pair(root, "col_dictionary_float_default.bin",
                     colenc::encodeDictionaryFloat(default_float_column()), "encodeDictionaryFloat");
    ok &= check_pair(root, "delta_binary_packed_int32_default.bin",
                     colenc::encodeDeltaZigZagVarint(default_int_column()), "encodeDeltaZigZagVarint");
    ok &= check_pair(root, "col_dictionary_int32_default.bin",
                     colenc::encodeDictionaryInt(default_int_column()), "encodeDictionaryInt");
    ok &= check_pair(root, "col_rle_int32_default.bin", colenc::encodeRleInt(default_int_column()),
                     "encodeRleInt");
    ok &= check_pair(root, "col_bss_planar_float_default.bin",
                     colenc::planarByteStreamSplitFloat(default_float_column()),
                     "planarByteStreamSplitFloat");

    ok &= check_pair(root, "delta_binary_packed_empty.bin",
                     colenc::encodeDeltaZigZagVarint({}), "encodeDeltaZigZagVarint(empty)");
    ok &= check_pair(root, "col_dictionary_int32_empty.bin", colenc::encodeDictionaryInt({}),
                     "encodeDictionaryInt(empty)");
    ok &= check_pair(root, "col_dictionary_float_empty.bin", colenc::encodeDictionaryFloat({}),
                     "encodeDictionaryFloat(empty)");
    ok &= check_pair(root, "col_rle_int32_empty.bin", colenc::encodeRleInt({}), "encodeRleInt(empty)");
    ok &= check_pair(root, "col_bss_planar_float_empty.bin", colenc::planarByteStreamSplitFloat({}),
                     "planarByteStreamSplitFloat(empty)");

    ok &= check_pair(root, "col_dictionary_int32_dupes.bin",
                     colenc::encodeDictionaryInt(int_column_dupes()), "encodeDictionaryInt(dupes)");
    ok &= check_pair(root, "col_dictionary_float_dupes.bin",
                     colenc::encodeDictionaryFloat(float_column_dupes()), "encodeDictionaryFloat(dupes)");
    ok &= check_pair(root, "col_rle_int32_runs.bin", colenc::encodeRleInt(int_column_runs()),
                     "encodeRleInt(runs)");
    ok &= check_pair(root, "delta_binary_packed_neg.bin",
                     colenc::encodeDeltaZigZagVarint(int_column_neg_delta()),
                     "encodeDeltaZigZagVarint(neg)");

    {
        std::vector<std::uint8_t> golden;
        if (!read_whole_file(root + "/col_bss_lz4_float_default.bin", golden)) {
            std::cerr << "missing golden: col_bss_lz4_float_default.bin\n";
            ok = false;
        } else {
            auto want = default_float_column();
            try {
                auto dec = colenc::decodeBssLz4Float(golden);
                ok &= floats_equal_report(dec, want, "decodeBssLz4Float(Java golden)");
                auto enc = colenc::encodeBssLz4Float(want);
                auto round = colenc::decodeBssLz4Float(enc);
                ok &= floats_equal_report(round, want, "encodeBssLz4Float round-trip");
            } catch (const std::exception &e) {
                std::cerr << "bss_lz4_float default: " << e.what() << '\n';
                ok = false;
            }
        }
    }

    ok &= check_pair(root, "col_bss_lz4_float_empty.bin", colenc::encodeBssLz4Float({}),
                     "encodeBssLz4Float(empty) vs Java");

    {
        std::vector<std::uint8_t> golden;
        if (read_whole_file(root + "/col_bss_lz4_float_empty.bin", golden)) {
            try {
                auto dec = colenc::decodeBssLz4Float(golden);
                ok &= floats_equal_report(dec, {}, "decodeBssLz4Float(empty Java blob)");
            } catch (const std::exception &e) {
                std::cerr << "decode empty bss lz4: " << e.what() << '\n';
                ok = false;
            }
        }
    }

    // Decode Java-encoded blobs (P1.3) — must match fixtures used for export.
    auto decode_blob = [&root](const char *fn, std::vector<std::uint8_t> &out) -> bool {
        return read_whole_file(root + "/" + fn, out);
    };
    std::vector<std::uint8_t> blob;
    try {
        if (decode_blob("delta_binary_packed_int32_default.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeDeltaZigZagVarint(blob), default_int_column(),
                                    "decodeDeltaZigZagVarint(default golden)");
        }
        if (decode_blob("delta_binary_packed_empty.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeDeltaZigZagVarint(blob), {},
                                    "decodeDeltaZigZagVarint(empty golden)");
        }
        if (decode_blob("delta_binary_packed_neg.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeDeltaZigZagVarint(blob), int_column_neg_delta(),
                                    "decodeDeltaZigZagVarint(neg golden)");
        }
        if (decode_blob("col_dictionary_int32_default.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeDictionaryInt(blob), default_int_column(),
                                    "decodeDictionaryInt(default golden)");
        }
        if (decode_blob("col_dictionary_int32_empty.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeDictionaryInt(blob), {},
                                    "decodeDictionaryInt(empty golden)");
        }
        if (decode_blob("col_dictionary_int32_dupes.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeDictionaryInt(blob), int_column_dupes(),
                                    "decodeDictionaryInt(dupes golden)");
        }
        if (decode_blob("col_dictionary_float_default.bin", blob)) {
            ok &= floats_equal_report(colenc::decodeDictionaryFloat(blob), default_float_column(),
                                      "decodeDictionaryFloat(default golden)");
        }
        if (decode_blob("col_dictionary_float_empty.bin", blob)) {
            ok &= floats_equal_report(colenc::decodeDictionaryFloat(blob), {},
                                      "decodeDictionaryFloat(empty golden)");
        }
        if (decode_blob("col_dictionary_float_dupes.bin", blob)) {
            ok &= floats_equal_report(colenc::decodeDictionaryFloat(blob), float_column_dupes(),
                                      "decodeDictionaryFloat(dupes golden)");
        }
        if (decode_blob("col_rle_int32_default.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeRleInt(blob), default_int_column(),
                                    "decodeRleInt(default golden)");
        }
        if (decode_blob("col_rle_int32_empty.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeRleInt(blob), {}, "decodeRleInt(empty golden)");
        }
        if (decode_blob("col_rle_int32_runs.bin", blob)) {
            ok &= ints_equal_report(colenc::decodeRleInt(blob), int_column_runs(),
                                    "decodeRleInt(runs golden)");
        }
        if (decode_blob("col_bss_planar_float_default.bin", blob)) {
            ok &= floats_equal_report(colenc::floatsFromPlanarByteStreamSplit(blob),
                                      default_float_column(), "floatsFromPlanar(default golden)");
        }
        if (decode_blob("col_bss_planar_float_empty.bin", blob)) {
            ok &= floats_equal_report(colenc::floatsFromPlanarByteStreamSplit(blob), {},
                                      "floatsFromPlanar(empty golden)");
        }
    } catch (const std::exception &e) {
        std::cerr << "decode golden: " << e.what() << '\n';
        ok = false;
    }

    // P1.5 — byte involution on goldens (skipped: non-empty BSS+LZ4; compressed block may differ).
    ok &= roundtrip_deterministic_blob(
        root, "delta_binary_packed_int32_default.bin", "roundtrip delta default",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDeltaZigZagVarint(b); },
        [](const std::vector<int> &v) { return colenc::encodeDeltaZigZagVarint(v); });
    ok &= roundtrip_deterministic_blob(
        root, "delta_binary_packed_empty.bin", "roundtrip delta empty",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDeltaZigZagVarint(b); },
        [](const std::vector<int> &v) { return colenc::encodeDeltaZigZagVarint(v); });
    ok &= roundtrip_deterministic_blob(
        root, "delta_binary_packed_neg.bin", "roundtrip delta neg",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDeltaZigZagVarint(b); },
        [](const std::vector<int> &v) { return colenc::encodeDeltaZigZagVarint(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_dictionary_int32_default.bin", "roundtrip dict int default",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDictionaryInt(b); },
        [](const std::vector<int> &v) { return colenc::encodeDictionaryInt(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_dictionary_int32_empty.bin", "roundtrip dict int empty",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDictionaryInt(b); },
        [](const std::vector<int> &v) { return colenc::encodeDictionaryInt(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_dictionary_int32_dupes.bin", "roundtrip dict int dupes",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDictionaryInt(b); },
        [](const std::vector<int> &v) { return colenc::encodeDictionaryInt(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_dictionary_float_default.bin", "roundtrip dict float default",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDictionaryFloat(b); },
        [](const std::vector<float> &v) { return colenc::encodeDictionaryFloat(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_dictionary_float_empty.bin", "roundtrip dict float empty",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDictionaryFloat(b); },
        [](const std::vector<float> &v) { return colenc::encodeDictionaryFloat(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_dictionary_float_dupes.bin", "roundtrip dict float dupes",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeDictionaryFloat(b); },
        [](const std::vector<float> &v) { return colenc::encodeDictionaryFloat(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_rle_int32_default.bin", "roundtrip RLE default",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeRleInt(b); },
        [](const std::vector<int> &v) { return colenc::encodeRleInt(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_rle_int32_empty.bin", "roundtrip RLE empty",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeRleInt(b); },
        [](const std::vector<int> &v) { return colenc::encodeRleInt(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_rle_int32_runs.bin", "roundtrip RLE runs",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeRleInt(b); },
        [](const std::vector<int> &v) { return colenc::encodeRleInt(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_bss_planar_float_default.bin", "roundtrip planar default",
        [](const std::vector<std::uint8_t> &b) { return colenc::floatsFromPlanarByteStreamSplit(b); },
        [](const std::vector<float> &v) { return colenc::planarByteStreamSplitFloat(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_bss_planar_float_empty.bin", "roundtrip planar empty",
        [](const std::vector<std::uint8_t> &b) { return colenc::floatsFromPlanarByteStreamSplit(b); },
        [](const std::vector<float> &v) { return colenc::planarByteStreamSplitFloat(v); });
    ok &= roundtrip_deterministic_blob(
        root, "col_bss_lz4_float_empty.bin", "roundtrip bss_lz4 empty frame",
        [](const std::vector<std::uint8_t> &b) { return colenc::decodeBssLz4Float(b); },
        [](const std::vector<float> &v) { return colenc::encodeBssLz4Float(v); });

    if (!ok) {
        return 1;
    }
    return 0;
}
