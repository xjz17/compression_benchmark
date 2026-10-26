#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace colenc {

inline constexpr const char *kDeltaBinaryPackedAlgorithm = "DELTA_BINARY_PACKED";

inline bool isDeltaBinaryPackedAlgorithm(const std::string &name) {
    return name == kDeltaBinaryPackedAlgorithm;
}

inline bool shouldRunDeltaBinaryPacked(const std::unordered_set<std::string> &existing) {
    return existing.find(kDeltaBinaryPackedAlgorithm) == existing.end();
}

/** Little-endian layout matches Java `ColumnarEncodings.encodeDeltaZigZagVarint`. */
std::vector<std::uint8_t> encodeDeltaZigZagVarint(const std::vector<int> &values);

/** Dictionary of float bit-patterns; matches Java `ColumnarEncodings.encodeDictionaryFloat`. */
std::vector<std::uint8_t> encodeDictionaryFloat(const std::vector<float> &values);

/** Matches Java `ColumnarEncodings.encodeDictionaryInt`. */
std::vector<std::uint8_t> encodeDictionaryInt(const std::vector<int> &values);

/** Matches Java `ColumnarEncodings.encodeRleInt`. */
std::vector<std::uint8_t> encodeRleInt(const std::vector<int> &values);

/** Matches Java `ColumnarEncodings.decodeDeltaZigZagVarint`. */
std::vector<int> decodeDeltaZigZagVarint(const std::vector<std::uint8_t> &blob);

/** Inverse of `encodeDictionaryInt` (Java has encode only; layout matches dictionary section). */
std::vector<int> decodeDictionaryInt(const std::vector<std::uint8_t> &blob);

/** Inverse of `encodeDictionaryFloat`. */
std::vector<float> decodeDictionaryFloat(const std::vector<std::uint8_t> &blob);

/** Inverse of `encodeRleInt`. */
std::vector<int> decodeRleInt(const std::vector<std::uint8_t> &blob);

/** BSS of LE floats, then LZ4 frame. Compressed bytes may differ from Java (different block encoder). */
std::vector<std::uint8_t> encodeBssLz4Float(const std::vector<float> &values);

/** Decompress LZ4 frame, inverse BSS, LE floats (interoperable with Java-framed blobs). */
std::vector<float> decodeBssLz4Float(const std::vector<std::uint8_t> &blob);

/** LE floats then `byteStreamSplitFloat`; matches Java `ColumnarEncodings.byteStreamSplitFloat` on LE raw. */
std::vector<std::uint8_t> planarByteStreamSplitFloat(const std::vector<float> &values);

/** Inverse BSS then LE floats; matches Java `inverseByteStreamSplitFloat` + buffer wrap. */
std::vector<float> floatsFromPlanarByteStreamSplit(const std::vector<std::uint8_t> &bss);

void benchDeltaZigZagVarint(const std::vector<std::vector<int>> &columns,
                            std::vector<long long> &result, LossAccum *loss = nullptr);

void benchDictionaryInt(const std::vector<std::vector<int>> &columns,
                        std::vector<long long> &result, LossAccum *loss = nullptr);

void benchDictionaryFloat(const std::vector<std::vector<float>> &columns,
                          std::vector<long long> &result, LossAccum *loss = nullptr);
void benchDictionaryInt64(const std::vector<std::vector<std::int64_t>> &columns,
                          std::vector<long long> &result, bool double_wire = true,
                          LossAccum *loss = nullptr);
void benchDictionaryDouble64(const std::vector<std::vector<double>> &columns,
                             std::vector<long long> &result, LossAccum *loss = nullptr);

void benchRleInt(const std::vector<std::vector<int>> &columns,
                 std::vector<long long> &result, LossAccum *loss = nullptr);

void benchDeltaZigZagVarintInt64(const std::vector<std::vector<std::int64_t>> &columns,
                                 std::vector<long long> &result, bool double_wire = true,
                                 LossAccum *loss = nullptr);

/** Complete DBF1 float wire, including scale/mode preparation and reconstruction. */
void benchDbpFloat64(const std::vector<std::vector<double>> &columns,
                    std::vector<long long> &result,
                    const std::vector<int> *max_points = nullptr, LossAccum *loss = nullptr);

std::vector<std::uint8_t> encodeDeltaBinaryPackedInt64Ablation(
    const std::vector<std::int64_t> &values, std::string_view omitted_operator);
std::vector<std::int64_t> decodeDeltaBinaryPackedInt64Ablation(
    const std::vector<std::uint8_t> &payload, std::size_t expected_count);

void benchRleInt64(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                   bool double_wire = true, LossAccum *loss = nullptr);

void benchBssLz4Float(const std::vector<std::vector<float>> &columns,
                      std::vector<long long> &result, LossAccum *loss = nullptr);
void benchBssLz4Double64(const std::vector<std::vector<double>> &columns,
                         std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace colenc
