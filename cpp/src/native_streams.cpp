#include "native_streams.hpp"

#include "benchmark_loss.hpp"

#include <lz4.h>
#include <lz4hc.h>
#include <lz4frame.h>
#include <snappy.h>
#include <zlib.h>
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

static void push_ns_timing(std::vector<long long> &result, std::size_t rawSize, std::size_t compSize,
                           long long encNs, long long decNs) {
    result.resize(4);
    result[0] = static_cast<long long>(rawSize);
    result[1] = static_cast<long long>(compSize);
    result[2] = encNs;
    result[3] = decNs;
}

void benchGzFramed(const std::vector<char> &data, std::vector<long long> &result, LossAccum *loss) {
    if (data.empty()) {
        throw std::runtime_error("empty column");
    }

    z_stream strm{};
    strm.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(data.data()));
    strm.avail_in = static_cast<uInt>(data.size());
    int ret = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8,
                           Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        throw std::runtime_error("deflateInit2 failed");
    }

    std::vector<char> compressed;
    compressed.reserve(deflateBound(&strm, static_cast<uLong>(data.size())));

    auto t0 = std::chrono::high_resolution_clock::now();
    do {
        char chunk[16384];
        strm.next_out = reinterpret_cast<Bytef *>(chunk);
        strm.avail_out = sizeof(chunk);
        ret = deflate(&strm, Z_FINISH);
        compressed.insert(compressed.end(), chunk,
                          chunk + (sizeof(chunk) - strm.avail_out));
    } while (ret == Z_OK);
    deflateEnd(&strm);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ret != Z_STREAM_END) {
        throw std::runtime_error("gzip compress failed");
    }

    z_stream zs{};
    zs.next_in = reinterpret_cast<Bytef *>(compressed.data());
    zs.avail_in = static_cast<uInt>(compressed.size());
    ret = inflateInit2(&zs, 16 + MAX_WBITS);
    if (ret != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }

    std::vector<char> dec;
    dec.reserve(data.size());

    auto t2 = std::chrono::high_resolution_clock::now();
    // Must use Z_NO_FLUSH until Z_STREAM_END: calling Z_FINISH on every iteration makes
    // system zlib return Z_BUF_ERROR (-5) mid-stream even with non-zero avail_out.
    // zlib also maps Z_OK -> Z_BUF_ERROR when a call made no input/output progress (inflate.c
    // inf_leave); that is not fatal — call inflate() again (see zlib.h inflate() notes).
    char outbuf[16384];
    std::size_t inflateIters = 0;
    // Worst case ~one loop per 16KiB out; add headroom for Z_BUF_ERROR no-progress retries.
    const std::size_t kMaxInflateIters =
        256 + (data.size() / 8192) + compressed.size() + 1024;
    for (;;) {
        uInt avail_in_before = zs.avail_in;
        zs.next_out = reinterpret_cast<Bytef *>(outbuf);
        zs.avail_out = sizeof(outbuf);
        ret = inflate(&zs, Z_NO_FLUSH);
        dec.insert(dec.end(), outbuf, outbuf + (sizeof(outbuf) - zs.avail_out));
        bool progressed = (zs.avail_in != avail_in_before) ||
                          (sizeof(outbuf) - zs.avail_out) > 0;
        if (ret == Z_STREAM_END) {
            break;
        }
        if (ret == Z_NEED_DICT) {
            inflateEnd(&zs);
            throw std::runtime_error("gzip decompress failed: preset dictionary required");
        }
        if (ret != Z_OK && ret != Z_BUF_ERROR) {
            inflateEnd(&zs);
            const char *msg = zs.msg ? zs.msg : "";
            throw std::runtime_error(std::string("gzip decompress failed: ") + msg);
        }
        ++inflateIters;
        if (inflateIters > kMaxInflateIters ||
            (!progressed && ret == Z_BUF_ERROR && zs.avail_in == 0)) {
            inflateEnd(&zs);
            throw std::runtime_error("gzip decompress failed: stalled or truncated stream");
        }
    }
    inflateEnd(&zs);
    auto t3 = std::chrono::high_resolution_clock::now();

    if (dec.size() != data.size()) {
        if (loss != nullptr) {
            benchmark_loss::accumulate_bytes(data, dec, *loss);
        }
        throw std::runtime_error("gzip size mismatch");
    }

    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, dec, *loss);
    }

    auto encNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto decNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    push_ns_timing(result, data.size(), compressed.size(), encNs, decNs);
}

void benchGzRawDeflate(const std::vector<char> &data, std::vector<long long> &result,
                       LossAccum *loss) {
    if (data.empty()) {
        throw std::runtime_error("empty column");
    }

    z_stream encoder{};
    int ret = deflateInit2(&encoder, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                           Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        throw std::runtime_error("raw deflateInit2 failed");
    }
    encoder.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(data.data()));
    encoder.avail_in = static_cast<uInt>(data.size());
    std::vector<char> compressed;
    compressed.reserve(deflateBound(&encoder, static_cast<uLong>(data.size())));

    const auto t0 = std::chrono::high_resolution_clock::now();
    do {
        char chunk[16384];
        encoder.next_out = reinterpret_cast<Bytef *>(chunk);
        encoder.avail_out = sizeof(chunk);
        ret = deflate(&encoder, Z_FINISH);
        compressed.insert(compressed.end(), chunk, chunk + (sizeof(chunk) - encoder.avail_out));
    } while (ret == Z_OK);
    deflateEnd(&encoder);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (ret != Z_STREAM_END) {
        throw std::runtime_error("raw deflate compression failed");
    }

    z_stream decoder{};
    ret = inflateInit2(&decoder, -MAX_WBITS);
    if (ret != Z_OK) {
        throw std::runtime_error("raw inflateInit2 failed");
    }
    decoder.next_in = reinterpret_cast<Bytef *>(compressed.data());
    decoder.avail_in = static_cast<uInt>(compressed.size());
    std::vector<char> decoded(data.size());
    decoder.next_out = reinterpret_cast<Bytef *>(decoded.data());
    decoder.avail_out = static_cast<uInt>(decoded.size());

    const auto t2 = std::chrono::high_resolution_clock::now();
    ret = inflate(&decoder, Z_FINISH);
    const auto t3 = std::chrono::high_resolution_clock::now();
    const std::size_t decoded_size = decoder.total_out;
    inflateEnd(&decoder);
    if (ret != Z_STREAM_END || decoded_size != data.size() || decoded != data) {
        if (loss != nullptr) {
            decoded.resize(std::min(decoded_size, decoded.size()));
            benchmark_loss::accumulate_bytes(data, decoded, *loss);
        }
        throw std::runtime_error("raw DEFLATE roundtrip failed");
    }
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decoded, *loss);
    }
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

namespace {

void bench_deflate_variant(const std::vector<char> &data, int window_bits, int level, int strategy,
                           std::vector<long long> &result, LossAccum *loss) {
    if (data.empty()) {
        throw std::runtime_error("empty column");
    }
    z_stream encoder{};
    int ret = deflateInit2(&encoder, level, Z_DEFLATED, window_bits, 8, strategy);
    if (ret != Z_OK) {
        throw std::runtime_error("deflate variant initialization failed");
    }
    encoder.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(data.data()));
    encoder.avail_in = static_cast<uInt>(data.size());
    std::vector<char> compressed;
    compressed.reserve(deflateBound(&encoder, static_cast<uLong>(data.size())));
    const auto t0 = std::chrono::high_resolution_clock::now();
    do {
        char chunk[16384];
        encoder.next_out = reinterpret_cast<Bytef *>(chunk);
        encoder.avail_out = sizeof(chunk);
        ret = deflate(&encoder, Z_FINISH);
        compressed.insert(compressed.end(), chunk, chunk + (sizeof(chunk) - encoder.avail_out));
    } while (ret == Z_OK);
    deflateEnd(&encoder);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (ret != Z_STREAM_END) {
        throw std::runtime_error("deflate variant compression failed");
    }

    z_stream decoder{};
    ret = inflateInit2(&decoder, window_bits);
    if (ret != Z_OK) {
        throw std::runtime_error("inflate variant initialization failed");
    }
    decoder.next_in = reinterpret_cast<Bytef *>(compressed.data());
    decoder.avail_in = static_cast<uInt>(compressed.size());
    std::vector<char> decoded(data.size());
    decoder.next_out = reinterpret_cast<Bytef *>(decoded.data());
    decoder.avail_out = static_cast<uInt>(decoded.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    ret = inflate(&decoder, Z_FINISH);
    const auto t3 = std::chrono::high_resolution_clock::now();
    const std::size_t decoded_size = decoder.total_out;
    inflateEnd(&decoder);
    if (ret != Z_STREAM_END || decoded_size != data.size() || decoded != data) {
        throw std::runtime_error("DEFLATE variant roundtrip failed");
    }
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decoded, *loss);
    }
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

}  // namespace

void benchZlibWrapped(const std::vector<char> &data, std::vector<long long> &result,
                      LossAccum *loss) {
    bench_deflate_variant(data, MAX_WBITS, 9, Z_DEFAULT_STRATEGY, result, loss);
}

void benchZlibRawDeflate(const std::vector<char> &data, std::vector<long long> &result,
                         LossAccum *loss) {
    bench_deflate_variant(data, -MAX_WBITS, 9, Z_DEFAULT_STRATEGY, result, loss);
}

void benchDeflateOperatorAblation(const std::vector<char> &data, std::string_view method,
                                  std::string_view omitted_operator,
                                  std::vector<long long> &result, LossAccum *loss) {
    const int window_bits = method == "gzip" ? MAX_WBITS + 16 : MAX_WBITS;
    if (method != "gzip" && method != "zlib") {
        throw std::invalid_argument("unsupported DEFLATE method");
    }
    int level = method == "gzip" ? Z_DEFAULT_COMPRESSION : 9;
    int strategy = Z_DEFAULT_STRATEGY;
    if (omitted_operator == "lz_match" || omitted_operator == "match_parse_policy") {
        strategy = Z_HUFFMAN_ONLY;
    } else if (omitted_operator == "huffman code") {
        level = Z_NO_COMPRESSION;
    } else if (!omitted_operator.empty()) {
        throw std::invalid_argument("unsupported DEFLATE operator ablation");
    }
    bench_deflate_variant(data, window_bits, level, strategy, result, loss);
}

/** Match Apache Commons ``FramedLZ4CompressorOutputStream.Parameters.DEFAULT``. */
static void apply_apache_framed_lz4_prefs(LZ4F_preferences_t *prefs) {
    *prefs = LZ4F_INIT_PREFERENCES;
    prefs->frameInfo.blockSizeID = LZ4F_max4MB;
    prefs->frameInfo.blockMode = LZ4F_blockIndependent;
    prefs->frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
    prefs->frameInfo.blockChecksumFlag = LZ4F_noBlockChecksum;
    prefs->autoFlush = 1;
}

void benchLz4Framed(const std::vector<char> &data, std::vector<long long> &result, LossAccum *loss) {
    if (data.empty()) {
        throw std::runtime_error("empty column");
    }
    LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
    apply_apache_framed_lz4_prefs(&prefs);
    size_t dstCap = LZ4F_compressFrameBound(data.size(), &prefs);
    std::vector<char> compressed(dstCap);

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t produced =
        LZ4F_compressFrame(compressed.data(), dstCap, data.data(), data.size(), &prefs);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (LZ4F_isError(produced)) {
        throw std::runtime_error("LZ4F_compressFrame failed");
    }
    compressed.resize(produced);

    std::vector<char> dst(data.size());
    LZ4F_dctx *dctx = nullptr;
    LZ4F_errorCode_t er = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
    if (LZ4F_isError(er)) {
        throw std::runtime_error("LZ4F_createDecompressionContext failed");
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    size_t dstPos = 0;
    size_t srcPos = 0;
    while (srcPos < compressed.size()) {
        size_t dstChunk = dst.size() - dstPos;
        size_t srcChunk = compressed.size() - srcPos;
        size_t hint = LZ4F_decompress(dctx, dst.data() + dstPos, &dstChunk,
                                      compressed.data() + srcPos, &srcChunk, nullptr);
        if (LZ4F_isError(hint)) {
            LZ4F_freeDecompressionContext(dctx);
            throw std::runtime_error("LZ4F_decompress failed");
        }
        srcPos += srcChunk;
        dstPos += dstChunk;
        if (hint == 0) {
            break;
        }
    }
    LZ4F_freeDecompressionContext(dctx);
    auto t3 = std::chrono::high_resolution_clock::now();

    if (dstPos != data.size() || std::memcmp(dst.data(), data.data(), data.size()) != 0) {
        if (loss != nullptr) {
            std::vector<char> got(dst.begin(), dst.begin() + dstPos);
            benchmark_loss::accumulate_bytes(data, got, *loss);
        }
        throw std::runtime_error("LZ4 frame roundtrip failed");
    }

    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, dst, *loss);
    }

    auto encNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto decNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    push_ns_timing(result, data.size(), compressed.size(), encNs, decNs);
}

void benchLz4RawBlock(const std::vector<char> &data, std::vector<long long> &result,
                      LossAccum *loss) {
    if (data.empty() || data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid raw LZ4 column size");
    }
    const int input_size = static_cast<int>(data.size());
    std::vector<char> compressed(static_cast<std::size_t>(LZ4_compressBound(input_size)));
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int compressed_size = LZ4_compress_default(
        data.data(), compressed.data(), input_size, static_cast<int>(compressed.size()));
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (compressed_size <= 0) {
        throw std::runtime_error("raw LZ4 compression failed");
    }
    compressed.resize(static_cast<std::size_t>(compressed_size));

    std::vector<char> decoded(data.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    const int decoded_size = LZ4_decompress_safe(
        compressed.data(), decoded.data(), compressed_size, input_size);
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (decoded_size != input_size || decoded != data) {
        if (loss != nullptr) {
            decoded.resize(decoded_size > 0 ? static_cast<std::size_t>(decoded_size) : 0);
            benchmark_loss::accumulate_bytes(data, decoded, *loss);
        }
        throw std::runtime_error("raw LZ4 roundtrip failed");
    }
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decoded, *loss);
    }
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

namespace {

void bench_lz4_framed_level(const std::vector<char> &data, int level,
                            std::vector<long long> &result, LossAccum *loss) {
    if (data.empty()) {
        throw std::runtime_error("empty column");
    }
    LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
    apply_apache_framed_lz4_prefs(&prefs);
    prefs.compressionLevel = level;
    const std::size_t capacity = LZ4F_compressFrameBound(data.size(), &prefs);
    std::vector<char> compressed(capacity);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::size_t produced = LZ4F_compressFrame(
        compressed.data(), capacity, data.data(), data.size(), &prefs);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (LZ4F_isError(produced)) {
        throw std::runtime_error("LZ4 frame variant compression failed");
    }
    compressed.resize(produced);
    std::vector<char> decoded(data.size());
    LZ4F_dctx *context = nullptr;
    if (LZ4F_isError(LZ4F_createDecompressionContext(&context, LZ4F_VERSION))) {
        throw std::runtime_error("LZ4 frame variant decoder initialization failed");
    }
    std::size_t input_position = 0;
    std::size_t output_position = 0;
    const auto t2 = std::chrono::high_resolution_clock::now();
    while (input_position < compressed.size()) {
        std::size_t input_size = compressed.size() - input_position;
        std::size_t output_size = decoded.size() - output_position;
        const std::size_t hint = LZ4F_decompress(
            context, decoded.data() + output_position, &output_size,
            compressed.data() + input_position, &input_size, nullptr);
        if (LZ4F_isError(hint)) {
            LZ4F_freeDecompressionContext(context);
            throw std::runtime_error("LZ4 frame variant decompression failed");
        }
        input_position += input_size;
        output_position += output_size;
        if (hint == 0) break;
    }
    const auto t3 = std::chrono::high_resolution_clock::now();
    LZ4F_freeDecompressionContext(context);
    if (output_position != data.size() || decoded != data) {
        throw std::runtime_error("LZ4 frame variant roundtrip failed");
    }
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decoded, *loss);
    }
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

void bench_lz4_raw_variant(const std::vector<char> &data, int level,
                           std::vector<long long> &result, LossAccum *loss) {
    if (data.empty() || data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid raw LZ4 variant column size");
    }
    const int input_size = static_cast<int>(data.size());
    std::vector<char> compressed(static_cast<std::size_t>(LZ4_compressBound(input_size)));
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int compressed_size = level > 0
        ? LZ4_compress_HC(data.data(), compressed.data(), input_size,
                          static_cast<int>(compressed.size()), level)
        : LZ4_compress_fast(data.data(), compressed.data(), input_size,
                            static_cast<int>(compressed.size()), -level);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (compressed_size <= 0) {
        throw std::runtime_error("raw LZ4 variant compression failed");
    }
    compressed.resize(static_cast<std::size_t>(compressed_size));
    std::vector<char> decoded(data.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    const int decoded_size = LZ4_decompress_safe(
        compressed.data(), decoded.data(), compressed_size, input_size);
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (decoded_size != input_size || decoded != data) {
        throw std::runtime_error("raw LZ4 variant roundtrip failed");
    }
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decoded, *loss);
    }
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

}  // namespace

void benchLz4FastFramed(const std::vector<char> &data, std::vector<long long> &result,
                        LossAccum *loss) {
    bench_lz4_framed_level(data, -17, result, loss);
}

void benchLz4FastRawBlock(const std::vector<char> &data, std::vector<long long> &result,
                          LossAccum *loss) {
    bench_lz4_raw_variant(data, -17, result, loss);
}

void benchLz4HcFramed(const std::vector<char> &data, std::vector<long long> &result,
                      LossAccum *loss) {
    bench_lz4_framed_level(data, 12, result, loss);
}

void benchLz4HcRawBlock(const std::vector<char> &data, std::vector<long long> &result,
                        LossAccum *loss) {
    bench_lz4_raw_variant(data, 12, result, loss);
}

void benchLz4OperatorAblation(const std::vector<char> &data, std::string_view method,
                              std::string_view omitted_operator,
                              std::vector<long long> &result, LossAccum *loss) {
    if (omitted_operator.empty()) {
        if (method == "LZ4") return benchLz4RawBlock(data, result, loss);
        if (method == "LZ4 Fast") return benchLz4FastRawBlock(data, result, loss);
        if (method == "LZ4HC") return benchLz4HcRawBlock(data, result, loss);
        throw std::invalid_argument("unsupported LZ4 method");
    }
    if (omitted_operator != "lz_match" && omitted_operator != "match_parse_policy" &&
        omitted_operator != "match_token_format" && omitted_operator != "integer coding") {
        throw std::invalid_argument("unsupported LZ4 operator ablation");
    }
    if (data.empty() || data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid literal-only LZ4 column size");
    }

    std::vector<char> compressed;
    compressed.reserve(data.size() + data.size() / 255 + 8);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::size_t literal_length = data.size();
    compressed.push_back(static_cast<char>((std::min<std::size_t>(literal_length, 15)) << 4));
    if (literal_length >= 15) {
        literal_length -= 15;
        while (literal_length >= 255) {
            compressed.push_back(static_cast<char>(255));
            literal_length -= 255;
        }
        compressed.push_back(static_cast<char>(literal_length));
    }
    compressed.insert(compressed.end(), data.begin(), data.end());
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<char> decoded(data.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    const int decoded_size = LZ4_decompress_safe(
        compressed.data(), decoded.data(), static_cast<int>(compressed.size()),
        static_cast<int>(decoded.size()));
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (decoded_size != static_cast<int>(data.size()) || decoded != data) {
        throw std::runtime_error("literal-only LZ4 roundtrip failed");
    }
    if (loss != nullptr) benchmark_loss::accumulate_bytes(data, decoded, *loss);
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

namespace {

void append_u24(std::vector<char> &buf, std::uint32_t v) {
    buf.push_back(static_cast<char>(v & 0xff));
    buf.push_back(static_cast<char>((v >> 8) & 0xff));
    buf.push_back(static_cast<char>((v >> 16) & 0xff));
}

void write_framed_chunk(std::vector<char> &out, std::uint8_t type, const char *pay,
                        std::size_t payLen) {
    out.push_back(static_cast<char>(type));
    if (payLen > 0xffffffu) {
        throw std::runtime_error("snappy chunk too large");
    }
    append_u24(out, static_cast<std::uint32_t>(payLen));
    out.insert(out.end(), pay, pay + payLen);
}

void append_snappy_varint(std::vector<char> &out, std::uint32_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<char>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

std::vector<char> snappy_literal_only_block(const char *data, std::size_t size) {
    if (size == 0 || size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid Snappy literal block size");
    }
    std::vector<char> out;
    out.reserve(size + 8);
    append_snappy_varint(out, static_cast<std::uint32_t>(size));
    const std::uint32_t length_minus_one = static_cast<std::uint32_t>(size - 1);
    if (size < 60) {
        out.push_back(static_cast<char>(length_minus_one << 2));
    } else {
        int bytes = 1;
        while (bytes < 4 && (length_minus_one >> (8 * bytes)) != 0) ++bytes;
        out.push_back(static_cast<char>((59 + bytes) << 2));
        for (int i = 0; i < bytes; ++i) {
            out.push_back(static_cast<char>((length_minus_one >> (8 * i)) & 0xff));
        }
    }
    out.insert(out.end(), data, data + size);
    return out;
}

void bench_snappy_framed(const std::vector<char> &data, bool literal_only,
                          std::vector<long long> &result, LossAccum *loss) {
    if (data.empty()) throw std::runtime_error("empty column");
    constexpr std::size_t kMaxUncomp = 65536;
    std::vector<char> compressed;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const char id[] = {'s', 'N', 'a', 'P', 'p', 'Y'};
    write_framed_chunk(compressed, 0xff, id, sizeof id);
    for (std::size_t off = 0; off < data.size();) {
        const std::size_t n = std::min(kMaxUncomp, data.size() - off);
        std::vector<char> block;
        if (literal_only) {
            block = snappy_literal_only_block(data.data() + off, n);
        } else {
            block.resize(snappy::MaxCompressedLength(n));
            std::size_t compressed_length = 0;
            snappy::RawCompress(data.data() + off, n, block.data(), &compressed_length);
            block.resize(compressed_length);
        }
        write_framed_chunk(compressed, 0x00, block.data(), block.size());
        off += n;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<char> decoded;
    decoded.reserve(data.size());
    std::size_t position = 0;
    const auto t2 = std::chrono::high_resolution_clock::now();
    while (position < compressed.size()) {
        if (position + 4 > compressed.size()) throw std::runtime_error("snappy frame truncated");
        const std::uint8_t type = static_cast<std::uint8_t>(compressed[position++]);
        const std::uint32_t length =
            static_cast<std::uint32_t>(static_cast<std::uint8_t>(compressed[position])) |
            (static_cast<std::uint32_t>(static_cast<std::uint8_t>(compressed[position + 1])) << 8) |
            (static_cast<std::uint32_t>(static_cast<std::uint8_t>(compressed[position + 2])) << 16);
        position += 3;
        if (position + length > compressed.size()) throw std::runtime_error("snappy frame payload truncated");
        if (type == 0xff) {
            if (length != sizeof id || std::memcmp(compressed.data() + position, id, sizeof id) != 0) {
                throw std::runtime_error("snappy stream id bad");
            }
        } else if (type == 0x00) {
            std::size_t uncompressed_length = 0;
            if (!snappy::GetUncompressedLength(compressed.data() + position, length,
                                               &uncompressed_length)) {
                throw std::runtime_error("snappy get length failed");
            }
            const std::size_t old_size = decoded.size();
            decoded.resize(old_size + uncompressed_length);
            if (!snappy::RawUncompress(compressed.data() + position, length,
                                       decoded.data() + old_size)) {
                throw std::runtime_error("snappy raw uncompress failed");
            }
        }
        position += length;
    }
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (decoded != data) throw std::runtime_error("snappy framed roundtrip mismatch");
    if (loss != nullptr) benchmark_loss::accumulate_bytes(data, decoded, *loss);
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}

}  // namespace

void benchSnappyFramed(const std::vector<char> &data, std::vector<long long> &result, LossAccum *loss) {
    bench_snappy_framed(data, false, result, loss);
}

void benchSnappyFramedLiteralOnly(const std::vector<char> &data,
                                  std::vector<long long> &result, LossAccum *loss) {
    bench_snappy_framed(data, true, result, loss);
}

void benchZstdOperatorAblation(const std::vector<char> &data,
                               std::string_view omitted_operator,
                               std::vector<long long> &result, LossAccum *loss) {
    if (data.empty()) throw std::runtime_error("empty column");
    std::vector<char> compressed(ZSTD_compressBound(data.size()));
    ZSTD_CCtx *context = ZSTD_createCCtx();
    if (context == nullptr) throw std::runtime_error("ZSTD_createCCtx failed");
    auto check_parameter = [&](ZSTD_cParameter parameter, int value) {
        const std::size_t status = ZSTD_CCtx_setParameter(context, parameter, value);
        if (ZSTD_isError(status)) {
            ZSTD_freeCCtx(context);
            throw std::runtime_error(std::string("Zstd parameter failed: ") +
                                     ZSTD_getErrorName(status));
        }
    };
    check_parameter(ZSTD_c_compressionLevel, 9);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::size_t compressed_size = 0;
    if (omitted_operator.empty()) {
        compressed_size = ZSTD_compress2(context, compressed.data(), compressed.size(),
                                         data.data(), data.size());
    } else if (omitted_operator == "literal code") {
        check_parameter(ZSTD_c_literalCompressionMode, ZSTD_lcm_uncompressed);
        compressed_size = ZSTD_compress2(context, compressed.data(), compressed.size(),
                                         data.data(), data.size());
    } else if (omitted_operator == "lz_match" ||
               omitted_operator == "match_parse_policy" ||
               omitted_operator == "fse/ans code") {
        compressed_size = ZSTD_compressSequences(
            context, compressed.data(), compressed.size(), nullptr, 0,
            data.data(), data.size());
    } else {
        ZSTD_freeCCtx(context);
        throw std::invalid_argument("unsupported Zstd operator ablation");
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    ZSTD_freeCCtx(context);
    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error(std::string("Zstd compression failed: ") +
                                 ZSTD_getErrorName(compressed_size));
    }
    compressed.resize(compressed_size);

    std::vector<char> decoded(data.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    const std::size_t decoded_size = ZSTD_decompress(
        decoded.data(), decoded.size(), compressed.data(), compressed.size());
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (ZSTD_isError(decoded_size) || decoded_size != data.size() || decoded != data) {
        throw std::runtime_error("Zstd operator-ablation roundtrip failed");
    }
    if (loss != nullptr) benchmark_loss::accumulate_bytes(data, decoded, *loss);
    push_ns_timing(
        result, data.size(), compressed.size(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
}
