#include "blosc_codec.hpp"
#include "benchmark_loss.hpp"

#include <blosc.h>
#include <blosclz.h>

#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace wc_blosc1_bench {

namespace {

constexpr int kClevel = 5;
constexpr int kShuffle = BLOSC_SHUFFLE;
constexpr int kNthreads = 1;

bool env_bench_on() {
    if (const char *env = std::getenv("WEB_COMPRESSION_BLOSC_BENCH")) {
        return env[0] != '0';
    }
    return true;
}

std::vector<float> column_to_float32(const std::vector<double> &col) {
    std::vector<float> out(col.size());
    for (std::size_t i = 0; i < col.size(); ++i) {
        out[i] = static_cast<float>(col[i]);
    }
    return out;
}

bool bench_one_buffer(const void *src, std::size_t nbytes, int typesize, bool is_float_wire,
                      long long &orig, long long &comp, long long &ct, long long &dt, LossAccum *loss) {
    if (nbytes == 0 || nbytes > static_cast<std::size_t>(BLOSC_MAX_BUFFERSIZE)) {
        return false;
    }
    const auto srcsize = static_cast<int>(nbytes);
    std::vector<std::uint8_t> dest(static_cast<std::size_t>(srcsize) + BLOSC_MAX_OVERHEAD);

    blosc_init();
    blosc_set_nthreads(kNthreads);
    if (blosc_set_compressor(BLOSC_BLOSCLZ_COMPNAME) < 0) {
        blosc_destroy();
        return false;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    const int csize = blosc_compress(kClevel, kShuffle, static_cast<size_t>(typesize), srcsize, src, dest.data(),
                                     dest.size());
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (csize <= 0) {
        blosc_destroy();
        return false;
    }

    std::vector<std::uint8_t> decoded(static_cast<std::size_t>(srcsize));
    const auto t2 = std::chrono::high_resolution_clock::now();
    const int dsize = blosc_decompress(dest.data(), decoded.data(), decoded.size());
    const auto t3 = std::chrono::high_resolution_clock::now();
    blosc_destroy();

    if (dsize != srcsize || std::memcmp(src, decoded.data(), nbytes) != 0) {
        return false;
    }

    if (loss != nullptr && is_float_wire) {
        const auto *fsrc = static_cast<const float *>(src);
        const auto *fdec = reinterpret_cast<const float *>(decoded.data());
        const std::size_t n = nbytes / sizeof(float);
        std::vector<double> orig_d(n);
        std::vector<double> dec_d(n);
        for (std::size_t i = 0; i < n; ++i) {
            orig_d[i] = static_cast<double>(fsrc[i]);
            dec_d[i] = static_cast<double>(fdec[i]);
        }
        benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
    } else if (loss != nullptr && typesize == static_cast<int>(sizeof(std::int64_t))) {
        const auto *isrc = static_cast<const std::int64_t *>(src);
        const auto *idec = reinterpret_cast<const std::int64_t *>(decoded.data());
        const std::size_t n = nbytes / sizeof(std::int64_t);
        std::vector<double> orig_d(n);
        std::vector<double> dec_d(n);
        for (std::size_t i = 0; i < n; ++i) {
            orig_d[i] = static_cast<double>(isrc[i]);
            dec_d[i] = static_cast<double>(idec[i]);
        }
        benchmark_loss::accumulate_double_column(orig_d, dec_d, *loss);
    }

    orig = static_cast<long long>(nbytes);
    comp = static_cast<long long>(csize);
    ct = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    return true;
}

} // namespace

bool bench_enabled() { return env_bench_on(); }

std::vector<std::uint8_t> encodeBufferExact(
    std::span<const std::uint8_t> input, int typesize) {
    if (input.size() > static_cast<std::size_t>(BLOSC_MAX_BUFFERSIZE)) {
        throw std::invalid_argument("invalid Blosc1 exact buffer size");
    }
    if (typesize <= 0 || typesize > 255) {
        throw std::invalid_argument("invalid Blosc1 exact typesize");
    }
    std::vector<std::uint8_t> output(input.size() + BLOSC_MAX_OVERHEAD);
    blosc_init();
    blosc_set_nthreads(kNthreads);
    if (blosc_set_compressor(BLOSC_BLOSCLZ_COMPNAME) < 0) {
        blosc_destroy();
        throw std::runtime_error("Blosc1 BLOSCLZ initialization failed");
    }
    std::uint8_t empty_storage = 0;
    const void* source = input.empty() ? static_cast<const void*>(&empty_storage)
                                       : static_cast<const void*>(input.data());
    const int size = blosc_compress(
        kClevel, kShuffle, static_cast<std::size_t>(typesize),
        static_cast<int>(input.size()), source, output.data(), output.size());
    blosc_destroy();
    if (size <= 0) throw std::runtime_error("Blosc1 exact compression failed");
    output.resize(static_cast<std::size_t>(size));
    return output;
}

std::vector<std::uint8_t> decodeBufferExact(
    std::span<const std::uint8_t> encoded) {
    if (encoded.size() < BLOSC_MIN_HEADER_LENGTH) {
        throw std::invalid_argument("truncated Blosc1 exact buffer");
    }
    std::size_t decoded_size = 0;
    std::size_t compressed_size = 0;
    std::size_t block_size = 0;
    if (blosc_cbuffer_validate(encoded.data(), encoded.size(), &decoded_size) != 0) {
        throw std::runtime_error("invalid Blosc1 exact buffer");
    }
    blosc_cbuffer_sizes(encoded.data(), &decoded_size, &compressed_size, &block_size);
    if (compressed_size != encoded.size()) {
        throw std::runtime_error("invalid Blosc1 header sizes");
    }
    if (decoded_size == 0) {
        constexpr std::uint8_t kEmptyFlags =
            BLOSC_DOSHUFFLE | BLOSC_MEMCPYED | 0x10;
        if (encoded.size() != BLOSC_MAX_OVERHEAD || block_size != 1 ||
            encoded[0] != BLOSC_VERSION_FORMAT ||
            encoded[1] != BLOSC_BLOSCLZ_VERSION_FORMAT ||
            encoded[2] != kEmptyFlags || encoded[3] == 0) {
            throw std::runtime_error("invalid Blosc1 empty buffer");
        }
    }
    std::vector<std::uint8_t> output(decoded_size);
    std::uint8_t empty_storage = 0;
    void* destination = output.empty() ? static_cast<void*>(&empty_storage)
                                       : static_cast<void*>(output.data());
    blosc_init();
    blosc_set_nthreads(kNthreads);
    const int size = blosc_decompress(encoded.data(), destination, output.size());
    blosc_destroy();
    if (size != static_cast<int>(decoded_size)) {
        throw std::runtime_error("Blosc1 exact decompression failed");
    }
    return output;
}

void benchFloatColumns(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                       LossAccum *loss) {
    if (!bench_enabled()) {
        result.assign(4, 0);
        return;
    }
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        const std::vector<float> fcol = column_to_float32(col);
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_buffer(fcol.data(), fcol.size() * sizeof(float), static_cast<int>(sizeof(float)), true, o, c,
                             tenc, tdec, loss)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchInt64Columns(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                       LossAccum *loss) {
    if (!bench_enabled()) {
        result.assign(4, 0);
        return;
    }
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_buffer(col.data(), col.size() * sizeof(std::int64_t), static_cast<int>(sizeof(std::int64_t)),
                             false, o, c, tenc, tdec, loss)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchByteColumns(const std::vector<std::vector<char>> &columns, std::vector<long long> &result,
                      LossAccum *loss) {
    if (!bench_enabled()) {
        result.assign(4, 0);
        return;
    }
    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_buffer(col.data(), col.size(), 1, false, o, c, tenc, tdec, loss)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
    result.assign({orig, comp, ct, dt});
}

void benchBufferOperatorAblation(const std::vector<char> &data, int typesize,
                                 std::string_view omitted_operator,
                                 std::vector<long long> &result, LossAccum *loss) {
    if (data.empty() || data.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error("invalid Blosc ablation buffer size");
    }
    if (typesize <= 0) throw std::invalid_argument("invalid Blosc typesize");
    const bool raw_blosclz = omitted_operator == "block/state";
    if (!omitted_operator.empty() && omitted_operator != "block transform" &&
        omitted_operator != "lz_match" && omitted_operator != "match_token_format" &&
        !raw_blosclz) {
        throw std::invalid_argument("unsupported Blosc operator ablation");
    }

    long long original_bytes = 0;
    long long compressed_bytes = 0;
    long long compression_ns = 0;
    long long decompression_ns = 0;
    constexpr std::size_t max_chunk = 1U << 20;
    blosc_init();
    blosc_set_nthreads(kNthreads);
    if (blosc_set_compressor(BLOSC_BLOSCLZ_COMPNAME) < 0) {
        blosc_destroy();
        throw std::runtime_error("Blosc ablation compressor initialization failed");
    }
    for (std::size_t offset = 0; offset < data.size();) {
        std::size_t chunk_size = std::min(max_chunk, data.size() - offset);
        if (chunk_size > static_cast<std::size_t>(typesize)) {
            chunk_size -= chunk_size % static_cast<std::size_t>(typesize);
        }
        if (chunk_size == 0) chunk_size = data.size() - offset;
        const char *chunk_data = data.data() + offset;
        std::vector<std::uint8_t> compressed(
            chunk_size * 2 + BLOSC_MAX_OVERHEAD + 66);
        int compressed_size = 0;
        bool raw_fallback = false;
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (raw_blosclz) {
            compressed_size = blosclz_compress(
                5, chunk_data, static_cast<int>(chunk_size), compressed.data(),
                static_cast<int>(compressed.size()), 0);
        } else {
            const int shuffle = omitted_operator == "block transform"
                ? BLOSC_NOSHUFFLE : BLOSC_SHUFFLE;
            const int level = omitted_operator == "lz_match" ||
                    omitted_operator == "match_token_format" ? 0 : kClevel;
            compressed_size = blosc_compress(
                level, shuffle, static_cast<std::size_t>(typesize),
                static_cast<int>(chunk_size), chunk_data, compressed.data(),
                compressed.size());
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (compressed_size <= 0) {
            raw_fallback = true;
            compressed.assign(reinterpret_cast<const std::uint8_t *>(chunk_data),
                              reinterpret_cast<const std::uint8_t *>(chunk_data) +
                                  chunk_size);
            compressed_size = static_cast<int>(chunk_size);
        }
        compressed.resize(static_cast<std::size_t>(compressed_size));
        std::vector<std::uint8_t> decoded(chunk_size);
        const auto t2 = std::chrono::high_resolution_clock::now();
        int decoded_size = 0;
        if (raw_fallback) {
            std::memcpy(decoded.data(), compressed.data(), chunk_size);
            decoded_size = static_cast<int>(chunk_size);
        } else {
            decoded_size = raw_blosclz
                ? blosclz_decompress(compressed.data(), compressed_size, decoded.data(),
                                     static_cast<int>(decoded.size()))
                : blosc_decompress(compressed.data(), decoded.data(), decoded.size());
        }
        const auto t3 = std::chrono::high_resolution_clock::now();
        if (decoded_size != static_cast<int>(chunk_size) ||
            std::memcmp(decoded.data(), chunk_data, chunk_size) != 0) {
            throw std::runtime_error("Blosc ablation roundtrip failed");
        }
        if (loss != nullptr) {
            const std::vector<char> original_chars(chunk_data, chunk_data + chunk_size);
            const std::vector<char> decoded_chars(decoded.begin(), decoded.end());
            benchmark_loss::accumulate_bytes(original_chars, decoded_chars, *loss);
        }
        original_bytes += static_cast<long long>(chunk_size);
        compressed_bytes += static_cast<long long>(compressed.size());
        compression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        decompression_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        offset += chunk_size;
    }
    blosc_destroy();
    result.assign({original_bytes, compressed_bytes, compression_ns, decompression_ns});
}

} // namespace wc_blosc1_bench
