#include "blosc2_codec.hpp"
#include "benchmark_loss.hpp"

#include <blosc2.h>
extern "C" {
int blosclz_compress(int level, const void* input, int length, void* output,
                     int max_output, blosc2_context* context);
int blosclz_decompress(const void* input, int length, void* output, int max_output);
}

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace wc_blosc2_bench {

namespace {

constexpr int kTypesizeFloat32 = 4;
constexpr int kTypesizeInt64 = 8;
constexpr int kTypesizeBytes = 1;

bool env_bench_on() {
    if (const char *env = std::getenv("WEB_COMPRESSION_BLOSC2_BENCH")) {
        return env[0] != '0';
    }
    return true;
}

bool is_canonical_empty_header(std::span<const std::uint8_t> encoded,
                               std::int32_t block_size, int typesize) {
    constexpr std::uint8_t kEmptyFlags =
        BLOSC_DOSHUFFLE | BLOSC_MEMCPYED | BLOSC_DOBITSHUFFLE;
    if (encoded.size() != BLOSC_EXTENDED_HEADER_LENGTH || block_size != 1 ||
        typesize <= 0 || typesize > BLOSC_MAX_TYPESIZE ||
        encoded[BLOSC2_CHUNK_VERSION] != BLOSC2_VERSION_FORMAT_STABLE ||
        encoded[BLOSC2_CHUNK_VERSIONLZ] != BLOSC_BLOSCLZ_VERSION_FORMAT ||
        encoded[BLOSC2_CHUNK_FLAGS] != kEmptyFlags ||
        encoded[BLOSC2_CHUNK_TYPESIZE] != static_cast<std::uint8_t>(typesize)) {
        return false;
    }
    const std::size_t shuffle_offset =
        static_cast<std::size_t>(BLOSC2_CHUNK_FILTER_CODES) +
        static_cast<std::size_t>(BLOSC2_MAX_FILTERS) - 1;
    for (std::size_t i = BLOSC2_CHUNK_FILTER_CODES; i < encoded.size(); ++i) {
        const std::uint8_t expected = i == shuffle_offset ? BLOSC_SHUFFLE : 0;
        if (encoded[i] != expected) {
            return false;
        }
    }
    return true;
}

std::string unique_tag() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream o;
    o << std::hex << rng();
    return o.str();
}

fs::path repo_root() {
    if (const char *r = std::getenv("WEB_COMPRESSION_REPO_ROOT"); r != nullptr && r[0] != '\0') {
        return fs::path(r);
    }
    return {};
}

fs::path resolve_blosc2_col_bench_exe() {
    if (const char *env = std::getenv("WEB_COMPRESSION_BLOSC2_COL_BENCH");
        env != nullptr && env[0] != '\0') {
        const fs::path p(env);
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    const fs::path root = repo_root();
    if (root.empty()) {
        return {};
    }
#ifdef _WIN32
    const char *suffix = ".exe";
#else
    const char *suffix = "";
#endif
    const fs::path candidates[] = {
        root / "external" / "compression" / "c-blosc2" / "build" / "bin" /
            ("blosc2_col_bench" + std::string(suffix)),
    };
    for (const auto &c : candidates) {
        if (fs::is_regular_file(c)) {
            return c;
        }
    }
    return {};
}

int run_cli(const fs::path &exe, const std::vector<std::string> &args) {
    std::ostringstream cmd;
#ifdef _WIN32
    cmd << "cmd /c \"\"" << exe.string() << "\"";
    for (const auto &a : args) {
        cmd << " \"" << a << "\"";
    }
    cmd << "\"";
#else
    cmd << "\"" << exe.string() << "\"";
    for (const auto &a : args) {
        cmd << " \"" << a << "\"";
    }
#endif
    return std::system(cmd.str().c_str());
}

bool write_file_bytes(const fs::path &path, const void *data, std::size_t nbytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    return static_cast<bool>(out.write(static_cast<const char *>(data), static_cast<std::streamsize>(nbytes)));
}

bool read_file_bytes(const fs::path &path, std::vector<std::uint8_t> &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff sz = in.tellg();
    if (sz < 0) {
        return false;
    }
    out.resize(static_cast<std::size_t>(sz));
    in.seekg(0, std::ios::beg);
    return static_cast<bool>(in.read(reinterpret_cast<char *>(out.data()), sz));
}

std::uintmax_t file_size_or_zero(const fs::path &path) {
    std::error_code ec;
    const auto sz = fs::file_size(path, ec);
    return ec ? 0 : sz;
}

std::vector<float> column_to_float32(const std::vector<double> &col) {
    std::vector<float> out(col.size());
    for (std::size_t i = 0; i < col.size(); ++i) {
        out[i] = static_cast<float>(col[i]);
    }
    return out;
}

bool bench_one_buffer(const fs::path &exe, const void *src, std::size_t nbytes, int typesize,
                      bool is_float_wire, long long &orig, long long &comp, long long &ct, long long &dt,
                      LossAccum *loss, const fs::path &tmp_root, std::size_t column_index) {
    if (nbytes == 0 || exe.empty()) {
        return false;
    }

    const fs::path in_path = tmp_root / ("in_" + std::to_string(column_index) + ".bin");
    const fs::path comp_path = tmp_root / ("comp_" + std::to_string(column_index) + ".b2");
    const fs::path out_path = tmp_root / ("out_" + std::to_string(column_index) + ".bin");

    std::error_code ec;
    fs::remove(comp_path, ec);
    fs::remove(out_path, ec);
    if (!write_file_bytes(in_path, src, nbytes)) {
        return false;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    if (run_cli(exe, {"-c", "-i", in_path.string(), "-o", comp_path.string(), "-s",
                      std::to_string(typesize)}) != 0) {
        return false;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::uintmax_t csize = file_size_or_zero(comp_path);
    if (csize == 0) {
        return false;
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    if (run_cli(exe, {"-d", "-i", comp_path.string(), "-o", out_path.string()}) != 0) {
        return false;
    }
    const auto t3 = std::chrono::high_resolution_clock::now();

    std::vector<std::uint8_t> decoded;
    if (!read_file_bytes(out_path, decoded) || decoded.size() != nbytes ||
        std::memcmp(src, decoded.data(), nbytes) != 0) {
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
    } else if (loss != nullptr && typesize == kTypesizeInt64) {
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

void accumulate_column_benches(const fs::path &exe, const fs::path &tmp_root,
                                 const std::vector<std::pair<const void *, std::size_t>> &buffers, int typesize,
                                 bool is_float_wire, long long &orig, long long &comp, long long &ct, long long &dt,
                                 LossAccum *loss) {
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto &[data, nbytes] = buffers[i];
        if (data == nullptr || nbytes == 0) {
            continue;
        }
        long long o = 0;
        long long c = 0;
        long long tenc = 0;
        long long tdec = 0;
        if (bench_one_buffer(exe, data, nbytes, typesize, is_float_wire, o, c, tenc, tdec, loss, tmp_root, i)) {
            orig += o;
            comp += c;
            ct += tenc;
            dt += tdec;
        }
    }
}

} // namespace

bool bench_enabled() { return env_bench_on(); }

std::vector<std::uint8_t> encodeBufferExact(
    std::span<const std::uint8_t> input, int typesize) {
    if (input.size() > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("invalid Blosc2 exact buffer size");
    }
    if (typesize <= 0 || typesize > 255) {
        throw std::invalid_argument("invalid Blosc2 exact typesize");
    }
    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = BLOSC_BLOSCLZ;
    cparams.clevel = 5;
    cparams.typesize = static_cast<std::uint8_t>(typesize);
    cparams.nthreads = 1;
    cparams.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE;
    blosc2_context* context = blosc2_create_cctx(cparams);
    if (context == nullptr) throw std::runtime_error("Blosc2 exact cctx creation failed");
    std::vector<std::uint8_t> output(input.size() + BLOSC2_MAX_OVERHEAD);
    std::uint8_t empty_storage = 0;
    const void* source = input.empty() ? static_cast<const void*>(&empty_storage)
                                       : static_cast<const void*>(input.data());
    const int size = blosc2_compress_ctx(
        context, source, static_cast<std::int32_t>(input.size()), output.data(),
        static_cast<std::int32_t>(output.size()));
    blosc2_free_ctx(context);
    if (size <= 0) throw std::runtime_error("Blosc2 exact compression failed");
    output.resize(static_cast<std::size_t>(size));
    return output;
}

std::vector<std::uint8_t> decodeBufferExact(
    std::span<const std::uint8_t> encoded, int typesize) {
    if (encoded.size() < BLOSC_MIN_HEADER_LENGTH ||
        encoded.size() > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("truncated Blosc2 exact buffer");
    }
    std::int32_t decoded_size = 0;
    std::int32_t compressed_size = 0;
    std::int32_t block_size = 0;
    const int status = blosc2_cbuffer_sizes(
        encoded.data(), &decoded_size, &compressed_size, &block_size);
    if (status < 0 || decoded_size < 0 ||
        compressed_size != static_cast<std::int32_t>(encoded.size())) {
        throw std::runtime_error("invalid Blosc2 header sizes");
    }
    if (decoded_size == 0 &&
        !is_canonical_empty_header(encoded, block_size, typesize)) {
        throw std::runtime_error("invalid Blosc2 empty buffer");
    }
    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    dparams.nthreads = 1;
    dparams.typesize = static_cast<std::uint8_t>(typesize);
    blosc2_context* context = blosc2_create_dctx(dparams);
    if (context == nullptr) throw std::runtime_error("Blosc2 exact dctx creation failed");
    std::vector<std::uint8_t> output(static_cast<std::size_t>(decoded_size));
    std::uint8_t empty_storage = 0;
    void* destination = output.empty() ? static_cast<void*>(&empty_storage)
                                       : static_cast<void*>(output.data());
    const int size = blosc2_decompress_ctx(
        context, encoded.data(), static_cast<std::int32_t>(encoded.size()),
        destination, decoded_size);
    blosc2_free_ctx(context);
    if (size != decoded_size) throw std::runtime_error("Blosc2 exact decompression failed");
    return output;
}

void benchFloatColumns(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                       LossAccum *loss) {
    result.assign(4, 0);
    if (!env_bench_on()) {
        return;
    }
    const fs::path exe = resolve_blosc2_col_bench_exe();
    if (exe.empty()) {
        return;
    }
    const fs::path tmp_root = fs::temp_directory_path() / ("wc_blosc2_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmp_root, ec);
    if (ec) {
        return;
    }

    std::vector<std::pair<const void *, std::size_t>> buffers;
    std::vector<std::vector<float>> wire;
    wire.reserve(columns.size());
    buffers.reserve(columns.size());
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        wire.push_back(column_to_float32(col));
        const auto &fcol = wire.back();
        buffers.emplace_back(fcol.data(), fcol.size() * sizeof(float));
    }

    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    accumulate_column_benches(exe, tmp_root, buffers, kTypesizeFloat32, true, orig, comp, ct, dt, loss);
    fs::remove_all(tmp_root, ec);
    result.assign({orig, comp, ct, dt});
}

void benchInt64Columns(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                       LossAccum *loss) {
    result.assign(4, 0);
    if (!env_bench_on()) {
        return;
    }
    const fs::path exe = resolve_blosc2_col_bench_exe();
    if (exe.empty()) {
        return;
    }
    const fs::path tmp_root = fs::temp_directory_path() / ("wc_blosc2_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmp_root, ec);
    if (ec) {
        return;
    }

    std::vector<std::pair<const void *, std::size_t>> buffers;
    buffers.reserve(columns.size());
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        buffers.emplace_back(col.data(), col.size() * sizeof(std::int64_t));
    }

    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    accumulate_column_benches(exe, tmp_root, buffers, kTypesizeInt64, false, orig, comp, ct, dt, loss);
    fs::remove_all(tmp_root, ec);
    result.assign({orig, comp, ct, dt});
}

void benchByteColumns(const std::vector<std::vector<char>> &columns, std::vector<long long> &result,
                      LossAccum *loss) {
    result.assign(4, 0);
    if (!env_bench_on()) {
        return;
    }
    const fs::path exe = resolve_blosc2_col_bench_exe();
    if (exe.empty()) {
        return;
    }
    const fs::path tmp_root = fs::temp_directory_path() / ("wc_blosc2_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmp_root, ec);
    if (ec) {
        return;
    }

    std::vector<std::pair<const void *, std::size_t>> buffers;
    buffers.reserve(columns.size());
    for (const auto &col : columns) {
        if (col.empty()) {
            continue;
        }
        buffers.emplace_back(col.data(), col.size());
    }

    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    accumulate_column_benches(exe, tmp_root, buffers, kTypesizeBytes, false, orig, comp, ct, dt, loss);
    fs::remove_all(tmp_root, ec);
    result.assign({orig, comp, ct, dt});
}

void benchBufferOperatorAblation(const std::vector<char> &data, int typesize,
                                 std::string_view omitted_operator,
                                 std::vector<long long> &result, LossAccum *loss) {
    if (data.empty() || data.size() > static_cast<std::size_t>(INT32_MAX)) {
        throw std::runtime_error("invalid Blosc2 ablation buffer size");
    }
    if (typesize <= 0 || typesize > 255) throw std::invalid_argument("invalid Blosc2 typesize");
    const bool raw_blosclz = omitted_operator == "block/state";
    if (!omitted_operator.empty() && omitted_operator != "block transform" &&
        omitted_operator != "lz_match" && omitted_operator != "match_token_format" &&
        !raw_blosclz) {
        throw std::invalid_argument("unsupported Blosc2 operator ablation");
    }

    blosc2_context *compression_context = nullptr;
    if (!raw_blosclz) {
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
        cparams.compcode = BLOSC_BLOSCLZ;
        cparams.clevel = omitted_operator == "lz_match" || omitted_operator == "match_token_format"
                            ? 0
                            : 5;
        cparams.typesize = typesize;
        cparams.nthreads = 1;
        cparams.filters[BLOSC2_MAX_FILTERS - 1] =
            omitted_operator == "block transform" ? BLOSC_NOFILTER : BLOSC_SHUFFLE;
        compression_context = blosc2_create_cctx(cparams);
        if (compression_context == nullptr) {
            throw std::runtime_error("Blosc2 cctx creation failed");
        }
    }
    std::vector<std::uint8_t> compressed(data.size() * 2 + BLOSC2_MAX_OVERHEAD + 66);
    const auto t0 = std::chrono::high_resolution_clock::now();
    int compressed_size = raw_blosclz
        ? blosclz_compress(5, data.data(), static_cast<int>(data.size()),
                           compressed.data(), static_cast<int>(compressed.size()), nullptr)
        : blosc2_compress_ctx(
              compression_context, data.data(), static_cast<int32_t>(data.size()),
              compressed.data(), static_cast<int32_t>(compressed.size()));
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (compression_context != nullptr) blosc2_free_ctx(compression_context);
    bool raw_fallback = false;
    if (compressed_size <= 0) {
        raw_fallback = true;
        compressed.assign(reinterpret_cast<const std::uint8_t*>(data.data()),
                          reinterpret_cast<const std::uint8_t*>(data.data()) + data.size());
        compressed_size = static_cast<int>(data.size());
    }
    compressed.resize(static_cast<std::size_t>(compressed_size));

    blosc2_context *decompression_context = nullptr;
    if (!raw_blosclz && !raw_fallback) {
        blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
        dparams.nthreads = 1;
        dparams.typesize = typesize;
        decompression_context = blosc2_create_dctx(dparams);
        if (decompression_context == nullptr) {
            throw std::runtime_error("Blosc2 dctx creation failed");
        }
    }
    std::vector<std::uint8_t> decoded(data.size());
    const auto t2 = std::chrono::high_resolution_clock::now();
    int decoded_size = 0;
    if (raw_fallback) {
        std::memcpy(decoded.data(), compressed.data(), decoded.size());
        decoded_size = static_cast<int>(decoded.size());
    } else if (raw_blosclz) {
        decoded_size = blosclz_decompress(compressed.data(), compressed_size, decoded.data(),
                                          static_cast<int>(decoded.size()));
    } else {
        decoded_size = blosc2_decompress_ctx(
            decompression_context, compressed.data(), compressed_size, decoded.data(),
            static_cast<int32_t>(decoded.size()));
    }
    const auto t3 = std::chrono::high_resolution_clock::now();
    if (decompression_context != nullptr) blosc2_free_ctx(decompression_context);
    if (decoded_size != static_cast<int>(data.size()) ||
        std::memcmp(decoded.data(), data.data(), data.size()) != 0) {
        throw std::runtime_error("Blosc2 ablation roundtrip failed");
    }
    if (loss != nullptr) {
        const std::vector<char> decoded_chars(decoded.begin(), decoded.end());
        benchmark_loss::accumulate_bytes(data, decoded_chars, *loss);
    }
    result.assign({
        static_cast<long long>(data.size()), static_cast<long long>(compressed.size()),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count(),
    });
}

} // namespace wc_blosc2_bench
