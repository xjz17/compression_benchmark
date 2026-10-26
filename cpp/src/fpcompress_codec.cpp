#include "fpcompress_codec.hpp"

#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace fpcompress_codec {

namespace {

enum class Variant { Dpratio, Dpspeed };

static std::string unique_tag() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream o;
    o << std::hex << rng();
    return o.str();
}

static fs::path repo_root() {
    if (const char *r = std::getenv("COMPRESSION_BENCHMARK_REPO_ROOT"); r != nullptr && r[0] != '\0') {
        return fs::path(r);
    }
    return {};
}

static fs::path bin_root() {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_FPCOMPRESS_BIN_ROOT");
        e != nullptr && e[0] != '\0') {
        return fs::path(e);
    }
    const fs::path root = repo_root();
    if (root.empty()) {
        return {};
    }
    return root / "external/runtime/fpcompress";
}

static fs::path upstream_root() {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_FPCOMPRESS_UPSTREAM_ROOT");
        e != nullptr && e[0] != '\0') {
        return fs::path(e);
    }
    const fs::path root = repo_root();
    if (root.empty()) {
        return {};
    }
    return root / "external/compression/FPCompress";
}

static const char *exe_suffix() {
#ifdef _WIN32
    return ".exe";
#else
    return "";
#endif
}

static const char *variant_compress_stem(Variant v) {
    switch (v) {
    case Variant::Dpratio:
        return "ratio-cpu-compress";
    case Variant::Dpspeed:
        return "speed-cpu-compress";
    }
    return "ratio-cpu-compress";
}

static const char *variant_decompress_stem(Variant v) {
    switch (v) {
    case Variant::Dpratio:
        return "ratio-cpu-decompress";
    case Variant::Dpspeed:
        return "speed-cpu-decompress";
    }
    return "ratio-cpu-decompress";
}

static fs::path resolve_exe_override(Variant v, bool compress) {
    const char *env_key = nullptr;
    switch (v) {
    case Variant::Dpratio:
        env_key = compress ? "COMPRESSION_BENCHMARK_FPCOMPRESS_DPRATIO_COMPRESS"
                           : "COMPRESSION_BENCHMARK_FPCOMPRESS_DPRATIO_DECOMPRESS";
        break;
    case Variant::Dpspeed:
        env_key = compress ? "COMPRESSION_BENCHMARK_FPCOMPRESS_DPSPEED_COMPRESS"
                           : "COMPRESSION_BENCHMARK_FPCOMPRESS_DPSPEED_DECOMPRESS";
        break;
    }
    if (env_key == nullptr) {
        return {};
    }
    if (const char *e = std::getenv(env_key); e != nullptr && e[0] != '\0') {
        const fs::path p(e);
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    return {};
}

static fs::path find_exe(Variant v, bool compress) {
    if (fs::path override_path = resolve_exe_override(v, compress); !override_path.empty()) {
        return override_path;
    }
    const std::string name =
        std::string(compress ? variant_compress_stem(v) : variant_decompress_stem(v)) + exe_suffix();

    const auto try_tree = [&](const fs::path &root) -> fs::path {
        if (root.empty()) {
            return {};
        }
        const fs::path in_bin = root / "double_src" / "bin" / name;
        if (fs::is_regular_file(in_bin)) {
            return in_bin;
        }
        const fs::path in_src = root / "double_src" / name;
        if (fs::is_regular_file(in_src)) {
            return in_src;
        }
        return {};
    };

    if (fs::path hit = try_tree(bin_root()); !hit.empty()) {
        return hit;
    }
    return try_tree(upstream_root());
}

static bool fpcompress_env_on() {
    // Legacy external FPCompress binaries are disabled for benchmark runs.
    // fpcompress_dpratio / fpcompress_dpspeed are emitted by native_lz_family.
    return false;
}

static int run_cli(const fs::path &exe, const fs::path &input, const fs::path &output) {
    std::ostringstream cmd;
#ifdef _WIN32
    cmd << "cmd /c \"\"" << exe.string() << "\" \"" << input.string() << "\" \"" << output.string()
        << "\"\"";
#else
    cmd << "\"" << exe.string() << "\" \"" << input.string() << "\" \"" << output.string() << "\"";
#endif
    const std::string cmd_str = cmd.str();
    return std::system(cmd_str.c_str());
}

static bool read_file_bytes(const fs::path &path, std::vector<std::uint8_t> &out) {
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
    if (!in.read(reinterpret_cast<char *>(out.data()), sz)) {
        return false;
    }
    return true;
}

static std::uintmax_t file_size_or_zero(const fs::path &path) {
    std::error_code ec;
    const auto sz = fs::file_size(path, ec);
    return ec ? 0 : sz;
}

void bench_double_columns_impl(const std::vector<std::vector<double>> &columns, Variant variant,
                               std::vector<long long> &result, LossAccum *loss) {
    result.assign(4, 0);
    if (!fpcompress_env_on()) {
        return;
    }
    const fs::path comp_exe = find_exe(variant, true);
    const fs::path decomp_exe = find_exe(variant, false);
    if (comp_exe.empty() || decomp_exe.empty()) {
        return;
    }

    long long orig = 0;
    long long comp = 0;
    long long ct = 0;
    long long dt = 0;
    const fs::path tmp_root = fs::temp_directory_path() / ("wc_fpcompress_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmp_root, ec);
    if (ec) {
        return;
    }

    for (std::size_t ci = 0; ci < columns.size(); ++ci) {
        const auto &col = columns[ci];
        const std::size_t n = col.size();
        if (n == 0) {
            continue;
        }
        const std::size_t nbytes = n * sizeof(double);
        std::vector<std::uint8_t> raw(nbytes);
        std::memcpy(raw.data(), col.data(), nbytes);

        const fs::path in_path = tmp_root / ("in_" + std::to_string(ci) + ".bin");
        const fs::path comp_path = tmp_root / ("comp_" + std::to_string(ci) + ".bin");
        const fs::path out_path = tmp_root / ("out_" + std::to_string(ci) + ".bin");
        {
            std::ofstream wf(in_path, std::ios::binary);
            if (!wf.write(reinterpret_cast<const char *>(raw.data()),
                          static_cast<std::streamsize>(nbytes))) {
                continue;
            }
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        if (run_cli(comp_exe, in_path, comp_path) != 0) {
            fs::remove(comp_path, ec);
            continue;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto comp_bytes = file_size_or_zero(comp_path);
        if (comp_bytes == 0) {
            fs::remove(comp_path, ec);
            continue;
        }

        const auto t2 = std::chrono::high_resolution_clock::now();
        if (run_cli(decomp_exe, comp_path, out_path) != 0) {
            fs::remove(comp_path, ec);
            fs::remove(out_path, ec);
            continue;
        }
        const auto t3 = std::chrono::high_resolution_clock::now();

        std::vector<std::uint8_t> decoded;
        if (!read_file_bytes(out_path, decoded) || decoded.size() != nbytes ||
            std::memcmp(decoded.data(), raw.data(), nbytes) != 0) {
            fs::remove(comp_path, ec);
            fs::remove(out_path, ec);
            continue;
        }

        if (loss != nullptr) {
            std::vector<double> dec_vals(n);
            std::memcpy(dec_vals.data(), decoded.data(), nbytes);
            benchmark_loss::accumulate_double_column(col, dec_vals, *loss);
        }

        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        comp += static_cast<long long>(comp_bytes);
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

        fs::remove(in_path, ec);
        fs::remove(comp_path, ec);
        fs::remove(out_path, ec);
    }

    fs::remove_all(tmp_root, ec);
    result = {orig, comp, ct, dt};
}

} // namespace

bool bench_enabled() { return fpcompress_env_on(); }

void benchDoubleColumnsDpratio(const std::vector<std::vector<double>> &columns,
                               std::vector<long long> &result, LossAccum *loss) {
    bench_double_columns_impl(columns, Variant::Dpratio, result, loss);
}

void benchDoubleColumnsDpspeed(const std::vector<std::vector<double>> &columns,
                               std::vector<long long> &result, LossAccum *loss) {
    bench_double_columns_impl(columns, Variant::Dpspeed, result, loss);
}

} // namespace fpcompress_codec
