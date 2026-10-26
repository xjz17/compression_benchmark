#include "research_cli.hpp"
#include "algorithm_allow_list.hpp"
#include "benchmark_loss.hpp"
#include "benchmark_wire.hpp"
#include "research_cli_serf_embed.hpp"
#include "research_cli_sz3_embed.hpp"

#ifdef COMPRESSION_BENCHMARK_EMBED_SZ
extern "C" {
#include "sz_api.h"
}
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "buff_kernel/buff_roaring_lite.hpp"
#include "buff_kernel/buff_simd_ops.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static std::string unique_tag() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream o;
    o << std::hex << rng();
    return o.str();
}

static int run_shell(const std::string &cmd) { return std::system(cmd.c_str()); }

/** Run native CLI with repo-relative DLL search paths (Windows sz3 needs libzstd.dll). */
static int run_shell_for_native_cli(const fs::path &exe, const fs::path &repo_root,
                                    const std::string &args_tail) {
    std::ostringstream cmd;
#ifdef _WIN32
    std::string path_extra;
    auto add_path = [&](const fs::path &p) {
        if (p.empty()) {
            return;
        }
        if (!path_extra.empty()) {
            path_extra += ";";
        }
        path_extra += p.string();
    };
    add_path(exe.parent_path());
    add_path(repo_root / "external/compression/SZ3/build/tools/zstd");
    if (const char *mb = std::getenv("COMPRESSION_BENCHMARK_MINGW_BIN"); mb != nullptr && *mb != '\0') {
        add_path(fs::path(mb));
    }
    if (const char *extra = std::getenv("COMPRESSION_BENCHMARK_NATIVE_CLI_PATH");
        extra != nullptr && *extra != '\0') {
        if (!path_extra.empty()) {
            path_extra += ";";
        }
        path_extra += extra;
    }
    cmd << "cmd /c \"set \"PATH=" << path_extra << ";%PATH%\"&& \"" << exe.string() << "\" "
        << args_tail << "\"";
#else
    cmd << "\"" << exe.string() << "\" " << args_tail;
#endif
    return run_shell(cmd.str());
}

#if !defined(_WIN32)
/** Run {@code sh -c cmd}. On timeout, kills the child process group (Serf CLI can hang after printing). */
static int run_shell_timeout(const std::string &cmd, int timeout_sec) {
    if (timeout_sec <= 0) {
        return run_shell(cmd);
    }
    pid_t child = fork();
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        setpgid(0, 0);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    (void)setpgid(child, child);
    const int step_ms = 100;
    int elapsed_ms = 0;
    const int max_ms = timeout_sec * 1000;
    while (elapsed_ms < max_ms) {
        int status = 0;
        const pid_t w = waitpid(child, &status, WNOHANG);
        if (w == child) {
            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            }
            return -1;
        }
        if (w < 0) {
            return -1;
        }
        usleep(static_cast<useconds_t>(step_ms * 1000));
        elapsed_ms += step_ms;
    }
    kill(-child, SIGKILL);
    (void)waitpid(child, nullptr, 0);
    std::cerr << "error: Serf shell command timed out after " << timeout_sec
              << "s (set COMPRESSION_BENCHMARK_SERF_SHELL_TIMEOUT_SEC or COMPRESSION_BENCHMARK_SERF_BENCH=0)\n";
    return -1;
}
#else
static int run_shell_timeout(const std::string &cmd, int timeout_sec) {
    (void)timeout_sec;
    return run_shell(cmd);
}
#endif

/** Per-invocation timeout for sfz compress/decompress; 0 = unlimited (default 300s). */
static int serf_shell_timeout_sec() {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_SERF_SHELL_TIMEOUT_SEC");
    if (!e || !*e) {
        return 300;
    }
    char *end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e) {
        return 300;
    }
    if (v <= 0) {
        return 0;
    }
    if (v > 86400) {
        return 86400;
    }
    return static_cast<int>(v);
}

static fs::path resolve_repo(const std::string &repoRoot) { return fs::path(repoRoot); }

static fs::path find_sz_cfg(const fs::path &root) {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_SZ_CONFIG")) {
        fs::path p(e);
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_SZ3_CONFIG")) {
        fs::path p(e);
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    fs::path a = root / "external/compression/SZ/example/sz.config";
    if (fs::is_regular_file(a)) {
        return a;
    }
    fs::path b = root / "external/compression/SZ3/sz.config";
    if (fs::is_regular_file(b)) {
        return b;
    }
    return {};
}

static fs::path resolve_native_cli(const fs::path &p) {
    if (fs::exists(p)) {
        return p;
    }
#ifdef _WIN32
    fs::path with_exe = p;
    with_exe += ".exe";
    if (fs::exists(with_exe)) {
        return with_exe;
    }
#endif
    return {};
}

static fs::path find_sz_exe(const fs::path &root) {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_SZ")) {
        fs::path hit = resolve_native_cli(fs::path(e));
        if (!hit.empty()) {
            return hit;
        }
    }
    fs::path hit = resolve_native_cli(root / "external/compression/SZ/example/sz");
    if (!hit.empty()) {
        return hit;
    }
    hit = resolve_native_cli(root / "external/compression/SZ/build/bin/sz");
    return hit;
}

static fs::path find_sz3_exe(const fs::path &root) {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_SZ3")) {
        fs::path hit = resolve_native_cli(fs::path(e));
        if (!hit.empty()) {
            return hit;
        }
    }
    fs::path hit = resolve_native_cli(root / "external/compression/SZ3/build/tools/sz3/sz3");
    if (!hit.empty()) {
        return hit;
    }
    return resolve_native_cli(root / "external/compression/SZ3/build/tools/sz3/Release/sz3");
}

static fs::path find_sfz_exe(const fs::path &root) {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_SFZ")) {
        fs::path p(e);
        if (fs::exists(p)) {
            return p;
        }
    }
    fs::path c = root / "external/compression/Serf/build/serf_cmd/sfz";
    if (fs::exists(c)) {
        return c;
    }
    return {};
}

static fs::path find_comp_profiler_exe(const fs::path &root) {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_COMP_PROFILER")) {
        fs::path p(e);
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    for (const char *rel : {"external/compression/buff/database/target/release/comp_profiler",
                           "external/compression/buff/database/target/debug/comp_profiler"}) {
        fs::path c = root / rel;
        if (fs::is_regular_file(c)) {
            return c;
        }
    }
    return {};
}

static fs::path find_cmix_exe(const fs::path &root) {
    if (const char *e = std::getenv("COMPRESSION_BENCHMARK_CMIX")) {
        fs::path p(e);
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    fs::path c = root / "external/compression/cmix/cmix";
    if (fs::is_regular_file(c)) {
        return c;
    }
    return {};
}

static int cmix_timeout_sec() {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_CMIX_TIMEOUT_SEC");
    if (!e || !*e) {
        return 600;
    }
    char *end = nullptr;
    long v = std::strtol(e, &end, 10);
    if (end == e) {
        return 600;
    }
    if (v <= 0) {
        return 0;
    }
    if (v > 86400) {
        return 86400;
    }
    return static_cast<int>(v);
}

static bool cmix_no_preprocess() {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS");
    return e && std::strcmp(e, "1") == 0;
}

/** 0 = unlimited; else compress only first this many bytes of each float column (multiple of 4). */
static std::size_t cmix_max_column_bytes() {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_CMIX_MAX_COLUMN_BYTES");
    if (!e || !*e) {
        return 0;
    }
    char *end = nullptr;
    unsigned long v = std::strtoul(e, &end, 10);
    if (end == e || v == 0) {
        return 0;
    }
    if (v > 256UL * 1024UL * 1024UL) {
        return 256UL * 1024UL * 1024UL;
    }
    return static_cast<std::size_t>(v);
}

/** Lossless cmix CLI on float32 column bytes; gated by {@code COMPRESSION_BENCHMARK_CMIX_BENCH=1}. */
static void bench_cmix_float(const std::vector<std::vector<float>> &cols, const fs::path &repoRoot,
                             std::ofstream &out) {
    const char *gate = std::getenv("COMPRESSION_BENCHMARK_CMIX_BENCH");
    if (!gate || std::strcmp(gate, "1") != 0) {
        return;
    }
    if (cols.empty()) {
        return;
    }
    fs::path cmix = find_cmix_exe(resolve_repo(repoRoot.string()));
    if (cmix.empty()) {
        return;
    }
    int timeoutSec = cmix_timeout_sec();
    std::size_t maxColBytes = cmix_max_column_bytes();
    fs::path tmpRoot = fs::temp_directory_path() / ("wc_cmix_cpp_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmpRoot, ec);
    if (ec) {
        return;
    }
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : cols) {
        int n = static_cast<int>(col.size());
        if (n < 1) {
            continue;
        }
        if (maxColBytes > 0) {
            int maxFloats = static_cast<int>(maxColBytes / 4U);
            if (n > maxFloats) {
                n = maxFloats;
            }
        }
        if (n < 1) {
            continue;
        }
        std::vector<char> raw(static_cast<std::size_t>(n) * 4);
        for (int i = 0; i < n; ++i) {
            float f = col[static_cast<std::size_t>(i)];
            std::memcpy(raw.data() + static_cast<std::size_t>(i) * 4, &f, 4);
        }
        fs::path in = tmpRoot / ("in_" + unique_tag() + ".dat");
        fs::path fcomp = tmpRoot / ("cmp_" + unique_tag() + ".cmix");
        fs::path fdec = tmpRoot / ("dec_" + unique_tag() + ".dat");
        {
            std::ofstream wf(in, std::ios::binary);
            wf.write(raw.data(), static_cast<std::streamsize>(raw.size()));
        }
        if (fs::exists(fcomp)) {
            fs::remove(fcomp, ec);
        }
        if (fs::exists(fdec)) {
            fs::remove(fdec, ec);
        }
        std::ostringstream enc;
        enc << "\"" << cmix.string() << "\" " << (cmix_no_preprocess() ? "-n" : "-c") << " \"" << in.string()
            << "\" \"" << fcomp.string() << "\"";
        auto t0 = std::chrono::high_resolution_clock::now();
        if (run_shell_timeout(enc.str(), timeoutSec) != 0) {
            fs::remove(in, ec);
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!fs::is_regular_file(fcomp)) {
            fs::remove(in, ec);
            continue;
        }
        long long csize = static_cast<long long>(fs::file_size(fcomp));

        std::ostringstream dec;
        dec << "\"" << cmix.string() << "\" -d \"" << fcomp.string() << "\" \"" << fdec.string() << "\"";
        auto t2 = std::chrono::high_resolution_clock::now();
        if (run_shell_timeout(dec.str(), timeoutSec) != 0) {
            fs::remove(in, ec);
            fs::remove(fcomp, ec);
            continue;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (!fs::is_regular_file(fdec) || static_cast<std::size_t>(fs::file_size(fdec)) != raw.size()) {
            fs::remove(in, ec);
            fs::remove(fcomp, ec);
            fs::remove(fdec, ec);
            continue;
        }
        std::vector<char> back(static_cast<std::size_t>(fs::file_size(fdec)));
        {
            std::ifstream rf(fdec, std::ios::binary);
            rf.read(back.data(), static_cast<std::streamsize>(back.size()));
        }
        if (back.size() != raw.size() || std::memcmp(back.data(), raw.data(), raw.size()) != 0) {
            fs::remove(in, ec);
            fs::remove(fcomp, ec);
            fs::remove(fdec, ec);
            continue;
        }
        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        comp += csize;
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        fs::remove(in, ec);
        fs::remove(fcomp, ec);
        fs::remove(fdec, ec);
    }
    fs::remove_all(tmpRoot, ec);
    if (orig > 0) {
        out << "ext_cmix," << orig << ',' << comp << ',' << ct << ',' << dt << '\n';
    }
}

static bool buff_float_bench_env_on() {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_BUFF_BENCH");
    return e && std::strcmp(e, "1") == 0;
}

/** x86_64: {@code buff-simd}; AArch64: {@code buff} (upstream SIMD is AVX2). Override: {@code COMPRESSION_BENCHMARK_BUFF_SIMD_MODE}. */
static const char *resolve_buff_simd_profiler_mode() {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_BUFF_SIMD_MODE");
    if (e) {
        if (std::strcmp(e, "buff-simd") == 0) {
            return "buff-simd";
        }
        if (std::strcmp(e, "buff") == 0) {
            return "buff";
        }
    }
#if defined(__x86_64__) || defined(_M_AMD64)
    return "buff-simd";
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    return "buff";
#else
    return "buff";
#endif
}

static void split_csv_commas(const std::string &s, std::vector<std::string> &out) {
    out.clear();
    std::size_t start = 0;
    while (start <= s.size()) {
        std::size_t p = s.find(',', start);
        if (p == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

/** Parses {@code Performance:path,...,ratio,enc_mbps,dec_mbps,...} allowing commas in {@code path}. */
static bool parse_buff_performance_stdout(const std::string &text, double &ratio, double &encMbps,
                                          double &decMbps) {
    const std::string needle = "Performance:";
    std::size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    std::size_t end = text.find('\n', pos);
    std::string line = end == std::string::npos ? text.substr(pos) : text.substr(pos, end - pos);
    std::vector<std::string> parts;
    split_csv_commas(line, parts);
    if (parts.size() < 10) {
        return false;
    }
    std::size_t n = parts.size();
    try {
        ratio = std::stod(parts[n - 7]);
        encMbps = std::stod(parts[n - 6]);
        decMbps = std::stod(parts[n - 5]);
    } catch (...) {
        return false;
    }
    return ratio > 0 && encMbps > 0 && decMbps > 0;
}

#if !defined(_WIN32)
static bool run_comp_profiler_capture(const fs::path &exe, const fs::path &databaseCwd,
                                      const fs::path &csvPath, const char *profilerMode, int intScale,
                                      double pred, std::string &stdoutOut) {
    std::ostringstream cmd;
    cmd << "cd \"" << databaseCwd.string() << "\" && \"" << exe.string() << "\" \""
        << csvPath.string() << "\" " << (profilerMode ? profilerMode : "buff") << " " << intScale << " "
        << std::fixed << std::setprecision(16) << pred << " 2>/dev/null";
    FILE *pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        return false;
    }
    char buf[4096];
    stdoutOut.clear();
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        stdoutOut += buf;
    }
    int st = pclose(pipe);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
#else
static bool run_comp_profiler_capture(const fs::path &, const fs::path &, const fs::path &, const char *, int,
                                      double, std::string &) {
    return false;
}
#endif

static void bench_buff_accumulate(const std::vector<std::vector<float>> &cols, const fs::path &exe,
                                  const fs::path &databaseCwd, const fs::path &tmpRoot, int scale,
                                  double pred, const char *profilerMode, long long &orig, long long &comp,
                                  long long &ct, long long &dt) {
    orig = comp = ct = dt = 0;
    std::error_code ec;
    for (const auto &col : cols) {
        int n = static_cast<int>(col.size());
        if (n < 1) {
            continue;
        }
        long long orgCol = static_cast<long long>(n) * 8LL;
        fs::path csv = tmpRoot / ("in_" + unique_tag() + ".csv");
        {
            std::ofstream wf(csv);
            if (!wf) {
                continue;
            }
            for (int i = 0; i < n; ++i) {
                double d = static_cast<double>(col[static_cast<std::size_t>(i)]);
                wf << std::setprecision(17) << d << '\n';
            }
        }
        std::string cap;
        if (!run_comp_profiler_capture(exe, databaseCwd, fs::absolute(csv), profilerMode, scale, pred, cap)) {
            fs::remove(csv, ec);
            continue;
        }
        double ratio = 0, encM = 0, decM = 0;
        if (!parse_buff_performance_stdout(cap, ratio, encM, decM)) {
            fs::remove(csv, ec);
            continue;
        }
        long long compCol = static_cast<long long>(ratio * static_cast<double>(orgCol));
        long long encNs = static_cast<long long>(static_cast<double>(orgCol) * 1e9 / (encM * 1048576.0));
        long long decNs = static_cast<long long>(static_cast<double>(orgCol) * 1e9 / (decM * 1048576.0));
        if (compCol < 1 || encNs < 1 || decNs < 1) {
            fs::remove(csv, ec);
            continue;
        }
        orig += orgCol;
        comp += compCol;
        ct += encNs;
        dt += decNs;
        fs::remove(csv, ec);
    }
}

static void bench_buff_rust(const std::vector<std::vector<float>> &cols, const fs::path &root,
                            std::ofstream &out) {
    if (!buff_float_bench_env_on()) {
        return;
    }
    fs::path exe = find_comp_profiler_exe(root);
    fs::path databaseCwd = root / "external/compression/buff/database";
    if (exe.empty() || !fs::is_directory(databaseCwd)) {
        return;
    }
    int scale = 10000;
    if (const char *es = std::getenv("COMPRESSION_BENCHMARK_BUFF_INT_SCALE")) {
        char *e2 = nullptr;
        long v = std::strtol(es, &e2, 10);
        if (e2 != es && v > 0) {
            scale = static_cast<int>(v);
        }
    }
    double pred = 1.1509;
    if (const char *ep = std::getenv("COMPRESSION_BENCHMARK_BUFF_PRED")) {
        char *e2 = nullptr;
        double v = std::strtod(ep, &e2);
        if (e2 != ep) {
            pred = v;
        }
    }

    fs::path tmpRoot = fs::temp_directory_path() / ("wc_buff_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmpRoot, ec);
    if (ec) {
        return;
    }
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    bench_buff_accumulate(cols, exe, databaseCwd, tmpRoot, scale, pred, "buff", orig, comp, ct, dt);
    fs::remove_all(tmpRoot, ec);
    if (orig > 0) {
        out << "buff_scalar," << orig << ',' << comp << ',' << ct << ',' << dt << '\n';
    }
    const char *simdMode = resolve_buff_simd_profiler_mode();
    long long o2 = 0, c2 = 0, t2 = 0, d2 = 0;
    if (std::strcmp(simdMode, "buff") == 0) {
        o2 = orig;
        c2 = comp;
        t2 = ct;
        d2 = dt;
    } else {
        fs::create_directories(tmpRoot, ec);
        if (!ec) {
            bench_buff_accumulate(cols, exe, databaseCwd, tmpRoot, scale, pred, simdMode, o2, c2, t2, d2);
            fs::remove_all(tmpRoot, ec);
        }
    }
    if (o2 > 0) {
        out << "buff_simd," << o2 << ',' << c2 << ',' << t2 << ',' << d2 << '\n';
    }
}

/** In-process BUFF kernel: bit-exact encode and real decode.
 * Opt in with {@code COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE=1}. Row {@code buff_kernel_native}. */
static void bench_buff_native_kernel(const std::vector<std::vector<double>> &cols, int scale, double pred,
                                     std::ofstream &out) {
    const char *e = std::getenv("COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE");
    if (!e || std::strcmp(e, "1") != 0) {
        return;
    }
    long long orig = 0, comp = 0, ct = 0, dt = 0;
    for (const auto &col : cols) {
        int n = static_cast<int>(col.size());
        if (n < 1) {
            continue;
        }
        const std::vector<double> &d = col;
        orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint8_t> packed = buff_kernel::encode_doubles(d, scale);
        auto t1 = std::chrono::high_resolution_clock::now();
        const auto decoded = buff_kernel::decode_doubles(packed);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (decoded.values.size() != d.size() ||
            std::memcmp(decoded.values.data(), d.data(), d.size() * sizeof(double)) != 0) {
            continue;
        }
        comp += static_cast<long long>(packed.size());
        ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    if (orig > 0) {
        out << "buff_kernel_native," << orig << ',' << comp << ',' << ct << ',' << dt << '\n';
    }
}

struct SzPreset {
    const char *rowName;
    const char *errMode;
    const char *boundFlag;
    const char *boundValue;
};

static const SzPreset kSzPresets[] = {
    {"sz_ABS_1e-1", "ABS", "-A", "1E-1"},
    {"sz_ABS_1e-2", "ABS", "-A", "1E-2"},
    {"sz_ABS_1e-3", "ABS", "-A", "1E-3"},
    {"sz_ABS_1e-4", "ABS", "-A", "1E-4"},
    {"sz_REL_1e-2", "REL", "-R", "1E-2"},
    {"sz_REL_1e-3", "REL", "-R", "1E-3"},
};

static const char *kSerfBounds[] = {"1e-1", "1e-2", "1e-3", "1e-4"};
static const char *kSerfRows[] = {"serf_ABS_1e-1", "serf_ABS_1e-2", "serf_ABS_1e-3", "serf_ABS_1e-4"};

static bool skip_algo(const std::unordered_set<std::string> *skip_algorithms,
                      const std::string &name) {
    return algorithm_allow_list::should_skip(skip_algorithms, name);
}

static void write_bench_row(std::ofstream &out, const std::string &row_name, long long orig,
                            long long comp, long long ct, long long dt, const LossAccum &loss = {}) {
    out << row_name << ',' << orig << ',' << comp << ',' << ct << ',' << dt;
    if (loss.err != 0.0 || loss.ref != 0.0) {
        out << ',' << loss.err << ',' << loss.ref;
    }
    out << '\n';
}

static fs::path resolve_sz_decompressed_path(const fs::path &input_path,
                                             const fs::path &compressed_path) {
    const fs::path candidates[] = {
        fs::path(compressed_path.string() + ".out"),
        fs::path(compressed_path.string() + "."),
        fs::path(input_path.string() + ".out"),
    };
    for (const fs::path &p : candidates) {
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    return {};
}

template <typename T>
static bool bench_sz_family(const std::vector<std::vector<T>> &cols, const fs::path &szExe,
                            const fs::path &cfg, const fs::path &repo_root,
                            const SzPreset *presets, std::size_t nPresets,
                            bool asSz3, std::ofstream &out,
                            const std::unordered_set<std::string> *skip_algorithms) {
    fs::path tmpRoot = fs::temp_directory_path() / ("wc_sz_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmpRoot, ec);
    if (ec) {
        return false;
    }

    for (std::size_t pi = 0; pi < nPresets; ++pi) {
        const SzPreset &pr = presets[pi];
        std::string row_name = pr.rowName;
        if (asSz3 && row_name.rfind("sz_", 0) == 0) {
            row_name.replace(0, 3, "sz3_");
        }
        if (skip_algo(skip_algorithms, row_name)) {
            continue;
        }
        long long orig = 0, comp = 0, ct = 0, dt = 0;
        LossAccum loss;
        for (const auto &col : cols) {
            int n = static_cast<int>(col.size());
            if (n < 1) {
                continue;
            }
            constexpr std::size_t elem_bytes = std::is_same_v<T, double> ? 8u : 4u;
            std::vector<char> raw(static_cast<std::size_t>(n) * elem_bytes);
            for (int i = 0; i < n; ++i) {
                T v = col[static_cast<std::size_t>(i)];
                std::memcpy(raw.data() + static_cast<std::size_t>(i) * elem_bytes, &v, elem_bytes);
            }
            fs::path in = tmpRoot / ("in_" + unique_tag() + ".dat");
            fs::path outs = fs::path(std::string(in.string()) + ".sz");
            {
                std::ofstream wf(in, std::ios::binary);
                wf.write(raw.data(), static_cast<std::streamsize>(raw.size()));
            }
            if (fs::exists(outs)) {
                fs::remove(outs);
            }
            std::ostringstream enc_args;
            const char *dtype_flag = std::is_same_v<T, double> ? "-d" : "-f";
            enc_args << "-z " << dtype_flag << " -c \"" << cfg.string() << "\" -M "
                     << pr.errMode << " " << pr.boundFlag << " " << pr.boundValue << " -i \""
                     << in.string() << "\" -1 " << n;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (run_shell_for_native_cli(szExe, repo_root, enc_args.str()) != 0) {
                fs::remove(in, ec);
                continue;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!fs::is_regular_file(outs)) {
                fs::remove(in, ec);
                continue;
            }
            long long csize = static_cast<long long>(fs::file_size(outs));

            std::ostringstream dec_args;
            dec_args << "-x " << dtype_flag << " -s \"" << outs.string() << "\" -1 " << n;
            auto t2 = std::chrono::high_resolution_clock::now();
            if (run_shell_for_native_cli(szExe, repo_root, dec_args.str()) != 0) {
                fs::remove(in, ec);
                fs::remove(outs, ec);
                continue;
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            const fs::path dec_path = resolve_sz_decompressed_path(in, outs);
            if (dec_path.empty()) {
                fs::remove(in, ec);
                fs::remove(outs, ec);
                continue;
            }
            std::vector<T> decoded(static_cast<std::size_t>(n));
            {
                std::ifstream rf(dec_path, std::ios::binary);
                rf.read(reinterpret_cast<char *>(decoded.data()),
                        static_cast<std::streamsize>(decoded.size() * elem_bytes));
                if (rf.gcount() != static_cast<std::streamsize>(n * elem_bytes)) {
                    fs::remove(in, ec);
                    fs::remove(outs, ec);
                    if (dec_path != in) {
                        fs::remove(dec_path, ec);
                    }
                    continue;
                }
            }
            std::vector<double> orig_d(static_cast<std::size_t>(n));
            std::vector<double> dec_d(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                orig_d[static_cast<std::size_t>(i)] =
                    static_cast<double>(col[static_cast<std::size_t>(i)]);
                dec_d[static_cast<std::size_t>(i)] =
                    static_cast<double>(decoded[static_cast<std::size_t>(i)]);
            }
            benchmark_loss::accumulate_double_column(orig_d, dec_d, loss);

            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += csize;
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

            fs::remove(in, ec);
            fs::remove(outs, ec);
            if (!dec_path.empty() && dec_path != in) {
                fs::remove(dec_path, ec);
            }
        }
        if (orig > 0) {
            write_bench_row(out, row_name, orig, comp, ct, dt, loss);
        }
    }

    fs::remove_all(tmpRoot, ec);
    return true;
}

#ifdef COMPRESSION_BENCHMARK_EMBED_SZ
template <typename T>
static bool bench_sz_family_inproc(const std::vector<std::vector<T>> &cols, const fs::path &cfg,
                                   const SzPreset *presets, std::size_t nPresets, bool asSz3,
                                   std::ofstream &out,
                                   const std::unordered_set<std::string> *skip_algorithms) {
    if (asSz3) {
        return false;
    }
    if (SZ_Init(cfg.string().c_str()) != SZ_SCES) {
        return false;
    }
    struct SzFin {
        ~SzFin() { SZ_Finalize(); }
    } sz_fin;

    for (std::size_t pi = 0; pi < nPresets; ++pi) {
        const SzPreset &pr = presets[pi];
        std::string row_name = pr.rowName;
        if (asSz3 && row_name.rfind("sz_", 0) == 0) {
            row_name.replace(0, 3, "sz3_");
        }
        if (skip_algo(skip_algorithms, row_name)) {
            continue;
        }
        int errMode = ABS;
        double absB = 0.0;
        double relB = 0.0;
        if (std::strcmp(pr.errMode, "ABS") == 0) {
            errMode = ABS;
            absB = std::stod(pr.boundValue);
        } else if (std::strcmp(pr.errMode, "REL") == 0) {
            errMode = REL;
            relB = std::stod(pr.boundValue);
        } else {
            continue;
        }

        long long orig = 0, comp = 0, ct = 0, dt = 0;
        LossAccum loss;
        for (const auto &col : cols) {
            int n = static_cast<int>(col.size());
            if (n < 1) {
                continue;
            }
            std::vector<T> mut = col;
            size_t compressedSize = 0;
            const int sz_type = std::is_same_v<T, double> ? SZ_DOUBLE : SZ_FLOAT;
            auto t0 = std::chrono::high_resolution_clock::now();
            unsigned char *bytes =
                SZ_compress_args(sz_type, mut.data(), &compressedSize, errMode, absB, relB, 0.0, 0, 0, 0, 0,
                                 static_cast<size_t>(n));
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!bytes || compressedSize == 0) {
                if (bytes) {
                    std::free(bytes);
                }
                continue;
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            void *dec =
                SZ_decompress(sz_type, bytes, compressedSize, 0, 0, 0, 0, static_cast<size_t>(n));
            auto t3 = std::chrono::high_resolution_clock::now();
            std::free(bytes);
            if (!dec) {
                continue;
            }
            const T *dec_ptr = static_cast<const T *>(dec);
            std::vector<double> orig_d(static_cast<std::size_t>(n));
            std::vector<double> dec_d(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                orig_d[static_cast<std::size_t>(i)] = static_cast<double>(col[static_cast<std::size_t>(i)]);
                dec_d[static_cast<std::size_t>(i)] = static_cast<double>(dec_ptr[static_cast<std::size_t>(i)]);
            }
            benchmark_loss::accumulate_double_column(orig_d, dec_d, loss);
            std::free(dec);

            orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
            comp += static_cast<long long>(compressedSize);
            ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
        if (orig > 0) {
            write_bench_row(out, row_name, orig, comp, ct, dt, loss);
        }
    }
    return true;
}
#endif

static bool bench_serf(const std::vector<std::vector<float>> &cols, const fs::path &sfz,
                       std::ofstream &out,
                       const std::unordered_set<std::string> *skip_algorithms) {
    const char *flag = std::getenv("COMPRESSION_BENCHMARK_SERF_BENCH");
    if (!flag || flag[0] != '1') {
        return false;
    }

#ifdef COMPRESSION_BENCHMARK_EMBED_SERF
    if (bench_serf_inproc(cols, out)) {
        return true;
    }
#endif

    fs::path tmpRoot = fs::temp_directory_path() / ("wc_serf_" + unique_tag());
    std::error_code ec;
    fs::create_directories(tmpRoot, ec);
    if (ec) {
        return false;
    }

    for (std::size_t pi = 0; pi < sizeof(kSerfBounds) / sizeof(kSerfBounds[0]); ++pi) {
        if (skip_algo(skip_algorithms, kSerfRows[pi])) {
            continue;
        }
        const char *boundStr = kSerfBounds[pi];
        double bound = std::stod(boundStr);
        long long orig = 0, comp = 0, ct = 0, dt = 0;
        LossAccum loss;

        const int serf_tmo = serf_shell_timeout_sec();
        for (std::size_t ci = 0; ci < cols.size(); ++ci) {
            const auto &col = cols[ci];
            int n = static_cast<int>(col.size());
            if (n < 1) {
                continue;
            }
            fs::path inTxt = tmpRoot / ("in_" + unique_tag() + ".txt");
            fs::path outBin = tmpRoot / ("cmp_" + unique_tag() + ".bin");
            fs::path outTxt = tmpRoot / ("dec_" + unique_tag() + ".txt");
            {
                std::ofstream w(inTxt);
                w << std::setprecision(9);
                for (float f : col) {
                    w << f << '\n';
                }
            }
            std::ostringstream enc;
            enc << "\"" << sfz.string() << "\" compress " << boundStr << " \"" << inTxt.string()
                << "\" \"" << outBin.string() << "\"";
            std::cerr << "Serf bound " << boundStr << " column " << (ci + 1) << "/" << cols.size()
                      << " (n=" << n << ") compress...\n"
                      << std::flush;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (run_shell_timeout(enc.str(), serf_tmo) != 0) {
                fs::remove(inTxt, ec);
                continue;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!fs::is_regular_file(outBin)) {
                fs::remove(inTxt, ec);
                continue;
            }
            long long csize = static_cast<long long>(fs::file_size(outBin));

            std::ostringstream dec;
            dec << "\"" << sfz.string() << "\" decompress " << boundStr << " \""
                << outBin.string() << "\" \"" << outTxt.string() << "\"";
            std::cerr << "Serf bound " << boundStr << " column " << (ci + 1) << "/" << cols.size()
                      << " decompress...\n"
                      << std::flush;
            auto t2 = std::chrono::high_resolution_clock::now();
            if (run_shell_timeout(dec.str(), serf_tmo) != 0) {
                fs::remove(inTxt, ec);
                fs::remove(outBin, ec);
                continue;
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            bool ok = false;
            {
                std::ifstream sc(outTxt);
                std::string line;
                int i = 0;
                ok = true;
                while (ok && i < n && std::getline(sc, line)) {
                    double dv = 0.0;
                    try {
                        dv = std::stod(line);
                    } catch (const std::exception &) {
                        ok = false;
                        break;
                    }
                    float got = static_cast<float>(dv);
                    float exp = col[static_cast<std::size_t>(i)];
                    if (std::fabs(static_cast<double>(got - exp)) > bound * 1.0001 + 1e-6) {
                        ok = false;
                    }
                    ++i;
                }
                if (i != n) {
                    ok = false;
                }
            }

            if (ok) {
                std::vector<double> orig_d(static_cast<std::size_t>(n));
                std::vector<double> dec_d(static_cast<std::size_t>(n));
                {
                    std::ifstream sc(outTxt);
                    std::string line;
                    int i = 0;
                    while (i < n && std::getline(sc, line)) {
                        orig_d[static_cast<std::size_t>(i)] =
                            static_cast<double>(col[static_cast<std::size_t>(i)]);
                        dec_d[static_cast<std::size_t>(i)] = std::stod(line);
                        ++i;
                    }
                }
                benchmark_loss::accumulate_double_column(orig_d, dec_d, loss);
                orig += static_cast<long long>(benchmark_wire::column_original_size(n, true));
                comp += csize;
                ct += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                dt += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }

            fs::remove(inTxt, ec);
            fs::remove(outBin, ec);
            fs::remove(outTxt, ec);
        }
        if (orig > 0) {
            write_bench_row(out, kSerfRows[pi], orig, comp, ct, dt, loss);
        }
    }

    fs::remove_all(tmpRoot, ec);
    return true;
}

void bench_sz_sz3_serf_double(const std::vector<std::vector<double>> &columnDoubleData,
                              const std::string &repoRoot, std::ofstream &out,
                              const std::unordered_set<std::string> *skip_algorithms) {
    if (columnDoubleData.empty()) {
        return;
    }
    fs::path root = resolve_repo(repoRoot);
    fs::path cfg = find_sz_cfg(root);
    if (fs::is_regular_file(cfg)) {
        fs::path sz = find_sz_exe(root);
#ifdef COMPRESSION_BENCHMARK_EMBED_SZ
        const bool did_inproc_sz =
            bench_sz_family_inproc(columnDoubleData, cfg, kSzPresets,
                                   sizeof(kSzPresets) / sizeof(kSzPresets[0]), false, out,
                                   skip_algorithms);
        if (!did_inproc_sz && !sz.empty()) {
            bench_sz_family(columnDoubleData, sz, cfg, root, kSzPresets,
                            sizeof(kSzPresets) / sizeof(kSzPresets[0]), false, out,
                            skip_algorithms);
        }
#else
        if (!sz.empty()) {
            bench_sz_family(columnDoubleData, sz, cfg, root, kSzPresets,
                            sizeof(kSzPresets) / sizeof(kSzPresets[0]), false, out,
                            skip_algorithms);
        }
#endif
        fs::path sz3 = find_sz3_exe(root);
#ifdef COMPRESSION_BENCHMARK_EMBED_SZ3
        const bool did_inproc_sz3 =
            bench_sz3_family_inproc_double(columnDoubleData, cfg.string(), out);
        if (!did_inproc_sz3 && !sz3.empty()) {
            bench_sz_family(columnDoubleData, sz3, cfg, root, kSzPresets,
                            sizeof(kSzPresets) / sizeof(kSzPresets[0]), true, out,
                            skip_algorithms);
        }
#else
        if (!sz3.empty()) {
            bench_sz_family(columnDoubleData, sz3, cfg, root, kSzPresets,
                            sizeof(kSzPresets) / sizeof(kSzPresets[0]), true, out,
                            skip_algorithms);
        }
#endif
    }
    std::vector<std::vector<float>> lossyFloatData;
    lossyFloatData.reserve(columnDoubleData.size());
    for (const auto &col : columnDoubleData) {
        std::vector<float> fcol;
        fcol.reserve(col.size());
        for (double v : col) {
            fcol.push_back(static_cast<float>(v));
        }
        lossyFloatData.push_back(std::move(fcol));
    }
    fs::path sfz = find_sfz_exe(root);
    if (!sfz.empty()) {
        bench_serf(lossyFloatData, sfz, out, skip_algorithms);
    }
    if (!skip_algo(skip_algorithms, "buff_scalar") || !skip_algo(skip_algorithms, "buff_simd")) {
        bench_buff_rust(lossyFloatData, root, out);
    }
    int buff_scale = 10000;
    if (const char *es = std::getenv("COMPRESSION_BENCHMARK_BUFF_INT_SCALE")) {
        char *e2 = nullptr;
        long v = std::strtol(es, &e2, 10);
        if (e2 != es && v > 0) {
            buff_scale = static_cast<int>(v);
        }
    }
    double buff_pred = 1.1509;
    if (const char *ep = std::getenv("COMPRESSION_BENCHMARK_BUFF_PRED")) {
        char *e2 = nullptr;
        double v = std::strtod(ep, &e2);
        if (e2 != ep) {
            buff_pred = v;
        }
    }
    if (!skip_algo(skip_algorithms, "buff_kernel_native")) {
        bench_buff_native_kernel(columnDoubleData, buff_scale, buff_pred, out);
    }
    if (!skip_algo(skip_algorithms, "ext_cmix")) {
        bench_cmix_float(lossyFloatData, root, out);
    }
}
