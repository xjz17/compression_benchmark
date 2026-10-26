#include "run_java_companion.hpp"

#include <cstdlib>
#include <sstream>
#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

#if !defined(_WIN32)
bool posix_have_command(const char *name) {
    std::string s = "command -v ";
    s += name;
    s += " >/dev/null 2>&1";
    int st = std::system(s.c_str());
    if (st == -1) {
        return false;
    }
#if defined(WIFEXITED) && defined(WEXITSTATUS)
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
#else
    return st == 0;
#endif
}
#endif

/** When ``COMPRESSION_BENCHMARK_PYTHON`` is unset: prefer ``python3``, else ``python``, else ``python3`` as last resort. */
std::string default_python_invocation() {
#if defined(_WIN32)
    return "python";
#else
    if (posix_have_command("python3")) {
        return "python3";
    }
    if (posix_have_command("python")) {
        return "python";
    }
    return "python3";
#endif
}

} // namespace

bool run_java_companion_phase(const std::string &repoRoot, const std::string &inputCsvPath,
                              const std::string &dataType, const std::string &appendReportPath,
                              const char *phase) {
    const char *skip = std::getenv("COMPRESSION_BENCHMARK_CPP_SKIP_JAVA_COMPANION");
    if (skip && skip[0] == '1') {
        return true;
    }
    std::string pyExe;
    const char *py = std::getenv("COMPRESSION_BENCHMARK_PYTHON");
    if (py && *py) {
        pyExe = py;
    } else {
        pyExe = default_python_invocation();
    }
    std::ostringstream cmd;
    cmd << pyExe << " \"" << repoRoot << "/scripts/cpp_java_companion.py\""
        << " --input \"" << inputCsvPath << "\""
        << " --data-type " << dataType << " --append \"" << appendReportPath << "\""
        << " --phase " << phase;
    int rc = std::system(cmd.str().c_str());
    return rc == 0;
}
