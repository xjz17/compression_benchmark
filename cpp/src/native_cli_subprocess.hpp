#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#ifndef popen
#define popen _popen
#endif
#ifndef pclose
#define pclose _pclose
#endif
inline long subprocess_pid() { return static_cast<long>(_getpid()); }
#else
#include <unistd.h>
inline long subprocess_pid() { return static_cast<long>(getpid()); }
#endif

namespace native_cli_subprocess {

/** Build the shell command used by {@link run_java_capture_32} (for diagnostics). */
std::string build_java_popen_command(const std::vector<std::string> &args);

/** Run {@code java} with argv-style args; read exactly 32 bytes from stdout. */
bool run_java_capture_32(const std::vector<std::string> &args, std::array<std::uint8_t, 32> &out,
                         int &exit_code);

} // namespace native_cli_subprocess
