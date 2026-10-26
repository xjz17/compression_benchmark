#include "native_cli_subprocess.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace native_cli_subprocess {

namespace {

bool path_has_space(const std::string &s) {
  return s.find(' ') != std::string::npos;
}

#ifdef _WIN32

/** MSVC-style quoting so {@code CreateProcess} receives one token per argument. */
std::string quote_windows_arg(const std::string &src) {
  if (src.empty()) {
    return "\"\"";
  }
  bool needs_quote = false;
  for (const char c : src) {
    if (c == ' ' || c == '\t' || c == '"' || c == '\\') {
      needs_quote = true;
      break;
    }
  }
  if (!needs_quote) {
    return src;
  }
  std::string out = "\"";
  std::size_t backslashes = 0;
  for (const char c : src) {
    if (c == '\\') {
      ++backslashes;
    } else if (c == '"') {
      out.append(backslashes * 2, '\\');
      backslashes = 0;
      out += "\\\"";
    } else {
      if (backslashes != 0) {
        out.append(backslashes, '\\');
        backslashes = 0;
      }
      out += c;
    }
  }
  out.append(backslashes * 2, '\\');
  out += '"';
  return out;
}

std::string build_windows_java_command_line(const std::vector<std::string> &args) {
  std::ostringstream cmd;
  cmd << "java";
  for (const std::string &a : args) {
    cmd << ' ' << quote_windows_arg(a);
  }
  return cmd.str();
}

bool run_java_capture_32_win(const std::vector<std::string> &args, std::array<std::uint8_t, 32> &out,
                             int &exit_code) {
  exit_code = -1;
  std::string cmdline = build_windows_java_command_line(args);
  std::vector<char> cmd_buf(cmdline.begin(), cmdline.end());
  cmd_buf.push_back('\0');

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;

  HANDLE read_pipe = nullptr;
  HANDLE write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
    return false;
  }
  SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  si.hStdOutput = write_pipe;
  si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  PROCESS_INFORMATION pi{};
  const char *cwd = std::getenv("WEB_COMPRESSION_REPO_ROOT");
  const BOOL created = CreateProcessA(nullptr, cmd_buf.data(), nullptr, nullptr, TRUE,
                                    CREATE_NO_WINDOW, nullptr, cwd, &si, &pi);
  CloseHandle(write_pipe);
  if (!created) {
    CloseHandle(read_pipe);
    return false;
  }

  std::size_t got = 0;
  while (got < out.size()) {
    DWORD chunk = 0;
    if (!ReadFile(read_pipe, out.data() + got,
                  static_cast<DWORD>(out.size() - got), &chunk, nullptr) || chunk == 0) {
      break;
    }
    got += chunk;
  }
  CloseHandle(read_pipe);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  exit_code = static_cast<int>(code);
  return got == out.size() && exit_code == 0;
}

#endif // _WIN32

} // namespace

std::string build_java_popen_command(const std::vector<std::string> &args) {
#ifdef _WIN32
  return build_windows_java_command_line(args);
#else
  std::ostringstream cmd;
  cmd << "java";
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string &a = args[i];
    if (a == "-cp" && i + 1 < args.size()) {
      const std::string &cp = args[i + 1];
      cmd << " -cp ";
      if (path_has_space(cp)) {
        cmd << '"' << cp << '"';
      } else {
        cmd << cp;
      }
      ++i;
      continue;
    }
    cmd << ' ';
    if (path_has_space(a)) {
      cmd << '"' << a << '"';
    } else {
      cmd << a;
    }
  }
  return cmd.str();
#endif
}

bool run_java_capture_32(const std::vector<std::string> &args, std::array<std::uint8_t, 32> &out,
                         int &exit_code) {
  exit_code = -1;
  if (args.empty()) {
    return false;
  }
#ifdef _WIN32
  return run_java_capture_32_win(args, out, exit_code);
#else
  const std::string cmd = build_java_popen_command(args);
  FILE *p = popen(cmd.c_str(), "r");
  if (p == nullptr) {
    return false;
  }
  const std::size_t n = std::fread(out.data(), 1, out.size(), p);
  exit_code = pclose(p);
  return n == out.size() && exit_code == 0;
#endif
}

} // namespace native_cli_subprocess
