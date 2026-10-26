#include "lzham_core.h"

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdio>

int sprintf_s(char* buffer, std::size_t capacity, const char* format, ...) {
  if (capacity == 0) return 0;
  va_list arguments;
  va_start(arguments, format);
  const int result = std::vsnprintf(buffer, capacity, format, arguments);
  va_end(arguments);
  buffer[capacity - 1] = '\0';
  return result < 0 ? static_cast<int>(capacity - 1)
                    : std::min(result, static_cast<int>(capacity - 1));
}

int vsprintf_s(char* buffer, std::size_t capacity, const char* format, va_list arguments) {
  if (capacity == 0) return 0;
  const int result = std::vsnprintf(buffer, capacity, format, arguments);
  buffer[capacity - 1] = '\0';
  return result < 0 ? static_cast<int>(capacity - 1)
                    : std::min(result, static_cast<int>(capacity - 1));
}

bool lzham_is_debugger_present() { return false; }
void lzham_debug_break() { assert(false && "LZHAM debug break"); }
void lzham_output_debug_string(const char* message) {
  if (message) std::fputs(message, stderr);
}
