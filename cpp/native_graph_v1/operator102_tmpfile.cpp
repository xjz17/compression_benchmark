#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

FILE* data_tmpfile() noexcept {
  const char* const configured = std::getenv("TMPDIR");
  if (configured == nullptr || configured[0] != '/') {
    errno = EINVAL;
    return nullptr;
  }
  struct stat info {};
  if (::stat(configured, &info) != 0) return nullptr;
  if (!S_ISDIR(info.st_mode)) {
    errno = ENOTDIR;
    return nullptr;
  }

  try {
    std::string pattern(configured);
    if (pattern.back() != '/') pattern.push_back('/');
    pattern += "operator102-tmp-XXXXXX";
    const int descriptor = ::mkstemp(pattern.data());
    if (descriptor < 0) return nullptr;
    const int flags = ::fcntl(descriptor, F_GETFD);
    if (flags >= 0) static_cast<void>(::fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC));
    if (::unlink(pattern.c_str()) != 0) {
      const int failure = errno;
      ::close(descriptor);
      errno = failure;
      return nullptr;
    }
    FILE* const stream = ::fdopen(descriptor, "w+b");
    if (stream == nullptr) {
      const int failure = errno;
      ::close(descriptor);
      errno = failure;
      return nullptr;
    }
    return stream;
  } catch (...) {
    errno = ENOMEM;
    return nullptr;
  }
}

}  // namespace

extern "C" FILE* tmpfile() { return data_tmpfile(); }

#if defined(__linux__)
extern "C" FILE* tmpfile64() { return data_tmpfile(); }
#endif
