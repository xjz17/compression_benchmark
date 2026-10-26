#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Bytes = std::vector<std::uint8_t>;
using Encode = int (*)(const std::uint8_t*, std::size_t, unsigned,
                       std::uint8_t**, std::size_t*, char*, std::size_t);
using Decode = int (*)(const std::uint8_t*, std::size_t, std::size_t, unsigned,
                       std::uint8_t**, std::size_t*, char*, std::size_t);
using Free = void (*)(void*);

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class Plugin {
 public:
  Plugin(const char* path, bool v2) : path_(path) {
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    const char* error = handle_ ? nullptr : dlerror();
    require(handle_ != nullptr,
            std::string("dlopen failed: ") + (error ? error : "unknown"));
    encode_ = symbol<Encode>(v2 ? "native_graph_paq_encode_v2"
                                : "native_graph_paq_encode");
    decode_ = symbol<Decode>(v2 ? "native_graph_paq_decode_v2"
                                : "native_graph_paq_decode");
    free_ = symbol<Free>("native_graph_paq_free");
  }
  ~Plugin() {
    if (handle_) dlclose(handle_);
  }
  Plugin(const Plugin&) = delete;
  Plugin& operator=(const Plugin&) = delete;

  Bytes encode(const Bytes& input) const {
    std::uint8_t* output = nullptr;
    std::size_t size = 0;
    std::array<char, 512> error{};
    const int status = encode_(input.data(), input.size(), 0, &output, &size,
                               error.data(), error.size());
    require(status == 0,
            std::string("encode failed: ") + error.data());
    require(output != nullptr, "encode returned null output");
    Bytes result(output, output + size);
    free_(output);
    return result;
  }

  Bytes decode(const Bytes& input, std::size_t raw_size) const {
    std::uint8_t* output = nullptr;
    std::size_t size = 0;
    std::array<char, 512> error{};
    const int status = decode_(input.data(), input.size(), raw_size, 0, &output,
                               &size, error.data(), error.size());
    require(status == 0,
            std::string("decode failed: ") + error.data());
    require(output != nullptr && size == raw_size,
            "decode returned a wrong-sized result");
    Bytes result(output, output + size);
    free_(output);
    return result;
  }

 private:
  template <class Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle_, name);
    const char* error = dlerror();
    require(error == nullptr && address != nullptr,
            std::string("missing symbol: ") + name);
    return reinterpret_cast<Function>(address);
  }
  const char* path_;
  void* handle_ = nullptr;
  Encode encode_ = nullptr;
  Decode decode_ = nullptr;
  Free free_ = nullptr;
};

void require_unloaded(const char* path) {
#ifdef RTLD_NOLOAD
  void* resident = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (resident) dlclose(resident);
  require(resident == nullptr, "PAQ DSO remained resident after dlclose");
#else
  (void)path;
#endif
}

Bytes encode_fresh(const char* path, const Bytes& input, bool v2) {
  Bytes result;
  {
    Plugin plugin(path, v2);
    result = plugin.encode(input);
  }
  require_unloaded(path);
  return result;
}

Bytes decode_fresh(const char* path, const Bytes& input, std::size_t raw_size,
                   bool v2) {
  Bytes result;
  {
    Plugin plugin(path, v2);
    result = plugin.decode(input, raw_size);
  }
  require_unloaded(path);
  return result;
}

std::string hex(const Bytes& bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const auto byte : bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

Bytes unhex(const char* text) {
  const std::string input(text);
  require(input.size() % 2 == 0, "odd hexadecimal input");
  const auto nibble = [](char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::runtime_error("invalid hexadecimal input");
  };
  Bytes result(input.size() / 2);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>((nibble(input[i * 2]) << 4) |
                                          nibble(input[i * 2 + 1]));
  return result;
}

void v2_roundtrip(const char* path, const Bytes& input) {
  const auto encoded = encode_fresh(path, input, true);
  require(decode_fresh(path, encoded, input.size(), true) == input,
          "v2 roundtrip mismatch");
}

void cold_decode_child(const char* executable, const char* path,
                       const Bytes& encoded,
                       const Bytes& expected) {
  const auto encoded_hex = hex(encoded);
  const auto expected_hex = hex(expected);
  const pid_t child = fork();
  require(child >= 0, "fork failed for cold decode");
  if (child == 0) {
    alarm(60);
    execl(executable, executable, "--cold-decode-v2", path,
          encoded_hex.c_str(), expected_hex.c_str(), nullptr);
    _exit(73);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child, "cold waitpid failed");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "cross-process v2 cold decode failed");
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc == 5 && std::string(argv[1]) == "--cold-decode-v2") {
    const auto encoded = unhex(argv[3]);
    const auto expected = unhex(argv[4]);
    return decode_fresh(argv[2], encoded, expected.size(), true) == expected
               ? 0
               : 71;
  }
  require(argc == 2, "usage: paq_raw_block_v2_test PLUGIN_PATH");
  const char* path = argv[1];

  Bytes empty;
  Bytes zeros(4096, 0);
  Bytes repeated(4096);
  for (std::size_t i = 0; i < repeated.size(); ++i)
    repeated[i] = static_cast<std::uint8_t>((i / 32) & 3U);
  Bytes random(4093);
  std::uint32_t state = 0x12345678U;
  for (auto& value : random) {
    state = state * 1664525U + 1013904223U;
    value = static_cast<std::uint8_t>(state >> 24);
  }
  const std::array<std::uint64_t, 10> float_bits{
      0x0000000000000000ULL, 0x8000000000000000ULL,
      0x3ff0000000000000ULL, 0xbff0000000000000ULL,
      0x7ff0000000000000ULL, 0xfff0000000000000ULL,
      0x7ff8000000000001ULL, 0x7ff0000000000001ULL,
      0x0000000000000001ULL, 0x000fffffffffffffULL};
  Bytes floating(sizeof(float_bits));
  std::memcpy(floating.data(), float_bits.data(), floating.size());

  const std::array<const Bytes*, 5> cases{
      &empty, &zeros, &repeated, &random, &floating};
  for (const auto* input : cases) {
    for (unsigned repeat = 0; repeat < 3; ++repeat) v2_roundtrip(path, *input);
    const auto frame = encode_fresh(path, *input, true);
    cold_decode_child(argv[0], path, frame, *input);
  }

  // v1 ABI remains usable for a previously valid cmix-preprocessed DEFAULT
  // block.  This is an ABI regression test, not a relabeling of old frames.
  Bytes old_valid{0, 0, 0, 0, 32};
  old_valid.resize(37, 0x41);
  const auto old_frame = encode_fresh(path, old_valid, false);
  require(decode_fresh(path, old_frame, old_valid.size(), false) == old_valid,
          "v1 valid-frame regression");

  std::cout << "PASS v2_same_process_roundtrip_cases=5 repeats=3\n"
               "PASS v2_cross_process_cold_decode_cases=5\n"
               "PASS v1_valid_frame_roundtrip\n"
               "PASS inputs=empty,zeros,repeated,random,float64_bits\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL " << error.what() << '\n';
  return 1;
}
