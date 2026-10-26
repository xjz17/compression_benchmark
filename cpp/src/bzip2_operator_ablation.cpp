#include "bzip2_operator_ablation.hpp"

#include <bzlib.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace bzip2_operator_ablation {
namespace {

int mode_for(std::string_view omitted_operator) {
  if (omitted_operator.empty()) return 0;
  if (omitted_operator == "block_transform") return 1;
  if (omitted_operator == "run_length") return 2;
  if (omitted_operator == "symbol_recode") return 3;
  if (omitted_operator == "entropy_huffman") return 4;
  throw std::invalid_argument("unsupported Bzip2 ablation operator: " +
                              std::string(omitted_operator));
}

std::mutex& ablation_mutex() {
  static std::mutex mutex;
  return mutex;
}

class ModeReset {
 public:
  explicit ModeReset(int mode) { BZ2_setAblationMode(mode); }
  ~ModeReset() { BZ2_setAblationMode(0); }
};

unsigned int checked_size(std::size_t size, std::string_view label) {
  if (size > std::numeric_limits<unsigned int>::max()) {
    throw std::runtime_error("Bzip2 " + std::string(label) + " exceeds uint32 size");
  }
  return static_cast<unsigned int>(size);
}

}  // namespace

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> input, int level,
                                 std::string_view omitted_operator) {
  if (level < 1 || level > 9) {
    throw std::invalid_argument("Bzip2 level must be in [1, 9]");
  }
  const unsigned int input_size = checked_size(input.size(), "input");
  unsigned int output_size = input_size + input_size / 100 + 601;
  std::vector<std::uint8_t> output(output_size);
  // libbz2 accepts a zero-length input and emits a valid native empty stream,
  // but it still requires a non-null source pointer.
  std::uint8_t empty_input = 0;
  const auto* input_data = input.empty() ? &empty_input : input.data();
  const std::lock_guard<std::mutex> lock(ablation_mutex());
  ModeReset reset(mode_for(omitted_operator));
  const int status = BZ2_bzBuffToBuffCompress(
      reinterpret_cast<char*>(output.data()), &output_size,
      const_cast<char*>(reinterpret_cast<const char*>(input_data)), input_size,
      level, 0, 30);
  if (status != BZ_OK) {
    throw std::runtime_error("Bzip2 compression failed: " + std::to_string(status));
  }
  output.resize(output_size);
  return output;
}

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded,
                                 std::size_t expected_size) {
  unsigned int input_size = checked_size(encoded.size(), "payload");
  unsigned int output_size = checked_size(expected_size, "decoded output");
  std::vector<std::uint8_t> output(expected_size);
  // As for compression, libbz2 requires a non-null destination pointer even
  // when the decoded output size is exactly zero.
  std::uint8_t empty_output = 0;
  auto* output_data = output.empty() ? &empty_output : output.data();
  const int status = BZ2_bzBuffToBuffDecompress(
      reinterpret_cast<char*>(output_data), &output_size,
      const_cast<char*>(reinterpret_cast<const char*>(encoded.data())), input_size,
      0, 0);
  if (status != BZ_OK || output_size != expected_size) {
    throw std::runtime_error("Bzip2 decompression failed: " + std::to_string(status));
  }
  return output;
}

}  // namespace bzip2_operator_ablation
