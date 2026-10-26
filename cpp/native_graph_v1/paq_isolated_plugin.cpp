#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// PAQ8 upstream keeps hundreds of mutable namespace/function statics.  This
// translation unit is built as a hidden-symbol shared object and loaded only
// for one encode or decode operation.  dlclose therefore destroys the complete
// upstream state before the next independent frame.
#include "../../external/compression/cmix/src/models/paq8.cpp"

char* dictionary_path = nullptr;

namespace {
using Bytes = std::vector<std::uint8_t>;

class Model {
 public:
  explicit Model(unsigned level,std::size_t raw_size,bool raw_block_header) {
    if (level > 3) throw std::runtime_error("PAQ level must be 0..3");
    paq8::level = static_cast<int>(level);
    paq8::buf.setsize(paq8::MEM() * 8);
    predictor_ = std::make_unique<paq8::Predictor>();
    // This vendored PAQ predictor consumes cmix preprocessor blocks, whose
    // first five bytes are a file-type and a big-endian payload length. Raw
    // numeric bytes must not be mistaken for an IMAGE/AUDIO block header.
    // The v2 initialization feeds a DEFAULT header derived from the complete
    // frame's raw length on both sides. No future payload symbol is observed.
    if(raw_block_header){
      const std::uint8_t header[5]={0,static_cast<std::uint8_t>(raw_size>>24),
        static_cast<std::uint8_t>(raw_size>>16),static_cast<std::uint8_t>(raw_size>>8),
        static_cast<std::uint8_t>(raw_size)};
      for(auto byte:header)for(int shift=7;shift>=0;--shift){
        (void)predict();observe((byte>>shift)&1U);
      }
    }
  }
  double predict() const { return predictor_->p() / 4096.0; }
  void observe(unsigned bit) {
    paq8::y = static_cast<int>(bit);
    predictor_->update();
  }

 private:
  std::unique_ptr<paq8::Predictor> predictor_;
};

std::uint32_t probability_one(double value) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0)
    throw std::runtime_error("PAQ returned an invalid probability");
  const auto scaled = 1U + static_cast<std::uint32_t>(value * 65534.0);
  return std::min<std::uint32_t>(scaled, 65535U);
}

struct BitWriter {
  Bytes bytes;
  unsigned used = 0;
  void put(unsigned bit) {
    if (used == 0) bytes.push_back(0);
    if (bit) bytes.back() |= static_cast<std::uint8_t>(1U << (7U - used));
    used = (used + 1U) & 7U;
  }
};

struct BitReader {
  std::span<const std::uint8_t> bytes;
  std::size_t bit = 0;
  unsigned get() {
    if (bit >= bytes.size() * 8U + 32U)
      throw std::runtime_error("PAQ arithmetic padding exhausted");
    const auto index = bit++;
    return index < bytes.size() * 8U
               ? (bytes[index / 8U] >> (7U - index % 8U)) & 1U
               : 0U;
  }
};

Bytes encode(std::span<const std::uint8_t> raw, unsigned level,bool raw_block_header=false) {
  if (raw.size() > 16ULL * 1024 * 1024)
    throw std::runtime_error("PAQ input exceeds 16 MiB");
  Model model(level,raw.size(),raw_block_header);
  constexpr std::uint32_t half = 0x80000000U;
  constexpr std::uint32_t quarter = 0x40000000U;
  constexpr std::uint32_t three = 0xc0000000U;
  constexpr std::uint32_t total = 65536U;
  std::uint32_t low = 0, high = 0xffffffffU;
  std::uint64_t pending = 0;
  BitWriter writer;
  auto emit = [&](unsigned bit) {
    writer.put(bit);
    while (pending) { writer.put(bit ^ 1U); --pending; }
  };
  for (const auto byte : raw) {
    for (int shift = 7; shift >= 0; --shift) {
      const auto bit = (byte >> shift) & 1U;
      const auto one = probability_one(model.predict());
      const auto zero = total - one;
      const std::uint64_t range = std::uint64_t{high} - low + 1;
      const auto split = low + static_cast<std::uint32_t>(range * zero / total);
      if (split == low || split > high)
        throw std::runtime_error("PAQ arithmetic interval underflow");
      if (bit) low = split; else high = split - 1;
      for (;;) {
        if (high < half) emit(0);
        else if (low >= half) { emit(1); low -= half; high -= half; }
        else if (low >= quarter && high < three) {
          ++pending; low -= quarter; high -= quarter;
        } else break;
        low <<= 1; high = (high << 1) | 1U;
      }
      model.observe(bit);
    }
  }
  ++pending;
  emit(low >= quarter ? 1U : 0U);
  return writer.bytes;
}

Bytes decode(std::span<const std::uint8_t> body, std::size_t length,
             unsigned level,bool raw_block_header=false) {
  if (length > 16ULL * 1024 * 1024 || body.empty())
    throw std::runtime_error("invalid PAQ entropy shape");
  Model model(level,length,raw_block_header);
  constexpr std::uint32_t half = 0x80000000U;
  constexpr std::uint32_t quarter = 0x40000000U;
  constexpr std::uint32_t three = 0xc0000000U;
  constexpr std::uint32_t total = 65536U;
  std::uint32_t low = 0, high = 0xffffffffU, code = 0;
  BitReader reader{body};
  for (unsigned i = 0; i < 32; ++i) code = (code << 1) | reader.get();
  Bytes raw;
  raw.reserve(length);
  for (std::size_t index = 0; index < length; ++index) {
    std::uint8_t byte = 0;
    for (unsigned j = 0; j < 8; ++j) {
      if (code < low || code > high)
        throw std::runtime_error("PAQ arithmetic code outside interval");
      const auto one = probability_one(model.predict());
      const auto zero = total - one;
      const std::uint64_t range = std::uint64_t{high} - low + 1;
      const auto split = low + static_cast<std::uint32_t>(range * zero / total);
      if (split == low || split > high)
        throw std::runtime_error("PAQ arithmetic interval underflow");
      const auto bit = code >= split ? 1U : 0U;
      if (bit) low = split; else high = split - 1;
      for (;;) {
        if (high < half) {}
        else if (low >= half) { low -= half; high -= half; code -= half; }
        else if (low >= quarter && high < three) {
          low -= quarter; high -= quarter; code -= quarter;
        } else break;
        low <<= 1; high = (high << 1) | 1U;
        code = (code << 1) | reader.get();
      }
      byte = static_cast<std::uint8_t>((byte << 1) | bit);
      model.observe(bit);
    }
    raw.push_back(byte);
  }
  return raw;
}

void set_error(char* error, std::size_t capacity, const std::string& message) {
  if (!error || capacity == 0) return;
  const auto size = std::min(capacity - 1, message.size());
  std::memcpy(error, message.data(), size);
  error[size] = '\0';
}

int copy_result(const Bytes& result, std::uint8_t** output,
                std::size_t* output_size) {
  if (!output || !output_size) return -1;
  auto* memory = static_cast<std::uint8_t*>(std::malloc(result.size() ? result.size() : 1));
  if (!memory) return -1;
  if (!result.empty()) std::memcpy(memory, result.data(), result.size());
  *output = memory;
  *output_size = result.size();
  return 0;
}
}  // namespace

#if defined(_WIN32)
#define NG_PAQ_EXPORT __declspec(dllexport)
#else
#define NG_PAQ_EXPORT __attribute__((visibility("default")))
#endif

extern "C" NG_PAQ_EXPORT int native_graph_paq_encode(
    const std::uint8_t* input, std::size_t input_size, unsigned level,
    std::uint8_t** output, std::size_t* output_size,
    char* error, std::size_t error_capacity) {
  try {
    if (!input && input_size) throw std::runtime_error("null PAQ input");
    return copy_result(encode({input, input_size}, level), output, output_size);
  } catch (const std::exception& exception) {
    set_error(error, error_capacity, exception.what());
    return -1;
  }
}

extern "C" NG_PAQ_EXPORT int native_graph_paq_decode(
    const std::uint8_t* input, std::size_t input_size, std::size_t raw_size,
    unsigned level, std::uint8_t** output, std::size_t* output_size,
    char* error, std::size_t error_capacity) {
  try {
    if (!input && input_size) throw std::runtime_error("null PAQ body");
    return copy_result(decode({input, input_size}, raw_size, level), output, output_size);
  } catch (const std::exception& exception) {
    set_error(error, error_capacity, exception.what());
    return -1;
  }
}

extern "C" NG_PAQ_EXPORT void native_graph_paq_free(void* memory) {
  std::free(memory);
}

// Separate ABI names preserve decoding of previously valid v1 frames; the
// graph stage records v2 explicitly in its complete-frame side metadata.
extern "C" NG_PAQ_EXPORT int native_graph_paq_encode_v2(
    const std::uint8_t* input,std::size_t input_size,unsigned level,
    std::uint8_t** output,std::size_t* output_size,char* error,std::size_t capacity){
  try{
    if(!input&&input_size)throw std::runtime_error("null PAQ input");
    return copy_result(encode({input,input_size},level,true),output,output_size);
  }catch(const std::exception& e){set_error(error,capacity,e.what());return -1;}
}
extern "C" NG_PAQ_EXPORT int native_graph_paq_decode_v2(
    const std::uint8_t* input,std::size_t input_size,std::size_t raw_size,unsigned level,
    std::uint8_t** output,std::size_t* output_size,char* error,std::size_t capacity){
  try{
    if(!input&&input_size)throw std::runtime_error("null PAQ body");
    return copy_result(decode({input,input_size},raw_size,level,true),output,output_size);
  }catch(const std::exception& e){set_error(error,capacity,e.what());return -1;}
}
