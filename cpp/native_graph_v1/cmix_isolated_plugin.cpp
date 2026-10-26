#include "../src/prediction_probability_model.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

// The upstream cmix predictor owns several multi-gigabyte function-static
// tables.  dlclose resets their values, but upstream intentionally never frees
// every table, so the process allocator retained roughly 4 GiB after each
// codec call.  A worker limited to 64 GiB consequently failed after eight
// calls.  On Linux, all allocation paths used by this DSO are therefore routed
// into one private mmap arena.  The host copies the ABI result while the DSO is
// loaded, runs dlclose (and hence all destructors), and only then munmaps the
// complete arena.  Loading, construction, copying, destruction, and unmapping
// all remain inside the timed stage call.
#if defined(__linux__)
namespace cmix_isolated_arena {
// Model construction has a much larger transient peak than the roughly 4 GiB
// that remained leaked after an upstream call.  Reserve 48 GiB of virtual
// address space so the real peak fits while retaining at least 16 GiB of the
// worker's enforced 64 GiB address-space limit for code and host buffers.  The
// macro is overridden only by the separate exhaustion-test build.
#ifndef NATIVE_GRAPH_V1_CMIX_ARENA_CAPACITY_BYTES
#define NATIVE_GRAPH_V1_CMIX_ARENA_CAPACITY_BYTES (UINT64_C(48) << 30)
#endif
constexpr std::size_t kCapacity =
    static_cast<std::size_t>(NATIVE_GRAPH_V1_CMIX_ARENA_CAPACITY_BYTES);
static_assert(kCapacity >= (std::size_t{1} << 20));
constexpr unsigned kBinCount = 64;
constexpr std::size_t kBlockAlignment = alignof(std::max_align_t);
constexpr std::uint64_t kBlockMagic = UINT64_C(0x434d4958424c4f43);
constexpr std::uint64_t kAllocationMagic = UINT64_C(0x434d4958414c4c4f);

struct Block {
  std::uint64_t magic;
  std::size_t total_size;
  std::size_t previous_size;
  Block* previous_free;
  Block* next_free;
  std::uint32_t free;
  std::uint32_t reserved;
};

struct Allocation {
  std::uint64_t magic;
  Block* block;
  std::size_t size;
};

struct State {
  std::uint8_t* base;
  Block* free_lists[kBinCount];
  std::size_t live_bytes;
  std::size_t peak_live_bytes;
};

State state{};

bool valid_alignment(std::size_t alignment) noexcept {
  return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

bool add_overflow(std::size_t lhs, std::size_t rhs,
                  std::size_t* result) noexcept {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) return true;
  *result = lhs + rhs;
  return false;
}

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    return 0;
  return (value + alignment - 1) & ~(alignment - 1);
}

unsigned bin_for(std::size_t size) noexcept {
  unsigned bin = 0;
  while (size > 1 && bin + 1 < kBinCount) {
    size >>= 1;
    ++bin;
  }
  return bin;
}

void insert_free(Block* block) noexcept {
  block->free = 1;
  block->previous_free = nullptr;
  const auto bin = bin_for(block->total_size);
  block->next_free = state.free_lists[bin];
  if (block->next_free) block->next_free->previous_free = block;
  state.free_lists[bin] = block;
}

void remove_free(Block* block) noexcept {
  const auto bin = bin_for(block->total_size);
  if (block->previous_free) block->previous_free->next_free = block->next_free;
  else state.free_lists[bin] = block->next_free;
  if (block->next_free) block->next_free->previous_free = block->previous_free;
  block->previous_free = nullptr;
  block->next_free = nullptr;
}

bool ensure_arena() noexcept {
  if (state.base) return true;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  void* mapping = mmap(nullptr, kCapacity, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (mapping == MAP_FAILED) return false;
  state.base = static_cast<std::uint8_t*>(mapping);
  state.live_bytes = 0;
  state.peak_live_bytes = 0;
  auto* block = reinterpret_cast<Block*>(state.base);
  *block = {kBlockMagic, kCapacity, 0, nullptr, nullptr, 1, 0};
  insert_free(block);
  return true;
}

Block* next_physical(Block* block) noexcept {
  auto* next = reinterpret_cast<std::uint8_t*>(block) + block->total_size;
  return next < state.base + kCapacity ? reinterpret_cast<Block*>(next) : nullptr;
}

void* allocate(std::size_t size, std::size_t alignment) noexcept {
  if (size == 0) size = 1;
  if (alignment < alignof(std::max_align_t)) alignment = alignof(std::max_align_t);
  if (!valid_alignment(alignment) || !ensure_arena()) return nullptr;

  std::size_t required = 0;
  if (add_overflow(sizeof(Block), sizeof(Allocation), &required) ||
      add_overflow(required, alignment - 1, &required) ||
      add_overflow(required, size, &required))
    return nullptr;
  required = align_up(required, kBlockAlignment);
  if (!required || required > kCapacity) return nullptr;

  Block* block = nullptr;
  for (unsigned bin = bin_for(required); bin < kBinCount && !block; ++bin)
    for (Block* candidate = state.free_lists[bin]; candidate;
         candidate = candidate->next_free)
      if (candidate->total_size >= required) {
        block = candidate;
        break;
      }
  if (!block) return nullptr;
  remove_free(block);

  constexpr std::size_t minimum_remainder =
      sizeof(Block) + sizeof(Allocation) + kBlockAlignment;
  const auto old_total = block->total_size;
  if (old_total - required >= minimum_remainder) {
    block->total_size = required;
    auto* remainder = reinterpret_cast<Block*>(
        reinterpret_cast<std::uint8_t*>(block) + required);
    *remainder = {kBlockMagic, old_total - required, required,
                  nullptr, nullptr, 1, 0};
    if (Block* following = next_physical(remainder))
      following->previous_size = remainder->total_size;
    insert_free(remainder);
  }
  block->magic = kBlockMagic;
  block->free = 0;
  const auto begin = reinterpret_cast<std::uintptr_t>(block);
  const auto candidate = begin + sizeof(Block) + sizeof(Allocation);
  const auto aligned = (candidate + alignment - 1) & ~(alignment - 1);
  auto* record = reinterpret_cast<Allocation*>(aligned - sizeof(Allocation));
  *record = {kAllocationMagic, block, size};
  state.live_bytes += block->total_size;
  state.peak_live_bytes = std::max(state.peak_live_bytes, state.live_bytes);
  return reinterpret_cast<void*>(aligned);
}

Allocation* allocation(void* memory) noexcept {
  if (!memory || !state.base) return nullptr;
  const auto address = reinterpret_cast<std::uintptr_t>(memory);
  const auto begin = reinterpret_cast<std::uintptr_t>(state.base);
  if (address < begin + sizeof(Block) + sizeof(Allocation) ||
      address >= begin + kCapacity)
    return nullptr;
  auto* value = reinterpret_cast<Allocation*>(address - sizeof(Allocation));
  if (value->magic != kAllocationMagic || !value->block ||
      value->block->magic != kBlockMagic || value->block->free != 0)
    return nullptr;
  return value;
}

void deallocate(void* memory) noexcept {
  Allocation* record = allocation(memory);
  if (!record) return;
  Block* block = record->block;
  record->magic = 0;
  state.live_bytes -= block->total_size;
  block->free = 1;

  if (Block* next = next_physical(block);
      next && next->magic == kBlockMagic && next->free != 0) {
    remove_free(next);
    block->total_size += next->total_size;
    next->magic = 0;
  }
  if (block->previous_size) {
    auto* previous = reinterpret_cast<Block*>(
        reinterpret_cast<std::uint8_t*>(block) - block->previous_size);
    if (previous->magic == kBlockMagic && previous->free != 0) {
      remove_free(previous);
      previous->total_size += block->total_size;
      block->magic = 0;
      block = previous;
    }
  }
  if (Block* following = next_physical(block))
    following->previous_size = block->total_size;
  insert_free(block);
}

void* reallocate(void* memory, std::size_t size) noexcept {
  if (!memory) return allocate(size, alignof(std::max_align_t));
  if (size == 0) {
    deallocate(memory);
    return nullptr;
  }
  auto* old = allocation(memory);
  if (!old) return nullptr;
  const auto block_end = reinterpret_cast<std::uintptr_t>(old->block) +
                         old->block->total_size;
  const auto available = block_end - reinterpret_cast<std::uintptr_t>(memory);
  if (size <= available) {
    old->size = size;
    return memory;
  }
  void* replacement = allocate(size, alignof(std::max_align_t));
  if (!replacement) return nullptr;
  std::memcpy(replacement, memory, old->size);
  deallocate(memory);
  return replacement;
}
}  // namespace cmix_isolated_arena

// cmix_allocation_redirect.hpp maps the direct C allocator spellings in the
// upstream translation units to these checked C++ functions.  Unlike linker
// wrapping a libc malloc declaration (which is noexcept in C++), these calls
// may safely unwind std::bad_alloc to the exported codec ABI catch boundary.
// The free-list allocator reuses temporaries within one call; the host then
// reclaims even intentionally leaked function-static tables in one munmap.
void* checked_allocate(std::size_t size, std::size_t alignment) {
  if (void* memory = cmix_isolated_arena::allocate(size, alignment)) return memory;
  throw std::bad_alloc();
}
extern "C" void* native_graph_cmix_checked_malloc(std::size_t size) {
  return checked_allocate(size, alignof(std::max_align_t));
}
extern "C" void* native_graph_cmix_checked_calloc(std::size_t count,
                                                   std::size_t size) {
  if (count != 0 && size > std::numeric_limits<std::size_t>::max() / count)
    throw std::bad_alloc();
  const auto bytes = count * size;
  void* memory = checked_allocate(bytes, alignof(std::max_align_t));
  std::memset(memory, 0, bytes);
  return memory;
}
extern "C" void* native_graph_cmix_checked_realloc(void* memory,
                                                    std::size_t size) {
  if (size == 0) {
    cmix_isolated_arena::deallocate(memory);
    return nullptr;
  }
  if (void* replacement = cmix_isolated_arena::reallocate(memory, size))
    return replacement;
  throw std::bad_alloc();
}
extern "C" void native_graph_cmix_checked_free(void* memory) noexcept {
  cmix_isolated_arena::deallocate(memory);
}
extern "C" void* native_graph_cmix_checked_aligned_alloc(
    std::size_t alignment, std::size_t size) {
  if (!cmix_isolated_arena::valid_alignment(alignment) ||
      size % alignment != 0)
    throw std::invalid_argument("invalid cmix aligned allocation");
  return checked_allocate(size, alignment);
}
extern "C" int native_graph_cmix_checked_posix_memalign(
    void** result, std::size_t alignment, std::size_t size) {
  if (!result || alignment < sizeof(void*) ||
      alignment % sizeof(void*) != 0 ||
      !cmix_isolated_arena::valid_alignment(alignment))
    return EINVAL;
  *result = checked_allocate(size, alignment);
  return 0;
}

void* operator new(std::size_t size) {
  if (void* memory = cmix_isolated_arena::allocate(size, alignof(std::max_align_t)))
    return memory;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return cmix_isolated_arena::allocate(size, alignof(std::max_align_t));
}
void* operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
  return ::operator new(size, tag);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  if (void* memory = cmix_isolated_arena::allocate(
          size, static_cast<std::size_t>(alignment)))
    return memory;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void* operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  return cmix_isolated_arena::allocate(size,
                                       static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t& tag) noexcept {
  return ::operator new(size, alignment, tag);
}
void operator delete(void* memory) noexcept { cmix_isolated_arena::deallocate(memory); }
void operator delete[](void* memory) noexcept { ::operator delete(memory); }
void operator delete(void* memory, std::size_t) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }
void operator delete(void* memory, const std::nothrow_t&) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, const std::nothrow_t&) noexcept { ::operator delete(memory); }
void operator delete(void* memory, std::align_val_t) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { ::operator delete(memory); }
void operator delete(void* memory, std::size_t, std::align_val_t) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept { ::operator delete(memory); }
void operator delete(void* memory, std::align_val_t,
                     const std::nothrow_t&) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::align_val_t,
                       const std::nothrow_t&) noexcept { ::operator delete(memory); }
#endif

// The full upstream cmix Predictor embeds PAQ/PPMd components whose source has
// process-global and function-static state.  This file is linked into a
// hidden-symbol DSO.  One exported call creates exactly one fresh predictor;
// the dispatcher unloads the DSO before starting another independent frame.
namespace {
using Bytes = std::vector<std::uint8_t>;

class Model {
 public:
  Model() {
    prediction_probability::Options options;
    options.provider = "cmix";
    options.allow_large_memory = true;
    model_ = std::make_unique<prediction_probability::ProbabilityModel>(options);
  }
  double predict() { return model_->predict(); }
  void observe(unsigned bit) { model_->observe(bit); }

 private:
  std::unique_ptr<prediction_probability::ProbabilityModel> model_;
};

std::uint32_t probability_one(double value) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0)
    throw std::runtime_error("cmix returned an invalid probability");
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
      throw std::runtime_error("cmix arithmetic padding exhausted");
    const auto index = bit++;
    return index < bytes.size() * 8U
               ? (bytes[index / 8U] >> (7U - index % 8U)) & 1U
               : 0U;
  }
};

Bytes encode(std::span<const std::uint8_t> raw) {
  if (raw.size() > 16ULL * 1024 * 1024)
    throw std::runtime_error("cmix input exceeds 16 MiB");
  Model model;
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
        throw std::runtime_error("cmix arithmetic interval underflow");
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

Bytes decode(std::span<const std::uint8_t> body, std::size_t length) {
  if (length > 16ULL * 1024 * 1024 || body.empty())
    throw std::runtime_error("invalid cmix entropy shape");
  Model model;
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
        throw std::runtime_error("cmix arithmetic code outside interval");
      const auto one = probability_one(model.predict());
      const auto zero = total - one;
      const std::uint64_t range = std::uint64_t{high} - low + 1;
      const auto split = low + static_cast<std::uint32_t>(range * zero / total);
      if (split == low || split > high)
        throw std::runtime_error("cmix arithmetic interval underflow");
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
  auto* memory = static_cast<std::uint8_t*>(
#if defined(__linux__)
      checked_allocate(result.size() ? result.size() : 1,
                       alignof(std::max_align_t))
#else
      std::malloc(result.size() ? result.size() : 1)
#endif
  );
  if (!memory) return -1;
  if (!result.empty()) std::memcpy(memory, result.data(), result.size());
  *output = memory;
  *output_size = result.size();
  return 0;
}
}  // namespace

#if defined(_WIN32)
#define NG_CMIX_EXPORT __declspec(dllexport)
#else
#define NG_CMIX_EXPORT __attribute__((visibility("default")))
#endif

extern "C" NG_CMIX_EXPORT int native_graph_cmix_encode(
    const std::uint8_t* input, std::size_t input_size,
    std::uint8_t** output, std::size_t* output_size,
    char* error, std::size_t error_capacity) {
  try {
    if (!input && input_size) throw std::runtime_error("null cmix input");
    return copy_result(encode({input, input_size}), output, output_size);
  } catch (const std::exception& exception) {
    set_error(error, error_capacity, exception.what());
    return -1;
  }
}

extern "C" NG_CMIX_EXPORT int native_graph_cmix_decode(
    const std::uint8_t* input, std::size_t input_size, std::size_t raw_size,
    std::uint8_t** output, std::size_t* output_size,
    char* error, std::size_t error_capacity) {
  try {
    if (!input && input_size) throw std::runtime_error("null cmix body");
    return copy_result(decode({input, input_size}, raw_size), output, output_size);
  } catch (const std::exception& exception) {
    set_error(error, error_capacity, exception.what());
    return -1;
  }
}

extern "C" NG_CMIX_EXPORT void native_graph_cmix_free(void* memory) {
#if defined(__linux__)
  cmix_isolated_arena::deallocate(memory);
#else
  std::free(memory);
#endif
}

extern "C" NG_CMIX_EXPORT void native_graph_cmix_arena_snapshot(
    void** base, std::size_t* reserved_bytes, std::size_t* used_bytes) {
  if (!base || !reserved_bytes || !used_bytes) return;
#if defined(__linux__)
  *base = cmix_isolated_arena::state.base;
  *reserved_bytes = cmix_isolated_arena::state.base
                        ? cmix_isolated_arena::kCapacity
                        : 0;
  *used_bytes = cmix_isolated_arena::state.peak_live_bytes;
#else
  *base = nullptr;
  *reserved_bytes = 0;
  *used_bytes = 0;
#endif
}
