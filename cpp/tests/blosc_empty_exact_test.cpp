#ifndef BLOSC_TEST_VARIANT
#error "compile with -DBLOSC_TEST_VARIANT=1 or -DBLOSC_TEST_VARIANT=2"
#endif

#if BLOSC_TEST_VARIANT == 1
#include "blosc_codec.hpp"
#include <blosc.h>
#elif BLOSC_TEST_VARIANT == 2
#include "blosc2_codec.hpp"
#include <blosc2.h>
#else
#error "BLOSC_TEST_VARIANT must be 1 or 2"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_throws(Fn&& fn, const std::string& message) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void store_le32(std::vector<std::uint8_t>& bytes, std::size_t offset,
                std::uint32_t value) {
    expect(offset + sizeof(value) <= bytes.size(), "test header write is in bounds");
    bytes[offset] = static_cast<std::uint8_t>(value);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
    bytes[offset + 2] = static_cast<std::uint8_t>(value >> 16);
    bytes[offset + 3] = static_cast<std::uint8_t>(value >> 24);
}

std::vector<std::uint8_t> sample_bytes() {
    std::vector<std::uint8_t> bytes(257);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((i * 37U + i / 3U) & 0xffU);
    }
    return bytes;
}

#if BLOSC_TEST_VARIANT == 1

std::vector<std::uint8_t> native_encode(std::span<const std::uint8_t> input,
                                        int typesize) {
    std::uint8_t empty_storage = 0;
    const void* source = input.empty() ? static_cast<const void*>(&empty_storage)
                                       : static_cast<const void*>(input.data());
    std::vector<std::uint8_t> encoded(input.size() + BLOSC_MAX_OVERHEAD);
    blosc_init();
    blosc_set_nthreads(1);
    expect(blosc_set_compressor(BLOSC_BLOSCLZ_COMPNAME) >= 0,
           "native Blosc1 BLOSCLZ setup");
    const int size = blosc_compress(5, BLOSC_SHUFFLE, static_cast<std::size_t>(typesize),
                                    input.size(), source, encoded.data(), encoded.size());
    blosc_destroy();
    expect(size > 0, "native Blosc1 compression");
    encoded.resize(static_cast<std::size_t>(size));
    return encoded;
}

std::vector<std::uint8_t> exact_encode(std::span<const std::uint8_t> input,
                                       int typesize) {
    return wc_blosc1_bench::encodeBufferExact(input, typesize);
}

std::vector<std::uint8_t> exact_decode(std::span<const std::uint8_t> encoded,
                                       int) {
    return wc_blosc1_bench::decodeBufferExact(encoded);
}

std::vector<std::uint8_t> native_decode(std::span<const std::uint8_t> encoded,
                                        std::size_t decoded_size) {
    std::uint8_t empty_storage = 0;
    std::vector<std::uint8_t> decoded(decoded_size);
    void* destination = decoded.empty() ? static_cast<void*>(&empty_storage)
                                        : static_cast<void*>(decoded.data());
    blosc_init();
    blosc_set_nthreads(1);
    const int size = blosc_decompress(encoded.data(), destination, decoded.size());
    blosc_destroy();
    expect(size == static_cast<int>(decoded_size), "native Blosc1 decompression");
    return decoded;
}

constexpr std::size_t kNativeEmptySize = BLOSC_MAX_OVERHEAD;

#elif BLOSC_TEST_VARIANT == 2

std::vector<std::uint8_t> native_encode(std::span<const std::uint8_t> input,
                                        int typesize) {
    blosc2_cparams params = BLOSC2_CPARAMS_DEFAULTS;
    params.compcode = BLOSC_BLOSCLZ;
    params.clevel = 5;
    params.typesize = static_cast<std::uint8_t>(typesize);
    params.nthreads = 1;
    params.filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE;
    blosc2_context* context = blosc2_create_cctx(params);
    expect(context != nullptr, "native Blosc2 compression context");
    std::uint8_t empty_storage = 0;
    const void* source = input.empty() ? static_cast<const void*>(&empty_storage)
                                       : static_cast<const void*>(input.data());
    std::vector<std::uint8_t> encoded(input.size() + BLOSC2_MAX_OVERHEAD);
    const int size = blosc2_compress_ctx(
        context, source, static_cast<std::int32_t>(input.size()), encoded.data(),
        static_cast<std::int32_t>(encoded.size()));
    blosc2_free_ctx(context);
    expect(size > 0, "native Blosc2 compression");
    encoded.resize(static_cast<std::size_t>(size));
    return encoded;
}

std::vector<std::uint8_t> exact_encode(std::span<const std::uint8_t> input,
                                       int typesize) {
    return wc_blosc2_bench::encodeBufferExact(input, typesize);
}

std::vector<std::uint8_t> exact_decode(std::span<const std::uint8_t> encoded,
                                       int typesize) {
    return wc_blosc2_bench::decodeBufferExact(encoded, typesize);
}

std::vector<std::uint8_t> native_decode(std::span<const std::uint8_t> encoded,
                                        std::size_t decoded_size) {
    blosc2_dparams params = BLOSC2_DPARAMS_DEFAULTS;
    params.nthreads = 1;
    blosc2_context* context = blosc2_create_dctx(params);
    expect(context != nullptr, "native Blosc2 decompression context");
    std::uint8_t empty_storage = 0;
    std::vector<std::uint8_t> decoded(decoded_size);
    void* destination = decoded.empty() ? static_cast<void*>(&empty_storage)
                                        : static_cast<void*>(decoded.data());
    const int size = blosc2_decompress_ctx(
        context, encoded.data(), static_cast<std::int32_t>(encoded.size()), destination,
        static_cast<std::int32_t>(decoded.size()));
    blosc2_free_ctx(context);
    expect(size == static_cast<int>(decoded_size), "native Blosc2 decompression");
    return decoded;
}

constexpr std::size_t kNativeEmptySize = BLOSC2_MAX_OVERHEAD;

#endif

void test_empty_native_parity() {
    const std::span<const std::uint8_t> empty;
    for (const int typesize : {1, 4, 8}) {
        const auto exact = exact_encode(empty, typesize);
        const auto native = native_encode(empty, typesize);
        expect(exact == native, "empty exact wire must equal native wire");
        expect(exact.size() == kNativeEmptySize, "empty wire has native header size");
        expect(exact_decode(native, typesize).empty(), "exact decoder accepts native empty wire");
        expect(native_decode(exact, 0).empty(), "native decoder accepts exact empty wire");
#if BLOSC_TEST_VARIANT == 1
        std::size_t native_nbytes = 1;
        expect(blosc_cbuffer_validate(exact.data(), exact.size(), &native_nbytes) == 0,
               "native Blosc1 validator accepts exact empty wire");
        expect(native_nbytes == 0, "native Blosc1 validator reports zero bytes");
#else
        std::int32_t native_nbytes = -1;
        std::int32_t native_cbytes = -1;
        std::int32_t native_blocksize = -1;
        expect(blosc2_cbuffer_sizes(exact.data(), &native_nbytes, &native_cbytes,
                                    &native_blocksize) == 0,
               "native Blosc2 parser accepts exact empty wire");
        expect(native_nbytes == 0 && native_cbytes == static_cast<std::int32_t>(exact.size()) &&
                   native_blocksize == 1,
               "native Blosc2 empty sizes");
#endif
    }
}

void test_nonempty_roundtrip_and_parity() {
    const auto input = sample_bytes();
    const auto exact = exact_encode(input, 1);
    const auto native = native_encode(input, 1);
    expect(exact == native, "nonempty exact wire remains native wire");
    expect(exact_decode(exact, 1) == input, "nonempty exact roundtrip");
    expect(native_decode(exact, input.size()) == input,
           "native decoder accepts nonempty exact wire");
}

void test_invalid_empty_wires() {
    const std::span<const std::uint8_t> empty;
    const auto canonical = native_encode(empty, 4);

    auto bad_header = canonical;
    bad_header[0] = 0;
    expect_throws([&] { (void)exact_decode(bad_header, 4); },
                  "bad empty version must be rejected");

    auto pseudo_empty = canonical;
#if BLOSC_TEST_VARIANT == 1
    store_le32(pseudo_empty, 8, 0);
#else
    pseudo_empty[BLOSC2_CHUNK_FILTER_CODES] = BLOSC_SHUFFLE;
#endif
    expect_throws([&] { (void)exact_decode(pseudo_empty, 4); },
                  "pseudo-empty native header must be rejected");

    auto trailing = canonical;
    trailing.push_back(0xa5);
    expect_throws([&] { (void)exact_decode(trailing, 4); },
                  "trailing bytes must be rejected");

    auto claimed_trailing = trailing;
    store_le32(claimed_trailing, 12, static_cast<std::uint32_t>(claimed_trailing.size()));
    expect_throws([&] { (void)exact_decode(claimed_trailing, 4); },
                  "claimed trailing bytes must not form an empty exact wire");
}

} // namespace

int main() {
    try {
        test_empty_native_parity();
        test_nonempty_roundtrip_and_parity();
        test_invalid_empty_wires();
        std::cout << "blosc_empty_exact_test variant " << BLOSC_TEST_VARIANT << ": ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "blosc_empty_exact_test variant " << BLOSC_TEST_VARIANT
                  << ": FAIL: " << e.what() << '\n';
        return 1;
    }
}
