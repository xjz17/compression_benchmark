#include "elf_float_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::int64_t> bits(const std::vector<double> &values) {
    std::vector<std::int64_t> out(values.size());
    if (!values.empty()) std::memcpy(out.data(), values.data(), values.size() * 8);
    return out;
}

void put_u32(std::vector<std::uint8_t> &out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t> reference_wire(const std::vector<double> &values,
                                         bool plus) {
    std::vector<std::uint8_t> out{'E', 'L', 'F', '1'};
    put_u32(out, static_cast<std::uint32_t>(values.size()));
    for (std::size_t start = 0; start < values.size(); start += 2048) {
        const auto end = std::min(values.size(), start + 2048);
        std::vector<float> chunk;
        chunk.reserve(end - start);
        for (std::size_t i = start; i < end; ++i)
            chunk.push_back(static_cast<float>(values[i]));
        const auto body = plus
            ? elf_float::encodeElfPlusFloatColumn(chunk)
            : elf_float::encodeElfOnChimpFloatColumn(chunk);
        put_u32(out, static_cast<std::uint32_t>(chunk.size()));
        put_u32(out, static_cast<std::uint32_t>(body.size()));
        out.insert(out.end(), body.begin(), body.end());
    }
    return out;
}
}  // namespace

int main() {
    using namespace exact_actions::elf;
    std::vector<double> boundary(2050);
    for (std::size_t i = 0; i < boundary.size(); ++i)
        boundary[i] = static_cast<double>(static_cast<float>((i % 31) * 0.25f));
    for (const std::vector<double> values : {
             std::vector<double>{},
             std::vector<double>{-19.25, -1.0, -0.0, 0.0, 0.125, 3.5, 17.25},
             std::vector<double>{1.0, 1.0, 1.25, 10.5, 100.0, 100.5},
             std::vector<double>{static_cast<double>(1.0e10f),
                                 static_cast<double>(-1.0e12f),
                                 static_cast<double>(std::numeric_limits<float>::max())},
             boundary}) {
        const auto raw = bits(values);
        const auto erased = eraser_forward_double_bits(raw);
        if (eraser_inverse_double_bits(erased) != raw)
            throw std::runtime_error("Elf eraser round trip");
        const auto elf_wire = elf_xor_forward(erased);
        if (elf_wire != reference_wire(values, false))
            throw std::runtime_error("Elf public wire parity");
        if (elf_xor_inverse(elf_wire) != erased)
            throw std::runtime_error("Elf XOR round trip");
        if (decompose_elf(elf_wire) != raw)
            throw std::runtime_error("Elf chain round trip");
        const auto plus_wire = elf_plus_xor_forward(erased);
        if (plus_wire != reference_wire(values, true))
            throw std::runtime_error("Elf+ public wire parity");
        if (elf_plus_xor_inverse(plus_wire) != erased)
            throw std::runtime_error("Elf+ XOR round trip");
        if (decompose_elf_plus(plus_wire) != raw)
            throw std::runtime_error("Elf+ chain round trip");
    }
    for (const double invalid : {
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try {
            (void)eraser_forward_double_bits(bits({invalid}));
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("Elf non-finite value accepted");
    }
    std::cout << "elf_exact_stage_actions=3 status=PASS\n";
}
