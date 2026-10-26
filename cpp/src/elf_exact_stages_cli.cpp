#include "elf_float_codec.hpp"

#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_all() {
    return {std::istreambuf_iterator<char>(std::cin),
            std::istreambuf_iterator<char>()};
}
std::vector<std::int64_t> words(const std::vector<std::uint8_t> &raw) {
    if (raw.size() % 8) throw std::runtime_error("float64 input byte count mismatch");
    std::vector<std::int64_t> values(raw.size() / 8);
    if (!raw.empty()) std::memcpy(values.data(), raw.data(), raw.size());
    return values;
}
template <class T>
void write(const std::vector<T> &values) {
    std::cout.write(reinterpret_cast<const char *>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}  // namespace

int main(int argc, char **argv) try {
    if (argc != 2) throw std::invalid_argument("missing Elf stage command");
    const std::string command = argv[1];
    const auto raw = read_all();
    using namespace exact_actions::elf;
    if (command == "eraser-forward") write(eraser_forward_double_bits(words(raw)));
    else if (command == "eraser-inverse") write(eraser_inverse_double_bits(raw));
    else if (command == "elf-xor-forward") write(elf_xor_forward(raw));
    else if (command == "elf-xor-inverse") write(elf_xor_inverse(raw));
    else if (command == "elf-plus-xor-forward") write(elf_plus_xor_forward(raw));
    else if (command == "elf-plus-xor-inverse") write(elf_plus_xor_inverse(raw));
    else if (command == "elf-compose") write(compose_elf(words(raw)));
    else if (command == "elf-decompose") write(decompose_elf(raw));
    else if (command == "elf-plus-compose") write(compose_elf_plus(words(raw)));
    else if (command == "elf-plus-decompose") write(decompose_elf_plus(raw));
    else throw std::invalid_argument("unknown command: " + command);
    if (!std::cout) throw std::runtime_error("failed to write stdout");
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
}
