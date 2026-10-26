#include "elf_double_codec.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) try {
    if (argc != 2) throw std::invalid_argument("expected an Elf64 stage command");
    const std::string command = argv[1];
    const std::vector<std::uint8_t> data{std::istreambuf_iterator<char>(std::cin), {}};
    std::vector<std::uint8_t> out;
    using namespace elf_double;
    if (command == "eraser-forward") out = eraser64Forward(data);
    else if (command == "eraser-inverse") out = eraser64Inverse(data);
    else if (command == "elf-xor-forward") out = xor64Forward(data, false);
    else if (command == "elf-xor-inverse") out = xor64Inverse(data, false);
    else if (command == "elf-plus-xor-forward") out = xor64Forward(data, true);
    else if (command == "elf-plus-xor-inverse") out = xor64Inverse(data, true);
    else if (command == "elf-original-encode") out = encodeDoubleColumnExact(data, false);
    else if (command == "elf-plus-original-encode") out = encodeDoubleColumnExact(data, true);
    else if (command == "elf-original-decode") out = decodeDoubleColumnExact(data, false);
    else if (command == "elf-plus-original-decode") out = decodeDoubleColumnExact(data, true);
    else throw std::invalid_argument("unknown Elf64 stage command");
    std::cout.write(reinterpret_cast<const char *>(out.data()), out.size());
    if (!std::cout) throw std::runtime_error("Elf64 stdout write failed");
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
}
