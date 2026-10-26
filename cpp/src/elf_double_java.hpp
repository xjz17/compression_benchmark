#pragma once

#include <cstdint>
#include <vector>

namespace elf_double_java {

void benchDoubleColumnsSubprocess(const std::vector<std::vector<double>> &columns,
                                const char *mode,
                                std::vector<long long> &out);

} // namespace elf_double_java
