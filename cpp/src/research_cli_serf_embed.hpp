#pragma once

#include <fstream>
#include <vector>

/** When built with -DBENCHMARK_CPP_EMBED_SERF=ON: Serf XOR codec in-process (same as sfz CLI). */
bool bench_serf_inproc(const std::vector<std::vector<float>> &column_float_data, std::ofstream &out);
