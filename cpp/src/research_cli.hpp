#pragma once

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

void bench_sz_sz3_serf_double(const std::vector<std::vector<double>> &columnDoubleData,
                              const std::string &repoRoot, std::ofstream &out,
                              const std::unordered_set<std::string> *skip_algorithms = nullptr);
