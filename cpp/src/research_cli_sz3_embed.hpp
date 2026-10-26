#pragma once

#include <fstream>
#include <string>
#include <vector>

/** When built with -DBENCHMARK_CPP_EMBED_SZ3=ON: SZ3 in-process using the same sz.config + error bounds as the sz3 CLI. */
bool bench_sz3_family_inproc(const std::vector<std::vector<float>> &column_float_data,
                             const std::string &sz_config_path,
                             std::ofstream &out);
bool bench_sz3_family_inproc_double(const std::vector<std::vector<double>> &column_double_data,
                                    const std::string &sz_config_path,
                                    std::ofstream &out);
