#pragma once

#include "benchmark_loss.hpp"

#include <string>
#include <vector>

const std::vector<std::string>& native_lz_family_algorithms();

std::vector<char> compress_native_lz_family(const std::vector<char>& data,
                                            const std::string& algorithm);

std::vector<char> compress_native_lz_family_ablation(
    const std::vector<char>& data, const std::string& algorithm,
    const std::string& omitted_operator);

std::vector<char> decompress_native_lz_family(const std::vector<char>& data,
                                              const std::string& algorithm,
                                              std::size_t expected_size);

std::vector<char> decompress_native_lz_family_ablation(
    const std::vector<char>& data, const std::string& algorithm,
    std::size_t expected_size, const std::string& omitted_operator);

void bench_native_lz_family_bytes(const std::vector<std::vector<char>>& columns,
                                  const std::string& algorithm,
                                  std::vector<long long>& result,
                                  LossAccum* loss = nullptr);
