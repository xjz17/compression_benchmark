#pragma once

#include "benchmark_loss.hpp"

#include <vector>

/** Byte-level compressors defined in main.cpp (linked into benchmark_cpp). */
void testBrotliCompression(const std::vector<char>& data, std::vector<long long>& result,
                           LossAccum* loss = nullptr);
void testLZ4Compression(const std::vector<char>& data, std::vector<long long>& result,
                        LossAccum* loss = nullptr);
void testGzipCompression(const std::vector<char>& data, std::vector<long long>& result,
                         LossAccum* loss = nullptr);
void testSnappyCompression(const std::vector<char>& data, std::vector<long long>& result,
                           LossAccum* loss = nullptr);
void testZstdCompression(const std::vector<char>& data, int zstdLevel, std::vector<long long>& result,
                         LossAccum* loss = nullptr);
void testLZMACompression(const std::vector<char>& data, int lzmaLevel, std::vector<long long>& result,
                         LossAccum* loss = nullptr);
