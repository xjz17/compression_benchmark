#pragma once

#include "benchmark_loss.hpp"

#include <string_view>
#include <vector>

void benchGzFramed(const std::vector<char> &data, std::vector<long long> &result,
                   LossAccum *loss = nullptr);
void benchGzRawDeflate(const std::vector<char> &data, std::vector<long long> &result,
                       LossAccum *loss = nullptr);
void benchZlibWrapped(const std::vector<char> &data, std::vector<long long> &result,
                      LossAccum *loss = nullptr);
void benchZlibRawDeflate(const std::vector<char> &data, std::vector<long long> &result,
                         LossAccum *loss = nullptr);
void benchDeflateOperatorAblation(const std::vector<char> &data, std::string_view method,
                                  std::string_view omitted_operator,
                                  std::vector<long long> &result,
                                  LossAccum *loss = nullptr);
void benchLz4Framed(const std::vector<char> &data, std::vector<long long> &result,
                    LossAccum *loss = nullptr);
void benchLz4RawBlock(const std::vector<char> &data, std::vector<long long> &result,
                      LossAccum *loss = nullptr);
void benchLz4FastFramed(const std::vector<char> &data, std::vector<long long> &result,
                        LossAccum *loss = nullptr);
void benchLz4FastRawBlock(const std::vector<char> &data, std::vector<long long> &result,
                          LossAccum *loss = nullptr);
void benchLz4HcFramed(const std::vector<char> &data, std::vector<long long> &result,
                      LossAccum *loss = nullptr);
void benchLz4HcRawBlock(const std::vector<char> &data, std::vector<long long> &result,
                        LossAccum *loss = nullptr);
void benchLz4OperatorAblation(const std::vector<char> &data, std::string_view method,
                              std::string_view omitted_operator,
                              std::vector<long long> &result,
                              LossAccum *loss = nullptr);
void benchSnappyFramed(const std::vector<char> &data, std::vector<long long> &result,
                       LossAccum *loss = nullptr);
void benchSnappyFramedLiteralOnly(const std::vector<char> &data,
                                  std::vector<long long> &result,
                                  LossAccum *loss = nullptr);
void benchZstdOperatorAblation(const std::vector<char> &data,
                               std::string_view omitted_operator,
                               std::vector<long long> &result,
                               LossAccum *loss = nullptr);
