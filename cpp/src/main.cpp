#include <brotli/decode.h>
#include <brotli/encode.h>
#include <lz4.h>
#include <lzma.h>
#include <snappy.h>
#include <zlib.h>
#include <zstd.h>

#include <CLI/CLI.hpp>
#include <csv2/csv2.hpp>

#include "alp_float_codec.hpp"
#include "benchmark_columns.hpp"
#include "camel_float_codec.hpp"
#include "elf_float_codec.hpp"
#include "simpiece_float_codec.hpp"
#include "column_encodings.hpp"
#include "csv_fractional_decimal_places.hpp"
#include "experiment_data_policy.hpp"
#include "leco_float_codec.hpp"
#include "java_fastpfor_codec.hpp"
#include "corad_tricklet_float_codec.hpp"
#include "machete_float_codec.hpp"
#include "most_float_codec.hpp"
#include "native_streams.hpp"
#include "native_lz_family.hpp"
#include "ans_rans.hpp"
#include "huffman_bytes.hpp"
#include "arithmetic_bytes.hpp"
#include "simple8b_int_codec.hpp"
#include "subcolumn_codec.hpp"
#include "spdp_codec.hpp"
#include "fpc_codec.hpp"
#include "pfpc_codec.hpp"
#include "zfp_codec.hpp"
#include "streamvbyte_int_codec.hpp"
#include "research_cli.hpp"
#include "run_java_companion.hpp"
#include "ts2diff_bos_native.hpp"
#include "elf_double_codec.hpp"
#include "elf_double_java.hpp"
#include "tsfile_bench_tsfile.hpp"
#include "process_csv_patch_bench.hpp"
#include "algorithm_allow_list.hpp"
#include "benchmark_loss.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

void testBrotliCompression(const std::vector<char> &data,
                           std::vector<long long> &result, LossAccum *loss) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> compressedData;

    size_t compressedSize = BrotliEncoderMaxCompressedSize(data.size());
    compressedData.resize(compressedSize);

    BrotliEncoderCompress(
        BROTLI_MAX_QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE,
        data.size(), reinterpret_cast<const uint8_t *>(data.data()),
        &compressedSize, reinterpret_cast<uint8_t *>(compressedData.data()));

    compressedData.resize(compressedSize);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> compressionTime =
        endTime - startTime;

    startTime = std::chrono::high_resolution_clock::now();

    size_t decompressedSize = data.size();
    std::vector<char> decompressedData(decompressedSize);
    if (BrotliDecoderDecompress(
            compressedData.size(),
            reinterpret_cast<const uint8_t *>(compressedData.data()),
            &decompressedSize,
            reinterpret_cast<uint8_t *>(decompressedData.data())) !=
        BROTLI_DECODER_RESULT_SUCCESS) {

        throw std::runtime_error("Failed to decompress data with Brotli.");
    }

    endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> decompressionTime =
        endTime - startTime;

    result[0] = data.size();
    result[1] = compressedSize;
    result[2] = static_cast<long long>(compressionTime.count());
    result[3] = static_cast<long long>(decompressionTime.count());
    if (loss != nullptr) {
        std::vector<char> got(decompressedData.begin(),
                              decompressedData.begin() + static_cast<std::ptrdiff_t>(decompressedSize));
        benchmark_loss::accumulate_bytes(data, got, *loss);
    }
}

void testLZ4Compression(const std::vector<char> &data,
                        std::vector<long long> &result, LossAccum *loss) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> compressedData;

    size_t compressedSize = LZ4_compressBound(data.size());
    compressedData.resize(compressedSize);

    compressedSize = LZ4_compress_default(data.data(), compressedData.data(),
                                          data.size(), compressedSize);

    compressedData.resize(compressedSize);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> compressionTime =
        endTime - startTime;

    startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> decompressedData(data.size());
    int decompressedSize =
        LZ4_decompress_safe(compressedData.data(), decompressedData.data(),
                            compressedData.size(), decompressedData.size());

    endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> decompressionTime =
        endTime - startTime;

    if (decompressedSize < 0) {
        throw std::runtime_error("Failed to decompress data with LZ4.");
    }

    result[0] = data.size();
    result[1] = compressedSize;
    result[2] = static_cast<long long>(compressionTime.count());
    result[3] = static_cast<long long>(decompressionTime.count());
    if (loss != nullptr) {
        std::vector<char> got(decompressedData.begin(),
                              decompressedData.begin() + decompressedSize);
        benchmark_loss::accumulate_bytes(data, got, *loss);
    }
}

void testGzipCompression(const std::vector<char> &data,
                         std::vector<long long> &result, LossAccum *loss) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> compressedData;
    uLongf compressedSize = compressBound(data.size());
    compressedData.resize(compressedSize);

    int res = compress(
        reinterpret_cast<Bytef *>(compressedData.data()), &compressedSize,
        reinterpret_cast<const Bytef *>(data.data()), data.size());

    if (res != Z_OK) {
        throw std::runtime_error("Failed to compress data with GZIP.");
    }

    compressedData.resize(compressedSize);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> compressionTime =
        endTime - startTime;

    startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> decompressedData(data.size());
    uLongf decompressedSize = data.size();
    res = uncompress(reinterpret_cast<Bytef *>(decompressedData.data()),
                     &decompressedSize,
                     reinterpret_cast<const Bytef *>(compressedData.data()),
                     compressedData.size());

    endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> decompressionTime =
        endTime - startTime;

    if (res != Z_OK) {
        throw std::runtime_error("Failed to decompress data with GZIP.");
    }

    result[0] = data.size();
    result[1] = compressedSize;
    result[2] = static_cast<long long>(compressionTime.count());
    result[3] = static_cast<long long>(decompressionTime.count());
    if (loss != nullptr) {
        std::vector<char> got(decompressedData.begin(),
                              decompressedData.begin() + static_cast<std::ptrdiff_t>(decompressedSize));
        benchmark_loss::accumulate_bytes(data, got, *loss);
    }
}

void testSnappyCompression(const std::vector<char> &data,
                           std::vector<long long> &result, LossAccum *loss) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> compressedData;
    size_t compressedSize = snappy::MaxCompressedLength(data.size());
    compressedData.resize(compressedSize);

    snappy::RawCompress(data.data(), data.size(), compressedData.data(),
                        &compressedSize);

    compressedData.resize(compressedSize);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> compressionTime =
        endTime - startTime;

    startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> decompressedData(data.size());
    if (!snappy::RawUncompress(compressedData.data(), compressedSize,
                               decompressedData.data())) {
        throw std::runtime_error("Failed to decompress data with Snappy.");
    }

    endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> decompressionTime =
        endTime - startTime;

    result[0] = data.size();
    result[1] = compressedSize;
    result[2] = static_cast<long long>(compressionTime.count());
    result[3] = static_cast<long long>(decompressionTime.count());
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decompressedData, *loss);
    }
}

void testZstdCompression(const std::vector<char> &data, int zstdLevel,
                         std::vector<long long> &result, LossAccum *loss) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> compressedData;
    size_t compressedBufferSize = ZSTD_compressBound(data.size());
    compressedData.resize(compressedBufferSize);

    size_t compressedSize =
        ZSTD_compress(compressedData.data(), compressedBufferSize, data.data(),
                      data.size(), zstdLevel);
    if (ZSTD_isError(compressedSize)) {
        throw std::runtime_error("Failed to compress data with ZSTD.");
    }

    compressedData.resize(compressedSize);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> compressionTime =
        endTime - startTime;

    startTime = std::chrono::high_resolution_clock::now();

    size_t decompressedSize =
        ZSTD_getFrameContentSize(compressedData.data(), compressedData.size());
    if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN ||
        decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
        throw std::runtime_error("Failed to get decompressed size with ZSTD.");
    }

    std::vector<char> decompressedData(decompressedSize);
    size_t res = ZSTD_decompress(decompressedData.data(), decompressedSize,
                                 compressedData.data(), compressedData.size());

    if (ZSTD_isError(res)) {
        throw std::runtime_error("Failed to decompress data with ZSTD.");
    }

    endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> decompressionTime =
        endTime - startTime;

    result[0] = data.size();
    result[1] = compressedSize;
    result[2] = static_cast<long long>(compressionTime.count());
    result[3] = static_cast<long long>(decompressionTime.count());
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decompressedData, *loss);
    }
}

void testLZMACompression(const std::vector<char> &data, int lzmaLevel,
                         std::vector<long long> &result, LossAccum *loss) {

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> compressedData;

    lzma_stream strm = LZMA_STREAM_INIT;
    // lzma2
    lzma_ret ret = lzma_easy_encoder(&strm, lzmaLevel, LZMA_CHECK_CRC64);

    if (ret != LZMA_OK) {
        throw std::runtime_error("Failed to initialize LZMA encoder.");
    }

    strm.next_in = reinterpret_cast<const uint8_t *>(data.data());
    strm.avail_in = data.size();

    const size_t buffer_size = 65536;
    std::vector<uint8_t> out_buffer(buffer_size);

    lzma_action action = LZMA_RUN;

    do {
        strm.next_out = out_buffer.data();
        strm.avail_out = buffer_size;

        if (strm.avail_in == 0) {
            action = LZMA_FINISH;
        }

        ret = lzma_code(&strm, action);

        if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
            std::cout << "Compression failed: " << ret << std::endl;
            lzma_end(&strm);
            return;
        }

        size_t compressed_size = buffer_size - strm.avail_out;
        compressedData.insert(compressedData.end(), out_buffer.begin(),
                              out_buffer.begin() + compressed_size);

    } while (ret != LZMA_STREAM_END);

    lzma_end(&strm);

    size_t compressedSize = compressedData.size();

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> compressionTime =
        endTime - startTime;

    startTime = std::chrono::high_resolution_clock::now();

    std::vector<char> decompressedData;

    lzma_stream strm2 = LZMA_STREAM_INIT;
    ret = lzma_stream_decoder(&strm2, UINT64_MAX, LZMA_CONCATENATED);

    if (ret != LZMA_OK) {
        std::cout << "Failed to initialize LZMA decoder: " << ret << std::endl;
        return;
    }

    strm2.next_in = compressedData.data();
    strm2.avail_in = compressedData.size();

    while (strm2.avail_in > 0) {
        strm2.next_out = out_buffer.data();
        strm2.avail_out = buffer_size;

        // 始终使用 LZMA_RUN，因为解码器会自动检测流的末尾
        ret = lzma_code(&strm2, LZMA_RUN);

        if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
            std::cout << "Decompression failed: " << ret << std::endl;
            lzma_end(&strm2);
            return;
        }

        size_t decompressed_size = buffer_size - strm2.avail_out;
        decompressedData.insert(
            decompressedData.end(), reinterpret_cast<char *>(out_buffer.data()),
            reinterpret_cast<char *>(out_buffer.data()) + decompressed_size);

        // 如果已经到达流的末尾，即使还有输入（可能是填充字节），也应提前退出
        if (ret == LZMA_STREAM_END) {
            break;
        }
    }

    lzma_end(&strm2);

    endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> decompressionTime =
        endTime - startTime;

    result[0] = data.size();
    result[1] = compressedSize;
    result[2] = static_cast<long long>(compressionTime.count());
    result[3] = static_cast<long long>(decompressionTime.count());
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, decompressedData, *loss);
    }
}

void processCsv(const std::string &csvPath, const std::string &dataType,
                int lzmaLevel, int zstdLevel,
                const std::string &outputCsvPath) {
    std::vector<std::vector<std::string>> columns;
    std::vector<std::string> headers;

    try {
        csv2::Reader<csv2::delimiter<','>, csv2::quote_character<'"'>,
                     csv2::first_row_is_header<true>>
            csv;

        if (!csv.mmap(csvPath)) {
            std::cerr << "CSV file is empty" << std::endl;
            return;
        }

        // 获取表头
        const auto header = csv.header();
        for (const auto &col_name : header) {
            std::string name;
            col_name.read_value(name);
            headers.push_back(name);
        }

        int colCount = headers.size();
        columns.resize(colCount);

        // 读取所有数据行
        for (const auto row : csv) {
            int colIndex = 0;

            for (const auto &cell : row) {
                std::string value;
                cell.read_value(value);
                columns[colIndex].push_back(value);
                colIndex++;
            }

            while (colIndex < colCount) {
                columns[colIndex].push_back("");
                colIndex++;
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Failed to read CSV: " << e.what() << std::endl;
        return;
    }

    std::vector<int> validColumnIndexes;
    std::vector<std::string> validColumnHeaders;

    std::vector<std::vector<char>> validColumnByteData;
    std::vector<std::vector<int>> validColumnIntData;
    std::vector<std::vector<float>> validColumnFloatData;

    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        const auto &col = columns[i];
        bool valid = true;

        std::vector<char> binaryData;
        std::vector<int> intData;
        std::vector<float> floatData;

        if (dataType == "int") {
            for (const auto &v : col) {
                if (v.empty() || experiment_data_policy::is_nan_cell(v))
                    continue;
                try {
                    // int intVal = std::stoi(v);
                    int intVal;
                    auto result =
                        std::from_chars(v.data(), v.data() + v.size(), intVal);

                    if (result.ec != std::errc{} ||
                        result.ptr != v.data() + v.size()) {
                        valid = false;
                        break;
                    }

                    const char *bytes = reinterpret_cast<const char *>(&intVal);
                    binaryData.insert(binaryData.end(), bytes,
                                      bytes + sizeof(int));
                    intData.push_back(intVal);

                } catch (const std::exception &) {
                    valid = false;
                    break;
                }
            }
        } else if (dataType == "float") {
            for (const auto &v : col) {
                if (v.empty())
                    continue;
                try {
                    // float floatVal = std::stof(v);
                    float floatVal;
                    auto result = std::from_chars(v.data(), v.data() + v.size(),
                                                  floatVal);
                    if (result.ec != std::errc{} ||
                        result.ptr != v.data() + v.size()) {
                        valid = false;
                        break;
                    }

                    const char *bytes =
                        reinterpret_cast<const char *>(&floatVal);
                    if (!experiment_data_policy::keep_numeric_value(floatVal))
                        continue;
                    binaryData.insert(binaryData.end(), bytes,
                                      bytes + sizeof(float));
                    floatData.push_back(floatVal);

                } catch (const std::exception &) {
                    valid = false;
                    break;
                }
            }
        }

        // 检查数据是否有效且不为空
        if (valid && !binaryData.empty()) {
            validColumnIndexes.push_back(i);
            validColumnHeaders.push_back(headers[i]);
            validColumnByteData.push_back(std::move(binaryData));

            if (dataType == "int") {
                validColumnIntData.push_back(std::move(intData));
            } else if (dataType == "float") {
                validColumnFloatData.push_back(std::move(floatData));
            }

            std::cout << "Column " << i << " (" << headers[i]
                      << ") preprocessing complete, data is valid" << std::endl;
        }
    }

    std::ofstream outputFile(outputCsvPath);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to create output file: " << outputCsvPath
                  << std::endl;
        return;
    }

    outputFile << "algorithm,originalSize,compressedSize,compressionTime,"
                  "decompressionTime\n";

    // gzip
    std::cout << "Processing GZIP compression" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testGzipCompression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "GZIP compression failed: " << e.what()
                          << std::endl;
                continue;
            }
        }

        outputFile << "gzip," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }

    // snappy
    std::cout << "Processing Snappy compression" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testSnappyCompression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "Snappy compression failed: " << e.what()
                          << std::endl;
                continue;
            }
        }

        outputFile << "snappy," << totalOriginal << "," << totalCompressed
                   << "," << totalCompTime << "," << totalDecompTime << "\n";
    }

    // LZ4
    std::cout << "Processing LZ4 compression" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testLZ4Compression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "LZ4 compression failed: " << e.what()
                          << std::endl;
                continue;
            }
        }

        outputFile << "lz4," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }

    // zstd
    std::cout << "Processing Zstd compression" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testZstdCompression(data, zstdLevel, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "Zstd compression failed: " << e.what()
                          << std::endl;
                continue;
            }
        }

        outputFile << "zstd," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }

    // XZ/LZMA2
    std::cout << "Processing XZ compression" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testLZMACompression(data, lzmaLevel, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "XZ compression failed: " << e.what()
                          << std::endl;
                continue;
            }
        }

        outputFile << "xz," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }

    // brotli
    std::cout << "Processing Brotli compression" << std::endl;
    {
        long totalOriginal = 0;
        long totalCompressed = 0;
        long totalCompTime = 0;
        long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testBrotliCompression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "Brotli compression failed: " << e.what()
                          << std::endl;
                continue;
            }
        }

        outputFile << "brotli," << totalOriginal << "," << totalCompressed
                   << "," << totalCompTime << "," << totalDecompTime << "\n";
    }

    // ans (order-0 rANS entropy coding)
    std::cout << "Processing ANS (rANS) entropy coding" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testAnsEntropyCompression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "ANS compression failed: " << e.what() << std::endl;
                continue;
            }
        }

        outputFile << "ans," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }

    // huffman (byte Huffman)
    std::cout << "Processing Huffman (byte) coding" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testHuffmanCompression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "Huffman compression failed: " << e.what() << std::endl;
                continue;
            }
        }

        outputFile << "huffman," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }

    // arithmetic (static order-0 range / carry-free)
    std::cout << "Processing Arithmetic (byte) coding" << std::endl;
    {
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;

        for (size_t idx = 0; idx < validColumnIndexes.size(); ++idx) {
            int originalColumnIndex = validColumnIndexes[idx];
            const std::string &columnHeader = validColumnHeaders[idx];
            const std::vector<char> &data = validColumnByteData[idx];

            std::cout << "processing " << originalColumnIndex << " ("
                      << columnHeader << ")" << std::endl;

            std::vector<long long> result(4);
            try {
                testArithmeticCompression(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << "Arithmetic compression failed: " << e.what() << std::endl;
                continue;
            }
        }

        outputFile << "arithmetic," << totalOriginal << "," << totalCompressed << ","
                   << totalCompTime << "," << totalDecompTime << "\n";
    }


}

void processCsvSimple(const std::string &csvPath, const std::string &dataType,
                      const std::string &outputCsvPath) {
    std::vector<std::vector<std::string>> columns;
    std::vector<std::string> headers;

    try {
        csv2::Reader<csv2::delimiter<','>, csv2::quote_character<'"'>,
                     csv2::first_row_is_header<true>>
            csv;

        if (!csv.mmap(csvPath)) {
            std::cerr << "CSV file is empty" << std::endl;
            return;
        }

        // 获取表头
        const auto header = csv.header();
        for (const auto &col_name : header) {
            std::string name;
            col_name.read_value(name);
            headers.push_back(name);
        }

        int colCount = headers.size();
        columns.resize(colCount);

        // 读取所有数据行
        for (const auto row : csv) {
            int colIndex = 0;

            for (const auto &cell : row) {
                std::string value;
                cell.read_value(value);
                columns[colIndex].push_back(value);
                colIndex++;
            }

            while (colIndex < colCount) {
                columns[colIndex].push_back("");
                colIndex++;
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Failed to read CSV: " << e.what() << std::endl;
        return;
    }

    std::vector<std::vector<char>> columnByteData;
    std::vector<std::vector<std::int64_t>> columnIntData;
    std::vector<std::vector<double>> columnFloatData;
    std::vector<int> columnFloatMaxPointPerColumn;

    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        const auto &col = columns[i];

        std::vector<char> binaryData;
        std::vector<std::int64_t> intData;
        std::vector<double> floatData;
        int floatMaxPointForCol = 0;

        if (dataType == "int") {
            for (const auto &v : col) {
                if (v.empty())
                    continue;

                std::int64_t intVal;
                auto result = std::from_chars(v.data(), v.data() + v.size(), intVal);
                if (result.ec == std::errc{} && result.ptr == v.data() + v.size()) {
                    const char *bytes = reinterpret_cast<const char *>(&intVal);
                    binaryData.insert(binaryData.end(), bytes, bytes + sizeof(intVal));
                    intData.push_back(intVal);
                }
            }
        } else if (dataType == "float") {
            for (const auto &v : col) {
                if (v.empty())
                    continue;

                double floatVal;
                auto result = std::from_chars(v.data(), v.data() + v.size(), floatVal);
                if (result.ec == std::errc{} && result.ptr == v.data() + v.size() &&
                    experiment_data_policy::keep_numeric_value(floatVal)) {
                    const char *bytes = reinterpret_cast<const char *>(&floatVal);
                    binaryData.insert(binaryData.end(), bytes, bytes + sizeof(floatVal));
                    floatData.push_back(floatVal);
                    floatMaxPointForCol =
                        std::max(floatMaxPointForCol, csv_fractional_decimal_places(v));
                }
            }
        }

        if (!binaryData.empty()) {
            columnByteData.push_back(std::move(binaryData));
            if (dataType == "int") {
                columnIntData.push_back(std::move(intData));
            } else if (dataType == "float") {
                columnFloatData.push_back(std::move(floatData));
                columnFloatMaxPointPerColumn.push_back(floatMaxPointForCol);
            }
            std::cout << "Column " << i << " (" << headers[i] << ") preprocessing complete" << std::endl;
        }
    }

    if (const char *regen = std::getenv("COMPRESSION_BENCHMARK_STATICDATA_REGEN");
        regen != nullptr && regen[0] == '1' && std::filesystem::is_regular_file(outputCsvPath)) {
        std::error_code ec;
        std::filesystem::remove(outputCsvPath, ec);
        if (ec) {
            std::cerr << "Warning: could not delete " << outputCsvPath << " for regen\n";
        }
    }

    std::unordered_set<std::string> existingAlgorithms;
    {
        std::ifstream in(outputCsvPath);
        std::string line;
        bool first = true;
        while (std::getline(in, line)) {
            if (first) {
                first = false;
                continue;
            }
            if (line.empty()) {
                continue;
            }
            std::size_t comma = line.find(',');
            if (comma == std::string::npos || comma == 0) {
                continue;
            }
            existingAlgorithms.insert(line.substr(0, comma));
        }
    }

    const bool appendMode = std::filesystem::is_regular_file(outputCsvPath);
    std::ofstream outputFile(outputCsvPath, appendMode ? std::ios::app : std::ios::out);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to create output file: " << outputCsvPath << std::endl;
        return;
    }

    if (!appendMode) {
        outputFile << "algorithm,originalSize,compressedSize,compressionTime,decompressionTime\n";
    }

    std::string repoRoot = ".";
    if (const char *rr = std::getenv("COMPRESSION_BENCHMARK_REPO_ROOT")) {
        repoRoot = rr;
    }
    const std::string csvAbs = std::filesystem::absolute(csvPath).string();
    const std::string outAbs = std::filesystem::absolute(outputCsvPath).string();
    auto should_run = [&](const std::string &algo) {
        return existingAlgorithms.find(algo) == existingAlgorithms.end() &&
               algorithm_allow_list::should_run(nullptr, algo);
    };

    auto bench_stream_algo = [&]<typename Fn>(const char *algoName, Fn fn) {
        if (!should_run(algoName)) {
            std::cout << "Skip existing " << algoName << '\n';
            return;
        }
        long long totalOriginal = 0;
        long long totalCompressed = 0;
        long long totalCompTime = 0;
        long long totalDecompTime = 0;
        for (size_t i = 0; i < columnByteData.size(); ++i) {
            const std::vector<char> &data = columnByteData[i];
            std::vector<long long> result(4);
            try {
                fn(data, result, nullptr);
                totalOriginal += result[0];
                totalCompressed += result[1];
                totalCompTime += result[2];
                totalDecompTime += result[3];
            } catch (const std::exception &e) {
                std::cerr << algoName << " failed: " << e.what() << std::endl;
            }
        }
        outputFile << algoName << ',' << totalOriginal << ',' << totalCompressed << ','
                   << totalCompTime << ',' << totalDecompTime << '\n';
        existingAlgorithms.insert(algoName);
    };

    const std::vector<std::string> nativeReplacementRows = {
        "blosc_blosclz",
        "blosc2_blosclz",
        "fpzip",
        "fpcompress_dpratio",
        "fpcompress_dpspeed",
        "elf_on_chimp",
        "elf_plus",
        "camel",
    };
    for (const auto &algoName : nativeReplacementRows) {
        if (!should_run(algoName)) {
            continue;
        }
        std::vector<long long> result(4);
        bench_native_lz_family_bytes(columnByteData, algoName, result, nullptr);
        if (result[0] > 0) {
            outputFile << algoName << ',' << result[0] << ',' << result[1] << ','
                       << result[2] << ',' << result[3] << '\n';
            existingAlgorithms.insert(algoName);
        }
    }

    std::cout << "Processing gz (zlib gzip)" << std::endl;
    bench_stream_algo("gz", &benchGzFramed);
    std::cout << "Processing snappy-framed" << std::endl;
    bench_stream_algo("snappy-framed", &benchSnappyFramed);
    std::cout << "Processing lz4-framed" << std::endl;
    bench_stream_algo("lz4-framed", &benchLz4Framed);

    outputFile.flush();
    outputFile.close();

    {
        std::ofstream out(outAbs, std::ios::app);
        if (!out.is_open()) {
            std::cerr << "Failed to append: " << outAbs << std::endl;
            return;
        }

        if (dataType == "int" && !columnIntData.empty()) {
            std::cout << "TSFile encodings (int): TS_2DIFF RLE GORILLA CHIMP SPRINTZ RLBE" << std::endl;
            bench_tsfile_rows_int64(columnIntData, out, &existingAlgorithms);
        } else if (dataType == "float" && !columnFloatData.empty()) {
            std::cout << "TSFile encodings (float): TS_2DIFF RLE GORILLA CHIMP SPRINTZ RLBE" << std::endl;
            bench_tsfile_rows_double(columnFloatData, columnFloatMaxPointPerColumn, out,
                                     &existingAlgorithms);
        }
        out.flush();
    }

    {
        std::cout << "Column encodings + research (native / JVM subprocess as implemented)" << std::endl;
        std::ofstream out(outAbs, std::ios::app);
        if (!out.is_open()) {
            std::cerr << "Failed to append: " << outAbs << std::endl;
            return;
        }

        std::vector<std::vector<double>> codecCols;
        if (dataType == "int" && !columnIntData.empty()) {
            codecCols = integers_as_double_columns(columnIntData);
            std::vector<long long> cr(4);
            auto reset_cr = [&cr]() { cr.assign({0, 0, 0, 0}); };
            if (colenc::shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                reset_cr();
                colenc::benchDeltaZigZagVarintInt64(columnIntData, cr);
            }
            if (cr[0] > 0 && colenc::shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                out << colenc::kDeltaBinaryPackedAlgorithm << "," << cr[0] << "," << cr[1] << "," << cr[2] << ","
                    << cr[3] << "\n";
                existingAlgorithms.insert(colenc::kDeltaBinaryPackedAlgorithm);
            }
            if (should_run("col_dictionary")) {
                reset_cr();
                colenc::benchDictionaryInt64(columnIntData, cr);
            }
            if (cr[0] > 0 && should_run("col_dictionary")) {
                out << "col_dictionary," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("col_dictionary");
            }
            if (should_run("streamvbyte_delta1")) {
                reset_cr();
                streamvbyte_int::benchInt64ColumnsStreamVByteDelta1(columnIntData, cr);
            }
            if (cr[0] > 0 && should_run("streamvbyte_delta1")) {
                out << "streamvbyte_delta1," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("streamvbyte_delta1");
            }
            const std::vector<std::vector<double>> intColsAsDouble =
                integers_as_double_columns(columnIntData);
            if (should_run("col_bss_lz4") && !intColsAsDouble.empty()) {
                reset_cr();
                colenc::benchBssLz4Double64(intColsAsDouble, cr);
            }
            if (cr[0] > 0 && should_run("col_bss_lz4")) {
                out << "col_bss_lz4," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("col_bss_lz4");
            }
            if (should_run("ts_2diff_bos")) {
                std::vector<long long> crb(4);
                ts2diff_bos_native::benchInt64ColumnsBos(columnIntData, crb);
                if (crb[0] > 0) {
                    out << "ts_2diff_bos," << crb[0] << "," << crb[1] << "," << crb[2] << "," << crb[3]
                        << "\n";
                    existingAlgorithms.insert("ts_2diff_bos");
                }
            }
        } else if (dataType == "float" && !columnFloatData.empty()) {
            codecCols = columnFloatData;
            std::vector<long long> cr(4);
            auto reset_cr = [&cr]() { cr.assign({0, 0, 0, 0}); };
            if (colenc::shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                reset_cr();
                colenc::benchDbpFloat64(columnFloatData, cr, &columnFloatMaxPointPerColumn);
                if (cr[0] > 0) {
                    out << colenc::kDeltaBinaryPackedAlgorithm << "," << cr[0] << "," << cr[1]
                        << "," << cr[2] << "," << cr[3] << "\n";
                    existingAlgorithms.insert(colenc::kDeltaBinaryPackedAlgorithm);
                }
            }
            const std::vector<std::vector<std::int64_t>> scaledIntFromFloat =
                scale_double_columns_to_int64(columnFloatData, &columnFloatMaxPointPerColumn);
            if (!scaledIntFromFloat.empty()) {
                if (should_run("streamvbyte_delta1")) {
                    reset_cr();
                    streamvbyte_int::benchInt64ColumnsStreamVByteDelta1(scaledIntFromFloat, cr);
                }
                if (cr[0] > 0 && should_run("streamvbyte_delta1")) {
                    out << "streamvbyte_delta1," << cr[0] << "," << cr[1] << ","
                        << cr[2] << "," << cr[3] << "\n";
                    existingAlgorithms.insert("streamvbyte_delta1");
                }
            }
            if (should_run("col_dictionary") && !columnFloatData.empty()) {
                reset_cr();
                colenc::benchDictionaryDouble64(columnFloatData, cr);
            }
            if (cr[0] > 0 && should_run("col_dictionary")) {
                out << "col_dictionary," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("col_dictionary");
            }
            if (should_run("col_bss_lz4") && !columnFloatData.empty()) {
                reset_cr();
                colenc::benchBssLz4Double64(columnFloatData, cr);
            }
            if (cr[0] > 0 && should_run("col_bss_lz4")) {
                out << "col_bss_lz4," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("col_bss_lz4");
            }
            if (should_run("ts_2diff_bos")) {
                std::vector<long long> crb(4);
                ts2diff_bos_native::benchDoubleColumnsBos(codecCols, &columnFloatMaxPointPerColumn, crb);
                if (crb[0] > 0) {
                    out << "ts_2diff_bos," << crb[0] << "," << crb[1] << "," << crb[2] << "," << crb[3]
                        << "\n";
                    existingAlgorithms.insert("ts_2diff_bos");
                }
            }
        }

        if (!codecCols.empty()) {
            const std::vector<std::vector<float>> codecColsFloat =
                (dataType == "int") ? integers_as_float_columns(columnIntData)
                                    : double_columns_as_float(codecCols);
            std::vector<long long> cr(4);
            for (unsigned spdpLevel : spdp_codec::levelsToBench()) {
                const std::string spdpRow = spdp_codec::rowName(spdpLevel);
                if (!should_run(spdpRow)) {
                    continue;
                }
                cr.assign(4, 0);
                if (dataType == "float") {
                    spdp_codec::benchDoubleColumnsSpdp(codecCols, spdpLevel, cr);
                } else {
                    spdp_codec::benchInt64ColumnsSpdp(columnIntData, spdpLevel, cr);
                }
                if (cr[0] > 0) {
                    out << spdpRow << "," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                        << "\n";
                    existingAlgorithms.insert(spdpRow);
                }
            }
            if (should_run(fpc_codec::kRowName)) {
                cr.assign(4, 0);
                if (dataType == "float") {
                    fpc_codec::benchDoubleColumnsFpc(codecCols, cr);
                } else if (dataType == "int") {
                    fpc_codec::benchInt64ColumnsFpc(columnIntData, cr);
                }
                if (cr[0] > 0) {
                    out << fpc_codec::kRowName << "," << cr[0] << "," << cr[1] << "," << cr[2] << ","
                        << cr[3] << "\n";
                    existingAlgorithms.insert(fpc_codec::kRowName);
                }
            }
            if (should_run(pfpc_codec::kRowName)) {
                cr.assign(4, 0);
                if (dataType == "float") {
                    pfpc_codec::benchDoubleColumnsPfpc(codecCols, cr);
                } else if (dataType == "int") {
                    pfpc_codec::benchInt64ColumnsPfpc(columnIntData, cr);
                }
                if (cr[0] > 0) {
                    out << pfpc_codec::kRowName << "," << cr[0] << "," << cr[1] << "," << cr[2] << ","
                        << cr[3] << "\n";
                    existingAlgorithms.insert(pfpc_codec::kRowName);
                }
            }
            if (zfp_codec::bench_enabled()) {
                if (should_run(zfp_codec::kRowReversible)) {
                    cr.assign(4, 0);
                    zfp_codec::benchReversibleDoubleColumns(codecCols, cr);
                    if (cr[0] > 0) {
                        out << zfp_codec::kRowReversible << "," << cr[0] << "," << cr[1] << ","
                            << cr[2] << "," << cr[3] << "\n";
                        existingAlgorithms.insert(zfp_codec::kRowReversible);
                    }
                }
                static const struct {
                    double tol;
                    const char *row;
                } kZfpTol[] = {
                    {1e-1, "ext_zfp_tol_1e-1"},
                    {1e-2, "ext_zfp_tol_1e-2"},
                    {1e-3, "ext_zfp_tol_1e-3"},
                };
                for (const auto &zp : kZfpTol) {
                    if (!should_run(zp.row)) {
                        continue;
                    }
                    cr.assign(4, 0);
                    zfp_codec::benchToleranceDoubleColumns(codecCols, zp.tol, zp.row, cr);
                    if (cr[0] > 0) {
                        out << zp.row << "," << cr[0] << "," << cr[1] << "," << cr[2] << ","
                            << cr[3] << "\n";
                        existingAlgorithms.insert(zp.row);
                    }
                }
            }
            if (should_run("most_minimal")) {
                if (dataType == "float") {
                    most_float::benchDoubleColumnsMostMinimal(codecCols, cr);
                } else {
                    most_float::benchInt64ColumnsMostMinimal(columnIntData, cr);
                }
            }
            if (cr[0] > 0 && should_run("most_minimal")) {
                out << "most_minimal," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("most_minimal");
            }
            static const struct {
                double rel;
                const char *rowName;
            } kMachetePresets[] = {
                {1e-1, "machete_rel_1e-1"},
                {1e-2, "machete_rel_1e-2"},
                {1e-3, "machete_rel_1e-3"},
                {1e-4, "machete_rel_1e-4"},
            };
            for (const auto &mp : kMachetePresets) {
                if (should_run(mp.rowName)) {
                    if (dataType == "float") {
                        machete_float::benchDoubleColumnsMacheteRel(codecCols, mp.rel, cr);
                    } else {
                        machete_float::benchInt64ColumnsMacheteRel(columnIntData, mp.rel, cr);
                    }
                }
                if (cr[0] > 0 && should_run(mp.rowName)) {
                    out << mp.rowName << ',' << cr[0] << ',' << cr[1] << ',' << cr[2] << ',' << cr[3]
                        << "\n";
                    existingAlgorithms.insert(mp.rowName);
                }
            }
            if (should_run("corad_tricklet")) {
                if (dataType == "float") {
                    corad_tricklet_float::benchDoubleColumnsCoradTricklet(codecCols, cr);
                } else {
                    corad_tricklet_float::benchInt64ColumnsCoradTricklet(columnIntData, cr);
                }
            }
            if (cr[0] > 0 && should_run("corad_tricklet")) {
                out << "corad_tricklet," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("corad_tricklet");
            }
            if (should_run("fastpfor")) {
                if (dataType == "float") {
                    java_fastpfor::benchDoubleColumnsLikeJava(codecCols, cr);
                } else {
                    java_fastpfor::benchInt64ColumnsLikeJava(columnIntData, cr);
                }
            }
            if (cr[0] > 0 && should_run("fastpfor")) {
                out << "fastpfor," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("fastpfor");
            }
            if (should_run("subcolumn")) {
                cr.assign({0, 0, 0, 0});
                if (dataType == "float") {
                    subcolumn_codec::benchDoubleColumnsSubcolumn(codecCols, cr, &columnFloatMaxPointPerColumn);
                } else {
                    subcolumn_codec::benchInt64ColumnsSubcolumn(columnIntData, cr);
                }
            }
            if (cr[0] > 0 && should_run("subcolumn")) {
                out << "subcolumn," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("subcolumn");
            }
            if (should_run("simple8b")) {
                if (dataType == "float") {
                    simple8b_int::benchDoubleColumnsSimple8b(codecCols, cr, &columnFloatMaxPointPerColumn);
                } else {
                    simple8b_int::benchInt64ColumnsSimple8b(columnIntData, cr);
                }
            }
            if (cr[0] > 0 && should_run("simple8b")) {
                out << "simple8b," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("simple8b");
            }
            if (should_run("leco")) {
                leco_double::benchDoubleColumnsLikeJava(codecCols, cr);
            }
            if (cr[0] > 0 && should_run("leco")) {
                out << "leco," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("leco");
            }
            if (should_run("alp")) {
                if (dataType == "float") {
                    alp_float::benchDoubleColumnsLikeJava(codecCols, cr);
                } else {
                    alp_float::benchInt64ColumnsLikeJava(columnIntData, cr);
                }
            }
            if (cr[0] > 0 && should_run("alp")) {
                out << "alp," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("alp");
            }
            if (should_run("CHIMP")) {
                elf_double::benchChimpDoubleColumnsLikeJava(codecCols, cr);
            }
            if (cr[0] > 0 && should_run("CHIMP")) {
                out << "CHIMP," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("CHIMP");
            }
            if (should_run("elf_on_chimp")) {
                elf_double::benchElfOnChimpDoubleColumnsLikeJava(codecCols, cr);
            }
            if (cr[0] > 0 && should_run("elf_on_chimp")) {
                out << "elf_on_chimp," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3]
                    << "\n";
                existingAlgorithms.insert("elf_on_chimp");
            }
            if (should_run("elf_plus")) {
                elf_double::benchElfPlusDoubleColumnsLikeJava(codecCols, cr);
            }
            if (cr[0] > 0 && should_run("elf_plus")) {
                out << "elf_plus," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("elf_plus");
            }
            if (should_run("camel")) {
                camel_float::benchDoubleColumnsLikeJava(codecCols, cr);
            }
            if (cr[0] > 0 && should_run("camel")) {
                out << "camel," << cr[0] << "," << cr[1] << "," << cr[2] << "," << cr[3] << "\n";
                existingAlgorithms.insert("camel");
            }
            if (should_run("simpiece_relRange_0p005")) {
                if (dataType == "float") {
                    simpiece_float::benchDoubleColumnsRelRange(codecCols, 0.005, cr);
                } else {
                    simpiece_float::benchInt64ColumnsRelRange(columnIntData, 0.005, cr);
                }
            }
            if (cr[0] > 0 && should_run("simpiece_relRange_0p005")) {
                out << "simpiece_relRange_0p005," << cr[0] << "," << cr[1] << "," << cr[2] << ","
                    << cr[3] << "\n";
                existingAlgorithms.insert("simpiece_relRange_0p005");
            }
            if (should_run("simpiece_relRange_0p01")) {
                if (dataType == "float") {
                    simpiece_float::benchDoubleColumnsRelRange(codecCols, 0.01, cr);
                } else {
                    simpiece_float::benchInt64ColumnsRelRange(columnIntData, 0.01, cr);
                }
            }
            if (cr[0] > 0 && should_run("simpiece_relRange_0p01")) {
                out << "simpiece_relRange_0p01," << cr[0] << "," << cr[1] << "," << cr[2] << ","
                    << cr[3] << "\n";
                existingAlgorithms.insert("simpiece_relRange_0p01");
            }
        }
        out.flush();
    }

    std::cout << "Java companion (CppBenchmarkCompanion external-only)..." << std::endl;
    if (!run_java_companion_phase(repoRoot, csvAbs, dataType, outAbs, "external-only")) {
        std::cerr << "Warning: Java companion (external-only) failed; optional research JAR rows missing.\n";
    }

    {
        std::ofstream out(outAbs, std::ios::app);
        if (!out.is_open()) {
            std::cerr << "Failed to append: " << outAbs << std::endl;
            return;
        }

        std::vector<std::vector<double>> lossyCodecCols;
        if (dataType == "int" && !columnIntData.empty()) {
            lossyCodecCols = integers_as_double_columns(columnIntData);
        } else if (dataType == "float" && !columnFloatData.empty()) {
            lossyCodecCols = columnFloatData;
        }
        if (!lossyCodecCols.empty()) {
            const char *skipSzSerf = std::getenv("COMPRESSION_BENCHMARK_CPP_SKIP_SZ_SERF");
            if (skipSzSerf != nullptr && skipSzSerf[0] == '1') {
                std::cout << "Skipping SZ/SZ3/Serf (COMPRESSION_BENCHMARK_CPP_SKIP_SZ_SERF=1)" << std::endl;
            } else {
                std::cout << "SZ / SZ3 / Serf / BUFF (int columns as float when applicable)..."
                          << std::endl;
                bench_sz_sz3_serf_double(lossyCodecCols, repoRoot, out, &existingAlgorithms);
            }
        }

        for (int level = 0; level <= 9; ++level) {
            const std::string rowName = "xz_level_" + std::to_string(level);
            if (!should_run(rowName)) {
                continue;
            }
            long long totalOriginal = 0;
            long long totalCompressed = 0;
            long long totalCompTime = 0;
            long long totalDecompTime = 0;
            for (size_t i = 0; i < columnByteData.size(); ++i) {
                const std::vector<char> &data = columnByteData[i];
                std::vector<long long> result(4);
                try {
                    testLZMACompression(data, level, result, nullptr);
                    totalOriginal += result[0];
                    totalCompressed += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                } catch (const std::exception &e) {
                    std::cerr << "XZ failed: " << e.what() << std::endl;
                }
            }
            out << "xz_level_" << level << ',' << totalOriginal << ',' << totalCompressed << ','
                << totalCompTime << ',' << totalDecompTime << '\n';
            existingAlgorithms.insert(rowName);
        }

        if (should_run("brotli")) {
            long long totalOriginal = 0;
            long long totalCompressed = 0;
            long long totalCompTime = 0;
            long long totalDecompTime = 0;
            for (size_t i = 0; i < columnByteData.size(); ++i) {
                std::vector<long long> result(4);
                try {
                    testBrotliCompression(columnByteData[i], result, nullptr);
                    totalOriginal += result[0];
                    totalCompressed += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                } catch (const std::exception &e) {
                    std::cerr << "Brotli failed: " << e.what() << std::endl;
                }
            }
            out << "brotli," << totalOriginal << ',' << totalCompressed << ',' << totalCompTime
                << ',' << totalDecompTime << '\n';
            existingAlgorithms.insert("brotli");
        }

        for (int level = 1; level <= 9; ++level) {
            const std::string rowName = "zstd_level_" + std::to_string(level);
            if (!should_run(rowName)) {
                continue;
            }
            long long totalOriginal = 0;
            long long totalCompressed = 0;
            long long totalCompTime = 0;
            long long totalDecompTime = 0;
            for (size_t i = 0; i < columnByteData.size(); ++i) {
                std::vector<long long> result(4);
                try {
                    testZstdCompression(columnByteData[i], level, result, nullptr);
                    totalOriginal += result[0];
                    totalCompressed += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                } catch (const std::exception &e) {
                    std::cerr << "Zstd failed: " << e.what() << std::endl;
                }
            }
            out << "zstd_level_" << level << ',' << totalOriginal << ',' << totalCompressed << ','
                << totalCompTime << ',' << totalDecompTime << '\n';
            existingAlgorithms.insert(rowName);
        }

        if (should_run("ans")) {
            long long totalOriginal = 0;
            long long totalCompressed = 0;
            long long totalCompTime = 0;
            long long totalDecompTime = 0;
            for (size_t i = 0; i < columnByteData.size(); ++i) {
                std::vector<long long> result(4);
                try {
                    testAnsEntropyCompression(columnByteData[i], result, nullptr);
                    totalOriginal += result[0];
                    totalCompressed += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                } catch (const std::exception &e) {
                    std::cerr << "ANS failed: " << e.what() << std::endl;
                }
            }
            out << "ans," << totalOriginal << ',' << totalCompressed << ',' << totalCompTime << ','
                << totalDecompTime << '\n';
            existingAlgorithms.insert("ans");
        }

        if (should_run("huffman")) {
            long long totalOriginal = 0;
            long long totalCompressed = 0;
            long long totalCompTime = 0;
            long long totalDecompTime = 0;
            for (size_t i = 0; i < columnByteData.size(); ++i) {
                std::vector<long long> result(4);
                try {
                    testHuffmanCompression(columnByteData[i], result, nullptr);
                    totalOriginal += result[0];
                    totalCompressed += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                } catch (const std::exception &e) {
                    std::cerr << "Huffman failed: " << e.what() << std::endl;
                }
            }
            out << "huffman," << totalOriginal << ',' << totalCompressed << ',' << totalCompTime << ','
                << totalDecompTime << '\n';
            existingAlgorithms.insert("huffman");
        }

        if (should_run("arithmetic")) {
            long long totalOriginal = 0;
            long long totalCompressed = 0;
            long long totalCompTime = 0;
            long long totalDecompTime = 0;
            for (size_t i = 0; i < columnByteData.size(); ++i) {
                std::vector<long long> result(4);
                try {
                    testArithmeticCompression(columnByteData[i], result, nullptr);
                    totalOriginal += result[0];
                    totalCompressed += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                } catch (const std::exception &e) {
                    std::cerr << "Arithmetic failed: " << e.what() << std::endl;
                }
            }
            out << "arithmetic," << totalOriginal << ',' << totalCompressed << ',' << totalCompTime << ','
                << totalDecompTime << '\n';
            existingAlgorithms.insert("arithmetic");
        }
    }

    std::cout << "Compression results written to " << outputCsvPath << std::endl;
}

int main(int argc, char **argv) {
    CLI::App app{"Benchmark CPP"};

    std::string inputPath;
    std::string dataType;
    std::string outputCsvPath;
    std::string columnInfoPath;
    int columnIndex = -1;
    int lzmaLevel = 0;
    int zstdLevel = 3;
    std::string algorithmsCsv;

    app.add_option("-i,--input", inputPath, "Input file path")->required();

    app.add_option("--data-type", dataType, "Data type")->required();

    app.add_option("--output-report", outputCsvPath, "Output report file path")
        ->required();

    app.add_option("--column-info", columnInfoPath,
                   "Catalog column_info CSV (max_point_number for float scaling)")
        ->required();

    app.add_option("--column-index", columnIndex,
                   "Original dataset column index for single-column mode")
        ->default_val(-1);

    app.add_option("--xz-level,--lzma-level", lzmaLevel, "XZ compression level")
        ->default_val(0);

    app.add_option("--zstd-level", zstdLevel, "ZSTD compression level")
        ->default_val(3);

    app.add_option(
        "--algorithms",
        algorithmsCsv,
        "Comma-separated algorithm row names for partial refresh (sets COMPRESSION_BENCHMARK_RUN_ALGORITHMS)");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    std::filesystem::path outputPath(outputCsvPath);
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    if (!algorithmsCsv.empty()) {
        algorithm_allow_list::set_allowlist_env(algorithmsCsv);
    }

    // processCsv(inputPath, dataType, lzmaLevel, zstdLevel, outputCsvPath);

    processCsvSimplePatch(inputPath, dataType, outputCsvPath, columnInfoPath, columnIndex);

    return 0;
}
