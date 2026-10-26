#include "camel_float_codec.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (char c : line) {
        if (c == '"') {
            quoted = !quoted;
        } else if (c == ',' && !quoted) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

struct Dataset {
    std::vector<std::vector<double>> columns;
    std::size_t rows = 0;
    std::size_t integer_columns = 0;
    std::size_t float_columns = 0;
    std::size_t timestamp_columns = 0;
};

Dataset load_numeric_columns(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open dataset");
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty dataset");
    const auto headers = split(line);
    Dataset dataset;
    std::vector<std::vector<std::string>> values(headers.size());
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line);
        if (fields.size() != headers.size()) throw std::runtime_error("CSV width mismatch");
        ++dataset.rows;
        for (std::size_t i = 0; i < fields.size(); ++i) values[i].push_back(fields[i]);
    }

    for (std::size_t column_index = 0; column_index < values.size(); ++column_index) {
        const auto& column = values[column_index];
        std::vector<double> numbers;
        numbers.reserve(column.size());
        for (const auto& text : column) numbers.push_back(std::stod(text));
        const std::string& name = headers[column_index];
        if (name == "timestamp" || name == "time") {
            ++dataset.timestamp_columns;
            continue;
        }
        const bool integral = std::all_of(column.begin(), column.end(), [](const std::string& text) {
            std::int64_t value = 0;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
        });
        if (integral) {
            ++dataset.integer_columns;
            continue;
        }
        dataset.columns.push_back(std::move(numbers));
        ++dataset.float_columns;
    }
    return dataset;
}

struct Result {
    const char* omitted;
    long long original = 0;
    long long compressed = 0;
    long long compression_ns = 0;
    long long decompression_ns = 0;
};

Result run(const std::vector<std::vector<double>>& columns,
           camel_float::AblationMode mode, const char* omitted) {
    camel_float::setAblationMode(mode);
    Result result{omitted};
    std::vector<long long> values;
    camel_float::benchDoubleColumnsLikeJava(columns, values);
    if (values.size() != 4 || values[0] <= 0 || values[1] <= 0 || values[2] <= 0 || values[3] <= 0)
        throw std::runtime_error("Camel native ablation did not encode all numeric columns");
    result.original = values[0];
    result.compressed = values[1];
    result.compression_ns = values[2];
    result.decompression_ns = values[3];
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: camel_source_air_ablation DATASET.csv OUTPUT.csv\n";
        return 2;
    }
    try {
        const auto dataset = load_numeric_columns(argv[1]);
        const std::vector<Result> results = {
            run(dataset.columns, camel_float::AblationMode::none, "none"),
            run(dataset.columns, camel_float::AblationMode::scaling, "scaling"),
            run(dataset.columns, camel_float::AblationMode::xor_transform, "xor"),
            run(dataset.columns, camel_float::AblationMode::xor_bit_reduction, "trailing_zero_pack"),
        };
        const double baseline = static_cast<double>(results.front().compressed) /
                                results.front().original;
        std::ofstream output(argv[2]);
        if (!output) throw std::runtime_error("cannot write output");
        output << "dataset,method,omitted_operator,original_bytes,compressed_bytes,compression_rate,baseline_rate,rate_degradation,compression_ns,decompression_ns,lossless,integer_columns,float_columns,timestamp_columns,columns,rows,configuration,implementation\n";
        output << std::setprecision(9);
        std::string dataset_name = argv[1];
        const std::size_t separator = dataset_name.find_last_of("/\\");
        if (separator != std::string::npos) dataset_name.erase(0, separator + 1);
        const std::size_t extension = dataset_name.rfind('.');
        if (extension != std::string::npos) dataset_name.erase(extension);
        for (const auto& result : results) {
            const double rate = static_cast<double>(result.compressed) / result.original;
            output << dataset_name << ",camel," << result.omitted << ','
                   << result.original << ',' << result.compressed << ',' << rate << ','
                   << baseline << ',' << rate - baseline << ','
                   << result.compression_ns << ',' << result.decompression_ns << ",1,"
                   << dataset.integer_columns << ',' << dataset.float_columns << ','
                   << dataset.timestamp_columns << ',' << dataset.columns.size() << ','
                   << dataset.rows << ",float_extended,float_extended_native_source_ablation\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
