#include "process_csv_patch_bench.hpp"

#include "wc_temp_path.hpp"

#include "algorithm_allow_list.hpp"
#include "alp_float_codec.hpp"
#include "ans_rans.hpp"
#include "arithmetic_bytes.hpp"
#include "benchmark_columns.hpp"
#include "benchmark_totals.hpp"
#include "column_encodings.hpp"
#include "column_info_loader.hpp"
#include "experiment_data_policy.hpp"
#include "compression_bench_api.hpp"
#include "corad_tricklet_float_codec.hpp"
#include "elf_double_codec.hpp"
#include "elf_float_codec.hpp"
#include "huffman_bytes.hpp"
#include "java_fastpfor_codec.hpp"
#include "leco_double_codec.hpp"
#include "machete_float_codec.hpp"
#include "most_float_codec.hpp"
#include "native_lz_family.hpp"
#include "native_streams.hpp"
#include "reger_codec.hpp"
#include "research_cli.hpp"
#include "run_java_companion.hpp"
#include "simple8b_int_codec.hpp"
#include "subcolumn_codec.hpp"
#include "simpiece_float_codec.hpp"
#include "spdp_codec.hpp"
#include "fpc_codec.hpp"
#include "grouping_codec.hpp"
#include "pfpc_codec.hpp"
#include "zfp_codec.hpp"
#include "streamvbyte_int_codec.hpp"
#include "ts2diff_bos_native.hpp"
#include "wc_temp_path.hpp"
#include "tsfile_bench_tsfile.hpp"
#include "buff_kernel/buff_roaring_lite.hpp"
#include "buff_kernel/buff_simd_ops.hpp"
#include "buff_kernel/buff_split_double_codec.hpp"

#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr const char* kRegerTimeColumn = "__reger_time__";

int patch_row_limit() {
  const char* raw = std::getenv("COMPRESSION_BENCHMARK_CSV_PATCH_ROWS");
  if (raw == nullptr || raw[0] == '\0') {
    return 10000;
  }
  try {
    const int n = std::stoi(raw);
    return n > 0 ? n : 10000;
  } catch (...) {
    return 10000;
  }
}

bool patch_bench_verbose() {
  const char* raw = std::getenv("COMPRESSION_BENCHMARK_PATCH_BENCH_VERBOSE");
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
         value == "YES" || value == "on" || value == "ON";
}

bool logical_type_matches_data_type(const std::string& logical_type, const std::string& data_type) {
  std::string logical = logical_type;
  for (char& ch : logical) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (data_type == "int") {
    return logical == "integer" || logical == "int" || logical == "long";
  }
  if (data_type == "float") {
    return logical == "float" || logical == "double";
  }
  if (data_type == "string") {
    return logical == "string";
  }
  return true;
}

std::string normalize_cell(std::string raw) {
  while (!raw.empty() && (raw.front() == ' ' || raw.front() == '\t')) {
    raw.erase(raw.begin());
  }
  while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\t' || raw.back() == '\r')) {
    raw.pop_back();
  }
  if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
    raw = raw.substr(1, raw.size() - 2);
  }
  return raw;
}

void split_csv_line(const std::string& line, std::vector<std::string>& out) {
  out.clear();
  std::size_t i = 0;
  while (i <= line.size()) {
    std::size_t j = line.find(',', i);
    if (j == std::string::npos) {
      out.push_back(normalize_cell(line.substr(i)));
      break;
    }
    out.push_back(normalize_cell(line.substr(i, j - i)));
    i = j + 1;
  }
}

struct PatchColumnData {
  std::vector<std::vector<char>> column_byte_data;
  std::vector<std::vector<std::int64_t>> column_int_data;
  std::vector<std::vector<double>> column_float_data;
  std::vector<int> column_float_max_point_per_column;
  std::vector<std::int64_t> row_time_data;
};

bool build_patch_column_data(const std::string& dataType, const std::vector<std::string>& headers,
                             const std::vector<std::vector<std::string>>& columns,
                             const std::unordered_map<std::string, int>& max_point_by_name,
                             PatchColumnData& out) {
  out = PatchColumnData{};
  for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
    if (i >= static_cast<int>(columns.size())) {
      break;
    }
    const auto& col = columns[static_cast<std::size_t>(i)];
    if (headers[static_cast<std::size_t>(i)] == kRegerTimeColumn) {
      out.row_time_data.clear();
      out.row_time_data.reserve(col.size());
      for (std::size_t row_idx = 0; row_idx < col.size(); ++row_idx) {
        const auto& v = col[row_idx];
        std::int64_t time_val = static_cast<std::int64_t>(row_idx);
        if (!v.empty()) {
          auto result = std::from_chars(v.data(), v.data() + v.size(), time_val);
          if (result.ec != std::errc{} || result.ptr != v.data() + v.size()) {
            time_val = static_cast<std::int64_t>(row_idx);
          }
        }
        out.row_time_data.push_back(time_val);
      }
      continue;
    }
    std::vector<char> binary_data;
    std::vector<std::int64_t> int_data;
    std::vector<double> float_data;
    int float_max_point = 0;
    auto it = max_point_by_name.find(headers[static_cast<std::size_t>(i)]);
    if (it != max_point_by_name.end()) {
      float_max_point = it->second;
    }

    if (dataType == "int") {
      for (const auto& v : col) {
        if (v.empty()) {
          continue;
        }
        std::int64_t int_val = 0;
        auto result = std::from_chars(v.data(), v.data() + v.size(), int_val);
        if (result.ec == std::errc{} && result.ptr == v.data() + v.size()) {
          const char* bytes = reinterpret_cast<const char*>(&int_val);
          binary_data.insert(binary_data.end(), bytes, bytes + sizeof(int_val));
          int_data.push_back(int_val);
        }
      }
    } else if (dataType == "float") {
      for (const auto& v : col) {
        if (v.empty()) {
          continue;
        }
        double float_val = 0;
        auto result = std::from_chars(v.data(), v.data() + v.size(), float_val);
        if (result.ec == std::errc{} && result.ptr == v.data() + v.size() &&
            experiment_data_policy::keep_numeric_value(float_val)) {
          const char* bytes = reinterpret_cast<const char*>(&float_val);
          binary_data.insert(binary_data.end(), bytes, bytes + sizeof(float_val));
          float_data.push_back(float_val);
        }
      }
    } else if (dataType == "string") {
      for (const auto& v : col) {
        binary_data.insert(binary_data.end(), v.begin(), v.end());
      }
    }

    if (!binary_data.empty()) {
      out.column_byte_data.push_back(std::move(binary_data));
      if (dataType == "int") {
        out.column_int_data.push_back(std::move(int_data));
      } else if (dataType == "float") {
        out.column_float_data.push_back(std::move(float_data));
        out.column_float_max_point_per_column.push_back(float_max_point);
      }
    }
  }
  return !out.column_byte_data.empty();
}

void accumulate_generic_byte_compression(
    const std::vector<std::vector<char>>& column_byte_data, BenchmarkTotals& totals,
    const std::unordered_set<std::string>* skip_algorithms) {
  for (const auto& algorithm : native_lz_family_algorithms()) {
    if (!algorithm_allow_list::should_run(skip_algorithms, algorithm)) {
      continue;
    }
    std::vector<long long> result(4);
    LossAccum loss;
    bench_native_lz_family_bytes(column_byte_data, algorithm, result, &loss);
    totals.merge(algorithm, result[0], result[1], result[2], result[3], loss);
  }
}

void accumulate_grouping_patch(
    const std::string& dataType, const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& patch_columns, std::uint64_t row_offset,
    const std::unordered_map<int, ColumnInfoMeta>& info_by_index, BenchmarkTotals& totals,
    const std::unordered_set<std::string>* skip_algorithms) {
  if (dataType != "int" && dataType != "float") {
    return;
  }
  if (!algorithm_allow_list::should_run(skip_algorithms, grouping_codec::kRowName)) {
    return;
  }
  std::vector<grouping_codec::ColumnStream> streams;
  for (std::size_t i = 0; i < headers.size() && i < patch_columns.size(); ++i) {
    if (headers[i] == kRegerTimeColumn) {
      continue;
    }
    auto meta = info_by_index.find(static_cast<int>(i));
    if (meta != info_by_index.end() &&
        !logical_type_matches_data_type(meta->second.logical_type, dataType)) {
      continue;
    }
    grouping_codec::ColumnStream stream =
        dataType == "int" ? grouping_codec::intColumnStream(patch_columns[i], row_offset)
                          : grouping_codec::doubleColumnStream(patch_columns[i], row_offset);
    if (!stream.values.empty()) {
      streams.push_back(std::move(stream));
    }
  }
  if (streams.empty()) {
    return;
  }
  std::vector<long long> result(4);
  grouping_codec::benchStreams(streams, result);
  totals.merge(grouping_codec::kRowName, result);
}

void bench_buff_kernel_native_columns(const std::vector<std::vector<double>>& cols,
                                      std::vector<long long>& result,
                                      LossAccum* loss) {
  result.assign(4, 0);
  constexpr int kScale = 10000;
  constexpr double kPredicate = 1.1509;
  for (const auto& col : cols) {
    if (col.empty()) {
      continue;
    }
    result[0] += static_cast<long long>(col.size() * sizeof(double));
    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::vector<std::uint8_t> packed = buff_kernel::encode_doubles(col, kScale);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto decoded = buff_kernel::decode_doubles(packed);
    const auto t2 = std::chrono::high_resolution_clock::now();
    if (loss != nullptr) {
      benchmark_loss::accumulate_double_column(col, decoded.values, *loss);
    }
    result[1] += static_cast<long long>(packed.size());
    result[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }
}

void accumulate_patch_columns(const std::string& dataType, const PatchColumnData& patch,
                              BenchmarkTotals& totals, const std::string& repo_root,
                              const std::unordered_set<std::string>* skip_algorithms) {
  const auto& column_byte_data = patch.column_byte_data;
  if (dataType == "string") {
    accumulate_generic_byte_compression(column_byte_data, totals, skip_algorithms);
    return;
  }

  const auto& column_int_data = patch.column_int_data;
  const auto& column_float_data = patch.column_float_data;
  const auto& column_float_max_point = patch.column_float_max_point_per_column;
  const std::vector<std::int64_t>* row_time_ptr =
      patch.row_time_data.empty() ? nullptr : &patch.row_time_data;

  if (dataType == "int" && !column_int_data.empty()) {
    bench_tsfile_rows_int64_accumulate(column_int_data, totals, skip_algorithms);
  } else if (dataType == "float" && !column_float_data.empty()) {
    bench_tsfile_rows_double_accumulate(column_float_data, column_float_max_point, totals,
                                        skip_algorithms);
  }

  std::vector<std::vector<double>> codec_cols;
  if (dataType == "int" && !column_int_data.empty()) {
    codec_cols = integers_as_double_columns(column_int_data);
    std::vector<long long> cr(4);
    LossAccum loss;
    if (algorithm_allow_list::should_run(skip_algorithms, colenc::kDeltaBinaryPackedAlgorithm)) {
      colenc::benchDeltaZigZagVarintInt64(column_int_data, cr, false, &loss);
      totals.merge(colenc::kDeltaBinaryPackedAlgorithm, cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "col_dictionary")) {
      colenc::benchDictionaryInt64(column_int_data, cr, true, &loss);
      totals.merge("col_dictionary", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "streamvbyte_delta1")) {
      streamvbyte_int::benchInt64ColumnsStreamVByteDelta1(column_int_data, cr, &loss);
      totals.merge("streamvbyte_delta1", cr, loss);
    }
    if (!codec_cols.empty() && algorithm_allow_list::should_run(skip_algorithms, "col_bss_lz4")) {
      colenc::benchBssLz4Double64(codec_cols, cr, &loss);
      totals.merge("col_bss_lz4", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "ts_2diff_bos")) {
      std::vector<long long> crb(4);
      ts2diff_bos_native::benchInt64ColumnsBos(column_int_data, crb);
      totals.merge("ts_2diff_bos", crb);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "subcolumn")) {
      subcolumn_codec::benchInt64ColumnsSubcolumn(column_int_data, cr, &loss);
      totals.merge("subcolumn", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "reger")) {
      LossAccum reger_loss;
      reger_codec::benchInt64ColumnsReger(column_int_data, cr, &reger_loss, row_time_ptr);
      totals.merge("reger", cr, reger_loss);
    }
  } else if (dataType == "float" && !column_float_data.empty()) {
    codec_cols = column_float_data;
    std::vector<long long> cr(4);
    LossAccum loss;
    if (algorithm_allow_list::should_run(skip_algorithms, colenc::kDeltaBinaryPackedAlgorithm)) {
      colenc::benchDbpFloat64(column_float_data, cr, &column_float_max_point, &loss);
      totals.merge(colenc::kDeltaBinaryPackedAlgorithm, cr, loss);
    }
    const std::vector<std::vector<std::int64_t>> scaled_int_from_float =
        scale_double_columns_to_int64(column_float_data, &column_float_max_point);
    if (!scaled_int_from_float.empty()) {
      if (algorithm_allow_list::should_run(skip_algorithms, "streamvbyte_delta1")) {
        streamvbyte_int::benchInt64ColumnsStreamVByteDelta1(scaled_int_from_float, cr, &loss);
        totals.merge("streamvbyte_delta1", cr, loss);
      }
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "reger")) {
      LossAccum reger_loss;
      reger_codec::benchDoubleColumnsReger(column_float_data, cr, &reger_loss, row_time_ptr);
      totals.merge("reger", cr, reger_loss);
    }
    if (!column_float_data.empty()) {
      if (algorithm_allow_list::should_run(skip_algorithms, "col_dictionary")) {
        colenc::benchDictionaryDouble64(column_float_data, cr, &loss);
        totals.merge("col_dictionary", cr, loss);
      }
      if (algorithm_allow_list::should_run(skip_algorithms, "col_bss_lz4")) {
        colenc::benchBssLz4Double64(column_float_data, cr, &loss);
        totals.merge("col_bss_lz4", cr, loss);
      }
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "ts_2diff_bos")) {
      std::vector<long long> crb(4);
      ts2diff_bos_native::benchDoubleColumnsBos(column_float_data, &column_float_max_point, crb);
      totals.merge("ts_2diff_bos", crb);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "subcolumn")) {
      subcolumn_codec::benchDoubleColumnsSubcolumn(column_float_data, cr, &column_float_max_point,
                                                   &loss);
      totals.merge("subcolumn", cr, loss);
    }
  }

  if (!codec_cols.empty()) {
    const std::vector<std::vector<float>> codec_cols_float =
        (dataType == "int") ? integers_as_float_columns(column_int_data)
                            : double_columns_as_float(codec_cols);
    std::vector<long long> cr(4);
    for (unsigned spdp_level : spdp_codec::levelsToBench()) {
      const std::string spdp_row = spdp_codec::rowName(spdp_level);
      if (!algorithm_allow_list::should_run(skip_algorithms, spdp_row)) {
        continue;
      }
      LossAccum loss;
      if (dataType == "float") {
        spdp_codec::benchDoubleColumnsSpdp(codec_cols, spdp_level, cr, &loss);
      } else {
        spdp_codec::benchInt64ColumnsSpdp(column_int_data, spdp_level, cr, &loss);
      }
      totals.merge(spdp_row, cr, loss);
    }
    if (dataType == "float" || dataType == "int") {
      LossAccum loss;
      if (algorithm_allow_list::should_run(skip_algorithms, fpc_codec::kRowName)) {
        if (dataType == "float") {
          fpc_codec::benchDoubleColumnsFpc(codec_cols, cr, &loss);
        } else {
          fpc_codec::benchInt64ColumnsFpc(column_int_data, cr, &loss);
        }
        totals.merge(fpc_codec::kRowName, cr, loss);
      }
      if (algorithm_allow_list::should_run(skip_algorithms, pfpc_codec::kRowName)) {
        if (dataType == "float") {
          pfpc_codec::benchDoubleColumnsPfpc(codec_cols, cr, &loss);
        } else {
          pfpc_codec::benchInt64ColumnsPfpc(column_int_data, cr, &loss);
        }
        totals.merge(pfpc_codec::kRowName, cr, loss);
      }
    }
    if (zfp_codec::bench_enabled()) {
      static const struct {
        double tol;
        const char* row;
      } k_zfp_tol[] = {
          {1e-1, "ext_zfp_tol_1e-1"},
          {1e-2, "ext_zfp_tol_1e-2"},
          {1e-3, "ext_zfp_tol_1e-3"},
      };
      for (const auto& zp : k_zfp_tol) {
        if (!algorithm_allow_list::should_run(skip_algorithms, zp.row)) {
          continue;
        }
        LossAccum loss;
        zfp_codec::benchToleranceDoubleColumns(codec_cols, zp.tol, zp.row, cr, &loss);
        totals.merge(zp.row, cr, loss);
      }
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "most_minimal")) {
      LossAccum loss;
      if (dataType == "float") {
        most_float::benchDoubleColumnsMostMinimal(codec_cols, cr, &loss);
      } else {
        most_float::benchInt64ColumnsMostMinimal(column_int_data, cr, &loss);
      }
      totals.merge("most_minimal", cr, loss);
    }
    static const struct {
      double rel;
      const char* row_name;
    } k_machete_presets[] = {
        {1e-1, "machete_rel_1e-1"},
        {1e-2, "machete_rel_1e-2"},
        {1e-3, "machete_rel_1e-3"},
        {1e-4, "machete_rel_1e-4"},
    };
    for (const auto& mp : k_machete_presets) {
      if (!algorithm_allow_list::should_run(skip_algorithms, mp.row_name)) {
        continue;
      }
      LossAccum loss;
      if (dataType == "float") {
        machete_float::benchDoubleColumnsMacheteRel(codec_cols, mp.rel, cr, &loss);
      } else {
        machete_float::benchInt64ColumnsMacheteRel(column_int_data, mp.rel, cr, &loss);
      }
      totals.merge(mp.row_name, cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "corad_tricklet")) {
      LossAccum loss;
      if (dataType == "float") {
        corad_tricklet_float::benchDoubleColumnsCoradTricklet(codec_cols, cr, &loss);
      } else {
        corad_tricklet_float::benchInt64ColumnsCoradTricklet(column_int_data, cr, &loss);
      }
      totals.merge("corad_tricklet", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "fastpfor")) {
      LossAccum loss;
      if (dataType == "float") {
        java_fastpfor::benchDoubleColumnsLikeJava(codec_cols, cr, &loss);
      } else {
        java_fastpfor::benchInt64ColumnsLikeJava(column_int_data, cr, &loss);
      }
      totals.merge("fastpfor", cr, loss);
    }
    const std::vector<int>* codec_max_points =
        (dataType == "float") ? &column_float_max_point : nullptr;
    if (algorithm_allow_list::should_run(skip_algorithms, "simple8b")) {
      LossAccum loss;
      if (dataType == "float") {
        simple8b_int::benchDoubleColumnsSimple8b(codec_cols, cr, codec_max_points, &loss);
      } else {
        simple8b_int::benchInt64ColumnsSimple8b(column_int_data, cr, &loss);
      }
      totals.merge("simple8b", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "leco")) {
      LossAccum loss;
      leco_double::benchDoubleColumnsLikeJava(codec_cols, cr, codec_max_points, &loss);
      totals.merge("leco", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "alp")) {
      LossAccum loss;
      if (dataType == "float") {
        alp_float::benchDoubleColumnsLikeJava(codec_cols, cr, &loss);
      } else {
        alp_float::benchInt64ColumnsLikeJava(column_int_data, cr, &loss);
      }
      totals.merge("alp", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "buff_kernel_native")) {
      LossAccum loss;
      bench_buff_kernel_native_columns(codec_cols, cr, &loss);
      totals.merge("buff_kernel_native", cr, loss);
    }
    if (dataType == "float" &&
        algorithm_allow_list::should_run(skip_algorithms, "CHIMP")) {
      LossAccum loss;
      elf_double::benchChimpDoubleColumnsLikeJava(codec_cols, cr, &loss);
      totals.merge("CHIMP", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "simpiece_relRange_0p005")) {
      LossAccum loss;
      if (dataType == "float") {
        simpiece_float::benchDoubleColumnsRelRange(codec_cols, 0.005, cr, &loss);
      } else {
        simpiece_float::benchInt64ColumnsRelRange(column_int_data, 0.005, cr, &loss);
      }
      totals.merge("simpiece_relRange_0p005", cr, loss);
    }
    if (algorithm_allow_list::should_run(skip_algorithms, "simpiece_relRange_0p01")) {
      LossAccum loss;
      if (dataType == "float") {
        simpiece_float::benchDoubleColumnsRelRange(codec_cols, 0.01, cr, &loss);
      } else {
        simpiece_float::benchInt64ColumnsRelRange(column_int_data, 0.01, cr, &loss);
      }
      totals.merge("simpiece_relRange_0p01", cr, loss);
    }
  }

  {
    std::vector<std::vector<double>> lossy_codec_cols;
    if (dataType == "int" && !column_int_data.empty()) {
      lossy_codec_cols = integers_as_double_columns(column_int_data);
    } else if (dataType == "float" && !column_float_data.empty()) {
      lossy_codec_cols = column_float_data;
    }
    if (!lossy_codec_cols.empty()) {
      bool wants_sz_serf = !algorithm_allow_list::has_allowlist();
      if (!wants_sz_serf) {
        for (const auto& name : algorithm_allow_list::from_allowlist_env()) {
          if (name.rfind("sz_", 0) == 0 || name.rfind("sz3_", 0) == 0 ||
              name.rfind("serf_", 0) == 0) {
            wants_sz_serf = true;
            break;
          }
        }
      }
      const char* skip_sz_serf = std::getenv("COMPRESSION_BENCHMARK_CPP_SKIP_SZ_SERF");
      if (wants_sz_serf && (skip_sz_serf == nullptr || skip_sz_serf[0] != '1')) {
        const std::filesystem::path tmp_path =
            std::filesystem::temp_directory_path() /
            ("wc_patch_sz_" + wc_temp::unique_tag() + ".csv");
        {
          std::ofstream tmp_out(tmp_path);
          bench_sz_sz3_serf_double(lossy_codec_cols, repo_root, tmp_out, skip_algorithms);
        }
        std::ifstream tmp_in(tmp_path);
        merge_report_stream(totals, tmp_in);
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
      }
    }
  }

  accumulate_generic_byte_compression(column_byte_data, totals, skip_algorithms);

  auto bench_with_loss = [&](const char* algo_name, auto fn) {
    if (!algorithm_allow_list::should_run(skip_algorithms, algo_name)) {
      return;
    }
    long long total_original = 0;
    long long total_compressed = 0;
    long long total_comp_time = 0;
    long long total_decomp_time = 0;
    LossAccum loss;
    for (const auto& data : column_byte_data) {
      std::vector<long long> result(4);
      try {
        fn(data, result, &loss);
        total_original += result[0];
        total_compressed += result[1];
        total_comp_time += result[2];
        total_decomp_time += result[3];
      } catch (...) {
      }
    }
    totals.merge(algo_name, total_original, total_compressed, total_comp_time, total_decomp_time,
                 loss);
  };

  bench_with_loss("ans", testAnsEntropyCompression);
  bench_with_loss("huffman", testHuffmanCompression);
  bench_with_loss("arithmetic", testArithmeticCompression);
}

void flush_totals(BenchmarkTotals& totals, std::ofstream& out,
                  const std::unordered_set<std::string>& existing_algorithms,
                  int column_index) {
  for (const auto& entry : totals.snapshot()) {
    if (algorithm_allow_list::should_skip(&existing_algorithms, entry.first)) {
      continue;
    }
    const auto& t = entry.second;
    if (t.size() < 4 || t[0] <= 0) {
      continue;
    }
    const int is_lossy = totals.is_lossy(entry.first);
    const std::string loss_rate = benchmark_loss::loss_rate_csv(totals.loss_rate(entry.first));
    if (column_index >= 0) {
      out << column_index << ',' << entry.first << ',' << t[0] << ',' << t[1] << ',' << t[2]
          << ',' << t[3] << ',' << is_lossy << ',' << loss_rate << '\n';
    } else {
      out << entry.first << ',' << t[0] << ',' << t[1] << ',' << t[2] << ',' << t[3] << ','
          << is_lossy << ',' << loss_rate << '\n';
    }
  }
}

} // namespace

void processCsvSimplePatch(const std::string& csvPath, const std::string& dataType,
                             const std::string& outputCsvPath,
                             const std::string& columnInfoPath,
                             int columnIndex) {
  if (dataType != "int" && dataType != "float" && dataType != "string") {
    std::cerr << "Unsupported data type: " << dataType << std::endl;
    return;
  }

  const bool single_column_mode = columnIndex >= 0;
  if (!single_column_mode) {
    if (const char* regen = std::getenv("COMPRESSION_BENCHMARK_STATICDATA_REGEN");
        regen != nullptr && regen[0] == '1' && std::filesystem::is_regular_file(outputCsvPath)) {
      std::error_code ec;
      std::filesystem::remove(outputCsvPath, ec);
    }
  }

  std::unordered_set<std::string> existing_algorithms;
  bool append_mode = false;
  if (!single_column_mode) {
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
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos || comma == 0) {
          continue;
        }
        existing_algorithms.insert(line.substr(0, comma));
      }
    }
    append_mode = std::filesystem::is_regular_file(outputCsvPath);
  }

  std::ofstream output_file(outputCsvPath, append_mode ? std::ios::app : std::ios::out);
  if (!output_file.is_open()) {
    std::cerr << "Failed to create output file: " << outputCsvPath << std::endl;
    return;
  }
  if (!append_mode) {
    if (single_column_mode) {
      output_file << "column_index,algorithm,originalSize,compressedSize,compressionTime,decompressionTime,isLossy,lossRate\n";
    } else {
      output_file << "algorithm,originalSize,compressedSize,compressionTime,decompressionTime,isLossy,lossRate\n";
    }
  }

  std::unordered_map<std::string, int> max_point_by_name;
  std::unordered_map<int, ColumnInfoMeta> info_by_index;
  if (!single_column_mode) {
    max_point_by_name = load_max_point_by_column_name(columnInfoPath);
    info_by_index = load_column_info_by_index(columnInfoPath);
  }

  std::vector<std::string> headers;
  {
    std::ifstream hdr_in(csvPath);
    std::string header_line;
    if (!std::getline(hdr_in, header_line)) {
      std::cerr << "CSV file is empty" << std::endl;
      return;
    }
    split_csv_line(header_line, headers);
  }
  if (headers.empty()) {
    return;
  }
  if (single_column_mode) {
    const auto info_by_index = load_column_info_by_index(columnInfoPath);
    auto it = info_by_index.find(columnIndex);
    int max_point = 0;
    if (it != info_by_index.end()) {
      max_point = it->second.max_point_number;
      const std::string logical = it->second.logical_type;
      const bool logical_int = (logical == "integer" || logical == "int" || logical == "long");
      const bool logical_float = (logical == "float" || logical == "double");
      const bool logical_string = (logical == "string");
      if ((dataType == "int" && !logical_int) || (dataType == "float" && !logical_float) ||
          (dataType == "string" && !logical_string)) {
        std::cerr << "Warning: column_info logical_type mismatch for column_index=" << columnIndex
                  << " data_type=" << dataType << " logical_type=" << logical << std::endl;
      }
    }
    max_point_by_name[headers[0]] = max_point;
  }

  std::string repo_root = ".";
  if (const char* rr = std::getenv("COMPRESSION_BENCHMARK_REPO_ROOT")) {
    repo_root = rr;
  }

  std::unordered_set<std::string> skip_from_env = algorithm_allow_list::from_incremental_skip_env();
  const std::unordered_set<std::string>* skip_algorithms =
      algorithm_allow_list::has_allowlist() ? nullptr : &skip_from_env;

  BenchmarkTotals totals;
  const int patch_rows = patch_row_limit();
  std::ifstream csv_in(csvPath);
  std::string header_discard;
  if (!std::getline(csv_in, header_discard)) {
    return;
  }

  std::string line;
  int rows_in_patch = 0;
  int patch_index = 0;
  long long rows_processed = 0;
  std::vector<std::vector<std::string>> patch_columns(headers.size());

  const auto process_patch = [&]() {
    if (rows_in_patch <= 0) {
      return;
    }
    const long long row_offset = rows_processed;
    patch_index++;
    rows_processed += rows_in_patch;
    const int out_column = single_column_mode ? columnIndex : -1;
    if (patch_bench_verbose()) {
      std::cout << "ProcessorPatchBench patch " << patch_index << ": " << rows_in_patch
                << " rows (limit=" << patch_rows << ", total=" << rows_processed
                << ", data_type=" << dataType << ", column_index=" << out_column << ")\n";
    }
    accumulate_grouping_patch(dataType, headers, patch_columns,
                              static_cast<std::uint64_t>(std::max<long long>(0, row_offset)),
                              info_by_index, totals, skip_algorithms);
    PatchColumnData patch;
    if (build_patch_column_data(dataType, headers, patch_columns, max_point_by_name, patch)) {
      accumulate_patch_columns(dataType, patch, totals, repo_root, skip_algorithms);
    }
    for (auto& col : patch_columns) {
      col.clear();
    }
    rows_in_patch = 0;
  };

  while (std::getline(csv_in, line)) {
    std::vector<std::string> values;
    split_csv_line(line, values);
    for (std::size_t i = 0; i < headers.size(); ++i) {
      if (i < values.size()) {
        patch_columns[i].push_back(values[i]);
      } else {
        patch_columns[i].push_back("");
      }
    }
    rows_in_patch++;
    if (rows_in_patch < patch_rows) {
      continue;
    }
    process_patch();
  }
  process_patch();

  flush_totals(totals, output_file, existing_algorithms, single_column_mode ? columnIndex : -1);
  output_file.flush();
  output_file.close();

  if (single_column_mode) {
    std::cout << "Compression results written to " << outputCsvPath << std::endl;
    return;
  }

  if (dataType == "string") {
    std::cout << "Compression results written to " << outputCsvPath << std::endl;
    return;
  }
  if (algorithm_allow_list::has_allowlist()) {
    const auto allowed = algorithm_allow_list::from_allowlist_env();
    if (allowed.size() == 1 &&
        (allowed.count("reger") > 0 || allowed.count(grouping_codec::kRowName) > 0)) {
      std::cout << "Compression results written to " << outputCsvPath << std::endl;
      return;
    }
  }

  const std::string csv_abs = std::filesystem::absolute(csvPath).string();
  const std::string out_abs = std::filesystem::absolute(outputCsvPath).string();
  std::cout << "Java companion (CppBenchmarkCompanion external-only)..." << std::endl;
  if (!run_java_companion_phase(repo_root, csv_abs, dataType, out_abs, "external-only")) {
    std::cerr << "Warning: Java companion (external-only) failed; optional research JAR rows missing.\n";
  }

  std::cout << "Compression results written to " << outputCsvPath << std::endl;
}
