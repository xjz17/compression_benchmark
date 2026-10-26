#pragma once

#include <string>
#include <unordered_map>

/** Load ``column_name`` -> ``max_point_number`` from ``{key}_column_info.csv``. */
std::unordered_map<std::string, int> load_max_point_by_column_name(
    const std::string& column_info_path);

struct ColumnInfoMeta {
  std::string logical_type;
  int max_point_number = 0;
};

/** Load ``column_index`` -> ``{logical_type,max_point_number}`` from ``{key}_column_info.csv``. */
std::unordered_map<int, ColumnInfoMeta> load_column_info_by_index(
    const std::string& column_info_path);
