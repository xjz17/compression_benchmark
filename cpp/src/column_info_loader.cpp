#include "column_info_loader.hpp"

#include <fstream>
#include <sstream>
#include <vector>

namespace {

std::string trim_copy(std::string s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
    s.erase(s.begin());
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
    s.pop_back();
  }
  return s;
}

} // namespace

std::unordered_map<std::string, int> load_max_point_by_column_name(
    const std::string& column_info_path) {
  std::unordered_map<std::string, int> out;
  if (column_info_path.empty()) {
    return out;
  }
  std::ifstream in(column_info_path);
  if (!in.is_open()) {
    return out;
  }
  std::string header;
  if (!std::getline(in, header)) {
    return out;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      fields.push_back(trim_copy(cell));
    }
    if (fields.size() < 4) {
      continue;
    }
    const std::string& name = fields[1];
    int mp = 0;
    try {
      mp = std::stoi(fields[3]);
    } catch (...) {
      mp = 0;
    }
    if (!name.empty()) {
      out[name] = std::max(0, mp);
    }
  }
  return out;
}

std::unordered_map<int, ColumnInfoMeta> load_column_info_by_index(
    const std::string& column_info_path) {
  std::unordered_map<int, ColumnInfoMeta> out;
  if (column_info_path.empty()) {
    return out;
  }
  std::ifstream in(column_info_path);
  if (!in.is_open()) {
    return out;
  }
  std::string header;
  if (!std::getline(in, header)) {
    return out;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      fields.push_back(trim_copy(cell));
    }
    if (fields.size() < 4) {
      continue;
    }
    int idx = -1;
    try {
      idx = std::stoi(fields[0]);
    } catch (...) {
      idx = -1;
    }
    if (idx < 0) {
      continue;
    }
    ColumnInfoMeta meta;
    meta.logical_type = trim_copy(fields[2]);
    try {
      meta.max_point_number = std::max(0, std::stoi(fields[3]));
    } catch (...) {
      meta.max_point_number = 0;
    }
    out[idx] = meta;
  }
  return out;
}
