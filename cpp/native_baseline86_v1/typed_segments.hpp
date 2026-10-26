#pragma once

#include "baseline.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <limits>

namespace native_baseline86_v1::typed_segments {
struct Range { std::uint64_t offset, length; };

inline Range range(const std::filesystem::path& path,
                   const nlohmann::json& row, const std::string& field) {
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("typed source is not a regular file: "+path.string());
  const auto size=std::filesystem::file_size(path);
  const auto offset_key=field+"_offset",length_key=field+"_length";
  const bool sliced=row.contains(offset_key)||row.contains(length_key);
  Range result{0,size};
  if (sliced) {
    if (!row.contains(offset_key)||!row.contains(length_key)||
        !row.at(offset_key).is_number_unsigned()||!row.at(length_key).is_number_unsigned())
      throw std::invalid_argument("typed segment requires unsigned offset and length: "+field);
    result={row.at(offset_key).get<std::uint64_t>(),row.at(length_key).get<std::uint64_t>()};
  }
  if (result.offset>size||result.length>size-result.offset||
      result.length>(256ULL<<20)||
      result.offset>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
    throw std::invalid_argument("typed segment bounds exceeded: "+field);
  return result;
}

inline Bytes read(const std::filesystem::path& path, Range slice) {
  std::ifstream input(path,std::ios::binary);
  if (!input) throw std::invalid_argument("cannot open typed segment: "+path.string());
  input.seekg(static_cast<std::streamoff>(slice.offset));
  Bytes bytes(static_cast<std::size_t>(slice.length));
  if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
  if (!input) throw std::invalid_argument("cannot read complete typed segment: "+path.string());
  return bytes;
}

inline Bytes read(const std::filesystem::path& path,const nlohmann::json& row,
                  const std::string& field) {
  return read(path,range(path,row,field));
}
} // namespace native_baseline86_v1::typed_segments
