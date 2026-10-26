#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace native_baseline86_v1::upstream_legacy {
using Bytes=std::vector<std::uint8_t>;
bool supports(const std::string& label);
Bytes encode(const std::string&,std::span<const std::uint8_t>);
Bytes decode(const std::string&,std::span<const std::uint8_t>,std::size_t);
}
