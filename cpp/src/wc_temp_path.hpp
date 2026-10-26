#pragma once

#include <random>
#include <sstream>
#include <string>

namespace wc_temp {

/** Random hex suffix for unique files under std::filesystem::temp_directory_path(). */
inline std::string unique_tag() {
  static std::mt19937_64 rng{std::random_device{}()};
  std::ostringstream o;
  o << std::hex << rng();
  return o.str();
}

}  // namespace wc_temp
