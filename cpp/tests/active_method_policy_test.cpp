#include "../src/active_method_policy.hpp"
#include "../src/native_lz_family.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
  using active_method_policy::is_active_algorithm_key;
  assert(std::string(active_method_policy::kPolicyId) ==
         "paper164-active85-glza-retired-v1");
  assert(active_method_policy::kCatalogMethodCount == 86);
  assert(active_method_policy::kActiveMethodCount == 85);
  assert(!is_active_algorithm_key("glza"));
  assert(is_active_algorithm_key("lz4-framed"));
  const auto& algorithms = native_lz_family_algorithms();
  assert(std::find(algorithms.begin(), algorithms.end(), "glza") ==
         algorithms.end());
  try {
    (void)compress_native_lz_family({'a', 'b', 'c'}, "glza");
    assert(false && "active C++ encode accepted GLZA");
  } catch (const std::invalid_argument& error) {
    assert(std::string(error.what()).find(active_method_policy::kPolicyId) !=
           std::string::npos);
  }
  const std::vector<char> literal_frame = {0, 'a', 'b', 'c'};
  assert(decompress_native_lz_family(literal_frame, "glza", 3) ==
         std::vector<char>({'a', 'b', 'c'}));
  return 0;
}
