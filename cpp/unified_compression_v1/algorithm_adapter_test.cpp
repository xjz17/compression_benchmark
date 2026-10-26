#include "algorithm_adapter.hpp"

#include "../native_graph_v1/graph_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using unified_compression_v1::Bytes;
using unified_compression_v1::Column;
using unified_compression_v1::Error;
using unified_compression_v1::MethodInfo;
using unified_compression_v1::detail::AlgorithmAdapter;

void require(bool condition, std::string message) {
  if (!condition) throw std::runtime_error(std::move(message));
}

template <class Function>
void require_error(Function&& function, std::string_view expected_code) {
  try {
    function();
  } catch (const Error& error) {
    require(error.code() == expected_code,
            "expected error " + std::string(expected_code) + ", got " +
                error.code() + ": " + error.what());
    return;
  }
  throw std::runtime_error("expected error " + std::string(expected_code));
}

template <class T>
Column dense_column(std::string dtype, const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable_v<T>);
  Bytes bytes(values.size() * sizeof(T));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return {std::move(dtype), static_cast<std::uint64_t>(values.size()),
          std::move(bytes), Bytes((values.size() + 7) / 8, 0)};
}

Column integer32_fixture() {
  std::vector<std::int32_t> values(128);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<std::int32_t>((i / 8) % 11) - 5;
  return dense_column("int32", values);
}

Column integer64_fixture() {
  std::vector<std::int64_t> values(128);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<std::int64_t>(1000 + i / 4 + (i % 3));
  return dense_column("int64", values);
}

Column float32_fixture() {
  std::vector<float> values(128);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.25f);
  return dense_column("float32", values);
}

Column ordinary_float64_fixture() {
  std::vector<double> values(128);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = (static_cast<int>(i % 19) - 9) * 0.25;
  return dense_column("float64", values);
}

Column exact_subset_fixture(std::string_view method) {
  if (method == "algorithm:CORAD")
    return dense_column("float64", std::vector<double>(128, 1.25));
  if (method == "algorithm:BUFF") {
    static constexpr double pattern[] =
        {0.5, 1.0, 1.25, 2.5, -1.5, -0.25, 64.0, 64.5};
    std::vector<double> values(128);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = pattern[i % (sizeof(pattern) / sizeof(pattern[0]))];
    return dense_column("float64", values);
  }
  return ordinary_float64_fixture();
}

Column fixture(const MethodInfo& method, std::string_view dtype) {
  if (dtype == "int32") return integer32_fixture();
  if (dtype == "int64") return integer64_fixture();
  if (dtype == "float32") return float32_fixture();
  if (dtype == "float64") return exact_subset_fixture(method.name);
  throw std::runtime_error("test encountered an unexpected public dtype");
}

const MethodInfo& named(const std::vector<MethodInfo>& methods,
                        std::string_view name) {
  const auto it = std::find_if(methods.begin(), methods.end(),
                               [&](const auto& method) { return method.name == name; });
  if (it == methods.end())
    throw std::runtime_error("missing catalog method " + std::string(name));
  return *it;
}

void test_catalog_and_disabled_policy(const AlgorithmAdapter& adapter) {
  const auto& methods = adapter.methods();
  require(methods.size() == 86, "algorithm adapter did not register all 86 labels");
  std::set<std::string> names;
  for (const auto& method : methods) {
    require(method.kind == "algorithm", method.name + " has wrong kind");
    require(method.name == "algorithm:" + method.display_name,
            method.name + " has a noncanonical display binding");
    require(names.emplace(method.name).second, method.name + " is duplicated");
    require(!method.aliases.empty() && method.aliases.front() == method.display_name,
            method.name + " lacks its exact inventory-label alias");
    require(!method.enabled || method.available,
            method.name + " is enabled without a linked implementation");
    const auto implementation_class =
        method.contract.at("implementation_class").get<std::string>();
    if (method.enabled)
      require(implementation_class == "direct_current" ||
                  implementation_class == "direct_generic" ||
                  implementation_class == "direct_exact_subset",
              method.name + " exposes a proxy/alias as the named algorithm");
  }

  const auto& glza = named(methods, "algorithm:GLZA");
  const auto& ppmd = named(methods, "algorithm:PPMD8");
  require(!glza.enabled, "GLZA must stay disabled");
  require(!ppmd.enabled, "PPMD8 must stay disabled");

  const auto input = integer64_fixture();
  require_error([&] { (void)adapter.encode(glza.name, input); }, "method_disabled");
  require_error([&] { (void)adapter.encode(ppmd.name, input); }, "method_disabled");
  for (const auto* denied : {"algorithm:NNCP", "algorithm:PPMd", "algorithm:DMC",
                            "algorithm:PAQ8", "algorithm:CMIX", "algorithm:LSTM"})
    require_error([&] { (void)adapter.encode(denied, input); }, "method_disabled");
}

void test_enabled_roundtrips(const AlgorithmAdapter& adapter) {
  std::size_t enabled = 0;
  std::size_t roundtrips = 0;
  std::size_t explicit_domain_rejections = 0;
  for (const auto& method : adapter.methods()) {
    if (!method.enabled) continue;
    ++enabled;
    require(!method.dtypes.empty(), method.name + " has no unified dtype");
    const bool exact_subset =
        method.contract.at("implementation_class") == "direct_exact_subset";
    for (const auto& dtype : method.dtypes) {
      const auto input = fixture(method, dtype);
      try {
        const auto frame = adapter.encode(method.name, input);
        require(!frame.empty(), method.name + " produced an empty complete frame");
        const auto restored = adapter.decode(method.name, frame);
        require(restored == input,
                method.name + " did not restore " + dtype + " bit-exactly");
        ++roundtrips;
      } catch (const Error& error) {
        if (exact_subset && error.code() == "input_domain") {
          ++explicit_domain_rejections;
          continue;
        }
        throw std::runtime_error(method.name + " " + dtype + ": " + error.code() +
                                 ": " + error.what());
      }
    }
  }
  require(enabled != 0 && roundtrips != 0,
          "no enabled algorithm roundtrip was exercised");

  // CORAD is admitted only on its exact constant-column branch.  A clear
  // domain rejection is retained as a rejection, never counted as a success.
  std::vector<double> nonconstant(128, 1.25);
  nonconstant.back() = 2.5;
  require_error(
      [&] {
        (void)adapter.encode("algorithm:CORAD",
                             dense_column("float64", nonconstant));
      },
      "input_domain");
  ++explicit_domain_rejections;
  require(explicit_domain_rejections != 0,
          "exact-subset domain rejection path was not exercised");

  std::cout << "enabled algorithms=" << enabled
            << " dtype roundtrips=" << roundtrips
            << " explicit domain rejections=" << explicit_domain_rejections << '\n';
}

void test_inner_binding_precedes_inverse(const AlgorithmAdapter& adapter) {
  const auto input = integer64_fixture();
  auto inner = adapter.encode("algorithm:RLE", input);

  // Damage a checksummed leaf, then recompute only the NGF1 container.  If an
  // inverse runs, the codec/root integrity check must fail.  The expected-name
  // mismatch has to win first, proving the inner label is preflighted before
  // dispatching the inverse kernel.
  auto parsed = native_graph_v1::detail::parse_frame(inner, {});
  bool changed = false;
  for (auto& [stream_id, stream] : parsed.leaves) {
    (void)stream_id;
    if (!stream.data.empty()) {
      stream.data.back() ^= 0x01;
      changed = true;
      break;
    }
  }
  require(changed, "RLE test frame has no leaf payload to corrupt");
  inner = native_graph_v1::detail::serialize_frame(parsed, {});
  require_error([&] { (void)adapter.decode("algorithm:Dict.", inner); },
                "inner_binding_mismatch");
}

}  // namespace

int main() try {
  const AlgorithmAdapter adapter;
  test_catalog_and_disabled_policy(adapter);
  test_enabled_roundtrips(adapter);
  test_inner_binding_precedes_inverse(adapter);
  std::cout << "unified algorithm adapter tests passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
