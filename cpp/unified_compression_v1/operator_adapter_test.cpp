#include "operator_adapter.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <span>
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
using unified_compression_v1::detail::OperatorAdapter;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("operator_adapter_test: " + message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

template <class Unsigned>
void append_little_endian(Bytes& output, Unsigned bits) {
  static_assert(std::is_unsigned_v<Unsigned>);
  for (std::size_t byte = 0; byte < sizeof(Unsigned); ++byte)
    output.push_back(static_cast<std::uint8_t>(bits >> (byte * 8)));
}

Column int64_fixture() {
  Column result{"int64", 128, {}, Bytes(16, 0)};
  result.values.reserve(128 * 8);
  for (std::int64_t index = 0; index < 128; ++index) {
    const std::int64_t daily = (index % 24) * 17;
    const std::int64_t weekly = ((index / 24) % 7) * 101;
    const auto value = static_cast<std::uint64_t>(100000 + daily + weekly +
                                                  (index % 5));
    append_little_endian(result.values, value);
  }
  return result;
}

Column int32_fixture() {
  Column result{"int32", 128, {}, Bytes(16, 0)};
  result.values.reserve(128 * 4);
  for (std::int32_t index = 0; index < 128; ++index) {
    const auto value = static_cast<std::uint32_t>(
        1200 + (index % 24) * 5 + ((index / 24) % 7) * 31);
    append_little_endian(result.values, value);
  }
  return result;
}

Column float64_fixture() {
  Column result{"float64", 128, {}, Bytes(16, 0)};
  result.values.reserve(128 * 8);
  for (std::uint64_t index = 0; index < 128; ++index) {
    const double value = 21.5 + 2.75 * std::sin(index * 0.11) +
                         0.003 * static_cast<double>(index);
    append_little_endian(result.values, std::bit_cast<std::uint64_t>(value));
  }
  return result;
}

Column float32_fixture() {
  Column result{"float32", 128, {}, Bytes(16, 0)};
  result.values.reserve(128 * 4);
  for (std::uint32_t index = 0; index < 128; ++index) {
    const float value = 7.25F + 1.5F * std::sin(index * 0.09F) + index * 0.001F;
    append_little_endian(result.values, std::bit_cast<std::uint32_t>(value));
  }
  return result;
}

Column fixture_for(const MethodInfo& method) {
  const auto width = method.contract.at("initial_width").get<std::uint32_t>();
  const auto view = method.contract.at("input_view").get<std::string>();
  if (width == 4)
    return view == "float32" ? float32_fixture() : int32_fixture();
  if (width == 8 && (view == "float64" || view == "physical_word"))
    return float64_fixture();
  if (width == 8) return int64_fixture();
  // Byte-stream operators admit every public numeric dtype.  A finite
  // float64 time-series fixture exercises the raw-byte view without a cast.
  return float64_fixture();
}

bool explicit_domain_rejection(const Error& error) {
  static const std::set<std::string> codes{
      "CAMEL_INT64_NOT_EXACTLY_REPRESENTABLE", "CAMEL_NONFINITE_INPUT",
      "CAMEL_DECIMAL_SCALE_UNREPRESENTABLE", "FPC_EMPTY_INPUT",
      "GORILLA_RESERVED_MARKER_INPUT"};
  return codes.contains(error.code());
}

std::string run_empty_case(const OperatorAdapter& adapter,
                           const MethodInfo& method) {
  if (!method.available) return "unavailable:" + method.reason;
  Column empty{method.contract.at("initial_width") == 4 ? "float32" : "float64",
               0, {}, {}};
  if (std::find(method.dtypes.begin(), method.dtypes.end(), empty.dtype) ==
      method.dtypes.end())
    empty.dtype = method.dtypes.front();
  try {
    const auto frame = adapter.encode(method.name, empty);
    const auto restored = OperatorAdapter::decode(method.name, frame);
    require(restored == empty, method.name + " empty roundtrip mismatch");
    return "roundtrip";
  } catch (const Error& error) {
    return "rejected:" + error.code() + ":" + error.what();
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: operator_adapter_test ABSOLUTE_CATALOG_PATH");
    OperatorAdapter adapter(argv[1]);
    const auto& methods = adapter.methods();
    require(methods.size() == 102, "registry size is not 102");

    std::size_t enabled = 0;
    std::size_t available = 0;
    std::set<std::string> disabled;
    for (const auto& method : methods) {
      enabled += method.enabled ? 1 : 0;
      available += method.available ? 1 : 0;
      if (!method.enabled) disabled.insert(method.name.substr(9));
      require(method.kind == "operator", method.name + " has wrong kind");
      require(method.name.starts_with("operator:A"),
              method.name + " is not canonical");
      require(method.contract.value("bit_preserving", false),
              method.name + " lacks bit-preserving contract");
      require(method.contract.value("numeric_conversion", true) == false,
              method.name + " incorrectly claims numeric conversion");
      require(method.contract.contains("adapter_rule"),
              method.name + " lacks explicit raw-lane adapter rule");
      const auto width =
          method.contract.at("initial_width").get<std::uint32_t>();
      if (width == 1 || width == 4)
        require(method.dtypes ==
                    std::vector<std::string>({"int32", "int64", "float32",
                                              "float64"}),
                method.name + " raw-lane dtype inventory is incomplete");
      else
        require(method.dtypes ==
                    std::vector<std::string>({"int64", "float64"}),
                method.name + " 64-bit dtype inventory is incorrect");
    }
    require(enabled == 97, "enabled count is not 97");
    require(disabled == std::set<std::string>({"A106", "A107", "A108",
                                               "A109", "A110"}),
            "disabled set differs from A106-A110");
    // With optional support dependencies linked every enabled method is
    // available; the ordinary local build exposes the 86-method closure.
    require(available == 86 || available == 97,
            "available count is neither the ordinary nor full closure");

    std::size_t roundtrips = 0;
    std::vector<std::string> domain_rejections;
    for (const auto& method : methods) {
      if (!method.enabled || !method.available) continue;
      const auto input = fixture_for(method);
      try {
        const auto frame = adapter.encode(method.name, input);
        const auto restored = OperatorAdapter::decode(method.name, frame);
        require(restored == input, method.name + " bitwise roundtrip mismatch");
        ++roundtrips;
      } catch (const Error& error) {
        if (!explicit_domain_rejection(error)) throw;
        domain_rejections.push_back(method.name + ":" + error.code());
      }
    }
    require(roundtrips + domain_rejections.size() == available,
            "not every available method produced a recorded outcome");

    for (const auto& id : disabled) {
      const auto name = "operator:" + id;
      bool rejected = false;
      try {
        (void)adapter.encode(name, int64_fixture());
      } catch (const Error& error) {
        rejected = error.code() == "operator_disabled";
      }
      require(rejected, name + " was callable");
    }
    bool cold_disabled_rejected = false;
    try {
      (void)OperatorAdapter::decode("operator:A106", {});
    } catch (const Error& error) {
      cold_disabled_rejected = error.code() == "operator_disabled";
    }
    require(cold_disabled_rejected, "cold decoder did not reject A106 first");

    const auto find = [&](std::string_view name) -> const MethodInfo& {
      const auto it = std::find_if(methods.begin(), methods.end(),
                                   [&](const auto& method) {
                                     return method.name == name;
                                   });
      require(it != methods.end(), "missing empty-input target");
      return *it;
    };
    const auto a048_empty = run_empty_case(adapter, find("operator:A048"));
    const auto a089_empty = run_empty_case(adapter, find("operator:A089"));
    const auto a090_empty = run_empty_case(adapter, find("operator:A090"));

    std::cout << "registered=102 enabled=" << enabled
              << " disabled=5 available=" << available
              << " fixture_roundtrips=" << roundtrips
              << " domain_rejections=" << domain_rejections.size() << '\n';
    for (const auto& rejection : domain_rejections)
      std::cout << "domain_rejected=" << rejection << '\n';
    std::cout << "empty_A048=" << a048_empty << '\n'
              << "empty_A089=" << a089_empty << '\n'
              << "empty_A090=" << a090_empty << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
