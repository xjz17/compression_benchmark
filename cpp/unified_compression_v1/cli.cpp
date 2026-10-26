#include "api.hpp"
#include "selection.hpp"
#include "recommender.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <openssl/evp.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace uc = unified_compression_v1;
using Json = nlohmann::json;

namespace {

[[noreturn]] void invalid(std::string message) {
  throw uc::Error("invalid_request", std::move(message));
}

std::uint64_t unsigned_integer(const Json& value, std::string_view field,
                               std::uint64_t maximum =
                                   std::numeric_limits<std::uint64_t>::max()) {
  std::uint64_t result = 0;
  if (value.is_number_unsigned()) {
    result = value.get<std::uint64_t>();
  } else if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value < 0) invalid(std::string(field) + " must be non-negative");
    result = static_cast<std::uint64_t>(signed_value);
  } else {
    invalid(std::string(field) + " must be an integer");
  }
  if (result > maximum) invalid(std::string(field) + " is out of range");
  return result;
}

std::uint64_t required_unsigned(const Json& request, std::string_view field,
                                std::uint64_t maximum =
                                    std::numeric_limits<std::uint64_t>::max()) {
  const auto key = std::string(field);
  if (!request.contains(key)) invalid("missing " + key);
  return unsigned_integer(request.at(key), field, maximum);
}

std::string required_string(const Json& request, std::string_view field) {
  const auto key = std::string(field);
  if (!request.contains(key) || !request.at(key).is_string())
    invalid("missing or non-string " + key);
  return request.at(key).get<std::string>();
}

int hex_digit(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

uc::Bytes from_hex(std::string_view input, std::string_view field) {
  if (input.size() % 2 != 0)
    invalid(std::string(field) + " must contain an even number of hex digits");
  uc::Bytes output;
  output.reserve(input.size() / 2);
  for (std::size_t index = 0; index < input.size(); index += 2) {
    const auto high = hex_digit(input[index]);
    const auto low = hex_digit(input[index + 1]);
    if (high < 0 || low < 0)
      invalid(std::string(field) + " contains a non-hex character");
    output.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return output;
}

std::string to_hex(std::span<const std::uint8_t> input) {
  constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(input.size() * 2);
  for (const auto byte : input) {
    output.push_back(digits[byte >> 4]);
    output.push_back(digits[byte & 15]);
  }
  return output;
}

std::string base64_encode(std::span<const std::uint8_t> input) {
  constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve(((input.size() + 2) / 3) * 4);
  for (std::size_t index = 0; index < input.size(); index += 3) {
    const auto remaining = input.size() - index;
    const std::uint32_t word =
        static_cast<std::uint32_t>(input[index]) << 16 |
        (remaining > 1 ? static_cast<std::uint32_t>(input[index + 1]) << 8 : 0) |
        (remaining > 2 ? input[index + 2] : 0);
    output.push_back(alphabet[(word >> 18) & 63]);
    output.push_back(alphabet[(word >> 12) & 63]);
    output.push_back(remaining > 1 ? alphabet[(word >> 6) & 63] : '=');
    output.push_back(remaining > 2 ? alphabet[word & 63] : '=');
  }
  return output;
}

int base64_digit(char value) {
  if (value >= 'A' && value <= 'Z') return value - 'A';
  if (value >= 'a' && value <= 'z') return value - 'a' + 26;
  if (value >= '0' && value <= '9') return value - '0' + 52;
  if (value == '+') return 62;
  if (value == '/') return 63;
  return -1;
}

uc::Bytes base64_decode(std::string_view input) {
  if (input.size() % 4 != 0) invalid("compressed_base64 has invalid length");
  uc::Bytes output;
  output.reserve((input.size() / 4) * 3);
  for (std::size_t index = 0; index < input.size(); index += 4) {
    const bool last = index + 4 == input.size();
    const auto first = base64_digit(input[index]);
    const auto second = base64_digit(input[index + 1]);
    const bool pad2 = input[index + 2] == '=';
    const bool pad3 = input[index + 3] == '=';
    const auto third = pad2 ? 0 : base64_digit(input[index + 2]);
    const auto fourth = pad3 ? 0 : base64_digit(input[index + 3]);
    if (first < 0 || second < 0 || third < 0 || fourth < 0 ||
        (!last && (pad2 || pad3)) || (pad2 && !pad3) ||
        (pad2 && (second & 15) != 0) || (pad3 && !pad2 && (third & 3) != 0))
      invalid("compressed_base64 is not canonical base64");
    const std::uint32_t word = static_cast<std::uint32_t>(first) << 18 |
                               static_cast<std::uint32_t>(second) << 12 |
                               static_cast<std::uint32_t>(third) << 6 |
                               static_cast<std::uint32_t>(fourth);
    output.push_back(static_cast<std::uint8_t>(word >> 16));
    if (!pad2) output.push_back(static_cast<std::uint8_t>(word >> 8));
    if (!pad3) output.push_back(static_cast<std::uint8_t>(word));
  }
  return output;
}

uc::Bytes read_bytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw uc::Error("io_error", "cannot read " + path.string());
  uc::Bytes output{std::istreambuf_iterator<char>(input), {}};
  if (input.bad()) throw uc::Error("io_error", "failed while reading " + path.string());
  return output;
}

void write_bytes(const fs::path& path, std::span<const std::uint8_t> bytes) {
  if (!path.is_absolute())
    invalid("output_path must be absolute");
  static std::uint64_t sequence = 0;
  const auto temporary = path.string() + ".tmp." + std::to_string(::getpid()) +
                         "." + std::to_string(++sequence);
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) throw uc::Error("io_error", "cannot write " + temporary);
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    fs::remove(temporary);
    throw uc::Error("io_error", "failed while writing " + path.string());
  }
  std::error_code error;
  fs::rename(temporary, path, error);
  if (error) {
    fs::remove(temporary);
    throw uc::Error("io_error", "cannot commit " + path.string() + ": " + error.message());
  }
}

std::string sha256(std::span<const std::uint8_t> input) {
  std::array<std::uint8_t, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  auto* context = EVP_MD_CTX_new();
  if (context == nullptr)
    throw uc::Error("internal_error", "cannot allocate SHA-256 context");
  const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                  EVP_DigestUpdate(context, input.data(), input.size()) == 1 &&
                  EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
  EVP_MD_CTX_free(context);
  if (!ok || digest_size != 32)
    throw uc::Error("internal_error", "SHA-256 failed");
  return to_hex({digest.data(), digest_size});
}

template <typename Unsigned>
void append_unsigned_le(uc::Bytes& output, Unsigned value) {
  static_assert(std::is_unsigned_v<Unsigned>);
  for (std::size_t index = 0; index < sizeof(value); ++index)
    output.push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}

std::int64_t json_signed(const Json& value, std::string_view dtype,
                         std::int64_t minimum, std::int64_t maximum) {
  std::int64_t result = 0;
  // Test unsigned first: nlohmann treats unsigned integers as integers too.
  // Converting uint64 above INT64_MAX to int64 would otherwise wrap.
  if (value.is_number_unsigned()) {
    const auto unsigned_value = value.get<std::uint64_t>();
    if (unsigned_value > static_cast<std::uint64_t>(maximum))
      invalid(std::string(dtype) + " value is out of range");
    result = static_cast<std::int64_t>(unsigned_value);
  } else if (value.is_number_integer()) {
    result = value.get<std::int64_t>();
  } else {
    invalid(std::string(dtype) + " values must be JSON integers or null");
  }
  if (result < minimum || result > maximum)
    invalid(std::string(dtype) + " value is out of range");
  return result;
}

void append_json_value(uc::Bytes& output, std::string_view dtype,
                       const Json& value) {
  if (dtype == "int32") {
    const auto parsed = static_cast<std::int32_t>(json_signed(
        value, dtype, std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max()));
    append_unsigned_le(output, std::bit_cast<std::uint32_t>(parsed));
    return;
  }
  if (dtype == "int64") {
    const auto parsed = json_signed(value, dtype,
                                    std::numeric_limits<std::int64_t>::min(),
                                    std::numeric_limits<std::int64_t>::max());
    append_unsigned_le(output, std::bit_cast<std::uint64_t>(parsed));
    return;
  }
  if (!value.is_number())
    invalid(std::string(dtype) + " values must be finite JSON numbers or null");
  const auto parsed = value.get<double>();
  if (!std::isfinite(parsed))
    invalid("non-finite values require the values_le_hex input channel");
  if (dtype == "float32") {
    const auto narrowed = static_cast<float>(parsed);
    if (!std::isfinite(narrowed)) invalid("float32 value is out of range");
    append_unsigned_le(output, std::bit_cast<std::uint32_t>(narrowed));
    return;
  }
  if (dtype == "float64") {
    append_unsigned_le(output, std::bit_cast<std::uint64_t>(parsed));
    return;
  }
  invalid("unsupported dtype " + std::string(dtype));
}

uc::Column parse_column(const Json& request) {
  if (!request.is_object()) invalid("request must be a JSON object");
  uc::Column column;
  column.dtype = required_string(request, "dtype");
  (void)uc::dtype_width(column.dtype);
  const bool array_channel = request.contains("values");
  const bool byte_channel = request.contains("values_le_hex") ||
                            request.contains("missing_bitmap_hex");
  if (array_channel == byte_channel)
    invalid("provide exactly one of values or the values_le_hex channel");
  if (array_channel) {
    const auto& values = request.at("values");
    if (!values.is_array()) invalid("values must be an array");
    column.row_count = values.size();
    if (request.contains("row_count") &&
        required_unsigned(request, "row_count") != column.row_count)
      invalid("row_count does not match values length");
    column.missing.assign((column.row_count + 7) / 8, 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (values[index].is_null()) {
        column.missing[index / 8] |= static_cast<std::uint8_t>(1u << (index % 8));
      } else {
        append_json_value(column.values, column.dtype, values[index]);
      }
    }
  } else {
    if (!request.contains("values_le_hex") ||
        !request.at("values_le_hex").is_string() ||
        !request.contains("missing_bitmap_hex") ||
        !request.at("missing_bitmap_hex").is_string())
      invalid("values_le_hex channel requires string values_le_hex and missing_bitmap_hex");
    column.row_count = required_unsigned(request, "row_count");
    column.values = from_hex(request.at("values_le_hex").get<std::string>(),
                             "values_le_hex");
    column.missing = from_hex(
        request.at("missing_bitmap_hex").get<std::string>(),
        "missing_bitmap_hex");
  }
  uc::validate_column(column);
  return column;
}

Json method_json(const uc::MethodInfo& method) {
  return Json{{"name", method.name},
              {"display_name", method.display_name},
              {"kind", method.kind},
              {"aliases", method.aliases},
              {"dtypes", method.dtypes},
              {"enabled", method.enabled},
              {"available", method.available},
              {"reason", method.reason},
              {"contract", method.contract}};
}

Json inventory(const uc::Registry& registry) {
  Json methods = Json::array();
  std::uint64_t enabled = 0;
  std::uint64_t available = 0;
  for (const auto& method : registry.methods()) {
    methods.push_back(method_json(method));
    enabled += method.enabled;
    available += method.available;
  }
  return Json{{"status", "ok"},
              {"command", "inventory"},
              {"method_count", methods.size()},
              {"enabled_count", enabled},
              {"available_count", available},
              {"tree_contract", registry.tree_contract()},
              {"methods", std::move(methods)}};
}

Json decompressed_column_response(uc::Column column) {
  return Json{{"status", "ok"},
              {"command", "decompress"},
              {"dtype", column.dtype},
              {"row_count", column.row_count},
              {"values_le_hex", to_hex(column.values)},
              {"missing_bitmap_hex", to_hex(column.missing)}};
}

Json decompress_response(const uc::Registry& registry,
                         std::span<const std::uint8_t> frame) {
  return decompressed_column_response(registry.decompress(frame));
}

uc::MeasurementOptions measurement_options(const Json& request) {
  uc::MeasurementOptions options;
  if (request.contains("warmups"))
    options.warmups = static_cast<std::uint32_t>(unsigned_integer(
        request.at("warmups"), "warmups", std::numeric_limits<std::uint32_t>::max()));
  if (request.contains("repeats"))
    options.repeats = static_cast<std::uint32_t>(unsigned_integer(
        request.at("repeats"), "repeats", std::numeric_limits<std::uint32_t>::max()));
  if (options.repeats == 0) invalid("repeats must be at least 1");
  return options;
}

Json result_response(const uc::Column& column, const uc::CompressionResult& result,
                     const Json& request, std::string command = "compress") {
  Json response{{"status", "ok"},
                {"command", std::move(command)},
                {"method", result.method},
                {"dtype", column.dtype},
                {"row_count", column.row_count},
                {"original_numeric_bytes", result.original_numeric_bytes},
                {"compressed_bytes", result.compressed_bytes},
                {"compressed_sha256", sha256(result.compressed_data)},
                {"compression_ratio",
                 result.compression_ratio ? Json(*result.compression_ratio)
                                          : Json(nullptr)},
                {"compression_time_ns", result.compression_time_ns},
                {"compression_time_ms", result.compression_time_ns / 1'000'000.0},
                {"decompression_time_ns", result.decompression_time_ns},
                {"decompression_time_ms",
                 result.decompression_time_ns / 1'000'000.0},
                {"compression_samples_ns", result.compression_samples_ns},
                {"decompression_samples_ns", result.decompression_samples_ns},
                {"bit_exact", result.bit_exact},
                {"execution_audit", result.execution_audit}};
  if (request.contains("output_path")) {
    if (!request.at("output_path").is_string()) invalid("output_path must be a string");
    const fs::path path = request.at("output_path").get<std::string>();
    write_bytes(path, result.compressed_data);
    response["output_path"] = path.string();
  } else {
    response["compressed_base64"] = base64_encode(result.compressed_data);
  }
  return response;
}

Json compress_response(const uc::Registry& registry, const Json& request) {
  auto column = parse_column(request);
  const bool has_method = request.contains("method");
  const bool has_tree = request.contains("tree");
  if (has_method == has_tree) invalid("compress requires exactly one of method or tree");
  const auto options = measurement_options(request);
  const auto result = has_tree
      ? registry.compress_tree(column, request.at("tree"), options)
      : registry.compress(column, required_string(request, "method"), options);
  return result_response(column, result, request);
}

// Forward declaration: model/split loading is outside the harness trial clock.
Json read_json(const fs::path& path);

Json select_response(const uc::Registry& registry, const Json& request, bool model_mode) {
  auto column = parse_column(request);
  auto options = uc::selection_options_from_json(request.value("selection", Json::object()));
  uc::SelectionResult result;
  Json recommendation = nullptr;
  if (model_mode) {
    const auto model_path = required_string(request, "model_path");
    const auto split_path = required_string(request, "split_path");
    using Service = uc::PreparedRecommender;
    struct LoadedService { std::shared_ptr<Service> service; std::string model_sha256; };
    static std::map<std::pair<std::string, std::string>, LoadedService> services;
    const auto key = std::make_pair(model_path, split_path);
    const auto setup_start = std::chrono::steady_clock::now();
    bool reused = services.contains(key);
    if (!reused) {
      if (services.size() >= 16) invalid("at most 16 prepared model/split pairs per process");
      const auto model_bytes = read_bytes(model_path);
      const auto split_bytes = read_bytes(split_path);
      auto model = Json::parse(model_bytes.begin(), model_bytes.end());
      auto split = Json::parse(split_bytes.begin(), split_bytes.end());
      services.emplace(key, LoadedService{std::make_shared<Service>(model, split,
          sha256(split_bytes)), sha256(model_bytes)});
    }
    const auto setup_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - setup_start).count();
    auto recommended = services.at(key).service->recommend(registry, column, options);
    result = std::move(recommended.selection);
    recommendation = {{"prediction", std::move(recommended.prediction)},
        {"provenance", std::move(recommended.provenance)},
        {"prediction_wall_time_ns", recommended.prediction_wall_time_ns},
        {"total_wall_time_ns", recommended.total_wall_time_ns},
        {"service_setup_wall_time_ns", setup_ns}, {"setup_reused", reused},
        {"model_sha256", services.at(key).model_sha256}};
  } else {
    result = uc::select(registry, column, options);
  }
  Json response = uc::selection_report(result);
  response["command"] = model_mode ? "recommend_select" : "select";
  response["dtype"] = column.dtype;
  response["row_count"] = column.row_count;
  if (model_mode) response["recommendation"] = std::move(recommendation);
  if (result.selected_result) {
    response["selected"] = result_response(column, *result.selected_result, request, "selected");
    response["status"] = "ok";
  } else {
    if (request.contains("output_path"))
      response["output_committed"] = false;
    response["status"] = "no_feasible_candidate";
  }
  return response;
}

Json dispatch(const uc::Registry& registry, const Json& request) {
  if (!request.is_object()) invalid("request must be a JSON object");
  const auto command = request.value("command", "compress");
  if (!request.contains("command") || !request.at("command").is_string()) {
    if (request.contains("command")) invalid("command must be a string");
  }
  if (command == "compress") return compress_response(registry, request);
  if (command == "select" || command == "recommend_select")
    return select_response(registry, request, command == "recommend_select");
  if (command == "tree_contract")
    return Json{{"status", "ok"}, {"contract", registry.tree_contract()}};
  if (command == "decompress") {
    const bool has_path = request.contains("frame_path");
    const bool has_base64 = request.contains("compressed_base64");
    if (has_path == has_base64)
      invalid("decompress requires exactly one of frame_path or compressed_base64");
    if (has_path) {
      if (!request.at("frame_path").is_string()) invalid("frame_path must be a string");
      const auto frame = read_bytes(request.at("frame_path").get<std::string>());
      return decompress_response(registry, frame);
    }
    if (!request.at("compressed_base64").is_string())
      invalid("compressed_base64 must be a string");
    const auto frame = base64_decode(request.at("compressed_base64").get<std::string>());
    return decompress_response(registry, frame);
  }
  if (command == "inventory") return inventory(registry);
  throw uc::Error("unknown_command", "unknown command " + command);
}

Json error_json(std::string code, std::string message) {
  return Json{{"status", "error"},
              {"error", Json{{"code", std::move(code)},
                             {"message", std::move(message)}}}};
}

Json handle_line(const uc::Registry& registry, std::string_view line) {
  try {
    return dispatch(registry, Json::parse(line));
  } catch (const uc::Error& error) {
    return error_json(error.code(), error.what());
  } catch (const Json::exception& error) {
    return error_json("invalid_request", error.what());
  } catch (const std::invalid_argument& error) {
    return error_json("invalid_request", error.what());
  } catch (const std::exception& error) {
    return error_json("internal_error", error.what());
  }
}

Json read_json(const fs::path& path) {
  std::ifstream input(path);
  if (!input) throw uc::Error("io_error", "cannot read " + path.string());
  try {
    return Json::parse(input);
  } catch (const Json::exception& error) {
    throw uc::Error("invalid_request", error.what());
  }
}

void write_inventory_markdown(const uc::Registry& registry) {
  std::cout
      << "# Unified compression method inventory\n\n"
      << "Every enabled method accepts the same `Column` envelope: `dtype`, "
         "`row_count`, compact little-endian valid numeric bytes, and a 1=NULL "
         "bitmap. Compression returns a self-contained UCC1 binary frame plus "
         "complete-frame bytes / valid numeric bytes (lower is better), encode "
         "time, decode time, and a bit-exact result. A null ratio means that the "
         "column contains no valid numeric bytes.\n\n"
      << "Disabled or unavailable rows remain visible for provenance and are "
         "rejected before codec execution. Per-method restrictions appear in "
         "the reason and limitations columns; the final column summarizes the "
         "common input and output.\n\n"
      << "The JSON form of `inventory` is the authoritative machine-readable "
         "contract. This table deliberately summarizes it for human review.\n\n"
      << "| name | display name | kind | aliases | input dtypes | status | reason | limitations | input/output contract |\n"
         "|---|---|---|---|---|---|---|---|---|\n";
  for (const auto& method : registry.methods()) {
    std::string dtypes;
    for (std::size_t index = 0; index < method.dtypes.size(); ++index) {
      if (index) dtypes += ", ";
      dtypes += method.dtypes[index];
    }
    std::string aliases;
    for (std::size_t index = 0; index < method.aliases.size(); ++index) {
      if (index) aliases += ", ";
      aliases += method.aliases[index];
    }
    auto clean = [](std::string value) {
      std::replace(value.begin(), value.end(), '|', '/');
      std::replace(value.begin(), value.end(), '\n', ' ');
      std::replace(value.begin(), value.end(), '\r', ' ');
      return value;
    };
    const auto status = method.enabled && method.available
                            ? "enabled"
                            : method.enabled
                                  ? "unavailable"
                                  : method.available ? "disabled"
                                                     : "disabled / unavailable";
    std::string limitations;
    if (method.contract.contains("limitations") &&
        method.contract.at("limitations").is_string())
      limitations = method.contract.at("limitations").get<std::string>();
    std::string contract;
    if (method.kind == "algorithm") {
      auto configuration = std::string("catalog-fixed");
      if (method.contract.contains("configuration") &&
          method.contract.at("configuration").is_string())
        configuration = method.contract.at("configuration").get<std::string>();
      contract =
          "Column(dtype, rows, compact values, null bitmap) → complete UCC1 "
          "binary; fixed configuration: " +
          configuration;
    } else {
      auto scalar = [&](const char* key, std::string fallback) {
        if (!method.contract.contains(key)) return fallback;
        const auto& value = method.contract.at(key);
        if (value.is_string()) return value.get<std::string>();
        if (value.is_number_integer() || value.is_number_unsigned())
          return value.dump();
        return fallback;
      };
      contract = "Column(dtype, rows, compact values, null bitmap) → in-place "
                 "adapter (kind=" +
                 scalar("initial_kind", "unknown") + ", width=" +
                 scalar("initial_width", "unknown") + ", view=" +
                 scalar("input_view", "unknown") +
                 "; no numeric conversion) → complete UCC1/NGF1 frame "
                 "(all original recipe ports preserved)";
    }
    std::cout << "| " << clean(method.name) << " | " << clean(method.display_name)
              << " | " << clean(method.kind) << " | " << clean(aliases) << " | "
              << clean(dtypes) << " | " << status << " | "
              << clean(method.reason) << " | " << clean(limitations) << " | "
              << clean(contract) << " |\n";
  }
}

struct Arguments {
  std::string command = "serve";
  std::string operand;
  std::string operator_catalog;
  bool markdown = false;
};

Arguments arguments(int argc, char** argv) {
  Arguments result;
  bool command_seen = false;
  for (int index = 1; index < argc; ++index) {
    const std::string value = argv[index];
    if (value == "--operator-catalog") {
      if (++index == argc) invalid("--operator-catalog requires a path");
      result.operator_catalog = argv[index];
    } else if (value == "--markdown") {
      result.markdown = true;
    } else if (!command_seen) {
      result.command = value;
      command_seen = true;
    } else if (result.operand.empty()) {
      result.operand = value;
    } else {
      invalid("too many command-line arguments");
    }
  }
  if (result.command != "serve" && result.command != "inventory" &&
      result.command != "compress" && result.command != "decompress" &&
      result.command != "select" && result.command != "recommend_select" &&
      result.command != "tree_contract")
    throw uc::Error("unknown_command", "unknown command " + result.command);
  if ((result.command == "compress" || result.command == "decompress" ||
       result.command == "select" || result.command == "recommend_select") !=
      !result.operand.empty())
    invalid(result.command == "compress"
                ? "usage: compress REQUEST.json"
                : result.command == "decompress"
                      ? "usage: decompress FRAME_PATH"
                      : "serve and inventory take no operand");
  if (result.markdown && result.command != "inventory")
    invalid("--markdown is only valid with inventory");
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = arguments(argc, argv);
    // The standalone decoder is deliberately catalog-free: the complete UCC1
    // frame contains its immutable method binding and must support cold decode.
    if (args.command == "decompress") {
      const auto frame = read_bytes(args.operand);
      std::cout << decompressed_column_response(uc::decompress_column(frame)).dump()
                << '\n';
      return 0;
    }
    const uc::Registry registry({args.operator_catalog});
    if (args.command == "tree_contract") {
      std::cout << registry.tree_contract().dump() << '\n';
      return 0;
    }
    if (args.command == "inventory") {
      if (args.markdown)
        write_inventory_markdown(registry);
      else
        std::cout << inventory(registry).dump() << '\n';
      return 0;
    }
    if (args.command == "compress" || args.command == "select" ||
        args.command == "recommend_select") {
      auto request = read_json(args.operand);
      if (request.contains("command") && request.at("command") != args.command)
        invalid("request command differs from CLI command");
      request["command"] = args.command;
      std::cout << dispatch(registry, request).dump() << '\n';
      return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
      std::cout << handle_line(registry, line).dump() << '\n';
      std::cout.flush();
    }
    if (!std::cin.eof())
      throw uc::Error("io_error", "failed while reading JSONL input");
    return 0;
  } catch (const uc::Error& error) {
    std::cout << error_json(error.code(), error.what()).dump() << '\n';
    return 1;
  } catch (const Json::exception& error) {
    std::cout << error_json("invalid_request", error.what()).dump() << '\n';
    return 1;
  } catch (const std::invalid_argument& error) {
    std::cout << error_json("invalid_request", error.what()).dump() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cout << error_json("internal_error", error.what()).dump() << '\n';
    return 1;
  }
}
