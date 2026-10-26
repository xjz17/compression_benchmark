#include "stage_dispatch.hpp"

#include <dlfcn.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace native_graph_v1;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "expected PAQ metadata rejection");
}

Bytes dump(const nlohmann::json& value) {
  const auto text = value.dump();
  return Bytes(text.begin(), text.end());
}

Bytes encode_v1(const Bytes& raw) {
#ifndef PAQ_PLUGIN_PATH
#error "PAQ_PLUGIN_PATH must name the freshly built isolated PAQ DSO"
#endif
  using Encode = int (*)(const std::uint8_t*, std::size_t, unsigned,
                         std::uint8_t**, std::size_t*, char*, std::size_t);
  using Free = void (*)(void*);
  void* handle = dlopen(PAQ_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL);
  const char* open_error = handle ? nullptr : dlerror();
  require(handle != nullptr,
          std::string("v1 plugin load failed: ") +
              (open_error ? open_error : "unknown"));
  dlerror();
  auto encode = reinterpret_cast<Encode>(
      dlsym(handle, "native_graph_paq_encode"));
  const char* encode_error = dlerror();
  require(encode_error == nullptr && encode != nullptr,
          "v1 encode ABI is missing");
  dlerror();
  auto free_output = reinterpret_cast<Free>(
      dlsym(handle, "native_graph_paq_free"));
  const char* free_error = dlerror();
  require(free_error == nullptr && free_output != nullptr,
          "PAQ free ABI is missing");

  std::uint8_t* output = nullptr;
  std::size_t output_size = 0;
  std::array<char, 512> error{};
  const int status = encode(raw.data(), raw.size(), 0, &output, &output_size,
                            error.data(), error.size());
  require(status == 0 && output != nullptr,
          std::string("v1 encode failed: ") + error.data());
  Bytes result(output, output + output_size);
  free_output(output);
  require(dlclose(handle) == 0, "v1 plugin unload failed");
  return result;
}
}  // namespace

int main() try {
  const StageRequest request{"op.prediction_probability.paq.provider_v1",
                             "bytes", {{"paq_level", "0"}}};
  Bytes raw(257);
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<std::uint8_t>(i * 37 + 11);
  const Stream input{Kind::Serial, 1, raw};

  auto encoder = make_default_dispatcher();
  const auto encoded = encoder->forward(request, {input});
  require(encoded.outputs.size() == 1, "PAQ v2 output count mismatch");
  const auto metadata = nlohmann::json::parse(encoded.metadata.begin(),
                                               encoded.metadata.end());
  require(metadata.at("side").at("raw_block_header_version") == 2,
          "new PAQ frame did not bind raw-block header v2");
  require(metadata.at("side").at("raw_bytes") == raw.size(),
          "new PAQ frame raw byte count mismatch");

  auto cold = make_default_dispatcher();
  const auto decoded =
      cold->inverse(request, encoded.outputs, encoded.metadata);
  require(decoded.size() == 1 && decoded[0] == input,
          "PAQ v2 stage roundtrip mismatch");

  for (const int bad_version : {0, 3, 255}) {
    auto bad = metadata;
    bad["side"]["raw_block_header_version"] = bad_version;
    const auto bad_bytes = dump(bad);
    rejects([&] { cold->inverse(request, encoded.outputs, bad_bytes); });
  }

  // Old frames used the v1 ABI on an already cmix-preprocessed block.  A
  // missing version field is the legacy on-wire form and must select v1.
  Bytes legacy_raw{0, 0, 0, 0, 32};
  legacy_raw.resize(37, 0x41);
  const auto legacy_template =
      encoder->forward(request, {{Kind::Serial, 1, legacy_raw}});
  auto legacy_metadata = nlohmann::json::parse(
      legacy_template.metadata.begin(), legacy_template.metadata.end());
  auto legacy_outputs = legacy_template.outputs;
  legacy_outputs[0].data = encode_v1(legacy_raw);
  legacy_metadata["outputs"][0]["bytes"] = legacy_outputs[0].data.size();
  legacy_metadata["side"].erase("raw_block_header_version");
  const auto legacy_decoded =
      cold->inverse(request, legacy_outputs, dump(legacy_metadata));
  require(legacy_decoded.size() == 1 &&
              legacy_decoded[0].data == legacy_raw,
          "legacy PAQ v1 stage frame did not decode");

  std::cout << "PASS paq_stage_v2_metadata\n"
               "PASS paq_stage_v2_roundtrip\n"
               "PASS paq_stage_unknown_versions_rejected=3\n"
               "PASS paq_stage_v1_missing_version_compatibility\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL " << error.what() << '\n';
  return 1;
}
