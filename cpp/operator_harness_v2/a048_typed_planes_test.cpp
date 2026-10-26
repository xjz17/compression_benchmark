#include "a048_typed_planes.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using native_graph_v1::Bytes;
using native_graph_v1::Kind;
using native_graph_v1::StageRequest;
using native_graph_v1::Stream;
using Json = nlohmann::json;

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void expect_rejected(const std::function<void()>& operation,
                     const std::string& message) {
  try {
    operation();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

Bytes words_bytes(const std::vector<std::uint64_t>& words) {
  Bytes result(words.size() * sizeof(std::uint64_t));
  if (!result.empty())
    std::memcpy(result.data(), words.data(), result.size());
  return result;
}

StageRequest request(unsigned bit_width) {
  return {std::string(operator_harness_v2::kA048TypedPlaneAction), "uint64",
          {{"bit_width", std::to_string(bit_width)}}};
}

std::vector<std::uint64_t> values_for(unsigned bit_width) {
  if (bit_width == 0) return {0, 0, 0, 0, 0};
  const auto maximum = bit_width == 64
                           ? ~std::uint64_t{0}
                           : (std::uint64_t{1} << bit_width) - 1;
  return {0,
          1,
          maximum,
          maximum >> 1,
          std::uint64_t{0xa55aa55af00ff00f} & maximum};
}

Bytes expected_plane(const std::vector<std::uint64_t>& values,
                     unsigned bit_width, std::size_t plane) {
  const auto planes = static_cast<std::size_t>((bit_width + 7) / 8);
  const auto padding = static_cast<unsigned>(planes * 8) - bit_width;
  const auto shift = static_cast<unsigned>((planes - 1 - plane) * 8);
  Bytes result;
  result.reserve(values.size());
  for (const auto value : values) {
    const auto aligned = padding == 0 ? value : value << padding;
    result.push_back(static_cast<std::uint8_t>(aligned >> shift));
  }
  return result;
}

Json metadata_json(const native_graph_v1::StageResult& encoded) {
  const std::string text(encoded.metadata.begin(), encoded.metadata.end());
  return Json::parse(text);
}

Bytes metadata_bytes(const Json& metadata) {
  const auto text = metadata.dump();
  return {text.begin(), text.end()};
}

}  // namespace

int main() {
  using operator_harness_v2::forward_a048_typed_planes;
  using operator_harness_v2::inverse_a048_typed_planes;

  for (unsigned bit_width = 0; bit_width <= 64; ++bit_width) {
    const auto values = values_for(bit_width);
    const Stream input{Kind::Numeric, 8, words_bytes(values)};
    const auto stage = request(bit_width);
    const auto encoded = forward_a048_typed_planes(stage, {input});
    const auto plane_count = static_cast<std::size_t>((bit_width + 7) / 8);
    const auto physical_count = std::max<std::size_t>(plane_count, 1);
    expect(encoded.outputs.size() == physical_count,
           "wrong physical plane count for width " +
               std::to_string(bit_width));
    if (plane_count == 0) {
      expect(encoded.outputs.front() == Stream{Kind::Serial, 1, {}},
             "width zero did not use the empty physical carrier");
    } else {
      for (std::size_t plane = 0; plane < plane_count; ++plane) {
        expect(encoded.outputs[plane].kind == Kind::Serial &&
                   encoded.outputs[plane].width == 1 &&
                   encoded.outputs[plane].data ==
                       expected_plane(values, bit_width, plane),
               "plane bytes differ at width " + std::to_string(bit_width));
      }
    }
    const auto metadata = metadata_json(encoded);
    expect(metadata.at("count") == values.size() &&
               metadata.at("bit_width") == bit_width &&
               metadata.at("semantic_plane_count") == plane_count &&
               metadata.at("zero_plane_carrier") == (plane_count == 0),
           "shape metadata differs at width " +
               std::to_string(bit_width));
    const auto decoded =
        inverse_a048_typed_planes(stage, encoded.outputs, encoded.metadata);
    expect(decoded.size() == 1 && decoded.front() == input,
           "inverse differs at width " + std::to_string(bit_width));
  }

  auto bad_width = request(64);
  bad_width.parameters.at("bit_width") = "65";
  expect_rejected(
      [&] { (void)forward_a048_typed_planes(
                bad_width, {{Kind::Numeric, 8, words_bytes({0})}}); },
      "bit_width=65 was accepted");
  bad_width.parameters.at("bit_width") = "01";
  expect_rejected(
      [&] { (void)forward_a048_typed_planes(
                bad_width, {{Kind::Numeric, 8, words_bytes({0})}}); },
      "noncanonical bit_width was accepted");

  const auto width9 = request(9);
  expect_rejected(
      [&] { (void)forward_a048_typed_planes(
                width9, {{Kind::Serial, 8, words_bytes({1})}}); },
      "serial input was accepted");
  expect_rejected(
      [&] { (void)forward_a048_typed_planes(
                width9, {{Kind::Numeric, 8, Bytes(7, 0)}}); },
      "misaligned input was accepted");
  expect_rejected(
      [&] { (void)forward_a048_typed_planes(
                width9,
                {{Kind::Numeric, 8, words_bytes({std::uint64_t{1} << 9})}}); },
      "value exceeding the requested width was accepted");

  const Stream input{Kind::Numeric, 8, words_bytes({0, 1, 0x1ff})};
  const auto encoded = forward_a048_typed_planes(width9, {input});
  auto missing_plane = encoded.outputs;
  missing_plane.pop_back();
  expect_rejected(
      [&] {
        (void)inverse_a048_typed_planes(width9, missing_plane,
                                        encoded.metadata);
      },
      "inverse accepted a missing plane");
  auto wrong_shape = encoded.outputs;
  wrong_shape.front().width = 2;
  expect_rejected(
      [&] {
        (void)inverse_a048_typed_planes(width9, wrong_shape,
                                        encoded.metadata);
      },
      "inverse accepted a non-byte plane");
  auto bad_padding = encoded.outputs;
  bad_padding.back().data.front() |= 1;
  expect_rejected(
      [&] {
        (void)inverse_a048_typed_planes(width9, bad_padding,
                                        encoded.metadata);
      },
      "inverse accepted nonzero padding bits");

  auto metadata = metadata_json(encoded);
  metadata["bit_width"] = 8;
  auto bytes = metadata_bytes(metadata);
  expect_rejected(
      [&] { (void)inverse_a048_typed_planes(width9, encoded.outputs, bytes); },
      "inverse accepted metadata/request width mismatch");
  metadata = metadata_json(encoded);
  metadata["extra"] = 1;
  bytes = metadata_bytes(metadata);
  expect_rejected(
      [&] { (void)inverse_a048_typed_planes(width9, encoded.outputs, bytes); },
      "inverse accepted extra metadata fields");
  metadata = metadata_json(encoded);
  metadata["count"] = -1;
  bytes = metadata_bytes(metadata);
  expect_rejected(
      [&] { (void)inverse_a048_typed_planes(width9, encoded.outputs, bytes); },
      "inverse accepted a negative count");

  std::cout << "operator_harness_v2 A048 typed-plane test passed\n";
}
