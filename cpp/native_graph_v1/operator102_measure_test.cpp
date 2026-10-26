#include "operator102_measure.hpp"
#include "operator102_dispatcher.hpp"

#include <algorithm>
#include <cmath>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace native_graph_v1;
using namespace native_graph_v1::operator102_measure;

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}

int main() {
  try {
    const Bytes constant(100, 7);
    const auto h_constant = entropy_one(constant);
    require(std::abs(h_constant.h0_bits) < 1e-9 &&
                std::abs(h_constant.markov3_conditional_bits) < 1e-9,
            "constant entropy regression");

    Bytes alternating;
    for (unsigned index = 0; index < 100; ++index) alternating.push_back(index & 1U);
    const auto h_alternating = entropy_one(alternating);
    require(std::abs(h_alternating.h0_bits - 100.0) < 1e-9,
            "binary H0 regression");
    require(std::abs(h_alternating.markov3_conditional_bits) < 1e-9,
            "order-3 deterministic regression");

    const Bytes zeros{0, 0, 0, 0};
    const Bytes ones{1, 1, 1, 1};
    const auto separated = entropy_streams({std::span<const std::uint8_t>(zeros),
                                            std::span<const std::uint8_t>(ones)});
    require(separated.h0_bits == 0.0 && separated.streams == 2,
            "stream-boundary entropy regression");

    NativeStageDispatcher dispatcher;
    Stream input{Kind::Serial, 1, Bytes{4, 4, 4, 4, 5, 5, 5, 5}};
    PipelineStep step{{"op.run_length.byte_optimal.cpp_v1", "bytes", {}}};
    const auto measured = measure_pipeline(dispatcher, input, {step}, 0);
    require(measured.bitexact, "stage roundtrip regression");
    require(measured.final_representation.bytes ==
                measured.output_payload_bytes + measured.metadata_bytes + 36,
            "side metadata/physical descriptor accounting regression");
    const std::vector<Bytes> timing_auxiliaries{{0xaa, 0x55}};
    const Bytes timing_adapter{'A', 'D', 'P', '2'};
    const auto timed = benchmark_pipeline(dispatcher, input, {step},
                                          timing_auxiliaries, &timing_adapter,
                                          1, 3);
    require(timed.bitexact && timed.encode_ns.size() == 3 &&
                timed.decode_ns.size() == 3 && timed.full_encode_ns.size() == 3 &&
                timed.full_decode_ns.size() == 3,
            "W1/R3 full timing sample regression");
    for (std::size_t repeat = 0; repeat < 3; ++repeat)
      require(timed.full_encode_ns[repeat] >= timed.encode_ns[repeat] &&
                  timed.full_decode_ns[repeat] >= timed.decode_ns[repeat],
              "full O2R1 timing is smaller than kernel timing");
    const std::vector<Stream> o2_outputs{
        {Kind::Serial, 1, Bytes{1, 2, 3}},
        {Kind::Numeric, 2, Bytes{4, 5, 6, 7}}};
    const std::vector<Bytes> o2_metadata{{8, 9}, {}};
    const auto packed = pack_o2r1(o2_outputs, o2_metadata,
                                  timing_auxiliaries, &timing_adapter);
    const auto parsed = parse_o2r1(packed);
    require(parsed.outputs == o2_outputs &&
                parsed.inverse_metadata == o2_metadata &&
                parsed.auxiliaries == timing_auxiliaries &&
                parsed.adapter && parsed.adapter.value() == timing_adapter,
            "O2R1 pack/parse regression");
    bool bad_o2r1_rejected = false;
    try {
      auto corrupted = packed;
      corrupted.front() ^= 1U;
      static_cast<void>(parse_o2r1(corrupted));
    } catch (const std::exception&) {
      bad_o2r1_rejected = true;
    }
    require(bad_o2r1_rejected, "corrupt O2R1 header was accepted");

    Operator102Dispatcher bridge;
    Stream integers{Kind::Numeric, 8, Bytes(8 * 32)};
    for (std::size_t index = 0; index < integers.data.size(); ++index)
      integers.data[index] = static_cast<std::uint8_t>(index * 17U);
    const std::vector<PipelineStep> sprintz{
        {{"op.delta.sprintz.cpp_v1", "int64", {}}},
        {{"op.zigzag.sprintz.cpp_v1", "bytes", {}}},
        {{"op.bitpack_int.sprintz.cpp_v1", "bytes", {}}}};
    const auto sprintz_measurement = measure_pipeline(bridge, integers, sprintz, 2);
    require(sprintz_measurement.bitexact && sprintz_measurement.metadata_bytes != 0,
            "source-separated bridge roundtrip/metadata regression");

    const std::vector<PipelineStep> elf_plus{
        {{"build.adjacent_xor_residual_words", "raw64", {}}},
        {{"op.trailing_zero_pack.elf_plus_residual.cpp_v1", "raw64", {}}}};
    const auto elf_plus_measurement = measure_pipeline(bridge, integers, elf_plus, 1);
    require(elf_plus_measurement.bitexact &&
                elf_plus_measurement.final_outputs.size() == 1 &&
                elf_plus_measurement.final_outputs.front().kind == Kind::Serial,
            "adjacent XOR residual/Elf+ bridge regression");

    const auto double_stream = [](std::initializer_list<double> values) {
      Stream stream{Kind::Numeric, 8, Bytes(values.size() * sizeof(double))};
      std::size_t offset = 0;
      for (const auto value : values) {
        const auto word = std::bit_cast<std::uint64_t>(value);
        for (unsigned shift = 0; shift < 64; shift += 8)
          stream.data[offset++] = static_cast<std::uint8_t>(word >> shift);
      }
      return stream;
    };
    const StageRequest fallback_request{
        "op.scaling.tsfile_rle.cpp_v1", "float64",
        {{"max_point", "0"}, {"exact_raw_fallback", "true"}}};
    const auto strict_input = double_stream({1.0, -2.0});
    const auto strict = bridge.forward(fallback_request, {strict_input});
    const auto strict_info = inspect_exact_raw_fallback_metadata(strict.metadata);
    require(strict_info.wrapper && !strict_info.used_raw_fallback,
            "strict domain wrapper selector regression");
    require(bridge.inverse(fallback_request, strict.outputs, strict.metadata) ==
                std::vector<Stream>{strict_input},
            "strict domain wrapper inverse regression");
    const auto fallback_input = double_stream({std::numeric_limits<double>::infinity()});
    const auto fallback = bridge.forward(fallback_request, {fallback_input});
    const auto fallback_info = inspect_exact_raw_fallback_metadata(fallback.metadata);
    require(fallback_info.wrapper && fallback_info.used_raw_fallback &&
                fallback_info.original_bytes == fallback_input.data.size() &&
                !fallback_info.reason.empty(),
            "raw domain wrapper selector regression");
    require(bridge.inverse(fallback_request, fallback.outputs, fallback.metadata) ==
                std::vector<Stream>{fallback_input},
            "raw domain wrapper inverse regression");
    const StageRequest elf_finite_request{
        "op.eraser.elf.cpp_v1", "float64",
        {{"exact_raw_fallback", "true"}}};
    const auto elf_finite = bridge.forward(elf_finite_request, {fallback_input});
    const auto elf_finite_info =
        inspect_exact_raw_fallback_metadata(elf_finite.metadata);
    require(elf_finite_info.wrapper && elf_finite_info.used_raw_fallback &&
                elf_finite_info.reason ==
                    "Elf32 exact action requires finite float64 values" &&
                bridge.inverse(elf_finite_request, elf_finite.outputs,
                               elf_finite.metadata) ==
                    std::vector<Stream>{fallback_input},
            "Elf32 finite-domain raw fallback regression");
    const StageRequest dbp_overflow_request{
        "op.scaling.delta_binary_packed.cpp_v1", "float64",
        {{"max_point", "18"}, {"exact_raw_fallback", "true"}}};
    const auto dbp_overflow_input =
        double_stream({std::numeric_limits<double>::max()});
    const auto dbp_overflow =
        bridge.forward(dbp_overflow_request, {dbp_overflow_input});
    const auto dbp_overflow_info =
        inspect_exact_raw_fallback_metadata(dbp_overflow.metadata);
    require(dbp_overflow_info.wrapper && dbp_overflow_info.used_raw_fallback &&
                dbp_overflow_info.reason ==
                    "DBP decimal scaling exceeds int64" &&
                bridge.inverse(dbp_overflow_request, dbp_overflow.outputs,
                               dbp_overflow.metadata) ==
                    std::vector<Stream>{dbp_overflow_input},
            "DBP overflow raw fallback regression");
    bool unknown_rejected = false;
    try {
      auto unknown = fallback_request;
      unknown.action = "op.invalid.must_not_be_hidden_by_fallback";
      static_cast<void>(bridge.forward(unknown, {strict_input}));
    } catch (const std::exception&) {
      unknown_rejected = true;
    }
    require(unknown_rejected, "lossless-domain wrapper hid an unknown target failure");
    const auto nncp_available = std::any_of(
        native_stage_capabilities().begin(), native_stage_capabilities().end(),
        [](const StageCapability& capability) {
          return capability.action == "op.prediction_probability.nncp.provider_v1";
        });
    if (nncp_available) {
      Stream nncp_input{Kind::Serial, 1, Bytes(4097)};
      const StageRequest nncp_request{
          "op.prediction_probability.nncp.provider_v1", "bytes", {}};
      const auto nncp = bridge.forward(nncp_request, {nncp_input});
      require(nncp.outputs.size() == 2 && nncp.output_logical_views.size() == 2 &&
                  nncp.metadata.size() >= 4 && nncp.metadata[0] == 'N' &&
                  nncp.metadata[1] == 'N' && nncp.metadata[2] == 'C' &&
                  nncp.metadata[3] == '1',
              "NNCP 4096-byte deterministic chunk adapter regression");
      require(bridge.inverse(nncp_request, nncp.outputs, nncp.metadata) ==
                  std::vector<Stream>{nncp_input},
              "NNCP chunk adapter inverse regression");
    }
    std::cout << "operator102_measure_test PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
