#include "pipeline.hpp"
#include "stage_dispatch.hpp"
#include "test_fixture.hpp"
#include <bit>
#include <iostream>

namespace ngpipe = wc::pipeline_v1;
namespace graph = native_graph_v1;
namespace np = weighted_grpo::native_policy_v1;
using namespace pipeline_test;

namespace {
void model_fixture(const std::filesystem::path& path, const std::string& registry) {
  Bytes body;
  text(body, np::checkpoint_schema); text(body, std::string(64, 'a'));
  text(body, std::string(64, 'b')); text(body, registry); text(body, np::feature_schema);
  for (auto n : {ngpipe::action_count, ngpipe::stop_action, 18U, 28U, 128U, 128U}) put(body, n, 4);
  for (unsigned i = 0; i < 2; ++i) {
    put(body, 1, 4); text(body, "SYNTHETIC_SMOKE_ONLY"); text(body, "not_a_trained_model");
  }
  const std::array<std::vector<std::uint32_t>, 6> shapes{{
      {49, 128}, {128}, {128, 128}, {128}, {128, ngpipe::action_count}, {ngpipe::action_count}}};
  put(body, 6, 4);
  for (unsigned parameter = 0; parameter < shapes.size(); ++parameter) {
    const auto& shape = shapes[parameter]; put(body, shape.size(), 4);
    std::uint64_t count = 1;
    for (auto dim : shape) { put(body, dim, 4); count *= dim; }
    put(body, count, 8);
    for (std::uint64_t i = 0; i < count; ++i) {
      // Known selected actions have higher logits than STOP. All other action
      // probabilities are controlled by the tested explicit mask.
      double value = 0;
      if (parameter == 5 && i == 101) value = 8;
      if (parameter == 5 && i == 296) value = 7;
      if (parameter == 5 && i == 100) value = 6;
      put(body, std::bit_cast<std::uint64_t>(value), 8);
    }
  }
  Bytes all{'W','G','P','N','A','T','1',0}; put(all, body.size(), 8);
  all.insert(all.end(), body.begin(), body.end()); const auto digest = np::sha256(all);
  all.insert(all.end(), digest.begin(), digest.end()); write(path, all);
}
template<class F> void rejected(F fn, const std::string& expected) {
  bool thrown = false;
  try { fn(); } catch (const std::exception& e) {
    thrown = true; check(std::string(e.what()).find(expected) != std::string::npos,
                         "rejection had unexpected reason");
  }
  check(thrown, "invalid pipeline input was accepted");
}
Bytes lanes(std::initializer_list<std::uint64_t> words) {
  Bytes out; for (const auto word : words) put(out, word, 8); return out;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 2, "usage: native_pipeline_v1_smoke FIXTURE_DIRECTORY");
    const std::filesystem::path dir(argv[1]); std::filesystem::create_directories(dir);
    const auto model = dir / "synthetic.wgpn";
    model_fixture(model, ngpipe::registry_fingerprint);
    const auto pipeline = ngpipe::Pipeline::load(model);
    auto dispatcher = graph::make_default_dispatcher();
    ngpipe::Request request;
    request.roots = {
        {{graph::Kind::Numeric, 8, lanes({0, 1, 3, 7, 15, 31, 63, 127})}, true, "int64"},
        {{graph::Kind::Numeric, 8, lanes({5, 5, 5, 5, 5, 5, 5, 5})}, true, "int64"},
        {{graph::Kind::Serial, 1, Bytes{0, 0, 1, 0, 0, 0, 0, 1}}, false, "bytes"}};
    request.enabled_actions.assign(ngpipe::action_count, 0);
    for (auto id : {0, 101, 296}) request.enabled_actions[id] = 1;
    const auto result = pipeline.benchmark(request, *dispatcher, 1, 2);
    check(result.samples.size() == 2, "wrong pipeline benchmark sample count");
    check(result.last.selected_action_ids == std::vector<std::uint32_t>{101, 101, 0},
          "policy ranking or active-flow slot mapping is wrong");
    check(result.last.original_numeric_bytes == 128, "auxiliary mask changed denominator");
    check(result.last.compression_ratio() == double(result.last.frame.size()) / 128.0,
          "wrong complete-frame compression ratio");
    for (const auto& sample : result.samples) {
      check(sample.compression_ns >= sample.feature_ns + sample.policy_ns +
                sample.selection_graph_ns + sample.final_encode_ns,
            "overlapping subtimers exceed total compression time");
      check(sample.decompression_ns > 0, "decode time missing");
    }
    write(dir / "multiple_roots.ngf1", result.last.frame);
    write(dir / "multiple_roots.expected", expected(request.roots));

    auto parity = request;
    wc::features_v1::State initial;
    initial.original_numeric_bytes = 128; initial.action_count = ngpipe::action_count;
    std::vector<wc::features_v1::FlowView> views;
    for (const auto& root : request.roots) {
      wc::features_v1::FlowView view{root.stream.kind == graph::Kind::Numeric ? "numeric" : "serial",
                                   root.stream.width, root.stream.data, root.logical_view};
      views.push_back(view); initial.flows.push_back({view, 0, 0});
    }
    parity.expected_initial_features = wc::features_v1::observation(views, initial, parity.weights);
    (void)pipeline.compress(parity, *dispatcher);
    (*parity.expected_initial_features)[0] += 1;
    rejected([&] { pipeline.compress(parity, *dispatcher); }, "49-D features disagree");
    auto invalid = request; invalid.enabled_actions[142] = 1;
    rejected([&] { pipeline.compress(invalid, *dispatcher); }, "UNSUPPORTED_NATIVE_POLICY_ACTION: 142");
    invalid = request; invalid.enabled_actions[0] = 0;
    rejected([&] { pipeline.compress(invalid, *dispatcher); }, "STOP permission");
    invalid = request; invalid.weights = {1.0, 1.0, 0.0};
    rejected([&] { pipeline.compress(invalid, *dispatcher); }, "weights");
    invalid = request; for (auto& root : invalid.roots) root.original_numeric_payload = false;
    rejected([&] { pipeline.compress(invalid, *dispatcher); }, "NO_ORIGINAL_NUMERIC_PAYLOAD");
    const ngpipe::Result empty;
    check(!empty.compression_ratio() && !empty.compression_ns_per_byte() &&
              !empty.decompression_ns_per_byte(), "B_raw=0 must normalize to N/A");

    ngpipe::Request floats;
    floats.roots = {{{graph::Kind::Numeric, 8, lanes({0, 0x8000000000000000ULL,
        0x7ff8000000000042ULL, 0xfff8000000001234ULL, 0x7ff0000000000000ULL,
        0xfff0000000000000ULL, 1, 0x0010000000000000ULL})}, true, "float64"}};
    floats.max_policy_steps = 1; floats.enabled_actions.assign(ngpipe::action_count, 0);
    floats.enabled_actions[0] = floats.enabled_actions[100] = 1;
    const auto floating = pipeline.benchmark(floats, *dispatcher, 0, 1);
    check(floating.last.selected_action_ids == std::vector<std::uint32_t>{100, 0},
          "max-step STOP was not respected");
    write(dir / "float_bits.ngf1", floating.last.frame);
    write(dir / "float_bits.expected", expected(floats.roots));
    model_fixture(dir / "wrong_registry.wgpn", "different-registry");
    rejected([&] { ngpipe::Pipeline::load(dir / "wrong_registry.wgpn"); }, "identity mismatch");
    // Cold decoding tests receive only NGF1 frames, never a policy file.
    std::filesystem::remove(model); std::filesystem::remove(dir / "wrong_registry.wgpn");
    std::cout << "native_pipeline_v1 finite-mapping smoke passed; complete_frame_bytes="
              << result.last.frame.size() << "; B_raw=" << result.last.original_numeric_bytes
              << "; E_total_ns=" << result.last.timing.compression_ns
              << "; D_total_ns=" << result.last.timing.decompression_ns << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
