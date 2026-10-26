#include "stage_dispatch.hpp"
#include "test_fixture.hpp"
#include <iostream>

int main(int argc, char** argv) {
  try {
    using namespace pipeline_test;
    check(argc == 2, "usage: native_pipeline_v1_cold_decode FIXTURE_DIRECTORY");
    const std::filesystem::path dir(argv[1]);
    check(!std::filesystem::exists(dir / "synthetic.wgpn"), "model must be absent for cold test");
    const auto dispatcher = native_graph_v1::make_default_dispatcher();
    for (const auto name : {"multiple_roots", "float_bits"}) {
      const auto frame = read(dir / (std::string(name) + ".ngf1"));
      const auto restored = native_graph_v1::decode(frame, *dispatcher);
      // Compare originals only after the decoder has returned.
      check(expected(restored.roots) == read(dir / (std::string(name) + ".expected")),
            "independent cold decode bitwise/root-metadata mismatch");
      auto corrupt = frame; corrupt[corrupt.size() / 2] ^= 1;
      bool rejected = false;
      try { native_graph_v1::decode(corrupt, *dispatcher); }
      catch (const std::exception&) { rejected = true; }
      check(rejected, "corrupt NGF1 frame was accepted");
    }
    std::cout << "independent cold decoder passed without model or encoder state\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
