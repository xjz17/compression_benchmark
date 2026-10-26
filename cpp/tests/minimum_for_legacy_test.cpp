// Exercise the actual standalone benchmark-86 REGER3 implementation without
// changing its public CLI or linking it to the distinct production REGER codec.
#define main minimum_for_embedded_legacy_main
#include "../src/reger_legacy_cpp.cpp"
#undef main
#include <iostream>
#include <string_view>
namespace {
struct ForGolden { std::string_view mode,label,input,wire; };
constexpr ForGolden for_golden[] = {
#include "minimum_for_golden.inc"
};
std::vector<std::uint8_t> for_unhex(std::string_view s) {
  std::vector<std::uint8_t> out;const auto n=[](char c){return c<='9'?c-'0':c-'a'+10;};
  for(std::size_t i=0;i<s.size();i+=2)out.push_back((n(s[i])<<4)|n(s[i+1]));return out;
}
}
int main() {
  std::size_t checked=0;
  for(const auto& g:for_golden)if(g.mode=="reger_legacy"){
    auto raw=for_unhex(g.input),wire=for_unhex(g.wire);
    if(reger_legacy_cpp::columns_forward(raw)!=wire)throw std::runtime_error("REGER3 full wire differs from frozen pre-FOR refactor");
    const auto values=reger_legacy_cpp::read_columns(raw);
    const auto restored=reger_legacy_cpp::read_columns(reger_legacy_cpp::columns_inverse(wire));
    if(values.rows!=restored.rows || values.values!=restored.values)throw std::runtime_error("REGER3 column roundtrip failed");
    ++checked;
  }
  std::cout<<"minimum_for_legacy=PASS complete_REGER3_wires="<<checked<<"\n";
}
