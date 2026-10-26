#include "native_lz_action_stages.hpp"
#include "legacy_numeric_action_stages.hpp"
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

using native_lz_action::Bytes;
namespace {
unsigned checks = 0;
void check(bool ok) { ++checks; if (!ok) throw std::runtime_error("test assertion failed"); }
void rejects(const std::function<void()>& fn) {
  bool thrown = false;
  try { fn(); } catch (const std::exception&) { thrown = true; }
  check(thrown);
}
void word(Bytes& raw,std::uint64_t value) { for (unsigned i = 0; i < 8; ++i) raw.push_back(value>>(8*i)); }
void real(Bytes& raw,double value) { std::uint64_t bits; std::memcpy(&bits,&value,8); word(raw,bits); }
constexpr const char* scaled[] = {
  "op.scaling.legacy_exact_decimal.python_v1", "op.block_transform.int64_lo_hi32.python_v1",
  "op.lz_match.native_lz_limb_batch.aligned8_greedy.python_v1", "op.match_token_format.native_lz_limb_batch.ctrl8.python_v1",
  "op.container_frame.legacy_numeric_limbs.python_v1"};
constexpr const char* alp[] = {
  "op.scaling.alp_legacy_fe_exceptions.python_v1", "op.block_transform.int64_lo_hi32.python_v1",
  "op.lz_match.native_lz_limb_batch.aligned8_greedy.python_v1", "op.match_token_format.native_lz_limb_batch.ctrl8.python_v1",
  "op.container_frame.alp2_legacy.python_v1"};
void legacy_chain(const Bytes& raw,const char* const* actions,const std::string& kind,int point) {
  Bytes current = raw;
  for (unsigned i = 0; i < 5; ++i) {
    const auto next = legacy_numeric_action::forward(actions[i],current,kind,point);
    check(legacy_numeric_action::inverse(actions[i],next,raw.size()/8) == current);
    current = next;
  }
  for (unsigned i = 5; i > 0; --i) current = legacy_numeric_action::inverse(actions[i-1],current,raw.size()/8);
  check(current == raw);
}
}
int main() {
  try {
    std::mt19937_64 random(1729);
    std::vector<Bytes> fixtures{Bytes{},Bytes{'a'},Bytes(1024)};
    Bytes repeated(4096), noise(2057);
    for (std::size_t i = 0; i < repeated.size(); ++i) repeated[i] = i%64;
    for (auto& b : noise) b = random();
    fixtures.push_back(repeated); fixtures.push_back(noise);
    for (const auto& raw : fixtures) {
      for (unsigned window : {3072,16384,19456,65535}) {
        for (unsigned minimum : {3,4,5}) {
          for (unsigned maximum : {18,32,35,66,175,255,258}) {
            const auto tokens = native_lz_action::parse(raw,window,minimum,maximum);
            const auto serialized = native_lz_action::serialize_tokens(tokens);
            check(native_lz_action::expand(native_lz_action::deserialize_tokens(serialized),raw.size()) == raw);
            const auto wire = native_lz_action::pack(tokens,minimum);
            check(native_lz_action::expand(native_lz_action::unpack(wire,minimum,raw.size()),raw.size()) == raw);
          }
        }
      }
    }
    Bytes integers;
    for (auto v : {std::uint64_t(0),std::uint64_t(-1),std::uint64_t(1)<<63,(std::uint64_t(1)<<63)-1,(std::uint64_t(1)<<53)+1}) word(integers,v);
    legacy_chain(integers,scaled,"int64",0); legacy_chain({},scaled,"int64",0);
    for (unsigned n : {0,1,2,1023,1024,1025,2051}) {
      Bytes floating;
      for (unsigned i = 0; i < n; ++i) real(floating,static_cast<double>(i%137)/4.);
      legacy_chain(floating,alp,"float64",2);
      if (n) legacy_chain(floating,scaled,"float64",2);
    }
    Bytes special;
    for (auto u : {std::uint64_t(0),std::uint64_t(1)<<63,std::uint64_t(1),std::uint64_t(0x7ff0000000000000),std::uint64_t(0xfff0000000000000),std::uint64_t(0x7ff8000000001234)}) word(special,u);
    legacy_chain(special,alp,"float64",15);
    Bytes negative_zero; word(negative_zero,std::uint64_t(1)<<63);
    rejects([&] { legacy_numeric_action::forward(scaled[0],negative_zero,"float64",2); });
    rejects([&] { legacy_numeric_action::forward(scaled[0],{},"float64",2); });
    Bytes exceptions; real(exceptions,1.25); real(exceptions,2.5); word(exceptions,std::uint64_t(1)<<63);
    legacy_chain(exceptions,alp,"float64",2);
    for (unsigned i = 0; i < 32; ++i) {
      Bytes words;
      for (unsigned j = 0; j < 33; ++j) word(words,random());
      legacy_chain(words,scaled,"int64",0); legacy_chain(words,alp,"float64",15);
    }
    const std::string window = "op.match_window_model.native_lz.window65535";
    const std::string match = "op.lz_match.native_lz.aligned8.min3.max258";
    const std::string parse = "op.match_parse_policy.native_lz.aligned8_greedy";
    const std::string pack = "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3";
    const auto w = native_lz_action::forward(window,repeated);
    const auto m = native_lz_action::forward(match,w);
    const auto p = native_lz_action::forward(parse,m);
    const auto wire = native_lz_action::forward(pack,p); const auto context = native_lz_action::context(p);
    check(native_lz_action::inverse(pack,wire,&context) == p);
    rejects([&] { native_lz_action::inverse(pack,wire); });
    rejects([&] { native_lz_action::forward(parse,w); });
    rejects([&] { native_lz_action::forward(pack,m); });
    rejects([&] { native_lz_action::inverse("op.match_window_model.native_lz.window16384",w); });
    rejects([&] { native_lz_action::unpack({1,0,0,0},3,3); });
    rejects([&] { native_lz_action::unpack({0,'a',0},3,1); });
    rejects([&] { native_lz_action::deserialize_tokens({1,0,0,0,0,0,0,0,2,0}); });
    rejects([&] { legacy_numeric_action::inverse(alp[4],{'A','L','P','2',1,0,0,0,0,1,0,8,0,0,0},1); });
    rejects([&] { legacy_numeric_action::forward(scaled[1],{'L','N','S','1','{','}'},"bytes"); });
    std::cout << "PASS " << checks << " checks\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
