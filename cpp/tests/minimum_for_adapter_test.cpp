#include "minimum_for.hpp"
#include "reference_operator_actions.hpp"
#include "ts2diff_exact_stages.hpp"
#include "bos_exact_stages.hpp"
#include "learning_codec_exact_stages.hpp"
#include "reger_codec.hpp"
#include "alp_float_codec.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include <cstring>
#include <iostream>
#include <string_view>

namespace {
using Bytes = std::vector<std::uint8_t>;
struct Golden { std::string_view mode,label,input,wire; };
constexpr Golden golden[] = {
#include "minimum_for_golden.inc"
};
Bytes unhex(std::string_view s) {
  Bytes out; const auto nibble=[](char c){return c<='9'?c-'0':c-'a'+10;};
  for(std::size_t i=0;i<s.size();i+=2)out.push_back((nibble(s[i])<<4)|nibble(s[i+1]));
  return out;
}
template<class T> std::vector<T> words(const Bytes& raw) {
  if(raw.size()%sizeof(T))throw std::runtime_error("invalid fixture words");
  std::vector<T> values(raw.size()/sizeof(T));
  if(!raw.empty())std::memcpy(values.data(),raw.data(),raw.size());return values;
}
template<class T> Bytes bytes(const std::vector<T>& values) {
  Bytes out(values.size()*sizeof(T));if(!out.empty())std::memcpy(out.data(),values.data(),out.size());return out;
}
Bytes alp_main(const Bytes& raw) {using namespace exact_actions::learning;return alp_bitpack_forward(alp_for_forward(alp_scaling_forward(raw)));}
Bytes alp_inverse(const Bytes& wire) {using namespace exact_actions::learning;return alp_scaling_inverse(alp_for_inverse(alp_bitpack_inverse(wire)));}
Bytes reger_main(const Bytes& raw) {using namespace exact_actions::learning;return reger_bitpack_forward(reger_for_forward(reger_autoregression_forward(reger_reorder_forward(raw))));}
Bytes reger_inverse(const Bytes& wire) {using namespace exact_actions::learning;return reger_reorder_inverse(reger_autoregression_inverse(reger_for_inverse(reger_bitpack_inverse(wire))));}
}
int main() {
  using namespace exact_actions; std::size_t checked=0;
  for(const auto& g:golden){
    if(g.mode=="reger_legacy")continue; // Separate isolated legacy translation-unit test.
    const auto input=unhex(g.input),wire=unhex(g.wire);Bytes encoded,decoded;
    if(g.mode=="reference") {encoded=reference_operator_actions::encode(input,{"op.frame_of_reference.reference_v1"});decoded=reference_operator_actions::decode(wire);}
    else if(g.mode=="ts2diff") {encoded=ts2diff::compose_int64(words<std::int64_t>(input));decoded=bytes(ts2diff::decompose_int64(wire));}
    else if(g.mode=="ts2diff_float") {encoded=ts2diff::compose_double(words<double>(input),2);decoded=bytes(ts2diff::decompose_double(wire,2));}
    else if(g.mode=="tsfile_original") {encoded=tsfile::p02::encode_ts_2diff_int64(words<std::int64_t>(input));std::vector<std::int64_t> v;if(!tsfile::p02::decode_ts_2diff_int64(wire,v))throw std::runtime_error("TSFile64 decode rejected golden");decoded=bytes(v);}
    else if(g.mode=="tsfile_int32") {encoded=tsfile::p02::encode_ts_2diff_int32(words<std::int32_t>(input));std::vector<std::int32_t> v;if(!tsfile::p02::decode_ts_2diff_int32(wire,v))throw std::runtime_error("TSFile32 decode rejected golden");decoded=bytes(v);}
    else if(g.mode=="bos") {encoded=bos::compose(words<std::int64_t>(input));decoded=bytes(bos::decompose(wire));}
    else if(g.mode=="ts2diff_for") {encoded=ts2diff::frame_of_reference_forward(input);decoded=ts2diff::frame_of_reference_inverse(wire);}
    else if(g.mode=="bos_for1") {encoded=bos::first_for_forward(input);decoded=bos::first_for_inverse(wire);}
    else if(g.mode=="bos_for2") {encoded=bos::second_for_forward(input);decoded=bos::second_for_inverse(wire);}
    else if(g.mode=="alp_for") {encoded=learning::alp_for_forward(input);decoded=learning::alp_for_inverse(wire);}
    else if(g.mode=="reger_for") {encoded=learning::reger_for_forward(input);decoded=learning::reger_for_inverse(wire);}
    else if(g.mode=="alp_main") {encoded=alp_main(input);decoded=alp_inverse(wire);}
    else if(g.mode=="reger_main") {encoded=reger_main(input);decoded=reger_inverse(wire);}
    else if(g.mode=="alp_complete") {encoded=learning::alp_residual_core_forward(alp_main(input));decoded=alp_inverse(learning::alp_residual_core_inverse(wire));}
    else if(g.mode=="reger_complete") {encoded=learning::reger_residual_core_forward(reger_main(input));decoded=reger_inverse(learning::reger_residual_core_inverse(wire));}
    else if(g.mode=="alp_original") {encoded=learning::alp_original_encode(input);decoded=bytes(alp_float::decodeDoubleColumnExact(wire));}
    else if(g.mode=="reger_original") {encoded=learning::reger_original_encode(input);decoded=bytes(reger_codec::decodeInt64Column(wire));}
    else throw std::runtime_error("unknown frozen fixture");
    if(encoded!=wire || decoded!=input)throw std::runtime_error("FOR frozen wire/roundtrip differs: "+std::string(g.mode)+"/"+std::string(g.label));
    ++checked;
  }
  std::cout<<"minimum_for_adapters=PASS frozen_fixture_count="<<checked<<" complete_wire=byte_identical\n";
}
