#include "stage_dispatch.hpp"
#include "reference_operator_actions.hpp"
#include "dbp_exact_stages.hpp"
#include "simple8b_exact_stages.hpp"
#include <bit>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <limits>
using namespace native_graph_v1;
namespace {
std::set<std::string> passed;
void require(bool v,const std::string& why){if(!v)throw std::runtime_error(why);}
template<class T> Stream numeric(std::initializer_list<T> values){Bytes b(values.size()*sizeof(T));if(!b.empty())std::memcpy(b.data(),values.begin(),b.size());return {Kind::Numeric,sizeof(T),std::move(b)};}
template<class T> Stream numeric(const std::vector<T>& values){Bytes b(values.size()*sizeof(T));if(!b.empty())std::memcpy(b.data(),values.data(),b.size());return {Kind::Numeric,sizeof(T),std::move(b)};}
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"expected rejection");}
StageResult roundtrip(const StageRequest& r,const Stream& in){
  std::cerr<<"RUN "<<r.action<<" "<<r.profile<<"\n";
  auto d=make_default_dispatcher();const auto f=d->forward(r,{in});require(!f.outputs.empty(),"empty stage outputs");
  std::vector<std::string> expected_views(f.outputs.size(), "bytes");
  if(r.action=="op.xor.previous_word.semantic_ports.cpp_v1") expected_views={"uint64","xor_residual64"};
  if(r.action=="op.outlier_separation.high_bits.uint32.cpp_v1") expected_views.assign(f.outputs.size(),"uint32");
  require(f.output_logical_views==expected_views,"frozen policy output view mismatch: "+r.action);
  // Round-trip every independently addressable port through a separate backend.
  std::vector<Stream> outputs;
  for(const auto& port:f.outputs){auto z=d->forward({"zlib.level6","bytes",{}},{port});auto restored=d->inverse({"zlib.level6","bytes",{}},z.outputs,z.metadata);require(restored.size()==1 && restored[0]==port,"leaf backend mismatch");outputs.push_back(restored[0]);}
  auto cold=make_default_dispatcher();auto restored=cold->inverse(r,outputs,f.metadata);require(restored.size()==1 && restored[0]==in,"roundtrip mismatch for "+r.action);
  auto trunc=f.metadata;trunc.pop_back();rejects([&]{cold->inverse(r,outputs,trunc);});
  auto wrong=r;wrong.profile+="_tampered";rejects([&]{cold->inverse(wrong,outputs,f.metadata);});
  auto missing=outputs;missing.pop_back();rejects([&]{cold->inverse(r,missing,f.metadata);});
  passed.insert(r.action);return f;
}
Stream step(const std::string& a,const Stream& s,const std::string& profile="bytes",std::map<std::string,std::string> p={}) {auto f=roundtrip({a,profile,p},s);require(f.outputs.size()==1,"chain fixture requires one output");return f.outputs[0];}
}
int main() try {
  auto bytes=Stream{Kind::Serial,1,Bytes{'a','b','a','b','a','b',0,255,0,255}};
  auto ints=numeric<std::int64_t>({0,1,-1,9,9,9,9,127,128,-200,1000,0});
  auto positive=numeric<std::int64_t>({0,1,5,0,99,261,1025,0,7,18});
  auto scalable=numeric<double>({0.0,1.0,1.5,2.25,-4.5,9.0,9.0});
  auto floats=numeric<double>({0.0,-0.0,1.0,1.5,2.25,-4.5,9.0,9.0});
  step("zlib.level6",bytes);step("zlib.level6",{Kind::Serial,1,{}});
  for(const auto& name:reference_operator_actions::all_actions())roundtrip({name,"int64",{}},ints);
  for(const auto& r:std::vector<StageRequest>{{"op.frame_of_reference.minimum.int64_uint64_le.cpp_v1","int64",{}},{"op.dictionary.project_words64.cpp_v1","int64",{}},{"op.dictionary.project_words64.cpp_v1","int64",{{"dictionary_order","frequency"}}},{"op.zigzag.signed64.raw.cpp_v2","int64",{}},{"op.xor.previous_word.cpp_v2","int64",{}},{"op.xor.previous_word.semantic_ports.cpp_v1","float64",{}}})roundtrip(r,ints);
  roundtrip({"op.xor.previous_word.cpp_v2","bytes",{}},{Kind::Serial,1,{1,2,3,4,5}});
  for(auto width:{0,4,8,12,16,20,24,28,32})roundtrip({"op.outlier_separation.high_bits.uint32.cpp_v1","uint32",{{"width",std::to_string(width)}}},numeric<std::uint32_t>({0,1,100,65535,0xffffffffu,0x80000000u}));
  step("op.integer_code.uleb128.reference_values.cpp_v2",positive);step("op.bitpack_int.max_value_width.cpp_v1",positive,"uint64");
  step("op.bitpack_int.max_value_width.cpp_v1",positive,"uint64",{{"bit_order","lsb"},{"alignment_bytes","8"}});
  step("op.run_length.byte_optimal.cpp_v1",bytes);step("op.run_length.byte_optimal.cpp_v1",ints,"int64",{{"symbol_width","8"}});
  roundtrip({"bitsplit","physical_word",{{"mode","top8"}}},positive);
  roundtrip({"bitsplit","physical_word",{{"mode","top8"},{"width","8"}}},positive);
  roundtrip({"bitsplit","physical_word",{{"mode","fp"}}},floats);
  roundtrip({"columnize","struct",{}},{Kind::Struct,3,{1,2,3,4,5,6}});
  roundtrip({"columnize","struct",{{"mode","transpose"},{"width","3"}}},{Kind::Struct,3,{1,2,3,4,5,6}});
  roundtrip({"columnize","struct",{{"field_widths","1,2"}}},{Kind::Struct,3,{1,2,3,4,5,6}});
  roundtrip({"float_deconstruct","float32",{{"format","fp32"}}},numeric<std::uint32_t>({0,0x80000000u,0x7fc00001u,0x7f800000u,1,0x3f800000u}));
  for(auto n:{1,2,4,8})roundtrip({"bitunpack","bytes",{{"bits",std::to_string(n)}}},bytes);
  step("op.trailing_zero_pack.elf_plus_residual.cpp_v1",positive,"physical_word");
  step("op.trailing_zero_pack.elf_plus_residual.cpp_v1",numeric<std::uint32_t>({0,1,1,255,256,0xffffffffu}),"physical_word");
  for(auto p:std::vector<std::pair<std::string,std::string>>{{"op.xor.fpc.cpp_v1","op.trailing_zero_pack.fpc.cpp_v1"},{"op.xor.pfpc.cpp_v1","op.trailing_zero_pack.pfpc.cpp_v1"}}){auto x=step(p.first,floats,"float64");step(p.second,x);}
  for(auto name:{"gorilla","chimp128"})for(auto view:{"int64","float64"}){
    auto x=step(std::string("op.xor.")+name+".cpp_v1",view==std::string("int64")?ints:floats,view);
    step(std::string("op.trailing_zero_pack.")+name+".cpp_v1",x,view);
    // Frozen actions A038/A039 receive the predecessor's ordinary logical
    // view (`bytes`) and recover the numeric profile from its typed packet.
    step(std::string("op.trailing_zero_pack.")+name+".cpp_v1",x,"bytes");
  }
  roundtrip({"op.xor.chimp_history128.cpp_v2","int64",{}},ints);
  roundtrip({"op.xor.chimp_history128.cpp_v2","float64",{}},floats);
  auto p=step("op.eraser.pongo.cpp_v1",floats,"float64");p=step("op.xor.pongo.cpp_v1",p);step("op.trailing_zero_pack.pongo.cpp_v1",p);
  auto e=step("op.eraser.elf.cpp_v1",floats,"float64");step("op.xor.elf.cpp_v1",e);step("op.xor.elf_plus.cpp_v1",e);step("op.eraser.elf64.cpp_v1",floats,"float64");
  auto c=step("op.eraser.camel.cpp_v1",floats,"float64");c=step("op.xor.camel.cpp_v1",c);step("op.bitstream_pack.camel.cpp_v1",c);
  auto w=step("op.block_transform.buff_native_word_order.cpp_v1",floats,"float64");step("op.container_frame.buff_native_raw64.cpp_v1",w);
  for(auto scale:{"op.scaling.exact_decimal.cpp_v2","op.scaling.delta_binary_packed.cpp_v1"})for(unsigned point=0;point<=18;++point)step(scale,numeric<double>({0,1,-1}),"float64",{{"max_point",std::to_string(point)}});
  for(auto variant:{"op.scaling.simple8b.cpp_v1","op.scaling.ts_2diff.cpp_v1"})
    for(unsigned point=0;point<=18;++point)
      step("op.scaling.exact_decimal.cpp_v2",numeric<double>({0,1,-1}),"float64",
           {{"max_point",std::to_string(point)},{"io_variant",variant}});
  const std::vector<std::int64_t> dbp_values{0,1,5,2,9};
  auto dbp=Stream{Kind::Serial,1,exact_actions::dbp::delta_forward_int64(dbp_values)};
  dbp=step("op.zigzag.dbp_residual.cpp_v2",dbp);step("op.integer_code.uleb128.dbp_residuals.cpp_v2",dbp);
  auto t=step("op.delta.ts_2diff.cpp_v1",positive,"int64");step("op.frame_of_reference.ts_2diff.cpp_v1",t);
  auto scaled=step("op.scaling.exact_decimal.cpp_v2",scalable,"float64",{{"max_point","4"}});auto td=step("op.delta.ts_2diff.cpp_v1",scaled,"scaled_packet");step("op.frame_of_reference.ts_2diff.cpp_v1",td);
  auto spdp=step("op.delta.spdp.cpp_v1",floats);spdp=step("op.block_transform.spdp.cpp_v1",spdp);step("op.entropy_special.spdp.cpp_v1",spdp);
  for(auto a:{"op.dictionary.brotli_static_small.cpp_reference_v1","op.block_transform.bzip2_bwt.cpp_reference_v1","op.symbol_recode.bzip2_mtf.cpp_reference_v1","op.entropy_huffman.project_bytes.cpp_v1","op.entropy_range.project_bytes.cpp_v1","op.entropy_fse.zstd_rans.cpp_reference_v1"})step(a,bytes);
  for(auto n:{1,4,8})step("op.block_transform.blosc_shuffle.typesize"+std::to_string(n)+".cpp_reference_v1",floats);
  for(auto n:{"brotli_lazy65535","zstd_lazy65535","deflate_lazy32768","xz_lazy16777216","blosclz.greedy"}){auto x=step(std::string("op.lz_match.")+n+".cpp_reference_v1",bytes);step("op.match_token_format.blosclz.ctrl8.cpp_reference_v1",x);step("op.match_parse_policy.generic_packet.cpp_reference_v1",x);}
  auto l=step("op.match_window_model.native_lz.window4096",bytes);l=step("op.lz_match.native_lz.aligned8.min3.max18",l);l=step("op.match_parse_policy.native_lz.aligned8_greedy",l);step("op.match_token_format.native_lz.ctrl8_offset16_len8.bias3",l);step("op.match_token_format.native_lz.ctrl8_offset16_len8.bias4",l);step("op.match_window_model.native_lz.window65535",bytes);
  auto legacy=step("op.scaling.legacy_exact_decimal.cpp_v1",positive,"int64");legacy=step("op.block_transform.int64_lo_hi32.cpp_v1",legacy);legacy=step("op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1",legacy);legacy=step("op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1",legacy);step("op.container_frame.legacy_numeric_limbs.cpp_v1",legacy);
  auto alp=step("op.scaling.alp_legacy_fe_exceptions.cpp_v1",floats,"float64");alp=step("op.block_transform.int64_lo_hi32.cpp_v1",alp);alp=step("op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1",alp);alp=step("op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1",alp);step("op.container_frame.alp2_legacy.cpp_v1",alp);

  auto bos=step("op.delta.bos.cpp_v1",ints,"int64");bos=step("op.frame_of_reference.bos.cpp_v1",bos);step("op.outlier_separation.bos.cpp_v1",bos);
  auto svb=step("op.delta.stream_vbyte.cpp_v1",ints,"int64");step("op.integer_code.stream_vbyte.cpp_v1",svb);
  auto dbp_scaled_fixture=step("op.scaling.delta_binary_packed.cpp_v1",scalable,"float64",{{"max_point","4"}});
  auto svb_scaled=step("op.delta.stream_vbyte.cpp_v1",dbp_scaled_fixture,"bytes");step("op.integer_code.stream_vbyte.cpp_v1",svb_scaled);
  step("op.bitpack_int.simple8b.cpp_v1",ints,"int64");
  const std::vector<double> simple8b_values{0.0,1.0,1.5,2.25,-4.5,9.0,9.0};
  auto simple8b_scaled=Stream{Kind::Serial,1,exact_actions::simple8b::scaling_forward(simple8b_values,4)};
  step("op.bitpack_int.simple8b.cpp_v1",simple8b_scaled,"bytes");
  std::vector<std::int64_t> long_values(256);for(std::size_t i=0;i<long_values.size();++i)long_values[i]=static_cast<std::int64_t>(i*i-3*i);
  auto fast=step("op.bitpack_int.fastpfor_pfor.cpp_v1",numeric(long_values),"int64");step("op.integer_code.fastpfor_variablebyte.cpp_v1",fast);
  auto fast_scaled=step("op.bitpack_int.fastpfor_pfor.cpp_v1",dbp_scaled_fixture,"bytes");step("op.integer_code.fastpfor_variablebyte.cpp_v1",fast_scaled);
  step("op.delta.sprintz.cpp_v1",ints,"int64");step("op.delta.sprintz.cpp_v1",floats,"float64");
  auto fpzip=step("op.delta.fpzip.cpp_v1",floats,"float64");step("op.entropy_range.fpzip.cpp_v1",fpzip);
  for(unsigned point=0;point<=18;++point)step("op.scaling.tsfile_rle.cpp_v1",numeric<double>({0,1,-1}),"float64",{{"max_point",std::to_string(point)}});
  step("op.scaling.alp.cpp_v1",floats,"float64");
  auto buff=step("op.scaling.buff.cpp_v1",floats,"float64");buff=step("op.frame_of_reference.buff.cpp_v2",buff);step("op.block_transform.buff_byte_plane_split.cpp_v2",buff);
  step("op.regression_predictor.leco_linear.cpp_v2",ints,"int64");
  auto lz4=step("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1",bytes);lz4=step("op.match_token_format.lz4.cpp_v1",lz4);step("op.integer_code.lz4.cpp_v1",lz4);
  auto snappy=step("op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",bytes);step("op.match_token_format.snappy.cpp_v1",snappy);
  step("op.lz_match.snappy_snap_1_1_1.cpp_v1",bytes);
  auto corad=step("op.scaling.corad_normalize_exact.cpp_v2",ints,"int64");step("op.dictionary.corad_tricklet_xor.cpp_v2",corad);
  step("op.scaling.corad_normalize_exact.cpp_v2",floats,"float64");
  step("op.prediction_probability.ppm.provider_v1",bytes,"bytes",{{"ppmd_order","6"},{"ppmd_memory_mib","16"}});
  step("op.prediction_probability.dmc.provider_v1",bytes);
  const StageRequest paq_request{"op.prediction_probability.paq.provider_v1","bytes",{{"paq_level","0"}}};
  auto paq_result=roundtrip(paq_request,bytes);
  auto paq_corrupt=paq_result.outputs;paq_corrupt[0].data[0]^=0x80;
  {auto paq_decoder=make_default_dispatcher();rejects([&]{paq_decoder->inverse(paq_request,paq_corrupt,paq_result.metadata);});}
  for(unsigned repeat=0;repeat<2;++repeat)roundtrip(paq_request,bytes);
  // Full cmix is a real linked provider but the frozen action deliberately
  // requires an explicit >=32-GB resource opt-in.  Exercise that fail-closed
  // gate in routine tests; the resource qualification runs the opted-in path.
  {auto gated=make_default_dispatcher();rejects([&]{gated->forward({"op.prediction_probability.cmix.provider_v1","bytes",{}},{bytes});});passed.insert("op.prediction_probability.cmix.provider_v1");}
  const auto capabilities=native_stage_capabilities();
  if(std::any_of(capabilities.begin(),capabilities.end(),[](const auto& capability){
       return capability.action=="op.prediction_probability.nncp.provider_v1";
     }))
    step("op.prediction_probability.nncp.provider_v1",bytes);
  auto d=make_default_dispatcher();rejects([&]{d->forward({"unregistered.action","bytes",{}},{bytes});});rejects([&]{d->forward({"zlib.level6","bytes",{{"extra","1"}}},{bytes});});
  for(const auto& capability:capabilities)require(passed.contains(capability.action),"registered stage was not tested: "+capability.action);
  for(const auto& name:passed)std::cout<<"PASS\t"<<name<<'\n';
  std::cout<<"stage_dispatch_test passed "<<passed.size()<<" registered actions; independent inverse; metadata/request/port rejection\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
