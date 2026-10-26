#include "native_graph_v1.h"
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
int main(int argc,char** argv){
  if(argc<2){std::cerr<<"usage: encode|decode|benchmark [warmups repeats]\n";return 64;}
  std::vector<uint8_t> in((std::istreambuf_iterator<char>(std::cin)),{}); wc_native_graph_v1_result* r=nullptr; std::string mode=argv[1];
  if(mode=="encode") r=wc_native_graph_v1_encode_zlib(in.data(),in.size());
  else if(mode=="decode") r=wc_native_graph_v1_decode(in.data(),in.size());
  else if(mode=="benchmark" && argc==4) r=wc_native_graph_v1_benchmark_zlib(in.data(),in.size(),std::stoul(argv[2]),std::stoul(argv[3]));
  else {std::cerr<<"invalid arguments\n";return 64;}
  if(std::string e=wc_native_graph_v1_result_error(r);!e.empty()){std::cerr<<e<<"\n";wc_native_graph_v1_result_free(r);return 1;}
  if(mode=="benchmark") {std::cout<<"{\"original_numeric_bytes\":"<<wc_native_graph_v1_original_numeric_bytes(r)<<",\"frame_bytes\":"<<wc_native_graph_v1_result_size(r)<<",\"encode_ns\":[";auto ep=wc_native_graph_v1_encode_ns(r);for(size_t i=0;i<wc_native_graph_v1_encode_ns_count(r);++i){if(i)std::cout<<',';std::cout<<ep[i];}std::cout<<"],\"decode_ns\":[";auto dp=wc_native_graph_v1_decode_ns(r);for(size_t i=0;i<wc_native_graph_v1_decode_ns_count(r);++i){if(i)std::cout<<',';std::cout<<dp[i];}std::cout<<"]}\n";}
  else std::cout.write(reinterpret_cast<const char*>(wc_native_graph_v1_result_data(r)),wc_native_graph_v1_result_size(r)); wc_native_graph_v1_result_free(r); return 0;
}
