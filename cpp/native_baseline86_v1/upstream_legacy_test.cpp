#include "upstream_legacy.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
using namespace native_baseline86_v1::upstream_legacy;
int main(){
  std::mt19937 rng(2815);unsigned runs=0;
  for(auto name:{"Pithy","Shrinker","XPACK"})for(auto size:{0,1,7,31,32,33,127,128,4096,65536})for(int pattern=0;pattern<3;++pattern){
    Bytes b(size);for(int i=0;i<size;++i)b[i]=pattern==0?0:pattern==1?(i/11)%13:rng();
    auto encoded=encode(name,b);if(decode(name,encoded,b.size())!=b||encode(name,b)!=encoded)throw std::runtime_error("legacy roundtrip/determinism");
    if(!encoded.empty()){
      auto invalid=encoded;invalid[0]=255;bool rejected=false;
      try{decode(name,invalid,b.size());}catch(...){rejected=true;}
      if(!rejected)throw std::runtime_error("bad marker accepted");
    }
    ++runs;
  }
  std::cout<<"legacy roundtrip/determinism cases="<<runs<<"\n";
}
