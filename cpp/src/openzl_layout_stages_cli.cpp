#include "openzl_layout_stages.hpp"

#include <charconv>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace openzl_layout;
std::uint64_t number(std::string_view text) {
  std::uint64_t n=0;
  auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),n);
  if(text.empty() || error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid unsigned integer argument");
  return n;
}
std::uint32_t width(std::string_view text) {
  const auto n=number(text);
  if(n>UINT32_MAX) throw std::invalid_argument("width overflows uint32");
  return static_cast<std::uint32_t>(n);
}
template<typename T> std::vector<T> list(std::string_view text) {
  std::vector<T> out;
  if(text=="-" || text.empty()) return out;
  std::size_t start=0;
  for(;;) {
    const auto comma=text.find(',',start), end=comma==std::string_view::npos ? text.size():comma;
    const auto n=number(text.substr(start,end-start));
    if(n>std::numeric_limits<T>::max()) throw std::invalid_argument("list item out of range");
    out.push_back(static_cast<T>(n));
    if(comma==std::string_view::npos) return out;
    start=comma+1;
  }
}
Kind kind(std::string_view text) {
  if(text=="serial") return Kind::Serial;
  if(text=="struct") return Kind::Struct;
  if(text=="numeric") return Kind::Numeric;
  throw std::invalid_argument("kind must be serial, struct or numeric");
}
void arity(int argc,int expected) {
  if(argc!=expected) throw std::invalid_argument("wrong number of arguments");
}
void write(const Bytes& bytes) {
  if(!bytes.empty()) std::cout.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
  if(!std::cout) throw std::runtime_error("stdout write failed");
}
} // namespace

int main(int argc,char** argv) try {
  using namespace openzl_layout;
  if(argc<2) throw std::invalid_argument("usage: encode ACTION [PARAMETERS...] | decode | decode-bundle");
  const std::string mode=argv[1];
  const Bytes raw(std::istreambuf_iterator<char>(std::cin),{});
  if(!std::cin.eof() && std::cin.fail()) throw std::runtime_error("stdin read failed");
  if(mode=="decode" || mode=="decode-bundle") {
    arity(argc,2); auto streams=inverse(deserialize(raw));
    if(mode=="decode-bundle") write(serialize(raw_bundle(std::move(streams))));
    else for(const auto& stream:streams) write(stream.data);
    return 0;
  }
  if(mode!="encode" || argc<3) throw std::invalid_argument("expected encode ACTION or decode");
  const std::string action=argv[2]; SplitResult result;
  if(action=="transpose") {
    arity(argc,4); result=transpose({Kind::Struct,width(argv[3]),raw});
  } else if(action=="split-struct") {
    arity(argc,4); result=split_struct(raw,list<std::uint32_t>(argv[3]));
  } else if(action=="float-deconstruct") {
    arity(argc,4); const std::string format=argv[3];
    result=float_deconstruct({Kind::Numeric,format=="fp32" ? 4u:2u,raw},format);
  } else if(action=="bitsplit") {
    arity(argc,5); result=bitsplit({Kind::Numeric,width(argv[3]),raw},list<std::uint8_t>(argv[4]));
  } else if(action=="bitsplit-top8") {
    arity(argc,4); result=bitsplit_top8({Kind::Numeric,width(argv[3]),raw});
  } else if(action=="bitsplit-fp") {
    arity(argc,4); result=bitsplit_fp({Kind::Numeric,width(argv[3]),raw});
  } else if(action=="bitsplit-bf16") {
    arity(argc,3); result=bitsplit_bf16({Kind::Numeric,2,raw});
  } else if(action=="bitunpack") {
    arity(argc,4); result=bitunpack(raw,width(argv[3]));
  } else if(action=="concat") {
    arity(argc,3); const auto inputs=deserialize(raw);
    if(inputs.operation!=Operation::Raw) throw std::invalid_argument("concat requires an OLS1 op=0 raw bundle");
    result=concat(inverse(inputs));
  } else if(action=="split-segments") {
    arity(argc,6); result=split_segments({kind(argv[3]),width(argv[4]),raw},list<std::uint64_t>(argv[5]));
  } else if(action=="split-by-range") {
    if(argc!=4 && argc!=5) throw std::invalid_argument("split-by-range requires WIDTH [MINIMUM]");
    const auto minimum=argc==5 ? number(argv[4]):0;
    if(argc==5 && minimum==0) throw std::invalid_argument("explicit minimum must be positive");
    result=split_by_range({Kind::Numeric,width(argv[3]),raw},minimum);
  } else throw std::invalid_argument("unknown layout action");
  write(serialize(result));
  return 0;
} catch(const std::exception& error) {
  std::cerr<<"openzl_layout_stages: "<<error.what()<<'\n'; return 2;
}
