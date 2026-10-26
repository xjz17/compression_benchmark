#include "python_wire_compat.hpp"
#include "parameterized_deflate.hpp"
#include <zlib.h>
#include <cfenv>

namespace corad_lossless_cpp {
using namespace python_wire;
constexpr std::size_t header_size=29, trick=40;
struct Header { unsigned kind;std::uint64_t count;double mean,sigma; };
Header header(const Bytes& raw,const char* expected) {
  require(raw.size()>=header_size && is_magic(raw,expected),"invalid CORAD header");
  std::size_t p=4;Header h;
  h.kind=get(raw,p,1);h.count=get(raw,p,8);h.mean=number(get(raw,p,8));h.sigma=number(get(raw,p,8));
  require(h.kind<=1 && std::isfinite(h.mean) && std::isfinite(h.sigma) && h.sigma>0,"invalid CORAD parameters");return h;
}
Bytes header_bytes(const Header& h,const char* m) {
  Bytes out;magic(out,m);put(out,h.kind,1);put(out,h.count,8);put(out,word(h.mean),8);put(out,word(h.sigma),8);return out;
}
std::uint64_t prediction(double z,const Header& h) {
  double value=z*h.sigma+h.mean;
  if (!h.kind) return word(value);
  if (!std::isfinite(value)) return 0;
  if (value>=9223372036854775808.0) return std::uint64_t(INT64_MAX);
  if (value<=-9223372036854775808.0) return std::uint64_t(INT64_MIN);
  return static_cast<std::uint64_t>(static_cast<std::int64_t>(std::nearbyint(value)));
}
void normalized(const Bytes& raw,Header& h,std::vector<std::uint64_t>& words,std::vector<std::uint64_t>& repair) {
  h=header(raw,"CNM2");require(h.count <= (raw.size()-header_size)/16 && raw.size()-header_size==16*h.count,"invalid normalized lane count");
  std::size_t p=header_size;
  for(std::uint64_t i=0;i<h.count;++i) { auto w=get(raw,p,8);require(std::isfinite(number(w)),"nonfinite normalized value");words.push_back(w); }
  for(std::uint64_t i=0;i<h.count;++i) repair.push_back(get(raw,p,8));
}
Bytes normalize_forward(const Bytes& raw,const std::string& kind) {
  require(kind=="int64" || kind=="float64","CORAD normalization expects int64 or float64");require(raw.size()%8==0,"incomplete lane");
  Header h{kind=="int64"?1u:0u,raw.size()/8,0.0,1.0};
  std::vector<std::uint64_t> originals;std::vector<double> values,finite;
  std::size_t p=0;double magnitude=0;
  while(p<raw.size()) { auto w=get(raw,p,8);originals.push_back(w);double v=h.kind?static_cast<double>(signed_word(w)):number(w);values.push_back(v);if(std::isfinite(v)){finite.push_back(v);magnitude=std::max(magnitude,std::abs(v));} }
  if(!finite.empty() && magnitude!=0) {
    std::vector<double> scaled,squares;for(auto v:finite)scaled.push_back(v/magnitude);
    double center=fsum(scaled)/static_cast<double>(scaled.size());
    for(auto v:scaled)squares.push_back(std::pow(v-center,2.0));
    double spread=std::sqrt(fsum(squares)/static_cast<double>(scaled.size()));
    h.mean=center*magnitude;double sigma=spread*magnitude;if(sigma>0 && std::isfinite(sigma))h.sigma=sigma;
  }
  Bytes out=header_bytes(h,"CNM2");std::vector<std::uint64_t> repairs;
  for(std::size_t i=0;i<values.size();++i) { double z=std::isfinite(values[i])?(values[i]-h.mean)/h.sigma:0.0;if(!std::isfinite(z))z=0.0;put(out,word(z),8);repairs.push_back(originals[i]^prediction(z,h)); }
  for(auto value:repairs)put(out,value,8);return out;
}
Bytes normalize_inverse(const Bytes& raw) {
  Header h;std::vector<std::uint64_t> words,repairs;normalized(raw,h,words,repairs);Bytes out;
  for(std::size_t i=0;i<words.size();++i)put(out,prediction(number(words[i]),h)^repairs[i],8);return out;
}
Bytes tricklet_forward(const Bytes& raw) {
  Header h;std::vector<std::uint64_t> words,repair;normalized(raw,h,words,repair);
  std::size_t blocks=(h.count+trick-1)/trick,atoms=std::min(std::size_t(4),blocks);
  std::vector<std::uint64_t> dictionary(atoms*trick,0),residual;
  for(std::size_t i=0;i<std::min(words.size(),dictionary.size());++i)dictionary[i]=words[i];
  Bytes selectors;
  for(std::size_t block=0;block<blocks;++block) {
    unsigned best=0;std::size_t cost_min=SIZE_MAX;
    for(unsigned atom=0;atom<atoms;++atom) {
      std::size_t cost=0;for(std::size_t lane=0;lane<trick && block*trick+lane<words.size();++lane){auto w=words[block*trick+lane]^dictionary[atom*trick+lane];cost+=w?64-__builtin_clzll(w):0;}
      if(cost<cost_min){cost_min=cost;best=atom;}
    }
    selectors.push_back(best);
    for(std::size_t lane=0;lane<trick && block*trick+lane<words.size();++lane)residual.push_back(words[block*trick+lane]^dictionary[best*trick+lane]);
  }
  Bytes out=header_bytes(h,"CTK2");put(out,trick,2);put(out,atoms,1);
  for(auto w:dictionary)put(out,w,8);append(out,selectors);for(auto w:residual)put(out,w,8);for(auto w:repair)put(out,w,8);return out;
}
Bytes tricklet_inverse(const Bytes& raw) {
  Header h=header(raw,"CTK2");std::size_t p=header_size;auto width=get(raw,p,2),atoms=get(raw,p,1);
  require(h.count <= raw.size()/16,"invalid tricklet lane count");std::size_t blocks=(h.count+trick-1)/trick;
  require(width==trick && atoms==std::min(std::size_t(4),blocks),"invalid tricklet dictionary shape");
  require(raw.size()==header_size+3+atoms*trick*8+blocks+h.count*16,"invalid tricklet length");
  std::vector<std::uint64_t> dictionary;for(std::size_t i=0;i<atoms*trick;++i)dictionary.push_back(get(raw,p,8));
  auto selectors=slice(raw,p,p+blocks);p+=blocks;Bytes out=header_bytes(h,"CNM2");
  for(std::size_t i=0;i<h.count;++i){auto selector=selectors[i/trick];require(selector<atoms,"invalid tricklet atom");put(out,get(raw,p,8)^dictionary[selector*trick+i%trick],8);}
  append(out,slice(raw,p,raw.size()));Header checked;std::vector<std::uint64_t>a,b;normalized(out,checked,a,b);return out;
}
std::uint64_t length_header(const Bytes& raw,const char* expected) {
  require(raw.size()>=12 && is_magic(raw,expected),"invalid CORAD length header");std::size_t p=4;return get(raw,p,8);
}
Bytes entropy_forward(const Bytes& raw, int level = 9) {
  exact_actions::parameterized_deflate::validate_level(level);
  tricklet_inverse(raw);
  auto compressed = exact_actions::parameterized_deflate::encode(
      raw, level, exact_actions::parameterized_deflate::Wrapper::zlib);
  Bytes out;magic(out,"CZL2");put(out,raw.size(),8);append(out,compressed);return out;
}
Bytes entropy_inverse(const Bytes& raw) {
  auto size=length_header(raw,"CZL2");require(size<SIZE_MAX && size<std::numeric_limits<uInt>::max(),"CORAD inflated size overflow");
  require(raw.size()-12<=std::numeric_limits<uInt>::max(),"CORAD compressed size overflow");
  auto restored = exact_actions::parameterized_deflate::decode(
      std::span<const std::uint8_t>(raw).subspan(12),
      exact_actions::parameterized_deflate::Wrapper::zlib, size);
  tricklet_inverse(restored);return restored;
}
Bytes frame_forward(const Bytes& raw) { length_header(raw,"CZL2");Bytes out;magic(out,"CRD2");put(out,raw.size(),8);append(out,raw);return out; }
Bytes frame_inverse(const Bytes& raw) { auto size=length_header(raw,"CRD2");require(size==raw.size()-12,"CORAD frame length mismatch");auto out=slice(raw,12,raw.size());length_header(out,"CZL2");return out; }
Bytes run(const std::string& operation,const Bytes& raw,const std::string& kind) {
  std::fesetround(FE_TONEAREST);
  if(operation=="normalize-forward")return normalize_forward(raw,kind);
  if(operation=="normalize-inverse")return normalize_inverse(raw);
  if(operation=="tricklet-forward")return tricklet_forward(raw);
  if(operation=="tricklet-inverse")return tricklet_inverse(raw);
  if(operation=="entropy-forward")return entropy_forward(raw);
  if(operation=="entropy-inverse")return entropy_inverse(raw);
  if(operation=="frame-forward")return frame_forward(raw);
  if(operation=="frame-inverse")return frame_inverse(raw);
  throw std::runtime_error("unknown CORAD stage");
}
}
int main(int argc, char** argv) {
  if (argc < 2 || (std::string_view(argv[1]) != "deflate-forward" &&
                   std::string_view(argv[1]) != "deflate-inverse"))
    return python_wire::cli(argc, argv, corad_lossless_cpp::run);
  try {
    const bool inverse = std::string_view(argv[1]) == "deflate-inverse";
    python_wire::require(argc == 4 || (inverse && argc == 3),
                         "expected deflate-forward corad_ctk2 LEVEL or deflate-inverse corad_ctk2 [LEVEL]");
    python_wire::require(std::string_view(argv[2]) == "corad_ctk2",
                         "CORAD DEFLATE expects io_variant corad_ctk2");
    const int level = argc == 4 ? exact_actions::parameterized_deflate::parse_level(argv[3]) : 9;
    std::fesetround(FE_TONEAREST);
    python_wire::Bytes raw((std::istreambuf_iterator<char>(std::cin)), {});
    const auto out = inverse ? corad_lossless_cpp::entropy_inverse(raw)
                             : corad_lossless_cpp::entropy_forward(raw, level);
    std::cout.write(reinterpret_cast<const char*>(out.data()), out.size());
    return std::cout ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
