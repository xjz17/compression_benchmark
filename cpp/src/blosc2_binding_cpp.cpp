// Exact C-Blosc2 3.3.3 macro plus B2P1 physical-block framing; no Python runtime.
#include "python_wire_compat.hpp"
#include <blosc2.h>

namespace blosc2_binding_cpp {
using namespace python_wire;
struct Header {
  unsigned version, codec_version,flags,typesize;
  std::uint32_t nbytes,blocksize;
  Bytes extension;
};
void validate(const Header& h) {
  require(h.version==5 && h.codec_version==1 && h.typesize==1 && (h.flags>>5)==0,"expected C-Blosc2 3.3.3 BLOSCLZ typesize-1 chunk");
  require((h.flags&5)==5 && h.blocksize>0 && h.blocksize<=INT32_MAX && h.nbytes<=INT32_MAX,"invalid extended Blosc2 header");
  require(h.extension.size()==16 && std::all_of(h.extension.begin(),h.extension.begin()+15,[](auto v){return v==0;}) && (h.extension[15]==0 || h.extension[15]==16),"unsupported Blosc2 extension");
}
Header read_header(const Bytes& raw,std::size_t& pos) {
  Header h;
  h.version=get(raw,pos,1);h.codec_version=get(raw,pos,1);h.flags=get(raw,pos,1);h.typesize=get(raw,pos,1);
  h.nbytes=get(raw,pos,4);h.blocksize=get(raw,pos,4);return h;
}
void put_header(Bytes& out,const Header& h) {
  put(out,h.version,1);put(out,h.codec_version,1);put(out,h.flags,1);put(out,h.typesize,1);put(out,h.nbytes,4);put(out,h.blocksize,4);
}
Bytes frame_inverse(const Bytes& wire) {
  require(wire.size()>=32,"truncated Blosc2 chunk");std::size_t pos=0;Header h=read_header(wire,pos);auto cbytes=get(wire,pos,4);h.extension=slice(wire,16,32);validate(h);
  require(cbytes==wire.size(),"Blosc2 cbytes differs from actual payload length");bool direct=(h.flags&2) || h.extension[15]==16;
  std::vector<std::pair<std::uint32_t,Bytes>> records;
  if(direct) {
    auto body=slice(wire,32,wire.size());require(!(h.flags&2) || body.size()==h.nbytes,"Blosc2 memcpy size mismatch");require(h.extension[15]!=16 || body.empty(),"Blosc2 special-zero payload");records.push_back({0,std::move(body)});
  } else {
    std::size_t count=(std::size_t(h.nbytes)+h.blocksize-1)/h.blocksize,start=32+4*count;
    require(count && start<=wire.size(),"truncated Blosc2 offsets");pos=32;
    std::vector<std::pair<std::size_t,std::uint32_t>> physical;
    for(std::size_t i=0;i<count;++i)physical.push_back({get(wire,pos,4),static_cast<std::uint32_t>(i)});
    std::sort(physical.begin(),physical.end());require(physical.front().first==start,"invalid first Blosc2 block offset");
    for(std::size_t i=0;i<count;++i){auto offset=physical[i].first,end=i+1<count?physical[i+1].first:wire.size();require(start<=offset && offset<end && end<=wire.size(),"invalid Blosc2 block span");records.push_back({physical[i].second,slice(wire,offset,end)});}
  }
  Bytes out;magic(out,"B2P1");put_header(out,h);append(out,h.extension);put(out,direct,1);put(out,records.size(),4);
  for(const auto& record:records){put(out,record.first,4);put(out,record.second.size(),4);append(out,record.second);}return out;
}
Bytes frame_forward(const Bytes& packet) {
  require(packet.size()>=37 && is_magic(packet,"B2P1"),"invalid B2P1 header");std::size_t pos=4;Header h=read_header(packet,pos);h.extension=slice(packet,pos,pos+16);pos+=16;validate(h);
  auto direct=get(packet,pos,1),count=get(packet,pos,4);bool expected=(h.flags&2)||h.extension[15]==16;
  require(direct<=1 && direct==expected,"B2P1 storage mode conflicts with flags");auto expected_count=direct?1:(std::size_t(h.nbytes)+h.blocksize-1)/h.blocksize;
  require(count && count==expected_count && count<=(packet.size()-37)/8,"B2P1 record count mismatch");
  std::vector<std::pair<std::uint32_t,Bytes>> records;std::vector<bool> seen(count,false);
  for(std::size_t i=0;i<count;++i){auto index=get(packet,pos,4),size=get(packet,pos,4);require(index<count && !seen[index],"invalid B2P1 logical ID");seen[index]=true;records.push_back({index,slice(packet,pos,pos+size)});pos+=size;}
  require(pos==packet.size(),"B2P1 trailing bytes");Bytes body;
  if(direct){body=records[0].second;require(!(h.flags&2)||body.size()==h.nbytes,"B2P1 memcpy size mismatch");require(h.extension[15]!=16 || body.empty(),"B2P1 special-zero payload");}
  else {
    std::vector<std::size_t> offsets(count);std::size_t cursor=32+4*count;
    for(const auto& record:records){require(!record.second.empty(),"empty B2P1 compressed block");offsets[record.first]=cursor;cursor+=record.second.size();require(cursor<=UINT32_MAX,"B2P1 chunk size overflow");}
    for(auto offset:offsets)put(body,offset,4);for(const auto& record:records)append(body,record.second);
  }
  Bytes out;put_header(out,h);put(out,32+body.size(),4);append(out,h.extension);append(out,body);return out;
}
void binding() {
  require(std::string(blosc2_get_version_string())=="3.3.3","requires C-Blosc2 3.3.3");
  blosc2_init();blosc1_set_compressor("blosclz");
  // The Python binding's default is one thread for these small single-block chunks.
  // Pinning one thread also gives deterministic physical block ordering for larger inputs.
  blosc2_set_nthreads(1);
}
Bytes core_forward(const Bytes& raw) {
  require(raw.size()<=INT32_MAX-32,"Blosc2 input exceeds chunk capacity");binding();Bytes chunk(raw.size()+BLOSC2_MAX_OVERHEAD);
  auto size=blosc2_compress(5,BLOSC_SHUFFLE,1,raw.data(),raw.size(),chunk.data(),chunk.size());require(size>0,"Blosc2 compression failed");chunk.resize(size);return frame_inverse(chunk);
}
Bytes core_inverse(const Bytes& packet) {
  auto chunk=frame_forward(packet);binding();std::size_t pos=4;auto size=get(chunk,pos,4);Bytes out(size?size:1);
  auto result=blosc2_decompress(chunk.data(),chunk.size(),out.data(),out.size());require(result==size,"Blosc2 decompressed size mismatch");out.resize(size);return out;
}
Bytes run(const std::string& operation,const Bytes& raw,const std::string&) {
  if(operation=="core-forward")return core_forward(raw);if(operation=="core-inverse")return core_inverse(raw);
  if(operation=="frame-forward")return frame_forward(raw);if(operation=="frame-inverse")return frame_inverse(raw);throw std::runtime_error("unknown Blosc2 stage");
}
}
int main(int argc,char**argv){return python_wire::cli(argc,argv,blosc2_binding_cpp::run);}
