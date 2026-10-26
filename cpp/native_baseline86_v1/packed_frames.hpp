#pragma once
#include "baseline.hpp"
#include <array>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <unistd.h>

namespace native_baseline86_v1::packed_frames {
// The default mode is custody-only. Compound mode preserves the accepted
// P172B861 scientific boundary: 16-byte header plus 8-byte lengths and NGF1.
inline constexpr std::array<std::uint8_t,8> compound_magic{'P','1','7','2','B','8','6','1'};
class Writer {
 public:
  explicit Writer(const std::filesystem::path&path,bool compound=false):fd_(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600)){
    if(fd_<0)throw std::runtime_error("cannot exclusively create frame custody pack");
    compound_=compound;if(compound_){write(compound_magic);write(std::array<std::uint8_t,8>{});offset_=16;}
  }
  ~Writer(){if(fd_>=0)::close(fd_);}
  Writer(const Writer&)=delete;
  Writer&operator=(const Writer&)=delete;
  std::uint64_t append(std::span<const std::uint8_t>frame){
    if(frame.size()>(1ULL<<30)||offset_>std::numeric_limits<std::uint64_t>::max()-8-frame.size())
      throw std::runtime_error("custody pack member/offset exceeds bound");
    const auto start=offset_;std::array<std::uint8_t,8>prefix{};
    for(unsigned i=0;i<8;++i)prefix[i]=static_cast<std::uint8_t>(frame.size()>>(8*i));
    write(prefix);write(frame);offset_+=8+frame.size();++count_;return start;
  }
  void finish(){if(fd_>=0){if(compound_){std::array<std::uint8_t,8> count{};for(unsigned i=0;i<8;++i)count[i]=static_cast<std::uint8_t>(count_>>(8*i));if(::pwrite(fd_,count.data(),8,8)!=8)throw std::runtime_error("cannot finalize compound frame count");}if(::fsync(fd_)!=0)throw std::runtime_error("cannot persist custody pack");const auto fd=fd_;fd_=-1;if(::close(fd)!=0)throw std::runtime_error("cannot close custody pack");}}
 private:
  void write(std::span<const std::uint8_t>bytes){
    while(!bytes.empty()){
      const auto n=::write(fd_,bytes.data(),bytes.size());
      if(n<0&&errno==EINTR)continue;
      if(n<=0)throw std::runtime_error("short custody pack write");
      bytes=bytes.subspan(static_cast<std::size_t>(n));
    }
  }
  int fd_;std::uint64_t offset_=0,count_=0;bool compound_=false;
};
class Reader {
 public:
  explicit Reader(const std::filesystem::path&path):input_(path,std::ios::binary),size_(std::filesystem::file_size(path)){
    if(!input_)throw std::runtime_error("cannot open custody pack");
    std::array<std::uint8_t,8> magic{};input_.read(reinterpret_cast<char*>(magic.data()),8);
    if(input_&&magic==compound_magic){compound_=true;std::array<std::uint8_t,8> count{};input_.read(reinterpret_cast<char*>(count.data()),8);if(!input_)throw std::runtime_error("truncated compound header");for(unsigned i=0;i<8;++i)declared_count_|=std::uint64_t(count[i])<<(8*i);offset_=16;}
    else{input_.clear();input_.seekg(0);}
  }
  bool is_compound()const{return compound_;}
  Bytes next(std::uint64_t expected_offset,std::uint64_t expected_length){
    if(expected_offset!=offset_||expected_length>(1ULL<<30))throw std::runtime_error("custody pack offset/length mismatch");
    std::array<std::uint8_t,8>prefix{};input_.read(reinterpret_cast<char*>(prefix.data()),8);
    if(!input_)throw std::runtime_error("truncated custody pack prefix");
    std::uint64_t length=0;for(unsigned i=0;i<8;++i)length|=std::uint64_t(prefix[i])<<(8*i);
    if(length!=expected_length)throw std::runtime_error("custody pack member length differs");
    if(offset_>std::numeric_limits<std::uint64_t>::max()-8-length)throw std::runtime_error("custody pack offset overflow");
    Bytes frame(static_cast<std::size_t>(length));if(length)input_.read(reinterpret_cast<char*>(frame.data()),length);
    if(!input_)throw std::runtime_error("truncated custody pack frame");
    offset_+=8+length;++count_;return frame;
  }
  void finish(){if(compound_&&count_!=declared_count_)throw std::runtime_error("compound member count mismatch");if(offset_!=size_||input_.peek()!=std::char_traits<char>::eof()||input_.bad())throw std::runtime_error("custody pack has extra/trailing bytes or I/O failure");}
 private:
  std::ifstream input_;std::uint64_t offset_=0,size_=0,count_=0,declared_count_=0;bool compound_=false;
};
} // namespace native_baseline86_v1::packed_frames
