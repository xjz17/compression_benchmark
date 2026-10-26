#pragma once
#include <openssl/evp.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace operator_recommend135_v1::detail {
using Json = nlohmann::json;
using Bytes = std::vector<std::uint8_t>;
inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
inline Bytes read_bytes(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  require(bool(f), "cannot read " + path.string());
  const auto end = f.tellg(); require(end >= 0 && end <= std::streamoff(1ULL<<31), "file size limit");
  Bytes b(static_cast<std::size_t>(end)); f.seekg(0);
  if (!b.empty()) f.read(reinterpret_cast<char*>(b.data()), b.size());
  require(bool(f), "short read " + path.string()); return b;
}
inline std::string sha(std::span<const std::uint8_t> b) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{}; unsigned int size=0;
  require(EVP_Digest(b.data(), b.size(), digest.data(), &size, EVP_sha256(), nullptr)==1, "SHA256 failed");
  std::ostringstream s; for(unsigned i=0;i<size;++i) s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(digest[i]);
  return s.str();
}
inline std::string sha(const std::string& s) { return sha(std::span(reinterpret_cast<const std::uint8_t*>(s.data()),s.size())); }
inline std::string file_sha(const std::filesystem::path& p) { return sha(read_bytes(p)); }
inline Json read_json(const std::filesystem::path& p) { auto b=read_bytes(p); return Json::parse(b.begin(),b.end()); }
inline void write_bytes(const std::filesystem::path& path, std::span<const std::uint8_t> b) {
  const auto parent=path.has_parent_path()?path.parent_path():std::filesystem::path(".");
  std::filesystem::create_directories(parent);
  auto tmp=path; tmp += ".tmp."+std::to_string(getpid());
  int fd=::open(tmp.c_str(), O_WRONLY|O_CREAT|O_EXCL,0600); require(fd>=0,"open temporary " + tmp.string());
  std::size_t done=0;
  try {
    while(done<b.size()) { auto n=::write(fd,b.data()+done,b.size()-done); require(n>0,"write failed"); done+=n; }
    require(::fsync(fd)==0,"fsync failed"); ::close(fd); fd=-1;
    std::filesystem::rename(tmp,path);
    int dir=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);
    require(dir>=0,"open parent directory"); auto r=::fsync(dir); ::close(dir); require(r==0,"directory fsync failed");
  } catch(...) { if(fd>=0)::close(fd); std::filesystem::remove(tmp); throw; }
}
inline void write_text(const std::filesystem::path& p,const std::string& s) { write_bytes(p,std::span(reinterpret_cast<const std::uint8_t*>(s.data()),s.size())); }
inline void write_json(const std::filesystem::path& p,const Json& j) { write_text(p,j.dump(2)+"\n"); }
inline std::uint64_t wall_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline std::uint64_t cpu_ns() { timespec t{}; require(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0,"thread clock failed"); return std::uint64_t(t.tv_sec)*1000000000+t.tv_nsec; }
}
