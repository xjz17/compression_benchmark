#pragma once
// Binary framing and IEEE arithmetic compatibility for the retired Python stages.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace python_wire {
using Bytes = std::vector<std::uint8_t>;
inline void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline void put(Bytes& out, std::uint64_t value, int width) {
  for (int i=0; i<width; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8*i)));
}
inline std::uint64_t get(const Bytes& raw, std::size_t& pos, int width) {
  require(pos <= raw.size() && std::size_t(width) <= raw.size()-pos, "truncated integer");
  std::uint64_t value=0;
  for (int i=0; i<width; ++i) value |= std::uint64_t(raw[pos++]) << (8*i);
  return value;
}
inline void append(Bytes& out, const Bytes& raw) { out.insert(out.end(),raw.begin(),raw.end()); }
inline Bytes slice(const Bytes& raw, std::size_t begin, std::size_t end) {
  require(begin <= end && end <= raw.size(), "truncated byte range");
  return Bytes(raw.begin()+begin,raw.begin()+end);
}
inline void magic(Bytes& out,const char* value) { out.insert(out.end(),value,value+4); }
inline bool is_magic(const Bytes& raw,const char* value) {
  return raw.size()>=4 && std::memcmp(raw.data(),value,4)==0;
}
inline double number(std::uint64_t bits) { double value; std::memcpy(&value,&bits,8); return value; }
inline std::uint64_t word(double value) { std::uint64_t bits; std::memcpy(&bits,&value,8); return bits; }
inline std::int64_t signed_word(std::uint64_t bits) { std::int64_t value;std::memcpy(&value,&bits,8);return value; }

// Error-free partial sums and the final half-even fixup used by math.fsum.
// This operates in binary64, including each intermediate rounding; do not use fast-math/FMA.
inline double fsum(const std::vector<double>& values) {
  std::vector<double> partials;
  for (double x : values) {
    require(std::isfinite(x), "fsum requires finite inputs");
    std::size_t i=0;
    for (double y : partials) {
      if (std::abs(x)<std::abs(y)) std::swap(x,y);
      double hi=x+y, yr=hi-x, lo=y-yr;
      if (lo != 0.0) partials[i++]=lo;
      x=hi;
    }
    partials.resize(i);
    require(std::isfinite(x), "fsum intermediate overflow");
    if (x != 0.0) partials.push_back(x);
  }
  double hi=0.0, lo=0.0;
  std::size_t n=partials.size();
  if (n) {
    hi=partials[--n];
    while (n) {
      double x=hi,y=partials[--n];
      hi=x+y;
      double yr=hi-x;
      lo=y-yr;
      if (lo != 0.0) break;
    }
    if (n && ((lo<0.0 && partials[n-1]<0.0) || (lo>0.0 && partials[n-1]>0.0))) {
      double y=lo*2.0,x=hi+y,yr=x-hi;
      if (y==yr) hi=x;
    }
  }
  return hi;
}
inline int cli(int argc,char** argv, Bytes(*run)(const std::string&,const Bytes&,const std::string&)) {
  try {
    require(argc>=2 && argc<=3,"expected OP-forward/OP-inverse [int64|float64|bytes]");
    Bytes raw((std::istreambuf_iterator<char>(std::cin)),{});
    Bytes out=run(argv[1],raw,argc==3?argv[2]:"bytes");
    std::cout.write(reinterpret_cast<const char*>(out.data()),out.size());
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
}
