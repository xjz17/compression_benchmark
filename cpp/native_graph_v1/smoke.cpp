#include "stage_dispatch.hpp"
#include <iostream>
#include <cstring>
using namespace native_graph_v1;
int main() {
  auto d=make_default_dispatcher();
  std::vector<RootInput> roots{{{Kind::Serial,1,Bytes{'a','b','a','b'}},true,"bytes"}};
  Recipe r; r.policy_registry="cpp-canonical-127-v4"; r.steps.push_back({StepKind::Stage,{0},{"zlib.level6","bytes",{}},{},-1}); r.steps.push_back({StepKind::Stop});
  auto frame=encode(roots,r,*d); auto out=decode(frame,*d);
  if(out.roots.size()!=1 || out.roots[0].stream.data!=roots[0].stream.data) return 2;
  auto b=benchmark(roots,r,*d,1,2); std::cout<<frame.size()<<" "<<b.encode_nanoseconds.size()<<"\n"; return 0;
}
