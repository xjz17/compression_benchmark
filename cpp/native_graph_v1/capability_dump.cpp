#include "stage_dispatch.hpp"
#include <iostream>
int main(){for(const auto& c:native_graph_v1::native_stage_capabilities())std::cout<<c.action<<"\t"<<c.source<<"\t"<<c.semantics<<"\n";}
