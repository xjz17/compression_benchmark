#include "policy.hpp"
#include "binary.hpp"
#include <charconv>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace np=weighted_grpo::native_policy_v1;
using np::binary::require;
static std::string quoted(const std::string& s) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') out<<'\\'<<c;
        else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c);
        else out<<c;
    }
    out<<'"';return out.str();
}
static void pairs(const auto& values) {
    std::cout<<'[';bool comma=false;
    for(const auto& [k,v]:values) {
        if(comma)std::cout<<',';comma=true;std::cout<<'['<<quoted(k)<<','<<quoted(v)<<']';
    }
    std::cout<<']';
}
static std::uint32_t number(const std::string& text) {
    std::uint32_t v=0;auto r=std::from_chars(text.data(),text.data()+text.size(),v);
    require(r.ec==std::errc{} && r.ptr==text.data()+text.size(),"invalid integer option");return v;
}
int main(int argc,char** argv) {
    try {
        require(argc>=3,"usage: weighted_grpo_native_policy_cli inspect MODEL | infer MODEL --registry R --features F --task-dim N --state-dim N --actions N --stop N --input FILE --output FILE");
        const std::string command=argv[1];
        if(command=="inspect") {
            require(argc==3,"inspect takes exactly one model path");
            auto [id,origin]=np::Policy::inspect(argv[2]);
            std::cout<<"{\"identity\":{\"registry_fingerprint\":"<<quoted(id.registry_fingerprint)
                <<",\"feature_schema\":"<<quoted(id.feature_schema)<<",\"action_count\":"<<id.action_count
                <<",\"stop_action\":"<<id.stop_action<<",\"task_feature_dim\":"<<id.task_feature_dim
                <<",\"state_feature_dim\":"<<id.state_feature_dim<<",\"hidden_sizes\":[128,128],\"data_fingerprints\":";
            pairs(id.data_fingerprints);std::cout<<",\"root_fingerprints\":";pairs(id.root_fingerprints);
            std::cout<<"},\"checkpoint_schema\":"<<quoted(origin.checkpoint_schema)
                <<",\"checkpoint_payload_sha256\":"<<quoted(origin.checkpoint_payload_sha256)
                <<",\"checkpoint_file_sha256\":"<<quoted(origin.checkpoint_file_sha256)<<"}\n";
            return 0;
        }
        require(command=="infer","unknown command");
        std::map<std::string,std::string> args;
        for(int i=3;i<argc;i+=2) {
            require(i+1<argc,"option missing value");
            require(args.emplace(argv[i],argv[i+1]).second,"duplicate option");
        }
        require(args.size()==8,"infer requires exactly eight named options");
        auto get=[&](const char* key)->const std::string& {require(args.contains(key),"missing infer option");return args.at(key);};
        const np::RuntimeIdentity expected{get("--registry"),get("--features"),number(get("--actions")),number(get("--stop")),number(get("--task-dim")),number(get("--state-dim"))};
        const auto policy=np::Policy::load(argv[2],expected);
        const auto data=np::binary::read(get("--input"),64*1024*1024);
        np::binary::Reader input{data};input.magic("WGPINP1\0");
        const auto rows=input.u32(),cols=input.u32();
        require(rows>0 && cols==expected.input_dim(),"invalid input batch dimensions");
        const std::uint64_t row_bytes=std::uint64_t(cols)*8+expected.action_count;
        require(std::uint64_t(rows)*row_bytes==input.remaining(),"input batch length mismatch");
        require(std::uint64_t(rows)*(std::uint64_t(expected.action_count)*20+4)<=128*1024*1024,
                "result batch exceeds memory limit");
        // Each input row is float64 features immediately followed by uint8 mask.
        std::vector<np::Result> results;results.reserve(rows);
        for(std::uint32_t row=0;row<rows;++row) {
            std::vector<double> features;features.reserve(cols);
            for(std::uint32_t j=0;j<cols;++j)features.push_back(input.f64());
            const auto mask=input.take(expected.action_count);
            results.push_back(policy.infer(features,mask));
        }
        input.end();
        require(!std::filesystem::exists(get("--output")),"output exists; refusing to overwrite");
        std::ofstream out(get("--output"),std::ios::binary);require(bool(out),"cannot open output");
        out.write("WGPRES1\0",8);np::binary::u32(out,rows);np::binary::u32(out,expected.action_count);
        for(const auto& result:results) {
            for(auto x:result.logits)np::binary::f64(out,x);
            for(auto x:result.probabilities)np::binary::f64(out,x);
            np::binary::u32(out,static_cast<std::uint32_t>(result.ordered_actions.size()));
            for(auto x:result.ordered_actions)np::binary::u32(out,x);
        }
        out.close();require(bool(out),"cannot write output");
        std::cout<<"{\"rows\":"<<rows<<",\"actions\":"<<expected.action_count<<",\"backend\":\"cpp_float64_2x128_tanh_v1\"}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"NATIVE_POLICY_ERROR: "<<e.what()<<'\n';return 2;}
}
