#include "binary.hpp"
#include "policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace np = weighted_grpo::native_policy_v1;
using np::binary::require;

namespace {
void u32(std::vector<std::uint8_t>& out,std::uint32_t value) {
    for(unsigned i=0;i<4;++i)out.push_back(std::uint8_t(value>>(8*i)));
}
void u64(std::vector<std::uint8_t>& out,std::uint64_t value) {
    for(unsigned i=0;i<8;++i)out.push_back(std::uint8_t(value>>(8*i)));
}
void f64(std::vector<std::uint8_t>& out,double value) {
    u64(out,std::bit_cast<std::uint64_t>(value));
}
void text(std::vector<std::uint8_t>& out,const std::string& value) {
    u32(out,static_cast<std::uint32_t>(value.size()));
    out.insert(out.end(),value.begin(),value.end());
}
void parameter(std::vector<std::uint8_t>& out,std::span<const std::uint32_t> shape,
               std::span<const double> values) {
    u32(out,static_cast<std::uint32_t>(shape.size()));
    std::uint64_t count=1;
    for(const auto dim:shape){u32(out,dim);count*=dim;}
    require(count==values.size(),"invalid synthetic parameter shape");
    u64(out,count);
    for(const auto value:values)f64(out,value);
}

struct TemporaryFile {
    std::filesystem::path path;
    ~TemporaryFile(){std::error_code ignored;std::filesystem::remove(path,ignored);}
};

TemporaryFile make_tie_model() {
    constexpr std::uint32_t actions=4,input=5,hidden=128;
    std::vector<std::uint8_t> body;
    for(const auto* value:{np::checkpoint_schema,
                           "0000000000000000000000000000000000000000000000000000000000000000",
                           "1111111111111111111111111111111111111111111111111111111111111111",
                           "greedy-contract-registry",np::feature_schema})text(body,value);
    for(const auto value:{actions,0u,1u,1u,hidden,hidden})u32(body,value);
    for(unsigned list=0;list<2;++list){
        u32(body,1);text(body,"fixture");text(body,list==0?"data":"root");
    }
    u32(body,6);
    std::vector<double> zeros_w1(input*hidden),zeros_b(hidden),zeros_w2(hidden*hidden),
                        zeros_w3(hidden*actions);
    const std::array<double,actions> biases{1.0,1.0,0.0,1.0};
    const std::array<std::uint32_t,2> w1_shape{input,hidden},w2_shape{hidden,hidden},
                                               w3_shape{hidden,actions};
    const std::array<std::uint32_t,1> hidden_shape{hidden},action_shape{actions};
    parameter(body,w1_shape,zeros_w1);parameter(body,hidden_shape,zeros_b);
    parameter(body,w2_shape,zeros_w2);parameter(body,hidden_shape,zeros_b);
    parameter(body,w3_shape,zeros_w3);parameter(body,action_shape,biases);

    std::vector<std::uint8_t> file;
    const std::array<std::uint8_t,8> magic{'W','G','P','N','A','T','1','\0'};
    file.insert(file.end(),magic.begin(),magic.end());u64(file,body.size());
    file.insert(file.end(),body.begin(),body.end());
    const auto digest=np::sha256(file);file.insert(file.end(),digest.begin(),digest.end());
    auto path=std::filesystem::temp_directory_path()/
        ("native_policy_greedy_contract_"+
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".wgpn");
    require(!std::filesystem::exists(path),"synthetic model path collision");
    std::ofstream out(path,std::ios::binary);
    require(bool(out),"cannot create synthetic model");
    out.write(reinterpret_cast<const char*>(file.data()),static_cast<std::streamsize>(file.size()));
    out.close();require(bool(out),"cannot write synthetic model");
    return {path};
}

template<class Function> void rejected(Function&& function,const char* label) {
    try{function();}catch(const std::exception&){return;}
    throw std::runtime_error(std::string("expected rejection: ")+label);
}

np::RuntimeIdentity runtime_identity(const np::Identity& id) {
    return {id.registry_fingerprint,id.feature_schema,id.action_count,id.stop_action,
            id.task_feature_dim,id.state_feature_dim};
}

void synthetic_contract() {
    const auto fixture=make_tie_model();
    const np::RuntimeIdentity expected{"greedy-contract-registry",np::feature_schema,4,0,1,1};
    const auto policy=np::Policy::load(fixture.path,expected);
    const std::array<double,5> features{0.25,-0.5,0.2,0.3,0.5};
    const std::array<std::uint32_t,3> tied{0,1,3};
    require(policy.greedy(features,tied)==0,"exact tie did not choose the smallest legal ID");
    const std::array<std::uint8_t,4> mask{1,1,0,1};
    require(policy.greedy(features,tied)==policy.infer(features,mask).ordered_actions.front(),
            "synthetic greedy/full-infer mismatch");

    const std::array<std::uint32_t,0> empty{};
    const std::array<std::uint32_t,2> unsorted{1,0},duplicate{1,1};
    const std::array<std::uint32_t,1> out_of_range{4};
    rejected([&]{policy.greedy(features,empty);},"empty legal IDs");
    rejected([&]{policy.greedy(features,unsorted);},"unsorted legal IDs");
    rejected([&]{policy.greedy(features,duplicate);},"duplicate legal IDs");
    rejected([&]{policy.greedy(features,out_of_range);},"out-of-range legal ID");
    rejected([&]{policy.greedy(std::span(features).first(4),tied);},"wrong feature width");
    auto nonfinite=features;nonfinite[0]=std::numeric_limits<double>::quiet_NaN();
    rejected([&]{policy.greedy(nonfinite,tied);},"non-finite feature");
    auto negative_weight=features;negative_weight[2]=-0.1;negative_weight[3]=0.6;
    rejected([&]{policy.greedy(negative_weight,tied);},"negative objective weight");
    auto bad_sum=features;bad_sum[4]=0.6;
    rejected([&]{policy.greedy(bad_sum,tied);},"objective weight sum");
}

std::vector<double> valid_features(std::size_t width,std::mt19937_64& rng) {
    std::uniform_real_distribution<double> values(-2.0,2.0),positive(0.01,1.0);
    std::vector<double> result(width);
    for(auto& value:result)value=values(rng);
    double sum=0;
    for(auto& value:std::span(result).last(3)){value=positive(rng);sum+=value;}
    for(auto& value:std::span(result).last(3))value/=sum;
    return result;
}

std::vector<std::uint32_t> legal_ids(std::uint32_t actions,std::size_t count,std::mt19937_64& rng) {
    std::vector<std::uint32_t> all(actions);std::iota(all.begin(),all.end(),0);
    std::shuffle(all.begin(),all.end(),rng);all.resize(std::min(count,all.size()));
    std::sort(all.begin(),all.end());return all;
}
std::vector<std::uint8_t> mask_for(std::uint32_t actions,std::span<const std::uint32_t> legal) {
    std::vector<std::uint8_t> mask(actions);
    for(const auto id:legal)mask[id]=1;
    return mask;
}

void real_model_contract(const std::filesystem::path& path) {
    const auto [identity,origin]=np::Policy::inspect(path);(void)origin;
    const auto policy=np::Policy::load(path,runtime_identity(identity));
    std::mt19937_64 rng(0x475245454459ULL);
    const std::array<std::size_t,7> mask_sizes{1,2,8,32,191,1000,65536};
    std::uint64_t checked=0;
    for(unsigned round=0;round<8;++round)for(const auto requested:mask_sizes) {
        const auto features=valid_features(identity.input_dim(),rng);
        const auto legal=legal_ids(identity.action_count,requested,rng);
        const auto mask=mask_for(identity.action_count,legal);
        const auto full=policy.infer(features,mask).ordered_actions.front();
        const auto fast=policy.greedy(features,legal);
        require(fast==full,"real-model greedy/full-infer top-1 mismatch");++checked;
    }

    constexpr std::size_t samples=256,warmups=16;
    const auto legal=legal_ids(identity.action_count,std::min<std::size_t>(191,identity.action_count),rng);
    const auto mask=mask_for(identity.action_count,legal);
    std::vector<std::vector<double>> inputs;signed char sink=0;
    inputs.reserve(samples);
    for(std::size_t i=0;i<samples;++i)inputs.push_back(valid_features(identity.input_dim(),rng));
    for(std::size_t i=0;i<warmups;++i){
        sink^=static_cast<signed char>(policy.greedy(inputs[i],legal));
        sink^=static_cast<signed char>(policy.infer(inputs[i],mask).ordered_actions.front());
    }
    using Clock=std::chrono::steady_clock;
    std::vector<std::uint64_t> greedy_ns,full_ns;greedy_ns.reserve(samples);full_ns.reserve(samples);
    for(const auto& features:inputs){
        auto begin=Clock::now();const auto action=policy.greedy(features,legal);auto end=Clock::now();
        greedy_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        sink^=static_cast<signed char>(action);
        begin=Clock::now();const auto full=policy.infer(features,mask).ordered_actions.front();end=Clock::now();
        full_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        sink^=static_cast<signed char>(full);
    }
    auto describe=[](std::vector<std::uint64_t> values){
        std::sort(values.begin(),values.end());
        return std::array<std::uint64_t,4>{values.front(),values[values.size()/2],
                                           values[(values.size()*95)/100],values.back()};
    };
    const auto g=describe(greedy_ns),f=describe(full_ns);
    std::cout<<"real_model_cases="<<checked<<" legal_actions="<<legal.size()
             <<" samples="<<samples<<" single_thread=true sink="<<int(sink)<<'\n'
             <<"greedy_ns_min_median_p95_max="<<g[0]<<','<<g[1]<<','<<g[2]<<','<<g[3]<<'\n'
             <<"full_ns_min_median_p95_max="<<f[0]<<','<<f[1]<<','<<f[2]<<','<<f[3]<<'\n';
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==1 || argc==2,"usage: native_policy_greedy_contract [REAL_MODEL.wgpn]");
        synthetic_contract();
        if(argc==2)real_model_contract(argv[1]);
        std::cout<<"native policy greedy contract passed\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"GREEDY_CONTRACT_ERROR: "<<e.what()<<'\n';return 2;
    }
}
