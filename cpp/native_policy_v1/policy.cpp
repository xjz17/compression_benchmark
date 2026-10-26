#include "policy.hpp"
#include "binary.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>

namespace weighted_grpo::native_policy_v1 {
using binary::require;

std::array<std::uint8_t, 32> sha256(std::span<const std::uint8_t> input) {
    // FIPS 180-4 SHA-256, unsigned modular arithmetic, no external dependency.
    constexpr std::array<std::uint32_t,64> k={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t,8> h={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const std::size_t blocks=(input.size()+9+63)/64;
    for(std::size_t block=0;block<blocks;++block) {
        std::array<std::uint32_t,64> w{};
        for(std::size_t i=0;i<64;++i) {
            const auto index=block*64+i;
            std::uint8_t b=0;
            if(index<input.size()) b=input[index];
            else if(index==input.size()) b=0x80;
            else if(index>=blocks*64-8) b=std::uint8_t((std::uint64_t(input.size())*8) >> ((blocks*64-1-index)*8));
            w[i/4]|=std::uint32_t(b)<<((3-i%4)*8);
        }
        for(unsigned i=16;i<64;++i) {
            const auto x=w[i-15],y=w[i-2];
            w[i]=w[i-16]+(std::rotr(x,7)^std::rotr(x,18)^(x>>3))+w[i-7]+(std::rotr(y,17)^std::rotr(y,19)^(y>>10));
        }
        auto [a,b,c,d,e,f,g,z]=h;
        for(unsigned i=0;i<64;++i) {
            auto t1=z+(std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25))+((e&f)^((~e)&g))+k[i]+w[i];
            auto t2=(std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22))+((a&b)^(a&c)^(b&c));
            z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        const std::array<std::uint32_t,8> v={a,b,c,d,e,f,g,z};
        for(unsigned i=0;i<8;++i) h[i]+=v[i];
    }
    std::array<std::uint8_t,32> result{};
    for(unsigned i=0;i<32;++i) result[i]=std::uint8_t(h[i/4]>>((3-i%4)*8));
    return result;
}

static auto read_pairs(binary::Reader& in) {
    std::vector<std::pair<std::string,std::string>> result;
    const auto count=in.u32(); require(count>0 && count<=65536,"invalid fingerprint count");
    for(std::uint32_t i=0;i<count;++i) {
        auto key=in.text(); auto value=in.text();
        require(result.empty() || result.back().first<key,"fingerprints must be sorted and unique");
        result.emplace_back(std::move(key),std::move(value));
    }
    return result;
}
static bool hex_digest(const std::string& s) {
    return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
Policy Policy::parse(const std::filesystem::path& path) {
    const auto bytes=binary::read(path,maximum_model_bytes);
    require(bytes.size()>=48,"truncated model");
    auto content=std::span(bytes).first(bytes.size()-32);
    const auto digest=sha256(content);
    require(std::equal(digest.begin(),digest.end(),bytes.end()-32),"model SHA256 mismatch");
    binary::Reader in{content}; in.magic("WGPNAT1\0");
    const auto body_length=in.u64();
    require(body_length==in.remaining(),"model body length mismatch");
    Policy p;
    p.origin_={in.text(),in.text(),in.text()};
    require(p.origin_.checkpoint_schema==checkpoint_schema,"unsupported checkpoint schema");
    require(hex_digest(p.origin_.checkpoint_payload_sha256)&&hex_digest(p.origin_.checkpoint_file_sha256),"invalid checkpoint digest");
    auto& id=p.identity_;
    id.registry_fingerprint=in.text();id.feature_schema=in.text();
    id.action_count=in.u32();id.stop_action=in.u32();id.task_feature_dim=in.u32();id.state_feature_dim=in.u32();
    id.hidden_sizes={in.u32(),in.u32()};
    require(id.action_count>0 && id.action_count<=65536 && id.stop_action<id.action_count,"invalid action identity");
    require(id.task_feature_dim>0 && id.state_feature_dim>0 && id.input_dim()<=65536,"invalid feature dimensions");
    require(id.feature_schema==feature_schema && id.hidden_sizes==std::array<std::uint32_t,2>{128,128},"unsupported feature schema or architecture");
    id.data_fingerprints=read_pairs(in);id.root_fingerprints=read_pairs(in);
    require(in.u32()==6,"invalid parameter count");
    const std::array<std::vector<std::uint32_t>,6> shapes={{{std::uint32_t(id.input_dim()),128},{128},{128,128},{128},{128,id.action_count},{id.action_count}}};
    for(std::size_t i=0;i<6;++i) {
        require(in.u32()==shapes[i].size(),"parameter rank mismatch");
        std::size_t size=1;
        for(auto dim:shapes[i]) { require(in.u32()==dim,"parameter shape mismatch"); size*=dim; }
        require(in.u64()==size,"parameter element count mismatch");
        require(size<=in.remaining()/8,"truncated parameter array");
        auto& values=p.parameters_[i];values.reserve(size);
        for(std::size_t j=0;j<size;++j) values.push_back(in.f64());
    }
    in.end();return p;
}
Policy Policy::load(const std::filesystem::path& path,const RuntimeIdentity& expected) {
    auto p=parse(path);
    require(static_cast<const RuntimeIdentity&>(p.identity_)==expected,"runtime registry/feature identity mismatch");
    const auto actions=std::size_t(p.identity_.action_count);
    const auto hidden=std::size_t(p.identity_.hidden_sizes[1]);
    p.output_weights_by_action_.resize(actions*hidden);
    for(std::size_t action=0;action<actions;++action)
        for(std::size_t i=0;i<hidden;++i)
            p.output_weights_by_action_[action*hidden+i]=p.parameters_[4][i*actions+action];
    return p;
}
std::pair<Identity,Origin> Policy::inspect(const std::filesystem::path& path) {
    const auto p=parse(path);return {p.identity_,p.origin_};
}
Result Policy::infer(std::span<const double> x,std::span<const std::uint8_t> mask) const {
    require(x.size()==identity_.input_dim(),"feature dimension mismatch");
    require(mask.size()==identity_.action_count,"legal mask dimension mismatch");
    require(std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);}),"non-finite features");
    double sum=0;
    for(auto v:x.last(3)) { require(v>=0,"negative objective weight");sum+=v; }
    require(std::abs(sum-1.0)<=1e-12,"objective weights must sum to one");
    Result result;
    for(std::uint32_t i=0;i<mask.size();++i) {
        require(mask[i]<=1,"mask must contain only 0 or 1");
        if(mask[i]) result.ordered_actions.push_back(i);
    }
    require(!result.ordered_actions.empty(),"empty legal mask");
    auto layer=[](std::span<const double> src,const auto& w,const auto& b,bool activate) {
        std::vector<double> out(b.size(),0.0);
        // Row-major input-major reduction; contraction/fast-math are disabled.
        for(std::size_t i=0;i<src.size();++i)
            for(std::size_t j=0;j<out.size();++j) out[j]+=src[i]*w[i*out.size()+j];
        for(std::size_t j=0;j<out.size();++j) {
            out[j]+=b[j];require(std::isfinite(out[j]),"non-finite affine result");
            if(activate) out[j]=std::tanh(out[j]);
        }
        return out;
    };
    const auto h1=layer(x,parameters_[0],parameters_[1],true);
    const auto h2=layer(h1,parameters_[2],parameters_[3],true);
    result.logits=layer(h2,parameters_[4],parameters_[5],false);
    std::sort(result.ordered_actions.begin(),result.ordered_actions.end(),[&](auto a,auto b){
        return result.logits[a]!=result.logits[b] ? result.logits[a]>result.logits[b] : a<b;
    });
    const auto maximum=result.logits[result.ordered_actions.front()];
    result.probabilities.assign(mask.size(),0.0);sum=0;
    // Ascending action order matches the reference summation order, independent of ranking.
    for(std::size_t i=0;i<mask.size();++i) if(mask[i]) { result.probabilities[i]=std::exp(result.logits[i]-maximum);sum+=result.probabilities[i]; }
    for(auto& probability:result.probabilities) probability/=sum;
    return result;
}

std::uint32_t Policy::greedy(std::span<const double> x,
                             std::span<const std::uint32_t> legal) const {
    require(x.size()==identity_.input_dim(),"feature dimension mismatch");
    require(!legal.empty(),"empty legal action list");
    require(std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);}),"non-finite features");
    double sum=0;
    for(auto v:x.last(3)) { require(v>=0,"negative objective weight");sum+=v; }
    require(std::abs(sum-1.0)<=1e-12,"objective weights must sum to one");
    for(std::size_t i=0;i<legal.size();++i) {
        require(legal[i]<identity_.action_count,"legal action ID out of range");
        require(i==0 || legal[i-1]<legal[i],"legal action IDs must be ascending and unique");
    }

    constexpr std::size_t hidden=128;
    require(identity_.hidden_sizes==std::array<std::uint32_t,2>{hidden,hidden},
            "unsupported greedy architecture");
    require(output_weights_by_action_.size()==std::size_t(identity_.action_count)*hidden,
            "greedy output weights are not initialized");
    std::array<double,hidden> h1{},h2{};
    // Preserve infer()'s input-major binary64 reduction order exactly.
    for(std::size_t i=0;i<x.size();++i)
        for(std::size_t j=0;j<hidden;++j)
            h1[j]+=x[i]*parameters_[0][i*hidden+j];
    for(std::size_t j=0;j<hidden;++j) {
        h1[j]+=parameters_[1][j];
        require(std::isfinite(h1[j]),"non-finite affine result");
        h1[j]=std::tanh(h1[j]);
    }
    for(std::size_t i=0;i<hidden;++i)
        for(std::size_t j=0;j<hidden;++j)
            h2[j]+=h1[i]*parameters_[2][i*hidden+j];
    for(std::size_t j=0;j<hidden;++j) {
        h2[j]+=parameters_[3][j];
        require(std::isfinite(h2[j]),"non-finite affine result");
        h2[j]=std::tanh(h2[j]);
    }

    auto best=legal.front();
    double best_logit=0.0;
    bool have_best=false;
    for(const auto action:legal) {
        double logit=0.0;
        const auto offset=std::size_t(action)*hidden;
        for(std::size_t i=0;i<hidden;++i)
            logit+=h2[i]*output_weights_by_action_[offset+i];
        logit+=parameters_[5][action];
        require(std::isfinite(logit),"non-finite affine result");
        // legal is ascending, so retaining the first exact tie matches infer().
        if(!have_best || logit>best_logit) {
            best=action;
            best_logit=logit;
            have_best=true;
        }
    }
    return best;
}
} // namespace weighted_grpo::native_policy_v1
