#include "campaign.hpp"
#include "common.hpp"
#include "model.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <iostream>

namespace ors = operator_recommend135_v1;
namespace d = ors::detail;
using Json = nlohmann::json;
using Path = std::filesystem::path;
struct ModelFile { Json model;std::string sha256; };
ModelFile load_model(const Path& file) {
  auto bytes=d::read_bytes(file);
  return {Json::parse(bytes.begin(),bytes.end()),d::sha(bytes)};
}

class PreparedService {
 public:
  explicit PreparedService(const Path& file):PreparedService(load_model(file)) {}

 private:
  explicit PreparedService(ModelFile loaded):model_(loaded.model),model_sha256_(std::move(loaded.sha256)) {
    const auto hash=d::sha(registry_.catalog().dump());
    d::require(model_.candidate_catalog_sha256()==hash,"model/runtime catalog SHA mismatch");
    Json ids=Json::array();for(const auto& c:registry_.catalog().at("candidates"))ids.push_back(c.at("candidate_id"));
    d::require(loaded.model.at("candidate_ids")==ids,"model candidate scope mismatch");
  }

 public:
  Json recommend(const Json& sample,const std::string& objective={},const Path& frame_path={}) const {
    d::require(objective.empty()==frame_path.empty(),"objective and frame_path must be supplied together");
    if(!objective.empty())d::require(objective=="space" || objective=="encode" || objective=="decode" || objective=="balanced","unsupported objective");
    auto cpu=d::cpu_ns(),wall=d::wall_ns();
    auto block=ors::load_input_part(sample,sample.contains("head")?"head":"tail");
    const auto load_cpu=d::cpu_ns()-cpu,load_wall=d::wall_ns()-wall;
    cpu=d::cpu_ns();wall=d::wall_ns();
    auto validated=registry_.prepare(block);
    auto eligible=std::move(validated.candidate_ids);std::erase(eligible,std::string("raw"));
    auto features=ors::extract_features(block.values,block.dtype,block.missing,block.row_count);
    Json prediction=model_.predict(features,eligible);
    Json selected=Json::object();
    for(const auto* target:{"space","encode","decode","balanced"}) {
      const auto& top=prediction.at("targets").at(target).at("top1");
      d::require(top.size()==1,"input has no scored compression candidate");selected[target]=top.at(0);
    }
    const auto select_cpu=d::cpu_ns()-cpu,select_wall=d::wall_ns()-wall;
    Json encoded=nullptr;
    if(!objective.empty()) {
      const auto candidate=selected.at(objective).get<std::string>();
      auto ec=d::cpu_ns(),ew=d::wall_ns();auto frame=ors::encode_candidate(candidate,block);
      const auto encode_cpu=d::cpu_ns()-ec,encode_wall=d::wall_ns()-ew;
      const auto combined_cpu=d::cpu_ns()-cpu,combined_wall=d::wall_ns()-wall;
      auto wc=d::cpu_ns(),ww=d::wall_ns();d::write_bytes(frame_path,frame);
      const auto write_cpu=d::cpu_ns()-wc,write_wall=d::wall_ns()-ww;
      encoded={{"candidate_id",candidate},{"frame_path",frame_path.string()},{"frame_bytes",frame.size()},
        {"frame_sha256",d::sha(frame)},{"encode_cpu_ns",encode_cpu},{"encode_wall_ns",encode_wall},
        {"decision_and_encode_cpu_ns",combined_cpu},{"decision_and_encode_wall_ns",combined_wall},
        {"durable_frame_write_cpu_ns",write_cpu},{"durable_frame_write_wall_ns",write_wall}};
    }
    prediction["profile"]="prepared_compression_only_no_raw";
    prediction["selected_ids"]=std::move(selected);
    prediction["sample_id"]=sample.value("sample_id",std::string("new-input"));
    prediction["dtype"]=block.dtype;prediction["row_count"]=block.row_count;
    prediction["source_values_sha256"]=d::sha(block.values);
    prediction["source_missing_sha256"]=d::sha(block.missing);
    prediction["model_sha256"]=model_sha256_;
    prediction["validation_passes"]=validated.validation_passes;
    prediction["features"]=std::move(features);
    prediction["encoded"]=std::move(encoded);
    prediction["request_cost"]={{"source_load_verify_cpu_ns",load_cpu},{"source_load_verify_wall_ns",load_wall},
      {"decision_cpu_ns",select_cpu},{"decision_wall_ns",select_wall},
      {"boundary","validated loaded Block -> features, eligibility, complete four-target prediction and selected IDs; source loading and model setup reported separately"}};
    return prediction;
  }
 private:
  ors::PreparedModel model_;
  ors::PreparedRegistry registry_;
  std::string model_sha256_;
};

int main(int argc,char** argv) {
  try {
    const bool stream=argc==3 && std::string(argv[1])=="--stream";
    d::require(stream || argc==4,"usage: prepared_serve MODEL SAMPLE PLAN | prepared_serve --stream MODEL (JSONL {sample_path:...})");
    auto cpu=d::cpu_ns(),wall=d::wall_ns();
    const PreparedService service(stream?argv[2]:argv[1]);
    const auto setup_cpu=d::cpu_ns()-cpu,setup_wall=d::wall_ns()-wall;
    Json setup{{"cpu_ns",setup_cpu},{"wall_ns",setup_wall},
      {"boundary","model JSON read/parse, PreparedModel, PreparedRegistry and model/catalog validation; once per service process"}};
    if(stream) {
      std::string line;std::uint64_t request=0;
      while(std::getline(std::cin,line)) {
        if(line.empty())continue;
        try {
          const auto input=Json::parse(line);
          d::require(input.contains("sample_path") && input.at("sample_path").is_string(),"request requires sample_path");
          auto result=service.recommend(d::read_json(input.at("sample_path").get<std::string>()),
            input.value("objective",std::string()),Path(input.value("frame_path",std::string())));
          result["request_index"]=++request;result["service_setup"]=setup;
          result["setup_reused"]=request>1;std::cout<<result.dump()<<'\n'<<std::flush;
        }catch(const std::exception& e){std::cout<<Json({{"status","request_failed"},{"error",e.what()}}).dump()<<'\n'<<std::flush;}
      }
    } else {
      auto result=service.recommend(d::read_json(argv[2]));result["service_setup"]=setup;
      d::write_json(argv[3],result);
      std::cout<<Json({{"status","complete"},{"plan_path",argv[3]},{"selected_ids",result.at("selected_ids")}}).dump()<<'\n';
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<Json({{"status","failed"},{"error",e.what()}}).dump()<<'\n';return 1;}
}
