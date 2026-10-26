#include "campaign.hpp"
#include "common.hpp"
#include "input_builder.hpp"
#include "model.hpp"
#include <iostream>
using namespace operator_recommend135_v1;
namespace d=operator_recommend135_v1::detail;
int main(int argc,char** argv) {
  try {
    d::require(argc>=2,"commands: catalog | inputs PLAN OUTPUT [MAX] | measure-sample SAMPLE OUTPUT [REPEATS] | encode CANDIDATE SAMPLE FRAME | decode FRAME OUTPUT | campaign INPUTS GROUPS OUTPUT [REPEATS] | evaluate ROWS OUTPUT | recommend MODEL SAMPLE OUTPUT");
    std::string command=argv[1]; nlohmann::json result;
    if(command=="catalog" && argc==2)result=candidate_catalog();
    else if(command=="inputs" && (argc==4||argc==5))result=build_inputs(argv[2],argv[3],argc==5?std::stoull(argv[4]):32768);
    else if(command=="measure-sample" && (argc==4||argc==5))result=measure_sample(d::read_json(argv[2]),argv[3],argc==5?std::stoi(argv[4]):5);
    else if(command=="decode" && argc==4)result=decode_to_directory(argv[2],argv[3]);
    else if(command=="encode" && argc==5) {
      auto sample=d::read_json(argv[3]);auto block=load_input_part(sample,sample.contains("tail")?"tail":"head");
      auto c=d::cpu_ns(),w=d::wall_ns();auto frame=encode_candidate(argv[2],block);
      auto cpu=d::cpu_ns()-c,wall=d::wall_ns()-w;d::write_bytes(argv[4],frame);
      result={{"candidate_id",argv[2]},{"frame_bytes",frame.size()},{"frame_sha256",d::sha(frame)},
        {"encode_cpu_ns",cpu},{"encode_wall_ns",wall},{"input_bytes",block.values.size()+block.missing.size()},
        {"dtype",block.dtype},{"row_count",block.row_count}};
      d::write_json(std::string(argv[4])+".json",result);
    }
    else if(command=="campaign" && (argc==5||argc==6))result=run_campaign(argv[2],argv[3],argv[4],std::filesystem::canonical(std::filesystem::exists("/proc/self/exe")?"/proc/self/exe":argv[0]),argc==6?std::stoi(argv[5]):5);
    else if(command=="evaluate" && argc==4) {
      const auto row_bytes=d::read_bytes(argv[2]);const auto rows=nlohmann::json::parse(row_bytes.begin(),row_bytes.end());
      const auto row_sha=d::sha(row_bytes);nlohmann::json source_bindings;
      const auto completion=std::filesystem::path(argv[2]).parent_path()/"CAMPAIGN_COMPLETE.json";
      if(std::filesystem::exists(completion)) {
        const auto done=d::read_json(completion);
        d::require(done.at("failed")==0 && done.at("samples")==rows.size(),"refusing incomplete/failed campaign rows");
        d::require(done.at("rows_sha256").get<std::string>()==row_sha,"campaign rows SHA mismatch");source_bindings=done.at("bindings");
      }
      std::filesystem::create_directories(argv[3]);
      const auto catalog=candidate_catalog();nlohmann::json ids=nlohmann::json::array();
      for(const auto& candidate:catalog.at("candidates"))ids.push_back(candidate.at("candidate_id"));
      nlohmann::json options{{"candidate_ids",ids},{"catalog_sha256",d::sha(catalog.dump())}};
      result=evaluate_leave_group_out(rows,options);result["training_rows_sha256"]=row_sha;result["source_bindings"]=source_bindings;
      d::write_json(std::filesystem::path(argv[3])/"EVALUATION.json",result);
      auto model=fit_model(rows,options);model["training_rows_sha256"]=row_sha;model["source_bindings"]=source_bindings;
      d::write_json(std::filesystem::path(argv[3])/"MODEL.json",model);
    } else if(command=="recommend" && argc==5) {
      auto sample=d::read_json(argv[3]);auto block=load_input_part(sample,sample.contains("head")?"head":"tail");
      auto c=d::cpu_ns(),w=d::wall_ns();auto features=extract_features(block.values,block.dtype,block.missing,block.row_count);
      auto fc=d::cpu_ns()-c,fw=d::wall_ns()-w;std::vector<std::string> eligible;
      auto ec=d::cpu_ns(),ew=d::wall_ns();const auto catalog=candidate_catalog();
      for(const auto& candidate:catalog.at("candidates")) { auto id=candidate.at("candidate_id").get<std::string>();if(candidate_eligible(id,block))eligible.push_back(id); }
      auto eligibility_cpu=d::cpu_ns()-ec,eligibility_wall=d::wall_ns()-ew;
      const auto model=d::read_json(argv[2]);d::require(model.at("candidate_catalog_sha256").get<std::string>()==d::sha(catalog.dump()),"model/runtime catalog SHA mismatch");
      result=predict(model,features,eligible);result["feature_cpu_ns"]=fc;result["feature_wall_ns"]=fw;result["eligibility_cpu_ns"]=eligibility_cpu;result["eligibility_wall_ns"]=eligibility_wall;
      result["sample_id"]=sample.value("sample_id",std::string("new-input"));result["features"]=features;
      d::write_json(argv[4],result);
    } else throw std::runtime_error("invalid command or argument count");
    if(command=="catalog"||command=="recommend"||command=="encode"||command=="decode"||command=="campaign")std::cout<<result.dump(2)<<'\n';
    else std::cout<<nlohmann::json({{"command",command},{"status","completed"}}).dump()<<'\n';
    return command=="campaign" && result.at("failed").get<std::uint64_t>()!=0 ? 2 : 0;
  } catch(const std::exception& e) { std::cerr<<nlohmann::json({{"status","failed"},{"error",e.what()}}).dump()<<'\n';return 1; }
}
