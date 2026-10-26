#include "campaign.hpp"
#include "common.hpp"
#include <iostream>
using namespace operator_recommend135_v1;
namespace d=operator_recommend135_v1::detail;
int main() {
  auto root=std::filesystem::temp_directory_path()/("recommend135-campaign-test-"+std::to_string(getpid()));
  std::filesystem::create_directories(root);
  try {
    d::Bytes values(8*8,0),missing{0,0};
    for(std::size_t i=0;i<8;++i) values[8*i]=std::uint8_t(i);
    missing[0]=1; // nine rows, eight dense values, one NULL
    d::write_bytes(root/"v",values); d::write_bytes(root/"m",missing);
    nlohmann::json part{{"values_path",(root/"v").string()},{"missing_path",(root/"m").string()},
      {"values_sha256",d::sha(values)},{"missing_sha256",d::sha(missing)},{"row_count",9}};
    nlohmann::json s{{"sample_id","fixture"},{"dataset_id","fixture"},{"dtype","int64"},
      {"scale","primary"},{"head",part},{"tail",part}};
    auto block=load_input_part(s,"tail");
    auto frame=encode_candidate("raw",block);
    d::write_bytes(root/"f",frame);
    decode_to_directory(root/"f",root/"decoded");
    d::require(verify_decoded(s,root/"decoded").at("verified").get<bool>(),"decode must recover shape and NULLs");
    d::write_bytes(root/"decoded/values.bin",d::Bytes{1});
    d::require(!verify_decoded(s,root/"decoded").at("verified").get<bool>(),"tampered decoded values must fail");
    s["tail"]["values_sha256"]=std::string(64,'0'); bool rejected=false;
    try { (void)load_input_part(s,"tail"); } catch(const std::exception&) { rejected=true; }
    d::require(rejected,"tampered source binding must fail before measurement");
    std::filesystem::remove_all(root); std::cout<<"campaign tests passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<" evidence="<<root<<"\n"; return 1; }
}
