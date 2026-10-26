#pragma once
#include "runtime.hpp"
#include <filesystem>
namespace operator_recommend135_v1 {
Block load_input_part(const nlohmann::json& sample, const std::string& part);
nlohmann::json measure_sample(const nlohmann::json& sample, const std::filesystem::path& output, int repeats=5);
nlohmann::json decode_to_directory(const std::filesystem::path& frame, const std::filesystem::path& output);
nlohmann::json verify_decoded(const nlohmann::json& sample,const std::filesystem::path& decoded);
nlohmann::json run_campaign(const std::filesystem::path& inputs,const std::filesystem::path& groups,
                            const std::filesystem::path& output,const std::filesystem::path& executable,int repeats=5);
}
