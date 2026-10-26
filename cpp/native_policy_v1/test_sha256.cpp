#include "policy.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main() {
    using weighted_grpo::native_policy_v1::sha256;
    const std::pair<std::string,std::string> vectors[]={
        {"","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc","ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq","248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000,'a'),"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}};
    for(const auto& [input,wanted]:vectors) {
        const auto actual=sha256({reinterpret_cast<const std::uint8_t*>(input.data()),input.size()});
        std::ostringstream hex;
        for(auto b:actual)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);
        if(hex.str()!=wanted)throw std::runtime_error("SHA256 known-answer test failed");
    }
    std::cout<<"4 SHA256 known-answer vectors passed\n";
}
