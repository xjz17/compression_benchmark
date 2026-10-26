#include "native_lz_action_stages.hpp"
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
  using namespace native_lz_action;
  try {
    if (argc < 2) throw std::invalid_argument("usage: forward|inverse ACTION [window min max raw_size]; compress|parse window min max; compress-chain|parse-chain window min max candidates lazy; literal; decompress|unpack min raw_size; pack min");
    const std::string mode = argv[1];
    const Bytes raw(std::istreambuf_iterator<char>(std::cin),{});
    Bytes out;
    if (mode == "forward" && argc == 3) out = forward(argv[2],raw);
    else if (mode == "inverse" && (argc == 3 || argc == 7)) {
      Context c;
      if (argc == 7) { c = {3,static_cast<std::uint32_t>(std::stoul(argv[3])),static_cast<std::uint32_t>(std::stoul(argv[4])),static_cast<std::uint32_t>(std::stoul(argv[5])),std::stoull(argv[6])}; }
      out = inverse(argv[2],raw,argc == 7 ? &c : nullptr);
    } else if ((mode == "compress" || mode == "parse") && argc == 5) {
      const auto tokens = parse(raw,std::stoul(argv[2]),std::stoul(argv[3]),std::stoul(argv[4]));
      out = mode == "compress" ? pack(tokens,std::stoul(argv[3])) : serialize_tokens(tokens);
    } else if ((mode == "compress-chain" || mode == "parse-chain") && argc == 7) {
      const auto tokens = parse_chain(raw,std::stoul(argv[2]),std::stoul(argv[3]),std::stoul(argv[4]),std::stoul(argv[5]),std::stoul(argv[6]) != 0);
      out = mode == "compress-chain" ? pack(tokens,std::stoul(argv[3])) : serialize_tokens(tokens);
    } else if (mode == "literal" && argc == 2) out = literal(raw);
    else if ((mode == "decompress" || mode == "unpack") && argc == 4) {
      const auto tokens = unpack(raw,std::stoul(argv[2]),std::stoull(argv[3]));
      out = mode == "decompress" ? expand(tokens,std::stoull(argv[3])) : serialize_tokens(tokens);
    } else if (mode == "pack" && argc == 3) out = pack(deserialize_tokens(raw),std::stoul(argv[2]));
    else if (mode == "expand" && argc == 3) out = expand(deserialize_tokens(raw),std::stoull(argv[2]));
    else throw std::invalid_argument("invalid native-LZ CLI arguments");
    std::cout.write(reinterpret_cast<const char*>(out.data()),out.size());
    if (!std::cout) throw std::runtime_error("cannot write stage output");
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
