#include "dbp_float64_codec.hpp"

#include <openssl/sha.h>
#include <openssl/evp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

namespace {
std::string sha256(const std::vector<std::uint8_t>& data) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), digest);
    constexpr char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i=0; i<SHA256_DIGEST_LENGTH; ++i) { out[2*i]=hex[digest[i]>>4]; out[2*i+1]=hex[digest[i]&15]; }
    return out;
}
std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 32) throw std::runtime_error("control character in aggregate ID");
        out += c;
    }
    return out + '"';
}
struct Metrics {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> sequence{EVP_MD_CTX_new(), EVP_MD_CTX_free};
    Metrics() {
        if (!sequence || EVP_DigestInit_ex(sequence.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("cannot initialize frame sequence digest");
    }
    std::string id;
    int width = 0;
    std::uint64_t blocks = 0, rows = 0, missing = 0, validity = 0, count = 0,
        original = 0, compressed = 0, enc = 0, dec = 0, scaled = 0, raw = 0;
    void addFrame(const std::vector<std::uint8_t>& frame) {
        const auto length = static_cast<std::uint32_t>(frame.size());
        std::uint8_t prefix[4];
        for (int b=0; b<4; ++b) prefix[b] = static_cast<std::uint8_t>(length >> (8*b));
        if (EVP_DigestUpdate(sequence.get(), prefix, 4) != 1 ||
            EVP_DigestUpdate(sequence.get(), frame.data(), frame.size()) != 1)
            throw std::runtime_error("cannot update frame sequence digest");
    }
    void write(std::ostream& out) {
        if (id.empty()) return;
        unsigned char digest[32];
        unsigned digestSize = 0;
        if (EVP_DigestFinal_ex(sequence.get(), digest, &digestSize) != 1 || digestSize != 32)
            throw std::runtime_error("cannot finalize frame sequence digest");
        constexpr char hex[] = "0123456789abcdef";
        std::string digestHex(64, '0');
        for (int i=0; i<32; ++i) { digestHex[2*i]=hex[digest[i]>>4]; digestHex[2*i+1]=hex[digest[i]&15]; }
        out << "{\"status\":" << jsonString(count ? "ok" : "n_a")
            << ",\"aggregate_id\":" << jsonString(id) << ",\"width\":" << width
            << ",\"blocks\":" << blocks << ",\"rows\":" << rows << ",\"missing_count\":" << missing
            << ",\"validity_bytes\":" << validity << ",\"header_bytes\":" << 10*blocks
            << ",\"valid_count\":" << count << ",\"original_bytes\":" << original
            << ",\"compressed_bytes\":" << compressed << ",\"encode_ns\":" << enc
            << ",\"decode_ns\":" << dec << ",\"scaled_blocks\":" << scaled
            << ",\"raw_blocks\":" << raw << ",\"verified_values\":" << count
            << ",\"frame_sequence_sha256\":" << jsonString(digestHex) << "}\n";
        out.flush();
    }
};
void runManifest(int argc, char** argv) {
    std::string manifest, output;
    for (int i = 2; i < argc; i += 2) {
        if (i+1 >= argc) throw std::runtime_error("missing option value");
        const std::string option = argv[i];
        if (option == "--manifest") manifest = argv[i+1];
        else if (option == "--output") output = argv[i+1];
        else if (option == "--aggregates") {} // The controller validates expected aggregate totals.
        else if (option == "--repeats" && std::string(argv[i+1]) == "1") {}
        else throw std::runtime_error("invalid manifest option");
    }
    std::ifstream catalog(manifest);
    std::ofstream result(output);
    if (!catalog || !result) throw std::runtime_error("cannot open manifest/result");
    std::ifstream valuesFile;
    std::string currentPath, line;
    Metrics metrics;
    std::size_t warmBlocks = 0;
    for (int pass = 0; pass < 2; ++pass) {
    catalog.clear(); catalog.seekg(0); metrics = Metrics{};
    while (std::getline(catalog, line)) {
        std::vector<std::string> fields;
        std::stringstream row(line);
        std::string field;
        while (std::getline(row, field, '\t')) fields.push_back(field);
        if (fields.size() != 15) throw std::runtime_error("invalid 15-field manifest row");
        const int width = std::stoi(fields[1]), cap = std::stoi(fields[2]);
        const auto count = std::stoull(fields[13]), size = std::stoull(fields[5]);
        if ((width != 4 && width != 8) || count > dbp_float64::kMaxBlock || size != count*width)
            throw std::runtime_error("invalid native input length/count/width");
        if (fields[0] != metrics.id) {
            if (pass == 1) metrics.write(result);
            metrics = Metrics{};
            metrics.id = fields[0]; metrics.width = width;
        }
        if (width != metrics.width) throw std::runtime_error("aggregate width changed");
        if (fields[3] != currentPath) {
            valuesFile.close(); valuesFile.clear(); valuesFile.open(fields[3], std::ios::binary);
            currentPath = fields[3];
        }
        valuesFile.clear(); valuesFile.seekg(std::stoull(fields[4]));
        std::vector<std::uint8_t> input(size);
        valuesFile.read(reinterpret_cast<char*>(input.data()), size);
        if (!valuesFile || static_cast<std::uint64_t>(valuesFile.gcount()) != size)
            throw std::runtime_error("truncated native input segment");
        if (sha256(input) != fields[6]) throw std::runtime_error("values segment SHA256 mismatch");
        const auto rows = std::stoull(fields[12]), missing = std::stoull(fields[14]), bitmapSize = std::stoull(fields[9]);
        if (rows != count + missing || bitmapSize != (rows+7)/8)
            throw std::runtime_error("invalid row/validity shape");
        if (fields[7] != currentPath) {
            valuesFile.close(); valuesFile.clear(); valuesFile.open(fields[7], std::ios::binary);
            currentPath = fields[7];
        }
        valuesFile.clear(); valuesFile.seekg(std::stoull(fields[8]));
        std::vector<std::uint8_t> bitmap(bitmapSize);
        valuesFile.read(reinterpret_cast<char*>(bitmap.data()), bitmapSize);
        if (!valuesFile || static_cast<std::uint64_t>(valuesFile.gcount()) != bitmapSize || sha256(bitmap) != fields[10])
            throw std::runtime_error("bitmap segment SHA256 mismatch");
        std::uint64_t ones = 0;
        for (const auto byte : bitmap) ones += __builtin_popcount(static_cast<unsigned>(byte));
        if (ones != missing || (rows%8 && (bitmap.back() >> (rows%8))))
            throw std::runtime_error("bitmap missing count/unused bits mismatch");
        std::vector<double> doubles;
        std::vector<std::uint32_t> words32;
        if (width == 8) {
            doubles.resize(count);
            for (std::size_t i=0; i<count; ++i) doubles[i] = dbp_float64::fromBits(dbp_float64::readWord(input.data()+8*i));
        } else {
            words32.resize(count);
            for (std::size_t i=0; i<count; ++i) {
                std::uint32_t u=0;
                for (int b=0; b<4; ++b) u |= std::uint32_t(input[4*i+b]) << (8*b);
                words32[i]=u;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto frame = width == 8 ? dbp_float64::encodeFrame(doubles, cap) : dbp_float64::encodeFrame32(words32, cap);
        const auto t1 = std::chrono::steady_clock::now();
        auto t2 = t1;
        if (width == 8) {
            const auto decoded = dbp_float64::decodeFrame(frame);
            t2 = std::chrono::steady_clock::now();
            dbp_float64::verify(doubles, decoded);
        } else {
            const auto decoded = dbp_float64::decodeFrame32(frame);
            t2 = std::chrono::steady_clock::now();
            if (decoded != words32) throw std::runtime_error("DBF2 bitwise mismatch");
        }
        if (pass == 0) {
            if (++warmBlocks == 20) break;
            continue;
        }
        metrics.addFrame(frame);
        ++metrics.blocks;
        metrics.rows += std::stoull(fields[12]); metrics.missing += std::stoull(fields[14]);
        metrics.validity += std::stoull(fields[9]); metrics.count += count; metrics.original += size;
        metrics.compressed += frame.size(); (frame[4] == 0 ? metrics.scaled : metrics.raw)++;
        metrics.enc += std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        metrics.dec += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    }
    if (pass == 1) {
        if (!catalog.eof()) throw std::runtime_error("manifest read error");
        metrics.write(result);
    }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: dbp_float64_cli encode MAX_POINT | decode | bench MAX_POINT");
        const std::string command(argv[1]);
        if (command == "bench-manifest") { runManifest(argc, argv); return 0; }
        const std::vector<std::uint8_t> input(std::istreambuf_iterator<char>(std::cin), {});
        if (command == "decode" && argc == 2) {
            if (input.size() >= 4 && input[3] == '2') {
                const auto words = dbp_float64::decodeFrame32(input);
                for (const auto word : words) for (int b=0; b<4; ++b) std::cout.put(static_cast<char>(word >> (8*b)));
                return 0;
            }
            const auto values = dbp_float64::decodeFrame(input);
            std::vector<std::uint8_t> raw;
            raw.reserve(values.size()*8);
            for (const auto value : values) dbp_float64::appendWord(raw, dbp_float64::bits(value));
            std::cout.write(reinterpret_cast<const char*>(raw.data()), raw.size());
            return 0;
        }
        if (argc != 3 || (command != "encode" && command != "encode32" && command != "bench"))
            throw std::runtime_error("invalid command");
        const int cap = std::stoi(argv[2]);
        if (command == "encode32") {
            if (input.size()%4) throw std::runtime_error("partial float32 lane");
            std::vector<std::uint32_t> words(input.size()/4);
            for (std::size_t i=0; i<words.size(); ++i) for (int b=0; b<4; ++b) words[i] |= std::uint32_t(input[4*i+b]) << (8*b);
            const auto frame = dbp_float64::encodeFrame32(words, cap);
            std::cout.write(reinterpret_cast<const char*>(frame.data()), frame.size());
            return 0;
        }
        if (input.size() % 8) throw std::runtime_error("partial float64 lane");
        std::vector<double> values(input.size()/8);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = dbp_float64::fromBits(dbp_float64::readWord(input.data()+8*i));
        if (command == "encode") {
            const auto frame = dbp_float64::encodeFrame(values, cap);
            std::cout.write(reinterpret_cast<const char*>(frame.data()), frame.size());
            return 0;
        }
        std::uint64_t compressed = 0, enc_ns = 0, dec_ns = 0, scaled = 0, raw = 0;
        for (std::size_t offset = 0; offset < values.size() || (!offset && values.empty());
             offset += dbp_float64::kMaxBlock) {
            const auto end = std::min(values.size(), offset+dbp_float64::kMaxBlock);
            const std::vector<double> block(values.begin()+offset, values.begin()+end);
            const auto t0 = std::chrono::steady_clock::now();
            const auto frame = dbp_float64::encodeFrame(block, cap);
            const auto t1 = std::chrono::steady_clock::now();
            const auto decoded = dbp_float64::decodeFrame(frame);
            const auto t2 = std::chrono::steady_clock::now();
            dbp_float64::verify(block, decoded);
            compressed += frame.size();
            enc_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
            dec_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
            (frame[4] == 0 ? scaled : raw)++;
            if (values.empty()) break;
        }
        std::cout << "{\"valid_count\":" << values.size() << ",\"original_bytes\":" << input.size()
                  << ",\"compressed_bytes\":" << compressed << ",\"encode_ns\":" << enc_ns
                  << ",\"decode_ns\":" << dec_ns << ",\"scaled_blocks\":" << scaled
                  << ",\"raw_blocks\":" << raw << ",\"verified_values\":" << values.size() << "}\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
