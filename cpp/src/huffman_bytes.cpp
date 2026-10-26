#include "huffman_bytes.hpp"
#include "benchmark_loss.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kHeader = 4 + 1024 + 4;

struct Node {
    long long freq = 0;
    int minSym = 0;
    int sym = -1;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    bool isLeaf() const { return !left && !right; }
};

struct NodeCmp {
    bool operator()(const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b) const {
        if (a->freq != b->freq) {
            return a->freq > b->freq;
        }
        return a->minSym > b->minSym;
    }
};

std::vector<long long> countFreq(const std::vector<char> &data) {
    std::vector<long long> f(256, 0);
    for (unsigned char uc : data) {
        f[uc]++;
    }
    return f;
}

std::shared_ptr<Node> buildTree(const std::vector<long long> &freq) {
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodeCmp> pq;
    for (int s = 0; s < 256; ++s) {
        if (freq[static_cast<std::size_t>(s)] > 0) {
            auto n = std::make_shared<Node>();
            n->freq = freq[static_cast<std::size_t>(s)];
            n->minSym = s;
            n->sym = s;
            pq.push(n);
        }
    }
    if (pq.empty()) {
        return nullptr;
    }
    if (pq.size() == 1) {
        return pq.top();
    }
    while (pq.size() > 1) {
        std::shared_ptr<Node> a = pq.top();
        pq.pop();
        std::shared_ptr<Node> b = pq.top();
        pq.pop();
        auto p = std::make_shared<Node>();
        p->freq = a->freq + b->freq;
        p->minSym = std::min(a->minSym, b->minSym);
        p->sym = -1;
        p->left = a;
        p->right = b;
        pq.push(p);
    }
    return pq.top();
}

void dfsAssign(const std::shared_ptr<Node> &n, int c, int d, std::vector<int> &code, std::vector<int> &len) {
    if (n->isLeaf()) {
        len[static_cast<std::size_t>(n->sym)] = d;
        code[static_cast<std::size_t>(n->sym)] = d == 0 ? 0 : c;
        return;
    }
    dfsAssign(n->left, (c << 1), d + 1, code, len);
    dfsAssign(n->right, (c << 1) | 1, d + 1, code, len);
}

void assignCodes(const std::shared_ptr<Node> &root, std::vector<int> &code, std::vector<int> &len) {
    code.assign(256, 0);
    len.assign(256, 0);
    if (!root) {
        return;
    }
    if (root->isLeaf()) {
        len[static_cast<std::size_t>(root->sym)] = 1;
        code[static_cast<std::size_t>(root->sym)] = 0;
        return;
    }
    dfsAssign(root, 0, 0, code, len);
}

class BitWriter {
   public:
    void append(int bit) {
        cur_ = (cur_ << 1) | (bit & 1);
        nbits_++;
        if (nbits_ == 8) {
            buf_.push_back(static_cast<std::uint8_t>(cur_ & 0xFF));
            cur_ = 0;
            nbits_ = 0;
        }
    }

    int bitLength() const { return static_cast<int>(buf_.size() * 8 + nbits_); }

    std::vector<std::uint8_t> toBytes() {
        std::vector<std::uint8_t> out = buf_;
        if (nbits_ > 0) {
            int pad = cur_ << (8 - nbits_);
            out.push_back(static_cast<std::uint8_t>(pad & 0xFF));
        }
        return out;
    }

   private:
    std::vector<std::uint8_t> buf_;
    int cur_ = 0;
    int nbits_ = 0;
};

class BitReader {
   public:
    BitReader(const std::uint8_t *base, int start, int numBits)
        : data_(base), start_(start), numBits_(numBits) {}

    int readBit() {
        if (pos_ >= numBits_) {
            throw std::runtime_error("huffman: read past end of bitstream");
        }
        int byteIdx = start_ + (pos_ >> 3);
        int sub = 7 - (pos_ & 7);
        pos_++;
        return (static_cast<int>(data_[static_cast<std::size_t>(byteIdx)]) >> sub) & 1;
    }

   private:
    const std::uint8_t *data_;
    int start_;
    int numBits_;
    int pos_ = 0;
};

std::vector<std::uint8_t> huffmanEncode(const std::vector<char> &dataIn) {
    const int origLen = static_cast<int>(dataIn.size());
    std::vector<long long> freq = countFreq(dataIn);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(kHeader), 0);
    std::uint32_t olen = static_cast<std::uint32_t>(origLen);
    std::memcpy(out.data(), &olen, 4);
    for (int i = 0; i < 256; ++i) {
        std::uint32_t v = static_cast<std::uint32_t>(freq[static_cast<std::size_t>(i)] & 0xffffffffLL);
        std::memcpy(out.data() + 4 + i * 4, &v, 4);
    }
    if (origLen == 0) {
        std::uint32_t nb = 0;
        std::memcpy(out.data() + 1028, &nb, 4);
        return out;
    }
    int nz = 0;
    for (long long x : freq) {
        if (x > 0) {
            nz++;
        }
    }
    if (nz == 1) {
        std::uint32_t nb = 0;
        std::memcpy(out.data() + 1028, &nb, 4);
        return out;
    }
    std::shared_ptr<Node> root = buildTree(freq);
    std::vector<int> symCode(256);
    std::vector<int> symLen(256);
    assignCodes(root, symCode, symLen);
    BitWriter bw;
    for (int i = 0; i < origLen; ++i) {
        const int s = static_cast<unsigned char>(dataIn[static_cast<std::size_t>(i)]);
        const int L = symLen[static_cast<std::size_t>(s)];
        const int v = symCode[static_cast<std::size_t>(s)];
        for (int b = L - 1; b >= 0; --b) {
            bw.append((v >> b) & 1);
        }
    }
    std::uint32_t numBits = static_cast<std::uint32_t>(bw.bitLength());
    std::memcpy(out.data() + 1028, &numBits, 4);
    std::vector<std::uint8_t> body = bw.toBytes();
    out.resize(static_cast<std::size_t>(kHeader) + body.size());
    std::memcpy(out.data() + kHeader, body.data(), body.size());
    return out;
}

std::vector<char> huffmanDecode(const std::vector<std::uint8_t> &comp) {
    if (comp.size() < static_cast<std::size_t>(kHeader)) {
        throw std::runtime_error("huffman: truncated header");
    }
    std::uint32_t origLen = 0;
    std::memcpy(&origLen, comp.data(), 4);
    std::vector<long long> freq(256);
    for (int i = 0; i < 256; ++i) {
        std::uint32_t v = 0;
        std::memcpy(&v, comp.data() + 4 + i * 4, 4);
        freq[static_cast<std::size_t>(i)] = v;
    }
    std::uint32_t numBits = 0;
    std::memcpy(&numBits, comp.data() + 1028, 4);
    if (origLen == 0) {
        return {};
    }
    int nz = 0;
    int onlySym = -1;
    for (int s = 0; s < 256; ++s) {
        if (freq[static_cast<std::size_t>(s)] > 0) {
            nz++;
            onlySym = s;
        }
    }
    if (nz == 1) {
        return std::vector<char>(origLen, static_cast<char>(static_cast<unsigned char>(onlySym)));
    }
    const int need = kHeader + (static_cast<int>(numBits) + 7) / 8;
    if (static_cast<int>(comp.size()) < need) {
        throw std::runtime_error("huffman: truncated payload");
    }
    std::shared_ptr<Node> root = buildTree(freq);
    BitReader br(comp.data(), kHeader, static_cast<int>(numBits));
    std::vector<char> out(origLen);
    for (std::uint32_t i = 0; i < origLen; ++i) {
        std::shared_ptr<Node> n = root;
        while (!n->isLeaf()) {
            const int bit = br.readBit();
            n = bit == 0 ? n->left : n->right;
        }
        out[i] = static_cast<char>(static_cast<unsigned char>(n->sym));
    }
    return out;
}

} // namespace

namespace huffman_bytes {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> data) {
    std::vector<char> input(data.size());
    std::transform(data.begin(), data.end(), input.begin(),
                   [](std::uint8_t value) { return static_cast<char>(value); });
    return huffmanEncode(input);
}

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded) {
    const std::vector<std::uint8_t> input(encoded.begin(), encoded.end());
    const std::vector<char> restored = huffmanDecode(input);
    std::vector<std::uint8_t> output(restored.size());
    std::transform(restored.begin(), restored.end(), output.begin(),
                   [](char value) { return static_cast<std::uint8_t>(value); });
    return output;
}

}  // namespace huffman_bytes

void testHuffmanCompression(const std::vector<char> &data, std::vector<long long> &result,
                            LossAccum *loss) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::uint8_t> enc = huffmanEncode(data);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<char> dec = huffmanDecode(enc);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (dec.size() != data.size() || !std::equal(data.begin(), data.end(), dec.begin())) {
        throw std::runtime_error("huffman roundtrip failed");
    }
    std::chrono::duration<double, std::nano> encNs = t1 - t0;
    std::chrono::duration<double, std::nano> decNs = t2 - t1;
    result.resize(4);
    result[0] = static_cast<long long>(data.size());
    result[1] = static_cast<long long>(enc.size());
    result[2] = static_cast<long long>(encNs.count());
    result[3] = static_cast<long long>(decNs.count());
    if (loss != nullptr) {
        benchmark_loss::accumulate_bytes(data, dec, *loss);
    }
}
