#include "deflate_container_stages.hpp"
#include "general_codec_reference_stages.hpp"
#include "general_codec_residual_stages.hpp"
#include "xz_container_stage.hpp"

#include <zlib.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;
namespace exact = exact_actions::general_residual;
namespace ref = reference_actions::general;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

Bytes brotli_prefix(const Bytes& raw) {
  return ref::huffman_forward(
      ref::lz_forward(ref::dictionary_forward(raw), 65535, 4, 258, true));
}
Bytes zstd_prefix(const Bytes& raw) {
  return ref::rans_forward(ref::huffman_forward(ref::parse_policy_forward(
      ref::lz_forward(raw, 65535, 3, 258, true))));
}
Bytes blosc_prefix(const Bytes& raw, int type_size) {
  return ref::token_forward(
      ref::lz_forward(ref::shuffle_forward(raw, type_size), 65535, 4, 258, false));
}
Bytes bzip2_prefix(const Bytes& raw) {
  return ref::huffman_forward(
      ref::rle_forward(ref::mtf_forward(ref::bwt_forward(raw))));
}
Bytes deflate_prefix(const Bytes& raw, bool zlib_kind) {
  auto value = ref::lz_forward(raw, 32768, 3, 258, true);
  if (zlib_kind) value = ref::parse_policy_forward(value);
  return ref::huffman_forward(value);
}
Bytes xz_prefix(const Bytes& raw) {
  return ref::range_forward(ref::parse_policy_forward(
      ref::lz_forward(raw, 16777216, 3, 273, true)));
}

void check(const Bytes& raw) {
  const auto brotli_p = brotli_prefix(raw);
  const auto brotli_wire = exact::brotli_forward(brotli_p);
  require(brotli_wire == exact::brotli_encode(raw), "Brotli byte parity");
  require(exact::brotli_inverse(brotli_wire) == brotli_p, "Brotli residual inverse");
  require(exact::brotli_decode(brotli_wire) == raw, "Brotli decode");

  const auto zstd_p = zstd_prefix(raw);
  const auto zstd_wire = exact::zstd_forward(zstd_p);
  require(zstd_wire == exact::zstd_encode(raw), "Zstd byte parity");
  require(exact::zstd_inverse(zstd_wire) == zstd_p, "Zstd residual inverse");
  require(exact::zstd_decode(zstd_wire) == raw, "Zstd decode");

  for (const int type_size : {1, 4, 8}) {
    const auto blosc_p = blosc_prefix(raw, type_size);
    const auto blosc_wire = exact::blosc1_forward(blosc_p, type_size);
    require(blosc_wire == exact::blosc1_encode(raw, type_size), "Blosc1 byte parity");
    require(exact::blosc1_inverse(blosc_wire, type_size) == blosc_p,
            "Blosc1 residual inverse");
    require(exact::blosc1_decode(blosc_wire) == raw, "Blosc1 decode");

    const auto blosc2_wire = exact::blosc2_forward(blosc_p, type_size);
    require(blosc2_wire == exact::blosc2_encode(raw, type_size), "Blosc2 byte parity");
    require(exact::blosc2_inverse(blosc2_wire, type_size) == blosc_p,
            "Blosc2 residual inverse");
    require(exact::blosc2_decode(blosc2_wire, type_size) == raw, "Blosc2 decode");
  }

  const auto bzip2_p = bzip2_prefix(raw);
  const auto bzip2_wire = exact::bzip2_forward(bzip2_p);
  require(bzip2_wire == exact::bzip2_encode(raw), "Bzip2 byte parity");
  require(exact::bzip2_inverse(bzip2_wire) == bzip2_p, "Bzip2 residual inverse");
  require(exact::bzip2_decode(bzip2_wire) == raw, "Bzip2 decode");

  const auto gzip_p = deflate_prefix(raw, false);
  const auto gzip_packet = exact::gzip_forward(gzip_p);
  const auto gzip_wire = exact_actions::deflate_container::gzip_forward(gzip_packet);
  require(gzip_wire == exact::gzip_encode(raw), "gzip byte parity");
  require(exact::gzip_inverse(gzip_packet) == gzip_p, "gzip residual inverse");
  require(exact::gzip_decode(gzip_wire) == raw, "gzip decode");

  const auto zlib_p = deflate_prefix(raw, true);
  const auto zlib_packet = exact::zlib_forward(zlib_p);
  const auto zlib_wire = exact_actions::deflate_container::zlib_forward(zlib_packet);
  require(zlib_wire == exact::zlib_encode(raw), "Zlib byte parity");
  require(exact::zlib_inverse(zlib_packet) == zlib_p, "Zlib residual inverse");
  require(exact::zlib_decode(zlib_wire) == raw, "Zlib decode");

  const auto xz_p = xz_prefix(raw);
  const auto xz_packet = exact::xz_forward(xz_p);
  const auto xz_wire = exact_actions::xz_container::forward(xz_packet);
  require(xz_wire == exact::xz_encode(raw), "XZ byte parity");
  require(exact::xz_inverse(xz_packet) == xz_p, "XZ residual inverse");
  require(exact::xz_decode(xz_wire) == raw, "XZ decode");
}

void check_deflate_parameters() {
  Bytes raw(4096);
  for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = (i / 64) % 7;
  const auto gzip_p = deflate_prefix(raw, false);
  const auto zlib_p = deflate_prefix(raw, true);
  for (int level = 0; level <= 9; ++level) {
    const auto gzip_packet = exact::gzip_forward(gzip_p, level);
    const auto zlib_packet = exact::zlib_forward(zlib_p, level);
    const auto gzip_wire = exact_actions::deflate_container::gzip_forward(gzip_packet);
    const auto zlib_wire = exact_actions::deflate_container::zlib_forward(zlib_packet);

    // An independent public zlib entry point checks wire compatibility for
    // every exposed level, not merely a round trip through the new adapter.
    uLongf size = compressBound(raw.size());
    Bytes reference(size);
    require(compress2(reference.data(), &size, raw.data(), raw.size(), level) == Z_OK,
            "zlib reference compression");
    reference.resize(size);
    require(zlib_wire == reference, "parameterized DEFLATE zlib wire parity");
    require(Bytes(gzip_wire.begin() + 10, gzip_wire.end() - 8) ==
                Bytes(zlib_wire.begin() + 2, zlib_wire.end() - 4),
            "same DEFLATE core across small gzip/zlib wrappers");
    require(exact::gzip_inverse(gzip_packet) == gzip_p,
            "parameterized gzip prefix restoration");
    require(exact::zlib_inverse(zlib_packet) == zlib_p,
            "parameterized zlib prefix restoration");
  }
  require(exact::zlib_forward(zlib_p, 0).size() > exact::zlib_forward(zlib_p, 9).size(),
          "DEFLATE level must affect the encoded result");
  for (int invalid : {-1, 10}) {
    bool rejected = false;
    try { (void)exact::gzip_forward(gzip_p, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range DEFLATE level accepted");
  }
}

}  // namespace

int main() {
  Bytes repeated(4096);
  for (std::size_t i = 0; i < repeated.size(); ++i) {
    repeated[i] = static_cast<std::uint8_t>((i / 64) % 7);
  }
  check(repeated);

  Bytes text;
  const std::string phrase = "temperature=21.50,status=stable,region=beijing\n";
  for (int i = 0; i < 96; ++i) text.insert(text.end(), phrase.begin(), phrase.end());
  text.resize(text.size() - text.size() % 8);
  check(text);

  Bytes random_bytes(4096);
  std::mt19937_64 random(20260904);
  for (auto& value : random_bytes) value = static_cast<std::uint8_t>(random());
  check(random_bytes);

  check_deflate_parameters();

  std::cout << "general_codec_residual_stages=PASS codecs=8 datasets=3 "
               "byte_parity=PASS roundtrip=PASS deflate_levels=10 invalid_levels=2\n";
}
