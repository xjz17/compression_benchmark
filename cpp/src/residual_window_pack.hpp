#pragma once
#include <cstdint>
#include <stdexcept>

// These packers consume residuals only. XOR, first-word state, chunk resets,
// EOS records and eraser fields are supplied by their composition adapters.
namespace exact_actions::residual_window_pack {

struct ChimpProfile {
  int word_bits;
  int history_index_bits;
  int significant_bits;
  int trailing_threshold;
};

template<class Write>
int chimp(Write write, std::uint64_t residual, unsigned previous, int leading,
          int trailing, unsigned leading_code, int& stored_leading, ChimpProfile p) {
  if ((p.word_bits != 32 && p.word_bits != 64) || p.history_index_bits < 0 ||
      p.history_index_bits > 7 || previous >= (1U << p.history_index_bits))
    throw std::invalid_argument("invalid Chimp window profile");
  if (residual == 0) {
    write(0, 2);
    write(previous, p.history_index_bits);
    stored_leading = p.word_bits + 1;
    return 2 + p.history_index_bits;
  }
  if (trailing > p.trailing_threshold) {
    const int significant = p.word_bits - leading - trailing;
    write(1, 2);
    write(previous, p.history_index_bits);
    write(leading_code, 3);
    write(significant, p.significant_bits);
    write(residual >> trailing, significant);
    stored_leading = p.word_bits + 1;
    return 5 + p.history_index_bits + p.significant_bits + significant;
  }
  const bool reuse = leading == stored_leading;
  write(reuse ? 2 : 3, 2);
  if (!reuse) write(leading_code, 3);
  write(residual, p.word_bits - leading);
  stored_leading = leading;
  return 2 + (reuse ? 0 : 3) + p.word_bits - leading;
}

struct ElfPlusProfile { int word_bits; int short_length_bits; int long_length_bits; };

struct ChimpResidual { std::uint64_t value; unsigned previous; };

template<class Read, class LeadingTable>
ChimpResidual chimp_inverse(Read read, const LeadingTable& leading_table,
                             int& stored_leading, ChimpProfile p) {
  const auto flag = read(2);
  unsigned previous = 0;
  if (flag <= 1) previous = static_cast<unsigned>(read(p.history_index_bits));
  if (flag == 0) { stored_leading = p.word_bits + 1; return {0, previous}; }
  if (flag == 1) {
    const int leading = leading_table[read(3)];
    int significant = read(p.significant_bits);
    if (significant == 0) significant = 1 << p.significant_bits;
    const int trailing = p.word_bits - leading - significant;
    if (trailing < 0) throw std::runtime_error("invalid Chimp window length");
    stored_leading = p.word_bits + 1;
    return {read(significant) << trailing, previous};
  }
  if (flag == 3) stored_leading = leading_table[read(3)];
  if (stored_leading < 0 || stored_leading >= p.word_bits)
    throw std::runtime_error("Chimp reuse before window initialization");
  return {read(p.word_bits - stored_leading), previous};
}

template<class Write>
void elf_plus(Write write, std::uint64_t residual, int leading, int trailing,
              unsigned leading_code, int& stored_leading, int& stored_trailing,
              ElfPlusProfile p) {
  if (residual == 0) { write(1, 2); return; }
  if (leading == stored_leading && trailing >= stored_trailing) {
    write(0, 2);
    write(residual >> stored_trailing, p.word_bits - stored_leading - stored_trailing);
    return;
  }
  stored_leading = leading;
  stored_trailing = trailing;
  const int significant = p.word_bits - leading - trailing;
  const bool short_header = significant <= (1 << p.short_length_bits);
  const int length_bits = short_header ? p.short_length_bits : p.long_length_bits;
  write(short_header ? 2 : 3, 2);
  write(leading_code, 3);
  write(significant & ((1U << length_bits) - 1), length_bits);
  if (significant > 1) write(residual >> (trailing + 1), significant - 1);
}

template<class Read, class LeadingTable>
std::uint64_t elf_plus_inverse(Read read, const LeadingTable& leading_table,
                               int& stored_leading, int& stored_trailing,
                               ElfPlusProfile p) {
  const auto flag = read(2);
  if (flag == 1) return 0;
  if (flag == 0) {
    const int significant = p.word_bits - stored_leading - stored_trailing;
    if (significant <= 0 || significant > p.word_bits)
      throw std::runtime_error("Elf+ reuse before window initialization");
    return read(significant) << stored_trailing;
  }
  stored_leading = leading_table[read(3)];
  const int length_bits = flag == 2 ? p.short_length_bits : p.long_length_bits;
  int significant = read(length_bits);
  if (significant == 0) significant = 1 << length_bits;
  stored_trailing = p.word_bits - stored_leading - significant;
  if (stored_trailing < 0) throw std::runtime_error("invalid Elf+ window length");
  return ((read(significant - 1) << 1) | 1U) << stored_trailing;
}
}  // namespace exact_actions::residual_window_pack
