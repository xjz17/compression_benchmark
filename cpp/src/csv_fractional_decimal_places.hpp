#pragma once

#include <cctype>
#include <string_view>

/** Max fractional decimal digits from a CSV numeric cell string (strip trailing zeros in fraction). */
inline int csv_fractional_decimal_places(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
    s.remove_prefix(1);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
    s.remove_suffix(1);
  }
  if (s.empty()) {
    return 0;
  }
  if (s.front() == '+' || s.front() == '-') {
    s.remove_prefix(1);
  }
  const std::size_t dot = s.find('.');
  if (dot == std::string_view::npos) {
    return 0;
  }
  std::string_view frac = s.substr(dot + 1);
  const std::size_t exp = frac.find_first_of("eE");
  if (exp != std::string_view::npos) {
    frac = frac.substr(0, exp);
  }
  while (!frac.empty() && frac.back() == '0') {
    frac.remove_suffix(1);
  }
  for (char ch : frac) {
    if (ch < '0' || ch > '9') {
      return 0;
    }
  }
  return static_cast<int>(frac.size());
}
