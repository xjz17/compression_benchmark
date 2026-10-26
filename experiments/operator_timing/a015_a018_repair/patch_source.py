#!/usr/bin/env python3
"""Apply only the already-reviewed A015/A018 reversible-kernel repairs.

The input source tree is an isolated copy.  This script deliberately refuses
unknown anchors and never edits the running v1 tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path


A015_OLD = """      const double rounded = std::nearbyint(b0 + b1 * static_cast<double>(out.back()));
      require(std::isfinite(rounded) && std::abs(rounded) < 0x1p126, "invalid regression prediction");
      const __int128 value = static_cast<__int128>(rounded) + residual;
      require(value >= kMin && value <= kMax, "reference regression inverse exceeds int64 (legacy Python also rejects)");
      out.push_back(static_cast<I>(value));"""

A015_NEW = """      const I prediction = numpy_cast(
          std::nearbyint(b0 + b1 * static_cast<double>(out.back())));
      out.push_back(add(prediction, residual));"""

A018_OLD = """    require(std::isfinite(lower) && std::isfinite(upper) && lower >= -0x1p63 && lower < 0x1p63 && upper >= -0x1p63 && upper < 0x1p63,
            "reference outlier threshold cannot be stored as int64 (legacy Python also rejects)");
    const I low = static_cast<I>(lower), high = static_cast<I>(upper);"""

A018_NEW = """    require(std::isfinite(lower) && std::isfinite(upper),
            "reference outlier threshold is not finite");
    const auto saturate = [](double value) -> I {
      if (value <= -0x1p63) return kMin;
      if (value >= 0x1p63) return kMax;
      return static_cast<I>(value);
    };
    const I low = saturate(lower), high = saturate(upper);"""

CMAKE_TEST = """add_executable(operator97_repair_test operator97_repair_test.cpp
  operator102_dispatcher.cpp)
target_link_libraries(operator97_repair_test PRIVATE native_graph_v1_stages)
add_test(NAME operator97_repair_test COMMAND operator97_repair_test)
"""


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) == 1:
        return text.replace(old, new, 1)
    if text.count(new) == 1 and old not in text:
        return text
    raise RuntimeError(f"{label} patch anchor count is not exactly one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    reference = source / "cpp/src/reference_operator_actions.cpp"
    cmake = source / "cpp/native_graph_v1/CMakeLists.txt"
    if not reference.is_file() or not cmake.is_file():
        raise RuntimeError("isolated source tree is incomplete")

    text = reference.read_text()
    text = replace_exact(text, A015_OLD, A015_NEW, "A015")
    text = replace_exact(text, A018_OLD, A018_NEW, "A018")
    reference.write_text(text)

    cmake_text = cmake.read_text()
    if CMAKE_TEST not in cmake_text:
        cmake_text += "\n" + CMAKE_TEST
        cmake.write_text(cmake_text)

    final = reference.read_text()
    if A015_OLD in final or A018_OLD in final:
        raise RuntimeError("legacy rejecting implementation remains after repair")
    if final.count(A015_NEW) != 1 or final.count(A018_NEW) != 1:
        raise RuntimeError("repaired implementation is not unique")
    print("A015_A018_SOURCE_PATCH_PASS")


if __name__ == "__main__":
    main()
