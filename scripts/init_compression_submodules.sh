#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

failing_recursive_path="external/compression/FCBench"
golden_paths=(
  external/compression/brotli
  external/compression/fpzip
  external/compression/zfp
  external/compression/c-blosc
  external/compression/c-blosc2
  external/compression/lz4
  external/compression/xz
  external/compression/snappy
  external/compression/zlib
  external/compression/zstd
  external/compression/lzbench
  external/compression/cmix
  external/compression/FastPFor
  external/operator_ablation_sources/TSXor
  external/operator_ablation_sources/lzbench_legacy
  external/submodules/CLI11
  external/submodules/csv2
  external/submodules/simde
)

list_paths() {
  git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}'
}

selected_paths() {
  if [[ -n "${COMPRESSION_BENCHMARK_GOLDEN_CI:-}" ]]; then
    printf '%s\n' "${golden_paths[@]}"
  else
    list_paths | while IFS= read -r path; do
      [[ "$path" == "$failing_recursive_path" ]] || printf '%s\n' "$path"
    done
  fi
}

git -c submodule.recurse=false submodule sync
while IFS= read -r submodule_path; do
  [[ -n "$submodule_path" ]] || continue
  printf 'Initializing %s\n' "$submodule_path"
  git -c submodule.recurse=false submodule update --init -- "$submodule_path"
done < <(selected_paths)

printf '%s\n' \
  "Top-level submodules initialized without recursive nested checkouts." \
  "FCBench is skipped because its upstream tree contains an incomplete nested gitlink."
