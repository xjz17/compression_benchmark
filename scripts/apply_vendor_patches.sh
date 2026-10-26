#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_patch_once() {
  local submodule_path="$1"
  local expected_commit="$2"
  local patch_path="$3"
  local absolute_submodule="$repo_root/$submodule_path"
  local absolute_patch="$repo_root/$patch_path"

  if [[ ! -d "$absolute_submodule/.git" && ! -f "$absolute_submodule/.git" ]]; then
    echo "Missing initialized submodule: $submodule_path" >&2
    echo "Run: ./scripts/init_compression_submodules.sh" >&2
    return 1
  fi

  local actual_commit
  actual_commit="$(git -C "$absolute_submodule" rev-parse HEAD)"
  if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "Unexpected commit for $submodule_path: $actual_commit" >&2
    echo "Expected: $expected_commit" >&2
    return 1
  fi

  if git -C "$absolute_submodule" apply --reverse --check "$absolute_patch" >/dev/null 2>&1; then
    echo "Already applied: $patch_path"
    return 0
  fi

  git -C "$absolute_submodule" apply --check "$absolute_patch"
  git -C "$absolute_submodule" apply "$absolute_patch"
  echo "Applied: $patch_path"
}

apply_patch_once \
  "external/compression/FPCompress" \
  "97f037249bd28682bfd83462ea1129e14df36d81" \
  "patches/vendor/fpcompress-tscb.patch"

apply_patch_once \
  "external/compression/lzbench" \
  "ec823b970911ee4edaf16bdce043be95a006f24b" \
  "patches/vendor/lzbench-tscb.patch"
