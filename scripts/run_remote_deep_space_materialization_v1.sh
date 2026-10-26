#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 CAMPAIGN_ROOT [CPU_ID] [OUTPUT_TAG]" >&2
  exit 2
fi

campaign_root=$1
cpu_id=${2:-63}
output_tag=${3:-deep_space_probes_exact_upstream_logical_v2}
source_root="$campaign_root/source"
exact_root="$campaign_root/data/exact_sources/deep_space_probes/ed64a3ba6e24baccb33574d3b36562ba5c1b7332"
source_parquet="$exact_root/data/deep_space_probes.parquet"
result_root="$campaign_root/results/data_materialization/$output_tag"
canonical_root="$campaign_root/data/canonical/$output_tag"
prepared_root="$campaign_root/data/prepared/$output_tag"
spec_path="$result_root/deep_space_probes_formal_spec_remote_v1.json"
audit_path="$result_root/deep_space_probes_source_audit_remote_v1.json"
receipt_path="$result_root/deep_space_probes_materialization_receipt_v1.json"

if [[ ! -f "$source_parquet" ]]; then
  echo "missing exact upstream Parquet: $source_parquet" >&2
  exit 2
fi
if [[ -e "$canonical_root" || -e "$prepared_root" || -e "$receipt_path" ]]; then
  echo "refusing to overwrite an existing canonical, prepared, or receipt artifact" >&2
  exit 2
fi

actual_source_sha=$(sha256sum "$source_parquet" | awk '{print $1}')
expected_source_sha=dd2b043ad74d1e049e7d1d6a85a51feb5a12244ef0cee6327e8cc6d80f0c76e8
if [[ "$actual_source_sha" != "$expected_source_sha" ]]; then
  echo "exact upstream Parquet SHA-256 mismatch" >&2
  exit 2
fi

mkdir -p "$result_root" "$(dirname "$canonical_root")" "$(dirname "$prepared_root")"
cd "$source_root"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

runner=(nice -n 19 ionice -c 3 taskset -c "$cpu_id" python3)

"${runner[@]}" scripts/rebuild_weighted_grpo_deep_space_v1.py generate-spec \
  --source "$source_parquet" \
  --output "$spec_path"
"${runner[@]}" scripts/rebuild_weighted_grpo_deep_space_v1.py audit \
  --spec "$spec_path" \
  --source "$source_parquet" \
  --output "$audit_path"
"${runner[@]}" scripts/rebuild_weighted_grpo_deep_space_v1.py validate-receipt \
  --spec "$spec_path" \
  --audit "$audit_path"
"${runner[@]}" scripts/rebuild_weighted_grpo_deep_space_v1.py rebuild \
  --spec "$spec_path" \
  --source "$source_parquet" \
  --output "$canonical_root"
"${runner[@]}" scripts/rebuild_weighted_grpo_deep_space_v1.py verify \
  --input "$canonical_root"
"${runner[@]}" scripts/rebuild_weighted_grpo_deep_space_v1.py prepare \
  --canonical "$canonical_root" \
  --output "$prepared_root"
"${runner[@]}" scripts/prepare_weighted_grpo_timeseries_v1.py verify \
  --input "$prepared_root"

"${runner[@]}" - "$source_parquet" "$spec_path" "$audit_path" \
  "$canonical_root" "$prepared_root" "$receipt_path" "$cpu_id" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source, spec, audit, canonical_root, prepared_root, receipt = map(Path, sys.argv[1:7])
canonical_manifest = canonical_root / "canonical_manifest.json"
prepared_manifest = prepared_root / "dataset_manifest.json"
canonical = json.loads(canonical_manifest.read_text(encoding="utf-8"))
prepared = json.loads(prepared_manifest.read_text(encoding="utf-8"))
document = {
    "schema": "weighted-grpo-deep-space-remote-materialization-receipt-v1",
    "status": "verified",
    "dataset_id": "deep-space-probes",
    "source": {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    },
    "formal_spec": {"path": str(spec), "sha256": sha256_file(spec)},
    "source_audit": {"path": str(audit), "sha256": sha256_file(audit)},
    "canonical": {
        "path": str(canonical_root),
        "manifest_sha256": sha256_file(canonical_manifest),
        "provenance_sha256": canonical["provenance_sha256"],
        "rows": canonical["totals"]["rows"],
        "timestamp_only_rows": canonical["totals"]["timestamp_only_rows"],
        "blocks": canonical["totals"]["blocks"],
        "original_numeric_bytes": canonical["totals"]["original_numeric_bytes"],
    },
    "prepared": {
        "path": str(prepared_root),
        "manifest_sha256": sha256_file(prepared_manifest),
        "provenance_sha256": prepared["provenance_sha256"],
        "source_rows": prepared["selection"]["source_rows"],
        "selected_rows": prepared["selection"]["selected_rows"],
        "blocks": prepared["totals"]["blocks"],
        "original_numeric_bytes": prepared["totals"]["original_numeric_bytes"],
        "missing_bitmap_bytes": prepared["totals"]["missing_bitmap_bytes"],
    },
    "verification": {
        "canonical_independent_cold_reread": True,
        "prepared_independent_full_verify": True,
        "worker_cpu_limit": 1,
        "cpu_id": int(sys.argv[7]),
    },
}
payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
temporary = receipt.with_name(receipt.name + ".tmp")
temporary.write_text(payload, encoding="utf-8")
temporary.replace(receipt)
print(json.dumps(document, sort_keys=True))
PY

sha256sum "$receipt_path"
