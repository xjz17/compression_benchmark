#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <web-repo> <tsfile-repo> <influxdb-repo> <dataset-dir> <output-dir>" >&2
  exit 2
fi

web_repo=$1
tsfile_repo=$2
influx_repo=$3
dataset_dir=$4
output_dir=$5
input_dir="$output_dir/inputs"
work_dir="$output_dir/work"
manifest="$output_dir/input_manifest.tsv"
results="$output_dir/system_codec_results.csv"
summary="$output_dir/system_codec_summary.csv"
environment_manifest="$output_dir/environment_manifest.tsv"
command_manifest="$output_dir/command.txt"

for repository in "$web_repo" "$tsfile_repo" "$influx_repo"; do
  if [[ -n $(git -C "$repository" status --porcelain) ]]; then
    echo "repository must be clean before benchmarking: $repository" >&2
    exit 1
  fi
done

mkdir -p "$input_dir" "$work_dir"
{
  printf '%q' "$0"
  for argument in "$@"; do
    printf ' %q' "$argument"
  done
  printf '\n'
} >"$command_manifest"

python3 "$web_repo/scripts/prepare_system_codec_inputs.py" \
  --dataset-dir "$dataset_dir" \
  --output-dir "$input_dir" \
  --manifest "$manifest" \
  --count 5 \
  --require-paper-19

cd "$tsfile_repo"
./mvnw -P with-java -pl :examples -am -DskipTests -Dcheckstyle.skip=true -Drat.skip=true package
./mvnw -P with-java -pl :examples dependency:build-classpath \
  -Dmdep.outputFile="$work_dir/tsfile_classpath.txt" \
  -DincludeScope=runtime
tsfile_classpath="$tsfile_repo/java/examples/target/classes:$tsfile_repo/java/tsfile/target/classes:$tsfile_repo/java/common/target/classes:$(<"$work_dir/tsfile_classpath.txt")"

cd "$influx_repo"
go_bin=${GO_BIN:-go}
flux_dir="$($go_bin list -m -f '{{.Dir}}' github.com/influxdata/flux)"
libflux_target="$work_dir/libflux-target"
libflux_pkgconfig="$work_dir/libflux-pkgconfig"
mkdir -p "$libflux_pkgconfig"
CARGO_TARGET_DIR="$libflux_target" cargo build \
  --manifest-path "$flux_dir/libflux/Cargo.toml" \
  --release \
  -p flux
cat >"$libflux_pkgconfig/flux.pc" <<EOF
prefix=$libflux_target/release
includedir=$flux_dir/libflux/include
libdir=\${prefix}

Name: flux
Description: Flux parser C API
Version: 0.200.0
Libs: -L\${libdir} -lflux
Cflags: -I\${includedir}
EOF
export PKG_CONFIG_PATH="$libflux_pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export DYLD_LIBRARY_PATH="$libflux_target/release${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$libflux_target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
influx_bench="$work_dir/tsm-codec-bench"
$go_bin build -o "$influx_bench" ./cmd/tsm-codec-bench

{
  printf 'key\tvalue\n'
  printf 'started_at\t%s\n' "$(date -Is)"
  printf 'hostname\t%s\n' "$(hostname)"
  printf 'uname\t%s\n' "$(uname -a)"
  printf 'java\t%s\n' "$(java -version 2>&1 | head -n 1)"
  printf 'go\t%s\n' "$($go_bin version)"
  printf 'rustc\t%s\n' "$(rustc --version)"
  printf 'cargo\t%s\n' "$(cargo --version)"
  printf 'web_commit\t%s\n' "$(git -C "$web_repo" rev-parse HEAD)"
  printf 'tsfile_commit\t%s\n' "$(git -C "$tsfile_repo" rev-parse HEAD)"
  printf 'influxdb_commit\t%s\n' "$(git -C "$influx_repo" rev-parse HEAD)"
  printf 'web_dirty\t%s\n' "$([[ -n $(git -C "$web_repo" status --porcelain) ]] && printf true || printf false)"
  printf 'tsfile_dirty\t%s\n' "$([[ -n $(git -C "$tsfile_repo" status --porcelain) ]] && printf true || printf false)"
  printf 'influxdb_dirty\t%s\n' "$([[ -n $(git -C "$influx_repo" status --porcelain) ]] && printf true || printf false)"
  printf 'iterations\t10\n'
  printf 'warmups\t3\n'
} >"$environment_manifest"

rm -f "$results"
first_header=1
tail -n +2 "$manifest" | while IFS=$'\t' read -r dataset source source_bytes source_sha256 rows columns input input_sha256; do
  tsfile_csv="$work_dir/${dataset}_tsfile.csv"
  influx_csv="$work_dir/${dataset}_influxdb.csv"
  java -cp "$tsfile_classpath" org.apache.tsfile.SystemCodecBenchmark \
    "$input" "$work_dir/tsfile" "$dataset" 10 3 \
    | awk -F, '$1 == "system" || $1 == "tsfile"' >"$tsfile_csv"
  "$influx_bench" "$input" "$work_dir/influxdb" "$dataset" 10 3 >"$influx_csv"
  if [[ $first_header -eq 1 ]]; then
    head -n 1 "$tsfile_csv" >"$results"
    first_header=0
  fi
  tail -n +2 "$tsfile_csv" >>"$results"
  tail -n +2 "$influx_csv" >>"$results"
done

python3 - "$results" "$summary" "$manifest" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
summary_path = sys.argv[2]
manifest_path = sys.argv[3]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("no benchmark rows")
failed = [row for row in rows if row["hash_ok"].lower() != "true"]
if failed:
    raise SystemExit(f"hash validation failed for {len(failed)} rows")
groups = defaultdict(list)
for row in rows:
    groups[(row["system"], row["dataset"], row["codec"])].append(row)
expected_codecs = {
    "tsfile": {"BOS", "SUBCOLUMN", "PLAIN", "GORILLA_V1", "GORILLA", "CHIMP", "RLBE", "CAMEL"},
    "influxdb": {"BOS", "SUBCOLUMN", "GORILLA", "INTEGER_DELTA_ADAPTIVE_BITCAST"},
}
datasets = {row["dataset"] for row in rows}
with open(manifest_path, newline="", encoding="utf-8") as handle:
    manifest_datasets = {row["dataset"] for row in csv.DictReader(handle, delimiter="\t")}
if len(manifest_datasets) != 5 or datasets != manifest_datasets:
    raise SystemExit(
        f"result datasets do not match five-dataset manifest: results={sorted(datasets)}, "
        f"manifest={sorted(manifest_datasets)}"
    )
expected_groups = {
    (system, dataset, codec)
    for system, codecs in expected_codecs.items()
    for dataset in datasets
    for codec in codecs
}
actual_groups = set(groups)
if actual_groups != expected_groups:
    missing = sorted(expected_groups - actual_groups)
    extra = sorted(actual_groups - expected_groups)
    raise SystemExit(f"invalid result matrix: missing={missing}, extra={extra}")
bad_repetitions = {key: len(values) for key, values in groups.items() if len(values) != 10}
if bad_repetitions:
    raise SystemExit(f"expected 10 repetitions per group: {bad_repetitions}")
expected_iterations = {str(iteration) for iteration in range(10)}
bad_iterations = sorted(
    key
    for key, values in groups.items()
    if {row["iteration"] for row in values} != expected_iterations
)
if bad_iterations:
    raise SystemExit(f"expected unique iterations 0..9 per group: {bad_iterations}")
with open(summary_path, "w", newline="", encoding="utf-8") as handle:
    fieldnames = [
        "system",
        "dataset",
        "codec",
        "repetitions",
        "rows",
        "columns",
        "median_write_ms",
        "median_read_ms",
        "median_file_bytes",
        "all_hash_ok",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for key in sorted(groups):
        values = groups[key]
        writer.writerow(
            {
                "system": key[0],
                "dataset": key[1],
                "codec": key[2],
                "repetitions": len(values),
                "rows": values[0]["rows"],
                "columns": values[0]["columns"],
                "median_write_ms": f'{statistics.median(float(row["write_ms"]) for row in values):.6f}',
                "median_read_ms": f'{statistics.median(float(row["read_ms"]) for row in values):.6f}',
                "median_file_bytes": f'{statistics.median(int(row["file_bytes"]) for row in values):.0f}',
                "all_hash_ok": "true",
            }
        )
print(f"validated_rows={len(rows)}")
PY

echo "manifest=$manifest"
echo "results=$results"
echo "summary=$summary"
echo "environment=$environment_manifest"
echo "command=$command_manifest"
