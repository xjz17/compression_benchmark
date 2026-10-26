#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/benchmark/reference/golden"
cd "${ROOT}/java"
mvn -q -DskipTests package
BENCH_JAR="${ROOT}/java/target/benchmark-java-1.0-SNAPSHOT.jar"
if [[ ! -f "${BENCH_JAR}" ]]; then
  echo "missing ${BENCH_JAR}" >&2
  exit 1
fi
java -cp "${BENCH_JAR}" org.example.SimPieceFloatGoldenExport "${OUT}"
echo "Sim-Piece float goldens + manifest (simpiece_float/) -> ${OUT}"
