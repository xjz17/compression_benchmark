#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/benchmark/reference/golden/fastpfor"
cd "${ROOT}/java"
mvn -q -DskipTests package
java -cp "${ROOT}/java/target/benchmark-java-1.0-SNAPSHOT.jar" \
  org.example.FastPFORGoldenExport "${OUT}"
echo "FastPFOR goldens + manifest.tsv -> ${OUT}"
