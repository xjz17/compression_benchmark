#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/benchmark/reference/golden/chimp32"
cd "${ROOT}/java"
mvn -q -DskipTests package
cd "${ROOT}/external/compression/chimp"
mvn -q -DskipTests package
CP="${ROOT}/java/target/benchmark-java-1.0-SNAPSHOT.jar:${ROOT}/external/compression/chimp/target/benchmark.jar"
java -cp "${CP}" org.example.Chimp32GoldenExport "${OUT}"
echo "Chimp32 goldens + manifest.tsv -> ${OUT}"
