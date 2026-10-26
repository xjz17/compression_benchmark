#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/benchmark/reference/golden"
cd "${ROOT}/external/compression/camel"
mvn -q -DskipTests package
cd "${ROOT}/java"
mvn -q -DskipTests package
java -cp "${ROOT}/java/target/benchmark-java-1.0-SNAPSHOT.jar" org.example.CamelFloatGoldenExport "${OUT}"
echo "Camel float goldens + manifest.tsv (camel_float/) -> ${OUT}"
