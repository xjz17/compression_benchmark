#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/benchmark/reference/golden"
cd "${ROOT}/external/compression/elf"
mvn -q -Dmaven.test.skip=true package
mvn -q -Dmaven.test.skip=true dependency:copy-dependencies
cd "${ROOT}/java"
mvn -q -DskipTests package
java -cp "${ROOT}/java/target/benchmark-java-1.0-SNAPSHOT.jar" org.example.ElfFloatGoldenExport "${OUT}"
echo "ELF float goldens + manifest.tsv (elf_on_chimp_float, elf_plus_float) -> ${OUT}"
