# Compression Benchmark

This repository contains the implementation of a reproducible benchmark for
lossless compression of numeric and time-series columns. It is a code-only
distribution: raw/processed datasets, benchmark outputs, experiment logs, and
publication artifacts are intentionally not included.

The codebase includes:

- C++ codecs, operator implementations, unified harnesses, and tests;
- Python benchmark orchestration and the web API;
- Java codec integrations;
- the Vue frontend and static benchmark-site builder;
- method/operator catalogs, schemas, and small synthetic or golden fixtures;
- standalone operator-timing and algorithm-by-operator experiment harnesses.

## Repository layout

| Path | Purpose |
| --- | --- |
| `cpp/` | Native algorithms, operators, unified interfaces, harnesses, and tests |
| `backend/` | Django API and Python codec bridges |
| `java/` | Java codec implementations and benchmark bridge |
| `frontend/` | Vue/Vite user interface |
| `benchmark/reference/` | Method/operator catalogs, schemas, and small test fixtures |
| `experiments/` | Reusable timing and Cartesian-composition experiment runners |
| `deploy/huggingface-compressionbenchmark/` | Static-site source and payload builders |
| `external/` | Git submodules and source-only operator dependencies |
| `patches/vendor/` | Project-specific changes on top of pinned third-party commits |
| `docs/` | Interfaces, implementation notes, and harness documentation |

See [`EXPORT_MANIFEST.md`](EXPORT_MANIFEST.md) for the exact inclusion and
exclusion policy.

## Clone and initialize dependencies

```bash
git clone https://github.com/xjz17/compression_benchmark.git
cd compression_benchmark
./scripts/init_compression_submodules.sh
./scripts/apply_vendor_patches.sh
```

If the repository was cloned without submodules, run:

```bash
./scripts/init_compression_submodules.sh
./scripts/apply_vendor_patches.sh
```

The initialization script checks out top-level dependencies without recursing
into broken upstream nested gitlinks; FCBench is therefore left for an explicit
manual checkout when that integration is needed. The patch script is
idempotent: it applies the two maintained vendor patches only when they have not
already been applied.

## Build the native benchmark

On Debian or Ubuntu, install the native build prerequisites first:

```bash
sudo apt-get install cmake build-essential nlohmann-json3-dev
```

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

Individual harnesses under `cpp/` and `experiments/` contain their own focused
entry points. The canonical algorithm I/O contract is documented in
[`docs/canonical182_algorithm_io.md`](docs/canonical182_algorithm_io.md), and
the operator catalog is under [`benchmark/reference/actions/`](benchmark/reference/actions/).

## Python API

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python backend/manage.py migrate
python backend/manage.py runserver
```

Dataset roots and native executable paths are configured at runtime; no local
dataset or compiled binary is embedded in this repository. The complete Django
test suite exercises native binaries and optional result-backed views; build the
native targets and provide those optional inputs before running it. The default
CI runs the source-only unit-test subset that has no dataset/result dependency.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

## Java components

```bash
cd java
mvn clean package -DskipTests
```

## Experiment runners

- [`experiments/operator_timing/`](experiments/operator_timing/) measures
  operator encode/decode timing while verifying exact round trips.
- [`experiments/operator_algorithm_cartesian/`](experiments/operator_algorithm_cartesian/)
  evaluates measured operator/algorithm compositions without treating
  independently optimal stages as a measured joint optimum.

Both runners write their outputs to caller-provided directories. Keep large
outputs outside the repository or under an ignored `results/` directory.

## Data and results

The benchmark expects caller-supplied, typed numeric columns. The repository
ships only small deterministic fixtures needed by tests. It does **not** ship:

- raw or cleaned benchmark datasets;
- final compression ratios or timing matrices;
- remote-run snapshots, receipts, or logs;
- generated paper figures, tables, PDFs, or deployed site payloads.

This separation keeps the repository reviewable and prevents measured outputs
from being mistaken for source code.

## Third-party code

Third-party projects are pinned as Git submodules and retain their upstream
licenses. Project-specific changes for FPCompress and lzbench are stored as
versioned patches; see [`patches/vendor/README.md`](patches/vendor/README.md).
