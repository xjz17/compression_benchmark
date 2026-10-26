# Code Export Manifest

This repository was assembled from the active `web_compression` working tree
as a source-only export. The source working tree was not modified.

## Included

- first-party C++, Python, Java, JavaScript, Vue, C, and Rust source code;
- build definitions, package manifests, tests, and small deterministic fixtures;
- method/operator catalogs and interface schemas;
- implementation and developer documentation;
- static-site source and payload-generation code;
- reusable operator-timing and algorithm-by-operator experiment harnesses;
- pinned third-party submodule commits and two project-specific vendor patches.

## Excluded

- raw, downloaded, cleaned, typed, or transformed benchmark datasets;
- final experiment results, score tables, ledgers, recommendation labels, and
  deployment payloads derived from measured results;
- JSONL result streams, archives, checkpoints, progress snapshots, receipts,
  audits, screenshots, and experiment logs;
- paper sources and generated publication figures, tables, PDFs, and slides;
- build trees, object files, executables, shared/static libraries, JARs, and
  frontend distribution bundles;
- Python virtual environments, `node_modules`, caches, IDE state, local Git
  worktrees, and nested Git histories;
- private files, credentials, local machine paths, and server-specific run state.

## External dependencies

The 41 external repositories listed in `.gitmodules` are represented by Git
submodule pins rather than copied worktrees. Local changes to FPCompress and
lzbench are preserved under `patches/vendor/` and can be applied with
`scripts/apply_vendor_patches.sh`.

## Generated outputs

Generated data should be written outside the repository or into ignored paths
such as `results/`, `logs/`, or `artifacts/`. Do not commit measured results to
the source branch.
