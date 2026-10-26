# Code Export Manifest

This repository is a source-only, double-blind review artifact. It was built as
an independent snapshot with no development-repository history.

## Included

- first-party C++, Python, Java, JavaScript, Vue, C, and Rust source code;
- build definitions, package manifests, tests, and small deterministic fixtures;
- method/operator catalogs and interface schemas;
- implementation and developer documentation;
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
- author-owned deployment and dataset-mirror utilities, institutional branding,
  and links to non-anonymous project pages;
- third-party source trees without a redistributable license; these are pinned
  as upstream references instead.

## External dependencies

External repositories listed in `.gitmodules` are represented by Git submodule
pins rather than copied worktrees. Local changes are preserved under
`patches/vendor/` and can be applied with `scripts/apply_vendor_patches.sh`.
The upstream identities and licenses are summarized in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Generated outputs

Generated data should be written outside the repository or into ignored paths
such as `results/`, `logs/`, or `artifacts/`. Do not commit measured results to
the source branch.
