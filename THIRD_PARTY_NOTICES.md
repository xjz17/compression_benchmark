# Third-party notices

This artifact benchmarks and integrates open-source compression projects. The
upstream project name, URL, and pinned revision for Git-submodule dependencies
are recorded in `.gitmodules` and the repository tree. Each dependency remains
subject to its upstream license.

Project-specific changes are provided as reviewable patches under
`patches/vendor/`; applying a patch does not replace the upstream license.

Additional source-only dependencies retained in this artifact include:

- BUFF (`external/operator_ablation_sources/buff-rs`), under its included
  license;
- Density (`external/operator_ablation_sources/density_c_0_14_2`), under its
  included license;
- Sprintz (`external/operator_ablation_sources/sprintz`), under its included
  license;
- TSXor, referenced from `https://github.com/andybbruno/TSXor.git` at revision
  `16899d15f54176c3a15babe138d795a7c4633877`;
- the legacy lzbench source used for Pithy, Shrinker, and XPACK experiments,
  referenced from `https://github.com/inikep/lzbench.git` at revision
  `f2e93cd2e5c1445e6057f49225df7ea71c9df83f`.

Third-party author names and project URLs are intentionally not anonymized: they
are citations and license metadata, not identities of this artifact's authors.
