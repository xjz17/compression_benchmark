# Vendor patches

The external projects remain pinned to their upstream commits as Git
submodules. Three submodule snapshots have maintained project-specific changes:

| Submodule | Pinned base commit | Patch |
| --- | --- | --- |
| FPCompress | `97f037249bd28682bfd83462ea1129e14df36d81` | `fpcompress-tscb.patch` |
| lzbench | `ec823b970911ee4edaf16bdce043be95a006f24b` | `lzbench-tscb.patch` |
| lzbench legacy ablation source | `f2e93cd2e5c1445e6057f49225df7ea71c9df83f` | `lzbench-legacy-ablation.patch` |

Apply the patches from the repository root with:

```bash
./scripts/apply_vendor_patches.sh
```

The script verifies that each submodule is at its expected base commit and
refuses to apply a patch to an unrelated checkout.

To apply only selected patches, pass their submodule paths. For example:

```bash
./scripts/apply_vendor_patches.sh external/operator_ablation_sources/lzbench_legacy
```
