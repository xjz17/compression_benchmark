# Double-blind review boundary

This repository is an anonymous, source-only artifact prepared for double-blind
review. It is intentionally independent of the development repository, with
neutral author and committer metadata throughout its review history.

The artifact excludes:

- author names, usernames, email addresses, affiliations, and self-identifying
  repository links;
- private hostnames, IP addresses, local absolute paths, credentials, and remote
  execution receipts;
- author-owned deployment and dataset-mirror utilities;
- raw or processed benchmark datasets, measured result matrices, logs, paper
  sources, and generated publication assets.

Names, email addresses, URLs, and copyright notices that remain under
`external/`, `.gitmodules`, or `patches/vendor/` identify third-party projects
and are retained for attribution and license compliance. They do not identify
the artifact authors.

The public review entry point is:

<https://anonymous.4open.science/r/compression_benchmark>

Do not replace this link with the private source repository during review.

## Content checks

Run `python3 scripts/audit_anonymity.py` on the source checkout or an extracted
review ZIP. The scanner includes itself and checks constant Python string
concatenations. Its default scan checks generic paths and secrets; author-specific
checks require `--private-terms /absolute/path/to/private-terms.txt`. Keep that
UTF-8 file outside this repository, with one literal search term per line.
Never publish private names, accounts, hosts, or IP addresses as a deny-list.

For a source checkout, `--check-git-identities` verifies author/committer metadata
only. `--check-history` also scans the content of every commit reachable from
HEAD; it rejects an earlier identifying version even after the current files
have been repaired. Updating current files does not erase earlier Git revisions.
The review ZIP contains the current snapshot without Git history; audit that
download separately and verify its file hashes with
`python3 scripts/verify_manifest.py`.
