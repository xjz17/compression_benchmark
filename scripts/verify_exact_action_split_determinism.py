#!/usr/bin/env python3
"""Run the requested split audits twice and record byte-level determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITS = {
    "requested": (
        ROOT / "scripts/audit_requested_codec_splits_realdata.py",
        ("requested_split_realdata_rows.csv", "requested_split_realdata_summary.json"),
    ),
    "general": (
        ROOT / "scripts/audit_general_codec_splits_realdata.py",
        ("general_split_realdata_rows.csv", "general_split_realdata_summary.json"),
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema": "exact-action-split-determinism-v1",
        "rows_per_column_limit": args.rows,
        "repeats": 2,
        "audits": {},
    }
    all_equal = True
    with tempfile.TemporaryDirectory(prefix="webcomp-exact-split-") as temp:
        temp_root = Path(temp)
        for audit_name, (script, filenames) in AUDITS.items():
            repeat_hashes: list[dict[str, str]] = []
            for repeat in (1, 2):
                output_dir = temp_root / audit_name / f"repeat_{repeat}"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--rows",
                        str(args.rows),
                        "--output-dir",
                        str(output_dir),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{audit_name} repeat {repeat} failed: "
                        f"{completed.stderr.strip() or completed.stdout.strip()}"
                    )
                repeat_hashes.append(
                    {filename: sha256(output_dir / filename) for filename in filenames}
                )
            equal = repeat_hashes[0] == repeat_hashes[1]
            all_equal = all_equal and equal
            result["audits"][audit_name] = {
                "byte_identical": equal,
                "repeat_hashes": repeat_hashes,
            }
    result["status"] = "PASS" if all_equal else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
