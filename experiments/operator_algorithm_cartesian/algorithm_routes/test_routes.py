#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
BUILDER = ROOT / "build_routes.py"
sys.path.insert(0, str(ROOT.parent / "source"))
from carrier import ACTIVE85  # type: ignore  # noqa: E402


def write_route_fixtures(root: Path) -> tuple[Path, Path]:
    """Create the smallest source-only route inputs accepted by the builder."""
    old84_path = root / "old84.json"
    bos_path = root / "bos.json"
    groups: list[list[str]] = [[] for _ in range(7)]
    for index, method in enumerate(name for name in ACTIVE85 if name != "BOS"):
        groups[index % len(groups)].append(method)
    routes = [
        {
            "route_id": f"synthetic-reviewed-route-{index}",
            "methods": methods,
            "binary": f"/runtime/route-{index}",
            "binary_sha256": f"{index + 1:x}" * 64,
            "command_profile": "standard_measure_decode",
            "configuration": f"fixture-{index}",
        }
        for index, methods in enumerate(groups)
    ]
    old84_path.write_text(json.dumps({
        "schema": "native85-nonnull-method-routes-v1",
        "routes": routes,
    }))
    bos_path.write_text(json.dumps({
        "schema": "new15-algorithm85-method-routes-v1",
        "routes": [{
            "route_id": "bos-bfe-ngf-hybrid-full-v1",
            "methods": ["BOS"],
            "binary": "/runtime/bos",
            "binary_sha256": "b" * 64,
            "command_profile": "standard_measure_decode",
            "configuration": "bos-bfe-ngf-hybrid-full-v1",
        }],
    }))
    return old84_path, bos_path


class RouteManifestTest(unittest.TestCase):
    def test_exact_active85_and_carrier_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            old84, bos = write_route_fixtures(temporary_root)
            output = temporary_root / "routes.json"
            subprocess.run([
                sys.executable, str(BUILDER), "--old84", str(old84),
                "--bos", str(bos), "--corad-binary", "/runtime/corad",
                "--corad-binary-sha256", "a" * 64, "--output", str(output),
            ], check=True, capture_output=True, text=True)
            value = json.loads(output.read_text())
            self.assertEqual(value["algorithms"], 85)
            self.assertEqual(value["carrier_counts"], {
                "direct_u8": 59, "int64_u56": 13, "float64_u48": 12,
                "float64_u8": 1,
            })
            records = value["records"]
            self.assertEqual(len({row["algorithm"] for row in records}), 85)
            self.assertEqual(next(row for row in records if row["algorithm"] == "CORAD")["binary"],
                             "/runtime/corad")
            self.assertEqual(next(row for row in records if row["algorithm"] == "BOS")["route_id"],
                             "bos-bfe-ngf-hybrid-full-v1")
            self.assertIn("PPMD8", {row["algorithm"] for row in records})
            camel = next(row for row in records if row["algorithm"] == "Camel")
            self.assertEqual(camel["carrier_kind"], "float64_u8")


if __name__ == "__main__":
    unittest.main()
