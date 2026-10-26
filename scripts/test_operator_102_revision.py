"""Native-core and topology checks for the 102-item source catalog."""
from pathlib import Path
import hashlib
import json
import random
import struct
import subprocess
import unittest
import operator_implementation_catalog as catalog

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis_outputs/operator_final_102_20260916"


class RevisionTests(unittest.TestCase):
    def test_topology_and_classification(self):
        self.assertEqual(catalog.IMPLEMENTATION_COUNT, 102)
        self.assertEqual(catalog.BY_ID["A013"]["family_numbers"], [3])
        self.assertEqual(catalog.BY_ID["A035"]["family_numbers"], [10])
        self.assertEqual(catalog.legacy_replacement("A034"), ("A031", "A039"))
        self.assertEqual(catalog.legacy_replacement("A025"), ("A020",))
        for key in ("A024", "A036"):
            with self.assertRaises(ValueError):
                catalog.legacy_replacement(key)
        recipe = catalog.replacement_recipe("A024")
        self.assertEqual(recipe["branches"]["exception_positions"]["width"], 32)
        self.assertEqual(recipe["branches"]["exception_values"]["width"], 64)
        recipe = catalog.replacement_recipe("A036")
        self.assertEqual(recipe["branches"], {"first": "anchor/reset", "literal": "A031", "decimal": "A028"})
        for record in catalog.inventory()["runtime_action_recipes"].values():
            self.assertLessEqual(set(record["recipe"]["replacement_implementation_ids"]), set(catalog.BY_ID))

    def test_old_wire_bytes_and_current_roundtrip(self):
        subprocess.run([str(RUN / "wire_after"), str(RUN / "golden_after")], check=True)
        before = sorted((RUN / "golden_before").glob("*.bin"))
        self.assertEqual(len(before), 71)
        for path in before:
            self.assertEqual(path.read_bytes(), (RUN / "golden_after" / path.name).read_bytes(), path.name)

    def test_standalone_residual_window_cold_decode(self):
        rng = random.Random(102)
        executable = str(RUN / "residual_window_pack_cli")
        for width in (32, 64):
            words = [0, 0, 1, 2, 3, (1 << width) - 1, 1 << (width - 1)]
            words += [1 << k for k in range(width)]
            words += [rng.getrandbits(width) for _ in range(300)]
            # IEEE special values are bit patterns here, with no eraser/EOS domain restriction.
            if width == 64:
                words += [0x8000000000000000, 0x7ff0000000000000, 0x7ff8000000001234, 1]
            raw = b"".join(v.to_bytes(width // 8, "little") for v in words)
            for kind in ("chimp", "elfplus"):
                profile = f"{kind}{width}"
                def call(mode, payload, check=True):
                    return subprocess.run([executable, mode, profile], input=payload, capture_output=True, check=check)
                for payload in (b"", raw):
                    wire = call("encode", payload).stdout
                    self.assertEqual(call("decode", wire).stdout, payload)
                    self.assertNotEqual(call("decode", wire + b"\x00", False).returncode, 0)
                    self.assertNotEqual(call("decode", wire[:-1], False).returncode, 0)
                forged = bytearray(call("encode", raw).stdout)
                forged[6:14] = struct.pack("<Q", 2**64 - 1)
                self.assertNotEqual(call("decode", forged, False).returncode, 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RevisionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = ["cpp/src/max_value_bitpack.hpp", "cpp/src/adjacent_residual.hpp",
               "cpp/src/residual_window_pack.hpp", "cpp/src/residual_window_pack_cli.cpp",
               "cpp/src/bos_exact_stages.cpp", "cpp/src/elf_float_codec.cpp",
               "cpp/src/elf_double_codec.cpp", "cpp/src/camel_exact_stages.cpp",
               "cpp/src/chimp128_exact_stages.cpp", "cpp/src/gorilla_exact_stages.cpp",
               "scripts/operator_implementation_catalog.py",
               "benchmark/reference/actions/operator_implementations_102.json"]
    (RUN / "revision_tests.json").write_text(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
        "wire_fixtures": 71, "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "scope": "local C++ correctness and old-wire parity; no performance benchmark",
        "known_preexisting_limit": "Elf32 uses a fixed 5000-byte output buffer; a high-entropy 2054-value probe exceeded it before these changes."
    }, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
