#!/usr/bin/env python3
"""Regression checks for self-scanning, external terms, and review archives."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("artifact_anonymity", SCRIPTS / "audit_anonymity.py")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)


class AnonymityToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "artifact"
        self.root.mkdir()

    def test_scanner_does_not_exempt_itself_or_split_strings(self) -> None:
        script = self.root / "audit_anonymity.py"
        script.write_bytes((SCRIPTS / script.name).read_bytes()
                           + b'\nidentity = "reviewer-" + "private-sentinel"\n')
        problems, count, checkout = SCANNER.audit(self.root, (b"reviewer-private-sentinel",), False)
        self.assertEqual(count, 1)
        self.assertFalse(checkout)
        self.assertIn("private identity/infrastructure term in audit_anonymity.py", problems)

    def test_generic_personal_path_is_rejected(self) -> None:
        private_path = "/".join(("", "Users", "private-person", "fixture"))
        (self.root / "sample.py").write_text(f"value = {private_path!r}\n")
        problems, _, _ = SCANNER.audit(self.root, (), False)
        self.assertIn("personal absolute path in sample.py", problems)

    def test_upstream_attribution_is_retained_but_private_terms_still_apply(self) -> None:
        source = self.root / "external" / "upstream.cpp"
        source.parent.mkdir()
        source.write_text("// " + "/".join(("", "Users", "upstream-author", "fixture")))
        self.assertEqual(SCANNER.audit(self.root, (), False)[0], [])
        self.assertTrue(SCANNER.audit(self.root, (b"upstream-author",), False)[0])

    def test_loopback_address_and_python_escapes_are_not_private_paths(self) -> None:
        (self.root / "sample.py").write_text('message = "output:\\n{result}\\n"\n')
        (self.root / "dev.sh").write_text("serve --host 127.0.0.1\n")
        self.assertEqual(SCANNER.audit(self.root, (), False)[0], [])

    def test_private_terms_inside_artifact_are_rejected(self) -> None:
        terms = self.root / "terms.txt"
        terms.write_text("private-sentinel\n")
        result = subprocess.run([sys.executable, str(SCRIPTS / "audit_anonymity.py"),
                                 "--root", str(self.root), "--private-terms", str(terms)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("outside the artifact", result.stderr)

    def test_archive_does_not_claim_git_history_was_verified(self) -> None:
        (self.root / "README.md").write_text("Anonymous artifact\n")
        result = subprocess.run([sys.executable, str(SCRIPTS / "audit_anonymity.py"),
                                 "--root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("private terms not supplied; Git history not checked", result.stdout)
        self.assertTrue(SCANNER.audit(self.root, (), True)[0])

    def test_history_scan_rejects_repaired_ancestor(self) -> None:
        command = ["git", "-C", str(self.root)]
        subprocess.run(command + ["init", "-q"], check=True)
        subprocess.run(command + ["config", "user.name", "Anonymous Authors"], check=True)
        subprocess.run(command + ["config", "user.email", "anonymous@invalid"], check=True)
        source = self.root / "sample.txt"
        source.write_text("/".join(("", "Users", "private-person", "fixture")))
        subprocess.run(command + ["add", "sample.txt"], check=True)
        subprocess.run(command + ["commit", "-qm", "Initial fixture"], check=True)
        source.write_text("Anonymous fixture\n")
        subprocess.run(command + ["commit", "-qam", "Repair fixture"], check=True)
        self.assertEqual(SCANNER.audit(self.root, (), False, True)[0], [])
        problems = SCANNER.audit(self.root, (), True)[0]
        self.assertTrue(any("historical commit" in problem for problem in problems))

    def test_manifest_verifies_archive_and_rejects_deleted_file(self) -> None:
        source = self.root / "README.md"
        payload = b"Anonymous artifact\n"
        source.write_bytes(payload)
        manifest = self.root / "MANIFEST.sha256"
        manifest.write_text(f"{hashlib.sha256(payload).hexdigest()}  README.md\n")
        command = [sys.executable, str(SCRIPTS / "verify_manifest.py"), "--root", str(self.root)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        source.unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing", result.stdout)

    def test_manifest_rejects_changed_bytes(self) -> None:
        source = self.root / "README.md"
        source.write_text("Changed bytes\n")
        (self.root / "MANIFEST.sha256").write_text(f"{hashlib.sha256(b'old').hexdigest()}  README.md\n")
        result = subprocess.run([sys.executable, str(SCRIPTS / "verify_manifest.py"),
                                 "--root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("digest mismatch", result.stdout)


if __name__ == "__main__":
    unittest.main()
