#!/usr/bin/env python3
"""Cold-process, cross-runtime DBF1/DBF2 wire and malformed-input verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import struct
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True)
    parser.add_argument("--java", required=True, help="quoted Java command including class")
    parser.add_argument("--python", required=True, help="quoted Python command including CLI script")
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    commands = {"cpp": shlex.split(args.cpp), "java": shlex.split(args.java), "python": shlex.split(args.python)}
    manifest = json.loads(Path(args.goldens).read_text())
    golden_root = Path(args.goldens).parent
    records = []
    cases = []
    for item in manifest["cases"]:
        raw = (golden_root / item["input_file"]).read_bytes()
        frame = (golden_root / item["frame_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["input_sha256"]
        assert hashlib.sha256(frame).hexdigest() == item["frame_sha256"]
        width = 4 if frame[:4] == b"DBF2" else 8
        cases.append((item["name"], width, item["max_point"], raw, frame))
    rng = random.Random(20261009)
    for width, code in [(4, "I"), (8, "Q")]:
        words = [rng.getrandbits(width * 8) for _ in range(511)]
        raw = struct.pack("<" + code * len(words), *words)
        cases.append((f"random_native_{width}", width, 18, raw, None))
        floats = [i * .25 for i in range(-64, 65)]
        raw = struct.pack("<" + ("f" if width == 4 else "d") * len(floats), *floats)
        cases.append((f"decimal_native_{width}", width, 2, raw, None))
    for name, width, cap, raw, expected in cases:
        encoded = {}
        for runtime, command in commands.items():
            encoded[runtime] = subprocess.run(command + ["encode32" if width == 4 else "encode", str(cap)], input=raw, capture_output=True, check=True).stdout
        assert len(set(encoded.values())) == 1, (name, "runtime encoder wire disagreement")
        frame = encoded["cpp"]
        if expected is not None:
            assert frame == expected, (name, "fixed golden disagreement")
        for source, wire in encoded.items():
            for target, command in commands.items():
                decoded = subprocess.run(command + ["decode"], input=wire, capture_output=True, check=True).stdout
                assert decoded == raw, (name, source, target, "original native bytes mismatch")
        records.append({"case": name, "width": width, "values": len(raw) // width,
                        "frame_bytes": len(frame), "input_sha256": hashlib.sha256(raw).hexdigest(),
                        "frame_sha256": hashlib.sha256(frame).hexdigest(), "cold_decoder_pairs": 9})
    # Bad headers, framing, count, first-word truncation, overlong and overflowing varints.
    invalid = [b"", b"DBF1", b"DBX1\x01\xff\0\0\0\0",
               b"DBF1\x02\xff\0\0\0\0", b"DBF1\x01\x00\0\0\0\0",
               b"DBF1\x00\x13\0\0\0\0", b"DBF1\x00\0\0\0\0\0",
               b"DBF1\x01\xff\0\0\0\0\0",
               b"DBF1\x01\xff" + struct.pack("<I", 10001),
               b"DBF1\x01\xff\x01\0\0\0" + b"\0" * 7]
    header = b"DBF1\x01\xff\x02\0\0\0" + b"\0" * 8
    invalid += [header, header + b"\x80", header + b"\x80\0", header + b"\xff" * 9 + b"\x02",
                header + b"\x80" * 10 + b"\0", header + b"\0\0",
                b"DBF2\x01\xff\x01\0\0\0" + struct.pack("<Q", 1 << 32)]
    for case_no, frame in enumerate(invalid):
        for runtime, command in commands.items():
            rejected = subprocess.run(command + ["decode"], input=frame, capture_output=True)
            assert rejected.returncode != 0, ("accepted malformed frame", case_no, runtime)
    receipt = {"status": "PASS", "random_seed": 20261009, "cases": records,
               "golden_manifest_sha256": hashlib.sha256(Path(args.goldens).read_bytes()).hexdigest(),
               "runtime_commands": commands, "cold_decoder_pairs": sum(r["cold_decoder_pairs"] for r in records),
               "malformed_cases": len(invalid), "malformed_runtime_rejections": len(invalid) * len(commands),
               "comparison": "original native bytes, including signed zero and NaN payload bits"}
    receipt["runtime_artifact_hashes"] = {
        runtime: {str(Path(arg).resolve()): hashlib.sha256(Path(arg).read_bytes()).hexdigest()
                  for arg in command if Path(arg).is_file()}
        for runtime, command in commands.items()
    }
    receipt["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "cases": len(records), "cold_decoder_pairs": receipt["cold_decoder_pairs"], "malformed_runtime_rejections": receipt["malformed_runtime_rejections"]}))


if __name__ == "__main__":
    main()
