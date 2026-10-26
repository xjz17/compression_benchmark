#!/usr/bin/env python3
"""Build the real NNCP/LibNC adapter on Linux x86-64, or explicit Docker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/prediction_probability/nncp-2024-06-05"
BUILD = ROOT / "build/nncp_probability"
DRIVER = BUILD / "nncp_probability_cli"
DOCKER_IMAGE = "gcc:14.3.0-bookworm"
VALIDATION = ROOT / "analysis_outputs/operator_consolidation_20260917/nncp_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprints() -> dict[str, str]:
    paths = [ROOT / "cpp/src/nncp_probability_bridge.c", ROOT / "cpp/src/nncp_probability_cli.cpp",
             ROOT / "cpp/include/nncp_probability.h", Path(__file__), UPSTREAM / "libnc.so"]
    paths += sorted(UPSTREAM.glob("*.c")) + sorted(UPSTREAM.glob("*.h"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def nncp_availability() -> dict:
    """Read-only capability check; never builds or starts a container/model."""
    native = platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64")
    override = os.environ.get("WEB_COMPRESSION_NNCP_DRIVER")
    driver = Path(override).expanduser().resolve() if override else DRIVER
    docker = os.environ.get("WEB_COMPRESSION_NNCP_USE_DOCKER") == "1" and not override
    result = {"available": False, "native_platform": native, "docker_configured": docker,
              "driver": str(driver), "driver_exists": driver.is_file(),
              "dependency_exists": (BUILD / "libnc.so").is_file(),
              "validated_current_build": False, "unit": "float32[256] next-byte probabilities",
              "profile": "transformer-stream1-v1"}
    stamp = _read_json(BUILD / "build.json")
    receipt = _read_json(VALIDATION)
    try:
        hashes = source_fingerprints()
        current = (driver.is_file() and (BUILD / "libnc.so").is_file()
                   and stamp.get("source_fingerprints") == hashes
                   and sha256(BUILD / "libnc.so") == hashes[str((UPSTREAM / "libnc.so").relative_to(ROOT))]
                   and receipt.get("result") == "passed"
                   and receipt.get("source_fingerprints") == hashes
                   and receipt.get("driver_sha256") == sha256(driver))
    except OSError:
        current = False
    result["validated_current_build"] = current
    if not current:
        result["reason"] = "NNCP independent adapter exists; current binary/dependency/source fingerprints require build and validation"
        return result
    if docker:
        try:
            daemon = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=5)
            image = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=5)
            ready = (daemon.returncode == 0 and daemon.stdout.strip() == "linux" and image.returncode == 0
                     and image.stdout.strip() == receipt.get("docker_image_id"))
        except (OSError, subprocess.SubprocessError):
            ready = False
        result["available"] = ready
        result["reason"] = ("Verified NNCP build; explicit Docker route, live Linux daemon and validated image match"
                            if ready else "NNCP build verified; explicit Docker route is not currently ready or image changed")
    else:
        result["available"] = native
        result["reason"] = ("Verified NNCP build and original LibNC on Linux x86-64"
                            if native else "NNCP requires Linux x86-64 or explicitly enabled and verified Docker execution")
    return result


def build_native(*, force: bool = False) -> Path:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("Bundled NNCP LibNC requires Linux x86-64; use explicit Docker mode")
    cc = os.environ.get("CC") or shutil.which("gcc")
    cxx = os.environ.get("CXX") or shutil.which("g++")
    if not cc or not cxx:
        raise RuntimeError("NNCP requires C and C++20 compilers")
    sources = [ROOT / "cpp/src/nncp_probability_bridge.c"] + [
        UPSTREAM / name for name in ("cmdopt.c", "cp_utils.c", "arith.c", "preprocess.c", "cutils.c")]
    dependencies = sources + [ROOT / "cpp/src/nncp_probability_cli.cpp", ROOT / "cpp/include/nncp_probability.h",
                              Path(__file__), UPSTREAM / "nncp.c", UPSTREAM / "libnc.so"] + sorted(UPSTREAM.glob("*.h"))
    digest = hashlib.sha256()
    digest.update(subprocess.check_output([cc, "--version"]))
    digest.update(subprocess.check_output([cxx, "--version"]))
    for path in dependencies:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    stamp = BUILD / "build.json"
    BUILD.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (BUILD / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _read_json(stamp)
        if (not force and DRIVER.is_file() and cached.get("fingerprint") == fingerprint
                and (BUILD / "libnc.so").is_file()
                and sha256(BUILD / "libnc.so") == sha256(UPSTREAM / "libnc.so")
                and cached.get("driver_sha256") == sha256(DRIVER)):
            return DRIVER
        with tempfile.TemporaryDirectory(prefix="compile_", dir=BUILD) as temporary:
            directory = Path(temporary)
            objects = []
            common = ["-O2", "-ffp-contract=off", "-DLIBNC_CONFIG_FULL", '-DCONFIG_VERSION="2024-06-05"',
                      "-I", str(ROOT / "cpp/include"), "-I", str(UPSTREAM)]
            commands = []
            for index, source in enumerate(sources):
                output = directory / f"source_{index}.o"
                commands.append([cc, "-std=gnu11", *common, "-c", str(source), "-o", str(output)])
                objects.append(output)
            commands.append([cxx, "-std=c++20", *common, str(ROOT / "cpp/src/nncp_probability_cli.cpp"),
                             *map(str, objects), "-L", str(UPSTREAM), "-Wl,-rpath,$ORIGIN", "-lnc", "-lz", "-lm", "-lpthread",
                             "-o", str(directory / DRIVER.name)])
            for command in commands:
                subprocess.run(command, check=True, timeout=180)
            shutil.copy2(UPSTREAM / "libnc.so", BUILD / "libnc.so")
            os.replace(directory / DRIVER.name, DRIVER)
            stamp.write_text(json.dumps({"fingerprint": fingerprint, "upstream": "NNCP 3.3 (2024-06-05)",
                "source_sha256": hashlib.sha256((UPSTREAM / "nncp.c").read_bytes()).hexdigest(),
                "libnc_sha256": hashlib.sha256((UPSTREAM / "libnc.so").read_bytes()).hexdigest(),
                "source_fingerprints": source_fingerprints(), "driver_sha256": sha256(DRIVER),
                "profile": "transformer-stream1-v1", "platform": platform.platform()}, indent=2) + "\n")
    return DRIVER


def docker_prefix() -> list[str]:
    if not shutil.which("docker"):
        raise RuntimeError("NNCP Docker mode requires Docker with the gcc:14.3.0-bookworm image")
    return ["docker", "run", "--rm", "-i", "--platform", "linux/amd64", "--network", "none",
            "--cpus", "2", "--memory", "2g", "-v", f"{ROOT}:/workspace", "-w", "/workspace", DOCKER_IMAGE]


def nncp_command(*, force: bool = False) -> list[str]:
    override = os.environ.get("WEB_COMPRESSION_NNCP_DRIVER")
    if override:
        driver = Path(override).expanduser().resolve()
        if not driver.is_file():
            raise RuntimeError(f"Configured NNCP driver is missing: {driver}")
        return [str(driver)]
    if os.environ.get("WEB_COMPRESSION_NNCP_USE_DOCKER") == "1":
        command = [*docker_prefix(), "python3", "scripts/build_nncp_probability.py", "--native-only"]
        if force:
            command.append("--force")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, timeout=240)
        return [*docker_prefix(), "timeout", "180", "build/nncp_probability/nncp_probability_cli"]
    return [str(build_native(force=force))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build_native(force=args.force) if args.native_only else json.dumps(nncp_command(force=args.force)))


if __name__ == "__main__":
    main()
