#!/usr/bin/env python3
"""Build the prepared recommend135 binaries against frozen runtime archives.

The build is deliberately sequential.  It writes only to --output, verifies
the same 270 frozen dependencies used by the recommend135 v1 build before and
after compilation, and refuses any base/overlay with NNCP enabled.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


COMPILE_UNITS = (
    "campaign",
    "input_builder",
    "main",
    "model",
    "runtime",
    "campaign_test",
    "input_builder_test",
    "model_test",
    "runtime_test",
    "prepared_replay",
    "prepared_columns",
)

TARGETS = {
    "operator_recommend135_cli": (
        "main",
        "campaign",
        "runtime",
        "model",
        "input_builder",
    ),
    "prepared_replay": (
        "prepared_replay",
        "campaign",
        "runtime",
        "model",
        "input_builder",
    ),
    "prepared_columns": (
        "prepared_columns",
        "campaign",
        "runtime",
        "model",
        "input_builder",
    ),
    "campaign_test": ("campaign_test", "campaign", "runtime", "model"),
    "input_builder_test": ("input_builder_test", "input_builder"),
    "model_test": ("model_test", "model"),
    "runtime_test": ("runtime_test", "runtime"),
}

TEST_TARGETS = (
    "campaign_test",
    "input_builder_test",
    "model_test",
    "runtime_test",
)

EXPECTED_FROZEN_DEPENDENCIES = 270
EXCLUDED_ACTION_IDS = (106, 107, 108, 109, 110)


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def frozen_dependencies(base, overlay, libraries, receipt_path, cache, flag_file,
                        overlay_receipt):
    protected = {cache, flag_file, receipt_path}
    for token in libraries:
        if not token.startswith("-"):
            protected.add(Path(token).resolve(strict=True))
    for binding in overlay_receipt.get("private_codec_plugins", {}).values():
        if isinstance(binding, str) and Path(binding).is_file():
            protected.add(Path(binding))
    protected.update(
        (overlay / "build").glob("liboperator102_blosc*_v4.so")
    )
    for root in (overlay / "source/cpp", base / "source/cpp"):
        for directory in ("operator_harness_v2", "native_graph_v1", "src"):
            for pattern in ("*.hpp", "*.h"):
                protected.update((root / directory).rglob(pattern))
    for directory in (
        base / "source/external/compression/lz4/lib",
        base / "source/external/compression/zstd/lib",
        base / "deps/include/nlohmann",
    ):
        for pattern in ("*.h", "*.hpp"):
            protected.update(directory.rglob(pattern))
    if len(protected) != EXPECTED_FROZEN_DEPENDENCIES:
        raise ValueError(
            "frozen dependency closure changed: "
            f"expected {EXPECTED_FROZEN_DEPENDENCIES}, found {len(protected)}"
        )
    return tuple(sorted(protected))


def build(base, overlay, source, output):
    base, overlay, source = (
        Path(path).resolve(strict=True) for path in (base, overlay, source)
    )
    output = Path(output).resolve()
    if output.is_relative_to(base) or output.is_relative_to(overlay):
        raise ValueError("prepared build output must be outside frozen trees")
    output.mkdir(parents=True, exist_ok=True)

    cache = require_file(base / "build/CMakeCache.txt", "base CMake cache")
    if "NATIVE_GRAPH_ENABLE_NNCP:BOOL=OFF" not in cache.read_text():
        raise ValueError("NNCP must be OFF in the frozen base")

    receipt_path = overlay / "build/BUILD_RECEIPT.json"
    if not receipt_path.is_file():
        receipt_path = overlay / "BUILD_RECEIPT.json"
    receipt_path = require_file(receipt_path, "overlay build receipt")
    overlay_receipt = json.loads(receipt_path.read_text())
    if overlay_receipt.get("NNCP") != "OFF":
        raise ValueError("NNCP must be OFF in the frozen overlay")

    link = next(
        command for command in overlay_receipt["commands"]
        if "-o" in command
        and Path(command[command.index("-o") + 1]).name
        == "operator_harness_v2_cli"
    )
    libraries = link[link.index("-Wl,--whole-archive"):]

    flag_file = require_file(
        base / "build/CMakeFiles/operator_harness_v2.dir/flags.make",
        "frozen compiler flags",
    )
    assignments = dict(
        line.split(" = ", 1)
        for line in flag_file.read_text().splitlines()
        if " = " in line
    )
    flags = sum(
        (shlex.split(assignments[key])
         for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS")),
        [],
    )
    flags = [
        "-I" + str(source / "cpp/operator_recommend135_v1"),
        "-I" + str(overlay / "source/cpp/operator_harness_v2"),
        "-I" + str(overlay / "source/cpp/src"),
        "-I" + str(overlay / "source/cpp/native_graph_v1"),
        *flags,
        "-I" + str(base / "source/external/compression/lz4/lib"),
        "-I" + str(base / "source/external/compression/zstd/lib"),
        "-O3",
        "-ffp-contract=off",
    ]

    protected = frozen_dependencies(
        base, overlay, libraries, receipt_path, cache, flag_file,
        overlay_receipt,
    )
    frozen_before = {str(path): sha256(path) for path in protected}

    source_root = source / "cpp/operator_recommend135_v1"
    compile_sources = {
        name: require_file(source_root / f"{name}.cpp", f"source unit {name}")
        for name in COMPILE_UNITS
    }
    headers = tuple(sorted(source_root.glob("*.hpp")))
    if not headers:
        raise ValueError(f"no recommend135 headers found under {source_root}")
    source_paths = tuple(compile_sources.values()) + headers
    source_before = {str(path): sha256(path) for path in source_paths}

    commands = []

    def run(command):
        commands.append(command)
        print(json.dumps({"command": command}), flush=True)
        subprocess.run(command, check=True)

    objects = {}
    for name in COMPILE_UNITS:
        object_path = output / f"{name}.o"
        unit_flags = [*flags]
        if name.endswith("_test"):
            # The frozen Release flags contain -DNDEBUG.  These test sources
            # use assert(), so override NDEBUG after all inherited flags while
            # leaving production objects on the original Release contract.
            unit_flags.append("-UNDEBUG")
        run([
            "/usr/bin/c++",
            *unit_flags,
            "-c",
            str(compile_sources[name]),
            "-o",
            str(object_path),
        ])
        objects[name] = object_path

    binary_sha256 = {}
    map_sha256 = {}
    for target, members in TARGETS.items():
        binary = output / target
        link_map = output / f"{target}.map"
        run([
            "/usr/bin/c++",
            "-O3",
            *(str(objects[member]) for member in members),
            "-o",
            str(binary),
            "-Wl,-Map," + str(link_map),
            *libraries,
            "-pthread",
        ])
        binary_sha256[target] = sha256(binary)
        map_sha256[link_map.name] = sha256(link_map)

    tests = {}
    for target in TEST_TARGETS:
        test_command = [str(output / target)]
        commands.append(test_command)
        completed = subprocess.run(
            test_command, text=True, capture_output=True
        )
        tests[target] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode:
            raise RuntimeError(json.dumps({target: tests[target]}))

    frozen_after = {str(path): sha256(path) for path in protected}
    if frozen_after != frozen_before:
        raise RuntimeError("frozen dependency changed during prepared build")
    source_after = {str(path): sha256(path) for path in source_paths}
    if source_after != source_before:
        raise RuntimeError("prepared source changed during build")

    receipt = {
        "schema": "operator-prepared135-build-v1",
        "status": "built-and-tested",
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "NNCP": "OFF",
        "excluded_action_ids": list(EXCLUDED_ACTION_IDS),
        "frozen_dependency_count": len(protected),
        "frozen_runtime_before": frozen_before,
        "frozen_runtime_after": frozen_after,
        "source_sha256": source_before,
        "source_after_sha256": source_after,
        "binaries": binary_sha256,
        "maps_sha256": map_sha256,
        "commands": commands,
        "tests": tests,
        "test_assertions": {
            "status": "enabled",
            "compile_flag": "-UNDEBUG",
            "inherited_ndebug_overridden": True,
            "targets": list(TEST_TARGETS),
            "production_compile_flags_unchanged": True,
        },
        "resource_contract": {
            "experiment_cpu": 38,
            "scheduler": "SCHED_IDLE",
            "nice": 19,
            "io_class": "idle",
            "address_space_limit_bytes": 2 * 1024 * 1024 * 1024,
            "enforced_by_build_script": False,
        },
        "evaluation_contract": {
            "columns": 3519,
            "split": "5-fold source-group dev OOF; dev-only fit to confirm",
            "confirm_labels_used_for_tuning": False,
            "parameter_retuning": False,
        },
    }
    atomic_json(output / "BUILD_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"],
                      "binaries": receipt["binaries"]}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ("base", "overlay", "source", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    options = parser.parse_args()
    build(options.base, options.overlay, options.source, options.output)
