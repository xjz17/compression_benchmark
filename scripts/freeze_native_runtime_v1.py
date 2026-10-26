#!/usr/bin/env python3
"""Freeze the admitted native source, build, Python, and ELF runtime closure.

This tool hashes provenance only. It performs no training, compression search,
or timing measurement, and it does not create an admission/pass assertion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


SCHEMA = "native-runtime-freeze-v1"
NATIVE_COMPONENTS = (
    "cpp/native_features_v1",
    "cpp/native_policy_v1",
    "cpp/native_graph_v1",
    "cpp/native_pipeline_v1",
    "cpp/native_baseline86_v1",
)
ENTRYPOINTS = {
    "pipeline_server": ("pipeline", "native_pipeline_v1_server"),
    "cold_decode_cli": ("pipeline", "native_graph/native_graph_request_cli"),
    "policy_cli": ("pipeline", "native_policy/weighted_grpo_native_policy_cli"),
    "features_cli": ("pipeline", "native_features/native_features_cli"),
    "baseline86_cli": ("baseline", "native_baseline86_cli"),
}
BUILD_CONFIGURATION_NAMES = {
    "CMakeCache.txt", "flags.make", "link.txt", "DependInfo.cmake",
    "Makefile.cmake", "CMakeSystem.cmake", "CMakeCCompiler.cmake",
    "CMakeCXXCompiler.cmake",
}
BUILD_DIRECTORY_NAMES = {"build", "__pycache__", ".git"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def _record(path: Path, *, code_root: Path | None = None) -> dict:
    absolute = path.expanduser().resolve(strict=True)
    require(absolute.is_file(), f"runtime freeze input is not a file: {absolute}")
    result = {
        "path": str(absolute),
        "sha256": digest_bytes(absolute.read_bytes()),
        "bytes": absolute.stat().st_size,
    }
    if code_root is not None:
        try:
            result["code_root_relative_path"] = absolute.relative_to(code_root).as_posix()
        except ValueError:
            pass
    return result


def _runtime_record(path: Path, *, code_root: Path) -> dict:
    # Keep the loader-reported absolute path as the admission key. Resolving a
    # symlink here would let a later retarget escape verification of the path
    # that the ELF loader actually opens.
    runtime_path = Path(os.path.abspath(path))
    resolved = runtime_path.resolve(strict=True)
    require(runtime_path.is_file(), f"runtime library is not a file: {runtime_path}")
    result = {
        "path": str(runtime_path),
        "resolved_path": str(resolved),
        "sha256": digest_bytes(runtime_path.read_bytes()),
        "bytes": runtime_path.stat().st_size,
    }
    if _inside(resolved, code_root):
        result["code_root_relative_path"] = resolved.relative_to(code_root).as_posix()
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root)
        return True
    except ValueError:
        return False


def _parse_depfile(path: Path, *, build_root: Path, code_root: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    require(raw.strip(), f"empty compiler dependency file: {path}")
    # GCC/Clang Make dependency output uses a trailing backslash for every
    # physical continuation. A remaining nonempty newline means another rule
    # or malformed/truncated output, which this v1 parser refuses to guess at.
    logical = re.sub(r"\\\r?\n", " ", raw)
    require(len([line for line in logical.splitlines() if line.strip()]) == 1,
            f"unsupported multiple/malformed dependency rules: {path}")
    require(":" in logical, f"compiler dependency target is missing: {path}")
    target, rhs = logical.split(":", 1)
    require(target.strip().endswith(".o"), f"unexpected compiler target in {path}")
    try:
        tokens = shlex.split(rhs, posix=True)
    except ValueError as error:
        raise RuntimeError(f"cannot parse compiler dependencies: {path}") from error
    require(tokens, f"compiler dependency list is empty: {path}")
    dependencies: list[Path] = []
    for token in tokens:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = build_root / candidate
        try:
            candidate = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"compiler dependency is missing (stale/incomplete build): {token} in {path}"
            ) from error
        require(candidate.is_file(), f"compiler dependency is not a file: {candidate}")
        dependencies.append(candidate)
    dependencies = sorted(set(dependencies), key=str)
    code_dependencies = [entry for entry in dependencies if _inside(entry, code_root)]
    require(code_dependencies,
            f"dependency rule has no source/header inside code root: {path}")
    return {
        "depfile": _record(path, code_root=code_root),
        "target": target.strip(),
        "code_dependencies": code_dependencies,
        "system_dependencies": [entry for entry in dependencies
                                if not _inside(entry, code_root)],
    }


def _parse_cmake_asm_dependency(object_path: Path, *, build_root: Path,
                                code_root: Path) -> dict:
    """Read CMake's generated dependency rule for an ASM object without .o.d.

    GNU Make emits ``depend.make`` rather than a compiler ``.o.d`` for the
    preprocessed zstd ``.S`` source.  Accept only that exact object kind and an
    exact target rule under its CMake ``*.dir`` directory; all dependencies
    still have to exist and the rule must include code-root input.
    """
    object_path = object_path.resolve(strict=True)
    require(object_path.name.endswith(".S.o"),
            f"non-ASM object is missing its compiler dependency file: {object_path}")
    target_directories = [parent for parent in object_path.parents
                          if parent.name.endswith(".dir")]
    require(len(target_directories) == 1,
            f"cannot identify one CMake target directory for ASM object: {object_path}")
    dependency_file = target_directories[0] / "depend.make"
    require(dependency_file.is_file(),
            f"ASM object lacks CMake depend.make: {object_path}")
    raw = dependency_file.read_text(encoding="utf-8")
    logical = re.sub(r"\\\r?\n", " ", raw)
    matches: list[tuple[str, str]] = []
    for line in logical.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        require(":" in line,
                f"malformed CMake ASM dependency rule in {dependency_file}")
        target, rhs = line.split(":", 1)
        try:
            targets = shlex.split(target, posix=True)
        except ValueError as error:
            raise RuntimeError(
                f"cannot parse CMake ASM dependency target: {dependency_file}"
            ) from error
        if any((build_root / token).resolve() == object_path for token in targets):
            matches.append((target.strip(), rhs))
    require(len(matches) == 1,
            f"expected one exact CMake ASM dependency rule for {object_path}, "
            f"found {len(matches)}")
    target, rhs = matches[0]
    try:
        tokens = shlex.split(rhs, posix=True)
    except ValueError as error:
        raise RuntimeError(
            f"cannot parse CMake ASM dependencies: {dependency_file}"
        ) from error
    require(tokens, f"CMake ASM dependency list is empty: {dependency_file}")
    dependencies: list[Path] = []
    for token in tokens:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = build_root / candidate
        try:
            candidate = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"CMake ASM dependency is missing: {token} in {dependency_file}"
            ) from error
        require(candidate.is_file(),
                f"CMake ASM dependency is not a file: {candidate}")
        dependencies.append(candidate)
    dependencies = sorted(set(dependencies), key=str)
    code_dependencies = [entry for entry in dependencies
                         if _inside(entry, code_root)]
    require(code_dependencies,
            f"CMake ASM rule has no source/header inside code root: {dependency_file}")
    return {
        "depfile": _record(dependency_file, code_root=code_root),
        "object": _record(object_path, code_root=code_root),
        "target": target,
        "dependency_kind": "cmake_asm_depend_make",
        "code_dependencies": code_dependencies,
        "system_dependencies": [entry for entry in dependencies
                                if not _inside(entry, code_root)],
    }


def _walk_files(root: Path, *, predicate) -> list[Path]:
    results: list[Path] = []
    for current, directories, files in os.walk(root):
        base = Path(current)
        retained = []
        for name in directories:
            candidate = base / name
            generated_build = (
                name in BUILD_DIRECTORY_NAMES
                or name.startswith(("build-", "build_", "cmake-build"))
                or (candidate / "CMakeCache.txt").is_file()
                or (candidate / "CMakeFiles").is_dir()
            )
            if not generated_build:
                retained.append(name)
        directories[:] = sorted(retained)
        for name in sorted(files):
            path = base / name
            if predicate(path):
                results.append(path.resolve(strict=True))
    return sorted(set(results), key=str)


def _source_configuration_files(code_root: Path) -> list[Path]:
    return _walk_files(
        code_root,
        predicate=lambda path: path.name == "CMakeLists.txt" or path.suffix == ".cmake",
    )


def _native_component_files(code_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in NATIVE_COMPONENTS:
        component = (code_root / relative).resolve(strict=True)
        require(component.is_dir(), f"missing native component: {component}")
        files.extend(_walk_files(component, predicate=lambda path: path.is_file()))
    return sorted(set(files), key=str)


def _python_files(code_root: Path) -> list[Path]:
    scripts = (code_root / "scripts").resolve(strict=True)
    require(scripts.is_dir(), f"missing scripts directory: {scripts}")
    return _walk_files(scripts, predicate=lambda path: path.suffix == ".py")


def _build_configuration_files(build_roots: Iterable[Path]) -> list[Path]:
    results: list[Path] = []
    for root in build_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.name in BUILD_CONFIGURATION_NAMES:
                results.append(path.resolve(strict=True))
    return sorted(set(results), key=str)


def _entrypoints(pipeline_build: Path, baseline_build: Path) -> dict[str, Path]:
    roots = {"pipeline": pipeline_build, "baseline": baseline_build}
    result = {}
    for label, (kind, relative) in ENTRYPOINTS.items():
        path = (roots[kind] / relative).resolve(strict=True)
        require(path.is_file() and os.access(path, os.X_OK),
                f"missing/non-executable native entry point {label}: {path}")
        result[label] = path
    return result


def _one_plugin(build: Path, basename: str) -> Path:
    exact = sorted(
        {path.resolve(strict=True) for path in build.rglob(basename) if path.is_file()},
        key=str,
    )
    require(len(exact) == 1,
            f"expected exactly one {basename} below {build}, found {len(exact)}")
    return exact[0]


def _runtime_plugins(pipeline_build: Path, baseline_build: Path) -> dict[str, Path]:
    result = {
        "pipeline_cmix": _one_plugin(
            pipeline_build, "libnative_graph_v1_cmix_isolated.so"
        ),
        "pipeline_paq": _one_plugin(
            pipeline_build, "libnative_graph_v1_paq_isolated.so"
        ),
    }
    # The baseline links the same stages target and therefore embeds both of
    # its own absolute dlopen paths even when no formal baseline label selects
    # CMIX. Both reachable DSOs belong to the frozen runtime closure.
    result["baseline_cmix"] = _one_plugin(
        baseline_build, "libnative_graph_v1_cmix_isolated.so"
    )
    result["baseline_paq"] = _one_plugin(
        baseline_build, "libnative_graph_v1_paq_isolated.so"
    )
    return result


def _run_ldd(path: Path) -> tuple[list[Path], list[str], str]:
    result = subprocess.run(
        ["ldd", str(path)], text=True, capture_output=True, check=False,
        timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr)
                           if part.strip())
    lower = combined.lower()
    if "not a dynamic executable" in lower or "statically linked" in lower:
        return [], [], combined
    require(result.returncode == 0, f"ldd failed for {path}: {combined[:2000]}")
    libraries: list[Path] = []
    virtual: list[str] = []
    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=> not found" in line:
            raise RuntimeError(f"unresolved dynamic library for {path}: {line}")
        if "=>" in line:
            _name, destination = line.split("=>", 1)
            token = destination.strip().split(" (", 1)[0].strip()
        else:
            token = line.split(" (", 1)[0].strip()
        if token.startswith("/"):
            dependency = Path(token)
            require(dependency.is_absolute() and dependency.resolve(strict=True).is_file(),
                    f"ldd dependency is not a file: {dependency}")
            libraries.append(dependency)
        elif token.startswith("linux-vdso") or token.startswith("linux-gate"):
            virtual.append(token)
        else:
            raise RuntimeError(f"unparsed ldd output for {path}: {line}")
    return sorted(set(libraries), key=str), sorted(set(virtual)), combined


def _dynamic_closure(roots: dict[str, Path]) -> tuple[dict, list[Path]]:
    pending = list(roots.values())
    seen: set[Path] = set()
    libraries: set[Path] = set()
    linkage = {}
    while pending:
        path = Path(os.path.abspath(pending.pop(0)))
        require(path.resolve(strict=True).is_file(), f"runtime root disappeared: {path}")
        if path in seen:
            continue
        seen.add(path)
        dependencies, virtual, raw = _run_ldd(path)
        linkage[str(path)] = {
            "dependencies": [str(item) for item in dependencies],
            "virtual_dependencies": virtual,
            "ldd_output": raw,
        }
        for dependency in dependencies:
            libraries.add(dependency)
            if dependency not in seen:
                pending.append(dependency)
    ordered = sorted(libraries, key=str)
    require(any(path.name.startswith("libnc.so") for path in ordered),
            "NNCP LibNC is absent from the recursive runtime dynamic closure")
    return linkage, ordered


def _embedded_plugin_paths(entrypoints: dict[str, Path],
                           plugins: dict[str, Path]) -> tuple[dict, dict]:
    result = {}
    for label, binary in entrypoints.items():
        payload = binary.read_bytes()
        hits = {
            plugin_label: str(path)
            for plugin_label, path in plugins.items()
            if str(path).encode() in payload
        }
        result[label] = hits
    for binary_label in ("pipeline_server", "cold_decode_cli"):
        require(
            {"pipeline_cmix", "pipeline_paq"}.issubset(result[binary_label]),
            f"{binary_label} does not embed both absolute CMIX/PAQ DSO paths",
        )
    baseline_labels = {"baseline_cmix", "baseline_paq"}
    baseline_hits = baseline_labels.intersection(result["baseline86_cli"])
    # The baseline build adds native_graph_v1 and therefore builds both
    # isolated plugins.  Its CLI currently supplies a baseline-owned
    # Dispatcher, so a normal static link need not extract stage_dispatch.o
    # and may embed neither dlopen path.  Freeze both DSOs as conservative
    # extra roots, but fail on a partial embedding because that indicates an
    # inconsistent build rather than either supported link shape.
    require(
        len(baseline_hits) in (0, len(baseline_labels)),
        "baseline86_cli embeds only one of the baseline CMIX/PAQ DSO paths",
    )
    reachability = {}
    for label in ("pipeline_cmix", "pipeline_paq"):
        reachability[label] = "embedded_absolute_path"
    for label in sorted(baseline_labels):
        reachability[label] = (
            "embedded_absolute_path" if label in baseline_hits
            else "built_present_not_embedded_conservative_extra_root"
        )
    return result, reachability


def _add_frozen(frozen: dict[str, str], records: Iterable[dict]) -> None:
    for record in records:
        path, sha = record["path"], record["sha256"]
        if path in frozen:
            require(frozen[path] == sha, f"conflicting digest for frozen file: {path}")
        frozen[path] = sha


def collect(arguments) -> dict:
    require(sys.platform.startswith("linux"),
            "native runtime freezing requires Linux ldd on the admitted host")
    require(shutil.which("ldd") is not None, "Linux ldd is unavailable")
    code_root = arguments.code_root.expanduser().resolve(strict=True)
    pipeline_build = arguments.pipeline_build.expanduser().resolve(strict=True)
    baseline_build = arguments.baseline_build.expanduser().resolve(strict=True)
    require(code_root.is_dir(), "--code-root is not a directory")
    for build in (pipeline_build, baseline_build):
        require(build.is_dir() and _inside(build, code_root),
                f"build must be a directory inside code root: {build}")

    immutable_inputs = {
        label: _record(getattr(arguments, label), code_root=code_root)
        for label in ("library", "catalog", "reference")
    }
    depfiles = sorted(
        set(pipeline_build.rglob("*.o.d")) | set(baseline_build.rglob("*.o.d")),
        key=str,
    )
    require(depfiles, "no CMake compiler .o.d files found in either build")
    objects = sorted({
        path.resolve(strict=True)
        for build in (pipeline_build, baseline_build)
        for path in build.rglob("*.o")
        if any(parent.name.endswith(".dir") for parent in path.parents)
    }, key=str)
    require(objects, "no compiled CMake target objects found in either build")
    expected_depfiles = {Path(str(path) + ".d").resolve() for path in objects}
    actual_depfiles = {path.resolve(strict=True) for path in depfiles}
    missing_objects = sorted(
        (path for path in objects
         if Path(str(path) + ".d").resolve() not in actual_depfiles),
        key=str,
    )
    orphan_depfiles = sorted(
        path for path in actual_depfiles if not Path(str(path)[:-2]).is_file()
    )
    require(not orphan_depfiles,
            "dependency record has no compiled object: " +
            ", ".join(map(str, orphan_depfiles[:8])))
    parsed_dependencies = []
    compiled_code: set[Path] = set()
    system_headers: set[Path] = set()
    for depfile in depfiles:
        build = pipeline_build if _inside(depfile, pipeline_build) else baseline_build
        parsed = _parse_depfile(depfile, build_root=build, code_root=code_root)
        compiled_code.update(parsed.pop("code_dependencies"))
        system_headers.update(parsed.pop("system_dependencies"))
        parsed_dependencies.append(parsed)
    asm_dependencies = []
    for object_path in missing_objects:
        build = pipeline_build if _inside(object_path, pipeline_build) else baseline_build
        parsed = _parse_cmake_asm_dependency(
            object_path, build_root=build, code_root=code_root,
        )
        compiled_code.update(parsed.pop("code_dependencies"))
        system_headers.update(parsed.pop("system_dependencies"))
        asm_dependencies.append(parsed)
    require(len(parsed_dependencies) + len(asm_dependencies) == len(objects),
            "not every compiled object has one verified dependency rule")
    require(compiled_code, "compiler dependency closure inside code root is empty")

    source_configs = _source_configuration_files(code_root)
    component_sources = _native_component_files(code_root)
    python_sources = _python_files(code_root)
    build_configs = _build_configuration_files((pipeline_build, baseline_build))
    require(source_configs and component_sources and python_sources and build_configs,
            "source/configuration closure is incomplete")

    entrypoints = _entrypoints(pipeline_build, baseline_build)
    plugins = _runtime_plugins(pipeline_build, baseline_build)
    embedded, plugin_reachability = _embedded_plugin_paths(entrypoints, plugins)
    runtime_roots = {**entrypoints, **plugins}
    linkage, dynamic_libraries = _dynamic_closure(runtime_roots)

    categories = {
        "compiled_code_dependencies": [_record(path, code_root=code_root)
                                       for path in sorted(compiled_code, key=str)],
        "source_cmake_configuration": [_record(path, code_root=code_root)
                                       for path in source_configs],
        "build_cmake_configuration": [_record(path, code_root=code_root)
                                      for path in build_configs],
        "native_component_files": [_record(path, code_root=code_root)
                                   for path in component_sources],
        "python_training_and_evaluation_files": [_record(path, code_root=code_root)
                                                 for path in python_sources],
        "entrypoint_binaries": {
            label: _record(path, code_root=code_root)
            for label, path in sorted(entrypoints.items())
        },
        "runtime_loaded_plugins": {
            label: _record(path, code_root=code_root)
            for label, path in sorted(plugins.items())
        },
        "dynamic_libraries": [_runtime_record(path, code_root=code_root)
                              for path in dynamic_libraries],
    }
    frozen_files: dict[str, str] = {}
    _add_frozen(frozen_files, immutable_inputs.values())
    _add_frozen(frozen_files, (row["depfile"] for row in parsed_dependencies))
    _add_frozen(frozen_files, (row["depfile"] for row in asm_dependencies))
    _add_frozen(frozen_files, (row["object"] for row in asm_dependencies))
    for name in (
        "compiled_code_dependencies", "source_cmake_configuration",
        "build_cmake_configuration", "native_component_files",
        "python_training_and_evaluation_files", "dynamic_libraries",
    ):
        _add_frozen(frozen_files, categories[name])
    _add_frozen(frozen_files, categories["entrypoint_binaries"].values())
    _add_frozen(frozen_files, categories["runtime_loaded_plugins"].values())

    document = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source/binary/runtime identity only; zero training and zero measurement",
        "creates_training_admission": False,
        "code_root": str(code_root),
        "pipeline_build": str(pipeline_build),
        "baseline_build": str(baseline_build),
        "immutable_inputs": immutable_inputs,
        "compiler_dependency_files": parsed_dependencies,
        "compiler_dependency_file_count": len(parsed_dependencies),
        "cmake_asm_dependency_rules": asm_dependencies,
        "cmake_asm_dependency_rule_count": len(asm_dependencies),
        "compiled_object_count": len(objects),
        "system_header_dependencies": [str(path) for path in sorted(system_headers, key=str)],
        "system_header_policy": "paths recorded for audit; individual system headers not frozen",
        **categories,
        "embedded_absolute_runtime_plugin_paths": embedded,
        "runtime_plugin_reachability": plugin_reachability,
        "recursive_ldd": linkage,
        "runtime_requirements": {
            "nncp_libnc_present": True,
            "pipeline_cmix_dso_present": True,
            "pipeline_paq_dso_present": True,
            "baseline_cmix_dso_present": True,
            "baseline_paq_dso_present": True,
        },
        "frozen_files": dict(sorted(frozen_files.items())),
    }
    unsigned = canonical(document)
    document["manifest_sha256"] = digest_bytes(unsigned)
    return document


def write_new(path: Path, document: dict) -> None:
    output = path.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite runtime freeze output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(
                document, sort_keys=True, indent=2, allow_nan=False,
            ).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise RuntimeError(
                f"refusing to overwrite runtime freeze output: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "code-root", "pipeline-build", "baseline-build", "library",
        "catalog", "reference", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite runtime freeze output: {output}")
    document = collect(arguments)
    write_new(output, document)
    print(json.dumps({
        "schema": document["schema"],
        "output": str(output),
        "manifest_sha256": document["manifest_sha256"],
        "frozen_file_count": len(document["frozen_files"]),
        "compiler_dependency_file_count": document["compiler_dependency_file_count"],
        "creates_training_admission": document["creates_training_admission"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
