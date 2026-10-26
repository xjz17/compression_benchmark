#!/usr/bin/env python3
"""Build the C++ empty-input revision against immutable runtime dependencies.

This script orchestrates the compiler only. It does not execute operator kernels,
generate numeric inputs or perform measurements. Prior runtime files are read-only.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build(runtime, source, output):
    runtime = runtime.resolve(strict=True)
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == runtime / "build" or output.is_relative_to(runtime):
        raise ValueError("build output must be outside the prior runtime")
    output.mkdir(parents=True, exist_ok=True)
    prior = runtime / "build"
    cache = (prior / "CMakeCache.txt").read_text()
    if "NATIVE_GRAPH_ENABLE_NNCP:BOOL=OFF" not in cache or "OPERATOR_HARNESS_V2_ENABLE_SUPPORT_DEPS:BOOL=ON" not in cache:
        raise ValueError("prebuilt runtime must have NNCP OFF and support deps ON")
    flag_file = prior / "CMakeFiles/operator_harness_v2.dir/flags.make"
    assignments = dict(line.split(" = ", 1) for line in flag_file.read_text().splitlines() if " = " in line)
    flags = shlex.split(assignments["CXX_DEFINES"]) + shlex.split(assignments["CXX_INCLUDES"]) + shlex.split(assignments["CXX_FLAGS"])
    harness = source / "cpp/operator_harness_v2"
    flags = ["-I" + str(harness), "-I" + str(source / "cpp/src"),
             "-I" + str(source / "cpp/native_graph_v1")] + flags
    link_file = prior / "CMakeFiles/operator_harness_v2_cli.dir/link.txt"
    old_link = shlex.split(link_file.read_text())
    start = old_link.index("liboperator_harness_v2.a") + 1
    libraries = []
    protected = {flag_file, link_file, prior / "CMakeCache.txt"}
    for token in old_link[start:]:
        if token.startswith("-"):
            if not token.startswith(("-l", "-Wl,")):
                raise ValueError("unexpected prebuilt link flag: " + token)
            libraries.append(token)
        else:
            path = Path(token)
            if not path.is_absolute():
                path = prior / path
            path = path.resolve(strict=True)
            protected.add(path)
            libraries.append(str(path))
    for path in (prior / "native_graph_v1").glob("liboperator102_blosc*.so"):
        protected.add(path.resolve(strict=True))
    for path in (prior / "native_graph_v1").glob("libnative_graph_v1_*isolated.so"):
        protected.add(path.resolve(strict=True))
    codec_dependencies = {}
    for variant in (1, 2):
        directory = prior / "native_graph_v1" / f"operator102_operator102_blosc{variant}_build/blosc"
        static = (directory / ("libblosc.a" if variant == 1 else "libblosc2.a")).resolve(strict=True)
        include = runtime / "source/external/compression" / (
            "c-blosc/blosc" if variant == 1 else "c-blosc2/include")
        codec_dependencies[variant] = (static, include.resolve(strict=True))
        protected.add(static)
    required = ["cpp/native_graph_v1/operator102_dispatcher.cpp",
                "cpp/native_graph_v1/operator102_blosc_host.cpp",
                "cpp/native_graph_v1/operator102_blosc_plugin.cpp",
                "cpp/src/chimp128_exact_stages.cpp", "cpp/src/bzip2_operator_ablation.cpp",
                "cpp/src/fpc_exact_stages.cpp", "cpp/src/fpc.c",
                "cpp/src/blosc_codec.cpp", "cpp/src/blosc2_codec.cpp",
                "cpp/tests/chimp128_exact_stages_test.cpp", "cpp/tests/bzip2_empty_exact_test.cpp",
                "cpp/tests/fpc_exact_stages_test.cpp", "cpp/tests/blosc_empty_exact_test.cpp"]
    for relative in required:
        (source / relative).resolve(strict=True)
    fingerprints = {str(path): digest(path) for path in sorted(protected)}
    lib_sources = [path for path in sorted(harness.glob("*.cpp"))
                   if path.name not in {"main.cpp", "batch_runner.cpp", "stress_input_builder.cpp"}
                   and not path.name.endswith("_test.cpp")]
    dispatcher_override = source / "cpp/native_graph_v1/operator102_dispatcher.cpp"
    lib_sources.append(dispatcher_override)
    # Whole translation-unit overrides are narrowly enumerated. The prior
    # archives remain unchanged; loading the new objects first prevents an old
    # archive member from silently satisfying the same codec symbols.
    for relative in ("cpp/src/chimp128_exact_stages.cpp",
                     "cpp/src/bzip2_operator_ablation.cpp",
                     "cpp/src/fpc_exact_stages.cpp",
                     "cpp/native_graph_v1/operator102_blosc_host.cpp"):
        replacement = source / relative
        if replacement.is_file():
            lib_sources.append(replacement)
    all_sources = lib_sources + [harness / "main.cpp", harness / "batch_runner.cpp"]
    all_sources += sorted(harness.glob("*_test.cpp"))
    for relative in ("cpp/tests/chimp128_exact_stages_test.cpp",
                     "cpp/tests/bzip2_empty_exact_test.cpp",
                     "cpp/tests/fpc_exact_stages_test.cpp"):
        test = source / relative
        if test.is_file():
            all_sources.append(test)
    if (harness / "stress_input_builder.cpp").exists():
        all_sources.append(harness / "stress_input_builder.cpp")
    # Include the project headers used by the overlay and prior runtime. The
    # libraries and the previous source tree remain read-only throughout.
    headers = set(harness.rglob("*.hpp"))
    for root in (source, runtime / "source"):
        for relative in ("cpp/src", "cpp/native_graph_v1"):
            headers.update((root / relative).rglob("*.hpp"))
            headers.update((root / relative).rglob("*.h"))
    for _, include in codec_dependencies.values():
        headers.update(include.rglob("*.h"))
    bzip_headers = runtime / "source/external/compression/lzbench/bwt/bzip2"
    headers.update(bzip_headers.glob("*.h"))
    codec_sources = [source / "cpp/src/blosc_codec.cpp", source / "cpp/src/blosc2_codec.cpp",
                     source / "cpp/native_graph_v1/operator102_blosc_plugin.cpp",
                     source / "cpp/tests/blosc_empty_exact_test.cpp", source / "cpp/src/fpc.c"]
    source_hashes = {str(path): digest(path) for path in [*all_sources, *codec_sources, *sorted(headers)]}
    commands = []
    def execute(command):
        commands.append(command)
        print(json.dumps({"command": command}), flush=True)
        subprocess.run(command, check=True)
    exports = output / "blosc_private.exports"
    exports.write_text("{ global: operator102_blosc_call; local: *; };\n")
    plugins = {}
    plugin_objects = {}
    for variant, (static, include) in codec_dependencies.items():
        codec = source / ("cpp/src/blosc_codec.cpp" if variant == 1 else "cpp/src/blosc2_codec.cpp")
        object_paths = []
        for path in (source / "cpp/native_graph_v1/operator102_blosc_plugin.cpp", codec):
            obj = output / (path.stem + f"_private_{variant}.o")
            execute(["/usr/bin/c++", *flags, "-I" + str(include), "-fPIC",
                     "-fvisibility=hidden", "-fvisibility-inlines-hidden",
                     "-DOPERATOR102_BLOSC_VARIANT=" + str(variant),
                     "-c", str(path), "-o", str(obj)])
            object_paths.append(obj)
        plugin_objects[variant] = object_paths[1]
        plugin = output / f"liboperator102_blosc{variant}_v4.so"
        execute(["/usr/bin/c++", "-shared", *map(str, object_paths), str(static),
                 "-pthread", "-ldl", "-Wl,--version-script=" + str(exports),
                 "-Wl,-Bsymbolic", "-Wl,-Map," + str(output / (plugin.name + ".map")),
                 "-o", str(plugin)])
        plugins[variant] = plugin
    plugin_macros = [
        "-DOPERATOR102_BLOSC1_PLUGIN=" + json.dumps(str(plugins[1])),
        "-DOPERATOR102_BLOSC2_PLUGIN=" + json.dumps(str(plugins[2])),
        "-DOPERATOR102_BLOSC333_PLUGIN=" + json.dumps(str(
            (prior / "native_graph_v1/liboperator102_blosc333.so").resolve(strict=True))),
    ]
    objects = []
    for path in lib_sources:
        obj = output / (path.stem + ".o")
        extra = ["-I" + str(bzip_headers)] if path.name == "bzip2_operator_ablation.cpp" else []
        if path.name == "operator102_blosc_host.cpp":
            extra += plugin_macros
        execute(["/usr/bin/c++", *flags, *extra, "-c", str(path), "-o", str(obj)])
        objects.append(obj)
    native_fpc = output / "native_fpc_v4.o"
    execute(["/usr/bin/cc", "-O3", "-DNDEBUG", "-std=gnu11", "-ffp-contract=off",
             "-I" + str(source / "cpp/src"), "-I" + str(runtime / "source/cpp/src"),
             "-c", str(source / "cpp/src/fpc.c"), "-o", str(native_fpc)])
    objects.append(native_fpc)
    archive = output / "liboperator_tree_v3.a"
    temporary_archive = output / "liboperator_tree_v3.new.a"
    if temporary_archive.exists():
        temporary_archive.unlink()
    execute(["ar", "rcs", str(temporary_archive), *map(str, objects)])
    temporary_archive.replace(archive)
    v2_catalog = source / "benchmark/reference/actions/operator_harness_batch_catalog_v2.json"
    v3_catalog = source / "benchmark/reference/actions/operator_harness_tree_catalog_v3.json"
    test_macros = ["-DOPERATOR_HARNESS_V2_TEST_CATALOG=" + json.dumps(str(v2_catalog)),
                   "-DOPERATOR_HARNESS_V3_TEST_CATALOG=" + json.dumps(str(v3_catalog)),
                   "-DOPERATOR_HARNESS_TREE_V3_TEST_CATALOG=" + json.dumps(str(v3_catalog))]
    binaries = {}
    for path in all_sources[len(lib_sources):]:
        obj = output / (path.stem + ".o")
        execute(["/usr/bin/c++", *flags, *(["-UNDEBUG", *test_macros] if path.name.endswith("_test.cpp") else []),
                 "-c", str(path), "-o", str(obj)])
        name = {"main.cpp": "operator_harness_v2_cli", "batch_runner.cpp": "operator_harness_v2_batch_runner",
                "stress_input_builder.cpp": "operator_harness_v2_stress_input_builder",
                "protocol_test.cpp": "operator_harness_v2_protocol_test"}.get(path.name, path.stem)
        binary = output / name
        if path.name in {"stress_input_builder.cpp", "batch_runner.cpp"}:
            # Preparation/coordinator processes do not need any codec code.
            crypto = [item for item in libraries
                      if Path(item).name.startswith("libcrypto.")]
            if not crypto:
                raise ValueError("prebuilt link metadata lacks libcrypto")
            closure = list(dict.fromkeys(crypto))
        else:
            closure = ["-Wl,--whole-archive", str(archive), "-Wl,--no-whole-archive", *libraries]
        execute(["/usr/bin/c++", "-O3", "-DNDEBUG", str(obj), "-o", str(binary),
                 "-Wl,-Map," + str(output / (name + ".map")), *closure])
        binaries[name] = digest(binary)
    for variant, (static, include) in codec_dependencies.items():
        test = source / "cpp/tests/blosc_empty_exact_test.cpp"
        obj = output / f"blosc_empty_exact_test_{variant}.o"
        execute(["/usr/bin/c++", *flags, "-UNDEBUG", "-I" + str(include),
                 "-DBLOSC_TEST_VARIANT=" + str(variant), "-c", str(test), "-o", str(obj)])
        binary = output / f"blosc_empty_exact_test_{variant}"
        execute(["/usr/bin/c++", "-O3", str(obj), str(plugin_objects[variant]), str(static),
                 "-pthread", "-ldl", "-Wl,-Map," + str(output / (binary.name + ".map")),
                 "-o", str(binary)])
        binaries[binary.name] = digest(binary)
    for path, before in fingerprints.items():
        if digest(Path(path)) != before:
            raise ValueError("prior runtime changed during build: " + path)
    for path, before in source_hashes.items():
        if digest(Path(path)) != before:
            raise ValueError("source changed during build: " + path)
    result = {"schema": "operator-tree-prebuilt-build-receipt-v4", "status": "verified",
              "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "runtime_root": str(runtime), "source_root": str(source), "output": str(output),
              "prior_runtime_unchanged_sha256": fingerprints, "source_sha256": source_hashes,
              "binaries": binaries, "commands": commands,
              "private_codec_plugins": {str(path): digest(path) for path in plugins.values()},
              "codec_host_bindings": {str(k): str(v) for k, v in plugins.items()},
              "blosc333_binding": str(prior / "native_graph_v1/liboperator102_blosc333.so"),
              "link_map_sha256": {path.name: digest(path) for path in sorted(output.glob("*.map"))},
              "tests_executed_by_builder": False,
              "NNCP": "OFF", "support_deps": "ON"}
    atomic_json(output / "BUILD_RECEIPT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.runtime_root, args.source_root, args.build_dir)
    print(json.dumps({"status": receipt["status"], "binaries": receipt["binaries"]}), flush=True)
