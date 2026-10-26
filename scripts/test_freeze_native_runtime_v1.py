#!/usr/bin/env python3
"""Fixture tests for fail-closed native runtime provenance freezing."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest import mock

import freeze_native_runtime_v1 as freeze


class RuntimeFreezeTests(unittest.TestCase):
    def test_plugin_embedding_contract_full_none_and_partial_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugins = {
                label: root / (label + ".so")
                for label in (
                    "pipeline_cmix", "pipeline_paq",
                    "baseline_cmix", "baseline_paq",
                )
            }
            for path in plugins.values():
                path.write_bytes(b"plugin")
            entrypoints = {
                label: root / label
                for label in (
                    "pipeline_server", "cold_decode_cli", "policy_cli",
                    "features_cli", "baseline86_cli",
                )
            }
            pipeline_payload = b"\0".join(
                str(plugins[label]).encode()
                for label in ("pipeline_cmix", "pipeline_paq")
            )
            for label, path in entrypoints.items():
                path.write_bytes(
                    pipeline_payload
                    if label in {"pipeline_server", "cold_decode_cli"}
                    else b"binary"
                )

            embedded, reachability = freeze._embedded_plugin_paths(
                entrypoints, plugins,
            )
            self.assertEqual(embedded["baseline86_cli"], {})
            self.assertEqual(
                reachability["baseline_cmix"],
                "built_present_not_embedded_conservative_extra_root",
            )
            self.assertEqual(
                reachability["baseline_paq"],
                "built_present_not_embedded_conservative_extra_root",
            )

            entrypoints["baseline86_cli"].write_bytes(b"\0".join(
                str(plugins[label]).encode()
                for label in ("baseline_cmix", "baseline_paq")
            ))
            embedded, reachability = freeze._embedded_plugin_paths(
                entrypoints, plugins,
            )
            self.assertEqual(
                set(embedded["baseline86_cli"]),
                {"baseline_cmix", "baseline_paq"},
            )
            self.assertEqual(
                reachability["baseline_cmix"], "embedded_absolute_path",
            )
            self.assertEqual(
                reachability["baseline_paq"], "embedded_absolute_path",
            )

            entrypoints["baseline86_cli"].write_bytes(
                str(plugins["baseline_cmix"]).encode()
            )
            with self.assertRaisesRegex(RuntimeError, "only one"):
                freeze._embedded_plugin_paths(entrypoints, plugins)

    def test_depfile_continuation_and_escaped_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "code"
            build = root / "build"
            build.mkdir(parents=True)
            source = root / "source file.hpp"
            source.write_text("header")
            depfile = build / "source.o.d"
            escaped = str(source).replace(" ", "\\ ")
            depfile.write_text(f"CMakeFiles/x.dir/source.o: \\\n  {escaped} \\\n  /etc/hosts\n")
            parsed = freeze._parse_depfile(
                depfile, build_root=build, code_root=root.resolve(),
            )
            self.assertEqual(parsed["code_dependencies"], [source.resolve()])
            self.assertEqual(parsed["system_dependencies"], [Path("/etc/hosts").resolve()])

    def test_depfile_rejects_multiple_or_missing_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "code"
            build = root / "build"
            build.mkdir(parents=True)
            source = root / "x.cpp"
            source.write_text("x")
            depfile = build / "x.o.d"
            depfile.write_text(f"x.o: {source}\ny.o: {source}\n")
            with self.assertRaisesRegex(RuntimeError, "multiple/malformed"):
                freeze._parse_depfile(
                    depfile, build_root=build, code_root=root.resolve(),
                )
            depfile.write_text(f"x.o: {root / 'missing.hpp'}\n")
            with self.assertRaisesRegex(RuntimeError, "stale/incomplete"):
                freeze._parse_depfile(
                    depfile, build_root=build, code_root=root.resolve(),
                )

    def test_fixture_closure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "code"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "driver.py").write_text("VALUE = 1\n")
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.18)\n")
            for relative in freeze.NATIVE_COMPONENTS:
                component = root / relative
                component.mkdir(parents=True)
                (component / "CMakeLists.txt").write_text("project(fixture)\n")
                (component / "source.cpp").write_text("int fixture;\n")
            historical = root / "cpp/build-native-old"
            historical.mkdir(parents=True)
            (historical / "CMakeCache.txt").write_text("STALE=1\n")
            (historical / "CMakeLists.txt").write_text("project(stale)\n")
            component_debug = root / "cpp/native_graph_v1/build-v2"
            component_debug.mkdir()
            (component_debug / "CMakeLists.txt").write_text("project(debug)\n")
            pipeline = root / "cpp/native_pipeline_v1/build"
            baseline = root / "cpp/native_baseline86_v1/build"
            for build in (pipeline, baseline):
                depdir = build / "CMakeFiles/fixture.dir"
                depdir.mkdir(parents=True)
                source = build.parent / "source.cpp"
                (depdir / "source.cpp.o.d").write_text(
                    f"CMakeFiles/fixture.dir/source.cpp.o: \\\n  {source.resolve()} \\\n  /etc/hosts\n"
                )
                (depdir / "source.cpp.o").write_bytes(b"object-fixture")
                assembly = build.parent / "assembly.S"
                assembly.write_text('#include "assembly.inc"\n')
                assembly_header = build.parent / "assembly.inc"
                assembly_header.write_text("/* fixture */\n")
                assembly_object = depdir / "assembly.S.o"
                assembly_object.write_bytes(b"assembly-object-fixture")
                (depdir / "depend.make").write_text(
                    "# CMAKE generated file\n\n"
                    f"CMakeFiles/fixture.dir/assembly.S.o: \\\n"
                    f" {assembly_header.resolve()} \\\n"
                    f" {assembly.resolve()}\n"
                )
                (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE=Release\n")

            cmix = pipeline / "native_graph/libnative_graph_v1_cmix_isolated.so"
            pipeline_paq = pipeline / "native_graph/libnative_graph_v1_paq_isolated.so"
            baseline_cmix = baseline / "native_graph/libnative_graph_v1_cmix_isolated.so"
            baseline_paq = baseline / "native_graph/libnative_graph_v1_paq_isolated.so"
            for path in (cmix, pipeline_paq, baseline_cmix, baseline_paq):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"ELF-fixture")
                path.chmod(0o755)
            binaries = {
                "pipeline_server": pipeline / "native_pipeline_v1_server",
                "cold_decode_cli": pipeline / "native_graph/native_graph_request_cli",
                "policy_cli": pipeline / "native_policy/weighted_grpo_native_policy_cli",
                "features_cli": pipeline / "native_features/native_features_cli",
                "baseline86_cli": baseline / "native_baseline86_cli",
            }
            for label, path in binaries.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = b"ELF-" + label.encode()
                if label in {"pipeline_server", "cold_decode_cli"}:
                    payload += b"\0" + str(cmix.resolve()).encode()
                    payload += b"\0" + str(pipeline_paq.resolve()).encode()
                if label == "baseline86_cli":
                    payload += b"\0" + str(baseline_cmix.resolve()).encode()
                    payload += b"\0" + str(baseline_paq.resolve()).encode()
                path.write_bytes(payload)
                path.chmod(0o755)

            external = root / "external"
            external.mkdir()
            libnc = external / "libnc.so"
            libnc.write_bytes(b"NNCP-LibNC-fixture")
            libnc.chmod(0o755)
            inputs = {}
            for name in ("library", "catalog", "reference"):
                inputs[name] = root / (name + ".json")
                inputs[name].write_text("{}\n")
            arguments = SimpleNamespace(
                code_root=root, pipeline_build=pipeline, baseline_build=baseline,
                **inputs,
            )

            def fake_ldd(path):
                if Path(path).name == "native_pipeline_v1_server":
                    return [libnc], ["linux-vdso.so.1"], (
                        f"libnc.so => {libnc} (0x1)\nlinux-vdso.so.1 (0x2)"
                    )
                return [], [], "statically linked"

            with mock.patch.object(freeze.sys, "platform", "linux"), \
                    mock.patch.object(freeze.shutil, "which", return_value="/usr/bin/ldd"), \
                    mock.patch.object(freeze, "_run_ldd", side_effect=fake_ldd):
                document = freeze.collect(arguments)
            self.assertNotIn("passed", document)
            self.assertIs(document["creates_training_admission"], False)
            self.assertEqual(document["compiler_dependency_file_count"], 2)
            self.assertEqual(document["cmake_asm_dependency_rule_count"], 2)
            self.assertEqual(document["compiled_object_count"], 4)
            for row in document["cmake_asm_dependency_rules"]:
                self.assertEqual(row["dependency_kind"], "cmake_asm_depend_make")
                self.assertTrue(row["object"]["path"].endswith("assembly.S.o"))
                self.assertIn(row["depfile"]["path"], document["frozen_files"])
                self.assertIn(row["object"]["path"], document["frozen_files"])
            self.assertEqual(document["runtime_requirements"], {
                "nncp_libnc_present": True,
                "pipeline_cmix_dso_present": True,
                "pipeline_paq_dso_present": True,
                "baseline_cmix_dso_present": True,
                "baseline_paq_dso_present": True,
            })
            self.assertEqual(
                document["runtime_plugin_reachability"]["baseline_cmix"],
                "embedded_absolute_path",
            )
            self.assertIn(str(libnc), document["frozen_files"])
            self.assertIn(str((scripts / "driver.py").resolve()), document["frozen_files"])
            source_configs = {row["path"] for row in document["source_cmake_configuration"]}
            build_configs = {row["path"] for row in document["build_cmake_configuration"]}
            self.assertNotIn(str((historical / "CMakeLists.txt").resolve()), source_configs)
            self.assertNotIn(str((component_debug / "CMakeLists.txt").resolve()), source_configs)
            self.assertIn(str((pipeline / "CMakeCache.txt").resolve()), build_configs)
            self.assertIn(str((baseline / "CMakeCache.txt").resolve()), build_configs)
            unsigned = {key: value for key, value in document.items()
                        if key != "manifest_sha256"}
            self.assertEqual(
                document["manifest_sha256"], freeze.digest_bytes(freeze.canonical(unsigned)),
            )
            output = Path(directory) / "runtime-freeze.json"
            freeze.write_new(output, document)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                freeze.write_new(output, document)


if __name__ == "__main__":
    unittest.main()
