"""Typed-port checks using real C++ stage outputs, plus edited-stream inverses.

Run: python3 -B scripts/test_action_semantic_ports.py
No build, training, benchmark, or repository data writes are performed.
"""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
import lzma
from pathlib import Path
import struct
import subprocess
import unittest
import zlib

import action_semantic_ports as ports
import unified_action_runtime as unified


ROOT = Path(__file__).resolve().parents[1]


def cpp(executable: str, command: str, payload: bytes, *args: str) -> bytes:
    completed = subprocess.run([str(ROOT / "cpp/build" / executable), command, *args],
                               input=payload, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(f"{executable} {command}: {completed.stderr.decode(errors='replace')}")
    return completed.stdout


def wrap(payload: bytes, variant: str) -> unified.ActionData:
    header = {"actions": [variant], "io_variant": variant, "source_kind": "bytes",
              "source_count": 37, "max_point": None,
              "upstream_result_header": {"actions": ["legacy-parent"], "source_kind": "float64", "source_count": 3}}
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return unified.ActionData.bytes(b"UAR1" + struct.pack("<I", len(raw)) + raw + payload)


def body(data: unified.ActionData) -> bytes:
    length = struct.unpack_from("<I", data.payload, 4)[0]
    return data.payload[8 + length:]


def edit(view: ports.ActionPorts, name: str, raw: bytes, *, kind: str | None = None) -> ports.ActionPorts:
    changed = []
    found = False
    for stream in view.streams:
        if stream.name == name:
            found = True
            changed.append(replace(stream, payload=raw, kind=kind or stream.kind))
        else:
            changed.append(stream)
    assert found, name
    return replace(view, streams=tuple(changed))


class SemanticPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ints = struct.pack("<5q", 3, 1, 1, 2, 5)
        floats = struct.pack("<5d", 1.25, 1.25, 2.5, 2.5, 3.75)
        frames: dict[str, bytes] = {}
        for family in ["frame_of_reference", "value_reorder", "delta", "xor", "zigzag", "run_length"]:
            name = f"op.{family}.reference_v1"
            frames[name] = cpp("reference_operator_actions_cli", "encode", ints, name)
        name = "op.xor.chimp_history128.reference_v1"
        frames[name] = cpp("reference_operator_actions_cli", "encode", ints, name)
        for codec, target in [("ts2diff", "ts_2diff"), ("simple8b", "simple8b")]:
            frames[f"op.scaling.{target}.cpp_v1"] = cpp(f"{codec}_exact_stages_cli", "scale-forward", floats, "2")
        bos_delta = cpp("bos_exact_stages_cli", "delta-forward", ints)
        frames["op.delta.bos.cpp_v1"] = bos_delta
        frames["op.frame_of_reference.bos.cpp_v1"] = cpp("bos_exact_stages_cli", "for1-forward", bos_delta)
        reorder = cpp("learning_codec_exact_stages_cli", "reger-reorder-forward", ints)
        frames["op.value_reorder.reger.cpp_v1"] = reorder
        ar = cpp("learning_codec_exact_stages_cli", "reger-autoregression-forward", reorder)
        frames["op.frame_of_reference.reger.cpp_v1"] = cpp("learning_codec_exact_stages_cli", "reger-for-forward", ar)
        frames["op.delta.stream_vbyte.cpp_v1"] = cpp("svb_exact_stages_cli", "delta-forward", ints)
        dbp_delta = cpp("dbp_exact_stages_cli", "delta-forward-int", ints)
        frames["op.delta.delta_binary_packed.cpp_v1"] = dbp_delta
        dbp_zigzag = cpp("dbp_exact_stages_cli", "zigzag-forward", dbp_delta)
        frames["op.zigzag.delta_binary_packed.cpp_v1"] = dbp_zigzag
        frames["op.integer_code.uleb128.dbp_residuals.cpp_v2"] = cpp(
            "dbp_exact_stages_cli", "varint-forward", dbp_zigzag)
        raw_zigzag = cpp("pure_numeric_v1/pure_numeric_operator_v1", "zigzag-forward", ints)
        frames["op.zigzag.signed64.raw.cpp_v2"] = raw_zigzag
        frames["op.integer_code.uleb128.reference_values.cpp_v2"] = cpp(
            "unsigned_leb128_cli", "raw-forward", raw_zigzag)
        leco_signed = cpp("learning_codec_exact_stages_cli", "leco-regression-forward", ints)
        frames["op.regression_predictor.leco_linear.cpp_v2"] = leco_signed
        frames["op.zigzag.leco_residual.cpp_v2"] = cpp(
            "learning_codec_exact_stages_cli", "leco-zigzag-forward", leco_signed)
        reger_signed = cpp("learning_codec_exact_stages_cli", "reger-delta-forward", reorder)
        frames["op.delta.reger_signed_residual.cpp_v2"] = reger_signed
        frames["op.zigzag.reger_residual.cpp_v2"] = cpp(
            "learning_codec_exact_stages_cli", "reger-zigzag-forward", reger_signed)
        bos64_signed = cpp("bos_exact_stages_cli", "adapter-plain-delta-forward", ints)
        frames["op.delta.bos64_signed_residual.cpp_v2"] = bos64_signed
        frames["op.zigzag.bos64_residual.cpp_v2"] = cpp(
            "bos_exact_stages_cli", "adapter-zigzag-forward", bos64_signed)
        snappy_lz = cpp("snappy_exact_stages_cli", "lz-forward", b"aaabbbccccccd")
        snappy_tokens = cpp("snappy_exact_stages_cli", "token-forward", snappy_lz)
        frames["op.integer_code.uleb128.snappy_raw_size.cpp_v2"] = cpp(
            "snappy_exact_stages_cli", "integer-forward", snappy_tokens)
        predictor = cpp("p04_exact_stages_cli", "sprintz-predictor-forward-double", floats * 2)
        frames["op.zigzag.sprintz.cpp_v1"] = cpp("p04_exact_stages_cli", "sprintz-zigzag-forward", predictor)
        gorilla_xor = cpp("gorilla_exact_stages_cli", "xor-forward-float", floats)
        frames["op.xor.gorilla.cpp_v1"] = gorilla_xor
        frames["op.trailing_zero_pack.gorilla.cpp_v1"] = cpp("gorilla_exact_stages_cli", "pack-forward", gorilla_xor)
        frames["op.xor.chimp128.cpp_v1"] = cpp("chimp128_exact_stages_cli", "xor-forward-float", floats)
        pongo_erase = cpp("pongo_exact_stages_cli", "eraser-forward", floats)
        pongo_xor = cpp("pongo_exact_stages_cli", "xor-forward", pongo_erase)
        frames["op.xor.pongo.cpp_v1"] = pongo_xor
        frames["op.trailing_zero_pack.pongo.cpp_v1"] = cpp("pongo_exact_stages_cli", "pack-forward", pongo_xor)
        for codec in ["fpc", "pfpc"]:
            residual = cpp(f"{codec}_exact_stages_cli", "xor-forward", floats)
            frames[f"op.trailing_zero_pack.{codec}.cpp_v1"] = cpp(f"{codec}_exact_stages_cli", "pack-forward", residual)
        for width in [1, 4, 8]:
            frames[f"op.block_transform.blosc_shuffle.typesize{width}.cpp_reference_v1"] = cpp(
                "general_codec_reference_stages_cli", f"shuffle{width}-forward", bytes(range(35)))
        frames["op.run_length.bzip2_uint8.cpp_reference_v1"] = cpp(
            "general_codec_reference_stages_cli", "rle-forward", b"aaabbbccccccd")
        for codec in ["blosc", "blosc2", "gzip", "zlib", "xz"]:
            family = "block_state_control" if codec.startswith("blosc") else "container_frame"
            named = "blosc1" if codec == "blosc" else codec
            frames[f"op.{family}.{named}.cpp_reference_v1"] = cpp(
                "general_codec_reference_stages_cli", f"{codec}-frame-forward", b"retained payload")
        frames["op.container_frame.corad_lossless.cpp_v2"] = b"CRD2" + struct.pack("<Q", 15) + b"CZL2" + struct.pack("<Q", 3) + b"abc"
        data = b"container parts preserved " * 3
        frames["op.container_frame.gzip.cpp_v1"] = gzip.compress(data, mtime=0)
        frames["op.container_frame.zlib.cpp_v1"] = zlib.compress(data)
        frames["op.container_frame.xz.cpp_v1"] = lzma.compress(data)
        # Existing wire-stage fixtures: the tokens are already encoded bytes.
        frames["op.match_token_format.native_lz.ctrl8_offset16_len8.bias5"] = b"\x00abcdefgh"
        obj = {"format": "alp", "stage": "wires", "count": 3,
               "blocks": [{"count": 2, "fac": 0, "exp": 2, "exceptions": [[0, 2**64 - 1]], "wires": ["006162", "006364"]},
                          {"count": 1, "raw": "000000000000f03f"}]}
        frames["op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1"] = b"LNS1" + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        buff_scale = cpp("learning_codec_exact_stages_cli", "buff-scaling-forward", floats)
        buff_for = cpp("learning_codec_exact_stages_cli", "buff-for-forward", buff_scale)
        width = cpp("learning_codec_exact_stages_cli", "buff-byte-plane-width", buff_for)[0]
        frames["op.scaling.buff.cpp_v1"] = buff_scale
        frames["op.frame_of_reference.buff.cpp_v2"] = buff_for
        frames["op.block_transform.buff_byte_plane_split.cpp_v2"] = cpp(
            "learning_codec_exact_stages_cli", "buff-byte-plane-forward", buff_for, str(width))
        cls.frames = frames

    def test_all_document_variants_rebuild_and_preserve_header(self):
        self.assertEqual(len(self.frames), 52)
        contracts = ports.supported_layouts()["document_merged_variant_layouts"]
        self.assertEqual(set(contracts), set(self.frames))
        typed, encoded = 0, 0
        for variant, payload in self.frames.items():
            with self.subTest(variant=variant):
                source = wrap(payload, variant)
                view = ports.to_ports(source)
                self.assertIn(view.metadata["schema"], contracts[variant]["schemas"])
                rebuilt = ports.from_ports(view)
                self.assertEqual(rebuilt, source)
                self.assertEqual(ports.to_ports(rebuilt).header, view.header)
                if view.metadata["schema"] == "opaque_bytes":
                    encoded += 1
                    self.assertEqual(view.streams[0].kind, "bytes")
                else:
                    typed += 1
        self.assertEqual((typed, encoded), (47, 5))

    def test_reference_stream_edit_changes_actual_cpp_inverse(self):
        name = "op.frame_of_reference.reference_v1"
        raw = struct.pack("<4q", 100, 101, 101, 102)
        view = ports.to_ports(wrap(cpp("reference_operator_actions_cli", "encode", raw, name), name))
        changed = edit(view, "main", struct.pack("<4Q", 0, 1, 1, 3))
        result = cpp("reference_operator_actions_cli", "decode", body(ports.from_ports(changed)))
        self.assertEqual(struct.unpack("<4q", result), (100, 101, 101, 103))
        self.assertEqual(changed.header, ports.to_ports(ports.from_ports(changed)).header)

    def test_unsigned_delta_lanes_edit_changes_inverse_without_signed_cast(self):
        raw = struct.pack("<2Q", 2**32 - 1, 2**32 + 1)
        packet = cpp("svb_exact_stages_cli", "delta-forward", raw)
        view = ports.to_ports(wrap(packet, "op.delta.stream_vbyte.cpp_v1"))
        self.assertEqual([s.kind for s in view.streams], ["uint32", "uint32"])
        changed = edit(view, "low_deltas", struct.pack("<2I", 2**32 - 1, 3))
        decoded = cpp("svb_exact_stages_cli", "delta-inverse", body(ports.from_ports(changed)))
        self.assertEqual(struct.unpack("<2Q", decoded), (2**32 - 1, 2**32 + 2))

    def test_scaling_edit_and_layout_validation(self):
        variant = "op.scaling.simple8b.cpp_v1"
        view = ports.to_ports(wrap(self.frames[variant], variant))
        changed = edit(view, "values", struct.pack("<5q", 126, 125, 250, 250, 375))
        result = cpp("simple8b_exact_stages_cli", "scale-inverse", body(ports.from_ports(changed)))
        self.assertEqual(struct.unpack("<5d", result)[0], 1.26)
        with self.assertRaises(ports.PortSchemaError):
            ports.from_ports(edit(view, "scale", struct.pack("<I", 19)))

    def test_modified_permutation_reorders_and_invalid_permutation_rejects(self):
        variant = "op.value_reorder.reger.cpp_v1"
        view = ports.to_ports(wrap(self.frames[variant], variant))
        changed = edit(view, "original_index", struct.pack("<5I", 0, 1, 2, 3, 4))
        result = cpp("learning_codec_exact_stages_cli", "reger-reorder-inverse", body(ports.from_ports(changed)))
        self.assertEqual(struct.unpack("<5q", result), (1, 1, 2, 3, 5))
        with self.assertRaises(ports.PortSchemaError):
            ports.from_ports(edit(view, "original_index", struct.pack("<5I", 0, 0, 2, 3, 4)))

    def test_sprintz_anchor_and_tail_are_not_zigzag_codes(self):
        variant = "op.zigzag.sprintz.cpp_v1"
        view = ports.to_ports(wrap(self.frames[variant], variant))
        self.assertEqual([(s.name, s.kind, s.count) for s in view.streams],
                         [("anchors", "int64", 1), ("residuals", "uint64", 8), ("raw_tail", "int64", 1)])
        self.assertEqual(view.metadata["word_kind"], "float64_bits")

    def test_container_fields_are_editable_bytes_without_recomputed_checksum(self):
        variant = "op.container_frame.gzip.cpp_v1"
        view = ports.to_ports(wrap(self.frames[variant], variant))
        self.assertEqual([s.name for s in view.streams], ["container_header", "core", "trailer"])
        core = next(s for s in view.streams if s.name == "core")
        changed = edit(view, "core", bytes([core.payload[0] ^ 1]) + core.payload[1:])
        rebuilt = body(ports.from_ports(changed))
        self.assertEqual(rebuilt[:10], self.frames[variant][:10])
        self.assertEqual(rebuilt[-8:], self.frames[variant][-8:])
        self.assertNotEqual(rebuilt, self.frames[variant])

    def test_lns_multistream_changes_are_serialized_and_side_metadata_survives(self):
        variant = "op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1"
        view = ports.to_ports(wrap(self.frames[variant], variant))
        self.assertEqual([s.name for s in view.streams], ["block0.wire0", "block0.wire1", "block1.raw"])
        changed = edit(view, "block0.wire1", b"\0XY")
        obj = json.loads(body(ports.from_ports(changed))[4:])
        self.assertEqual(obj["blocks"][0]["wires"][1], "005859")
        self.assertEqual(obj["blocks"][0]["exceptions"][0][1], 2**64 - 1)
        self.assertEqual(obj["blocks"][1]["raw"], "000000000000f03f")

    def test_raw_inputs_preserve_float_bits_and_do_not_magic_sniff(self):
        float_bits = struct.pack("<4Q", 0, 1 << 63, 0x7ff8000000000001, 0x3ff0000000000000)
        for source in [unified.ActionData("float64", float_bits, 4), unified.ActionData.ints([-(2**63), 2**63 - 1]),
                       unified.ActionData.bytes(b"SHF1not-a-packet")]:
            self.assertEqual(ports.from_ports(ports.to_ports(source)), source)
        self.assertEqual(ports.to_ports(unified.ActionData.bytes(b"SHF1not-a-packet")).metadata["schema"], "raw")

    def test_typed_shape_missing_extra_and_malformed_packet_rejected(self):
        for kind, width in ports.supported_layouts()["stream_kinds"].items():
            with self.subTest(kind=kind):
                with self.assertRaises(ports.PortSchemaError):
                    ports.TypedStream("x", kind, b"\0" * (width - 1), 1)
        for count in [-1, True, 1.5]:
            with self.assertRaises(ports.PortSchemaError):
                ports.TypedStream("x", "bytes", b"", count)
        variant = "op.value_reorder.reger.cpp_v1"
        view = ports.to_ports(wrap(self.frames[variant], variant))
        with self.assertRaises(ports.PortSchemaError):
            ports.from_ports(replace(view, streams=view.streams[:-1]))
        with self.assertRaises(ports.PortSchemaError):
            ports.from_ports(replace(view, streams=view.streams + (ports.TypedStream("extra", "bytes", b"a", 1),)))
        with self.assertRaises(ports.PortSchemaError):
            ports.to_ports(wrap(b"RRO1" + struct.pack("<I", 2**32 - 1), variant))
        with self.assertRaises(ports.PortSchemaError):
            ports.to_ports(unified.ActionData.bytes(b"UAR1\xff\xff\xff\xff"))

    def test_empty_actual_cpp_packets_rebuild(self):
        for executable, command, variant in [
            ("gorilla_exact_stages_cli", "xor-forward-int", "op.xor.gorilla.cpp_v1"),
            ("chimp128_exact_stages_cli", "xor-forward-int", "op.xor.chimp128.cpp_v1"),
            ("pongo_exact_stages_cli", "eraser-forward", "op.eraser.pongo.cpp_v1"),
            ("bos_exact_stages_cli", "delta-forward", "op.delta.bos.cpp_v1"),
            ("learning_codec_exact_stages_cli", "reger-reorder-forward", "op.value_reorder.reger.cpp_v1"),
        ]:
            with self.subTest(variant=variant):
                source = wrap(cpp(executable, command, b""), variant)
                self.assertEqual(ports.from_ports(ports.to_ports(source)), source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
