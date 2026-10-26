#!/usr/bin/env python3
"""Focused tests for the current native prepared-block baseline86 bridge."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepared_block_baseline86_native_executor_v1 as native  # noqa: E402
import prepared_block_baseline86_v1 as protocol  # noqa: E402


def block(dtype: str = "int16") -> protocol.PreparedBlock:
    values = {
        "int16": struct.pack("<hhhh", -3, 0, 17, 17),
        "int64": struct.pack("<qqqq", -3, 0, 17, 17),
        "float64": struct.pack("<dddd", -3.0, -0.0, 17.25, 17.25),
    }[dtype]
    bitmap = b"\x0f"
    return protocol.PreparedBlock(
        dataset_id="native-toy", dataset_provenance_sha256="b" * 64,
        block_id="validation__e__v__000000", entity_id="e",
        split="validation", column="v", dtype=dtype,
        row_count=4, valid_count=4, block_index=0,
        values=values, missing_bitmap=bitmap,
        original_numeric_bytes=len(values),
        values_sha256=hashlib.sha256(values).hexdigest(),
        missing_bitmap_sha256=hashlib.sha256(bitmap).hexdigest(),
    )


def request(label: str, dtype: str = "int16") -> protocol.LabelExecutionRequest:
    catalog = protocol.build_catalog_snapshot()
    record = next(item for item in catalog.labels if item.label == label)
    result, reason = protocol.make_execution_request(catalog, record, block(dtype))
    if result is None:
        raise AssertionError(reason)
    return result


class NativeSupportTests(unittest.TestCase):
    def test_unbridged_dtype_variant_is_explicitly_unsupported(self):
        decision = native.NativeBaseline86Executor().support(request("CORAD"))
        self.assertFalse(decision.supported)
        self.assertIn("int16", decision.reason)
        self.assertFalse(decision.implementation_evidence["upstream_native_claim"])

    def test_all_public_labels_have_an_explicit_bridge_or_dtype_gate(self):
        self.assertEqual(len(native.SUPPORTED_LABELS), 86)
        self.assertIn("Huffman", native.SUPPORTED_LABELS)
        self.assertIn("Density", native.SUPPORTED_LABELS)
        self.assertIn("ALP", native.SUPPORTED_LABELS)
        self.assertIn("Blosc2", native.SUPPORTED_LABELS)

    def test_typed_method_rejects_unregistered_width_without_widening(self):
        with self.assertRaisesRegex(ValueError, "int16"):
            native._logical_type("ALP", "int16")

    def test_known_elf_nan_wire_limit_is_unsupported_not_failed(self):
        values = struct.pack("<Q", 0x7ff8000000001234)
        self.assertIn(
            "NaN payloads",
            native._applicability_reason("Elf", "float64", values),
        )


class NativeRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = native._required_binaries("Huffman")
        if any(not path.is_file() for path in required):
            raise unittest.SkipTest("Huffman C++ action target is not built")

    def test_complete_frame_roundtrips_in_independent_process(self):
        work = request("Huffman")
        result = protocol.benchmark_request(
            native.NativeBaseline86Executor(), work, warmups=0, repeats=1
        )
        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["self_contained_frame"])
        self.assertTrue(result["independent_cold_decode"])
        self.assertTrue(result["bit_exact_values_and_missing_bitmap"])
        self.assertGreater(result["complete_compressed_bytes"], len(work.missing_bitmap))
        self.assertEqual(
            result["rho_complete_bytes_per_original_numeric_byte"],
            result["complete_compressed_bytes"] / len(work.values),
        )

    def test_frame_checksum_rejects_mutation(self):
        work = request("Huffman")
        frame = bytearray(native._make_complete_frame(work))
        frame[-33] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            native._parse_complete_frame(bytes(frame))

    def test_complete_frame_uses_compact_fingerprints_not_action_names(self):
        work = request("Huffman")
        frame = native._make_complete_frame(work)
        header, _, _ = native._parse_complete_frame(frame)
        self.assertLess(header["compact_header_bytes"], 256)
        for action_name in work.registered_pipeline:
            self.assertNotIn(action_name.encode("utf-8"), frame)
        self.assertEqual(
            header["pipeline_contract_sha256"],
            native._pipeline_contract_sha256(work),
        )

    def test_cold_worker_does_not_receive_original_request_bytes(self):
        work = request("Huffman")
        frame = native._make_complete_frame(work)
        decoded = native.NativeBaseline86Executor().decode(work, frame)
        self.assertEqual(decoded.values, work.values)
        self.assertEqual(decoded.missing_bitmap, work.missing_bitmap)
        self.assertTrue(decoded.independent_process)
        self.assertIn("fresh-process", decoded.decoder_identity)

    def test_typed_rlbe_path_roundtrips_as_a_complete_frame(self):
        if any(not path.is_file() for path in native._required_binaries("RLBE")):
            self.skipTest("RLBE C++ targets are not built")
        work = request("RLBE", "int64")
        result = protocol.benchmark_request(
            native.NativeBaseline86Executor(), work, warmups=0, repeats=1
        )
        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["bit_exact_values_and_missing_bitmap"])

    def test_elf_nan_block_is_reported_unsupported_before_encode(self):
        values = struct.pack("<QQ", 0x3ff0000000000000, 0x7ff8000000001234)
        bitmap = b"\x03"
        source = protocol.PreparedBlock(
            dataset_id="elf-nan", dataset_provenance_sha256="d" * 64,
            block_id="validation__e__f__000000", entity_id="e",
            split="validation", column="f", dtype="float64",
            row_count=2, valid_count=2, block_index=0,
            values=values, missing_bitmap=bitmap,
            original_numeric_bytes=len(values),
            values_sha256=hashlib.sha256(values).hexdigest(),
            missing_bitmap_sha256=hashlib.sha256(bitmap).hexdigest(),
        )
        catalog = protocol.build_catalog_snapshot()
        record = next(item for item in catalog.labels if item.label == "Elf")
        work, reason = protocol.make_execution_request(catalog, record, source)
        self.assertIsNone(reason)
        assert work is not None
        decision = native.NativeBaseline86Executor().support(work)
        self.assertFalse(decision.supported)
        self.assertIn("NaN payloads", decision.reason)


if __name__ == "__main__":
    unittest.main()
