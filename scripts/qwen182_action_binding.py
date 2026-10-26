"""Versioned Qwen bit-container binding for the current algorithm registry.

This is an execution/search-space adapter, NOT a Beam50 or RL benchmark. The
frozen canonical206 Qwen experiment files are deliberately not imported or
modified. A numeric BF16-to-float conversion would change the task: here all
root views preserve the exact input bytes, including NaNs and signed zeros.

Public recipes are records, never bare integer lists. Use ``normalize_record``
to migrate an explicitly versioned old recipe, then persist its IDs AND its
I/O variants and per-step parameters. Cache keys retain all three. Candidate
enumeration is the broad Qwen bytes-compatible search, not the narrower generic registered-sequence grammar;
an enumerated edge can still fail the runtime's stage-packet/type checks.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import zlib

import primitive_operator_action_runtime as p
import unified_action_runtime as u
import native_lz_parameters as lz
import native_window_parameters as window_parameters
import current_action_parameters as action_parameters
import max_value_bitpack as bitpack
import minimum_for as minimum
from canonical182_policy_paths import (
    PolicyPath, catalog_fingerprint, path_from_names, path_from_record,
    path_record, variants_for,
)
from canonical206_search_env import HEADER, decode_frame
from compact_pipeline_frame import WIRE_FORMAT, compact_frame, expand_frame


BINDING_VERSION = 'qwen-canonical127-bitviews-parameters-v10'
LEGACY_VERSION = 'cpp-canonical-206-v1'
PREVIOUS_VERSION = 'cpp-canonical-182-v1'
ROOT_KIND_PRIORITY = ('int64', 'float64', 'bytes')
ROOT_SEMANTICS = {
    'schema': 'original-bytes-bit-view-v1',
    'source_representation': 'unchanged original file/payload bytes',
    'typed_byte_order': 'little-endian',
    'kind_priority': list(ROOT_KIND_PRIORITY),
    'numeric_cast': False,
    'uint64_value_profile': 'max_value_width.uint64_le uses original bytes as unsigned LE words',
    'typed_alignment_bytes': 8,
    'c206_original_alignment_bytes': 8,
}
# Exact old *variant names*, not IDs in the newly reordered algorithm table.
FLOAT_SCALE_ROOT_VARIANTS = frozenset((
    'op.scaling.tsfile_rle.cpp_v1',
    'op.scaling.ts_2diff.cpp_v1',
    'op.scaling.simple8b.cpp_v1',
    'op.scaling.delta_binary_packed.cpp_v1',
))


class QwenActionBinding:
    """Current algorithms with explicit I/O profiles and per-step parameters."""

    def __init__(self, max_depth=10):
        if type(max_depth) is not int or not 1 <= max_depth <= 10:
            raise ValueError('max_depth must be an integer in 1..10')
        self.max_depth = max_depth
        self.registry_version = p.ACTION_REGISTRY_VERSION
        self.algorithms = {s.action_id: s for s in p.POLICY_ACTIONS}
        self.steps = tuple(
            path_from_names((variant,))
            for action in self.algorithms
            for variant in variants_for(action)
        )
        self.descriptors = {
            step.io_variants[0]: u.describe_policy_action(
                step[0], io_variant=step.io_variants[0])
            for step in self.steps
        }
        # This fixed slice belongs explicitly to the old 206 registry. Testing
        # current_id < 29 would silently treat unrelated algorithms as refs.
        self.reference_variants = frozenset(
            spec.name for spec in p.LEGACY_206_POLICY_ACTIONS[:29]
            if spec.name in self.descriptors
        )
        if not self.reference_variants <= self.descriptors.keys():
            raise ValueError('legacy reference variants are missing; re-audit binding')
        if not FLOAT_SCALE_ROOT_VARIANTS <= self.descriptors.keys():
            raise ValueError('legacy float-scale variants are missing; re-audit binding')
        self.fingerprint = catalog_fingerprint({
            'binding_version': BINDING_VERSION,
            'root_semantics': ROOT_SEMANTICS,
            'max_depth': max_depth,
            'candidate_rule': 'all-roots-then-bytes-or-reference-prefix-v1',
            'storage_wire_format': WIRE_FORMAT,
        })

    def _record(self, path):
        return dict(path_record(path), binding_version=BINDING_VERSION,
                    max_depth=self.max_depth,
                    storage_wire_format=WIRE_FORMAT,
                    root_semantics=dict(ROOT_SEMANTICS,
                                        kind_priority=list(ROOT_KIND_PRIORITY)))

    def normalize_record(self, record):
        """Validate a recipe or migrate historical IDs, variants and parameters.

        Historical registry-only recipes can be translated. An asserted old
        binding/storage contract is rejected explicitly: its measured sizes
        do not describe the current execution and metadata contract.
        """
        if not isinstance(record, Mapping):
            raise ValueError('recipe must be a versioned record, not bare IDs')
        if ('policy_registry_version' in record and 'registry_version' in record
                and record['policy_registry_version'] != record['registry_version']):
            raise ValueError('conflicting registry version fields')
        version = record.get('policy_registry_version', record.get('registry_version'))
        if version not in (self.registry_version, p.PREVIOUS_ACTION_REGISTRY_VERSION,
                           p.LEGACY_ACTION_REGISTRY_VERSION,
                           "cpp-canonical-129-v1", "cpp-canonical-137-v1",
                           "cpp-canonical-140-v1", "cpp-canonical-143-v1",
                           "cpp-canonical-156-v1", PREVIOUS_VERSION, LEGACY_VERSION):
            raise ValueError('recipe needs an explicit supported policy_registry_version')
        if record.get('binding_version', BINDING_VERSION) != BINDING_VERSION:
            raise ValueError('Qwen binding version mismatch')
        if record.get('storage_wire_format', WIRE_FORMAT) != WIRE_FORMAT:
            raise ValueError('storage wire format mismatch; old sizes require remeasurement')
        if record.get('max_depth', self.max_depth) != self.max_depth:
            raise ValueError('recorded max_depth differs from this binding')
        if record.get('root_semantics', ROOT_SEMANTICS) != ROOT_SEMANTICS:
            raise ValueError('root semantics mismatch; numeric conversion is not allowed')
        if version == self.registry_version:
            variants = record.get('io_variants', record.get('selected_io_variants'))
            if variants is None or any(v is None for v in variants):
                raise ValueError('current recipes require explicit io_variants')
        path = path_from_record(record)
        if not 1 <= len(path) <= self.max_depth:
            raise ValueError(f'recipe requires 1..{self.max_depth} codec steps; STOP is not a step')
        # path_from_record rejects STOP, out-of-range IDs and mismatched variants.
        normalized = self._record(path)
        if 'input_binding' in record:
            source = record['input_binding']
            if not isinstance(source, Mapping) or set(source) != {
                'root_kind', 'root_count', 'original_bytes', 'original_sha256'
            }:
                raise ValueError('malformed input_binding')
            normalized['input_binding'] = dict(source)
        return normalized

    def path(self, record):
        return path_from_record(self.normalize_record(record))

    def cache_key(self, record):
        """Bind explicit source records; unbound keys need a per-input cache."""
        normalized = self.normalize_record(record)
        source = normalized.get('input_binding')
        source_key = None if source is None else tuple(source[name] for name in (
            'root_kind', 'root_count', 'original_bytes', 'original_sha256'))
        return self.registry_version, self.fingerprint, self.path(normalized), source_key

    def reference_count(self, record):
        count = 0
        for variant in self.path(record).io_variants:
            if variant not in self.reference_variants:
                break
            count += 1
        return count

    def candidate_steps(self, prefix=None):
        """Return versioned one-step records; no claim each branch is legal.

        Root: enumerate every explicit I/O branch. Later: bytes-compatible
        variants plus reference variants while the entire prefix is reference.
        This deliberately allows repeats and cross-family attempts as before.
        """
        path = PolicyPath() if prefix is None else self.path(prefix)
        if len(path) >= self.max_depth:
            return ()
        all_reference = bool(path) and all(
            v in self.reference_variants for v in path.io_variants)
        return tuple(self._record(step) for step in self.steps
                     if not path
                     or 'bytes' in self.descriptors[step.io_variants[0]].accepted_input_kinds
                     or (all_reference and step.io_variants[0] in self.reference_variants))

    def root_data(self, raw, record):
        """Choose the original int64 -> float64 -> bytes bit-view priority."""
        raw = bytes(raw)
        variant = self.path(record).io_variants[0]
        kinds = self.descriptors[variant].accepted_input_kinds
        kind = 'bytes' if variant in (bitpack.PLAIN_VARIANT, minimum.PLAIN_VARIANT) else next(kind for kind in ROOT_KIND_PRIORITY if kind in kinds)
        if kind != 'bytes' and len(raw) % 8:
            raise ValueError('typed bit view requires raw length divisible by eight')
        return u.ActionData(kind, raw, len(raw) if kind == 'bytes' else len(raw) // 8)

    def input_record(self, raw, record):
        """Persist the concrete root kind/count and source hash for a replay."""
        normalized = self.normalize_record(record)
        source = self.root_data(raw, normalized)
        expected = {
            'root_kind': source.kind, 'root_count': source.count,
            'original_bytes': len(source.payload),
            'original_sha256': hashlib.sha256(source.payload).hexdigest(),
        }
        if 'input_binding' in normalized and normalized['input_binding'] != expected:
            raise ValueError('input_binding differs from supplied original bytes/root view')
        return dict(normalized, input_binding=expected)

    def encode_recipe(self, raw, record, *, wire_format='compact'):
        """Encode with per-layer exact inverses and a complete C206 decode gate.

        Leading reference variants remain a grouped compatibility pipeline;
        grouping does not assert that those implementations are atomic. Every
        other layer uses current run_policy_action with an explicit variant.
        Default storage compacts the final metadata once; 'legacy' is an
        explicit compatibility/diagnostic mode. This method includes inverse
        gates, so its wall time must not be called pure compression time.
        """
        raw = bytes(raw)
        if wire_format not in ('compact', 'legacy'):
            raise ValueError('wire_format must be compact or legacy')
        if len(raw) % 8:
            raise ValueError('C206 original bit-container length must be divisible by eight')
        record = self.input_record(raw, record)
        path = self.path(record)
        current = self.root_data(raw, record)
        nrefs = self.reference_count(record)
        layers = 0
        if nrefs:
            names = list(path.io_variants[:nrefs])
            before = current
            # Reference variants have no parameters. Keep their grouped
            # compatibility execution; a parameter list selects per-step
            # policy execution and would change this pipeline's semantics.
            current = u.run_actions(names, before, max_point=None)
            if u.run_actions(names, current, reverse=True) != before:
                raise ValueError('reference-prefix bit-exact gate failed')
            layers = 1
        for action, variant, parameters in zip(
                path[nrefs:], path.io_variants[nrefs:], path.parameters[nrefs:]):
            before = current
            current = u.run_policy_action(
                action, before, registry_version=self.registry_version,
                io_variant=variant,
                parameters=parameters,
                max_point=0 if not layers and variant in FLOAT_SCALE_ROOT_VARIANTS else None,
            )
            inverse = u.run_policy_action(
                action, current, registry_version=self.registry_version,
                io_variant=variant, parameters=parameters, reverse=True,
            )
            if inverse != before:
                raise ValueError(f'action {action}/{variant} bit-exact gate failed')
            layers += 1
        # Keep the previous C206 bit-container convention. The code is not a
        # claim the BF16 bytes are numeric int64 values; no cast is performed.
        frame = HEADER.pack(b'C206', 1, layers, len(raw) // 8,
                            zlib.crc32(raw)) + current.payload
        if decode_frame(frame) != raw:
            raise ValueError('complete C206 bit-exact gate failed')
        if wire_format == 'legacy':
            return frame
        wire = compact_frame(frame)
        if expand_frame(wire) != frame:
            raise ValueError('compact frame must preserve the exact original frame')
        return wire

    @staticmethod
    def decode_recipe(wire):
        """Decode compact or legacy storage through the same codec inverses."""
        return decode_frame(expand_frame(wire))


def main():
    """Print the binding inventory only; no model download or experiment run."""
    import json
    binding = QwenActionBinding()
    print(json.dumps({
        'binding_version': BINDING_VERSION,
        'policy_registry_version': binding.registry_version,
        'algorithm_count': len(binding.algorithms),
        'io_variant_count': len(binding.steps),
        'parameterized_algorithm_count': len(action_parameters.schemas()),
        'parameter_schemas': action_parameters.schemas(),
        'lz_parameter_choice_count': len(lz.parameter_candidates()),
        'lz_default_parameters': lz.default_parameters(),
        'window_parameter_choice_count': len(window_parameters.parameter_candidates()),
        'window_default_parameters': window_parameters.default_parameters(),
        'stop_id': p.STOP_ACTION_ID,
        'max_depth': binding.max_depth,
        'catalog_fingerprint': binding.fingerprint,
        'storage_wire_format': WIRE_FORMAT,
        'root_semantics': ROOT_SEMANTICS,
        'performance_experiment_run': False,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
