"""Algorithm paths retain independent I/O and parameter choices in all identities."""
from __future__ import annotations
import hashlib
import json
import operator
import primitive_operator_action_runtime as p
import native_lz_parameters as lz
import current_action_parameters as action_parameters
import max_value_bitpack as bitpack
import minimum_for as minimum
import optimal_rle_actions as rle


class PolicyPath(tuple):
    def __new__(cls, ids=(), io_variants=None, parameters=None):
        if isinstance(ids, cls) and io_variants is None and parameters is None:
            return ids
        source = ids if isinstance(ids, cls) else None
        try:
            values = tuple(operator.index(i) if not isinstance(i, bool) else None for i in ids)
        except TypeError as exc:
            raise ValueError('policy IDs must be integers') from exc
        if any(i is None for i in values):
            raise ValueError('boolean values are not policy IDs')
        variants = tuple(io_variants) if io_variants is not None else (source.io_variants if source else (None,) * len(values))
        params = tuple(parameters) if parameters is not None else (source.parameters if source else ({},) * len(values))
        if len(values) != len(variants) or len(values) != len(params):
            raise ValueError('one io_variant and parameter mapping required per policy ID')
        if any(not isinstance(v, dict) for v in params):
            raise ValueError('step parameters must be dictionaries')
        obj = tuple.__new__(cls, values)
        obj.io_variants = variants
        obj._parameter_json = tuple(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False) for v in params)
        return obj

    @property
    def parameters(self):
        # Returning copies prevents a caller from mutating a cache key.
        return tuple(json.loads(v) for v in self._parameter_json)

    def __setattr__(self, name, value):
        if hasattr(self, name): raise AttributeError('PolicyPath is immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name): raise AttributeError('PolicyPath is immutable')
    def __reduce__(self): return (PolicyPath, (tuple(self), self.io_variants, self.parameters))
    def _has_choices(self): return any(self.io_variants) or any(v != '{}' for v in self._parameter_json)
    def __hash__(self):
        return hash((tuple(self), self.io_variants, self._parameter_json)) if self._has_choices() else hash(tuple(self))
    def __eq__(self, other):
        if isinstance(other, PolicyPath):
            return tuple.__eq__(self, other) and self.io_variants == other.io_variants and self._parameter_json == other._parameter_json
        return not self._has_choices() and tuple.__eq__(self, other)
    def __ne__(self, other): return not self == other
    def __lt__(self, other):
        other = PolicyPath(other)
        return (tuple(self), tuple(v or '' for v in self.io_variants), self._parameter_json) < (tuple(other), tuple(v or '' for v in other.io_variants), other._parameter_json)
    def __le__(self, other): return self == other or self < other
    def __gt__(self, other): return not self <= other
    def __ge__(self, other): return not self < other
    def __getitem__(self, key):
        value = tuple.__getitem__(self, key)
        return PolicyPath(value, self.io_variants[key], self.parameters[key]) if isinstance(key, slice) else value
    def __add__(self, other):
        other = PolicyPath(other)
        return PolicyPath(tuple(self) + tuple(other), self.io_variants + other.io_variants, self.parameters + other.parameters)
    def __radd__(self, other): return PolicyPath(other) + self


def variants_for(action):
    try:
        spec = p.POLICY_ACTION_BY_ID[int(action)] if isinstance(action, int) else p.POLICY_ACTION_BY_NAME[action]
    except KeyError as exc:
        raise ValueError(f'unknown algorithm policy ID/name (STOP is not a codec step): {action!r}') from exc
    return tuple(p.POLICY_IO_VARIANTS.get(spec.name, (spec.name,)))


def resolve_variant_name(name):
    if name == rle.ALGORITHM_NAME:
        return p.POLICY_ACTION_BY_NAME[name].action_id, rle.WORDS_VARIANT
    name = rle.upgrade_variant(name)
    module = action_parameters.module_for_name(name)
    if module is not None:
        action = p.POLICY_ACTION_BY_NAME[module.ALGORITHM_NAME].action_id
        variant = action_parameters.canonical_io_variant(name)
        if variant not in variants_for(action):
            raise ValueError(f'{name} requires an explicit io_variant')
        return action, variant
    if name in (bitpack.ALGORITHM_NAME, minimum.ALGORITHM_NAME):
        return p.POLICY_ACTION_BY_NAME[name].action_id, bitpack.PLAIN_VARIANT if name == bitpack.ALGORITHM_NAME else minimum.PLAIN_VARIANT
    canonical = p.LEGACY_206_ACTION_CANONICAL_NAMES.get(name, name)
    policy_name = p.IO_VARIANT_TO_POLICY_NAME.get(canonical, canonical)
    if policy_name not in p.POLICY_ACTION_BY_NAME: raise ValueError(f'unknown action/variant name: {name}')
    policy = p.POLICY_ACTION_BY_NAME[policy_name]
    choices = variants_for(policy.action_id)
    if canonical not in choices:
        if len(choices) != 1: raise ValueError(f'{name} requires an explicit io_variant')
        canonical = choices[0]
    return policy.action_id, canonical


def concrete_path(ids=(), io_variants=None, parameters=None):
    path = PolicyPath(ids, io_variants, parameters)
    variants, params = [], []
    for action, variant, supplied in zip(path, path.io_variants, path.parameters):
        choices = variants_for(action)
        module = action_parameters.BY_NAME.get(p.POLICY_ACTION_BY_ID[action].name)
        original_variant = variant
        if variant is not None and module is not None:
            variant = action_parameters.canonical_io_variant(variant)
        if variant is None:
            if p.POLICY_ACTION_BY_ID[action].name == bitpack.ALGORITHM_NAME:
                variant = bitpack.PLAIN_VARIANT
            elif p.POLICY_ACTION_BY_ID[action].name == minimum.ALGORITHM_NAME:
                variant = minimum.PLAIN_VARIANT
            elif p.POLICY_ACTION_BY_ID[action].name == rle.ALGORITHM_NAME:
                variant = rle.WORDS_VARIANT
            elif len(choices) != 1: raise ValueError(f'policy ID {action} requires io_variant; choices={choices}')
            else:
                variant = choices[0]
        if variant not in choices: raise ValueError(f'io_variant {variant!r} does not belong to policy ID {action}')
        if module is not None:
            supplied = action_parameters.resolve_parameters(
                module.ALGORITHM_NAME, original_variant or variant, supplied or None)
        elif supplied:
            raise ValueError('this algorithm has no learnable parameters')
        variants.append(variant); params.append(supplied)
    return PolicyPath(path, variants, params)


def path_from_names(names, parameters=None):
    names = tuple(names)
    pairs = [resolve_variant_name(rle.upgrade_variant(name, names[index - 1] if index else None))
             for index, name in enumerate(names)]
    params = parameters if parameters is not None else tuple(action_parameters.parameters_from_name(name) or {} for name in names)
    for name, supplied in zip(names, params):
        historical = action_parameters.parameters_from_name(name)
        if historical is not None and supplied and supplied != historical:
            raise ValueError('historical action name conflicts with explicit parameters')
    return concrete_path((i for i, _ in pairs), (v for _, v in pairs), params)


def path_from_record(record, *, registry_version=None):
    ids = record.get('ids', record.get('selected_ids'))
    if ids is None: raise ValueError('path record requires ids or selected_ids')
    version = record.get('policy_registry_version', record.get('registry_version', registry_version))
    if version is None: raise ValueError('integer path IDs require an explicit policy_registry_version')
    variants = record.get('io_variants', record.get('selected_io_variants'))
    params = record.get('parameters', record.get('selected_parameters'))
    path = PolicyPath(ids, variants, params)
    if version == p.ACTION_REGISTRY_VERSION: return concrete_path(path)
    translated = [p.migrate_policy_action(i, registry_version=version, io_variant=v, parameters=q or None)
                  for i, v, q in zip(path, path.io_variants, path.parameters)]
    # A005 was an identity stage. Versioned recipe migration drops that step;
    # old encoded frames still require the frozen v1 decoder path.
    translated = [row for row in translated
                  if row.get('migration_kind') != 'retired_identity_bypass']
    expanded = []
    for row in translated:
        if row.get('migration_kind') != 'replacement_recipe':
            expanded.append(row)
            continue
        ids = row.get('replacement_policy_action_ids', ())
        io_variants = row.get('replacement_io_variants', ())
        parameters = row.get('replacement_parameters', ())
        if not (len(ids) == len(io_variants) == len(parameters)):
            raise ValueError('invalid replacement recipe migration')
        expanded.extend(
            {'policy_action_id': action_id, 'io_variant': variant, 'parameters': supplied}
            for action_id, variant, supplied in zip(ids, io_variants, parameters)
        )
    translated = expanded
    if not translated:
        return concrete_path(())
    if any(r['io_variant'] is None for r in translated):
        raise ValueError('historical merged actions require their original explicit I/O profiles')
    # The historical TsFile selector consumed RLS1 after decimal scaling. The
    # replacement has an explicit profile for that contract, distinct from raw
    # 64-bit input. This is recipe migration, never old-frame replay.
    for index, row in enumerate(translated):
        if (index and row['io_variant'] == rle.WORDS_VARIANT
                and translated[index - 1]['io_variant'] == rle.PRODUCERS[rle.SCALED_VARIANT]):
            row['io_variant'] = rle.SCALED_VARIANT
    return concrete_path([r['policy_action_id'] for r in translated], [r['io_variant'] for r in translated],
                         [r.get('parameters', {}) for r in translated])


def path_record(path):
    path = concrete_path(path)
    return {'ids': list(path), 'io_variants': list(path.io_variants), 'parameters': list(path.parameters),
            'policy_registry_version': p.ACTION_REGISTRY_VERSION}


def stage_signature(path):
    path = PolicyPath(path)
    return tuple(zip(path, path.io_variants))


def same_stages(left, right): return stage_signature(left) == stage_signature(right)


def catalog_fingerprint(extra=None):
    payload = {'registry_version': p.ACTION_REGISTRY_VERSION,
               'actions': [(s.action_id, s.name, variants_for(s.action_id)) for s in p.POLICY_ACTIONS],
               'parameter_catalog': action_parameters.all_candidates(), 'parameter_fingerprint': action_parameters.parameter_fingerprint(),
               'feature_schema': 'algorithm-last-id-and-codec-parameters-v6', 'extra': extra}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_current_registry(registry):
    if registry.get('policy_registry_version') != p.ACTION_REGISTRY_VERSION:
        raise ValueError('baseline registry differs from current policy registry; migrate paths with '
                         'explicit io_variants and parameters and remeasure complete frames before a new run')
