"""Whole-recipe controls over the current versioned algorithm runtime.

An option is search control, not a new codec: executing it expands to the exact
registered policy IDs and retains all C206/UAR1 metadata.  Only *completed*
options can become incumbents.  Unlike the original seven-action grammar,
three option decisions may execute more than seven atomic actions.

The incumbent can be seeded from a previously measured fixed recipe.  That
fallback guarantees non-regression only; a strict win requires a smaller new
complete frame, and the provenance fields keep these cases separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from canonical206_search_env import (
    DESC, ENTROPY, EXCLUDED, REFS, SEQUENCES, SPECS, Env, decode_frame,
    make_frame, VARIANT_DESC, REF_VARIANTS, is_reference, blosc_prefix_width,
)
import primitive_operator_action_runtime as p
import native_lz_parameters as lz
import current_action_parameters as action_parameters
import block_state_parameters as block_parameters
from canonical182_policy_paths import (
    PolicyPath, concrete_path, path_from_names, path_from_record, path_record,
    catalog_fingerprint, require_current_registry, stage_signature,
)


class BudgetExceeded(RuntimeError):
    """A runner-owned search budget ended before another atomic execution."""


@dataclass(frozen=True)
class RecipeOption:
    option_id: int
    name: str
    ids: tuple[int, ...]
    accepted_input_kinds: tuple[str, ...]
    is_single_ref: bool
    labels: tuple[str, ...] = ()
    io_variants: tuple[str, ...] = ()
    parameters: tuple[dict, ...] = ()
    fixed_parameter_profiles: tuple[tuple[dict, ...], ...] = ()

    def __post_init__(self):
        path = concrete_path(self.ids, self.io_variants or None, self.parameters or None)
        object.__setattr__(self, 'ids', path)
        object.__setattr__(self, 'io_variants', path.io_variants)
        object.__setattr__(self, 'parameters', path.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {'option_id': self.option_id, 'name': self.name, 'ids': list(self.ids),
                'io_variants': list(self.io_variants), 'parameters': list(self.parameters), 'accepted_input_kinds': self.accepted_input_kinds,
                'is_single_ref': self.is_single_ref, 'labels': self.labels,
                'fixed_parameter_profiles': self.fixed_parameter_profiles}


def build_catalog(registry: Mapping[str, Any]) -> tuple[RecipeOption, ...]:
    """Deduplicate registered full recipes, stage sequences and single actions.

    ``registry`` is the frozen action_registry.json, not a newly invented codec
    mapping.  Labels preserve every fixed-method alias; IDs retain their exact
    implementation identity.  Camel and internally repeated-ID recipes are
    excluded consistently with the requested grammar.
    """
    require_current_registry(registry)
    given = [(r['policy_action_id'], r['name']) for r in registry['policy_actions']]
    expected = [(i, SPECS[i].name) for i in sorted(SPECS)]
    if given != expected:
        raise ValueError('frozen policy IDs/names differ from loaded algorithm runtime')
    version = registry.get('policy_registry_version')
    if version is not None and version != p.ACTION_REGISTRY_VERSION:
        raise ValueError('frozen policy registry version differs from loaded runtime')
    by_name = {name: i for i, name in expected}
    recipes: dict[tuple[int, ...], set[str]] = {}
    fixed_profiles = {}

    def add(ids: Iterable[int], label: str | None = None) -> None:
        chain = concrete_path(ids)
        if not chain or EXCLUDED.intersection(chain) or len(set(chain)) != len(chain):
            return
        if any(i not in SPECS for i in chain):
            raise ValueError(f'unknown canonical policy ID in recipe {chain}')
        original = chain
        # Blosc's width is constrained by this option's fixed SHF1 producer;
        # resetting it to a global default can make an otherwise valid recipe
        # impossible to invert. Other independent parameter profiles normalize.
        chain = concrete_path(chain, parameters=tuple(
            q if SPECS[i].name in (block_parameters.BLOSC1, block_parameters.BLOSC2) else {}
            for i,q in zip(chain,chain.parameters)))
        labels = recipes.setdefault(chain, set())
        if label is not None:
            labels.add(label)
            fixed_profiles.setdefault(chain, set()).add(original)

    for sequence in SEQUENCES:
        add(sequence)
    for variant, descriptor in VARIANT_DESC.items():
        path = path_from_names((variant,))
        if descriptor.execution_mode == 'standalone' or path[0] in ENTROPY:
            add(path)
    mappings = registry.get('benchmark_86_io_variant_pipelines',
                            registry['benchmark_86_canonical_action_pipelines'])
    for label, mapping in mappings.items():
        variants = mapping.items() if isinstance(mapping, dict) else [('shared', mapping)]
        for _variant, names in variants:
            params = registry.get('benchmark_86_action_parameters', {}).get(label)
            params = params.get(_variant) if isinstance(params, dict) else params
            add(path_from_names(names, params), label)
    output = []
    for option_id, (ids, labels) in enumerate(sorted(recipes.items())):
        names = tuple(SPECS[i].name for i in ids)
        output.append(RecipeOption(
            option_id=option_id,
            name=' / '.join(sorted(labels)) if labels else ' -> '.join(names),
            ids=ids,
            accepted_input_kinds=tuple(VARIANT_DESC[ids.io_variants[0]].accepted_input_kinds),
            is_single_ref=len(ids) == 1 and is_reference(ids),
            labels=tuple(sorted(labels)),
            fixed_parameter_profiles=tuple(q.parameters for q in sorted(fixed_profiles.get(ids, ()))),
        ))
    return tuple(output)


class RecipeEnv:
    """Full-column, whole-option environment for GRPO or matched controls.

    Prefixes contain *option IDs*, not atomic policy IDs.  ``expanded(prefix)``
    provides the latter for cold encoding with the original runtime.
    ``deadline`` is an absolute ``time.perf_counter()`` value.  The unique-edge
    budget starts after ``seed_fixed_recipes``; cached atomic edges cost no new
    edge.  BudgetExceeded is deliberately never converted to codec rejection.
    A runner should also bound each underlying subprocess independently.
    """

    def __init__(
        self,
        column: Mapping[str, Any],
        options: Sequence[RecipeOption],
        baseline_fullbytes: int,
        max_steps: int = 3,
        *,
        sample_rows: int = 0,
        max_unique_edges: int | None = None,
        deadline: float | None = None,
        auto_complete_option: int | None = None,
        complete_on_transition: bool = True,
    ) -> None:
        if sample_rows != 0:
            raise ValueError('RecipeEnv rewards require full columns; sample_rows must be 0')
        if not 1 <= max_steps <= 3 or baseline_fullbytes < 1:
            raise ValueError('positive baseline bytes and 1..3 option steps are required')
        if max_unique_edges is not None and max_unique_edges < 0:
            raise ValueError('max_unique_edges must be non-negative or None')
        self.options = tuple(options)
        if not self.options or tuple(o.option_id for o in self.options) != tuple(range(len(self.options))):
            raise ValueError('option IDs must be dense, ordered and non-empty')
        self.option_by_ids = {o.ids: o.option_id for o in self.options}
        self.option_by_stages = {stage_signature(o.ids): o.option_id for o in self.options}
        if len(self.option_by_stages) != len(self.options):
            raise ValueError('option recipes must be deduplicated')
        self.column = dict(column)
        self.baseline_fullbytes = int(baseline_fullbytes)
        self.max_steps = int(max_steps)
        self.stop = len(self.options)
        self.action_count = self.stop + 1
        self.stop_id = self.stop
        self.policy_registry_version = p.ACTION_REGISTRY_VERSION
        self.catalog_fingerprint = catalog_fingerprint([o.to_dict() for o in self.options])
        self.max_unique_edges = max_unique_edges
        self.deadline = deadline
        if auto_complete_option is not None:
            if (type(auto_complete_option) is not int
                    or not 0 <= auto_complete_option < len(self.options)):
                raise ValueError('auto_complete_option must be a catalog option ID or None')
            if 'bytes' not in self.options[auto_complete_option].accepted_input_kinds:
                raise ValueError('fixed backend completion must accept byte input')
        self.auto_complete_option = auto_complete_option
        self.complete_on_transition = bool(complete_on_transition)
        self._completion_cache: dict[tuple[int, ...], tuple[int, ...] | None] = {}
        self.base = Env(self.column, sample_rows=0,
                        max_steps=max_steps * max(len(o.ids) for o in self.options))
        self.nodes: dict[tuple[int, ...], tuple[int, ...]] = {(): ()}
        self.invalid: dict[tuple[int, ...], str] = {}
        self.records: list[dict[str, Any]] = []
        self.completed: dict[tuple[int, ...], dict[str, Any]] = {}
        self.best_prefix: tuple[int, ...] = ()
        self.best_bytes = self.base.nodes[()].frame_bytes
        self.best_provenance = 'raw'
        self.best_new_prefix: tuple[int, ...] | None = None
        self.fixed_recipe_ids = {concrete_path(o.ids, parameters=q) for o in self.options if o.labels
                                 for q in (o.fixed_parameter_profiles or (o.parameters,))}
        self.seeded_fixed_ids: set[tuple[int, ...]] = set()
        self.search_budget_origin = 0
        self._seeding = False
        self._features_cache: dict[tuple[int, ...], np.ndarray] = {}
        self.feature_dim = self.base.feature_dim + len(self.options) + 3

    def _check_budget(self, child: tuple[int, ...]) -> None:
        if self._seeding:
            return
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise BudgetExceeded('recipe search wall-clock deadline reached')
        new_edge = child not in self.base.nodes and child not in self.base.invalid
        if (new_edge and self.max_unique_edges is not None
                and len(self.base.records) - self.search_budget_origin >= self.max_unique_edges):
            raise BudgetExceeded('recipe search unique-atomic-edge budget reached')

    def option_path_for_ids(self, ids):
        """Resolve a recipe and retain stage parameters in the scoped option map."""
        chain = concrete_path(ids)
        try:
            option = self.option_by_stages[stage_signature(chain)]
        except KeyError as exc:
            raise ValueError('chain is not a registered whole recipe option') from exc
        parameters = {key:value for i,q in zip(chain,chain.parameters)
                      for key,value in action_parameters.for_option(SPECS[i].name,q).items()}
        return PolicyPath((option,), parameters=(parameters,))

    def expanded(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        path = PolicyPath()
        option_path = PolicyPath(prefix)
        for option_id, override in zip(option_path, option_path.parameters):
            selected = self.options[option_id].ids
            if override:
                override = action_parameters.normalize_option((SPECS[i].name for i in selected), override)
                selected = concrete_path(selected, parameters=tuple(action_parameters.for_stage(SPECS[i].name, override) if SPECS[i].name in action_parameters.BY_NAME else q for i, q in zip(selected, selected.parameters)))
            path = path + selected
        return path

    def frame(self, prefix: tuple[int, ...]) -> bytes:
        if prefix not in self.nodes:
            raise KeyError('only complete successful option prefixes have frames')
        return make_frame(self.column, self.base.nodes[self.nodes[prefix]])

    def features(self, prefix: tuple[int, ...]) -> np.ndarray:
        if prefix not in self.nodes:
            raise KeyError('features require a completed option prefix')
        if prefix not in self._features_cache:
            atomic = self.nodes[prefix]
            x = np.zeros(self.feature_dim, dtype=np.float64)
            x[:self.base.feature_dim] = self.base.features(atomic)
            if prefix:
                x[self.base.feature_dim + prefix[-1]] = 1.0
            size = self.base.nodes[atomic].frame_bytes
            x[-3:] = [len(prefix) / self.max_steps,
                      math.log(self.baseline_fullbytes / size) / 10,
                      len(atomic) / max(1, self.base.max_steps)]
            self._features_cache[prefix] = x
        return self._features_cache[prefix]

    def legal(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        if prefix not in self.nodes:
            raise KeyError('legality requires a completed option prefix')
        if len(prefix) >= self.max_steps:
            return (self.stop,)
        used = set(self.expanded(prefix))
        second_reference = (len(prefix) == 1 and self.options[prefix[0]].is_single_ref)
        choices = []
        for option in self.options:
            if used.intersection(option.ids) or EXCLUDED.intersection(option.ids):
                continue
            if not prefix:
                allowed = (self.column['kind'] in option.accepted_input_kinds
                           or 'bytes' in option.accepted_input_kinds)
            else:
                allowed = ('bytes' in option.accepted_input_kinds
                           or (second_reference and option.is_single_ref))
            if allowed:
                choices.append(option.option_id)
        return tuple(choices + [self.stop])

    def execute_ids(
        self, ids: Sequence[int], *, provenance: str = 'search'
    ) -> dict[str, Any]:
        """Execute a complete expanded chain without the old grammar filter.

        A failed intermediate stage is recorded as rejection of the entire
        requested chain.  Successful intermediate states remain cache entries,
        never returned as successful substitutes for a failed final option.
        This low-level method does not insert an option-prefix node; seed and
        transition provide that higher-level bookkeeping.
        """
        chain = concrete_path(ids)
        if any(i not in SPECS for i in chain) or EXCLUDED.intersection(chain):
            raise ValueError('unknown or excluded ID in expanded recipe')
        if len(set(chain)) != len(chain):
            raise ValueError('repeated atomic IDs are outside this recipe grammar')
        started = time.perf_counter()
        before_count = len(self.base.records)
        prefix: tuple[int, ...] = ()
        cached = chain in self.completed
        try:
            for action, variant, params in zip(chain, chain.io_variants, chain.parameters):
                child = PolicyPath(prefix) + concrete_path((action,), (variant,), (params,))
                self._check_budget(child)
                result = self.base.transition(prefix, action, io_variant=variant, parameters=params)
                if result is None:
                    row = {**path_record(chain), 'status': 'rejected',
                           'failed_prefix': list(child), 'error': self.base.invalid[child],
                           'provenance': provenance, 'wall_sec': time.perf_counter() - started,
                           'new_atomic_edges': len(self.base.records) - before_count}
                    self.records.append(row)
                    return row
                prefix = result
        except BudgetExceeded as exc:
            self.records.append({
                **path_record(chain), 'status': 'budget_exhausted',
                'completed_atomic_prefix': list(prefix), 'error': str(exc),
                'provenance': provenance, 'wall_sec': time.perf_counter() - started,
                'new_atomic_edges': len(self.base.records) - before_count,
            })
            raise
        frame = make_frame(self.column, self.base.nodes[prefix])
        try:
            if not cached and decode_frame(frame) != self.column['raw']:
                raise AssertionError('complete recipe C206 frame is not bit-exact')
        except BudgetExceeded:
            raise
        except Exception as exc:
            row = {
                **path_record(chain), 'status': 'rejected',
                'failed_phase': 'complete_decode',
                'error': f'{type(exc).__name__}: {exc}'[:1000],
                'provenance': provenance, 'wall_sec': time.perf_counter() - started,
                'new_atomic_edges': len(self.base.records) - before_count,
            }
            self.records.append(row)
            return row
        row = {**path_record(chain), 'status': 'ok', 'compressed_bytes': len(frame),
               'compression_ratio': len(self.column['raw']) / len(frame),
               'frame_sha256': hashlib.sha256(frame).hexdigest(), 'lossless': True,
               'provenance': provenance, 'cached_complete': cached,
               'is_registered_fixed_recipe': chain in self.fixed_recipe_ids,
               'first_provenance': self.completed[chain]['first_provenance'] if cached else provenance,
               'wall_sec': time.perf_counter() - started,
               'new_atomic_edges': len(self.base.records) - before_count}
        self.completed.setdefault(chain, dict(row))
        self.records.append(row)
        return row

    def parameter_choices(self, prefix, action):
        if not 0 <= action < len(self.options):
            return ()
        path = self.options[action].ids
        choices = action_parameters.choices_for_names(SPECS[i].name for i in path)
        for j,i in enumerate(path):
            name = SPECS[i].name
            if name in (block_parameters.BLOSC1, block_parameters.BLOSC2):
                width = blosc_prefix_width(path[:j])
                key = action_parameters.BY_NAME[name].option_key
                choices = tuple(q for q in choices if q[key] == width)
        return choices

    def transition(
        self, prefix: tuple[int, ...], action: int, *, provenance: str = 'search', parameters=None
    ) -> tuple[int, ...] | None:
        if action == self.stop:
            return prefix
        if action not in self.legal(prefix):
            raise ValueError('illegal whole-recipe option')
        if parameters:
            if not self.parameter_choices(prefix, action):
                raise ValueError('this option has no input parameters')
            parameters = action_parameters.normalize_option([SPECS[i].name for i in self.options[action].ids], parameters)
        child = PolicyPath(prefix) + PolicyPath((int(action),), parameters=(parameters or {},))
        if child in self.invalid:
            return None
        atomic = self.expanded(child)
        result = self.execute_ids(atomic, provenance=provenance)
        if result['status'] != 'ok':
            self.invalid[child] = result['error']
            return None
        self.nodes[child] = atomic
        size = int(result['compressed_bytes'])
        if (size, atomic, child) < (self.best_bytes, self.expanded(self.best_prefix), self.best_prefix):
            self.best_prefix, self.best_bytes = child, size
            self.best_provenance = str(result['first_provenance'])
        if provenance != 'baseline' and atomic not in self.fixed_recipe_ids:
            old = self.best_new_prefix
            if old is None or (size, atomic) < (self.base.nodes[self.nodes[old]].frame_bytes, self.expanded(old)):
                self.best_new_prefix = child
        if self.complete_on_transition and provenance == 'search':
            # A discovered short file remains valuable even if the policy later
            # explores an expanding or invalid continuation.  Completion itself
            # uses a distinct provenance and therefore cannot recurse here.
            self.reward(child)
        return child

    def reward(self, prefix: tuple[int, ...]) -> float:
        """Try one current-prefix completion, then score this trajectory path.

        This deterministic completion is candidate/environment design, not a
        learned suffix oracle: exactly one runner-selected fixed backend can be
        tested.  It must be given identically to GRPO and its matched control.
        Empty prefixes are not completed.  All newly executed atomic stages
        retain ordinary search-budget checks and full-frame inverse validation.
        """
        if prefix not in self.nodes:
            raise KeyError('reward requires a complete successful option prefix')
        option = self.auto_complete_option
        if (prefix and len(prefix) < self.max_steps and option is not None
                and option in self.legal(prefix)):
            if prefix not in self._completion_cache:
                # BudgetExceeded must propagate; an interrupted completion is
                # not cached as an invalid codec or a successful short prefix.
                self._completion_cache[prefix] = self.transition(
                    prefix, option, provenance='fixed_backend_completion',
                )
        return self.path_reward(prefix)

    def path_reward(self, prefix: tuple[int, ...]) -> float:
        """Best full file on this option path and its cached fixed completions.

        This is read-only: it does not execute an ancestor completion, inspect
        another trajectory's unrelated nodes, or use the global incumbent.
        Empty paths score the complete RAW frame.  All nonempty path states
        must already have completed successfully.
        """
        if prefix not in self.nodes:
            raise KeyError('path reward requires a complete successful option prefix')
        path = [prefix[:end] for end in range(1, len(prefix) + 1)] or [()]
        sizes = []
        for ancestor in path:
            if ancestor not in self.nodes:
                raise KeyError('trajectory contains an uncompleted option ancestor')
            sizes.append(self.base.nodes[self.nodes[ancestor]].frame_bytes)
            completed = self._completion_cache.get(ancestor)
            if completed is not None:
                sizes.append(self.base.nodes[self.nodes[completed]].frame_bytes)
        return math.log(self.baseline_fullbytes / min(sizes))

    def reward_on_invalid(self, prefix: tuple[int, ...]) -> float:
        """Keep valid path discoveries and charge a small failed-probe penalty.

        The 0.02 log-reward penalty affects policy learning only.  Final file
        selection remains minimum verified C206/UAR1 bytes, without this term.
        """
        return self.path_reward(prefix) - 0.02

    def seed_fixed_recipes(
        self, recipes: Sequence[Mapping[str, Any] | Sequence[int]], *,
        verify_baseline: bool = True,
    ) -> list[dict[str, Any]]:
        """Seed the supplied fixed winner(s), then start the search-edge budget.

        Passing only the prior fixed winner avoids repeating all prior baseline
        measurements.  Passing all eligible recipes is also supported.  The
        seed cost is measured separately and remains visible in the summary.
        Each successful seed is inserted as a root option node for warm-start
        supervised policy initialization; merely populating atomic cache would
        not be sufficient.
        """
        if not recipes:
            raise ValueError('at least one fixed recipe is required for seeding')
        if self.search_budget_origin or any(r['provenance'] != 'baseline' for r in self.records):
            raise ValueError('seed fixed recipes once before search')
        rows = []
        self._seeding = True
        try:
            for recipe in recipes:
                values = recipe.get('ids', recipe.get('selected_ids')) if isinstance(recipe, Mapping) else recipe
                if values is None:
                    raise ValueError('seed recipe needs ids or selected_ids')
                ids = (path_from_record(recipe) if isinstance(recipe, Mapping)
                       else concrete_path(values))
                if stage_signature(ids) not in self.option_by_stages or ids not in self.fixed_recipe_ids:
                    raise ValueError('seed must be a registered fixed complete recipe option')
                option_path = self.option_path_for_ids(ids)
                result = self.transition((), option_path[0], parameters=option_path.parameters[0], provenance='baseline')
                rows.append(dict(self.records[-1]))
                if result is not None:
                    self.seeded_fixed_ids.add(ids)
            sizes = [int(r['compressed_bytes']) for r in rows if r['status'] == 'ok']
            if not sizes:
                raise ValueError('all supplied baseline seed recipes were rejected')
            if verify_baseline and min(sizes) != self.baseline_fullbytes:
                raise AssertionError(f'baseline replay differs: expected {self.baseline_fullbytes}, got {min(sizes)}')
        finally:
            self._seeding = False
            self.search_budget_origin = len(self.base.records)
        return rows

    def summary(self) -> dict[str, Any]:
        new = self.best_new_prefix
        new_bytes = self.base.nodes[self.nodes[new]].frame_bytes if new is not None else None
        return {
            'options': len(self.options), 'action_count': self.action_count,
            'max_option_steps': self.max_steps, 'full_column_reward': True,
            'auto_complete_option': self.auto_complete_option,
            'complete_on_transition': self.complete_on_transition,
            'reward_scope': 'path-best complete frame plus cached fixed-backend completions; no global-incumbent reward',
            'invalid_reward_penalty': 0.02,
            'auto_complete_io_variants': (list(self.options[self.auto_complete_option].io_variants)
                                          if self.auto_complete_option is not None else None),
            'auto_complete_ids': (list(self.options[self.auto_complete_option].ids)
                                  if self.auto_complete_option is not None else None),
            'fixed_backend_completion_attempts': sum(
                r['provenance'] == 'fixed_backend_completion' for r in self.records),
            'baseline_fullbytes': self.baseline_fullbytes,
            'seed_atomic_edges': self.search_budget_origin,
            'search_unique_atomic_edges': len(self.base.records) - self.search_budget_origin,
            'seed_wall_sec': sum(r['wall_sec'] for r in self.records if r['provenance'] == 'baseline'),
            'search_execution_wall_sec': sum(r['wall_sec'] for r in self.records if r['provenance'] != 'baseline'),
            'completed_option_prefixes': len(self.nodes) - 1,
            'rejected_option_prefixes': len(self.invalid),
            'best_option_prefix': list(self.best_prefix),
            'best_ids': list(self.expanded(self.best_prefix)),
            'best_io_variants': list(self.expanded(self.best_prefix).io_variants),
            'best_parameters': list(self.expanded(self.best_prefix).parameters),
            'best_option_parameters': list(PolicyPath(self.best_prefix).parameters),
            'policy_registry_version': p.ACTION_REGISTRY_VERSION,
            'catalog_fingerprint': self.catalog_fingerprint, 'best_frame_bytes': self.best_bytes,
            'best_first_provenance': self.best_provenance,
            'best_is_seeded_baseline': self.expanded(self.best_prefix) in self.seeded_fixed_ids,
            'best_is_registered_fixed_recipe': self.expanded(self.best_prefix) in self.fixed_recipe_ids,
            'best_new_option_prefix': list(new) if new is not None else None,
            'best_new_ids': list(self.expanded(new)) if new is not None else None,
            'best_new_io_variants': list(self.expanded(new).io_variants) if new is not None else None,
            'best_new_parameters': list(self.expanded(new).parameters) if new is not None else None,
            'best_new_frame_bytes': new_bytes,
            'strict_new_win': new_bytes is not None and new_bytes < self.baseline_fullbytes,
            'atomic_runtime': self.base.summary(),
        }
