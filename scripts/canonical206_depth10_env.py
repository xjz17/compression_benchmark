"""Depth-limited recipe/atomic controls over the versioned algorithm registry.

Depth counts expanded algorithm steps, not recipe decisions or subprocesses.
Each step preserves a concrete I/O variant. The recipe catalog is versioned;
its option IDs must not be interpreted using a historical catalog. Old frozen
reference variants retain their original compound codec behavior.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from canonical206_recipe_env import (
    BudgetExceeded, RecipeEnv, RecipeOption, build_catalog as build_recipe_catalog,
)
from canonical206_search_env import DESC, ENTROPY, EXCLUDED, NEXT, REFS, ROOTS, SEQUENCES, SPECS, VARIANT_DESC, VARIANT_PATHS, is_reference, is_standalone

from canonical182_policy_paths import PolicyPath, concrete_path, path_from_record, path_from_names, path_record, stage_signature


def build_catalog(registry: Mapping[str, Any]) -> tuple[RecipeOption, ...]:
    """Preserve this version's recipe IDs and append explicit-variant singletons."""
    original = build_recipe_catalog(registry)
    output = list(original)
    present = {option.ids for option in original}
    for variant, path in sorted(VARIANT_PATHS.items()):
        action = path[0]
        descriptor = VARIANT_DESC[variant]
        if (action in EXCLUDED or path in present
                or descriptor.execution_mode not in {'standalone', 'stagewise'}):
            continue
        output.append(RecipeOption(
            option_id=len(output), name=SPECS[action].name, ids=path,
            accepted_input_kinds=tuple(descriptor.accepted_input_kinds),
            is_single_ref=is_reference(path), labels=(),
        ))
    return tuple(output)


def _histogram(depths) -> dict[str, int]:
    return {str(depth): count for depth, count in sorted(Counter(depths).items())}


class Depth10Env(RecipeEnv):
    """At most ten registered IDs, with packet-aware macro/atomic transitions.

    A successful atomic output is a complete, independently decodable C206 file,
    but can still be an *unfinished registered codec packet*.  Such a packet may
    continue along its registered recipe; it may not randomly start another
    codec.  New byte segments are enabled only at registered completion points.
    Any number of completed segments is allowed within the actual-ID depth cap.

    The original full-column execution, inverse verification, timeout rejection,
    cache, incumbent protection, and path-local reward are inherited unchanged.
    Fixed-backend automatic completion is deliberately disabled; a runner may
    perform and label a separate, identically budgeted public probe phase.
    """

    def __init__(
        self,
        column: Mapping[str, Any],
        options: Sequence[RecipeOption],
        baseline_fullbytes: int,
        max_atomic_depth: int = 10,
        *,
        sample_rows: int = 0,
        max_unique_edges: int | None = None,
        deadline: float | None = None,
        auto_complete_option: int | None = None,
    ) -> None:
        if type(max_atomic_depth) is not int or not 1 <= max_atomic_depth <= 10:
            raise ValueError('max_atomic_depth must be an integer from 1 to 10')
        if auto_complete_option is not None:
            raise ValueError('Depth10Env disables automatic fixed-backend completion')
        # The parent validates its old 1..3 option horizon.  It is used only for
        # construction; all depth-sensitive fields are replaced before any
        # path feature or codec action is requested.
        super().__init__(
            column, options, baseline_fullbytes, max_steps=3,
            sample_rows=sample_rows, max_unique_edges=max_unique_edges,
            deadline=deadline, auto_complete_option=None,
            complete_on_transition=False,
        )
        self.max_atomic_depth = max_atomic_depth
        self.max_steps = max_atomic_depth
        self.base.max_steps = max_atomic_depth
        self._features_cache.clear()
        self.depth_guard_rejections = 0
        self.grammar_guard_rejections = 0
        self.seeded_known_ids: set[tuple[int, ...]] = set()
        self.known_chain_sources: dict[tuple[int, ...], list[dict[str, Any]]] = {}

        # Whole options are trusted registered recipes.  A single ID is a
        # closure only if it was independently executable in the old grammar,
        # not merely because it was appended to this new singleton catalog.
        self._complete_recipes = frozenset(
            option.ids for option in self.options
            if (len(option.ids) > 1 or option.labels
                or option.ids in SEQUENCES
                or option.ids[0] in ENTROPY
                or is_standalone(option.ids[:1]))
        )
        self._proper_prefixes = frozenset(
            recipe[:length] for recipe in self._complete_recipes
            for length in range(1, len(recipe))
        )
        self._root_ids = frozenset(
            (set(ROOTS) | set(ENTROPY)
             | {recipe[0] for recipe in self._complete_recipes}) - set(EXCLUDED)
        )
        # Fixed frozen recipes are admitted even if their full mapping was not
        # present in the old SEQUENCES set.  Their own registered edges are not
        # a license to splice unrelated packet formats.
        self._registered_next = {action: set(targets) for action, targets in NEXT.items()}
        for recipe in self._complete_recipes:
            for first, second in zip(recipe, recipe[1:]):
                self._registered_next.setdefault(first, set()).add(second)
        self._grammar_cache: dict[tuple[int, ...], frozenset[tuple[int, ...]]] = {
            (): frozenset({()}),
        }

    def _can_start_segment(self, action: int, position: int, chain: tuple[int, ...]) -> bool:
        if action not in self._root_ids:
            return False
        if position == 0:
            return (self.column['kind'] in VARIANT_DESC[chain.io_variants[position]].accepted_input_kinds
                    or 'bytes' in VARIANT_DESC[chain.io_variants[position]].accepted_input_kinds)
        # The original runtime re-encodes the first two reference IDs together
        # from the typed source.  It does not support a third typed reference or
        # arbitrary typed re-interpretation of a serialized codec packet.
        if position == 1 and is_reference(chain[:2]):
            return True
        return 'bytes' in VARIANT_DESC[chain.io_variants[position]].accepted_input_kinds

    def _grammar_states(self, ids: Sequence[int]) -> frozenset[tuple[int, ...]]:
        """Return viable pending recipe prefixes; () denotes a closed boundary.

        Keeping all viable parses handles a stage sequence which is both a
        registered complete recipe and the prefix of a longer registered one.
        This is packet-context matching, not just a union of last-ID neighbors.
        """
        chain = concrete_path(ids)
        if chain in self._grammar_cache:
            return self._grammar_cache[chain]
        if not chain or any(action not in SPECS for action in chain):
            return frozenset()
        parent = self._grammar_states(chain[:-1])
        action = chain[-1]
        states: set[tuple[int, ...]] = set()
        for pending in parent:
            if pending:
                if action not in self._registered_next.get(pending[-1], ()):
                    continue
                candidate = pending + chain[-1:]
            else:
                if not self._can_start_segment(action, len(chain) - 1, chain):
                    continue
                candidate = chain[-1:]
            if any(stage_signature(candidate) == stage_signature(r) for r in self._complete_recipes):
                states.add(())
            if any(stage_signature(candidate) == stage_signature(r) for r in self._proper_prefixes):
                states.add(candidate)
        result = frozenset(states)
        self._grammar_cache[chain] = result
        return result

    def is_closed(self, prefix: tuple[int, ...]) -> bool:
        """Whether this successful option path can open a new byte segment."""
        if prefix not in self.nodes:
            raise KeyError('closure requires a successful option prefix')
        return () in self._grammar_states(self.expanded(prefix))

    def remaining_depth(self, prefix: tuple[int, ...]) -> int:
        return self.max_atomic_depth - len(self.expanded(prefix))

    def grammar_state(self, prefix: tuple[int, ...]) -> dict[str, Any]:
        if prefix not in self.nodes:
            raise KeyError('grammar state requires a successful option prefix')
        expanded = self.expanded(prefix)
        states = self._grammar_states(expanded)
        return {
            'expanded_ids': list(expanded), 'expanded_io_variants': list(expanded.io_variants), 'expanded_parameters': list(expanded.parameters),
            'registered_id_depth': len(expanded),
            'remaining_depth': self.max_atomic_depth - len(expanded),
            'closed_boundary': () in states,
            'pending_registered_prefixes': [list(s) for s in sorted(states) if s],
        }

    def option_path_for_ids(self, ids: Sequence[int]) -> tuple[int, ...]:
        """Represent an existing legal chain without inventing a new codec.

        Use an existing whole option when the complete chain matches one;
        otherwise use appended singleton options.  This is a pure grammar/path
        conversion: it does not execute or materialize intermediate candidates.
        """
        chain = concrete_path(ids)
        if (len(chain) > self.max_atomic_depth or len(set(chain)) != len(chain)
                or EXCLUDED.intersection(chain) or not self._grammar_states(chain)):
            raise ValueError('known chain violates actual-depth, no-repeat or packet grammar')
        if not chain:
            return ()
        if stage_signature(chain) in self.option_by_stages:
            return super().option_path_for_ids(chain)
        try:
            path = PolicyPath()
            for k in range(len(chain)):
                path += super().option_path_for_ids(chain[k:k+1])
            return path
        except ValueError as exc:
            raise ValueError(f'known chain contains a nonselectable singleton: {exc}') from exc

    def seed_known_chains(
        self, recipes: Sequence[Mapping[str, Any]], *, verify_baseline: bool = True,
    ) -> list[dict[str, Any]]:
        """Verify and seed complete archived chains from any previously run method.

        Each mapping must contain ``ids`` or ``selected_ids`` and the prior
        complete ``frame_sha256``.  ``compressed_bytes`` is checked if supplied;
        ``source``/``source_path``, ``method`` and ``seed`` preserve lineage.
        The caller sets baseline_fullbytes to the best supplied known result,
        while retaining the original fixed-algorithm comparison separately.

        Only complete archive endpoints are promoted.  Atomic encoding cache
        entries are not implicitly promoted as newly discovered intermediate
        files.  Therefore a seeded singleton option path can have unmaterialized
        ancestors; path_reward uses only already-verified nodes on that path.
        The explicit archive-seed provenance never counts as a new search win.
        """
        if not recipes:
            raise ValueError('at least one archived chain is required')
        if self.records or self.seeded_known_ids or self.search_budget_origin:
            raise ValueError('seed known chains once, before all other probes')
        pending = []
        self._seeding = True
        try:
            for recipe in recipes:
                values = recipe.get('ids', recipe.get('selected_ids'))
                if values is None:
                    raise ValueError('archive seed needs ids or selected_ids')
                chain = path_from_record(recipe)
                option_path = self.option_path_for_ids(chain)
                expected_sha = recipe.get('frame_sha256')
                if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                        or any(c not in '0123456789abcdefABCDEF' for c in expected_sha)):
                    raise ValueError('every archive seed needs its original complete frame_sha256')
                row = self.execute_ids(chain, provenance='baseline')
                row['seed_kind'] = 'known_archive'
                row['archive_source'] = str(recipe.get('source', recipe.get('source_path', 'known_archive')))
                row['archive_method'] = recipe.get('method')
                row['archive_seed'] = recipe.get('seed')
                row['archive_expected_frame_sha256'] = expected_sha.lower()
                row['option_path'] = list(option_path)
                if row['status'] != 'ok':
                    raise ValueError(f"archive complete chain failed replay: {chain}: {row.get('error', '')}")
                if row['frame_sha256'] != expected_sha.lower():
                    raise AssertionError(f'archive full frame SHA256 differs: {chain}')
                if ('compressed_bytes' in recipe
                        and int(recipe['compressed_bytes']) != row['compressed_bytes']):
                    raise AssertionError(f'archive full frame byte count differs: {chain}')
                pending.append((chain, option_path, row))
            sizes = [int(row['compressed_bytes']) for _, _, row in pending]
            if verify_baseline and min(sizes) != self.baseline_fullbytes:
                raise AssertionError(
                    f'known archive best differs: expected {self.baseline_fullbytes}, got {min(sizes)}')
            # Do not promote any archive endpoint before every required replay
            # and expected-hash check has passed.
            for chain, option_path, row in pending:
                self.nodes[option_path] = chain
                self.seeded_known_ids.add(chain)
                if chain in self.fixed_recipe_ids:
                    self.seeded_fixed_ids.add(chain)
                self.known_chain_sources.setdefault(chain, []).append({
                    'source': row['archive_source'], 'method': row['archive_method'],
                    'seed': row['archive_seed'], 'frame_sha256': row['frame_sha256'],
                    'compressed_bytes': row['compressed_bytes'],
                })
                size = int(row['compressed_bytes'])
                if ((size, chain, option_path)
                        < (self.best_bytes, self.expanded(self.best_prefix), self.best_prefix)):
                    self.best_prefix, self.best_bytes = option_path, size
                    self.best_provenance = 'baseline'
        finally:
            self._seeding = False
            self.search_budget_origin = len(self.base.records)
        return [row for _, _, row in pending]

    def legal(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        if prefix not in self.nodes:
            raise KeyError('legality requires a successful option prefix')
        expanded = self.expanded(prefix)
        remaining = self.max_atomic_depth - len(expanded)
        if remaining <= 0 or len(prefix) >= self.max_steps:
            return (self.stop,)
        used = set(expanded)
        choices = []
        for option in self.options:
            if (len(option.ids) > remaining or used.intersection(option.ids)
                    or EXCLUDED.intersection(option.ids)
                    or len(set(option.ids)) != len(option.ids)):
                continue
            if self._grammar_states(expanded + option.ids):
                choices.append(option.option_id)
        return tuple(choices + [self.stop])

    def execute_ids(self, ids: Sequence[int], *, provenance: str = 'search') -> dict[str, Any]:
        """Guard direct execution too, including seeds and public probe calls."""
        chain = concrete_path(ids)
        if len(chain) > self.max_atomic_depth:
            self.depth_guard_rejections += 1
            raise ValueError(f'expanded registered-ID depth {len(chain)} exceeds {self.max_atomic_depth}')
        if not self._grammar_states(chain):
            self.grammar_guard_rejections += 1
            raise ValueError(f'chain violates registered packet/segment grammar: {chain}')
        return super().execute_ids(chain, provenance=provenance)

    def transition(
        self, prefix: tuple[int, ...], action: int, *, provenance: str = 'search', parameters=None,
    ) -> tuple[int, ...] | None:
        previous_new = self.best_new_prefix
        result = super().transition(prefix, action, provenance=provenance, parameters=parameters)
        if (result is not None and self.expanded(result) in self.seeded_known_ids
                and self.best_new_prefix == result):
            # Replaying an archive chain under a search-labelled proposal does
            # not turn an inherited result into a fresh algorithmic discovery.
            self.best_new_prefix = previous_new
        return result

    def path_reward(self, prefix: tuple[int, ...]) -> float:
        if prefix not in self.nodes:
            raise KeyError('path reward requires a complete successful option prefix')
        ancestors = [prefix[:end] for end in range(1, len(prefix) + 1)] or [()]
        sizes = [self.base.nodes[self.nodes[ancestor]].frame_bytes
                 for ancestor in ancestors if ancestor in self.nodes]
        return math.log(self.baseline_fullbytes / min(sizes))

    def reward(self, prefix: tuple[int, ...]) -> float:
        # No hidden suffix execution: all probes must be explicit and costed by
        # the runner.  A failed later probe still uses the inherited path-local
        # best-file reward and the separately declared invalid-action penalty.
        return self.path_reward(prefix)

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        attempted = [path_from_record(row) for row in self.records]
        successful = [path_from_record(row) for row in self.records if row['status'] == 'ok']
        unique_attempted = set(attempted)
        verified = set(self.completed)
        atomic_success = [tuple(row['prefix']) for row in self.base.records if row['status'] == 'ok']
        by_provenance = {}
        for provenance in sorted({str(row['provenance']) for row in self.records}):
            rows = [row for row in self.records if row['provenance'] == provenance]
            by_provenance[provenance] = {
                'attempt': _histogram(len(row['ids']) for row in rows),
                'ok': _histogram(len(row['ids']) for row in rows if row['status'] == 'ok'),
                'unique': _histogram(len(chain) for chain in {path_from_record(row) for row in rows}),
            }
        evidence = {}
        for depth in (9, 10):
            tried = sorted(chain for chain in unique_attempted if len(chain) == depth)
            passed = sorted(chain for chain in verified if len(chain) == depth)
            evidence[str(depth)] = {
                'unique_attempted': len(tried), 'unique_verified_complete': len(passed),
                'attempted_examples': [list(chain) for chain in tried[:5]],
                'verified_examples': [list(chain) for chain in passed[:5]],
                'attempted_parameters': [list(chain.parameters) for chain in tried[:5]],
                'verified_parameters': [list(chain.parameters) for chain in passed[:5]],
                'attempted_io_variants': [list(chain.io_variants) for chain in tried[:5]],
                'verified_io_variants': [list(chain.io_variants) for chain in passed[:5]],
                'atomic_stage_reached': any(len(chain) == depth for chain in atomic_success),
            }
        result.update({
            'depth_definition': 'number of expanded current algorithm policy IDs with explicit I/O variants, not recipe decisions, bare primitives, UAR1 layers or subprocess calls',
            'reference_units': 'frozen canonical206 reference variants retain their compound implementations including packing and zlib6',
            'max_atomic_depth': self.max_atomic_depth,
            'max_decision_steps': self.max_steps,
            'automatic_backend_completion': False,
            'segment_limit': 'only expanded-ID depth cap and no-repeat constraint; no two-segment cap',
            'candidate_depth_histogram_attempt': _histogram(map(len, attempted)),
            'candidate_depth_histogram_ok': _histogram(map(len, successful)),
            'candidate_depth_histogram_unique': _histogram(map(len, unique_attempted)),
            'candidate_depth_histogram_verified_unique': _histogram(map(len, verified)),
            'candidate_depth_histograms_by_provenance': by_provenance,
            'atomic_attempt_depth_histogram': _histogram(len(row['prefix']) for row in self.base.records),
            'max_candidate_depth_attempted': max(map(len, attempted), default=0),
            'max_atomic_depth_reached': max(map(len, atomic_success), default=0),
            'max_verified_candidate_depth': max(map(len, verified), default=0),
            'depth_guard_rejections': self.depth_guard_rejections,
            'grammar_guard_rejections': self.grammar_guard_rejections,
            'depth_9_10_evidence': evidence,
            'seeded_known_chains': [
                {**path_record(chain), 'sources': sources}
                for chain, sources in sorted(self.known_chain_sources.items())
            ],
            'best_is_seeded_known_chain': self.expanded(self.best_prefix) in self.seeded_known_ids,
            'best_known_archive_source': self.known_chain_sources.get(self.expanded(self.best_prefix), []),
        })
        return result


__all__ = ['BudgetExceeded', 'Depth10Env', 'RecipeOption', 'build_catalog']
