#!/usr/bin/env python3
"""Budgeted GRPO-guided and Beam30 search for compression graphs.

This module is deliberately an orchestration layer.  It never compresses data
itself and it never treats an ordering heuristic as a proof.  A task owns the
native operator transitions, complete framing, cold decode, and physical
measurements.  Search owns only:

* the 64-candidate first-block / 16-candidate continuation budgets;
* exact objective comparison of complete candidates;
* optional certified lower-bound pruning;
* policy or Beam30 ordering; and
* prediction-time accounting.

Every call to ``task.evaluate_terminal`` consumes one complete-candidate
evaluation, including a repeated recipe.  This matches the experiment contract
that online measurements are part of operator-combination prediction cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from weight_conditioned_grpo_v1 import (
    FrozenPolicy,
    GraphCompressionTask,
    GraphContractError,
    TerminalOutcome,
    objective_value,
    validate_weight,
)
from weighted_objective_pruning_v1 import (
    BoundState,
    certified_prune,
    feature_priority,
    objective_guided_ordering_score,
)


SCHEMA = "weight-conditioned-budgeted-search-v1"
FIRST_BLOCK_EVALUATIONS = 64
CONTINUATION_EVALUATIONS = 16
BEAM_WIDTH = 30


class SearchContractError(RuntimeError):
    """A task or policy violated the frozen online-search contract."""


class BoundProvider(Protocol):
    def __call__(self, task: GraphCompressionTask, state: Any) -> BoundState: ...


class ActionOrdering(Protocol):
    def __call__(
        self,
        task: GraphCompressionTask,
        state: Any,
        action: int,
        weight: tuple[float, float, float],
    ) -> float: ...


@dataclass(frozen=True)
class SearchConfig:
    max_evaluations: int
    beam_width: int = BEAM_WIDTH
    max_steps: int | None = None
    keep_objective_ties: bool = True
    use_objective_guided_ordering: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_evaluations, bool) or self.max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer")
        if isinstance(self.beam_width, bool) or self.beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or self.max_steps < 1
        ):
            raise ValueError("max_steps must be positive when supplied")
        if type(self.use_objective_guided_ordering) is not bool:
            raise ValueError("use_objective_guided_ordering must be boolean")

    @classmethod
    def for_block(cls, block_index: int, *, beam_width: int = BEAM_WIDTH) -> "SearchConfig":
        if isinstance(block_index, bool) or block_index < 0:
            raise ValueError("block_index must be a non-negative integer")
        return cls(
            max_evaluations=(FIRST_BLOCK_EVALUATIONS if block_index == 0
                             else CONTINUATION_EVALUATIONS),
            beam_width=beam_width,
        )


@dataclass(frozen=True)
class PredictionTiming:
    feature_time_ns: int = 0
    model_inference_time_ns: int = 0
    action_ordering_time_ns: int = 0
    environment_transition_time_ns: int = 0
    certified_pruning_time_ns: int = 0
    candidate_evaluation_time_ns: int = 0
    total_prediction_time_ns: int = 0


@dataclass(frozen=True)
class CandidateSummary:
    # ``actions`` is always the complete, replayable operator recipe.  The
    # policy/search trace can differ from it for continuation tasks, where one
    # recipe edit is represented by edit-type/position/operator tokens.
    actions: tuple[int, ...]
    objective: float
    compression_ratio: float
    compression_ns_per_byte: float
    decompression_ns_per_byte: float
    complete_bytes: int
    search_actions: tuple[int, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    method: str
    weight: tuple[float, float, float]
    best_actions: tuple[int, ...]
    best_outcome: TerminalOutcome
    best_objective: float
    evaluations: int
    evaluation_budget: int
    certified_pruned_states: int
    budget_unexplored_actions: int
    duplicate_states_skipped: int
    objective_guided_ordering: bool
    timing: PredictionTiming
    evaluated_candidates: tuple[CandidateSummary, ...]

    @property
    def best_recipe_action_ids(self) -> tuple[int, ...]:
        return _outcome_recipe(self.best_outcome)

    def json_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "method": self.method,
            "weight": list(self.weight),
            # ``best_actions`` is retained for v1 reader compatibility.  It is
            # the search/edit-token trace, not necessarily a graph recipe.
            "best_actions": list(self.best_actions),
            "best_search_actions": list(self.best_actions),
            "best_recipe_action_ids": list(self.best_recipe_action_ids),
            "best_objective": self.best_objective,
            "evaluations": self.evaluations,
            "evaluation_budget": self.evaluation_budget,
            "certified_pruned_states": self.certified_pruned_states,
            "budget_unexplored_actions": self.budget_unexplored_actions,
            "duplicate_states_skipped": self.duplicate_states_skipped,
            "objective_guided_ordering": self.objective_guided_ordering,
            "timing": asdict(self.timing),
            "evaluated_candidates": [asdict(row) for row in self.evaluated_candidates],
        }


@dataclass(frozen=True)
class _Node:
    state: Any
    actions: tuple[int, ...]
    rank: float
    depth: int


def _validate_task(task: GraphCompressionTask) -> None:
    required = (
        "task_features", "initial_state", "state_features", "legal_actions",
        "transition", "evaluate_terminal",
    )
    if any(not callable(getattr(task, name, None)) for name in required):
        raise SearchContractError("task does not implement the graph-compression contract")
    if int(task.action_count) < 1 or not 0 <= int(task.stop_action) < int(task.action_count):
        raise SearchContractError("task action contract is invalid")
    if int(task.max_steps) < 1:
        raise SearchContractError("task max_steps must be positive")


def _state_key(node: _Node) -> tuple[Any, ...]:
    recipe = getattr(node.state, "recipe", None)
    if recipe is not None:
        if bool(getattr(node.state, "continuation_edit_mode", False)):
            return (
                "continuation_recipe",
                tuple(int(value) for value in recipe),
                getattr(node.state, "pending_edit_kind", None),
                getattr(node.state, "pending_edit_index", None),
                int(getattr(node.state, "edit_count", 0)),
                tuple(getattr(node.state, "edit_history", ())),
            )
        return ("recipe", *tuple(int(value) for value in recipe))
    return ("actions", *node.actions)


def _outcome_recipe(outcome: TerminalOutcome) -> tuple[int, ...]:
    graph = outcome.operator_graph
    if not isinstance(graph, dict):
        try:
            graph = dict(graph)
        except (TypeError, ValueError) as error:
            raise SearchContractError(
                "terminal operator_graph must expose recipe_action_ids"
            ) from error
    raw = graph.get("recipe_action_ids")
    if (not isinstance(raw, (list, tuple))
            or any(isinstance(action, bool) or not isinstance(action, (int, np.integer))
                   or int(action) < 0 for action in raw)):
        raise SearchContractError("terminal recipe_action_ids is not replayable")
    return tuple(int(action) for action in raw)


def _summary(actions: tuple[int, ...], outcome: TerminalOutcome, objective: float) -> CandidateSummary:
    return CandidateSummary(
        actions=_outcome_recipe(outcome),
        objective=objective,
        compression_ratio=outcome.compression_ratio,
        compression_ns_per_byte=outcome.metrics.compression_ns_per_byte,
        decompression_ns_per_byte=outcome.metrics.decompression_ns_per_byte,
        complete_bytes=len(outcome.compressed_payload),
        search_actions=actions,
    )


def _is_terminal_evaluable(task: GraphCompressionTask, state: Any) -> bool:
    """Whether ``state`` is a complete recipe that may consume an evaluation.

    Continuation edits are multi-token actions.  Their edit-type and position
    prefix states are legal search states, but they are not complete operator
    recipes and therefore must neither consume a candidate evaluation nor be
    passed to ``evaluate_terminal``.
    """
    checker = getattr(task, "is_terminal_evaluable", None)
    if callable(checker):
        result = checker(state)
        if not isinstance(result, (bool, np.bool_)):
            raise SearchContractError("is_terminal_evaluable must return a boolean")
        return bool(result)
    return getattr(state, "pending_edit_kind", None) is None


def _legal_nonstop(
    task: GraphCompressionTask, state: Any, *, require_stop: bool | None = None,
) -> tuple[int, ...]:
    actions = tuple(int(action) for action in task.legal_actions(state))
    if not actions or len(set(actions)) != len(actions):
        raise SearchContractError("task returned an empty or duplicate legal-action menu")
    if any(not 0 <= action < int(task.action_count) for action in actions):
        raise SearchContractError("task returned an out-of-range legal action")
    if require_stop is None:
        require_stop = _is_terminal_evaluable(task, state)
    if require_stop and int(task.stop_action) not in actions:
        raise SearchContractError("STOP must remain legal for every searchable state")
    if not require_stop and int(task.stop_action) in actions:
        raise SearchContractError("STOP cannot be legal for an incomplete edit state")
    return tuple(action for action in actions if action != int(task.stop_action))


def _bound_is_suffix_safe(task: GraphCompressionTask, state: Any) -> bool:
    """Reject a prefix bound while a continuation recipe can still be edited.

    The current typed-graph byte bound includes already replayed graph steps.
    Insert-only suffix expansion cannot remove those bytes, but delete/replace
    continuation actions can.  Until the edit budget is exhausted, skipping the
    bound is the only certified behavior; the branch remains explorable.
    """
    if not bool(getattr(state, "continuation_edit_mode", False)):
        return True
    pending = getattr(state, "pending_edit_kind", None)
    edit_count = getattr(state, "edit_count", None)
    edit_limit = getattr(task, "_max_recipe_edits", None)
    if (not isinstance(edit_count, int) or isinstance(edit_count, bool)
            or not isinstance(edit_limit, int) or isinstance(edit_limit, bool)):
        return False
    return pending is None and edit_count >= edit_limit


def _maybe_certified_prune(
    task: GraphCompressionTask,
    state: Any,
    weight: tuple[float, float, float],
    incumbent: float | None,
    provider: BoundProvider | None,
    keep_ties: bool,
) -> bool:
    if provider is None or incumbent is None:
        return False
    if not _bound_is_suffix_safe(task, state):
        return False
    bound = provider(task, state)
    if not isinstance(bound, BoundState):
        raise SearchContractError("bound_provider must return BoundState")
    return certified_prune(
        bound, weight, incumbent, keep_ties=keep_ties
    ).prune


def _policy_distribution(
    policy: FrozenPolicy,
    task: GraphCompressionTask,
    state: Any,
    static_features: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    feature_started = time.perf_counter_ns()
    dynamic = np.asarray(task.state_features(state), dtype=np.float64)
    if dynamic.shape != (policy.identity.state_feature_dim,) or not np.all(np.isfinite(dynamic)):
        raise SearchContractError("task state features do not match the frozen policy")
    observation = np.concatenate((static_features, dynamic, weight))
    feature_ns = time.perf_counter_ns() - feature_started
    inference_started = time.perf_counter_ns()
    logits = policy.forward(observation[None, :])[0]
    legal = tuple(int(action) for action in task.legal_actions(state))
    mask = np.zeros(policy.identity.action_count, dtype=bool)
    mask[list(legal)] = True
    shifted = np.where(mask, logits, -np.inf)
    shifted -= np.max(shifted)
    probabilities = np.where(mask, np.exp(shifted), 0.0)
    probabilities /= probabilities.sum()
    inference_ns = time.perf_counter_ns() - inference_started
    return probabilities, feature_ns, inference_ns


def _validate_policy(policy: FrozenPolicy, task: GraphCompressionTask) -> np.ndarray:
    if not isinstance(policy, FrozenPolicy):
        raise TypeError("policy search requires a FrozenPolicy")
    if (policy.identity.registry_fingerprint != str(task.registry_fingerprint)
            or policy.identity.action_count != int(task.action_count)
            or policy.identity.stop_action != int(task.stop_action)):
        raise GraphContractError("task action/registry contract does not match the policy")
    static = np.asarray(task.task_features(), dtype=np.float64)
    if static.shape != (policy.identity.task_feature_dim,) or not np.all(np.isfinite(static)):
        raise SearchContractError("task features do not match the frozen policy")
    return static


def policy_candidate_search(
    policy: FrozenPolicy,
    task: GraphCompressionTask,
    weight: Sequence[float],
    *,
    config: SearchConfig,
    bound_provider: BoundProvider | None = None,
) -> SearchResult:
    """Explore complete candidates in frozen-policy probability order.

    The policy only orders legal branches.  Exact terminal measurements choose
    the returned candidate.  Therefore model score is never substituted for
    compression ratio or physical time.
    """
    total_started = time.perf_counter_ns()
    _validate_task(task)
    static_started = time.perf_counter_ns()
    static = _validate_policy(policy, task)
    static_feature_ns = time.perf_counter_ns() - static_started
    weights_array = validate_weight(weight)
    weights = tuple(float(value) for value in weights_array)
    horizon = min(int(task.max_steps), config.max_steps or int(task.max_steps))
    root = _Node(task.initial_state(), (), 0.0, 0)
    frontier = [root]
    seen: set[tuple[Any, ...]] = set()
    evaluated: list[CandidateSummary] = []
    best_outcome: TerminalOutcome | None = None
    best_actions: tuple[int, ...] = ()
    incumbent: float | None = None
    certified_pruned_states = budget_unexplored = duplicates = 0
    feature_ns = static_feature_ns
    inference_ns = ordering_ns = transition_ns = pruning_ns = evaluation_ns = 0

    while frontier and len(evaluated) < config.max_evaluations:
        proposals: list[tuple[float, _Node, int]] = []
        budget_hit = False
        for node_index, node in enumerate(frontier):
            key = _state_key(node)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            pruning_started = time.perf_counter_ns()
            pruned = _maybe_certified_prune(
                task, node.state, weights, incumbent, bound_provider,
                config.keep_objective_ties,
            )
            pruning_ns += time.perf_counter_ns() - pruning_started
            if pruned:
                certified_pruned_states += 1
                continue
            evaluable = _is_terminal_evaluable(task, node.state)
            if evaluable:
                evaluation_started = time.perf_counter_ns()
                outcome = task.evaluate_terminal(node.state)
                objective = objective_value(outcome, weights)
                evaluation_ns += time.perf_counter_ns() - evaluation_started
                evaluated.append(_summary(node.actions, outcome, objective))
                if incumbent is None or objective < incumbent:
                    incumbent = objective
                    best_outcome = outcome
                    best_actions = node.actions
                if len(evaluated) >= config.max_evaluations:
                    budget_unexplored += len(
                        _legal_nonstop(task, node.state, require_stop=True)
                    )
                    for remaining in frontier[node_index + 1:]:
                        budget_unexplored += len(_legal_nonstop(
                            task, remaining.state,
                            require_stop=_is_terminal_evaluable(task, remaining.state),
                        ))
                    budget_hit = True
                    break
            if node.depth >= horizon - 1:
                continue
            probabilities, spent_features, spent_inference = _policy_distribution(
                policy, task, node.state, static, weights_array
            )
            feature_ns += spent_features
            inference_ns += spent_inference
            legal = _legal_nonstop(
                task, node.state, require_stop=evaluable
            )
            ordering_started = time.perf_counter_ns()
            scored = [
                (
                    -float(probabilities[action]),
                    (_default_action_order(task, node.state, action, weights)
                     if config.use_objective_guided_ordering else 0.0),
                    action,
                )
                for action in legal
            ]
            scored.sort()
            for _negative_probability, objective_hint, action in scored:
                proposals.append((
                    node.rank
                    + math.log(max(float(probabilities[action]), 1.0e-300))
                    - 0.25 * objective_hint,
                    node,
                    action,
                ))
            ordering_ns += time.perf_counter_ns() - ordering_started
        if budget_hit:
            break
        if not proposals:
            break
        proposals.sort(key=lambda item: (-item[0], item[1].actions, item[2]))
        selected = (
            _select_with_exploration(
                task, proposals, config.beam_width, action_index=2
            )
            if config.use_objective_guided_ordering
            else proposals[:config.beam_width]
        )
        budget_unexplored += len(proposals) - len(selected)
        next_nodes: list[_Node] = []
        for rank, node, action in selected:
            transition_started = time.perf_counter_ns()
            following = task.transition(node.state, action)
            transition_ns += time.perf_counter_ns() - transition_started
            if following is None:
                raise SearchContractError("a legal action produced no next state")
            next_nodes.append(_Node(
                following,
                node.actions + (action,),
                rank,
                node.depth + 1,
            ))
        frontier = next_nodes

    if best_outcome is None or incumbent is None:
        raise SearchContractError("search produced no complete legal candidate")
    total_ns = time.perf_counter_ns() - total_started
    return SearchResult(
        method="frozen_weight_conditioned_grpo",
        weight=weights,
        best_actions=best_actions,
        best_outcome=best_outcome,
        best_objective=incumbent,
        evaluations=len(evaluated),
        evaluation_budget=config.max_evaluations,
        certified_pruned_states=certified_pruned_states,
        budget_unexplored_actions=budget_unexplored,
        duplicate_states_skipped=duplicates,
        objective_guided_ordering=config.use_objective_guided_ordering,
        timing=PredictionTiming(
            feature_time_ns=feature_ns,
            model_inference_time_ns=inference_ns,
            action_ordering_time_ns=ordering_ns,
            environment_transition_time_ns=transition_ns,
            certified_pruning_time_ns=pruning_ns,
            candidate_evaluation_time_ns=evaluation_ns,
            total_prediction_time_ns=total_ns,
        ),
        evaluated_candidates=tuple(evaluated),
    )


def _default_action_order(
    task: GraphCompressionTask,
    _state: Any,
    action: int,
    weight: tuple[float, float, float],
) -> float:
    """Transparent ordering hint; it does not remove an operator group."""
    try:
        record = task.action_record(action)  # type: ignore[attr-defined]
        text = " ".join((
            str(getattr(record, "key", "")),
            str(getattr(record, "runtime_action", "")),
            str(getattr(record, "category", "")),
        )).lower()
    except (AttributeError, KeyError, TypeError, ValueError):
        text = ""
    if any(term in text for term in (
        "delta", "zigzag", "bitpack", "bitunpack", "frame_of_reference",
    )):
        group = "delta_zigzag_pack"
    elif any(term in text for term in ("dictionary", "run_length", "rle")):
        group = "dictionary_run_length"
    elif any(term in text for term in (
        "xor", "window", "scaling", "outlier", "exception",
    )):
        group = "xor_window_scaling_exception"
    else:
        group = "other_legal"
    feature_provider = getattr(task, "ordering_features", None)
    features = feature_provider() if callable(feature_provider) else {}
    priorities = feature_priority(weight, features)
    measurement_provider = getattr(task, "measured_action_cost", None)
    measured = measurement_provider(action) if callable(measurement_provider) else None
    count_provider = getattr(task, "action_measurement_count", None)
    count = int(count_provider(action)) if callable(count_provider) else 0
    e0 = float(getattr(task, "compression_reference_ns_per_byte", 1.0))
    d0 = float(getattr(task, "decompression_reference_ns_per_byte", 1.0))
    return objective_guided_ordering_score(
        weight=weight,
        estimated_cost=measured,
        compression_reference_ns_per_byte=e0,
        decompression_reference_ns_per_byte=d0,
        feature_priority_mass=priorities[group],
        measurement_count=count,
    ) + float(action) * 1.0e-12


def _action_measurement_count(task: GraphCompressionTask, action: int) -> int:
    provider = getattr(task, "action_measurement_count", None)
    if not callable(provider):
        return 0
    count = provider(action)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise SearchContractError("action_measurement_count returned an invalid count")
    return int(count)


def _select_with_exploration(
    task: GraphCompressionTask,
    ordered: Sequence[tuple[Any, ...]],
    limit: int,
    *,
    action_index: int,
) -> list[tuple[Any, ...]]:
    """Reserve one fifth of a beam for least-measured legal actions.

    ``ordered`` is already in exploit order.  The exploration lane only changes
    traversal order and never removes an action by proof.
    """
    if limit <= 0:
        return []
    if len(ordered) <= limit or limit == 1:
        return list(ordered[:limit])
    exploration_slots = max(1, limit // 5)
    exploit_slots = limit - exploration_slots
    selected = list(ordered[:exploit_slots])
    selected_ids = {id(row) for row in selected}
    remaining = [row for row in ordered if id(row) not in selected_ids]
    remaining.sort(key=lambda row: (
        _action_measurement_count(task, int(row[action_index])),
        int(row[action_index]),
    ))
    selected.extend(remaining[:exploration_slots])
    return selected


def beam30_search(
    task: GraphCompressionTask,
    weight: Sequence[float],
    *,
    config: SearchConfig,
    bound_provider: BoundProvider | None = None,
    action_ordering: ActionOrdering | None = None,
) -> SearchResult:
    """Beam30 with the same complete-candidate budget as policy search."""
    total_started = time.perf_counter_ns()
    _validate_task(task)
    weights_array = validate_weight(weight)
    weights = tuple(float(value) for value in weights_array)
    horizon = min(int(task.max_steps), config.max_steps or int(task.max_steps))
    if action_ordering is not None:
        ordering = action_ordering
    elif config.use_objective_guided_ordering:
        ordering = _default_action_order
    else:
        ordering = lambda _task, _state, action, _weight: float(action)
    frontier = [_Node(task.initial_state(), (), 0.0, 0)]
    seen: set[tuple[Any, ...]] = set()
    evaluated: list[CandidateSummary] = []
    best_outcome: TerminalOutcome | None = None
    best_actions: tuple[int, ...] = ()
    incumbent: float | None = None
    certified_pruned_states = budget_unexplored = duplicates = 0
    ordering_ns = transition_ns = pruning_ns = evaluation_ns = 0

    while frontier and len(evaluated) < config.max_evaluations:
        ranked_current: list[tuple[float, _Node]] = []
        for node in frontier:
            key = _state_key(node)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            pruning_started = time.perf_counter_ns()
            pruned = _maybe_certified_prune(
                task, node.state, weights, incumbent, bound_provider,
                config.keep_objective_ties,
            )
            pruning_ns += time.perf_counter_ns() - pruning_started
            if pruned:
                certified_pruned_states += 1
                continue
            evaluable = _is_terminal_evaluable(task, node.state)
            if evaluable:
                evaluation_started = time.perf_counter_ns()
                outcome = task.evaluate_terminal(node.state)
                objective = objective_value(outcome, weights)
                evaluation_ns += time.perf_counter_ns() - evaluation_started
                evaluated.append(_summary(node.actions, outcome, objective))
                ranked_current.append((objective, node))
                if incumbent is None or objective < incumbent:
                    incumbent = objective
                    best_outcome = outcome
                    best_actions = node.actions
                if len(evaluated) >= config.max_evaluations:
                    break
            else:
                # Multi-token continuation edit prefixes have no physical
                # objective yet.  Keep them in the frontier with their policy
                # ordering score; exact measurements resume once the edit is
                # complete.
                ranked_current.append((node.rank, node))
        if len(evaluated) >= config.max_evaluations:
            budget_unexplored += sum(
                len(_legal_nonstop(task, node.state)) for _score, node in ranked_current
            )
            break
        ranked_current.sort(key=lambda item: (item[0], item[1].actions))
        proposals: list[tuple[float, _Node, int, float]] = []
        remaining_slots = config.max_evaluations - len(evaluated)
        for parent_objective, node in ranked_current[:config.beam_width]:
            if node.depth >= horizon - 1:
                continue
            ordering_started = time.perf_counter_ns()
            legal = _legal_nonstop(
                task, node.state,
                require_stop=_is_terminal_evaluable(task, node.state),
            )
            ordered = sorted(
                ((ordering(task, node.state, action, weights), action)
                 for action in legal),
                key=lambda item: (item[0], item[1]),
            )
            ordering_ns += time.perf_counter_ns() - ordering_started
            for score, action in ordered:
                proposals.append((
                    score, node, action, parent_objective,
                ))
        proposals.sort(key=lambda item: (item[0], item[1].actions, item[2]))
        child_limit = min(config.beam_width, remaining_slots)
        selected = (
            _select_with_exploration(
                task, proposals, child_limit, action_index=2
            )
            if config.use_objective_guided_ordering
            else proposals[:child_limit]
        )
        budget_unexplored += len(proposals) - len(selected)
        next_nodes: list[_Node] = []
        for _score, node, action, parent_objective in selected:
            transition_started = time.perf_counter_ns()
            following = task.transition(node.state, action)
            transition_ns += time.perf_counter_ns() - transition_started
            if following is None:
                raise SearchContractError("a legal action produced no next state")
            next_nodes.append(_Node(
                following, node.actions + (action,), parent_objective, node.depth + 1
            ))
        frontier = next_nodes

    if best_outcome is None or incumbent is None:
        raise SearchContractError("search produced no complete legal candidate")
    total_ns = time.perf_counter_ns() - total_started
    return SearchResult(
        method=f"beam{config.beam_width}",
        weight=weights,
        best_actions=best_actions,
        best_outcome=best_outcome,
        best_objective=incumbent,
        evaluations=len(evaluated),
        evaluation_budget=config.max_evaluations,
        certified_pruned_states=certified_pruned_states,
        budget_unexplored_actions=budget_unexplored,
        duplicate_states_skipped=duplicates,
        objective_guided_ordering=config.use_objective_guided_ordering,
        timing=PredictionTiming(
            action_ordering_time_ns=ordering_ns,
            environment_transition_time_ns=transition_ns,
            certified_pruning_time_ns=pruning_ns,
            candidate_evaluation_time_ns=evaluation_ns,
            total_prediction_time_ns=total_ns,
        ),
        evaluated_candidates=tuple(evaluated),
    )


def accept_continuation_only_if_better(
    baseline: SearchResult,
    adapted: SearchResult,
) -> SearchResult:
    """Keep the prior block recipe when at most two local edits do not improve J."""
    if baseline.weight != adapted.weight:
        raise ValueError("baseline and adapted search must use the same weight")
    return adapted if adapted.best_objective < baseline.best_objective else baseline


__all__ = [
    "BEAM_WIDTH", "CONTINUATION_EVALUATIONS", "FIRST_BLOCK_EVALUATIONS",
    "CandidateSummary", "PredictionTiming", "SCHEMA", "SearchConfig",
    "SearchContractError", "SearchResult", "accept_continuation_only_if_better",
    "beam30_search", "policy_candidate_search",
]
