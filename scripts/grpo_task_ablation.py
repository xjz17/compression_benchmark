#!/usr/bin/env python3
"""Feature and task-adapted GRPO ablations for depth-3 operator search.

All variants keep the 106-dimensional actor input shape and zero disabled
feature blocks.  This preserves paired actor initialization across variants.
The study is exploratory outer leave-one-column-out evaluation: every variant
is reported, and no test-column result is used to select another variant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FEATURE_SLICES = {
    "raw": slice(0, 18),
    "probes": slice(18, 46),
    "rate": slice(46, 47),
    "depth": slice(47, 50),
    "prefix": slice(50, 106),
}


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    disabled_features: tuple[str, ...] = ()
    advantage: str = "std"
    clip_mode: str = "fixed"
    clip_beta: float = 0.0
    clip_higher: bool = False
    global_rezscore: bool = True
    trajectory_equal_weight: bool = True
    what_it_tests: str = ""
    expected_if_component_matters: str = ""


VARIANTS = (
    Variant(
        "feat_full_std_fixed", "feature", what_it_tests="106D full-state reference",
        expected_if_component_matters="reference for every leave-one-block-out comparison",
    ),
    Variant(
        "feat_no_raw", "feature", ("raw",), what_it_tests="18 raw column statistics",
        expected_if_component_matters="held-out codec rate worsens without cross-column shape/type statistics",
    ),
    Variant(
        "feat_no_probes", "feature", ("probes",), what_it_tests="28 target-column single-operator probes",
        expected_if_component_matters="held-out codec rate worsens when the policy cannot observe operator affinity",
    ),
    Variant(
        "feat_no_rate", "feature", ("rate",), what_it_tests="current-prefix proxy rate ratio",
        expected_if_component_matters="STOP and later-action decisions worsen without progress feedback",
    ),
    Variant(
        "feat_no_depth", "feature", ("depth",), what_it_tests="3D depth one-hot",
        expected_if_component_matters="action choice worsens when the policy cannot distinguish decision stage",
    ),
    Variant(
        "feat_no_prefix", "feature", ("prefix",), what_it_tests="56D prior-action identity",
        expected_if_component_matters="operator interaction modeling worsens without the chosen-prefix identity",
    ),
    Variant(
        "feat_no_history", "feature", ("rate", "depth", "prefix"),
        what_it_tests="column descriptors without sequential state",
        expected_if_component_matters="performance drops if stateful composition is more than a per-column static ranking",
    ),
    Variant(
        "method_robust_fixed", "method", advantage="robust", global_rezscore=False,
        what_it_tests="median plus MAD dispersion with a training-only EMA floor instead of mean/std",
        expected_if_component_matters="less sensitivity to one unusually good or bad sampled pipeline",
    ),
    Variant(
        "method_rank_fixed", "method", advantage="rank", global_rezscore=False,
        what_it_tests="tie-aware rank advantage instead of magnitude advantage",
        expected_if_component_matters="better stability when return magnitudes are heavy-tailed across operator paths",
    ),
    Variant(
        "method_std_adaptive", "method", clip_mode="dispersion_adaptive", clip_beta=0.5,
        global_rezscore=False,
        what_it_tests="training-EMA dispersion-adaptive clipping with standard advantages",
        expected_if_component_matters="tight updates for weakly separated groups and wider updates for informative groups",
    ),
    Variant(
        "method_robust_adaptive", "method", advantage="robust", clip_mode="dispersion_adaptive",
        clip_beta=0.5, global_rezscore=False,
        what_it_tests="robust dispersion advantage plus dispersion-adaptive clipping",
        expected_if_component_matters="combined robustness and confidence-aware update size improve held-out rate",
    ),
    Variant(
        "method_std_no_global", "method", global_rezscore=False,
        what_it_tests="remove the second global advantage z-score",
        expected_if_component_matters="preserving within-column GRPO normalization avoids cross-group rescaling distortion",
    ),
    Variant(
        "method_std_action_weight", "method", global_rezscore=False, trajectory_equal_weight=False,
        what_it_tests="action-level rather than 1/L trajectory-equal weighting",
        expected_if_component_matters="a drop would show that short and long paths should contribute equal total mass",
    ),
    Variant(
        "method_robust_clip_higher", "method", advantage="robust",
        clip_mode="dispersion_adaptive", clip_beta=0.5, clip_higher=True,
        global_rezscore=False,
        what_it_tests="dispersion-adaptive asymmetric Clip-Higher",
        expected_if_component_matters="positive trajectories need a wider trust region without destabilizing negative updates",
    ),
    Variant(
        "method_robust_reverse_clip", "method", advantage="robust",
        clip_mode="dispersion_adaptive", clip_beta=-0.5, global_rezscore=False,
        what_it_tests="reverse-beta falsifier for the dispersion interpretation",
        expected_if_component_matters="should lose if high dispersion is reliable separation; a win means dispersion is instability instead",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def load_base(script_dir: Path):
    path = script_dir / "cross_column_rl_compare.py"
    spec = importlib.util.spec_from_file_location("grpo_ablation_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mask_features(full: np.ndarray, variant: Variant) -> np.ndarray:
    if full.shape[1] != 106:
        raise AssertionError(f"expected 106 features, got {full.shape}")
    result = full.copy()
    for block in variant.disabled_features:
        result[:, FEATURE_SLICES[block]] = 0.0
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def robust_dispersion(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def group_advantage(returns: np.ndarray, mode: str, scale_reference: float) -> tuple[np.ndarray, float]:
    returns = np.asarray(returns, dtype=np.float64)
    dispersion = robust_dispersion(returns)
    if mode == "std":
        scale = float(np.std(returns))
        if scale < 1.0e-8:
            return np.zeros_like(returns), dispersion
        return (returns - float(np.mean(returns))) / (scale + 1.0e-8), dispersion
    if mode == "robust":
        scale = max(dispersion, 0.05 * scale_reference, 1.0e-8)
        advantage = (returns - float(np.median(returns))) / (scale + 1.0e-8)
        advantage = np.clip(advantage, -3.0, 3.0)
        return advantage - float(np.mean(advantage)), dispersion
    if mode == "rank":
        ranks = average_ranks(returns)
        centered = ranks - float(np.mean(ranks))
        if not np.any(centered):
            return np.zeros_like(returns), dispersion
        scale = math.sqrt((returns.size * returns.size - 1.0) / 12.0)
        return centered / scale, dispersion
    raise ValueError(mode)


def collect_group_relative(base, env, features, actor, rng, train_columns, groups, group_size, variant, scale_reference=1.0e-3):
    pieces: list[dict[str, object]] = []
    group_columns = rng.choice(train_columns, size=groups, replace=True)
    episode_columns = np.repeat(group_columns, group_size)
    rollout = base.collect_episodes(env, features, actor, rng, episode_columns, None)
    for group in range(groups):
        start = group * group_size
        stop = start + group_size
        returns = rollout["episode_returns"][start:stop]
        advantage, dispersion = group_advantage(returns, variant.advantage, scale_reference)
        selected = (rollout["episode"] >= start) & (rollout["episode"] < stop)
        local_episode = rollout["episode"][selected] - start
        pieces.append({
            "states": rollout["states"][selected],
            "actions": rollout["actions"][selected],
            "old_logp": rollout["old_logp"][selected],
            "advantages": advantage[local_episode],
            "lengths": rollout["lengths"][start:stop],
            "local_episode": local_episode,
            "dispersion": dispersion,
        })

    dispersions = np.asarray([float(piece["dispersion"]) for piece in pieces], dtype=np.float64)
    if variant.clip_mode == "fixed":
        group_epsilon_low = np.full(groups, 0.2, dtype=np.float64)
        group_epsilon_high = np.full(groups, 0.2, dtype=np.float64)
    elif variant.clip_mode == "dispersion_adaptive":
        relative_log_dispersion = np.clip(
            np.log((dispersions + 1.0e-3) / (scale_reference + 1.0e-3)),
            -3.0,
            3.0,
        )
        group_epsilon_low = np.clip(
            0.2 * np.exp(variant.clip_beta * relative_log_dispersion),
            0.10,
            0.30,
        )
        if variant.clip_higher:
            group_epsilon_high = np.minimum(1.25 * group_epsilon_low, 0.35)
        else:
            group_epsilon_high = group_epsilon_low.copy()
    else:
        raise ValueError(variant.clip_mode)

    states = np.concatenate([piece["states"] for piece in pieces])
    if variant.trajectory_equal_weight:
        weights = np.concatenate([
            1.0 / piece["lengths"][piece["local_episode"]] for piece in pieces
        ])
    else:
        weights = np.ones(states.size, dtype=np.float64)
    return {
        "states": states,
        "actions": np.concatenate([piece["actions"] for piece in pieces]),
        "old_logp": np.concatenate([piece["old_logp"] for piece in pieces]),
        "advantages": np.concatenate([piece["advantages"] for piece in pieces]),
        "targets": None,
        "weights": weights,
        "clip_low": np.concatenate([
            np.full(len(piece["states"]), group_epsilon_low[index], dtype=np.float64)
            for index, piece in enumerate(pieces)
        ]),
        "clip_high": np.concatenate([
            np.full(len(piece["states"]), group_epsilon_high[index], dtype=np.float64)
            for index, piece in enumerate(pieces)
        ]),
    }, {
        "sampled_groups": groups,
        "filtered_groups": 0,
        "mean_group_dispersion": float(np.mean(dispersions)),
        "batch_nonzero_dispersion_median": float(np.median(dispersions[dispersions > 0])) if np.any(dispersions > 0) else 0.0,
        "scale_reference": scale_reference,
        "median_clip_epsilon_low": float(np.median(group_epsilon_low)),
        "median_clip_epsilon_high": float(np.median(group_epsilon_high)),
        "min_clip_epsilon": float(np.min(group_epsilon_low)),
        "max_clip_epsilon": float(np.max(group_epsilon_high)),
    }


def actor_update(base, env, features, actor, optimizer, rollout, rng, epochs, minibatch, entropy_coef, global_rezscore):
    states = rollout["states"]
    actions = rollout["actions"]
    old_logp = rollout["old_logp"]
    advantages = rollout["advantages"].astype(np.float64, copy=True)
    weights = rollout["weights"].astype(np.float64, copy=True)
    clip_low = rollout["clip_low"]
    clip_high = rollout["clip_high"]
    if global_rezscore and float(np.std(advantages)) > 1.0e-8:
        advantages = (advantages - float(np.mean(advantages))) / (float(np.std(advantages)) + 1.0e-8)
    indices = np.arange(states.size)
    clipped: list[float] = []
    gradient_norms: list[float] = []
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, indices.size, minibatch):
            take = indices[offset : offset + minibatch]
            x, hidden, probs = base.masked_distribution(actor, features, env, states[take])
            selected = probs[np.arange(take.size), actions[take]]
            ratio = np.exp(np.log(np.maximum(selected, 1.0e-30)) - old_logp[take])
            advantage = advantages[take]
            active = ((advantage >= 0) & (ratio <= 1.0 + clip_high[take])) | (
                (advantage < 0) & (ratio >= 1.0 - clip_low[take])
            )
            local_weight = weights[take]
            local_weight /= float(np.sum(local_weight))
            coefficient = np.where(active, -advantage * ratio * local_weight, 0.0)
            grad_logits = -coefficient[:, None] * probs
            grad_logits[np.arange(take.size), actions[take]] += coefficient
            log_probs = np.log(np.maximum(probs, 1.0e-30))
            entropy = -np.sum(probs * log_probs, axis=1, keepdims=True)
            grad_logits += entropy_coef * local_weight[:, None] * probs * (log_probs + entropy)
            gradients = actor.backward(x, hidden, grad_logits)
            gradient_norms.append(math.sqrt(sum(float(np.sum(grad * grad)) for grad in gradients)))
            optimizer.step(gradients)
            clipped.append(float(np.mean(~active)))
    _x, _hidden, final_probs = base.masked_distribution(actor, features, env, states)
    final_selected = final_probs[np.arange(states.size), actions]
    final_logp = np.log(np.maximum(final_selected, 1.0e-30))
    entropy = -np.sum(final_probs * np.log(np.maximum(final_probs, 1.0e-30)), axis=1)
    stop_states = env.state_depth[states] >= 1
    return {
        "clip_fraction": float(np.mean(clipped)) if clipped else 0.0,
        "approx_kl": float(np.mean(old_logp - final_logp)),
        "policy_entropy": float(np.mean(entropy)),
        "effective_action_count": float(np.exp(np.mean(entropy))),
        "mean_stop_probability": float(np.mean(final_probs[stop_states, base.STOP])) if np.any(stop_states) else 0.0,
        "sampled_stop_fraction": float(np.mean(actions == base.STOP)),
        "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        "max_gradient_norm": float(np.max(gradient_norms)) if gradient_norms else 0.0,
    }


def evaluate(base, env, features, actor, validation_columns, variant, seed, fold):
    rows = base.evaluate_columns(env, features, actor, validation_columns, variant.name, seed, fold)
    for row in rows:
        row["variant_family"] = variant.family
        row["disabled_features"] = "+".join(variant.disabled_features) or "none"
        row["advantage_mode"] = variant.advantage
        row["clip_mode"] = variant.clip_mode
        row["global_rezscore"] = int(variant.global_rezscore)
        row["trajectory_equal_weight"] = int(variant.trajectory_equal_weight)
    return rows


def train_one(base, env, full_features, train_columns, validation_columns, variant, seed, fold, args):
    features = mask_features(full_features, variant)
    base_seed = seed + 1009 * fold
    init_rng = np.random.default_rng(base_seed)
    rng = np.random.default_rng(base_seed + 100_003)
    actor = base.MLP(features.shape[1], args.hidden, 29, init_rng)
    optimizer = base.Adam(actor.params, args.actor_lr)
    training_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    scale_reference = 1.0e-3
    for iteration in range(1, args.iterations + 1):
        rollout, sampling = collect_group_relative(
            base, env, features, actor, rng, train_columns,
            args.groups_per_iteration, args.group_size, variant, scale_reference,
        )
        update_stats = actor_update(
            base, env, features, actor, optimizer, rollout, rng,
            args.policy_epochs, args.minibatch, args.entropy_coef,
            variant.global_rezscore,
        )
        if sampling["batch_nonzero_dispersion_median"] > 0:
            scale_reference = 0.9 * scale_reference + 0.1 * sampling["batch_nonzero_dispersion_median"]
        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            train_eval = evaluate(base, env, features, actor, train_columns, variant, seed, fold)
            gains = np.asarray([row["gain_vs_baseline_pct"] for row in train_eval])
            regrets = np.asarray([row["oracle_regret_pct"] for row in train_eval])
            row = {
                "algorithm": variant.name,
                "variant_family": variant.family,
                "seed": seed,
                "fold": fold,
                "iteration": iteration,
                "transitions": int(rollout["states"].size),
                "train_median_gain_vs_baseline_pct": float(np.median(gains)),
                "train_columns_better_than_baseline": int(np.sum(gains > 0)),
                "train_median_oracle_regret_pct": float(np.median(regrets)),
                **update_stats,
                **sampling,
                "elapsed_sec": time.perf_counter() - started,
            }
            training_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return (
        evaluate(base, env, features, actor, validation_columns, variant, seed, fold),
        training_rows,
        actor,
        {
            "algorithm": variant.name,
            "variant_family": variant.family,
            "seed": seed,
            "fold": fold,
            "requested_iterations": args.iterations,
            "completed_iterations": args.iterations,
            "early_stop_reason": "",
            "early_stop_sampled_groups": 0,
            "early_stop_filtered_groups": 0,
        },
    )


def self_check(base, env, folds, args) -> None:
    base.self_check(env, folds, args)
    train = np.flatnonzero(folds != 0)
    full = base.build_state_features(env, train)
    if full.shape[1] != 106:
        raise AssertionError(full.shape)
    for variant in VARIANTS:
        features = mask_features(full, variant)
        for block in variant.disabled_features:
            if np.any(features[:, FEATURE_SLICES[block]]):
                raise AssertionError((variant.name, block))
    tied = np.asarray([1.0, 1.0, 3.0, 7.0])
    if not np.allclose(average_ranks(tied), [0.5, 0.5, 2.0, 3.0]):
        raise AssertionError("tie-aware ranks failed")

    # Exact one-update witness: the full standard/fixed variant must reproduce
    # the original GRPO collector and actor update.
    variant = VARIANT_BY_NAME["feat_full_std_fixed"]
    init_a = np.random.default_rng(123)
    init_b = np.random.default_rng(123)
    actor_a = base.MLP(106, 8, 29, init_a)
    actor_b = base.MLP(106, 8, 29, init_b)
    opt_a = base.Adam(actor_a.params, 0.003)
    opt_b = base.Adam(actor_b.params, 0.003)
    rng_a = np.random.default_rng(456)
    rng_b = np.random.default_rng(456)
    old, _ = base.collect_group_relative(env, full, actor_a, rng_a, train, 4, 8, False)
    new, _ = collect_group_relative(base, env, full, actor_b, rng_b, train, 4, 8, variant)
    for key in ("states", "actions", "old_logp", "advantages", "weights"):
        if not np.array_equal(old[key], new[key]):
            raise AssertionError(f"baseline collector divergence: {key}")
    base.actor_update(env, full, actor_a, opt_a, old, rng_a, 1, 512, 0.2, 0.2, 0.01)
    actor_update(base, env, full, actor_b, opt_b, new, rng_b, 1, 512, 0.01, True)
    if any(not np.array_equal(left, right) for left, right in zip(actor_a.params, actor_b.params)):
        raise AssertionError("baseline actor-update divergence")
    print("ABLATION_SELF_CHECK exact_full_variant_match=PASS", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_BY_NAME), default=tuple(VARIANT_BY_NAME))
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260825, 20260826, 20260827))
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--groups-per-iteration", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=0.003)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--self-check-only", action="store_true")
    # Compatibility fields read by the base self-check.
    parser.set_defaults(episodes_per_iteration=512, gae_lambda=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base, base_path = load_base(Path(__file__).resolve().parent)
    env, data_inputs, operator_source = base.load_environment(
        args.repo_root.resolve(), args.measurement_root.resolve(),
        args.summary_csv.resolve(), args.baseline_csv.resolve(), tuple(args.datasets),
    )
    if args.folds != len(env.columns):
        raise ValueError("this ablation requires leave-one-column-out: folds must equal selected columns")
    folds = base.assign_folds(env.columns, args.folds)
    self_check(base, env, folds, args)
    if args.self_check_only:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --self-check-only")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "fold_assignments.csv", [
        {
            "dataset": column.dataset,
            "column_index": column.index,
            "column_name": column.name,
            "logical_type": column.logical_type,
            "fold": int(folds[position]),
        }
        for position, column in enumerate(env.columns)
    ])

    selected_variants = [VARIANT_BY_NAME[name] for name in args.variants]
    evaluation_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for variant in selected_variants:
        for seed in args.seeds:
            for fold in range(args.folds):
                train_columns = np.flatnonzero(folds != fold)
                validation_columns = np.flatnonzero(folds == fold)
                full_features = base.build_state_features(env, train_columns)
                evaluation, training, actor, status = train_one(
                    base, env, full_features, train_columns, validation_columns,
                    variant, seed, fold, args,
                )
                evaluation_rows.extend(evaluation)
                training_rows.extend(training)
                status_rows.append(status)
                write_csv(output_dir / "evaluation.csv", evaluation_rows)
                write_csv(output_dir / "training_curve.csv", training_rows)
                write_csv(output_dir / "run_status.csv", status_rows)
                np.savez_compressed(
                    output_dir / f"policy_{variant.name}_seed{seed}_fold{fold}.npz",
                    W1=actor.params[0], b1=actor.params[1], W2=actor.params[2], b2=actor.params[3],
                )

    script_path = Path(__file__).resolve()
    expected = len(selected_variants) * len(args.seeds) * len(env.columns)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "script": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "base_script": str(base_path),
        "base_script_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "operator_source": str(operator_source),
        "operator_source_sha256": hashlib.sha256(operator_source.read_bytes()).hexdigest(),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "variants": [asdict(variant) for variant in selected_variants],
        "protocol": {
            "split": f"{args.folds}-fold leave-one-column-out; normalization and rewards use training columns only",
            "evaluation_status": "exploratory outer-test comparison; all variants reported; not a nested method-selection estimate",
            "paired_initialization": "same 106D actor shape and same actor seed; disabled feature blocks are zeroed",
            "full_variant_witness": "one collector and actor update exactly match cross_column_rl_compare.py GRPO",
            "reward": "log(best-single/final proxy rate) within a same-column group of eight trajectories",
            "robust_advantage": "median centered; scale=max(1.4826*MAD,0.05*training-fold EMA,1e-8); clip to [-3,3] then recenter",
            "rank_advantage": "tie-aware average ranks, centered and standardized within group",
            "adaptive_clip": "epsilon_g=clip(0.2*exp(beta*clip(log((1.4826*MAD+1e-3)/(training-fold EMA+1e-3)),-3,3)),0.10,0.30); optional upper=1.25*epsilon capped at 0.35",
            "codec_followup": "evaluation.csv must be replayed through standalone byte-stream codec before compression claims",
        },
        "claim_boundary": {
            "probe_conditioned": True,
            "production_claim_allowed": False,
            "method_selection_allowed": False,
            "reason": "same two datasets are used for exploratory ablation; confirm the chosen formula on fresh datasets or nested meta-validation",
        },
        "inputs": {
            "measurement_manifests": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(args.measurement_root.resolve().glob("*/manifest.json"))
            },
            "summary_csv": hashlib.sha256(args.summary_csv.resolve().read_bytes()).hexdigest(),
            "baseline_csv": hashlib.sha256(args.baseline_csv.resolve().read_bytes()).hexdigest(),
            "datasets": data_inputs,
        },
        "counts": {
            "columns": len(env.columns),
            "variants": len(selected_variants),
            "seeds": len(args.seeds),
            "evaluation_rows": len(evaluation_rows),
            "expected_evaluation_rows": expected,
            "training_runs": len(status_rows),
        },
        "wall_sec": time.perf_counter() - started,
    }
    if len(evaluation_rows) != expected or len(status_rows) != expected:
        raise AssertionError(manifest["counts"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"OUTPUT {output_dir}", flush=True)


if __name__ == "__main__":
    main()
