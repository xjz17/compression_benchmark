"""Inference-weight checkpoints bound to an exact action/feature catalog."""
import json
from pathlib import Path
import numpy as np
from search_physical_operator_variants_v2 import MLP
import current_action_parameters as action_parameters
from canonical156_parameter_policy import attach, CANDIDATES


def identity(env):
    return {'format': 'algorithm-parameter-policy-weights-v3',
            'policy_registry_version': getattr(env, 'policy_registry_version', None),
            'catalog_fingerprint': getattr(env, 'catalog_fingerprint', None),
            'action_count': int(env.action_count),
            'stop_id': int(getattr(env, 'stop_id', getattr(env, 'stop', -1))),
            'feature_dim': int(np.asarray(env.features(())).size),
            'environment_type': type(env).__module__ + '.' + type(env).__qualname__,
            'feature_schema': 'algorithm-and-option-and-codec-parameters-v6',
            'parameter_count': len(CANDIDATES) if callable(getattr(env, 'parameter_choices', None)) else 0,
            'parameter_fingerprint': action_parameters.parameter_fingerprint(),
            'feature_config': {
                'max_steps': int(env.max_steps),
                'base_max_steps': (int(env.base.max_steps) if hasattr(env, 'base') else None),
                'max_atomic_depth': getattr(env, 'max_atomic_depth', None),
                'auto_complete_option': getattr(env, 'auto_complete_option', None),
                'complete_on_transition': getattr(env, 'complete_on_transition', None),
            }}


def bind(actor, env):
    attach(actor, env)
    actor.policy_identity = identity(env)
    return actor


def validate(actor, env):
    expected = identity(env)
    actual = getattr(actor, 'policy_identity', None)
    if actual != expected:
        raise ValueError('policy checkpoint catalog/registry/parameters/features mismatch; old weights cannot '
                         'be relabeled as the current algorithm and conditional parameter heads. '
                         'Migrate archived paths with io_variants and parameters, remeasure and retrain.')
    if actor.params[0].shape[0] != expected['feature_dim'] or actor.params[2].shape[1] != expected['action_count']:
        raise ValueError('policy parameter shapes differ from checkpoint identity')
    head = getattr(actor, 'parameter_head', None)
    if expected['parameter_count'] and (head is None or head.params[0].shape[0] != expected['feature_dim'] or head.params[2].shape[1] != expected['parameter_count']):
        raise ValueError('conditional parameter head differs from checkpoint identity')


def save(actor, path):
    metadata = getattr(actor, 'policy_identity', None)
    if metadata is None:
        raise ValueError('cannot save unversioned policy; initialize/bind it to its environment')
    with Path(path).open('wb') as handle:
        extra = {f'parameter_{i}': a for i, a in enumerate(actor.parameter_head.params)} if hasattr(actor, 'parameter_head') else {}
        np.savez_compressed(handle, *actor.params, policy_identity=json.dumps(metadata, sort_keys=True), **extra)


def load(path, env=None):
    with np.load(Path(path), allow_pickle=False) as archive:
        if 'policy_identity' not in archive:
            raise ValueError('unversioned legacy checkpoint: retain its frozen 206 runtime or migrate '
                             'archived paths with io_variants and retrain; do not resize/relabel logits')
        metadata = json.loads(str(archive['policy_identity'].item()))
        params = [np.array(archive[f'arr_{i}'], dtype=np.float64) for i in range(4)]
        parameter_params = [np.array(archive[f'parameter_{i}'], dtype=np.float64) for i in range(4)] if metadata.get('parameter_count') else None
    if (metadata.get('format') != 'algorithm-parameter-policy-weights-v3'
            or params[0].ndim != 2 or params[2].ndim != 2
            or min(params[0].shape + params[2].shape) < 1
            or params[0].shape[1] != params[2].shape[0]
            or params[1].shape != (params[0].shape[1],)
            or params[3].shape != (params[2].shape[1],)
            or metadata.get('action_count') != params[2].shape[1]
            or metadata.get('feature_dim') != params[0].shape[0]
            or any(not np.all(np.isfinite(a)) for a in params)):
        raise ValueError('invalid saved actor parameters/identity')
    actor = MLP(params[0].shape[0], params[0].shape[1], params[2].shape[1], np.random.default_rng(0))
    actor.params = params
    actor.policy_identity = metadata
    if parameter_params is not None:
        w1, b1, w2, b2 = parameter_params
        if (w1.shape != params[0].shape or b1.shape != params[1].shape or w2.shape != (w1.shape[1], metadata['parameter_count']) or b2.shape != (metadata['parameter_count'],) or any(not np.all(np.isfinite(a)) for a in parameter_params)):
            raise ValueError('invalid conditional parameter head shapes/values')
        actor.parameter_head = MLP(w1.shape[0], w1.shape[1], w2.shape[1], np.random.default_rng(0))
        actor.parameter_head.params = parameter_params
    if env is not None:
        validate(actor, env)
    return actor
