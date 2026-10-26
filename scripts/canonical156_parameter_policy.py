"""Conditional codec parameter head with a joint clipped policy objective.

P(action, parameters|state) = P(action|state) P(parameters|state, parameterized action).
The second factor only exists for actions/options with explicit codec parameters. Both
heads receive gradients from the same joint probability ratio and advantage.
"""
from __future__ import annotations
import hashlib
import numpy as np
import current_action_parameters as action_parameters
from search_physical_operator_variants_v2 import Adam, MLP, distribution

CANDIDATES = action_parameters.all_candidates()
_CANDIDATE_KEYS = tuple(tuple(sorted(choice.items())) for choice in CANDIDATES)


def legal_mask(env, prefix, action):
    choices = tuple(getattr(env, 'parameter_choices', lambda *_: ())(prefix, action))
    keys = {tuple(sorted(choice.items())) for choice in choices}
    return np.array([key in keys for key in _CANDIDATE_KEYS], dtype=bool)

def attach(actor, env):
    if callable(getattr(env, 'parameter_choices', None)) and not hasattr(actor, 'parameter_head'):
        seed = int.from_bytes(hashlib.sha256(actor.params[0].tobytes()).digest()[:8], 'little')
        actor.parameter_head = MLP(actor.params[0].shape[0], actor.params[0].shape[1], len(CANDIDATES), np.random.default_rng(seed))
    return actor


def choose(actor, env, prefix, action, features, rng, greedy=False):
    choices = tuple(getattr(env, 'parameter_choices', lambda *_: ())(prefix, action))
    if not choices: return -1, 0.0, None
    mask = legal_mask(env, prefix, action)
    if not mask.any(): raise ValueError('environment parameter choices are outside the checkpoint catalog')
    if actor is None:
        probabilities = mask.astype(float) / mask.sum()
    else:
        if not hasattr(actor, 'parameter_head'): raise ValueError('policy is missing the conditional parameter head')
        probabilities = distribution(actor.parameter_head, features[None, :], mask[None, :])[1][0]
    selected = int(np.argmax(probabilities)) if greedy else int(rng.choice(len(CANDIDATES), p=probabilities / probabilities.sum()))
    return selected, float(np.log(max(probabilities[selected], 1e-30))), dict(CANDIDATES[selected])


def optimizers(actor, lr):
    return (Adam(actor.params, lr), Adam(actor.parameter_head.params, lr) if hasattr(actor, 'parameter_head') else None)


def _stats(actor, rollout):
    x, masks, actions = rollout['features'], rollout['legal'], rollout['actions']
    hidden, prob, _ = distribution(actor, x, masks)
    logp = np.log(np.maximum(prob[np.arange(len(actions)), actions], 1e-30))
    indices = np.asarray(rollout.get('parameter_actions', np.full(len(actions), -1)), dtype=int)
    rows = np.flatnonzero(indices >= 0)
    params = None
    if len(rows):
        parameter_masks = rollout.get('parameter_legal')
        pmask = parameter_masks[rows] if parameter_masks is not None else np.ones((len(rows), len(CANDIDATES)), bool)
        phidden, pprob, _ = distribution(actor.parameter_head, x[rows], pmask)
        logp[rows] += np.log(np.maximum(pprob[np.arange(len(rows)), indices[rows]], 1e-30))
        params = (rows, indices[rows], phidden, pprob)
    ratios = np.exp(logp - rollout['old_logp'])
    return hidden, prob, ratios, params


def objective(actor, rollout, clip_ratio, entropy_coefficient):
    _h, prob, ratios, params = _stats(actor, rollout)
    weights = rollout['weights'] / rollout['weights'].sum()
    advantage = rollout['advantages']
    surrogate = np.minimum(ratios * advantage, np.clip(ratios, 1-clip_ratio, 1+clip_ratio) * advantage)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-30)), axis=1)
    if params is not None:
        rows, _i, _ph, pp = params
        entropy[rows] += -np.sum(pp * np.log(np.maximum(pp, 1e-30)), axis=1)
    loss = float(-np.sum(weights * surrogate)); ent = float(np.sum(weights * entropy))
    return {'loss': loss - entropy_coefficient * ent, 'policy_loss': loss, 'entropy': ent,
            'approx_kl': float(np.sum(weights * ((ratios - 1) - np.log(ratios))))}


def _gradient(probabilities, actions, coefficient, weights, entropy_coefficient):
    gradient = -coefficient[:, None] * probabilities
    gradient[np.arange(len(actions)), actions] += coefficient
    logs = np.log(np.maximum(probabilities, 1e-30))
    entropy = -np.sum(probabilities * logs, axis=1, keepdims=True)
    return gradient + entropy_coefficient * weights[:, None] * probabilities * (logs + entropy)


def update(actor, optimizer_pair, rollout, epochs, clip_ratio, entropy_coefficient,
           rng=None, minibatch=None, normalize_advantages=False):
    before = objective(actor, rollout, clip_ratio, entropy_coefficient)
    working = dict(rollout)
    if normalize_advantages:
        advantage = np.asarray(working['advantages'], dtype=float).copy()
        if advantage.std() > 1e-8: advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        working['advantages'] = advantage
    indices = np.arange(len(working['actions']))
    batch_size = len(indices) if minibatch is None else minibatch
    clips = []
    steps = 0
    for _ in range(epochs):
        if rng is not None: rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            take = indices[start:start+batch_size]
            batch = {k: np.asarray(working[k])[take] for k in ('features','legal','actions','old_logp','advantages','weights')}
            for k in ('parameter_actions', 'parameter_legal'):
                if k in working: batch[k] = np.asarray(working[k])[take]
            hidden, prob, ratios, params = _stats(actor, batch)
            adv = batch['advantages']; weights = batch['weights'] / batch['weights'].sum()
            active = ((adv >= 0) & (ratios <= 1+clip_ratio)) | ((adv < 0) & (ratios >= 1-clip_ratio))
            coefficient = np.where(active, -adv * ratios * weights, 0.)
            base_gradient = _gradient(prob, batch['actions'], coefficient, weights, entropy_coefficient)
            base_grads = actor.backward(batch['features'], hidden, base_gradient)
            parameter_grads = None
            if params is not None:
                rows, pa, ph, pp = params
                parameter_grads = actor.parameter_head.backward(batch['features'][rows], ph,
                    _gradient(pp, pa, coefficient[rows], weights[rows], entropy_coefficient))
            optimizer_pair[0].step(base_grads)
            if parameter_grads is not None: optimizer_pair[1].step(parameter_grads)
            steps += 1
            clips.append(float(np.sum(weights * ~active)))
    after = objective(actor, rollout, clip_ratio, entropy_coefficient)
    return {'loss_before': before['loss'], 'loss_after': after['loss'],
            'policy_loss_before': before['policy_loss'], 'policy_loss_after': after['policy_loss'],
            'entropy_before': before['entropy'], 'entropy_after': after['entropy'],
            'approx_kl_after': after['approx_kl'], 'clip_fraction': float(np.mean(clips)),
            'optimizer_steps': steps, 'parameter_decisions': int(np.sum(np.asarray(rollout.get('parameter_actions', [])) >= 0))}
