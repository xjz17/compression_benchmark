"""Optional single-device PyTorch execution for the existing float64 GRPO policy.

The NumPy policy/checkpoint remains the portable source of truth. Only neural
policy arithmetic runs here; graph execution and compression measurement do not.
There is no implicit CPU fallback when CUDA was explicitly requested.
"""
from __future__ import annotations

import re
import os
from typing import Any, Mapping

import numpy as np


def normalize_device(value: str) -> str:
    if value == "cuda":
        return "cuda:0"
    if value == "cpu" or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value or ""):
        return value
    raise ValueError("policy device must be cpu or one explicit cuda:N device")


def torch_device(value: str) -> tuple[Any, Any]:
    value = normalize_device(value)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the requested policy backend") from error
    if value.startswith("cuda:"):
        index = int(value.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"requested policy device {value} is unavailable; no CPU fallback")
    return torch, torch.device(value)


def device_receipt(value: str) -> dict[str, Any]:
    value = normalize_device(value)
    if value == "cpu":
        return {"policy_update_device": "cpu", "policy_backend": "numpy_float64"}
    torch, device = torch_device(value)
    # Allocate now, before the first native rollout, to fail early on an unusable GPU.
    torch.zeros(1, dtype=torch.float64, device=device).cpu()
    return {
        "policy_update_device": value,
        "policy_backend": "pytorch_float64",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_gpu_count": 1,
        "rollout_policy_device": "cpu",
        "compression_execution": "owned_by_task_factory; not certified by policy backend",
    }


def frozen_parameters(parameters: Any, value: str) -> tuple[Any, ...]:
    torch, device = torch_device(value)
    return tuple(torch.tensor(np.asarray(p), dtype=torch.float64, device=device)
                 for p in parameters)


def frozen_forward(inputs: np.ndarray, parameters: tuple[Any, ...], value: str) -> np.ndarray:
    torch, device = torch_device(value)
    with torch.inference_mode():
        x = torch.as_tensor(inputs, dtype=torch.float64, device=device)
        w1, b1, w2, b2, w3, b3 = parameters
        logits = torch.tanh(torch.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3
        # The host copy synchronizes GPU work, including when inference is timed.
        return logits.cpu().numpy()


def update_policy(policy: Any, optimizer: Any, rollout: Mapping[str, np.ndarray],
                  config: Any, *, device: str | None = None) -> dict[str, float | int | str]:
    """Execute the same analytic GRPO/Adam update as the NumPy reference.

    An explicit CPU device is useful for numerical parity tests on CPU-only
    hosts. Production dispatch calls this function only for requested CUDA.
    All epochs, hidden-layer gradients, clipping and optimizer moments execute
    on the requested device. Portable arrays are committed after a full update.
    """
    torch, target = torch_device(device or optimizer.execution_device)
    def tensor(value: Any, dtype: Any = None) -> Any:
        return torch.tensor(np.asarray(value), dtype=dtype or torch.float64, device=target)
    with torch.no_grad():
        params = [tensor(p) for p in policy.params]
        first = [tensor(p) for p in optimizer.first]
        second = [tensor(p) for p in optimizer.second]
        steps = optimizer.steps
        x = tensor(rollout["features"])
        masks = tensor(rollout["legal"], torch.bool)
        actions = tensor(rollout["actions"], torch.int64)
        old_logp = tensor(rollout["old_logp"])
        advantages = tensor(rollout["advantages"])
        weights = tensor(rollout["token_weights"])
        weights /= weights.sum()
        indices = torch.arange(actions.numel(), device=target)
        if not bool(masks.any(dim=1).all()):
            raise ValueError("every policy row must have at least one legal action")
        def forward() -> tuple[Any, Any, Any]:
            w1, b1, w2, b2, w3, b3 = params
            h1 = torch.tanh(x @ w1 + b1)
            h2 = torch.tanh(h1 @ w2 + b2)
            logits = (h2 @ w3 + b3).masked_fill(~masks, -torch.inf)
            return torch.softmax(logits, dim=1), h1, h2
        clip_fractions = []
        for _ in range(config.epochs):
            probabilities, h1, h2 = forward()
            ratios = torch.exp(probabilities[indices, actions].clamp_min(1e-300).log() - old_logp)
            active = (((advantages >= 0) & (ratios <= 1 + config.clip_ratio))
                      | ((advantages < 0) & (ratios >= 1 - config.clip_ratio)))
            coefficient = torch.where(active, -advantages * ratios * weights, 0.)
            gradient = -coefficient[:, None] * probabilities
            gradient[indices, actions] += coefficient
            logp = probabilities.clamp_min(1e-300).log()
            entropy = -(probabilities * logp).sum(dim=1, keepdim=True)
            gradient += config.entropy_coefficient * weights[:, None] * probabilities * (logp + entropy)
            w1, b1, w2, b2, w3, b3 = params
            grad_h2 = (gradient @ w3.T) * (1 - h2 * h2)
            grad_h1 = (grad_h2 @ w2.T) * (1 - h1 * h1)
            gradients = [x.T @ grad_h1, grad_h1.sum(0), h1.T @ grad_h2,
                         grad_h2.sum(0), h2.T @ gradient, gradient.sum(0)]
            norm = torch.sqrt(sum(g.square().sum() for g in gradients))
            scale = torch.clamp(config.gradient_max_norm / norm.clamp_min(1e-30), max=1.)
            steps += 1
            for i, (parameter, g) in enumerate(zip(params, gradients)):
                g = g * scale
                first[i] = .9 * first[i] + .1 * g
                second[i] = .999 * second[i] + .001 * g * g
                m = first[i] / (1 - .9 ** steps)
                v = second[i] / (1 - .999 ** steps)
                parameter -= optimizer.learning_rate * m / (torch.sqrt(v) + 1e-8)
            clip_fractions.append((weights * ~active).sum())
        probabilities, _, _ = forward()
        ratios = torch.exp(probabilities[indices, actions].clamp_min(1e-300).log() - old_logp)
        entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1)
        surrogate = torch.minimum(ratios * advantages,
                                  ratios.clamp(1-config.clip_ratio, 1+config.clip_ratio) * advantages)
        if not all(bool(torch.isfinite(p).all()) for p in (*params, *first, *second)):
            raise FloatingPointError("non-finite PyTorch GRPO update; portable policy not committed")
        for destination, source in zip(policy.params, params):
            destination[:] = source.cpu().numpy()
        optimizer.first = [p.cpu().numpy().copy() for p in first]
        optimizer.second = [p.cpu().numpy().copy() for p in second]
        optimizer.steps = steps
        return {
            "policy_loss": float(-(weights * surrogate).sum()),
            "entropy": float((weights * entropy).sum()),
            "approx_kl": float((weights * (ratios - 1 - ratios.log())).sum()),
            "clip_fraction": float(torch.stack(clip_fractions).mean()),
            "optimizer_steps": config.epochs,
            "policy_update_device": str(target),
            "policy_update_backend": "pytorch_float64",
        }
