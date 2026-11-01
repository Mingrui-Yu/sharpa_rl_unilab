"""Shared tensor-native on-policy rollout collection and timeout bootstrap."""

from __future__ import annotations

import time
from collections import defaultdict

import torch
from tensordict import TensorDict
from unilab.rl.runtime.collector_timing import extract_env_step_breakdown_timing_ms

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import transition_next

from .policy import inference_distribution


def _reward_totals(info: dict, totals: dict[str, torch.Tensor | float]) -> None:
    for key, value in info.get("log", {}).items():
        if not key.startswith("reward/"):
            continue
        if isinstance(value, torch.Tensor):
            totals[key] = totals.get(key, 0.0) + value.detach()
        else:
            totals[key] = totals.get(key, 0.0) + float(value)


def _empty_like_row(obs, sample, state, *, horizon: int, store_distribution: bool):
    def values(value: torch.Tensor, dtype=None):
        return torch.empty((horizon, *value.shape), dtype=dtype or value.dtype, device=value.device)

    row = {key: values(obs[key]) for key in ("obs", "priv_info", "critic")}
    row["next_critic"] = values(state.obs["critic"])
    row.update(
        actions=values(sample.raw),
        actions_log_prob=values(sample.log_prob),
        rewards=values(state.reward),
        terminated=values(state.terminated),
        truncated=values(state.truncated),
    )
    if store_distribution:
        row.update(behavior_mean=values(sample.mean), behavior_std=values(sample.std))
    return row


def collect(env, obs, actor, horizon, device, *, count=None, metrics=None, store_distribution=True):
    """Collect a time-major rollout without host serialization in the hot loop."""
    rollout_started = time.perf_counter()
    totals = defaultdict(float)
    reward_component_totals: dict[str, torch.Tensor | float] = {}
    batch: dict[str, torch.Tensor] | None = None
    next_obs = obs

    for step in range(horizon):
        inference_started = time.perf_counter()
        with torch.no_grad():
            sample = inference_distribution(actor, next_obs, device).sample()
        inference_seconds = time.perf_counter() - inference_started

        step_started = time.perf_counter()
        state = env.step(sample.action)
        step_seconds = time.perf_counter() - step_started

        if batch is None:
            batch = _empty_like_row(
                next_obs, sample, state, horizon=horizon, store_distribution=store_distribution
            )
        batch["obs"][step].copy_(next_obs["obs"])
        batch["priv_info"][step].copy_(next_obs["priv_info"])
        batch["critic"][step].copy_(next_obs["critic"])
        batch["actions"][step].copy_(sample.raw)
        assert sample.log_prob is not None
        batch["actions_log_prob"][step].copy_(sample.log_prob)
        batch["rewards"][step].copy_(state.reward)
        batch["terminated"][step].copy_(state.terminated)
        batch["truncated"][step].copy_(state.truncated)
        batch["next_critic"][step].copy_(transition_next(state)["critic"])
        if store_distribution:
            batch["behavior_mean"][step].copy_(sample.mean)
            batch["behavior_std"][step].copy_(sample.std)

        if metrics is not None:
            _reward_totals(state.info, reward_component_totals)
            totals["timing/collector_mlp_infer_ms"] += inference_seconds * 1000
            totals["timing/collector_env_step_ms"] += step_seconds * 1000
            for key, value in extract_env_step_breakdown_timing_ms(state.info).items():
                totals[f"timing/collector_{key}"] += value
        next_obs = state.obs

    assert batch is not None
    if count is not None:
        with count.get_lock():
            count.value += env.num_envs * horizon
    if metrics is not None:
        for key, value in reward_component_totals.items():
            totals[key] = float(value) / horizon
        metrics.update({key: value / horizon for key, value in totals.items()})
        rollout_ms = (time.perf_counter() - rollout_started) * 1000
        metrics["timing/collector_rollout_ms"] = rollout_ms
        metrics["perf/collector_active_steps_per_sec"] = (
            env.num_envs * horizon / (rollout_ms / 1000.0)
        )
    return batch, next_obs


@torch.no_grad()
def timeout_rewards(batch, critic, gamma):
    clean = batch["next_critic"]
    values = critic(
        TensorDict({"critic": clean.flatten(0, 1)}, batch_size=clean.shape[0] * clean.shape[1])
    ).view(*clean.shape[:2])
    pure_timeout = batch["truncated"].to(torch.bool) & ~batch["terminated"].to(torch.bool)
    return batch["rewards"] + gamma * values * pure_timeout.to(batch["rewards"].dtype)
