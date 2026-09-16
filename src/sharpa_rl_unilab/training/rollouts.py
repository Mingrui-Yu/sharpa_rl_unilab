"""Shared on-policy rollout collection and timeout bootstrap."""

import time
from collections import defaultdict

import numpy as np
import torch
from tensordict import TensorDict
from uni_rl.algos.appo.worker import compute_rollout_active_steps_per_sec
from uni_rl.algos.common.collector_timing import extract_env_step_breakdown_timing_ms

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import transition_next

from .policy import policy_td


def collect(env, obs, actor, horizon, device, *, count=None, metrics=None, store_distribution=True):
    rows = []
    rollout_started = time.perf_counter()
    totals = defaultdict(float)
    for _ in range(horizon):
        inference_started = time.perf_counter()
        td = policy_td(obs, device)
        with torch.no_grad():
            sample = actor.policy(td).sample()
        sampled_actions = sample.raw.cpu().numpy().copy()
        executed_actions = sample.action.cpu().numpy()
        inference_seconds = time.perf_counter() - inference_started
        step_started = time.perf_counter()
        state = env.step(executed_actions)
        step_seconds = time.perf_counter() - step_started
        if metrics is not None:
            for key, value in state.info.get("log", {}).items():
                if key.startswith("reward/"):
                    totals[key] += float(value)
            totals["timing/collector_mlp_infer_ms"] += inference_seconds * 1000
            totals["timing/collector_env_step_ms"] += step_seconds * 1000
            for key, value in extract_env_step_breakdown_timing_ms(state.info).items():
                totals[f"timing/collector_{key}"] += value
        nxt = transition_next(state)
        row = {key: obs[key].copy() for key in ("obs", "priv_info", "critic")}
        row["next_critic"] = nxt["critic"]
        row.update(
            actions=sampled_actions,
            actions_log_prob=sample.log_prob.cpu().numpy().copy(),
            rewards=state.reward.copy(),
            terminated=state.terminated.copy(),
            truncated=state.truncated.copy(),
        )
        if store_distribution:
            row["behavior_mean"] = sample.mean.cpu().numpy().copy()
            row["behavior_std"] = sample.std.cpu().numpy().copy()
        rows.append(row)
        obs = state.obs
        if count is not None:
            with count.get_lock():
                count.value += env.num_envs
    batch = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    if metrics is not None:
        # Per-step means for this fresh rollout; rollout wall includes packing,
        # but excludes waiting to publish the packet to the learner.
        metrics.update({key: value / horizon for key, value in totals.items()})
        rollout_ms = (time.perf_counter() - rollout_started) * 1000
        metrics["timing/collector_rollout_ms"] = rollout_ms
        throughput = compute_rollout_active_steps_per_sec(
            num_envs=env.num_envs, steps_per_env=horizon, rollout_ms=rollout_ms
        )
        if throughput is not None:
            metrics["perf/collector_active_steps_per_sec"] = throughput
    return batch, obs


@torch.no_grad()
def timeout_rewards(batch, critic, gamma):
    clean = batch["next_critic"]
    values = critic(
        TensorDict({"critic": clean.flatten(0, 1)}, batch_size=clean.shape[0] * clean.shape[1])
    ).view(*clean.shape[:2])
    pure_timeout = batch["truncated"] * (1 - batch["terminated"])
    return batch["rewards"] + gamma * values * pure_timeout
