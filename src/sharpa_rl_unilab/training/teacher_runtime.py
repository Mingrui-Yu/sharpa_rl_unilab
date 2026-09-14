"""Task-owned collection, budgets and checkpoints; losses stay in uni_rl/RSL-RL."""

from __future__ import annotations

import inspect
import math
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from sharpa_rl_unilab.algos.hora.teacher import (
    CleanValue,
    TeacherActor,
    TeacherAPPOLearner,
    TeacherFlashActor,
    TeacherFlashLearner,
    frozen_weights,
    observe_new_samples,
    pack_actor,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import (
    CONTRACT_VERSION,
    SharpaTeacherEnv,
    transition_next,
)

from .logging import EpisodeStatistics, TrainingLogger


def tensor_obs(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


def policy_td(obs, device):
    data = tensor_obs({k: obs[k] for k in ("obs", "priv_info", "critic")}, device)
    return TensorDict(
        {"policy": pack_actor(data), "critic": data["critic"]}, batch_size=data["obs"].shape[0]
    )


def config_dict(cfg) -> dict[str, Any]:
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def make_models(cfg, device):
    model = config_dict(cfg.model)
    if cfg.algo.algo == "flashsac":
        params = config_dict(cfg.algo.algo_params)
        for key in (
            "use_compile",
            "use_cuda_graph_actor",
            "use_cuda_graph_critic",
            "use_cuda_graph_actor_packed_staging",
            "use_cuda_graph_critic_packed_staging",
        ):
            if params.get(key, False):
                raise ValueError(
                    "The Sharpa FlashSAC adapter does not support compilation or CUDA graphs"
                )
        if params["n_step"] != 1:
            raise ValueError("Sharpa replay stores one-step transitions")
        learner = TeacherFlashLearner(
            model,
            device=device,
            gamma=float(cfg.algo.gamma),
            tau=float(cfg.algo.tau),
            critic_hidden_dim=int(cfg.algo.critic_hidden_dim),
            num_atoms=int(cfg.algo.num_atoms),
            use_amp=bool(cfg.algo.use_amp),
            **params,
        )
        return learner.actor, learner.critic, learner
    return TeacherActor(model).to(device), CleanValue(model).to(device), None


def algorithm_options(cfg, cls):
    params = config_dict(cfg.algo.algorithm)
    allowed = inspect.signature(cls.__init__).parameters
    if unsupported := params.keys() - allowed.keys():
        raise ValueError(f"Unsupported algorithm options: {sorted(unsupported)}")
    return {k: v for k, v in params.items() if k in allowed}


def collect(env, obs, actor, horizon, device, *, count=None):
    rows = []
    for _ in range(horizon):
        td = policy_td(obs, device)
        with torch.no_grad():
            if isinstance(actor, TeacherFlashActor):
                previous_done = (
                    None
                    if env.state is None
                    else torch.as_tensor(env.state.terminated | env.state.truncated, device=device)
                )
                actions = actor.explore(td["policy"], dones=previous_done)
                logp = torch.zeros(env.num_envs, device=device)
            else:
                actions = actor(td, stochastic_output=True)
                logp = actor.get_output_log_prob(actions)
        sampled_actions = actions.cpu().numpy().copy()
        state = env.step(sampled_actions)
        nxt = transition_next(state)
        row = {key: obs[key].copy() for key in ("obs", "priv_info", "critic")}
        row.update({"next_" + key: nxt[key] for key in ("obs", "priv_info", "critic")})
        row.update(
            actions=sampled_actions,
            actions_log_prob=logp.cpu().numpy().copy(),
            rewards=state.reward.copy(),
            terminated=state.terminated.copy(),
            truncated=state.truncated.copy(),
        )
        if not isinstance(actor, TeacherFlashActor):
            row["behavior_mean"] = actor.output_mean.cpu().numpy().copy()
            row["behavior_std"] = actor.output_std.cpu().numpy().copy()
        rows.append(row)
        obs = state.obs
        if count is not None:
            with count.get_lock():
                count.value += env.num_envs
    return {k: np.stack([row[k] for row in rows]) for k in rows[0]}, obs


def _collector(config, initial_weights, output, weights, stop, count):
    env = None
    try:
        cfg = OmegaConf.create(config)
        assert isinstance(cfg, DictConfig)
        torch.set_num_threads(int(cfg.hardware.torch_threads))
        torch.manual_seed(int(cfg.algo.seed))
        np.random.seed(int(cfg.algo.seed))
        device = str(cfg.hardware.collector_device)
        actor = TeacherActor(config_dict(cfg.model)).to(device).eval()
        actor.load_state_dict(initial_weights)
        env = SharpaTeacherEnv(cfg, int(cfg.algo.num_envs))
        obs, _ = env.reset(seed=int(cfg.algo.seed))
        version = 0
        target = math.ceil(int(cfg.budget.transitions) / env.num_envs) * env.num_envs
        while not stop.is_set() and count.value < target:
            try:
                while True:
                    version, current_weights = weights.get_nowait()
                    actor.load_state_dict(current_weights)
            except queue.Empty:
                pass
            horizon = min(int(cfg.algo.steps_per_env), (target - count.value) // env.num_envs)
            batch, obs = collect(env, obs, actor, horizon, device, count=count)
            collected = count.value
            packet = (batch, {k: v.copy() for k, v in obs.items()}, version, collected)
            while not stop.is_set():
                try:
                    output.put(packet, timeout=0.5)
                    break
                except queue.Full:
                    pass
    except Exception:
        output.put({"error": traceback.format_exc()})
    finally:
        if env is not None:
            env.close()


class AsyncRollouts:
    def __init__(self, cfg, actor):
        ctx = mp.get_context("spawn")
        self.capacity = int(cfg.budget.async_queue_size)
        self.output = ctx.Queue(maxsize=self.capacity)
        self.weights = ctx.Queue(maxsize=1)
        self.stop = ctx.Event()
        self.count = ctx.Value("q", 0)
        self.process = ctx.Process(
            target=_collector,
            args=(
                OmegaConf.to_container(cfg, resolve=True),
                frozen_weights(actor),
                self.output,
                self.weights,
                self.stop,
                self.count,
            ),
            daemon=True,
        )
        self.process.start()

    def receive(self):
        while True:
            try:
                result = self.output.get(timeout=1.0)
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
                return result
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        f"Collector exited without a rollout (exitcode={self.process.exitcode})"
                    )

    def receive_ready(self):
        packets = [self.receive()]
        # Like the native APPO runner, drain the bounded ingress before learning.
        for _ in range(self.capacity):
            try:
                packet = self.output.get_nowait()
                if isinstance(packet, dict):
                    raise RuntimeError(packet["error"])
                packets.append(packet)
            except queue.Empty:
                break
        return packets

    def publish(self, version, actor):
        try:
            self.weights.put_nowait((version, frozen_weights(actor)))
        except queue.Full:
            pass

    def close(self):
        self.stop.set()
        # A multiprocessing Queue's feeder may be waiting for the reader.
        deadline = time.monotonic() + 10
        while self.process.is_alive() and time.monotonic() < deadline:
            try:
                self.output.get(timeout=0.1)
            except queue.Empty:
                pass
            self.process.join(timeout=0.1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        for q in (self.output, self.weights):
            q.cancel_join_thread()
            q.close()


class Replay:
    """Raw fields remain separate in replay, including both privilege vectors."""

    def __init__(self, capacity, device):
        self.capacity, self.device = capacity, device
        self.data, self.size, self.cursor = {}, 0, 0

    def add(self, batch):
        data = {k: v.flatten(0, 1) for k, v in batch.items() if k != "actions_log_prob"}
        n = data["obs"].shape[0]
        if n > self.capacity:
            data = {k: v[-self.capacity :] for k, v in data.items()}
            n = self.capacity
        ids = (torch.arange(n, device=self.device) + self.cursor) % self.capacity
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = torch.empty(
                    (self.capacity, *value.shape[1:]), device=self.device, dtype=value.dtype
                )
            self.data[key][ids] = value
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, size):
        ids = torch.randint(self.size, (size,), device=self.device)
        batch = {key: value[ids] for key, value in self.data.items()}
        batch["obs"] = torch.cat((batch["obs"], batch["priv_info"]), -1)
        batch["next_obs"] = torch.cat((batch["next_obs"], batch["next_priv_info"]), -1)
        batch["dones"] = (batch["terminated"].bool() | batch["truncated"].bool()).float()
        batch["truncated"] = batch["truncated"] * (1 - batch["terminated"])
        return batch


def optimizer_metadata(learner):
    result = {}
    for name in ("optimizer", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        optimizer = getattr(learner, name, None)
        if optimizer is not None:
            result[name] = [
                {k: v for k, v in group.items() if k != "params"}
                for group in optimizer.param_groups
            ]
    return result


def save_teacher(path, cfg, actor, critic, learner, counters, elapsed):
    snapshot = {
        "contract": CONTRACT_VERSION,
        "stage": "teacher",
        "algorithm": str(cfg.algo.algo),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "actor": frozen_weights(actor),
        "critic": frozen_weights(critic),
        "counters": dict(counters),
        "wall_seconds": elapsed,
        "optimizer_parameters": optimizer_metadata(learner),
    }
    snapshot["parameter_counts"] = {
        "actor": sum(p.numel() for p in actor.parameters()),
        "priv_encoder": sum(p.numel() for p in actor.shared.priv_encoder.parameters()),
        "critic": sum(p.numel() for p in critic.parameters()),
    }
    if isinstance(learner, TeacherFlashLearner):
        snapshot["learner"] = learner.get_state_dict()
    else:
        snapshot["optimizer"] = learner.optimizer.state_dict()
        if isinstance(learner, TeacherAPPOLearner):
            snapshot["target_actor"] = frozen_weights(learner.target_actor)
    torch.save(snapshot, path)
    return snapshot


def load_policy(path, device="cpu", *, stage=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("contract") != CONTRACT_VERSION:
        raise ValueError(
            f"Checkpoint must use {CONTRACT_VERSION}; old flat/shared-HORA/FlashSAC weights require retraining"
        )
    if stage is not None and checkpoint.get("stage") != stage:
        raise ValueError(f"Expected a {stage} checkpoint")
    if checkpoint.get("stage") not in ("teacher", "student") or checkpoint.get("algorithm") not in (
        "ppo",
        "appo",
        "flashsac",
    ):
        raise ValueError("Unknown checkpoint stage or algorithm")
    cfg = OmegaConf.create(checkpoint["config"])
    assert isinstance(cfg, DictConfig)
    model = config_dict(cfg.model)
    student = checkpoint["stage"] == "student"
    if checkpoint["algorithm"] == "flashsac":
        actor = TeacherFlashActor(
            model,
            student=student,
            noise_zeta_mu=float(cfg.algo.algo_params.actor_noise_zeta_mu),
            noise_zeta_max=int(cfg.algo.algo_params.actor_noise_zeta_max),
        ).to(device)
    else:
        actor = TeacherActor(model, student=student).to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    return actor, cfg, checkpoint


@torch.no_grad()
def timeout_rewards(batch, critic, gamma):
    clean = batch["next_critic"]
    values = critic(
        TensorDict({"critic": clean.flatten(0, 1)}, batch_size=clean.shape[0] * clean.shape[1])
    ).view(*clean.shape[:2])
    pure_timeout = batch["truncated"] * (1 - batch["terminated"])
    return batch["rewards"] + gamma * values * pure_timeout


def stage_rollout(stages, raw, last_obs, version, actor, critic, device):
    batch = tensor_obs(raw, device)
    packed = pack_actor(batch)
    observe_new_samples(actor, critic, packed, batch["critic"])
    if stages and stages[0]["observations"].shape[0] != packed.shape[0]:
        stages.clear()
    stages.append(
        dict(
            observations=packed,
            critic=batch["critic"],
            rewards=batch["rewards"],
            next_critic=batch["next_critic"],
            truncated=batch["truncated"],
            terminated=batch["terminated"],
            dones=(batch["terminated"].bool() | batch["truncated"].bool()).float(),
            actions=batch["actions"],
            actions_log_prob=batch["actions_log_prob"],
            behavior_version=torch.full_like(batch["rewards"], version),
            last_obs=policy_td(last_obs, device)["policy"],
            last_critic=tensor_obs(last_obs, device)["critic"],
        )
    )
    return packed.shape[0] * packed.shape[1]


def train_teacher(cfg):
    from .evaluation import evaluate_checkpoint, write_run_metadata

    started = time.monotonic()
    if cfg.budget.checkpoint is not None:
        raise ValueError(
            "budget.checkpoint selects evaluation weights; training resume is not supported"
        )
    torch.set_num_threads(int(cfg.hardware.torch_threads))
    torch.manual_seed(int(cfg.algo.seed))
    np.random.seed(int(cfg.algo.seed))
    device = str(cfg.training.get("device") or cfg.hardware.device)
    cfg.hardware.device = device
    if cfg.training.get("devices"):
        raise ValueError("The fixed-hardware comparison runner currently uses one learner device")
    n = int(cfg.algo.num_envs)
    target = math.ceil(int(cfg.budget.transitions) / n) * n
    if target <= 0:
        raise ValueError("budget.transitions must be positive")
    algo = str(cfg.algo.algo)
    run = Path(cfg.training.log_dir or f"logs/{algo}/seed_{cfg.algo.seed}_{time.time_ns()}")
    run.mkdir(parents=True, exist_ok=False)
    write_run_metadata(run, cfg)
    actor, critic, learner = make_models(cfg, device)
    env = asynchronous = None
    obs = {}
    packets = []
    counters = dict(
        collected=0,
        received=0,
        training_samples=0,
        optimizer_updates=0,
        actor_updates=0,
        critic_updates=0,
        temperature_updates=0,
        policy_version=0,
    )
    stages = deque(maxlen=int(cfg.budget.async_queue_size))
    replay = (
        Replay(int(cfg.algo.get("replay_buffer_n", 1280)) * n, device)
        if algo == "flashsac"
        else None
    )
    if algo == "appo":
        learner = TeacherAPPOLearner(
            actor=cast(
                Any, actor
            ),  # Native annotations require MLPModel; adapters implement its interface.
            critic=cast(Any, critic),
            device=device,
            **algorithm_options(cfg, TeacherAPPOLearner.__mro__[1]),
        )
        asynchronous = AsyncRollouts(cfg, actor)
    else:
        env = SharpaTeacherEnv(cfg, n)
        obs, _ = env.reset(seed=int(cfg.algo.seed))
    save_due, eval_due, log_due = (
        int(cfg.budget[k]) for k in ("save_every", "evaluate_every", "log_every")
    )
    next_save, next_eval, next_log = save_due, eval_due, log_due
    episodes = EpisodeStatistics(n)
    logger = None
    try:
        logger = TrainingLogger(run, cfg, target)
        logger.start(status="Waiting for first rollout...")
        while counters["received"] < target:
            if asynchronous is not None:
                packets = asynchronous.receive_ready()
                raw, last_obs, behavior_version, _ = packets[0]
                counters["collected"] = asynchronous.count.value
            else:
                horizon = int(
                    cfg.algo.get("num_steps_per_env", cfg.training.get("env_steps_per_sync", 2))
                )
                horizon = min(horizon, (target - counters["received"]) // n)
                raw, last_obs = collect(env, obs, actor, horizon, device)
                obs = last_obs
                behavior_version = counters["policy_version"]
                counters["collected"] += horizon * n
            fresh_rollouts = [packet[0] for packet in packets] if asynchronous is not None else [raw]
            for fresh in fresh_rollouts:
                episodes.update(fresh["rewards"], fresh["terminated"], fresh["truncated"])
            batch = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in raw.items()
            }
            packed = torch.cat((batch["obs"], batch["priv_info"]), -1)
            t = packed.shape[0]
            if asynchronous is None:
                counters["received"] += t * n
            dones = (batch["terminated"].bool() | batch["truncated"].bool()).float()
            if algo == "flashsac":
                assert isinstance(learner, TeacherFlashLearner) and replay is not None
                observe_new_samples(actor, critic, packed, batch["critic"])
                replay.add(batch)
                learner.update_reward_stats(batch["rewards"], dones)
                metrics = {}
                if counters["received"] >= int(cfg.algo.learning_starts) * n:
                    for index in range(int(cfg.algo.updates_per_step)):
                        sampled = replay.sample(int(cfg.algo.batch_size))
                        metrics.update(learner.update_critic(sampled))
                        counters["optimizer_updates"] += 1
                        counters["critic_updates"] += 1
                        if index % int(cfg.algo.policy_frequency) == 0:
                            metrics.update(learner.update_actor(sampled))
                            counters["optimizer_updates"] += 2
                            counters["actor_updates"] += 1
                            counters["temperature_updates"] += 1
                            counters["training_samples"] += int(cfg.algo.batch_size)
                        learner.soft_update_target()
                        counters["training_samples"] += int(cfg.algo.batch_size)
                    metrics.update(learner.saturation_metrics)
            else:
                if algo == "appo":
                    assert isinstance(learner, TeacherAPPOLearner)
                    for raw, last_obs, version, packet_end in packets:
                        size = raw["obs"].shape[0] * n
                        if packet_end != counters["received"] + size:
                            raise RuntimeError(
                                "Collector packet gap or duplicate; refusing to recount normalization samples"
                            )
                        counters["received"] += stage_rollout(
                            stages, raw, last_obs, version, actor, critic, device
                        )
                    learner.sync_target_actor_buffers()
                    # Each staged rollout has its own bootstrap; concatenate on
                    # the environment axis so unrelated trajectories never join.
                    combined = {
                        k: torch.cat(
                            [stage[k] for stage in stages], dim=0 if k.startswith("last_") else 1
                        )
                        for k in stages[0]
                    }
                    combined["rewards"] = timeout_rewards(
                        combined, critic, float(cfg.algo.algorithm.gamma)
                    )
                    learner.process_batch(combined)
                    metrics = learner.update(combined)
                    updates = int(metrics["appo/updates_executed"])
                    batch_size = (
                        combined["observations"].shape[0] * combined["observations"].shape[1]
                    )
                    counters["training_samples"] += (
                        batch_size // int(cfg.algo.algorithm.num_mini_batches) * updates
                    )
                    lag = counters["policy_version"] - combined["behavior_version"]
                    metrics["policy_lag_mean"] = float(lag.mean())
                    metrics["policy_lag_max"] = float(lag.max())
                    metrics["staging_rollouts"] = len(stages)
                    counters["optimizer_updates"] += updates
                    counters["actor_updates"] += updates
                    counters["critic_updates"] += updates
                else:
                    corrected_reward = timeout_rewards(
                        batch, critic, float(cfg.algo.algorithm.gamma)
                    )
                    td = TensorDict(
                        {"policy": packed[0], "critic": batch["critic"][0]}, batch_size=n
                    )
                    storage = RolloutStorage("rl", n, t, td, [22], device)
                    if learner is None:
                        learner = PPO(
                            cast(Any, actor),
                            cast(Any, critic),
                            storage,
                            device=device,
                            **algorithm_options(cfg, PPO),
                        )
                    assert isinstance(learner, PPO)
                    learner.storage = storage
                    # Preserve the log-prob generated by the collecting policy.
                    for step in range(t):
                        tr = RolloutStorage.Transition()
                        tr.observations = TensorDict(
                            {"policy": packed[step], "critic": batch["critic"][step]}, batch_size=n
                        )
                        with torch.no_grad():
                            tr.values = critic(tr.observations)
                        tr.actions = batch["actions"][step]
                        tr.actions_log_prob = batch["actions_log_prob"][step]
                        tr.distribution_params = (
                            batch["behavior_mean"][step],
                            batch["behavior_std"][step],
                        )
                        tr.rewards, tr.dones = corrected_reward[step], dones[step]
                        storage.add_transition(tr)
                    with torch.no_grad():
                        learner.compute_returns(policy_td(last_obs, device))
                    # Values/returns use the collecting critic's frozen statistics.
                    # The update may change normalization, but behavior density
                    # and old values remain exactly those of collection.
                    observe_new_samples(actor, critic, packed, batch["critic"])
                    metrics = learner.update()
                    updates = int(cfg.algo.algorithm.num_learning_epochs) * int(
                        cfg.algo.algorithm.num_mini_batches
                    )
                    counters["optimizer_updates"] += updates
                    counters["actor_updates"] += updates
                    counters["critic_updates"] += updates
                    counters["training_samples"] += (
                        t * n // int(cfg.algo.algorithm.num_mini_batches)
                    ) * updates
            counters["policy_version"] += 1
            if asynchronous is not None:
                asynchronous.publish(counters["policy_version"], actor)
                counters["collected"] = asynchronous.count.value
            metrics.update(counters)
            metrics["policy_lag"] = counters["policy_version"] - behavior_version - 1
            metrics["wall_seconds"] = time.monotonic() - started
            metrics["reuse_ratio"] = counters["training_samples"] / counters["received"]
            if counters["received"] >= next_log or counters["received"] == target:
                metrics.update(episodes.metrics())
                logger.log(metrics)
                next_log = counters["received"] + max(log_due, 1)
            do_save = save_due > 0 and counters["collected"] >= next_save
            do_eval = eval_due > 0 and counters["collected"] >= next_eval
            if do_save or do_eval:
                path = run / f"teacher_{counters['collected']}.pt"
                save_teacher(
                    path, cfg, actor, critic, learner, counters, time.monotonic() - started
                )
                logger.log_save(str(path))
                if do_eval:
                    logger.status(f"Evaluating {path.name}...")
                    evaluate_checkpoint(
                        path, output=run / f"evaluation_{counters['collected']}.json", device=device
                    )
                    logger.status("Training")
                next_save = (counters["collected"] // max(save_due, 1) + 1) * max(save_due, 1)
                if do_eval:
                    next_eval = (counters["collected"] // max(eval_due, 1) + 1) * max(eval_due, 1)
        final = run / "teacher_final.pt"
        save_teacher(final, cfg, actor, critic, learner, counters, time.monotonic() - started)
        logger.log_save(str(final))
        if eval_due > 0:
            logger.status("Evaluating final checkpoint...")
            evaluate_checkpoint(final, output=run / "evaluation_final.json", device=device)
        logger.finish()
        return final
    finally:
        try:
            if logger is not None:
                logger.close()
        finally:
            if asynchronous is not None:
                asynchronous.close()
            if env is not None:
                env.close()
