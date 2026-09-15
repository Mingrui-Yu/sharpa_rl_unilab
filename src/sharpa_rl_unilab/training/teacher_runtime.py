"""Task-owned collection, budgets and checkpoints; losses stay in uni_rl/RSL-RL."""

from __future__ import annotations

import inspect
import math
import multiprocessing as mp
import queue
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from uni_rl.algos.appo.staging import RolloutStagingPool
from uni_rl.algos.appo.worker import compute_rollout_active_steps_per_sec
from uni_rl.algos.common.collector_timing import extract_env_step_breakdown_timing_ms

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

from .configuration import configure_threads, migrate_checkpoint_config, resolve_device
from .logging import EpisodeStatistics, TrainingLogger, write_run_metadata


def tensor_obs(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


def policy_td(obs, device):
    data = tensor_obs({k: obs[k] for k in ("obs", "priv_info", "critic")}, device)
    return TensorDict(
        {"policy": pack_actor(data), "critic": data["critic"]}, batch_size=data["obs"].shape[0]
    )


def config_dict(cfg) -> dict[str, Any]:
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def training_budget(cfg):
    """Exactly one stopping budget; all teacher snapshots use update rounds."""
    iterations = cfg.algo.get("max_iterations")
    transitions = cfg.training.max_transitions
    if cfg.algo.algo == "flashsac" and (iterations is None or transitions is not None):
        raise ValueError(
            "FlashSAC requires algo.max_iterations > 0 and training.max_transitions=null"
        )
    if (iterations is None) == (transitions is None):
        raise ValueError(
            "Select exactly one budget: algo.max_iterations or training.max_transitions; "
            "set the other to null"
        )
    if cfg.algo.algo == "appo":
        if (
            min(
                int(cfg.algo.async_queue_size),
                int(cfg.algo.staging_pool_size),
                int(cfg.algo.steps_per_env),
            )
            < 1
        ):
            raise ValueError(
                "APPO queue capacity, staging capacity and rollout length must be positive"
            )
    if int(cfg.algo.save_interval) < 0:
        raise ValueError("algo.save_interval must be nonnegative")
    n = int(cfg.algo.num_envs)
    target = int(iterations) if iterations is not None else math.ceil(int(transitions) / n) * n
    if target <= 0:
        raise ValueError("Training budget must be positive")
    return ("policy_version" if iterations is not None else "received"), target


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


def collect(env, obs, actor, horizon, device, *, count=None, metrics=None):
    rows = []
    rollout_started = time.perf_counter()
    totals = defaultdict(float)
    for _ in range(horizon):
        inference_started = time.perf_counter()
        td = policy_td(obs, device)
        with torch.no_grad():
            actions = actor(td, stochastic_output=True)
            logp = actor.get_output_log_prob(actions)
        sampled_actions = actions.cpu().numpy().copy()
        inference_seconds = time.perf_counter() - inference_started
        step_started = time.perf_counter()
        state = env.step(sampled_actions)
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
            actions_log_prob=logp.cpu().numpy().copy(),
            rewards=state.reward.copy(),
            terminated=state.terminated.copy(),
            truncated=state.truncated.copy(),
        )
        row["behavior_mean"] = actor.output_mean.cpu().numpy().copy()
        row["behavior_std"] = actor.output_std.cpu().numpy().copy()
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


def _collector(config, initial_weights, output, weights, stop, count):
    env = None
    try:
        cfg = OmegaConf.create(config)
        assert isinstance(cfg, DictConfig)
        configure_threads(cfg, role="collector")
        seed = int(cfg.algo.collector_seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = str(cfg.algo.collector_device)
        actor = TeacherActor(config_dict(cfg.model)).to(device).eval()
        actor.load_state_dict(initial_weights)
        env = SharpaTeacherEnv(cfg, int(cfg.algo.num_envs))
        obs, _ = env.reset(seed=seed)
        version = 0
        target = (
            math.ceil(int(cfg.training.max_transitions) / env.num_envs) * env.num_envs
            if cfg.training.max_transitions is not None
            else None
        )
        while not stop.is_set() and (target is None or count.value < target):
            current_weights = None
            try:
                while True:
                    version, current_weights = weights.get_nowait()
            except queue.Empty:
                pass
            # Install weights and RMS buffers together, only at a rollout boundary.
            if current_weights is not None:
                actor.load_state_dict(current_weights)
            horizon = int(cfg.algo.steps_per_env)
            if target is not None:
                horizon = min(horizon, (target - count.value) // env.num_envs)
            collector_metrics = {}
            batch, obs = collect(
                env, obs, actor, horizon, device, count=count, metrics=collector_metrics
            )
            collector_metrics.update(
                {
                    "runtime/collector_num_threads": torch.get_num_threads(),
                    "runtime/collector_num_interop_threads": torch.get_num_interop_threads(),
                }
            )
            collected = count.value
            packet = (
                batch,
                {k: v.copy() for k, v in obs.items()},
                version,
                collected,
                collector_metrics,
            )
            while not stop.is_set():
                try:
                    output.put(packet, timeout=0.5)
                    break
                except queue.Full:
                    pass
    except Exception:
        error = {"error": traceback.format_exc()}
        while not stop.is_set():
            try:
                output.put(error, timeout=0.5)
                break
            except queue.Full:
                pass
    finally:
        if env is not None:
            env.close()


class AsyncRollouts:
    def __init__(self, cfg, actor):
        ctx = mp.get_context("spawn")
        self.capacity = int(cfg.algo.async_queue_size)
        # Four pending packets plus at most one rollout being collected/put.
        # Unlike main's overwrite ring, a full queue applies backpressure.
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
        from uni_rl.offpolicy.thread_budget import resolve_torch_thread_runtime, torch_thread_env

        runtime = resolve_torch_thread_runtime(cfg.training.torch_threads)
        with torch_thread_env(runtime, role="collector"):
            self.process.start()
        self.closed = False

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
        for _ in range(self.capacity - 1):
            try:
                packet = self.output.get_nowait()
                if isinstance(packet, dict):
                    raise RuntimeError(packet["error"])
                packets.append(packet)
            except queue.Empty:
                break
        return packets

    def publish(self, version, actor):
        snapshot = (version, frozen_weights(actor))
        while True:
            try:
                self.weights.put_nowait(snapshot)
                return
            except queue.Full:
                # Queue.put uses a feeder thread: Full can precede readability.
                # Retry after either we or the collector remove the stale state.
                try:
                    self.weights.get(timeout=0.01)
                except queue.Empty:
                    pass

    def close(self):
        if self.closed:
            return
        self.closed = True
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


def load_policy(path, device: str | None = "cpu", *, stage=None, configure_runtime=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    cfg = migrate_checkpoint_config(checkpoint["config"])
    device = resolve_device(device if device is not None else cfg.training.device)
    if configure_runtime:
        configure_threads(cfg)
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


def stage_rollout(stages, raw, last_obs, version, actor, critic, device, capacity=8):
    """Stage raw inputs once; reused pool views never update statistics."""
    fields = {
        key: value
        for key, value in raw.items()
        if key
        in (
            "critic",
            "rewards",
            "next_critic",
            "truncated",
            "terminated",
            "actions",
            "actions_log_prob",
        )
    }
    fields.update(
        observations=np.concatenate((raw["obs"], raw["priv_info"]), axis=-1),
        dones=raw["terminated"].astype(bool) | raw["truncated"].astype(bool),
        behavior_version=np.full_like(raw["rewards"], version),
    )
    fields = {
        key: value.swapaxes(0, 1).astype(np.float32, copy=False) for key, value in fields.items()
    }
    fields.update(
        last_obs=np.concatenate((last_obs["obs"], last_obs["priv_info"]), axis=-1),
        last_critic=last_obs["critic"],
    )
    n, t = fields["observations"].shape[:2]
    # A final short rollout replaces old full-length stages, as before.
    if stages is None or stages.batch()["observations"].shape[0] != t:
        stages = RolloutStagingPool(
            capacity=capacity,
            num_envs=n,
            slot_shapes={k: v.shape for k, v in fields.items()},
            device=device,
        )
    slot = stages.stage_numpy_views(fields)
    batch = stages.batch()
    rows = slice(slot * n, (slot + 1) * n)
    observe_new_samples(actor, critic, batch["observations"][:, rows], batch["critic"][:, rows])
    return stages, t * n


def train_teacher(cfg):
    started = time.monotonic()
    if cfg.algo.checkpoint is not None:
        raise ValueError(
            "algo.checkpoint selects evaluation weights; training resume is not supported"
        )
    progress_key, target = training_budget(cfg)
    if cfg.algo.algo == "flashsac":
        from .flashsac_runtime import train_flashsac

        return train_flashsac(cfg)
    configure_threads(cfg)
    torch.manual_seed(int(cfg.algo.seed))
    np.random.seed(int(cfg.algo.seed))
    device = resolve_device(cfg.training.device)
    cfg.training.device = device
    if cfg.algo.algo == "appo":
        cfg.algo.collector_device = resolve_device(cfg.algo.collector_device or device)
        if cfg.algo.collector_seed is None:
            cfg.algo.collector_seed = int(cfg.algo.seed) + 1
    if cfg.training.get("devices"):
        raise ValueError("The fixed-hardware comparison runner currently uses one learner device")
    n = int(cfg.algo.num_envs)
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
    stages = None
    with ExitStack() as resources:
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
            resources.callback(asynchronous.close)
        else:
            env = SharpaTeacherEnv(cfg, n)
            resources.callback(env.close)
            obs, _ = env.reset(seed=int(cfg.algo.seed))
        save_interval = int(cfg.algo.save_interval)
        next_save = save_interval
        episodes = EpisodeStatistics(n)
        logger = TrainingLogger(run, cfg, target)
        resources.callback(logger.close)
        logger.start(status="Waiting for first rollout...")
        reward_sums = defaultdict(float)
        reward_steps = 0
        while counters[progress_key] < target:
            iteration_started = time.perf_counter()
            timings = {}
            if asynchronous is not None:
                wait_started = time.perf_counter()
                packets = asynchronous.receive_ready()
                timings["collector_wait_time"] = time.perf_counter() - wait_started
                raw, last_obs, behavior_version, _, _ = packets[0]
                collector_metrics = packets[-1][4]
                counters["collected"] = asynchronous.count.value
            else:
                horizon = int(cfg.algo.num_steps_per_env)
                if progress_key == "received":
                    horizon = min(horizon, (target - counters["received"]) // n)
                collector_metrics = {}
                raw, last_obs = collect(env, obs, actor, horizon, device, metrics=collector_metrics)
                obs = last_obs
                behavior_version = counters["policy_version"]
                counters["collected"] += horizon * n
            fresh_rollouts = (
                [(packet[0], packet[4]) for packet in packets]
                if asynchronous is not None
                else [(raw, collector_metrics)]
            )
            for fresh, fresh_metrics in fresh_rollouts:
                episodes.update(fresh["rewards"], fresh["terminated"], fresh["truncated"])
                size = fresh["rewards"].size
                reward_steps += size
                for key, value in fresh_metrics.items():
                    if key.startswith("reward/"):
                        reward_sums[key] += value * size
            if asynchronous is not None:
                assert isinstance(learner, TeacherAPPOLearner)
                stage_started = time.perf_counter()
                for raw, last_obs, version, packet_end, _ in packets:
                    size = raw["obs"].shape[0] * n
                    if packet_end != counters["received"] + size:
                        raise RuntimeError(
                            "Collector packet gap or duplicate; refusing to recount normalization samples"
                        )
                    stages, size = stage_rollout(
                        stages,
                        raw,
                        last_obs,
                        version,
                        actor,
                        critic,
                        device,
                        capacity=int(cfg.algo.staging_pool_size),
                    )
                    counters["received"] += size
                timings["learner_replay_stage_time"] = time.perf_counter() - stage_started
                learner.sync_target_actor_buffers()
                assert stages is not None
                sample_started = time.perf_counter()
                combined = stages.batch()
                timings["learner_replay_sample_time"] = time.perf_counter() - sample_started
                train_started = time.perf_counter()
                combined["rewards"] = timeout_rewards(
                    combined, critic, float(cfg.algo.algorithm.gamma)
                )
                learner.process_batch(combined)
                metrics = learner.update(combined)
                timings["train_time"] = time.perf_counter() - train_started
                updates = int(metrics["appo/updates_executed"])
                batch_size = combined["observations"].shape[0] * combined["observations"].shape[1]
                counters["training_samples"] += (
                    batch_size // int(cfg.algo.algorithm.num_mini_batches) * updates
                )
                lag = counters["policy_version"] - combined["behavior_version"]
                metrics["policy_lag_mean"] = float(lag.mean())
                metrics["policy_lag_max"] = float(lag.max())
                metrics["staging_rollouts"] = stages.active_count
                metrics["staging_pool_capacity"] = stages.capacity
                metrics["rollouts_read"] = len(packets)
                counters["optimizer_updates"] += updates
                counters["actor_updates"] += updates
                counters["critic_updates"] += updates
            else:
                batch = tensor_obs(raw, device)
                packed = pack_actor(batch)
                t = packed.shape[0]
                counters["received"] += t * n
                dones = (batch["terminated"].bool() | batch["truncated"].bool()).float()
                corrected_reward = timeout_rewards(batch, critic, float(cfg.algo.algorithm.gamma))
                td = TensorDict({"policy": packed[0], "critic": batch["critic"][0]}, batch_size=n)
                storage = (
                    learner.storage
                    if isinstance(learner, PPO) and learner.storage.num_transitions_per_env == t
                    else RolloutStorage("rl", n, t, td, [22], device)
                )
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
                metrics = cast(PPO, learner).update()
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
            finished = counters[progress_key] >= target
            if asynchronous is not None:
                publish_started = time.perf_counter()
                asynchronous.publish(counters["policy_version"], actor)
                timings["weight_sync_time"] = time.perf_counter() - publish_started
                counters["collected"] = asynchronous.count.value
                timings["iteration_time"] = time.perf_counter() - iteration_started
                if finished:
                    # Stop and drain before recording final sampling counts: an
                    # in-flight rollout may finish without entering the learner.
                    asynchronous.close()
                    counters["collected"] = asynchronous.count.value
            metrics.update(counters)
            metrics["policy_lag"] = counters["policy_version"] - behavior_version - 1
            metrics["wall_seconds"] = time.monotonic() - started
            metrics["reuse_ratio"] = counters["training_samples"] / counters["received"]
            # One log per outer loop; counters remain the true sample/update work.
            metrics.update(collector_metrics)
            metrics.update({key: value / reward_steps for key, value in reward_sums.items()})
            metrics.update(episodes.metrics())
            logger.log(metrics, timings=timings if asynchronous is not None else None)
            reward_sums.clear()
            reward_steps = 0
            iteration = counters["policy_version"]
            if save_interval > 0 and iteration >= next_save:
                path = run / f"teacher_iteration_{iteration}.pt"
                save_teacher(
                    path, cfg, actor, critic, learner, counters, time.monotonic() - started
                )
                logger.log_save(str(path))
                next_save = iteration + save_interval
        final = run / "teacher_final.pt"
        save_teacher(final, cfg, actor, critic, learner, counters, time.monotonic() - started)
        logger.log_save(str(final))
        logger.finish()
        return final
