"""APPO staging, target-policy updates and asynchronous training lifecycle."""

import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, cast

import numpy as np
from uni_rl.algos.appo.learner import APPOLearner
from uni_rl.algos.appo.staging import RolloutStagingPool

from sharpa_rl_unilab.algos.hora.on_policy import TeacherAPPOLearner

from .appo_collector import AsyncRollouts
from .checkpoints import save_teacher
from .configuration import algorithm_options
from .logging import EpisodeStatistics, TrainingLogger, create_run, training_counters
from .policy import make_models, observe_new_samples
from .rollouts import timeout_rewards


def stage_rollout(stages, raw, last_obs, version, actor, critic, device, capacity=8):
    """Stage raw inputs once; reused pool views never update statistics."""
    fields = {
        key: raw[key]
        for key in (
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


def train_appo(cfg, progress_key, target, started):
    device = cfg.training.device
    n = int(cfg.algo.num_envs)
    run = create_run(cfg)
    actor, critic, _ = make_models(cfg, device)
    counters = training_counters()
    stages = None
    with ExitStack() as resources:
        # Native annotations require MLPModel; these adapters implement its interface.
        learner = TeacherAPPOLearner(
            actor=cast(Any, actor),
            critic=cast(Any, critic),
            device=device,
            **algorithm_options(cfg, APPOLearner, extra_options=("kl_mode",)),
        )
        collector = AsyncRollouts(cfg, actor)
        resources.callback(collector.close)
        episodes = EpisodeStatistics(n)
        logger = TrainingLogger(run, cfg, target)
        resources.callback(logger.close)
        logger.start(status="Waiting for first rollout...")
        save_interval = int(cfg.algo.save_interval)
        while counters[progress_key] < target:
            iteration_started = time.perf_counter()
            wait_started = time.perf_counter()
            packets = collector.receive_ready()
            timings = {"collector_wait_time": time.perf_counter() - wait_started}
            counters["collected"] = collector.count.value
            reward_sums = defaultdict(float)
            reward_steps = 0
            for raw, _, _, _, fresh_metrics in packets:
                episodes.update(raw["rewards"], raw["terminated"], raw["truncated"])
                size = raw["rewards"].size
                reward_steps += size
                for key, value in fresh_metrics.items():
                    if key.startswith("reward/"):
                        reward_sums[key] += value * size
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
            combined["rewards"] = timeout_rewards(combined, critic, float(cfg.algo.algorithm.gamma))
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
            counters["policy_version"] += 1
            publish_started = time.perf_counter()
            collector.publish(counters["policy_version"], actor)
            timings["weight_sync_time"] = time.perf_counter() - publish_started
            counters["collected"] = collector.count.value
            timings["iteration_time"] = time.perf_counter() - iteration_started
            if counters[progress_key] >= target:
                # Drain before final counters; a last rollout may not reach the learner.
                collector.close()
                counters["collected"] = collector.count.value
            metrics.update(counters)
            metrics.update(packets[-1][4])
            metrics.update({key: value / reward_steps for key, value in reward_sums.items()})
            metrics.update(episodes.metrics())
            metrics.update(
                policy_lag=counters["policy_version"] - packets[0][2] - 1,
                wall_seconds=time.monotonic() - started,
                reuse_ratio=counters["training_samples"] / counters["received"],
            )
            logger.log(metrics, timings=timings)
            iteration = counters["policy_version"]
            if save_interval > 0 and iteration % save_interval == 0:
                path = run / f"teacher_iteration_{iteration}.pt"
                save_teacher(
                    path, cfg, actor, critic, learner, counters, time.monotonic() - started
                )
                logger.log_save(str(path))
        final = run / "teacher_final.pt"
        save_teacher(final, cfg, actor, critic, learner, counters, time.monotonic() - started)
        logger.log_save(str(final))
        logger.finish()
        return final
