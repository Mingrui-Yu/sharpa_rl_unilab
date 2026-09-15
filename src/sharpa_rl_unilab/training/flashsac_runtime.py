"""HORA observation/checkpoint adapters around UniLab's native FlashSAC runner."""

from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from uni_rl.offpolicy.thread_budget import apply_torch_thread_runtime, resolve_torch_thread_runtime
from unilab.base.np_env import NpEnvState

from sharpa_rl_unilab.algos.hora.teacher import ACTOR_DIM, CRITIC_DIM, PRIV_DIM, observe_new_samples
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv

from .evaluation import write_json, write_run_metadata
from .teacher_runtime import make_models, resolve_device, save_teacher


def transport_obs(obs):
    """Pack current privilege with actor history; keep clean Q inputs separate."""
    return {"obs": np.concatenate((obs["obs"], obs["priv_info"]), axis=-1), "critic": obs["critic"]}


class FlashTeacherEnv:
    def __init__(self, num_envs, env_cfg_override, *, config):
        if env_cfg_override:
            raise ValueError("FlashTeacherEnv takes task overrides through its composed config")
        cfg = OmegaConf.create(config)
        assert isinstance(cfg, DictConfig)
        self.teacher = SharpaTeacherEnv(cfg, num_envs)
        self.state = None

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    @property
    def obs_groups_spec(self):
        return {"obs": ACTOR_DIM + PRIV_DIM, "critic": CRITIC_DIM}

    def _state(self, state):
        self.state = NpEnvState(
            transport_obs(state.obs),
            state.reward,
            state.terminated,
            state.truncated,
            state.info,
            None if state.final_observation is None else transport_obs(state.final_observation),
        )
        return self.state

    def init_state(self):
        return self._state(self.teacher.init_state())

    def reset(self, env_indices=None, *, seed=None):
        obs, info = self.teacher.reset(env_indices, seed=seed)
        self._state(self.teacher.state)
        return transport_obs(obs), info

    def step(self, actions):
        return self._state(self.teacher.step(actions))

    def close(self):
        self.teacher.close()


class TeacherDoubleBufferRunner(DoubleBufferOffPolicyRunner):
    """Keep native collection, inference scheduling, replay and optimizer loops.

    Only fresh committed rows update HORA statistics. Inference and learning
    share the very same actor, including its normalization buffers.
    """

    def __init__(self, *, cfg, run, started, **kwargs):
        super().__init__(**kwargs)
        self.cfg, self.run, self.started = cfg, run, started
        self.received = 0
        self.collected = 0
        self.runtime_manifest.update(
            teacher_contract=str(cfg.protocol.version),
            stopping_budget="update_rounds",
            use_amp=bool(cfg.algo.use_amp),
            learner_torch_threads=torch.get_num_threads(),
            learner_torch_interop_threads=torch.get_num_interop_threads(),
            torch_thread_runtime=self.torch_thread_runtime,
        )

    def _update_reward_stats_from_replay(
        self, replay_buffer, start_ptr, end_ptr, replay_source=None
    ):
        # Read all fields in one stable snapshot, even when reward normalization
        # is disabled. The native reward hook and warmup hook share this cursor.
        assert replay_source is not None
        end, fields = replay_source.read_committed_fields(
            ("obs", "critic", "rewards", "dones"),
            start_ptr=self.received,
        )
        if end - self.received > replay_buffer.capacity:
            raise RuntimeError("Fresh HORA observations were overwritten before normalization")
        if end > self.received:
            observe_new_samples(
                self.learner.actor, self.learner.critic, fields["obs"], fields["critic"]
            )
            self.learner.update_reward_stats(
                fields["rewards"].view(-1, self.num_envs),
                fields["dones"].view(-1, self.num_envs),
            )
            self.received = end
        return end

    def _wait_for_inference_request(self, *args, **kwargs):
        tick = super()._wait_for_inference_request(*args, **kwargs)
        # Warmup can span many ticks before the first optimizer update.
        self._update_reward_stats_from_replay(
            kwargs["replay_buffer"],
            self.received,
            0,
            kwargs["replay_pipeline"],
        )
        return tick

    def counters(self):
        rounds = int(self.learner.update_count)
        critic = rounds * self.updates_per_step
        actor = rounds * math.ceil(self.updates_per_step / self.policy_frequency)
        return dict(
            collected=self.collected,
            received=self.received,
            policy_version=rounds,
            critic_updates=critic,
            actor_updates=actor,
            temperature_updates=actor,
            optimizer_updates=critic + 2 * actor,
            training_samples=(critic + actor) * self.batch_size,
        )

    def save(self, path):
        save_teacher(
            path,
            self.cfg,
            self.learner.actor,
            self.learner.critic,
            self.learner,
            self.counters(),
            time.monotonic() - self.started,
        )
        return str(path)

    def _aggregate_log_statistics(self, logger, **kwargs):
        self.collected = int(logger._total_steps)
        payload = super()._aggregate_log_statistics(logger, **kwargs)
        payload["metrics"].update(self.learner.saturation_metrics)
        row = {
            **payload["metrics"],
            **self.counters(),
            **payload["reward_components"],
            "episode/return": payload["reward"],
            "timing/learner_train_ms": payload["train_time"] * 1000,
            "timing/learner_inference_ms": payload["inference_time"] * 1000,
            "perf/iter_ms": payload["iteration_time"] * 1000,
            "wall_seconds": time.monotonic() - self.started,
        }
        with (self.run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return payload

    def _save_checkpoint(self, *, log_dir, iteration, logger):
        # The native runner also calls this at finalization. Keep intermediate
        # names iteration-based; the drained final snapshot is written below.
        interval = int(self.cfg.algo.save_interval)
        if interval == 0 or iteration % interval:
            return None
        path = self.run / f"teacher_iteration_{iteration}.pt"
        if not path.exists():
            self.save(path)
            logger.log_save(str(path))
        return str(path)


def train_flashsac(cfg):
    started = time.monotonic()
    if cfg.training.get("devices"):
        raise ValueError("The HORA FlashSAC runner uses one learner device")
    for name in (
        "num_envs",
        "batch_size",
        "replay_buffer_n",
        "updates_per_step",
        "policy_frequency",
    ):
        if int(cfg.algo[name]) < 1:
            raise ValueError(f"algo.{name} must be positive")
    if int(cfg.algo.replay_buffer_n) * int(cfg.algo.num_envs) < max(
        int(cfg.algo.batch_size),
        int(cfg.algo.learning_starts) * int(cfg.algo.num_envs),
    ):
        raise ValueError("Replay capacity cannot satisfy the learning_starts/batch_size threshold")
    if int(cfg.algo.save_interval) < 0:
        raise ValueError("algo.save_interval must be nonnegative")
    device = resolve_device(cfg.training.device)
    cfg.training.device = device
    runtime = resolve_torch_thread_runtime(cfg.training.torch_threads)
    apply_torch_thread_runtime(runtime, role="learner")
    torch.manual_seed(int(cfg.algo.seed))
    np.random.seed(int(cfg.algo.seed))
    _, _, learner = make_models(cfg, device)
    run = Path(cfg.training.log_dir or f"logs/flashsac/seed_{cfg.algo.seed}_{time.time_ns()}")
    run.mkdir(parents=True, exist_ok=False)
    write_run_metadata(run, cfg)
    runner = TeacherDoubleBufferRunner(
        cfg=cfg,
        run=run,
        started=started,
        learner=learner,
        env_name=str(cfg.training.task_name),
        algo_type="flashsac",
        env_factory=partial(FlashTeacherEnv, config=OmegaConf.to_container(cfg, resolve=True)),
        num_envs=int(cfg.algo.num_envs),
        replay_buffer_n=int(cfg.algo.replay_buffer_n),
        batch_size=int(cfg.algo.batch_size),
        learning_starts=int(cfg.algo.learning_starts),
        updates_per_step=int(cfg.algo.updates_per_step),
        policy_frequency=int(cfg.algo.policy_frequency),
        env_steps_per_sync=int(cfg.training.env_steps_per_sync),
        device=device,
        obs_normalization=False,
        sim_backend=str(cfg.training.sim_backend),
        seed=int(cfg.algo.seed),
        torch_thread_runtime=runtime,
    )
    try:
        runner.learn(
            max_iterations=int(cfg.algo.max_iterations),
            save_interval=int(cfg.algo.save_interval),
            log_dir=str(run),
            logger_type=str(cfg.training.logger),
        )
        final = run / "teacher_final.pt"
        runner.collected = int(runner.last_run_summary["total_env_steps"])
        runner.last_run_summary["last_checkpoint"] = runner.save(final)
        write_json(run / "summary.json", {**runner.last_run_summary, "counters": runner.counters()})
    finally:
        runner.close()
    return final
