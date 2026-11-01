"""HORA observation/checkpoint adapters around UniLab's native FlashSAC runner."""

from __future__ import annotations

import math
import time
from functools import partial
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from unilab.base.contracts.env import EnvAlgoCapabilities
from unilab.rl.logging.metric_schema import metric_spec
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.rl.runtime.offpolicy.thread_budget import (
    apply_torch_thread_runtime,
    resolve_torch_thread_runtime,
)
from unilab.rl.runtime.placement import resolve_inference_placement

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTOR_DIM, CRITIC_DIM, PRIV_DIM
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv, TorchEnvState

from .checkpoints import save_teacher
from .configuration import resolve_device
from .exploration import HeldGaussianNoise
from .logging import append_metrics, create_run, write_json
from .policy import make_models, observe_new_samples


def transport_obs(obs):
    """Pack current privilege with actor history; keep clean Q inputs separate."""
    return {"obs": torch.cat((obs["obs"], obs["priv_info"]), dim=-1), "critic": obs["critic"]}


class FlashTeacherEnv:
    def __init__(self, num_envs, env_cfg_override, *, config, tensor_device=None):
        if env_cfg_override:
            raise ValueError("FlashTeacherEnv takes task overrides through its composed config")
        cfg = OmegaConf.create(config)
        assert isinstance(cfg, DictConfig)
        self.teacher = SharpaTeacherEnv(cfg, num_envs, tensor_device=tensor_device)
        self.state = None

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    @property
    def obs_groups_spec(self):
        return {"obs": ACTOR_DIM + PRIV_DIM, "critic": CRITIC_DIM}

    @property
    def algo_capabilities(self):
        names = tuple(self.teacher.env.reward_manager.step_reward_names)
        width = int(self.teacher.env.reward_manager.step_rewards.shape[-1])
        if not names and width:
            names = tuple(f"reward/component_{index}" for index in range(width))
        return EnvAlgoCapabilities(reward_component_names=names)

    @property
    def produces_completion_receipts(self) -> bool:
        # MuJoCo's host bridge executes synchronous tensor steps and does not
        # publish the event receipts used only by asynchronous CUDA backends.
        return False

    def _state(self, state):
        self.state = TorchEnvState(
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
        self.held_noise = HeldGaussianNoise(
            float(cfg.algo.exploration.noise_zeta_mu), cfg.algo.exploration.noise_zeta_max
        )
        self.learner.actor.exploration_sampler = self._sample_exploration
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

    def _sample_exploration(self, obs, dones):
        distribution = self.learner.actor.policy(obs)
        noise = self.held_noise.sample(distribution.mean, dones)
        return distribution.sample(noise, log_prob=False).action

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
        saturation = {key: float(value) for key, value in self.learner.saturation_metrics.items()}
        payload = super()._aggregate_log_statistics(logger, **kwargs)
        payload["metrics"].update(
            {key: value for key, value in saturation.items() if metric_spec(key) is not None}
        )
        reward = payload["return_mean_ep100"]
        row = {
            **payload["metrics"],
            **saturation,
            **self.counters(),
            **payload["reward_components"],
            "episode/return": reward,
            "timing/learner_train_ms": payload["train_time"] * 1000,
            "timing/learner_inference_ms": payload["inference_time"] * 1000,
            "perf/iter_ms": payload["iteration_time"] * 1000,
            "wall_seconds": time.monotonic() - self.started,
        }
        append_metrics(self.run, row)
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
    run = create_run(cfg)
    placement = resolve_inference_placement(
        learner_device=device, collector_tensor_device="auto", algo_name="sharpa-flashsac"
    )
    runner = TeacherDoubleBufferRunner(
        cfg=cfg,
        run=run,
        started=started,
        learner=learner,
        env_name=str(cfg.training.task_name),
        algo_type="flashsac",
        env_factory=partial(
            FlashTeacherEnv,
            config=OmegaConf.to_container(cfg, resolve=True),
            tensor_device=placement.env_device,
        ),
        inference_placement=placement,
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
        summary = cast(dict[str, Any], runner.last_run_summary)
        runner.collected = int(summary["total_env_steps"])
        summary["last_checkpoint"] = runner.save(final)
        write_json(run / "summary.json", {**summary, "counters": runner.counters()})
    finally:
        runner.close()
    return final
