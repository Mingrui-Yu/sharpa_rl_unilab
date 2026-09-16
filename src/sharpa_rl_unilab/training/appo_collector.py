"""APPO collector process, bounded rollout queue and policy publication."""

import math
import multiprocessing as mp
import queue
import time
import traceback

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv

from .checkpoints import frozen_weights
from .configuration import configure_threads
from .policy import make_actor
from .rollouts import collect


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
        actor = make_actor(cfg, device).eval()
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
                env,
                obs,
                actor,
                horizon,
                device,
                count=count,
                metrics=collector_metrics,
                store_distribution=False,
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
                {k: obs[k].copy() for k in ("obs", "priv_info", "critic")},
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
