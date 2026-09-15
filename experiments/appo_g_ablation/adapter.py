"""Experiment-only hooks. No external dependency or production restriction changes."""

import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from uni_rl.algos.appo import learner as upstream

from sharpa_rl_unilab.algos.hora import teacher
from sharpa_rl_unilab.algos.hora.distribution import PolicyDistribution
from sharpa_rl_unilab.training import teacher_runtime as rt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/appo_g_ablation"
_NATIVE_COLLECTOR = rt._collector
_NATIVE_COLLECT = rt.collect
_NATIVE_SAVE = rt.save_teacher
_NATIVE_RECEIVE = rt.AsyncRollouts.receive_ready
_NATIVE_LOG = rt.TrainingLogger.log
_NATIVE_GRAD = upstream._grad_norm
_INSTALLED = False


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def append(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj) + "\n")


def tensor_hash(state):
    return {
        k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for k, v in state.items()
    }


def rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
    }


def legacy_module():
    name = "sharpa_rl_unilab.algos.hora.frozen_ablation_teacher"
    if name not in sys.modules:
        path = OUT / "frozen/original/src/sharpa_rl_unilab/algos/hora/teacher.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def config(group, run, iterations=501):
    c = json.loads((OUT / "audit/original_config.json").read_text())
    c["algo"].update(max_iterations=iterations, save_interval=100, collector_seed=2)
    c["training"].update(
        max_transitions=None,
        action_diagnostics=False,
        save_transitions=0,
        log_dir=str(run),
        device="cuda:0",
        logger="no_print",
    )
    c["model"].update(
        std_parameterization="legacy_scalar" if group in "RAB" else "log",
        std_mode="state_independent",
        action_mapping="tanh",
        initial_std=1.0,
        log_std_bounds=None if group in "RABN" else [-13.815510557964274, 13.815510557964274],
    )
    c["protocol"].pop("appo_baseline", None)
    c["evaluation"].update(diagnostics=True, manifest=str(OUT / "scenes.json"))
    c["experiment"] = {"group": group, "rng_reset_after_construction": True}
    return OmegaConf.create(c)


def old_kl(mean, std, old_mean, old_std):
    return (
        torch.log(std / old_std + 1e-5)
        + (old_std.square() + (old_mean - mean).square()) / (2 * std.square())
        - 0.5
    ).sum(-1)


def branch(kl):
    return "down" if kl > 0.048 else ("up" if 0 < kl < 0.04 / 1.2 else "same")


def collector_entry(config, *args):
    install(config["experiment"]["group"], Path(config["training"]["log_dir"]), collector=True)
    return _NATIVE_COLLECTOR(config, *args)


def install(group, run, *, collector=False):
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    old = legacy_module()
    actor_type = old.TeacherActor if group == "R" else teacher.TeacherActor
    base = old.TeacherAPPOLearner if group == "R" else teacher.TeacherAPPOLearner
    state = {"packets": [], "norms": []}

    def make_models(cfg, device):
        actor = actor_type(rt.config_dict(cfg.model)).to(device)
        critic = teacher.CleanValue(rt.config_dict(cfg.model)).to(device)
        canonical = OUT / "audit/canonical_initial.pt"
        # All constructors consume the same RNG; explicit copy verifies/fixes names.
        if not canonical.exists():
            a = dict(actor.state_dict())
            a.pop("shared.distribution.std_param", None)
            a.pop("std_module.std_param", None)
            a.pop("std_module.log_std", None)
            torch.save({"actor": a, "critic": rt.frozen_weights(critic)}, canonical)
        reference = torch.load(canonical, map_location=device, weights_only=False)
        own = actor.state_dict()
        assert all(torch.equal(own[k], v) for k, v in reference["actor"].items()), (
            "Initialization changed"
        )
        assert all(torch.equal(critic.state_dict()[k], v) for k, v in reference["critic"].items())
        own.update(reference["actor"])
        actor.load_state_dict(own)
        critic.load_state_dict(reference["critic"])
        construction_rng = rng_state()
        torch.manual_seed(1)
        np.random.seed(1)
        torch.save(
            {
                "actor": rt.frozen_weights(actor),
                "critic": rt.frozen_weights(critic),
                "construction_rng": construction_rng,
                "rng": rng_state(),
            },
            run / "initial.pt",
        )
        write(
            run / "initial_hashes.json",
            {"actor": tensor_hash(actor.state_dict()), "critic": tensor_hash(critic.state_dict())},
        )
        return actor, critic, None

    class Instrumented(base):
        def _minibatch_policy_value(self, *args):
            values = super()._minibatch_policy_value(*args)
            self._audit_mu, self._audit_std = values[:2]
            return values

        def _minibatch_loss_tensors(self, *args):
            result = super()._minibatch_loss_tensors(*args)
            mu, std = self._audit_mu, self._audit_std
            with torch.no_grad():
                ko = old_kl(mu, std, args[-2], args[-1]).mean()
                ke = PolicyDistribution(mu, std, "tanh").kl_from(args[-2], args[-1]).mean()
                # std checks at every minibatch, all joints; buffered until update end.
                self._audit_buffer.append(torch.stack([ko, ke, std.min(), std.max()]))
            selected = ko if group in "RAC" else ke
            return (*result[:4], selected, *result[5:])

        def _update_adaptive_learning_rate(self, kl):
            before = self.learning_rate
            super()._update_adaptive_learning_rate(kl)
            self._audit_lr.append([before, self.learning_rate, branch(kl)])

        def update(self, batch):
            self._audit_buffer = []
            self._audit_lr = []
            state["norms"] = []
            start = time.perf_counter()
            result = super().update(batch)
            values = torch.stack(self._audit_buffer).cpu().tolist()
            param = (
                self.actor.shared.distribution.std_param
                if group == "R"
                else (
                    self.actor.std_module.std_param
                    if group in "AB"
                    else self.actor.std_module.log_std
                )
            )
            raw = param.detach().cpu()
            std = (
                raw.clamp(1e-6, 1e6)
                if group in "RAB"
                else raw.clamp(-13.815510557964274, 13.815510557964274).exp()
                if group in "CD"
                else raw.exp()
            )
            mb = [
                {
                    "old_kl": v[0],
                    "exact_kl": v[1],
                    "std_min": v[2],
                    "std_max": v[3],
                    "lr_before": lr[0],
                    "lr_after": lr[1],
                    "branch": lr[2],
                    "shadow_branch": branch(v[1] if group in "RAC" else v[0]),
                    "grad_norm": norm,
                    "clip_coefficient": min(1.0, self.max_grad_norm / (norm + 1e-6)),
                }
                for v, lr, norm in zip(values, self._audit_lr, state["norms"])
            ]
            assert len(mb) == 20
            versions, counts = torch.unique(batch["behavior_version"], return_counts=True)
            append(
                run / "updates.jsonl",
                {
                    "iteration": self._update_counter,
                    "seconds": time.perf_counter() - start,
                    "minibatches": mb,
                    "std": std.tolist(),
                    "raw_std_parameter": raw.tolist(),
                    "boundary_hits": sum(v[2] <= 1e-6 or v[3] >= 1e6 for v in values),
                    "staging_versions": list(zip(versions.cpu().tolist(), counts.cpu().tolist())),
                    "packets": state["packets"],
                },
            )
            return result

        def process_batch(self, batch):
            if os.environ.get("APPO_CAPTURE_BATCH") == "1" and self._update_counter == 0:
                torch.save(
                    {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    run / "real_batch.pt",
                )
                torch.save(
                    {
                        "actor": rt.frozen_weights(self.actor),
                        "critic": rt.frozen_weights(self.critic),
                        "target_actor": rt.frozen_weights(self.target_actor),
                        "rng": rng_state(),
                    },
                    run / "batch_state.pt",
                )
            return super().process_batch(batch)

    def grad(params):
        result = _NATIVE_GRAD(params)
        state["norms"].append(result)
        return result

    def receive(self):
        packets = _NATIVE_RECEIVE(self)
        state["packets"] = [
            {"behavior_version": p[2], "end": p[3], "size": p[0]["rewards"].size} for p in packets
        ]
        return packets

    def log(self, metrics, *, timings=None):
        append(run / "runtime_metrics.jsonl", {"metrics": metrics, "timings": timings})
        return _NATIVE_LOG(self, metrics, timings=timings)

    def save(path, cfg, actor, critic, learner, counters, elapsed):
        start = time.perf_counter()
        result = _NATIVE_SAVE(path, cfg, actor, critic, learner, counters, elapsed)
        append(
            run / "saves.jsonl",
            {
                "path": str(path),
                "iteration": counters["policy_version"],
                "seconds": time.perf_counter() - start,
                "counters": counters,
            },
        )
        return result

    collect_impl = _NATIVE_COLLECT
    if group == "R":
        source = (
            OUT / "frozen/original/src/sharpa_rl_unilab/training/teacher_runtime.py"
        ).read_text()
        node = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "collect"
        )
        scope = dict(vars(rt))
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "<frozen original collect>", "exec"),
            scope,
        )
        collect_impl = scope["collect"]
    first = [True]

    def collect(*args, **kwargs):
        if first[0]:
            first[0] = False
            before = rng_state()
            torch.manual_seed(2)
            np.random.seed(2)
            torch.save({"construction_rng": before, "rng": rng_state()}, run / "collector_rng.pt")
        return collect_impl(*args, **kwargs)

    rt.make_models = make_models
    rt.TeacherActor = actor_type
    rt.TeacherAPPOLearner = Instrumented
    rt._collector = collector_entry
    rt.collect = collect
    rt.AsyncRollouts.receive_ready = receive
    rt.save_teacher = save
    rt.TrainingLogger.log = log
    upstream._grad_norm = grad
    return Instrumented
