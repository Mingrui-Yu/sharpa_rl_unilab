"""Model construction and stateless policy input/inference helpers."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from sharpa_rl_unilab.algos.hora.flashsac import CleanQ, TeacherFlashActor, TeacherFlashLearner
from sharpa_rl_unilab.algos.hora.models import split_actor
from sharpa_rl_unilab.algos.hora.on_policy import CleanValue, TeacherActor
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTOR_DIM, CRITIC_DIM

from .configuration import config_dict


def tensor_obs(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


def make_actor(cfg, device="cpu", *, student=False, model=None):
    """Use saved model settings as-is; training validation belongs to make_models."""
    model = config_dict(cfg.model) if model is None else model
    if cfg.algo.algo == "flashsac":
        actor = TeacherFlashActor(model, student=student)
    else:
        actor = TeacherActor(
            model, student=student, kl_mode=cfg.algo.algorithm.get("kl_mode", "exact")
        )
    return actor.to(device)


def make_models(cfg, device):
    model = config_dict(cfg.model)
    parameterization = model.get("std_parameterization")
    if parameterization not in {"log", "direct", "legacy_scalar"}:
        raise ValueError(
            "New training requires model.std_parameterization=log or direct "
            "(legacy_scalar is an alias); legacy_tanh is checkpoint-only"
        )
    if parameterization in {"direct", "legacy_scalar"}:
        if cfg.algo.algo not in {"ppo", "appo"}:
            raise ValueError("Direct std new training is supported only for PPO/APPO")
        if model.get("std_mode") != "state_independent":
            raise ValueError("Direct std requires model.std_mode=state_independent")
        if model.get("log_std_bounds") is not None:
            raise ValueError("Direct std requires model.log_std_bounds=null; std uses [1e-6, 1e6]")
        # Canonicalize only new construction; old checkpoints retain their semantics.
        model["std_parameterization"] = "direct"
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
    actor = make_actor(cfg, device, model=model)
    return actor, CleanValue(model).to(device), None


@torch.no_grad()
def observe_new_samples(actor, critic, packed: torch.Tensor, clean: torch.Tensor) -> None:
    base, _ = split_actor(packed)
    actor.shared.obs_normalizer.update(base.reshape(-1, ACTOR_DIM))
    if isinstance(critic, (CleanValue, CleanQ)):
        critic.obs_normalizer.update(clean.reshape(-1, CRITIC_DIM))


@torch.no_grad()
def inference_distribution(actor, obs, device, history_normalizer=None):
    keys = ("obs", "priv_info") if history_normalizer is None else ("obs", "proprio_hist")
    data = tensor_obs({key: obs[key] for key in keys}, device)
    inputs = {"actor": data["obs"]}
    if history_normalizer is None:
        inputs["priv_info"] = data["priv_info"]
    else:
        inputs["proprio_hist"] = history_normalizer(data["proprio_hist"], update=False)
    return actor.policy(TensorDict(inputs, batch_size=data["obs"].shape[0]))


@torch.no_grad()
def deterministic_actions(actor, obs, device, history_normalizer=None):
    return (
        inference_distribution(actor, obs, device, history_normalizer).deterministic().cpu().numpy()
    )
