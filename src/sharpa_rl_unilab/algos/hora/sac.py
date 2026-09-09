"""HORA-owned SAC entry helpers.

Importing this module registers the HORA-SAC off-policy actor adapter with
``uni_rl.offpolicy.actor_adapter`` so the generic off-policy runtime (learner
process and spawn collector subprocesses) can build, sample, and feed the
privileged HORA actor without hardcoding ``hora_sac`` branches in uni_rl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from uni_rl.offpolicy.actor_adapter import OffPolicyActorAdapter, register_offpolicy_actor_adapter
from uni_rl.offpolicy.runtime import OffPolicyRuntime

from sharpa_rl_unilab.algos.hora.observations import split_hora_obs_with_priv_info
from sharpa_rl_unilab.algos.hora.runtime import HORA_SAC_RUNTIME_IMPL, is_hora_sac_runtime
from sharpa_rl_unilab.algos.hora.sac_learner import HoraSACLearner

# Dotted path of this module; declared on the runtime so the generic
# off-policy runner imports it (re-running the registration below) inside
# spawn collector subprocesses, which do not inherit parent registrations.
_ACTOR_ADAPTER_MODULE = "sharpa_rl_unilab.algos.hora.sac"


def _build_hora_sac_actor(
    *,
    obs_dim: int,
    action_dim: int,
    actor_hidden_dim: int,
    use_layer_norm: bool,
    device: str | torch.device,
    priv_info_dim: int | None = None,
    priv_info_embed_dim: int = 9,
    priv_mlp_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 9),
) -> Any:
    """Build the privileged HORA-SAC actor (OffPolicyActorAdapter build hook)."""
    if priv_info_dim is None:
        raise ValueError("build_actor(algo_type='hora_sac') requires priv_info_dim.")
    from sharpa_rl_unilab.algos.hora.sac_models import HoraSACActor

    return HoraSACActor(
        obs_dim=obs_dim,
        priv_info_dim=int(priv_info_dim),
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        priv_info_embed_dim=priv_info_embed_dim,
        priv_mlp_hidden_dims=tuple(priv_mlp_hidden_dims),
        use_layer_norm=use_layer_norm,
        device=device,
    )


def _sample_hora_sac_actions(
    actor: Any,
    obs_torch: torch.Tensor,
    prev_dones_torch: torch.Tensor,
    priv_info_torch: torch.Tensor | None,
) -> torch.Tensor:
    """Sample HORA-SAC exploration actions (OffPolicyActorAdapter sample hook)."""
    del prev_dones_torch
    if priv_info_torch is None:
        raise ValueError("HORA-SAC action sampling requires priv_info_torch.")
    return cast(torch.Tensor, actor.explore(obs_torch, priv_info_torch, deterministic=False))


def _resolve_hora_sac_priv_info(
    obs_np: np.ndarray,
    critic_np: np.ndarray,
    info: dict | None,
) -> np.ndarray:
    """Extract HORA privileged info from env observations (adapter hook)."""
    _, _, priv_info_np = split_hora_obs_with_priv_info(
        {"obs": obs_np, "critic": critic_np},
        info,
    )
    if priv_info_np is None:
        raise ValueError(
            "HORA-SAC requires privileged info from info['critic_info'] "
            "or the critic observation tail."
        )
    return np.asarray(priv_info_np, dtype=np.float32)


def _hora_sac_actor_context_from_obs(obs_device: torch.Tensor, obs_dim: int) -> torch.Tensor:
    """Slice the packed learner-side inference observation into the priv tail."""
    return obs_device[:, obs_dim:]


# Module-level registration runs exactly once per process (Python imports a
# module at most once); spawn collectors re-run it by importing this module
# through the runtime's ``actor_adapter_modules``.
register_offpolicy_actor_adapter(
    OffPolicyActorAdapter(
        algo_type=HORA_SAC_RUNTIME_IMPL,
        build_actor=_build_hora_sac_actor,
        sample_actions=_sample_hora_sac_actions,
        resolve_priv_info=_resolve_hora_sac_priv_info,
        actor_context_from_obs=_hora_sac_actor_context_from_obs,
    )
)


@dataclass(frozen=True)
class HoraSACRuntime(OffPolicyRuntime):
    """Resolved HORA-SAC hooks consumed by the generic off-policy script."""

    learner_cls: type[Any] | None = HoraSACLearner
    algo_type: str | None = HORA_SAC_RUNTIME_IMPL
    actor_adapter_modules: tuple[str, ...] = (_ACTOR_ADAPTER_MODULE,)
    actor_cfg: dict[str, Any] = field(default_factory=dict)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        """Build HORA learner actor kwargs from privileged observation dimensions."""
        priv_info_dim = int(critic_obs_dim - obs_dim)
        if priv_info_dim <= 0:
            raise ValueError(
                "HORA-SAC requires critic observations to contain privileged tail "
                f"features; got obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}."
            )
        return {
            "priv_info_dim": priv_info_dim,
            "priv_info_embed_dim": int(self.actor_cfg.get("priv_info_embed_dim", 9)),
            "priv_mlp_hidden_dims": tuple(
                self.actor_cfg.get("priv_mlp_hidden_dims", (256, 128, 9))
            ),
        }


def resolve_hora_sac_runtime(rl_cfg: dict[str, Any]) -> HoraSACRuntime | None:
    """Resolve HORA-SAC hooks from an explicit owner-config runtime marker."""
    if not is_hora_sac_runtime(rl_cfg):
        return None
    actor_cfg_raw = rl_cfg.get("actor", {})
    actor_cfg = actor_cfg_raw if isinstance(actor_cfg_raw, dict) else {}
    return HoraSACRuntime(actor_cfg=dict(actor_cfg))


__all__ = ["HoraSACRuntime", "resolve_hora_sac_runtime"]
