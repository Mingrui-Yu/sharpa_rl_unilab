"""Persistent random external object-force event."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from unilab.managers import ManagerTermBase
from unisim.backend.base import TensorExecution

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.compat import rng_tensor
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.randomization import SharpaDomainRandomization
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_name,
    require_real,
    resolve_tensor_env_ids,
    validate_term_params,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers import ManagerTermBaseCfg

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


class SharpaPersistentObjectForce(ManagerTermBase):
    """Source-equivalent decaying random object force."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(
            term,
            cfg,
            {
                "entity_name",
                "domain_randomization_name",
                "object_body_name",
                "force_scale",
                "force_probability",
                "force_decay",
                "force_decay_interval",
            },
        )
        self._entity = cast(
            "Entity", env.scene[require_name(term, "entity_name", cfg.params.get("entity_name"))]
        )
        self._randomization_name = require_name(
            term, "domain_randomization_name", cfg.params.get("domain_randomization_name")
        )
        self._randomization: SharpaDomainRandomization | None = None
        self.rng: np.random.Generator | None = None
        self._force_scale = require_real(
            term, "force_scale", cfg.params.get("force_scale", 2.0), minimum=0.0
        )
        self._probability = require_real(
            term, "force_probability", cfg.params.get("force_probability", 0.25), minimum=0.0
        )
        decay = require_real(term, "force_decay", cfg.params.get("force_decay", 0.9), minimum=0.0)
        interval = require_real(
            term,
            "force_decay_interval",
            cfg.params.get("force_decay_interval", 0.08),
            minimum=0.0,
        )
        if interval == 0.0:
            raise ValueError(f"{term} force_decay_interval must be positive")
        self._decay = float(np.power(decay, env.step_dt / interval))
        body_name = require_name(
            term, "object_body_name", cfg.params.get("object_body_name", "object")
        )
        requested_body_ids, requested_body_names = self._entity.find_bodies([body_name])
        if tuple(requested_body_names) != (body_name,):
            raise ValueError(f"{term} could not resolve the unique object body")
        local_ids, backend_ids = self._entity.bind_body_wrench(
            np.asarray(requested_body_ids, dtype=np.intp), torque=False, term_name=term
        )
        self._backend_body_ids = np.asarray(backend_ids, dtype=np.intp)
        self._backend_body_ids.setflags(write=False)
        self._force = torch.zeros((env.num_envs, 1, 3), dtype=torch.float32, device=env.device)
        self._host_wrench = (
            env.backend.get_tensor_capabilities().execution is TensorExecution.DEVICE_RESIDENT
        )
        del local_ids

    def _apply_force(self, env: SharpaEnv, ids: torch.Tensor | None) -> None:
        if self._host_wrench:
            # HOST_BRIDGE backends reject device wrench writes while the
            # asynchronous step receipt is pending. Their public host staging API is the only
            # wrench path that preserves the persistent-force protocol here.
            env.backend.apply_body_force(self._backend_body_ids, self._force.detach().cpu().numpy())
            return
        self._entity.apply_body_wrench_to_sim(
            self._force if ids is None else self._force[ids],
            None,
            self._backend_body_ids,
            env_ids=ids,
            term_name=type(self).__name__,
        )

    def __call__(
        self,
        env: SharpaEnv,
        env_ids: torch.Tensor | None,
        **params: Any,
    ) -> None:
        del env_ids, params
        if self._randomization is None:
            dr = env.event_manager.get_term_cfg(self._randomization_name).func
            if not isinstance(dr, SharpaDomainRandomization):
                raise TypeError(f"{type(self).__name__} requires SharpaDomainRandomization")
            self._randomization = dr
        self._force.mul_(self._decay)
        rng = env.rng if self.rng is None else self.rng
        trigger = (
            rng_tensor(rng.random(env.num_envs), device=env.device, dtype=self._force.dtype)
            < self._probability
        )
        ids = trigger.nonzero(as_tuple=False).flatten()
        if ids.numel():
            mass = self._randomization.mass[ids]
            noise = rng_tensor(
                rng.standard_normal((ids.numel(), 3)),
                device=env.device,
                dtype=self._force.dtype,
            )
            self._force[ids, 0, :] = noise * mass[:, None] * self._force_scale
        self._apply_force(env, None)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = resolve_tensor_env_ids(self._env, env_ids)
        self._force[ids] = 0.0
        if ids.numel():
            self._apply_force(cast("SharpaEnv", self._env), ids)
