"""Sharpa policy/critic observation terms and shared tensor task state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import FRAME_DIM, PRIV_DIM
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.action import SharpaIncrementalPositionAction
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.compat import (
    entity_host_data,
    numpy_dtype,
    rng_tensor,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import (
    HAND_JOINT_NAMES,
    NUM_HAND_JOINTS,
    SOURCE_DEFAULT_HAND_JOINT_POS,
    TACTILE_SENSOR_NAMES,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.geometry import (
    torch_quat_angular_velocity_from_pair,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.randomization import SharpaDomainRandomization
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.reset import SharpaHandObjectReset
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_bool,
    require_name,
    require_names,
    require_real,
    resolve_tensor_env_ids,
    validate_term_params,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity, SceneTensorReadPlan
    from unilab.managers import ManagerTermBaseCfg

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


class SharpaRotationObservation(ManagerTermBase):
    """Own one Sharpa policy frame plus state shared by termination and rewards."""

    observation_width = FRAME_DIM

    def bind_read_phase(self) -> None:
        pass

    _ALLOWED_PARAMS = {
        "entity_name",
        "action_name",
        "event_state_name",
        "domain_randomization_name",
        "sensor_names",
        "joint_noise",
        "contact_smoothing",
        "contact_latency",
        "contact_sensor_noise",
        "binary_contact",
        "contact_threshold",
        "tactile_force_clip_max",
        "enable_tactile",
        "clip_obs",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(term, cfg, self._ALLOWED_PARAMS)
        entity_name = require_name(term, "entity_name", cfg.params.get("entity_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        self.tensor_state_entity = entity_name
        root_body_name = self._entity.root_body_name
        if not isinstance(root_body_name, str) or not root_body_name:
            raise ValueError(f"{term} requires a declared entity root body")
        self.tensor_body_names = (root_body_name,)
        self._action_name = require_name(term, "action_name", cfg.params.get("action_name"))
        self._action: SharpaIncrementalPositionAction | None = None
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(list(HAND_JOINT_NAMES))
        if tuple(joint_names) != HAND_JOINT_NAMES or len(joint_ids) != NUM_HAND_JOINTS:
            raise ValueError(f"{term} could not resolve all Sharpa hand joints in order")
        state_name = require_name(term, "event_state_name", cfg.params.get("event_state_name"))
        state = env.event_manager.get_term_cfg(state_name).func
        if not isinstance(state, SharpaHandObjectReset):
            raise TypeError(f"{term} event state must be SharpaHandObjectReset")
        self.task_state = state
        dr_name = require_name(
            term, "domain_randomization_name", cfg.params.get("domain_randomization_name")
        )
        dr = env.event_manager.get_term_cfg(dr_name).func
        if not isinstance(dr, SharpaDomainRandomization):
            raise TypeError(f"{term} DR state must be SharpaDomainRandomization")
        self.domain_randomization = dr

        sensor_names = require_names(
            term, "sensor_names", cfg.params.get("sensor_names", TACTILE_SENSOR_NAMES)
        )
        self._sensor_view = env.scene.bind_sensor_data(sensor_names)
        self._sensor_dims = self._sensor_view.dimensions
        self.tensor_sensor_names = tuple(sensor_names)
        self._enable_tactile = require_bool(
            term, "enable_tactile", cfg.params.get("enable_tactile", True)
        )
        self._joint_noise = require_real(
            term, "joint_noise", cfg.params.get("joint_noise", 0.02), minimum=0.0
        )
        self._contact_smoothing = require_real(
            term, "contact_smoothing", cfg.params.get("contact_smoothing", 0.5), minimum=0.0
        )
        self._contact_latency = require_real(
            term, "contact_latency", cfg.params.get("contact_latency", 0.005), minimum=0.0
        )
        self._contact_noise = require_real(
            term, "contact_sensor_noise", cfg.params.get("contact_sensor_noise", 0.01), minimum=0.0
        )
        self._binary_contact = require_bool(
            term, "binary_contact", cfg.params.get("binary_contact", False)
        )
        self._contact_threshold = require_real(
            term, "contact_threshold", cfg.params.get("contact_threshold", 0.05), minimum=0.0
        )
        self._tactile_clip = require_real(
            term, "tactile_force_clip_max", cfg.params.get("tactile_force_clip_max", 4.0)
        )
        self._clip_obs = require_real(term, "clip_obs", cfg.params.get("clip_obs", 5.0))

        dtype = get_global_dtype()
        device = env.device
        self.dof_pos = torch.zeros((env.num_envs, NUM_HAND_JOINTS), dtype=dtype, device=device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.object_pos = torch.zeros((env.num_envs, 3), dtype=dtype, device=device)
        self.object_quat = torch.zeros((env.num_envs, 4), dtype=dtype, device=device)
        self.object_quat[:, 0] = 1.0
        self.object_linvel = torch.zeros_like(self.object_pos)
        self.object_angvel = torch.zeros_like(self.object_pos)
        self.torques = torch.zeros_like(self.dof_pos)
        self._previous_dof_pos = self.dof_pos.clone()
        self._previous_object_pos = self.object_pos.clone()
        self._previous_object_quat = self.object_quat.clone()
        self._prev_tactile_force = torch.zeros(
            (env.num_envs, len(sensor_names)), dtype=dtype, device=device
        )
        self.last_contacts = torch.zeros_like(self._prev_tactile_force)
        self._tactile_output = self.last_contacts.clone()
        self._last_counter = int(env.common_step_counter)
        self._needs_reset = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        self._clean_tactile = torch.zeros_like(self._prev_tactile_force)
        self._actor_frame = torch.zeros((env.num_envs, self.frame_dim), dtype=dtype, device=device)
        self._clean_frame = torch.zeros_like(self._actor_frame)
        self.priv_info = torch.zeros((env.num_envs, PRIV_DIM), dtype=dtype, device=device)
        self._default_dof_pos = torch.as_tensor(
            SOURCE_DEFAULT_HAND_JOINT_POS, dtype=dtype, device=device
        )

        ranges = np.asarray(
            entity_host_data(self._entity, "actuator_ctrl_range"),
            dtype=numpy_dtype(dtype),
        )
        actuator_indices = np.arange(NUM_HAND_JOINTS, dtype=np.intp)
        self._dof_mid = torch.as_tensor(
            (ranges[actuator_indices, 1] + ranges[actuator_indices, 0]) / 2.0,
            dtype=dtype,
            device=device,
        )
        self._dof_range = torch.as_tensor(
            ranges[actuator_indices, 1] - ranges[actuator_indices, 0],
            dtype=dtype,
            device=device,
        )

    def _resolve_action(self, env: SharpaEnv) -> SharpaIncrementalPositionAction | None:
        if self._action is None:
            action_manager = getattr(env, "action_manager", None)
            if action_manager is None:
                return None
            action = action_manager.get_term(self._action_name)
            if not isinstance(action, SharpaIncrementalPositionAction):
                raise TypeError(
                    f"{type(self).__name__} action must be SharpaIncrementalPositionAction"
                )
            self._action = action
        return self._action

    @property
    def last_counter(self) -> int:
        return self._last_counter

    @property
    def frame_dim(self) -> int:
        return NUM_HAND_JOINTS * 2 + (len(self._sensor_view.names) if self._enable_tactile else 0)

    @staticmethod
    def _read_plan(env: SharpaEnv) -> SceneTensorReadPlan:
        plan = env.scene.read_plan
        if plan is None:
            raise RuntimeError("Sharpa tensor observations require the scene packed read plan")
        return plan

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = resolve_tensor_env_ids(self._env, env_ids)
        self._needs_reset[ids] = True
        self._prev_tactile_force[ids] = 0.0
        self.last_contacts[ids] = 0.0

    def snapshot(self, env: SharpaEnv) -> SharpaRotationObservation:
        counter = int(env.common_step_counter)
        if counter != self._last_counter:
            if counter != self._last_counter + 1:
                raise RuntimeError(
                    f"Sharpa observation missed a control-step update: "
                    f"last={self._last_counter}, current={counter}"
                )
            ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        else:
            ids = self._needs_reset.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return self

        reset_ids = ids[self._needs_reset[ids]]
        plan = self._read_plan(env)
        joints = plan.joint_tensor_view(self._entity)
        body = plan.body_tensor_view(self._entity, self.tensor_body_names)
        if body.pos_w is None or body.quat_w is None:
            raise ValueError("Sharpa root body packet is missing position or quaternion")
        if joints.joint_pos.shape != self.dof_pos.shape:
            raise ValueError(
                f"Sharpa entity joint packet has shape {tuple(joints.joint_pos.shape)}; "
                f"expected {tuple(self.dof_pos.shape)}"
            )
        current_pos = body.pos_w[:, 0]
        current_quat = body.quat_w[:, 0]
        self.dof_pos[ids] = joints.joint_pos[ids]
        self.object_pos[ids] = current_pos[ids]
        self.object_quat[ids] = current_quat[ids]
        self.dof_vel[ids] = (self.dof_pos[ids] - self._previous_dof_pos[ids]) / env.step_dt
        self.object_linvel[ids] = (
            self.object_pos[ids] - self._previous_object_pos[ids]
        ) / env.step_dt
        self.object_angvel[ids] = torch_quat_angular_velocity_from_pair(
            self.object_quat[ids], self._previous_object_quat[ids], env.step_dt
        )
        self.dof_vel[reset_ids] = 0.0
        self.object_linvel[reset_ids] = 0.0
        self.object_angvel[reset_ids] = 0.0
        action = self._resolve_action(env)
        if action is None:
            self.torques[ids] = 0.0
            target = torch.zeros_like(self.dof_pos[ids])
        else:
            target = action.target[ids]
            self.torques[ids] = (
                self.domain_randomization.kp[ids] * (target - self.dof_pos[ids])
                - self.domain_randomization.kd[ids] * self.dof_vel[ids]
            )
        self._previous_dof_pos[ids] = self.dof_pos[ids]
        self._previous_object_pos[ids] = self.object_pos[ids]
        self._previous_object_quat[ids] = self.object_quat[ids]

        self._clean_tactile[ids] = self._read_tactile(env)[ids]
        if self._enable_tactile:
            self._tactile_output[ids] = self._process_tactile(env, ids)
        normalized = 2.0 * (self.dof_pos[ids] - self._dof_mid) / (self._dof_range + 1.0e-8)
        clean_parts = [normalized, target]
        noisy = normalized
        if self._joint_noise > 0.0:
            noise = rng_tensor(
                env.rng.uniform(-self._joint_noise, self._joint_noise, noisy.shape),
                device=env.device,
                dtype=noisy.dtype,
            )
            noisy = noisy + noise
        actor_parts = [noisy, target]
        if self._enable_tactile:
            clean_parts.append(self._clean_tactile[ids])
            actor_parts.append(self._tactile_output[ids])
        self._clean_frame[ids] = torch.cat(clean_parts, dim=1)
        self._actor_frame[ids] = torch.cat(actor_parts, dim=1)
        if self._clip_obs > 0.0:
            self._clean_frame[ids] = self._clean_frame[ids].clamp(-self._clip_obs, self._clip_obs)
            self._actor_frame[ids] = self._actor_frame[ids].clamp(-self._clip_obs, self._clip_obs)
        self.priv_info[ids] = self.domain_randomization.privileged_info(
            self.object_pos, self.task_state.object_pos_anchor
        )[ids]
        self._needs_reset[ids] = False
        self._last_counter = counter
        return self

    def _read_tactile(self, env: SharpaEnv) -> torch.Tensor:
        views = self._read_plan(env).sensor_tensor_views(self._entity, self._sensor_view.names)
        values = [views.values[name] for name in views.names]
        result = torch.zeros(
            (env.num_envs, len(values)), dtype=get_global_dtype(), device=env.device
        )
        columns = []
        for value, width in zip(values, self._sensor_dims, strict=True):
            columns.append(value[:, :3].norm(dim=1) if width >= 3 else value[:, 0])
        result[:] = torch.stack(columns, dim=1)
        if self._tactile_clip > 0.0:
            result.clamp_(0.0, self._tactile_clip)
        return result

    def _process_tactile(self, env: SharpaEnv, ids: torch.Tensor) -> torch.Tensor:
        current = self._clean_tactile[ids]
        smooth = current * self._contact_smoothing + self._prev_tactile_force[ids] * (
            1.0 - self._contact_smoothing
        )
        self._prev_tactile_force[ids] = current
        delayed = (
            rng_tensor(env.rng.random(smooth.shape), device=env.device, dtype=smooth.dtype)
            < self._contact_latency
        )
        contact = (
            (smooth > self._contact_threshold).to(smooth.dtype) if self._binary_contact else smooth
        )
        self.last_contacts[ids] = torch.where(delayed, self.last_contacts[ids], contact)
        result = self.last_contacts[ids]
        if self._binary_contact:
            keep = (
                rng_tensor(env.rng.random(result.shape), device=env.device, dtype=result.dtype)
                >= self._contact_noise
            )
            result = torch.where(result > 0.1, keep * result, result)
        return result

    def frame(self, env: SharpaEnv, *, noisy: bool) -> torch.Tensor:
        self.snapshot(env)
        return self._actor_frame if noisy else self._clean_frame

    def __call__(self, env: SharpaEnv, **params: Any) -> torch.Tensor:
        del params
        return self.frame(env, noisy=True)


class SharpaFlattenedObservation(SharpaRotationObservation):
    """Legacy noisy policy frame plus privileged channels."""

    observation_width = FRAME_DIM + PRIV_DIM

    def __call__(self, env: SharpaEnv, **params: Any) -> torch.Tensor:
        del params
        return torch.cat((self.frame(env, noisy=True), self.priv_info), dim=1)


class _SharpaObservationView(ManagerTermBase):
    """Read a named frame owner without another sensor read or noise draw."""

    output_dim: int

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(term, cfg, {"observation_group", "observation_term"})
        self._group = require_name(term, "observation_group", cfg.params.get("observation_group"))
        self._term = require_name(term, "observation_term", cfg.params.get("observation_term"))

    def __call__(self, env: SharpaEnv, **params: Any) -> torch.Tensor:
        del params
        manager = getattr(env, "observation_manager", None)
        if manager is None:
            return torch.zeros(
                (env.num_envs, self.output_dim), dtype=torch.float32, device=env.device
            )
        source = manager.get_term_cfg(self._group, self._term).func
        if not isinstance(source, SharpaRotationObservation) or source.frame_dim != FRAME_DIM:
            raise ValueError(f"{type(self).__name__} requires a 49D Sharpa frame owner")
        return self._read(source.snapshot(env))

    def _read(self, source: SharpaRotationObservation) -> torch.Tensor:
        raise NotImplementedError


class SharpaCriticObservation(_SharpaObservationView):
    """Current clean 49D frame plus current 9D privileged information."""

    output_dim = FRAME_DIM + PRIV_DIM

    def _read(self, source: SharpaRotationObservation) -> torch.Tensor:
        return torch.cat((source._clean_frame, source.priv_info), dim=1)


class SharpaPrivilegedObservation(_SharpaObservationView):
    """Explicit current privileged vector, independent of critic history layout."""

    output_dim = PRIV_DIM

    def _read(self, source: SharpaRotationObservation) -> torch.Tensor:
        return source.priv_info


class SharpaProprioObservation(_SharpaObservationView):
    """Same noisy frame as the Actor; configure a 30-frame manager history."""

    output_dim = FRAME_DIM

    def _read(self, source: SharpaRotationObservation) -> torch.Tensor:
        return source._actor_frame
