"""Sharpa policy/critic observation terms and shared task state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase
from unilab.utils.geometry import np_quat_angular_velocity_from_pair

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import FRAME_DIM, PRIV_DIM
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.action import SharpaIncrementalPositionAction
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import (
    HAND_JOINT_NAMES,
    NUM_HAND_JOINTS,
    TACTILE_SENSOR_NAMES,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.randomization import SharpaDomainRandomization
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.reset import SharpaHandObjectReset
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_bool,
    require_name,
    require_names,
    require_real,
    resolve_env_ids,
    validate_term_params,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers import ManagerTermBaseCfg

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


class SharpaRotationObservation(ManagerTermBase):
    """Own one Sharpa policy frame plus state shared by termination and rewards."""

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
        self._entity = cast(
            "Entity", env.scene[require_name(term, "entity_name", cfg.params.get("entity_name"))]
        )
        self._action_name = require_name(term, "action_name", cfg.params.get("action_name"))
        self._action: SharpaIncrementalPositionAction | None = None
        self._joint_ids_array, joint_names = self._entity.find_joints_by_actuator_names(
            list(HAND_JOINT_NAMES)
        )
        if tuple(joint_names) != HAND_JOINT_NAMES:
            raise ValueError(f"{term} could not resolve all Sharpa hand joints in order")
        self._joint_ids_array = np.asarray(self._joint_ids_array, dtype=np.intp)
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
        self.dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._joint_ids_array], dtype=dtype
        ).copy()
        self.dof_vel = np.zeros_like(self.dof_pos)
        self.object_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=dtype).copy()
        self.object_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype).copy()
        self.object_linvel = np.zeros_like(self.object_pos)
        self.object_angvel = np.zeros_like(self.object_pos)
        self.torques = np.zeros_like(self.dof_pos)
        self._previous_dof_pos = self.dof_pos.copy()
        self._previous_object_pos = self.object_pos.copy()
        self._previous_object_quat = self.object_quat.copy()
        self._prev_tactile_force = np.zeros((env.num_envs, len(sensor_names)), dtype=dtype)
        self.last_contacts = np.zeros_like(self._prev_tactile_force)
        self._tactile_output = self.last_contacts.copy()
        self._last_counter = int(env.common_step_counter)
        self._needs_reset = np.ones(env.num_envs, dtype=np.bool_)
        self._clean_tactile = np.zeros_like(self._prev_tactile_force)
        self._actor_frame = np.zeros((env.num_envs, self.frame_dim), dtype=dtype)
        self._clean_frame = np.zeros_like(self._actor_frame)
        self.priv_info = np.zeros((env.num_envs, PRIV_DIM), dtype=dtype)

        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        actuator_indices = np.arange(NUM_HAND_JOINTS, dtype=np.intp)
        self._dof_mid = (ranges[actuator_indices, 1] + ranges[actuator_indices, 0]) / 2.0
        self._dof_range = ranges[actuator_indices, 1] - ranges[actuator_indices, 0]

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

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        # Reset events commit after manager.reset(). Read the new episode only
        # when observations are next computed, after action targets also reset.
        ids = resolve_env_ids(self._env, env_ids)
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
            ids = np.arange(env.num_envs, dtype=np.int32)
        else:
            ids = np.flatnonzero(self._needs_reset)
        if ids.size == 0:
            return self

        reset_ids = ids[self._needs_reset[ids]]
        self.dof_pos[ids] = self._entity.data.joint_pos[ids][:, self._joint_ids_array]
        self.object_pos[ids] = self._entity.data.root_link_pos_w[ids]
        self.object_quat[ids] = self._entity.data.root_link_quat_w[ids]
        self.dof_vel[ids] = (self.dof_pos[ids] - self._previous_dof_pos[ids]) / env.step_dt
        self.object_linvel[ids] = (
            self.object_pos[ids] - self._previous_object_pos[ids]
        ) / env.step_dt
        self.object_angvel[ids] = np_quat_angular_velocity_from_pair(
            self.object_quat[ids], self._previous_object_quat[ids], env.step_dt
        )
        self.dof_vel[reset_ids] = 0.0
        self.object_linvel[reset_ids] = 0.0
        self.object_angvel[reset_ids] = 0.0
        action = self._resolve_action(env)
        if action is None:
            self.torques[ids] = 0.0
        else:
            self.torques[ids] = (
                self.domain_randomization.kp[ids] * (action.target[ids] - self.dof_pos[ids])
                - self.domain_randomization.kd[ids] * self.dof_vel[ids]
            )
        self._previous_dof_pos[ids] = self.dof_pos[ids]
        self._previous_object_pos[ids] = self.object_pos[ids]
        self._previous_object_quat[ids] = self.object_quat[ids]

        # Read contact forces once. Only the Actor branch sees artificial
        # smoothing, latency and dropout; both branches describe this step.
        self._clean_tactile[ids] = self._read_tactile(env)[ids]
        if self._enable_tactile:
            self._tactile_output[ids] = self._process_tactile(env, ids)
        normalized = 2.0 * (self.dof_pos[ids] - self._dof_mid) / (self._dof_range + 1.0e-8)
        target = action.target[ids] if action is not None else np.zeros_like(normalized)
        clean_parts = [normalized, target]
        noisy = normalized.copy()
        if self._joint_noise > 0.0:
            noisy += env.rng.uniform(-self._joint_noise, self._joint_noise, size=noisy.shape)
        actor_parts = [noisy, target]
        if self._enable_tactile:
            clean_parts.append(self._clean_tactile[ids])
            actor_parts.append(self._tactile_output[ids])
        self._clean_frame[ids] = np.concatenate(clean_parts, axis=1)
        self._actor_frame[ids] = np.concatenate(actor_parts, axis=1)
        if self._clip_obs > 0.0:
            for frame in (self._clean_frame, self._actor_frame):
                frame[ids] = np.clip(frame[ids], -self._clip_obs, self._clip_obs)
        self.priv_info[ids] = self.domain_randomization.privileged_info(
            self.object_pos, self.task_state.object_pos_anchor
        )[ids]
        self._needs_reset[ids] = False
        self._last_counter = counter
        return self

    def _read_tactile(self, env: SharpaEnv) -> np.ndarray:
        dtype = get_global_dtype()
        raw = self._sensor_view.read()
        result = np.zeros((env.num_envs, len(self._sensor_view.names)), dtype=dtype)
        offset = 0
        for index, width in enumerate(self._sensor_dims):
            values = np.asarray(raw[:, offset : offset + width], dtype=dtype)
            result[:, index] = np.linalg.norm(values[:, :3], axis=1) if width >= 3 else values[:, 0]
            offset += width
        if self._tactile_clip > 0.0:
            np.clip(result, 0.0, self._tactile_clip, out=result)
        return result

    def _process_tactile(self, env: SharpaEnv, ids: np.ndarray) -> np.ndarray:
        current = self._clean_tactile[ids]
        smooth = current * self._contact_smoothing + self._prev_tactile_force[ids] * (
            1.0 - self._contact_smoothing
        )
        self._prev_tactile_force[ids] = current
        delayed = env.rng.random(smooth.shape) < self._contact_latency
        contact = (
            (smooth > self._contact_threshold).astype(get_global_dtype())
            if self._binary_contact
            else smooth
        )
        self.last_contacts[ids] = np.where(delayed, self.last_contacts[ids], contact)
        result = self.last_contacts[ids].copy()
        if self._binary_contact:
            keep = env.rng.random(result.shape) >= self._contact_noise
            result = np.where(result > 0.1, keep * result, result)
        return result

    def frame(self, env: SharpaEnv, *, noisy: bool) -> np.ndarray:
        self.snapshot(env)
        return self._actor_frame if noisy else self._clean_frame

    def __call__(self, env: SharpaEnv, **params: Any) -> np.ndarray:
        del params
        return self.frame(env, noisy=True)


class SharpaFlattenedObservation(SharpaRotationObservation):
    """Legacy noisy policy frame plus privileged channels."""

    def __call__(self, env: SharpaEnv, **params: Any) -> np.ndarray:
        del params
        return np.concatenate((self.frame(env, noisy=True), self.priv_info), axis=1)


class _SharpaObservationView(ManagerTermBase):
    """Read a named frame owner without another sensor read or noise draw."""

    output_dim: int

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(term, cfg, {"observation_group", "observation_term"})
        self._group = require_name(term, "observation_group", cfg.params.get("observation_group"))
        self._term = require_name(term, "observation_term", cfg.params.get("observation_term"))

    def __call__(self, env: SharpaEnv, **params: Any) -> np.ndarray:
        del params
        # ObservationManager probes dimensions before env.observation_manager
        # is assigned. Dependencies are resolved on the first real compute.
        manager = getattr(env, "observation_manager", None)
        if manager is None:
            return np.zeros((env.num_envs, self.output_dim), dtype=get_global_dtype())
        source = manager.get_term_cfg(self._group, self._term).func
        if not isinstance(source, SharpaRotationObservation) or source.frame_dim != FRAME_DIM:
            raise ValueError(f"{type(self).__name__} requires a 49D Sharpa frame owner")
        return self._read(source.snapshot(env))

    def _read(self, source: SharpaRotationObservation) -> np.ndarray:
        raise NotImplementedError


class SharpaCriticObservation(_SharpaObservationView):
    """Current clean 49D frame plus current 9D privileged information."""

    output_dim = FRAME_DIM + PRIV_DIM

    def _read(self, source: SharpaRotationObservation) -> np.ndarray:
        return np.concatenate((source._clean_frame, source.priv_info), axis=1)


class SharpaPrivilegedObservation(_SharpaObservationView):
    """Explicit current privileged vector, independent of critic history layout."""

    output_dim = PRIV_DIM

    def _read(self, source: SharpaRotationObservation) -> np.ndarray:
        return source.priv_info


class SharpaProprioObservation(_SharpaObservationView):
    """Same noisy frame as the Actor; configure a 30-frame manager history."""

    output_dim = FRAME_DIM

    def _read(self, source: SharpaRotationObservation) -> np.ndarray:
        return source._actor_frame
