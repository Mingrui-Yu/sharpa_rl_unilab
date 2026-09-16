"""Grasp-generation quality term and cache recorder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from unilab.base.run_control import RunComplete
from unilab.managers import ManagerTermBase, RecorderTerm, RecorderTermCfg
from unilab.utils.rotation import np_quat_error_magnitude

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import grasp_cache_output_file
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.observation import SharpaRotationObservation
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.reset import SharpaHandObjectReset
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_bool,
    require_int,
    require_name,
    require_real,
    resolve_env_ids,
    validate_term_params,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers import ManagerTermBaseCfg

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


class SharpaGraspQualityTermination(ManagerTermBase):
    """Reject timeouts unless the object is a stable multi-finger grasp."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(
            term,
            cfg,
            {
                "observation_group",
                "observation_term",
                "event_state_name",
                "reset_height_band",
                "maximum_orientation_error",
                "minimum_contacts",
            },
        )
        group = require_name(term, "observation_group", cfg.params.get("observation_group"))
        name = require_name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, SharpaRotationObservation):
            raise TypeError(f"{term} requires SharpaRotationObservation")
        self.observation = observation
        state = env.event_manager.get_term_cfg(
            require_name(term, "event_state_name", cfg.params.get("event_state_name"))
        ).func
        if not isinstance(state, SharpaHandObjectReset):
            raise TypeError(f"{term} requires SharpaHandObjectReset")
        self.task_state = state
        self._height_band = require_real(
            term, "reset_height_band", cfg.params.get("reset_height_band", 0.01)
        )
        self._orientation_error = require_real(
            term,
            "maximum_orientation_error",
            cfg.params.get("maximum_orientation_error", np.pi / 6.0),
            minimum=0.0,
        )
        self._minimum_contacts = require_int(
            term,
            "minimum_contacts",
            cfg.params.get("minimum_contacts", 3),
            positive=True,
        )
        self.invalid = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self.invalid[resolve_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("SharpaEnv", self._env).common_step_counter)

    def __call__(self, env: SharpaEnv, **params: Any) -> np.ndarray:
        del params
        state = self.observation.snapshot(env)
        height = state.object_pos[:, 2]
        center = self.task_state.object_default_pose[:, 2]
        contacts = (
            state._tactile_output
            if self.observation._enable_tactile
            else np.zeros((env.num_envs, 0))
        )
        enough_contacts = np.sum(contacts > 0.5, axis=1) >= self._minimum_contacts
        orientation_error = np_quat_error_magnitude(
            self.task_state.object_default_pose[:, 3:7], state.object_quat
        )
        valid = (
            (np.abs(height - center) <= self._height_band)
            & enough_contacts
            & (orientation_error < self._orientation_error)
        )
        self.invalid[:] = ~valid
        self._last_counter = int(env.common_step_counter)
        return self.invalid


class SharpaGraspRecorder(RecorderTerm):
    """Collect valid timeout states into a per-scale grasp cache."""

    def __init__(self, cfg: RecorderTermCfg, env: SharpaEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        validate_term_params(
            term,
            cfg,
            {
                "entity_name",
                "event_state_name",
                "termination_term_name",
                "observation_group",
                "observation_term",
                "output_prefix",
                "target",
                "auto_save",
            },
        )
        self._entity = cast(
            "Entity", env.scene[require_name(term, "entity_name", cfg.params.get("entity_name"))]
        )
        self.task_state = env.event_manager.get_term_cfg(
            require_name(term, "event_state_name", cfg.params.get("event_state_name"))
        ).func
        if not isinstance(self.task_state, SharpaHandObjectReset):
            raise TypeError(f"{term} requires SharpaHandObjectReset")
        quality = env.termination_manager.get_term_cfg(
            require_name(term, "termination_term_name", cfg.params.get("termination_term_name"))
        ).func
        if not isinstance(quality, SharpaGraspQualityTermination):
            raise TypeError(f"{term} requires SharpaGraspQualityTermination")
        self._quality = quality
        group = require_name(term, "observation_group", cfg.params.get("observation_group"))
        name = require_name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, SharpaRotationObservation):
            raise TypeError(f"{term} requires SharpaRotationObservation")
        self._observation = observation
        self._prefix = require_name(term, "output_prefix", cfg.params.get("output_prefix"))
        self._target = require_int(term, "target", cfg.params.get("target"), positive=True)
        self._auto_save = require_bool(term, "auto_save", cfg.params.get("auto_save", True))
        self._rows: list[np.ndarray] = []
        self._num_rows = 0
        self._saved = False
        self._target_notified = False

    @property
    def total_saved(self) -> int:
        return self._num_rows

    def _log(self, name: str, value: float) -> None:
        env = cast("SharpaEnv", self._env)
        log = env.extras.setdefault("log", {})
        log[name] = value

    def _save(self, *, force: bool = False) -> None:
        if self._saved or not self._rows or (not force and self.total_saved < self._target):
            return
        rows = np.concatenate(self._rows, axis=0)[: self._target].astype(np.float32)
        scale = float(np.asarray(self.task_state.scale_values)[0])
        output = grasp_cache_output_file(self._prefix, scale)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, rows)
        self._saved = True
        self._log("grasp_cache/saved", 1.0)
        self._log("grasp_cache/num_states", float(rows.shape[0]))

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("SharpaEnv", self._env)
        ids = np.asarray(env_ids, dtype=np.intp)
        success = env.reset_time_outs[ids] & ~env.reset_terminated[ids]
        success_ids = ids[np.flatnonzero(success)]
        if success_ids.size:
            state = self._observation
            rows = np.concatenate(
                (
                    state.dof_pos[success_ids],
                    state.object_pos[success_ids],
                    state.object_quat[success_ids],
                ),
                axis=1,
                dtype=np.float32,
            )
            if rows.shape != (success_ids.size, 29):
                raise ValueError(f"{type(self).__name__} collected invalid rows")
            if self.total_saved < self._target:
                rows = rows[: self._target - self.total_saved]
                self._rows.append(rows)
                self._num_rows += len(rows)
            self._log("grasp/cache_size", float(self.total_saved))
        self._save()
        if self.total_saved >= self._target and not self._target_notified:
            self._target_notified = True
            self._log("grasp/target_reached", 1.0)
            raise RunComplete(
                reason="grasp_collection_target_reached",
                summary={
                    "collected_grasps": self.total_saved,
                    "grasp_collection_target": self._target,
                },
            )

    def close(self) -> None:
        self._save(force=self._auto_save)
