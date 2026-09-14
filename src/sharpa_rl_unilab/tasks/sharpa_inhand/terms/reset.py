"""Reset term for the Sharpa hand and fixed-scale manipulated object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase
from unilab.utils.geometry import np_normalize_axis

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import (
    resolve_grasp_cache_file,
    sample_scale_grasp_caches,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import (
    ACTUATOR_NAMES,
    HAND_JOINT_NAMES,
    NUM_HAND_JOINTS,
    SOURCE_DEFAULT_HAND_JOINT_POS,
)
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


class SharpaHandObjectReset(ManagerTermBase):
    """Reset hand/object state and own task-reset caches."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(
            term,
            cfg,
            {
                "entity_name",
                "action_name",
                "actuator_names",
                "grasp_cache_path",
                "reset_height_band",
                "rotation_axis",
                "use_default_object_pose_for_object_pos_anchor",
                "grasp_generation",
                "grasp_pose_noise",
            },
        )
        self._entity = cast(
            "Entity", env.scene[require_name(term, "entity_name", cfg.params.get("entity_name"))]
        )
        actuator_names = require_names(
            term, "actuator_names", cfg.params.get("actuator_names", [".*_ctrl"])
        )
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(list(HAND_JOINT_NAMES))
        actuator_ids, matched_actuators = self._entity.find_actuators(
            list(actuator_names), preserve_order=True
        )
        if tuple(joint_names) != HAND_JOINT_NAMES or tuple(matched_actuators) != ACTUATOR_NAMES:
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete or out of order")
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        actuator_indices = np.asarray(actuator_ids, dtype=np.intp)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[actuator_indices, 0], copy=True)
        self._ctrl_upper = np.array(ranges[actuator_indices, 1], copy=True)
        self._grasp_generation = require_bool(
            term, "grasp_generation", cfg.params.get("grasp_generation", False)
        )
        self._grasp_pose_noise = require_real(
            term, "grasp_pose_noise", cfg.params.get("grasp_pose_noise", 0.15), minimum=0.0
        )
        self._reset_height_band = require_real(
            term, "reset_height_band", cfg.params.get("reset_height_band", 0.04)
        )
        axis = np.asarray(cfg.params.get("rotation_axis", (0.0, 0.0, 1.0)), dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.any(axis):
            raise ValueError(f"{term} rotation_axis must be a finite non-zero 3-D vector")
        self._rotation_axis = np_normalize_axis(axis).astype(get_global_dtype())
        self._use_default_anchor = require_bool(
            term,
            "use_default_object_pose_for_object_pos_anchor",
            cfg.params.get("use_default_object_pose_for_object_pos_anchor", False),
        )

        if self._entity.data.default_root_state.shape != (env.num_envs, 13):
            raise ValueError(f"{term} requires a 13-D floating object root state")
        self.object_default_pose = np.asarray(
            self._entity.data.default_root_state[:, :7], dtype=get_global_dtype()
        ).copy()
        self.reset_height_lower = np.full((env.num_envs,), np.inf, dtype=get_global_dtype())
        self.reset_height_upper = np.full((env.num_envs,), -np.inf, dtype=get_global_dtype())
        self.object_pos_anchor = np.zeros((env.num_envs, 3), dtype=get_global_dtype())
        self.rotation_axis = np.broadcast_to(self._rotation_axis, (env.num_envs, 3)).copy()

        self._variant_ids = self._resolve_variant_ids(env)
        self.scale_values = self._resolve_variant_scales(env)
        self._grasp_caches: tuple[np.ndarray, ...] | None = None
        if not self._grasp_generation:
            prefix = require_name(term, "grasp_cache_path", cfg.params.get("grasp_cache_path"))
            caches: list[np.ndarray] = []
            missing: list[str] = []
            for scale in self.scale_values:
                path = resolve_grasp_cache_file(prefix, float(scale))
                if not path.is_file():
                    missing.append(str(path))
                    continue
                caches.append(np.load(path).astype(np.float64))
            if missing:
                raise FileNotFoundError(f"{term} missing grasp cache(s): {', '.join(missing)}")
            self._grasp_caches = tuple(caches)

    def _resolve_variant_ids(self, env: SharpaEnv) -> np.ndarray:
        scene = env.cfg.scene
        plan = None if scene is None else scene.fixed_variant_plan
        if plan is None:
            return np.zeros(env.num_envs, dtype=np.int32)
        return np.asarray(plan.assignment, dtype=np.int32)

    def _resolve_variant_scales(self, env: SharpaEnv) -> np.ndarray:
        catalog = env.cfg.fixed_model_variants
        if catalog is None:
            return np.asarray((1.0,), dtype=np.float64)
        scales = []
        for variant in catalog.variants:
            prefix = "scene_scale_"
            if not variant.name.startswith(prefix):
                raise ValueError(f"unsupported Sharpa object variant name {variant.name!r}")
            scales.append(float(variant.name.removeprefix(prefix).replace("_", ".")))
        values = np.asarray(scales, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"invalid Sharpa object scales: {values.tolist()}")
        return values

    def _sample_rows(self, count: int, env: SharpaEnv, variant_ids: np.ndarray) -> np.ndarray:
        if self._grasp_generation:
            hand = np.broadcast_to(SOURCE_DEFAULT_HAND_JOINT_POS, (count, NUM_HAND_JOINTS)).copy()
            if self._grasp_pose_noise > 0.0:
                hand += env.rng.uniform(
                    -self._grasp_pose_noise,
                    self._grasp_pose_noise,
                    size=hand.shape,
                )
            np.clip(hand, self._ctrl_lower, self._ctrl_upper, out=hand)
            root_pose = np.broadcast_to(self.object_default_pose[0], (count, 7)).copy()
            return np.concatenate((hand, root_pose), axis=1)
        assert self._grasp_caches is not None
        return sample_scale_grasp_caches(self._grasp_caches, variant_ids)

    def __call__(
        self,
        env: SharpaEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = resolve_env_ids(env, env_ids)
        if ids.size == 0:
            return
        rows = self._sample_rows(ids.size, env, self._variant_ids[ids])
        hand_pos = np.asarray(rows[:, :NUM_HAND_JOINTS], dtype=get_global_dtype())
        root_pose = np.asarray(rows[:, NUM_HAND_JOINTS:], dtype=get_global_dtype())
        # Cache rows are global/default poses. CPU batch env origins are zero; the
        # formal root write remains world-frame for backends that introduce origins.
        self._entity.write_joint_state_to_sim(
            hand_pos,
            np.zeros_like(hand_pos),
            env_ids=ids,
        )
        self._entity.write_root_link_pose_to_sim(root_pose, env_ids=ids)
        self._entity.write_root_link_velocity_to_sim(
            np.zeros((ids.size, 6), dtype=get_global_dtype()), env_ids=ids
        )

        self.reset_height_lower[ids] = root_pose[:, 2] - 0.5 * self._reset_height_band
        self.reset_height_upper[ids] = root_pose[:, 2] + 0.5 * self._reset_height_band
        if self._use_default_anchor:
            self.object_pos_anchor[ids] = self.object_default_pose[ids, :3]
        else:
            self.object_pos_anchor[ids] = root_pose[:, :3]
