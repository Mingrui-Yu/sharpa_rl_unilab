"""Reset term for the Sharpa hand and fixed-scale manipulated object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.action import SharpaIncrementalPositionAction
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import (
    resolve_grasp_cache_file,
    sample_scale_grasp_caches,
    validate_grasp_caches,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.compat import (
    entity_default_root_state,
    entity_host_data,
    manager_tensor,
    numpy_dtype,
    rng_uniform,
    torch_ids,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import (
    ACTUATOR_NAMES,
    HAND_JOINT_NAMES,
    NUM_HAND_JOINTS,
    SOURCE_DEFAULT_HAND_JOINT_POS,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.geometry import np_normalize_axis
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
        self._action_name = require_name(term, "action_name", cfg.params.get("action_name"))
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
        ranges = np.asarray(
            entity_host_data(self._entity, "actuator_ctrl_range"),
            dtype=numpy_dtype(get_global_dtype()),
        )
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
        self._rotation_axis = np_normalize_axis(axis).astype(numpy_dtype(get_global_dtype()))
        self._use_default_anchor = require_bool(
            term,
            "use_default_object_pose_for_object_pos_anchor",
            cfg.params.get("use_default_object_pose_for_object_pos_anchor", False),
        )

        entity_default_state = entity_default_root_state(self._entity, env.device)
        if entity_default_state.shape != (env.num_envs, 13):
            raise ValueError(f"{term} requires a 13-D floating object root state")
        self.object_default_pose = np.asarray(
            entity_default_state[:, :7], dtype=numpy_dtype(get_global_dtype())
        ).copy()
        self._object_default_pose_tensor = torch.as_tensor(
            self.object_default_pose, dtype=get_global_dtype(), device=env.device
        )
        self.reset_height_lower = torch.full(
            (env.num_envs,), torch.inf, dtype=get_global_dtype(), device=env.device
        )
        self.reset_height_upper = torch.full(
            (env.num_envs,), -torch.inf, dtype=get_global_dtype(), device=env.device
        )
        self.object_pos_anchor = torch.zeros(
            (env.num_envs, 3), dtype=get_global_dtype(), device=env.device
        )
        self.rotation_axis = torch.as_tensor(
            np.broadcast_to(self._rotation_axis, (env.num_envs, 3)).copy(),
            dtype=get_global_dtype(),
            device=env.device,
        )

        self._variant_ids = self._resolve_variant_ids(env)
        self.scale_values = self._resolve_variant_scales(env)
        self._grasp_caches: tuple[np.ndarray, ...] | None = None
        self.cache_row_indices = np.full(env.num_envs, -1, dtype=np.int64)
        self._sampled_indices = np.empty(0, dtype=np.int64)
        if not self._grasp_generation:
            prefix = require_name(term, "grasp_cache_path", cfg.params.get("grasp_cache_path"))
            caches: list[np.ndarray] = []
            missing: list[str] = []
            for scale in self.scale_values:
                try:
                    path = resolve_grasp_cache_file(prefix, float(scale))
                except FileNotFoundError:
                    missing.append(f"{prefix} (scale={scale:g})")
                    continue
                caches.append(np.load(path).astype(np.float64))
            if missing:
                raise FileNotFoundError(f"{term} missing grasp cache(s): {', '.join(missing)}")
            self._grasp_caches = tuple(caches)
            validate_grasp_caches(self._grasp_caches)

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
                hand += rng_uniform(
                    env.rng, -self._grasp_pose_noise, self._grasp_pose_noise, hand.shape
                ).astype(hand.dtype, copy=False)
            np.clip(hand, self._ctrl_lower, self._ctrl_upper, out=hand)
            root_pose = np.broadcast_to(self.object_default_pose[0], (count, 7)).copy()
            return np.concatenate((hand, root_pose), axis=1)
        assert self._grasp_caches is not None
        self._sampled_indices = np.empty(count, dtype=np.int64)
        return sample_scale_grasp_caches(
            self._grasp_caches, variant_ids, rng=env.rng, selected_rows=self._sampled_indices
        )

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
        if not self._grasp_generation:
            self.cache_row_indices[ids] = self._sampled_indices
        hand_pos = np.asarray(rows[:, :NUM_HAND_JOINTS], dtype=numpy_dtype(get_global_dtype()))
        root_pose = np.asarray(rows[:, NUM_HAND_JOINTS:], dtype=numpy_dtype(get_global_dtype()))
        # UniLab 1.5 dispatches Manager reset terms through a fixed-width mask
        # transaction. Publish full-width tables so the packed reset composer can
        # select rows without a deprecated host env_ids transaction.
        selected_rows = torch_ids(ids, env.device)
        reset_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        reset_mask[selected_rows] = True
        hand_table = torch.zeros(
            (env.num_envs, NUM_HAND_JOINTS), dtype=torch.float32, device=env.device
        )
        hand_table[selected_rows] = torch.as_tensor(
            hand_pos, dtype=torch.float32, device=env.device
        )
        root_state = manager_tensor(env, entity_default_root_state(self._entity, env.device))
        root_state[selected_rows, :7] = torch.as_tensor(
            root_pose, dtype=torch.float32, device=env.device
        )
        root_state[selected_rows, 7:] = 0.0
        self._entity.write_joint_state_to_sim_mask(
            hand_table,
            torch.zeros_like(hand_table),
            reset_mask,
            # The packed state tables stay on the env device, while this legacy
            # entity API still normalizes its static joint-id selector on host.
            joint_ids=tuple(int(value) for value in self._joint_ids),
        )
        # UniLab <=1.2 reset events committed before ActionManager.reset(), so
        # the incremental-position owner synchronized its target to the newly
        # sampled grasp. UniLab 1.5 clears action state before reset events;
        # restore that post-grasp target explicitly at this boundary.
        action = env.action_manager.get_term(self._action_name)
        if not isinstance(action, SharpaIncrementalPositionAction):
            raise TypeError(f"{type(self).__name__} requires SharpaIncrementalPositionAction")
        action.reset_to_positions(hand_table[selected_rows], selected_rows)
        self._entity.write_root_state_tensor_to_sim(root_state, reset_mask)

        root_pose_tensor = torch.as_tensor(root_pose, dtype=get_global_dtype(), device=env.device)
        self.reset_height_lower[selected_rows] = (
            root_pose_tensor[:, 2] - 0.5 * self._reset_height_band
        )
        self.reset_height_upper[selected_rows] = (
            root_pose_tensor[:, 2] + 0.5 * self._reset_height_band
        )
        if self._use_default_anchor:
            anchor = torch.as_tensor(
                self.object_default_pose[ids, :3], dtype=get_global_dtype(), device=env.device
            )
        else:
            anchor = root_pose_tensor[:, :3]
        self.object_pos_anchor[selected_rows] = anchor
