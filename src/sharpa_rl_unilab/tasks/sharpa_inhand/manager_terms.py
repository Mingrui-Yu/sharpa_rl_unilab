"""Sharpa Wave in-hand Manager-Based terms.

The terms in this module use UniLab's Manager-Based API and Entity facade only.
They intentionally do not inspect a backend model, qpos layout, or native data
object.  Fixed object-size variants are declared by the registered environment
configuration and materialized by UniLab's fixed-variant plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from unilab.base.run_control import RunComplete
from unilab.dtype_config import get_global_dtype
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    ManagerTermBase,
    ManagerTermBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from unilab.utils.geometry import (
    np_normalize_axis,
    np_quat_angular_velocity_from_pair,
    np_sample_uniform_quaternion,
)
from unilab.utils.rotation import (
    np_quat_apply,
    np_quat_error_magnitude,
)

from sharpa_rl_unilab.assets import cache_root, resolve_asset

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.event_manager import EventManager
    from unilab.managers.observation_manager import ObservationManager
    from unilab.managers.termination_manager import TerminationManager

    class _SharpaEnv(ManagerBasedRlEnv, Protocol):
        @property
        def cfg(self) -> Any: ...

        @property
        def common_step_counter(self) -> int: ...

        @property
        def event_manager(self) -> EventManager: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def observation_manager(self) -> ObservationManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...

        @property
        def reset_time_outs(self) -> np.ndarray: ...

        @property
        def reset_terminated(self) -> np.ndarray: ...

        @property
        def extras(self) -> dict[str, Any]: ...


NUM_HAND_JOINTS = 22
_HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
HAND_JOINT_NAMES: tuple[str, ...] = _HAND_JOINT_NAMES
ACTUATOR_NAMES: tuple[str, ...] = tuple(f"{name}_ctrl" for name in _HAND_JOINT_NAMES)
FINGERTIP_BODY_NAMES: tuple[str, ...] = (
    "right_thumb_DP",
    "right_index_DP",
    "right_middle_DP",
    "right_ring_DP",
    "right_pinky_DP",
)
TACTILE_SENSOR_NAMES: tuple[str, ...] = (
    "contact_right_thumb_elastomer_force",
    "contact_right_index_elastomer_force",
    "contact_right_middle_elastomer_force",
    "contact_right_ring_elastomer_force",
    "contact_right_pinky_elastomer_force",
)
SOURCE_DEFAULT_HAND_JOINT_POS_DEG: tuple[float, ...] = (
    95.12771,
    -3.11244,
    14.81626,
    -1.03493,
    12.23986,
    65.21091,
    6.1133,
    15.58495,
    5.90325,
    31.74149,
    -0.95812,
    41.88173,
    12.844,
    31.72383,
    9.84458,
    35.22366,
    18.02839,
    10.9712,
    68.30895,
    7.99151,
    5.89626,
    5.89875,
)
SOURCE_DEFAULT_HAND_JOINT_POS = np.deg2rad(
    np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64)
)


def _name(term: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {field} must be a non-empty string")
    return value


def _names(term: str, field: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {field} must be a sequence of names")
    result = tuple(_name(term, field, item) for item in value)
    if not result:
        raise ValueError(f"{term} {field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{term} {field} contains duplicate names")
    return result


def _real(
    term: str,
    field: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{term} {field} must be at least {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{term} {field} must be at most {maximum}, got {result}")
    return result


def _int(term: str, field: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{term} {field} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{term} {field} must be positive, got {result}")
    return result


def _bool(term: str, field: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{term} {field} must be boolean")
    return bool(value)


def _pair(term: str, field: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {field} must be a numeric (min, max) pair")
    if len(value) != 2:
        raise ValueError(f"{term} {field} must contain two values")
    lower = _real(term, f"{field}[0]", value[0])
    upper = _real(term, f"{field}[1]", value[1])
    if lower > upper:
        raise ValueError(f"{term} {field} lower bound exceeds upper bound")
    return lower, upper


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return np.asarray(env_ids, dtype=np.int32)


def _validate_params(term: str, cfg: ManagerTermBaseCfg, allowed: set[str]) -> None:
    unexpected = set(cfg.params) - allowed
    if unexpected:
        raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")


def _scale_tag(value: float) -> str:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"object scale must be finite and positive, got {value}")
    return f"{value:g}"


def resolve_grasp_cache_file(prefix: str, scale: float) -> Path:
    """Resolve the package/writable-cache path for one fixed object scale."""
    prefix_path = Path(prefix)
    if prefix_path.suffix == ".npy":
        path = prefix_path.with_name(f"{prefix_path.stem}_{_scale_tag(scale)}.npy")
    else:
        path = Path(f"{prefix}_{_scale_tag(scale)}.npy")
    if path.is_absolute():
        return path
    try:
        return Path(resolve_asset(str(path)))
    except FileNotFoundError:
        return cache_root() / path


def sample_scale_grasp_caches(
    caches: tuple[np.ndarray, ...], variant_ids: np.ndarray
) -> np.ndarray:
    """Sample one cached grasp for each environment from its fixed variant."""
    if not caches:
        raise ValueError("at least one grasp cache is required")
    for index, cache in enumerate(caches):
        cache = np.asarray(cache)
        if cache.ndim != 2 or cache.shape[1] != 29 or cache.shape[0] == 0:
            raise ValueError(f"grasp cache {index} must have shape (N, 29), got {cache.shape}")
        if not np.isfinite(cache).all():
            raise ValueError(f"grasp cache {index} contains NaN or Inf")
    sampled = np.zeros((len(variant_ids), 29), dtype=np.float64)
    for variant_id, cache in enumerate(caches):
        ids = np.flatnonzero(variant_ids == variant_id)
        if ids.size == 0:
            continue
        choices = np.random.randint(0, np.asarray(cache).shape[0], size=ids.size)
        sampled[ids] = np.asarray(cache)[choices]
    return sampled


@dataclass(kw_only=True)
class SharpaIncrementalPositionActionCfg(ActionTermCfg):
    actuator_names: tuple[str, ...] | list[str]
    action_scale: float
    raw_action_clip: tuple[float, float] | list[float]
    dof_limits_scale: float = 0.9
    zero_action: bool = False

    def build(self, env: ManagerBasedRlEnv) -> SharpaIncrementalPositionAction:
        return SharpaIncrementalPositionAction(self, env)


class SharpaIncrementalPositionAction(ActionTerm):
    """Integrate bounded policy deltas into hand position targets."""

    cfg: SharpaIncrementalPositionActionCfg
    _entity: Entity

    def __init__(self, cfg: SharpaIncrementalPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(f"{term} uses raw_action_clip, not actuator-name clip")
        names = _names(term, "actuator_names", cfg.actuator_names)
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(list(HAND_JOINT_NAMES))
        actuator_ids, matched_actuators = self._entity.find_actuators(
            list(names), preserve_order=True
        )
        if tuple(joint_names) != _HAND_JOINT_NAMES or tuple(matched_actuators) != ACTUATOR_NAMES:
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete or out of order")
        if len(joint_ids) != NUM_HAND_JOINTS:
            raise ValueError(f"{term} requires {NUM_HAND_JOINTS} actuators, got {len(joint_ids)}")

        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._joint_ids.setflags(write=False)
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[self._actuator_ids, 0], copy=True)
        self._ctrl_upper = np.array(ranges[self._actuator_ids, 1], copy=True)
        if np.any(self._ctrl_lower >= self._ctrl_upper):
            raise ValueError(f"{term} actuator control ranges must have lower < upper")
        self._action_scale = _real(term, "action_scale", cfg.action_scale, minimum=0.0)
        self._raw_clip = _pair(term, "raw_action_clip", cfg.raw_action_clip)
        self._dof_limits_scale = _real(
            term, "dof_limits_scale", cfg.dof_limits_scale, minimum=0.0
        )
        self._zero_action = _bool(term, "zero_action", cfg.zero_action)
        if self._dof_limits_scale == 0.0:
            raise ValueError(f"{term} dof_limits_scale must be positive")

        dtype = get_global_dtype()
        self._raw_action = np.zeros((env.num_envs, NUM_HAND_JOINTS), dtype=dtype)
        self._clipped_action = np.zeros_like(self._raw_action)
        self._target = np.broadcast_to(
            SOURCE_DEFAULT_HAND_JOINT_POS.astype(dtype), (env.num_envs, NUM_HAND_JOINTS)
        ).copy()
        self._target_lower = self._ctrl_lower * self._dof_limits_scale
        self._target_upper = self._ctrl_upper * self._dof_limits_scale

    @property
    def action_dim(self) -> int:
        return NUM_HAND_JOINTS

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def ctrl_lower(self) -> np.ndarray:
        return self._ctrl_lower

    @property
    def ctrl_upper(self) -> np.ndarray:
        return self._ctrl_upper

    @property
    def joint_ids(self) -> np.ndarray:
        return self._joint_ids

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"expected np.ndarray actions, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("received NaN or Inf actions")
        self._raw_action[:] = actions
        if self._zero_action:
            actions = np.zeros_like(actions)
        np.clip(actions, self._raw_clip[0], self._raw_clip[1], out=self._clipped_action)
        self._target += self._action_scale * self._clipped_action
        np.clip(self._target, self._target_lower, self._target_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._clipped_action[ids] = 0.0
        self._target[ids] = self._entity.data.joint_pos[ids][:, self._joint_ids]


class SharpaHandObjectReset(ManagerTermBase):
    """Reset hand/object state and own task-reset caches."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        _validate_params(
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
        self._entity = cast("Entity", env.scene[_name(term, "entity_name", cfg.params.get("entity_name"))])
        actuator_names = _names(term, "actuator_names", cfg.params.get("actuator_names", [".*_ctrl"]))
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(list(HAND_JOINT_NAMES))
        actuator_ids, matched_actuators = self._entity.find_actuators(
            list(actuator_names), preserve_order=True
        )
        if tuple(joint_names) != _HAND_JOINT_NAMES or tuple(matched_actuators) != ACTUATOR_NAMES:
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete or out of order")
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        actuator_indices = np.asarray(actuator_ids, dtype=np.intp)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[actuator_indices, 0], copy=True)
        self._ctrl_upper = np.array(ranges[actuator_indices, 1], copy=True)
        self._grasp_generation = _bool(
            term, "grasp_generation", cfg.params.get("grasp_generation", False)
        )
        self._grasp_pose_noise = _real(
            term, "grasp_pose_noise", cfg.params.get("grasp_pose_noise", 0.15), minimum=0.0
        )
        self._reset_height_band = _real(
            term, "reset_height_band", cfg.params.get("reset_height_band", 0.04)
        )
        axis = np.asarray(cfg.params.get("rotation_axis", (0.0, 0.0, 1.0)), dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.any(axis):
            raise ValueError(f"{term} rotation_axis must be a finite non-zero 3-D vector")
        self._rotation_axis = np_normalize_axis(axis).astype(get_global_dtype())
        self._use_default_anchor = _bool(
            term,
            "use_default_object_pose_for_object_pos_anchor",
            cfg.params.get("use_default_object_pose_for_object_pos_anchor", False),
        )

        if self._entity.data.default_root_state.shape != (env.num_envs, 13):
            raise ValueError(f"{term} requires a 13-D floating object root state")
        self.object_default_pose = np.asarray(
            self._entity.data.default_root_state[:, :7], dtype=get_global_dtype()
        ).copy()
        self.reset_height_lower = np.full(
            (env.num_envs,), np.inf, dtype=get_global_dtype()
        )
        self.reset_height_upper = np.full(
            (env.num_envs,), -np.inf, dtype=get_global_dtype()
        )
        self.object_pos_anchor = np.zeros((env.num_envs, 3), dtype=get_global_dtype())
        self.rotation_axis = np.broadcast_to(
            self._rotation_axis, (env.num_envs, 3)
        ).copy()

        self._variant_ids = self._resolve_variant_ids(env)
        self._scale_values = self._resolve_variant_scales(env)
        self._grasp_caches: tuple[np.ndarray, ...] | None = None
        if not self._grasp_generation:
            prefix = _name(
                term, "grasp_cache_path", cfg.params.get("grasp_cache_path")
            )
            caches: list[np.ndarray] = []
            missing: list[str] = []
            for scale in self._scale_values:
                path = resolve_grasp_cache_file(prefix, float(scale))
                if not path.is_file():
                    missing.append(str(path))
                    continue
                caches.append(np.load(path).astype(np.float64))
            if missing:
                raise FileNotFoundError(
                    f"{term} missing grasp cache(s): {', '.join(missing)}"
                )
            self._grasp_caches = tuple(caches)

    def _resolve_variant_ids(self, env: _SharpaEnv) -> np.ndarray:
        scene = env.cfg.scene
        plan = None if scene is None else scene.fixed_variant_plan
        if plan is None:
            return np.zeros(env.num_envs, dtype=np.int32)
        return np.asarray(plan.assignment, dtype=np.int32)

    def _resolve_variant_scales(self, env: _SharpaEnv) -> np.ndarray:
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

    def _sample_rows(
        self, count: int, env: ManagerBasedRlEnv, variant_ids: np.ndarray
    ) -> np.ndarray:
        if self._grasp_generation:
            hand = np.broadcast_to(
                SOURCE_DEFAULT_HAND_JOINT_POS, (count, NUM_HAND_JOINTS)
            ).copy()
            if self._grasp_pose_noise > 0.0:
                hand += env.rng.uniform(
                    -self._grasp_pose_noise,
                    self._grasp_pose_noise,
                    size=hand.shape,
                )
            np.clip(hand, self._ctrl_lower, self._ctrl_upper, out=hand)
            root_pose = np.broadcast_to(
                self.object_default_pose[0], (count, 7)
            ).copy()
            return np.concatenate((hand, root_pose), axis=1)
        assert self._grasp_caches is not None
        return sample_scale_grasp_caches(self._grasp_caches, variant_ids)

    def __call__(
        self,
        env: _SharpaEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
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


class SharpaDomainRandomization(ManagerTermBase):
    """Own all reset-time physical DR and its privileged-value caches."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        _validate_params(
            term,
            cfg,
            {
                "entity_name",
                "object_body_name",
                "object_geom_name",
                "randomize_pd_gains",
                "p_gain_scale_range",
                "d_gain_scale_range",
                "randomize_friction",
                "friction_scale_range",
                "elastomer_base_friction",
                "metal_base_friction",
                "object_base_friction",
                "randomize_mass",
                "mass_range",
                "randomize_com",
                "com_range",
                "randomize_gravity_direction",
                "gravity_magnitude",
                "include_friction_scale",
                "include_gravity_direction",
            },
        )
        self._entity = cast("Entity", env.scene[_name(term, "entity_name", cfg.params.get("entity_name"))])
        self._randomize_pd = _bool(
            term, "randomize_pd_gains", cfg.params.get("randomize_pd_gains", True)
        )
        self._p_range = _pair(
            term, "p_gain_scale_range", cfg.params.get("p_gain_scale_range", (0.5, 2.0))
        )
        self._d_range = _pair(
            term, "d_gain_scale_range", cfg.params.get("d_gain_scale_range", (0.5, 2.0))
        )
        self._randomize_friction = _bool(
            term, "randomize_friction", cfg.params.get("randomize_friction", True)
        )
        self._friction_range = _pair(
            term, "friction_scale_range", cfg.params.get("friction_scale_range", (0.75, 1.25))
        )
        self._friction_bases = {
            material: _real(
                term,
                f"{material}_base_friction",
                cfg.params.get(f"{material}_base_friction", 1.0),
                minimum=0.0,
            )
            for material in ("elastomer", "metal", "object")
        }
        self._randomize_mass = _bool(
            term, "randomize_mass", cfg.params.get("randomize_mass", True)
        )
        self._mass_range = _pair(term, "mass_range", cfg.params.get("mass_range", (0.01, 0.25)))
        if self._mass_range[0] <= 0.0:
            raise ValueError(f"{term} mass_range must be positive")
        self._randomize_com = _bool(
            term, "randomize_com", cfg.params.get("randomize_com", True)
        )
        self._com_range = _pair(term, "com_range", cfg.params.get("com_range", (-0.01, 0.01)))
        self._randomize_gravity_direction = _bool(
            term,
            "randomize_gravity_direction",
            cfg.params.get("randomize_gravity_direction", False),
        )
        self._gravity_magnitude = _real(
            term,
            "gravity_magnitude",
            cfg.params.get("gravity_magnitude", 9.81),
            minimum=0.0,
        )
        self._include_friction = _bool(
            term, "include_friction_scale", cfg.params.get("include_friction_scale", True)
        )
        self._include_gravity = _bool(
            term, "include_gravity_direction", cfg.params.get("include_gravity_direction", False)
        )

        # Hand actuator gains.
        self._actuator_ids, bound_kp, bound_kd = self._entity.bind_actuator_gain_write(
            None, term_name=term
        )
        dtype = get_global_dtype()
        bound_kp_array = np.asarray(bound_kp, dtype=dtype)
        bound_kd_array = np.asarray(bound_kd, dtype=dtype)
        self._default_kp = (
            bound_kp_array[0] if bound_kp_array.ndim > 1 else bound_kp_array
        ).reshape(NUM_HAND_JOINTS)
        self._default_kd = (
            bound_kd_array[0] if bound_kd_array.ndim > 1 else bound_kd_array
        ).reshape(NUM_HAND_JOINTS)
        self.kp = np.broadcast_to(self._default_kp, (env.num_envs, NUM_HAND_JOINTS)).copy()
        self.kd = np.broadcast_to(self._default_kd, (env.num_envs, NUM_HAND_JOINTS)).copy()

        # Object body mass/CoM bindings.
        object_body = _name(term, "object_body_name", cfg.params.get("object_body_name", "object"))
        body_ids, body_names = self._entity.find_bodies([object_body])
        if tuple(body_names) != (object_body,) or len(body_ids) != 1:
            raise ValueError(f"{term} could not resolve the unique object body")
        self._object_body_ids = np.asarray(body_ids, dtype=np.intp)
        self._object_body_ids.setflags(write=False)
        _, default_mass = self._entity.bind_body_mass_write(
            self._object_body_ids, term_name=term
        )
        _, default_ipos = self._entity.bind_body_ipos_write(
            self._object_body_ids, term_name=term
        )
        default_mass_array = np.asarray(default_mass, dtype=dtype)
        default_ipos_array = np.asarray(default_ipos, dtype=dtype)
        self._default_mass = (
            default_mass_array[0, ...] if default_mass_array.ndim > 1 else default_mass_array
        ).reshape(())
        self._default_ipos = (
            default_ipos_array[0, 0, :] if default_ipos_array.ndim == 3 else default_ipos_array.reshape(3)
        )
        self.mass = np.full((env.num_envs,), float(self._default_mass), dtype=dtype)
        self.com_offset = np.zeros((env.num_envs, 3), dtype=dtype)
        default_gravity = np.asarray(env.scene.bind_gravity_write(term_name=term), dtype=dtype)
        self._default_gravity = (
            default_gravity[0] if default_gravity.ndim > 1 else default_gravity
        ).reshape(3)
        self.gravity = np.broadcast_to(self._default_gravity, (env.num_envs, 3)).copy()
        self.friction_scale = np.ones((env.num_envs, 1), dtype=dtype)

        # Collision-geom friction bindings. Entity names are public facade data;
        # material classification never touches backend model arrays.
        object_geom = _name(term, "object_geom_name", cfg.params.get("object_geom_name", "object"))
        object_geom_ids, object_geoms = self._entity.find_geoms([object_geom])
        if tuple(object_geoms) != (object_geom,) or len(object_geom_ids) != 1:
            raise ValueError(f"{term} could not resolve the unique object geom")
        elastomer_ids, _ = self._entity.find_geoms([".*_elastomer$"])
        all_geom_names = self._entity.geom_names
        metal_ids = [
            index
            for index, name in enumerate(all_geom_names)
            if name.startswith("right_")
            and not name.endswith("_visual")
            and not name.endswith("_elastomer")
        ]
        if not elastomer_ids or not metal_ids:
            raise ValueError(f"{term} could not resolve Sharpa collision-geom materials")
        selected = np.asarray(
            [int(object_geom_ids[0]), *map(int, elastomer_ids), *metal_ids], dtype=np.intp
        )
        if np.unique(selected).size != selected.size:
            raise ValueError(f"{term} selected duplicate friction geoms")
        self._material_ids = {
            "object": np.asarray(object_geom_ids, dtype=np.intp),
            "elastomer": np.asarray(elastomer_ids, dtype=np.intp),
            "metal": np.asarray(metal_ids, dtype=np.intp),
        }
        self._friction_geom_ids = selected
        self._friction_geom_ids.setflags(write=False)
        _, bound_friction = self._entity.bind_geom_friction_write(selected, term_name=term)
        bound_friction_array = np.asarray(bound_friction, dtype=np.float64)
        self._default_friction = (
            bound_friction_array[0]
            if bound_friction_array.ndim == 3
            else bound_friction_array
        ).reshape(selected.size, 3)
        self._friction_material = np.full(selected.size, -1, dtype=np.int8)
        material_numbers = {"object": 0, "elastomer": 1, "metal": 2}
        selected_set = {int(value) for value in selected}
        for material, ids in self._material_ids.items():
            for geom_id in ids:
                geom_id = int(geom_id)
                if geom_id not in selected_set:
                    raise ValueError(f"{term} friction binding is inconsistent")
                self._friction_material[np.flatnonzero(selected == geom_id)] = material_numbers[material]
        if np.any(self._friction_material < 0):
            raise ValueError(f"{term} friction material binding is incomplete")

        scale_catalog = env.cfg.fixed_model_variants
        self.scale = np.ones((env.num_envs, 1), dtype=dtype)
        if scale_catalog is not None and env.cfg.scene is not None:
            if env.cfg.scene.fixed_variant_plan is None:
                raise ValueError(f"{term} fixed variant plan was not materialized")
            variant_index = {
                variant.name: index for index, variant in enumerate(scale_catalog.variants)
            }
            scales = np.asarray(
                [
                    float(variant.name.removeprefix("scene_scale_").replace("_", "."))
                    for variant in scale_catalog.variants
                ],
                dtype=dtype,
            )
            ids = np.asarray(env.cfg.scene.fixed_variant_plan.assignment, dtype=np.intp)
            self.scale[:] = scales[ids][:, None]
            del variant_index

    @property
    def privileged_dim(self) -> int:
        return 9 + (3 if self._include_gravity else 0)

    def _sample_split_around_one(
        self, rng: np.random.Generator, bounds: tuple[float, float], shape: tuple[int, int]
    ) -> np.ndarray:
        lower, upper = bounds
        if lower > 1.0 or upper < 1.0:
            raise ValueError(f"gain scale bounds must satisfy lower <= 1 <= upper: {bounds}")
        small = rng.uniform(lower, 1.0, size=shape)
        large = rng.uniform(1.0, upper, size=shape)
        return np.where(rng.random(shape) > 0.5, small, large).astype(get_global_dtype())

    def __call__(
        self,
        env: _SharpaEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        if ids.size == 0:
            return
        dtype = get_global_dtype()
        count = ids.size

        if self._randomize_pd:
            kp = np.broadcast_to(
                np.asarray(self._default_kp, dtype=dtype), (count, NUM_HAND_JOINTS)
            ).copy()
            kd = np.broadcast_to(
                np.asarray(self._default_kd, dtype=dtype), (count, NUM_HAND_JOINTS)
            ).copy()
            kp *= self._sample_split_around_one(env.rng, self._p_range, (count, NUM_HAND_JOINTS))
            kd *= self._sample_split_around_one(env.rng, self._d_range, (count, NUM_HAND_JOINTS))
        else:
            kp = np.broadcast_to(self.kp[ids], (count, NUM_HAND_JOINTS)).copy()
            kd = np.broadcast_to(self.kd[ids], (count, NUM_HAND_JOINTS)).copy()
        self._entity.write_actuator_gains_to_sim(
            kp,
            kd,
            actuator_ids=self._actuator_ids,
            env_ids=ids,
            term_name=type(self).__name__,
        )
        self.kp[ids] = kp
        self.kd[ids] = kd

        if self._randomize_friction:
            self.friction_scale[ids] = env.rng.uniform(
                self._friction_range[0], self._friction_range[1], size=(count, 1)
            )
            friction = np.asarray(self._default_friction, dtype=np.float64).copy()
            friction = np.broadcast_to(friction, (count, friction.shape[-2], 3)).copy()
            scale = self.friction_scale[ids, 0][:, None]
            for material_id, material in enumerate(("object", "elastomer", "metal")):
                mask = self._friction_material == material_id
                template = friction[:, mask, :].copy()
                sliding = template[:, :, 0]
                if np.any(sliding <= 0.0):
                    raise ValueError(f"{material} default sliding friction must be positive")
                profile = template / sliding[:, :, None] * self._friction_bases[material]
                friction[:, mask, :] = profile * scale[:, None]
            self._entity.write_geom_friction_to_sim(
                np.asarray(friction, dtype=dtype),
                geom_ids=self._friction_geom_ids,
                env_ids=ids,
                term_name=type(self).__name__,
            )
        else:
            self.friction_scale[ids] = 1.0

        if self._randomize_mass:
            mass = env.rng.uniform(
                self._mass_range[0], self._mass_range[1], size=count
            ).astype(dtype)
        else:
            mass = np.full((count,), float(self._default_mass), dtype=dtype)
        self._entity.write_body_mass_to_sim(
            mass[:, None],
            body_ids=self._object_body_ids,
            env_ids=ids,
            term_name=type(self).__name__,
        )
        self.mass[ids] = mass

        if self._randomize_com:
            com = env.rng.uniform(
                self._com_range[0], self._com_range[1], size=(count, 3)
            ).astype(dtype)
        else:
            com = np.zeros((count, 3), dtype=dtype)
        default_ipos = np.broadcast_to(
            np.asarray(self._default_ipos, dtype=np.float64).reshape(1, 3), (count, 3)
        ).copy()
        self._entity.write_body_ipos_to_sim(
            np.asarray(default_ipos + com, dtype=dtype)[:, None, :],
            body_ids=self._object_body_ids,
            env_ids=ids,
            term_name=type(self).__name__,
        )
        self.com_offset[ids] = com

        if self._randomize_gravity_direction:
            downward = np.zeros((count, 3), dtype=np.float64)
            downward[:, 2] = -self._gravity_magnitude
            quat = np_sample_uniform_quaternion(count)
            gravity = np_quat_apply(quat, downward).astype(dtype)
        else:
            gravity = np.broadcast_to(self._default_gravity, (count, 3)).copy()
        env.scene.write_gravity_to_sim(gravity, ids, term_name=type(self).__name__)
        self.gravity[ids] = gravity

    def privileged_info(self, object_pos: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        object_pos = np.asarray(object_pos, dtype=get_global_dtype())
        anchor = np.asarray(anchor, dtype=get_global_dtype())
        parts = [object_pos - anchor]
        if self._include_friction:
            parts.append(self.friction_scale)
        parts.extend((self.mass[:, None], self.com_offset, self.scale))
        if self._include_gravity:
            parts.append(self.gravity)
        return np.concatenate(parts, axis=1, dtype=get_global_dtype())


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
        "disable_tactile_ids",
        "enable_tactile",
        "clip_obs",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        _validate_params(term, cfg, self._ALLOWED_PARAMS)
        self._entity = cast("Entity", env.scene[_name(term, "entity_name", cfg.params.get("entity_name"))])
        self._action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._action: SharpaIncrementalPositionAction | None = None
        self._joint_ids_array, joint_names = self._entity.find_joints_by_actuator_names(
            list(HAND_JOINT_NAMES)
        )
        if tuple(joint_names) != _HAND_JOINT_NAMES:
            raise ValueError(f"{term} could not resolve all Sharpa hand joints in order")
        self._joint_ids_array = np.asarray(self._joint_ids_array, dtype=np.intp)
        state_name = _name(term, "event_state_name", cfg.params.get("event_state_name"))
        state = env.event_manager.get_term_cfg(state_name).func
        if not isinstance(state, SharpaHandObjectReset):
            raise TypeError(f"{term} event state must be SharpaHandObjectReset")
        self._state = state
        dr_name = _name(
            term, "domain_randomization_name", cfg.params.get("domain_randomization_name")
        )
        dr = env.event_manager.get_term_cfg(dr_name).func
        if not isinstance(dr, SharpaDomainRandomization):
            raise TypeError(f"{term} DR state must be SharpaDomainRandomization")
        self._dr = dr

        sensor_names = _names(
            term, "sensor_names", cfg.params.get("sensor_names", TACTILE_SENSOR_NAMES)
        )
        self._sensor_view = env.scene.bind_sensor_data(sensor_names)
        self._sensor_dims = self._sensor_view.dimensions
        self._enable_tactile = _bool(
            term, "enable_tactile", cfg.params.get("enable_tactile", True)
        )
        self._joint_noise = _real(
            term, "joint_noise", cfg.params.get("joint_noise", 0.02), minimum=0.0
        )
        self._contact_smoothing = _real(
            term, "contact_smoothing", cfg.params.get("contact_smoothing", 0.5), minimum=0.0
        )
        self._contact_latency = _real(
            term, "contact_latency", cfg.params.get("contact_latency", 0.005), minimum=0.0
        )
        self._contact_noise = _real(
            term, "contact_sensor_noise", cfg.params.get("contact_sensor_noise", 0.01), minimum=0.0
        )
        self._binary_contact = _bool(
            term, "binary_contact", cfg.params.get("binary_contact", False)
        )
        self._contact_threshold = _real(
            term, "contact_threshold", cfg.params.get("contact_threshold", 0.05), minimum=0.0
        )
        self._tactile_clip = _real(
            term, "tactile_force_clip_max", cfg.params.get("tactile_force_clip_max", 4.0)
        )
        self._clip_obs = _real(term, "clip_obs", cfg.params.get("clip_obs", 5.0))
        disabled = np.asarray(
            cfg.params.get("disable_tactile_ids", ()), dtype=np.intp
        )
        if np.any(disabled < 0) or np.any(disabled >= len(sensor_names)):
            raise ValueError(f"{term} disable_tactile_ids are out of range: {disabled.tolist()}")

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

        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        actuator_indices = np.arange(NUM_HAND_JOINTS, dtype=np.intp)
        self._dof_mid = (ranges[actuator_indices, 1] + ranges[actuator_indices, 0]) / 2.0
        self._dof_range = ranges[actuator_indices, 1] - ranges[actuator_indices, 0]

    def _resolve_action(self, env: _SharpaEnv) -> SharpaIncrementalPositionAction | None:
        if self._action is None:
            action_manager = getattr(env, "action_manager", None)
            if action_manager is None:
                return None
            action = action_manager.get_term(self._action_name)
            if not isinstance(action, SharpaIncrementalPositionAction):
                raise TypeError(f"{type(self).__name__} action must be SharpaIncrementalPositionAction")
            self._action = action
        return self._action

    @property
    def last_counter(self) -> int:
        return self._last_counter

    @property
    def frame_dim(self) -> int:
        return NUM_HAND_JOINTS * 2 + (len(self._sensor_view.names) if self._enable_tactile else 0)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self.dof_pos[ids] = self._entity.data.joint_pos[ids][:, self._joint_ids_array]
        self.dof_vel[ids] = 0.0
        self.object_pos[ids] = self._entity.data.root_link_pos_w[ids]
        self.object_quat[ids] = self._entity.data.root_link_quat_w[ids]
        self.object_linvel[ids] = 0.0
        self.object_angvel[ids] = 0.0
        self.torques[ids] = 0.0
        self._previous_dof_pos[ids] = self.dof_pos[ids]
        self._previous_object_pos[ids] = self.object_pos[ids]
        self._previous_object_quat[ids] = self.object_quat[ids]
        self._prev_tactile_force[ids] = 0.0
        self.last_contacts[ids] = 0.0
        self._tactile_output[ids] = 0.0
        self._last_counter = int(cast("_SharpaEnv", self._env).common_step_counter)

    def snapshot(self, env: _SharpaEnv) -> SharpaRotationObservation:
        counter = int(env.common_step_counter)
        if counter == self._last_counter:
            return self
        if counter != self._last_counter + 1:
            raise RuntimeError(
                f"Sharpa observation missed a control-step update: "
                f"last={self._last_counter}, current={counter}"
            )
        dtype = get_global_dtype()
        dof_pos = np.asarray(self._entity.data.joint_pos[:, self._joint_ids_array], dtype=dtype)
        object_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=dtype)
        object_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype)
        np.subtract(dof_pos, self._previous_dof_pos, out=self.dof_vel)
        self.dof_vel /= env.step_dt
        np.subtract(object_pos, self._previous_object_pos, out=self.object_linvel)
        self.object_linvel /= env.step_dt
        self.object_angvel[:] = np_quat_angular_velocity_from_pair(
            object_quat, self._previous_object_quat, env.step_dt
        )
        self.dof_pos[:] = dof_pos
        self.object_pos[:] = object_pos
        self.object_quat[:] = object_quat
        action = self._resolve_action(env)
        if action is None:
            self.torques[:] = 0.0
        else:
            self.torques[:] = self._dr.kp * (action.target - self.dof_pos)
            self.torques -= self._dr.kd * self.dof_vel
        self._previous_dof_pos[:] = dof_pos
        self._previous_object_pos[:] = object_pos
        self._previous_object_quat[:] = object_quat
        self._tactile_output = self.tactile(env)
        self._last_counter = counter
        return self

    def _read_tactile(self, env: _SharpaEnv) -> np.ndarray:
        dtype = get_global_dtype()
        raw = self._sensor_view.read()
        result = np.zeros((env.num_envs, len(self._sensor_view.names)), dtype=dtype)
        offset = 0
        for index, width in enumerate(self._sensor_dims):
            values = np.asarray(raw[:, offset : offset + width], dtype=dtype)
            result[:, index] = (
                np.linalg.norm(values[:, :3], axis=1)
                if width >= 3
                else values[:, 0]
            )
            offset += width
        if self._tactile_clip > 0.0:
            np.clip(result, 0.0, self._tactile_clip, out=result)
        return result

    def tactile(self, env: _SharpaEnv) -> np.ndarray:
        if not self._enable_tactile:
            return np.zeros((env.num_envs, 0), dtype=get_global_dtype())
        current = self._read_tactile(env)
        smooth = current * self._contact_smoothing + self._prev_tactile_force * (
            1.0 - self._contact_smoothing
        )
        self._prev_tactile_force[:] = current
        latency = (
            env.rng.random(smooth.shape) < self._contact_latency
        ).astype(get_global_dtype())
        if self._binary_contact:
            contact = (smooth > self._contact_threshold).astype(get_global_dtype())
            self.last_contacts[:] = self.last_contacts * latency + contact * (1.0 - latency)
            keep = (
                env.rng.random(self.last_contacts.shape)
                >= self._contact_noise
            ).astype(get_global_dtype())
            return np.where(
                self.last_contacts > 0.1,
                keep * self.last_contacts,
                self.last_contacts,
            )
        self.last_contacts[:] = self.last_contacts * latency + smooth * (1.0 - latency)
        return self.last_contacts.copy()

    def frame(self, env: _SharpaEnv, *, noisy: bool) -> np.ndarray:
        self.snapshot(env)
        normalized = 2.0 * (self.dof_pos - self._dof_mid) / (self._dof_range + 1.0e-8)
        if noisy and self._joint_noise > 0.0:
            normalized = normalized + env.rng.uniform(
                -self._joint_noise, self._joint_noise, size=normalized.shape
            )
        action = self._resolve_action(env)
        target = action.target if action is not None else np.zeros_like(normalized)
        parts = [normalized, target]
        if self._enable_tactile:
            parts.append(self._tactile_output)
        frame = np.concatenate(parts, axis=1, dtype=get_global_dtype())
        if self._clip_obs > 0.0:
            np.clip(frame, -self._clip_obs, self._clip_obs, out=frame)
        return frame

    def __call__(self, env: _SharpaEnv, **params: Any) -> np.ndarray:
        del params
        return self.frame(env, noisy=True)


class SharpaFlattenedObservation(SharpaRotationObservation):
    """Noisy policy frame plus privileged channels in one flattened group."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(cfg, env)
        self._include_privileged = True
        self._noisy_output = True

    def __call__(self, env: _SharpaEnv, **params: Any) -> np.ndarray:
        del params
        return np.concatenate(
            (
                self.frame(env, noisy=self._noisy_output),
                self._dr.privileged_info(
                    self.object_pos,
                    self._state.object_pos_anchor,
                ),
            ),
            axis=1,
            dtype=get_global_dtype(),
        )


class SharpaCriticObservation(SharpaRotationObservation):
    """Clean policy frame plus privileged channels for asymmetric learners."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(cfg, env)
        self._include_privileged = True
        self._noisy_output = False

    def __call__(self, env: _SharpaEnv, **params: Any) -> np.ndarray:
        del params
        return np.concatenate(
            (
                self.frame(env, noisy=self._noisy_output),
                self._dr.privileged_info(
                    self.object_pos,
                    self._state.object_pos_anchor,
                ),
            ),
            axis=1,
            dtype=get_global_dtype(),
        )


class SharpaDropTermination(ManagerTermBase):
    """Terminate when the object leaves its sampled reset-height band."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        _validate_params(term, cfg, {"observation_group", "observation_term"})
        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, SharpaRotationObservation):
            raise TypeError(f"{term} requires SharpaRotationObservation")
        self.observation = observation
        self.dropped = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self.dropped[_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("_SharpaEnv", self._env).common_step_counter)

    def __call__(self, env: _SharpaEnv, **params: Any) -> np.ndarray:
        del params
        state = self.observation.snapshot(env)
        self.dropped[:] = (
            (state.object_pos[:, 2] > self.observation._state.reset_height_upper)
            | (state.object_pos[:, 2] < self.observation._state.reset_height_lower)
        )
        self._last_counter = int(env.common_step_counter)
        return self.dropped


def _rotation_state(
    env: _SharpaEnv, state_term_name: str
) -> tuple[SharpaDropTermination, SharpaRotationObservation]:
    state = env.termination_manager.get_term_cfg(
        _name("Sharpa reward", "state_term_name", state_term_name)
    ).func
    if not isinstance(state, SharpaDropTermination):
        raise TypeError("Sharpa reward state term must be SharpaDropTermination")
    if state.last_counter != int(env.common_step_counter):
        raise RuntimeError("Sharpa reward state was not computed for the current control step")
    return state, state.observation


def rotate_reward(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(
        np.clip(
            np.sum(state.object_angvel * state._state.rotation_axis, axis=1),
            -0.5,
            0.5,
        ),
        dtype=get_global_dtype(),
    )


def object_linear_velocity_l1(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.abs(state.object_linvel), axis=1), dtype=get_global_dtype())


def hand_pose_deviation_l2(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    default = np.broadcast_to(
        SOURCE_DEFAULT_HAND_JOINT_POS.astype(get_global_dtype()),
        state.dof_pos.shape,
    )
    return np.asarray(np.sum(np.square(state.dof_pos - default), axis=1), dtype=get_global_dtype())


def estimated_torque_l2(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.square(state.torques), axis=1), dtype=get_global_dtype())


def estimated_work_l2(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(
        np.square(np.sum(state.torques * state.dof_vel, axis=1)), dtype=get_global_dtype()
    )


def object_position_reward(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    delta = state.object_pos - state._state.object_pos_anchor
    return np.asarray(1.0 / (np.linalg.norm(delta, axis=1) + 0.001), dtype=get_global_dtype())


def dropped(env: _SharpaEnv, state_term_name: str) -> np.ndarray:
    state, _ = _rotation_state(env, state_term_name)
    return np.asarray(state.dropped, dtype=get_global_dtype())


class SharpaPersistentObjectForce(ManagerTermBase):
    """Source-equivalent decaying random object force."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        _validate_params(
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
        self._entity = cast("Entity", env.scene[_name(term, "entity_name", cfg.params.get("entity_name"))])
        self._dr_name = _name(
            term, "domain_randomization_name", cfg.params.get("domain_randomization_name")
        )
        self._dr: SharpaDomainRandomization | None = None
        self._force_scale = _real(
            term, "force_scale", cfg.params.get("force_scale", 2.0), minimum=0.0
        )
        self._probability = _real(
            term, "force_probability", cfg.params.get("force_probability", 0.25), minimum=0.0
        )
        decay = _real(term, "force_decay", cfg.params.get("force_decay", 0.9), minimum=0.0)
        interval = _real(
            term,
            "force_decay_interval",
            cfg.params.get("force_decay_interval", 0.08),
            minimum=0.0,
        )
        if interval == 0.0:
            raise ValueError(f"{term} force_decay_interval must be positive")
        self._decay = float(np.power(decay, env.step_dt / interval))
        body_name = _name(term, "object_body_name", cfg.params.get("object_body_name", "object"))
        requested_body_ids, requested_body_names = self._entity.find_bodies([body_name])
        if tuple(requested_body_names) != (body_name,):
            raise ValueError(f"{term} could not resolve the unique object body")
        local_ids, backend_ids = self._entity.bind_body_wrench(
            np.asarray(requested_body_ids, dtype=np.intp), torque=False, term_name=term
        )
        self._backend_body_ids = np.asarray(backend_ids, dtype=np.intp)
        self._backend_body_ids.setflags(write=False)
        self._force = np.zeros((env.num_envs, 1, 3), dtype=np.float64)
        del local_ids

    def __call__(
        self,
        env: _SharpaEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del env_ids, params
        if self._dr is None:
            dr = env.event_manager.get_term_cfg(self._dr_name).func
            if not isinstance(dr, SharpaDomainRandomization):
                raise TypeError(f"{type(self).__name__} requires SharpaDomainRandomization")
            self._dr = dr
        self._force *= self._decay
        trigger = env.rng.random(env.num_envs) < self._probability
        if np.any(trigger):
            ids = np.flatnonzero(trigger)
            mass = self._dr.mass[ids]
            self._force[ids, 0, :] = env.rng.standard_normal((ids.size, 3)) * mass[:, None]
            self._force[ids] *= self._force_scale
        self._entity.apply_body_wrench_to_sim(
            self._force,
            None,
            self._backend_body_ids,
            env_ids=None,
            term_name=type(self).__name__,
        )

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._force[ids] = 0.0
        if ids.size:
            self._entity.apply_body_wrench_to_sim(
                self._force[ids],
                None,
                self._backend_body_ids,
                env_ids=ids,
                term_name=type(self).__name__,
            )


class SharpaGraspReset(SharpaHandObjectReset):
    """Grasp-generation reset with cache loading disabled."""


class SharpaGraspQualityTermination(ManagerTermBase):
    """Reject timeouts unless the object is a stable multi-finger grasp."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        _validate_params(
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
        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, SharpaRotationObservation):
            raise TypeError(f"{term} requires SharpaRotationObservation")
        self.observation = observation
        state = env.event_manager.get_term_cfg(
            _name(term, "event_state_name", cfg.params.get("event_state_name"))
        ).func
        if not isinstance(state, SharpaHandObjectReset):
            raise TypeError(f"{term} requires SharpaHandObjectReset")
        self._state = state
        self._height_band = _real(
            term, "reset_height_band", cfg.params.get("reset_height_band", 0.01)
        )
        self._orientation_error = _real(
            term,
            "maximum_orientation_error",
            cfg.params.get("maximum_orientation_error", np.pi / 6.0),
            minimum=0.0,
        )
        self._minimum_contacts = _int(
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
        self.invalid[_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("_SharpaEnv", self._env).common_step_counter)

    def __call__(self, env: _SharpaEnv, **params: Any) -> np.ndarray:
        del params
        state = self.observation.snapshot(env)
        height = state.object_pos[:, 2]
        center = self._state.object_default_pose[:, 2]
        contacts = (
            state._tactile_output
            if self.observation._enable_tactile
            else np.zeros((env.num_envs, 0))
        )
        enough_contacts = np.sum(contacts > 0.5, axis=1) >= self._minimum_contacts
        orientation_error = np_quat_error_magnitude(
            self._state.object_default_pose[:, 3:7], state.object_quat
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

    def __init__(self, cfg: RecorderTermCfg, env: _SharpaEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        _validate_params(
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
        self._entity = cast("Entity", env.scene[_name(term, "entity_name", cfg.params.get("entity_name"))])
        self._state = env.event_manager.get_term_cfg(
            _name(term, "event_state_name", cfg.params.get("event_state_name"))
        ).func
        if not isinstance(self._state, SharpaHandObjectReset):
            raise TypeError(f"{term} requires SharpaHandObjectReset")
        quality = env.termination_manager.get_term_cfg(
            _name(term, "termination_term_name", cfg.params.get("termination_term_name"))
        ).func
        if not isinstance(quality, SharpaGraspQualityTermination):
            raise TypeError(f"{term} requires SharpaGraspQualityTermination")
        self._quality = quality
        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, SharpaRotationObservation):
            raise TypeError(f"{term} requires SharpaRotationObservation")
        self._observation = observation
        self._prefix = _name(term, "output_prefix", cfg.params.get("output_prefix"))
        self._target = _int(term, "target", cfg.params.get("target"), positive=True)
        self._auto_save = _bool(term, "auto_save", cfg.params.get("auto_save", True))
        self._rows: list[np.ndarray] = []
        self._saved = False
        self._target_notified = False

    @property
    def total_saved(self) -> int:
        return len(self._rows)

    def _log(self, name: str, value: float) -> None:
        env = cast("_SharpaEnv", self._env)
        log = env.extras.setdefault("log", {})
        log[name] = value

    def _save(self, *, force: bool = False) -> None:
        if self._saved or not self._rows or (not force and len(self._rows) < self._target):
            return
        rows = np.concatenate(self._rows, axis=0)[: self._target].astype(np.float32)
        scale = float(np.asarray(self._state._scale_values)[0])
        output = resolve_grasp_cache_file(self._prefix, scale)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, rows)
        self._saved = True
        self._log("grasp_cache/saved", 1.0)
        self._log("grasp_cache/num_states", float(rows.shape[0]))

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("_SharpaEnv", self._env)
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
                self._rows.append(rows[: self._target - self.total_saved])
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


__all__ = [
    "ACTUATOR_NAMES",
    "FINGERTIP_BODY_NAMES",
    "HAND_JOINT_NAMES",
    "SOURCE_DEFAULT_HAND_JOINT_POS",
    "SOURCE_DEFAULT_HAND_JOINT_POS_DEG",
    "TACTILE_SENSOR_NAMES",
    "SharpaCriticObservation",
    "SharpaDomainRandomization",
    "SharpaDropTermination",
    "SharpaFlattenedObservation",
    "SharpaGraspQualityTermination",
    "SharpaGraspRecorder",
    "SharpaGraspReset",
    "SharpaHandObjectReset",
    "SharpaIncrementalPositionAction",
    "SharpaIncrementalPositionActionCfg",
    "SharpaPersistentObjectForce",
    "SharpaRotationObservation",
    "dropped",
    "estimated_torque_l2",
    "estimated_work_l2",
    "hand_pose_deviation_l2",
    "object_linear_velocity_l1",
    "object_position_reward",
    "resolve_grasp_cache_file",
    "rotate_reward",
    "sample_scale_grasp_caches",
]
