"""Reset-time physical domain randomization and privileged state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase
from unilab.utils.geometry import np_sample_uniform_quaternion
from unilab.utils.rotation import np_quat_apply

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import NUM_HAND_JOINTS
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_bool,
    require_name,
    require_pair,
    require_real,
    resolve_env_ids,
    validate_term_params,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers import ManagerTermBaseCfg

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


class SharpaDomainRandomization(ManagerTermBase):
    """Own all reset-time physical DR and its privileged-value caches."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(
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
        self._entity = cast(
            "Entity", env.scene[require_name(term, "entity_name", cfg.params.get("entity_name"))]
        )
        self._randomize_pd = require_bool(
            term, "randomize_pd_gains", cfg.params.get("randomize_pd_gains", True)
        )
        self._p_range = require_pair(
            term, "p_gain_scale_range", cfg.params.get("p_gain_scale_range", (0.5, 2.0))
        )
        self._d_range = require_pair(
            term, "d_gain_scale_range", cfg.params.get("d_gain_scale_range", (0.5, 2.0))
        )
        self._randomize_friction = require_bool(
            term, "randomize_friction", cfg.params.get("randomize_friction", True)
        )
        self._friction_range = require_pair(
            term, "friction_scale_range", cfg.params.get("friction_scale_range", (0.75, 1.25))
        )
        self._friction_bases = {
            material: require_real(
                term,
                f"{material}_base_friction",
                cfg.params.get(f"{material}_base_friction", 1.0),
                minimum=0.0,
            )
            for material in ("elastomer", "metal", "object")
        }
        self._randomize_mass = require_bool(
            term, "randomize_mass", cfg.params.get("randomize_mass", True)
        )
        self._mass_range = require_pair(
            term, "mass_range", cfg.params.get("mass_range", (0.01, 0.25))
        )
        if self._mass_range[0] <= 0.0:
            raise ValueError(f"{term} mass_range must be positive")
        self._randomize_com = require_bool(
            term, "randomize_com", cfg.params.get("randomize_com", True)
        )
        self._com_range = require_pair(
            term, "com_range", cfg.params.get("com_range", (-0.01, 0.01))
        )
        self._randomize_gravity_direction = require_bool(
            term,
            "randomize_gravity_direction",
            cfg.params.get("randomize_gravity_direction", False),
        )
        self._gravity_magnitude = require_real(
            term,
            "gravity_magnitude",
            cfg.params.get("gravity_magnitude", 9.81),
            minimum=0.0,
        )
        self._include_friction = require_bool(
            term, "include_friction_scale", cfg.params.get("include_friction_scale", True)
        )
        self._include_gravity = require_bool(
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
        object_body = require_name(
            term, "object_body_name", cfg.params.get("object_body_name", "object")
        )
        body_ids, body_names = self._entity.find_bodies([object_body])
        if tuple(body_names) != (object_body,) or len(body_ids) != 1:
            raise ValueError(f"{term} could not resolve the unique object body")
        self._object_body_ids = np.asarray(body_ids, dtype=np.intp)
        self._object_body_ids.setflags(write=False)
        _, default_mass = self._entity.bind_body_mass_write(self._object_body_ids, term_name=term)
        _, default_ipos = self._entity.bind_body_ipos_write(self._object_body_ids, term_name=term)
        default_mass_array = np.asarray(default_mass, dtype=dtype)
        default_ipos_array = np.asarray(default_ipos, dtype=dtype)
        self._default_mass = (
            default_mass_array[0, ...] if default_mass_array.ndim > 1 else default_mass_array
        ).reshape(())
        self._default_ipos = (
            default_ipos_array[0, 0, :]
            if default_ipos_array.ndim == 3
            else default_ipos_array.reshape(3)
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
        object_geom = require_name(
            term, "object_geom_name", cfg.params.get("object_geom_name", "object")
        )
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
            bound_friction_array[0] if bound_friction_array.ndim == 3 else bound_friction_array
        ).reshape(selected.size, 3)
        self._friction_material = np.full(selected.size, -1, dtype=np.int8)
        material_numbers = {"object": 0, "elastomer": 1, "metal": 2}
        selected_set = {int(value) for value in selected}
        for material, ids in self._material_ids.items():
            for geom_id in ids:
                geom_id = int(geom_id)
                if geom_id not in selected_set:
                    raise ValueError(f"{term} friction binding is inconsistent")
                self._friction_material[np.flatnonzero(selected == geom_id)] = material_numbers[
                    material
                ]
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
        env: SharpaEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = resolve_env_ids(env, env_ids)
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
            mass = env.rng.uniform(self._mass_range[0], self._mass_range[1], size=count).astype(
                dtype
            )
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
            com = env.rng.uniform(self._com_range[0], self._com_range[1], size=(count, 3)).astype(
                dtype
            )
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
