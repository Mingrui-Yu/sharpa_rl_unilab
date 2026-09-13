"""Registry owner for Sharpa grasp-cache generation."""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .config import SharpaInhandRotationGraspCfg

registry.register_env_config("SharpaInhandRotationGrasp", SharpaInhandRotationGraspCfg)
for _backend_type in ("mujoco", "motrix"):
    registry.register_env(
        "SharpaInhandRotationGrasp", make_manager_based_rl_env, sim_backend=_backend_type
    )

__all__ = ["SharpaInhandRotationGraspCfg"]
