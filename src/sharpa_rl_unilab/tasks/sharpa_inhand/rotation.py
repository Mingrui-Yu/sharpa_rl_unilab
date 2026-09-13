"""Registry owner for the Sharpa Wave in-hand rotation task."""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .config import SharpaInhandRotationCfg

registry.register_env_config("SharpaInhandRotation", SharpaInhandRotationCfg)
for _backend_type in ("mujoco", "motrix"):
    registry.register_env(
        "SharpaInhandRotation", make_manager_based_rl_env, sim_backend=_backend_type
    )

__all__ = ["SharpaInhandRotationCfg"]
