"""Registry owner for Sharpa grasp-cache generation."""

from unilab.base import registry
from unilab.envs import make_manager_based_rl_env

from .config import SharpaInhandRotationGraspCfg

registry.register_env_config("SharpaInhandRotationGrasp", SharpaInhandRotationGraspCfg)
registry.register_env("SharpaInhandRotationGrasp", make_manager_based_rl_env, sim_backend="mujoco")
