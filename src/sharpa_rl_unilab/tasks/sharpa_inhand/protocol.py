"""Fixed HORA v2 observation layout, shared by environments and policies."""

from omegaconf import DictConfig, OmegaConf

from .terms.constants import NUM_HAND_JOINTS, TACTILE_SENSOR_NAMES

CONTRACT_VERSION = "sharpa-hora-v2"
ACTION_DIM = NUM_HAND_JOINTS
FRAME_DIM = ACTION_DIM * 2 + len(TACTILE_SENSOR_NAMES)
ACTOR_HISTORY = 3
HISTORY_LENGTH = 30
HISTORY_SHAPE = (HISTORY_LENGTH, FRAME_DIM)
PRIV_DIM = 9
ACTOR_DIM = FRAME_DIM * ACTOR_HISTORY
CRITIC_DIM = (FRAME_DIM + PRIV_DIM) * ACTOR_HISTORY
OBS_SHAPES = {
    "obs": (ACTOR_DIM,),
    "critic": (CRITIC_DIM,),
    "priv_info": (PRIV_DIM,),
    "proprio_hist": HISTORY_SHAPE,
}


def validate_observation_config(cfg: DictConfig) -> None:
    """Reject incompatible v2 options before constructing simulation or tensors."""
    expected = {
        "env.observations.actor.terms.frame.params.enable_tactile": True,
        "env.observations.actor.terms.frame.params.sensor_names": list(TACTILE_SENSOR_NAMES),
        "env.events.domain_randomization.params.include_friction_scale": True,
        "env.events.domain_randomization.params.include_gravity_direction": False,
        "env.observations.actor.history_length": ACTOR_HISTORY,
        "env.observations.actor.flatten_history_dim": True,
        "env.observations.critic.history_length": ACTOR_HISTORY,
        "env.observations.critic.flatten_history_dim": True,
        "env.observations.proprio_hist.history_length": HISTORY_LENGTH,
        "env.observations.proprio_hist.flatten_history_dim": False,
    }
    for field, required in expected.items():
        value = OmegaConf.select(cfg, field)
        if value != required:
            raise ValueError(f"{CONTRACT_VERSION} requires {field}={required!r}; got {value!r}")
    for group in ("actor", "critic", "priv_info", "proprio_hist"):
        field = f"env.observations.{group}.concatenate_terms"
        value = OmegaConf.select(cfg, field, default=True)
        if value is not True:
            raise ValueError(f"{CONTRACT_VERSION} requires {field}=True; got {value!r}")
    # Privilege has no manager history. An omitted group setting delegates to
    # the term's default, unlike the explicit actor/critic/history settings above.
    field = "env.observations.priv_info.history_length"
    value = OmegaConf.select(cfg, field)
    if value is None:
        field = "env.observations.priv_info.terms.current.history_length"
        value = OmegaConf.select(cfg, field, default=0)
    if value != 0:
        raise ValueError(f"{CONTRACT_VERSION} requires {field}=0; got {value!r}")
