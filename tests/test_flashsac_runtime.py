from types import SimpleNamespace

import numpy as np
from unilab.base.np_env import NpEnvState

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import transition_next
from sharpa_rl_unilab.training.flashsac_runtime import FlashTeacherEnv


def test_transport_preserves_terminal_privilege_and_clean_critic():
    def observations(offset):
        return {
            key: np.full((2, width), offset + i, dtype=np.float32)
            for i, (key, width) in enumerate((("obs", 147), ("priv_info", 9), ("critic", 174)))
        }

    state = NpEnvState(
        observations(0),
        np.ones(2),
        np.array([True, False]),
        np.array([False, True]),
        {},
        observations(10),
    )
    env = FlashTeacherEnv.__new__(FlashTeacherEnv)
    env.teacher = SimpleNamespace(step=lambda _: state)
    packed = env.step(np.zeros((2, 22)))
    assert packed.obs["obs"].shape == (2, 156)
    nxt = transition_next(packed)
    np.testing.assert_array_equal(nxt["obs"][:, :147], state.final_observation["obs"])
    np.testing.assert_array_equal(nxt["obs"][:, 147:], state.final_observation["priv_info"])
    np.testing.assert_array_equal(nxt["critic"], state.final_observation["critic"])
    np.testing.assert_array_equal(packed.obs["obs"][:, 147:], state.obs["priv_info"])
