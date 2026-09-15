from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from uni_rl.algos.common.normalization import EmpiricalNormalization

from sharpa_rl_unilab.algos.hora.teacher import TeacherActor
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import CONTRACT_VERSION
from sharpa_rl_unilab.training import playback


def save_policy(path, cfg, student=False):
    actor = TeacherActor(OmegaConf.to_container(cfg.model, resolve=True), student=student)
    normalizer = EmpiricalNormalization((30, 49), "cpu")
    normalizer(torch.ones(8, 30, 49))
    torch.save(
        {
            "contract": CONTRACT_VERSION,
            "algorithm": "ppo",
            "stage": "student" if student else "teacher",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "actor": actor.state_dict(),
            "history_normalizer": normalizer.state_dict(),
        },
        path,
    )


@pytest.mark.parametrize("student", [False, True])
def test_playback_loads_policy_and_closes_environment(tmp_path, monkeypatch, student):
    cfg = compose_config("ppo", "mujoco", ["training.device=cpu", "training.play_steps=1"])
    path = tmp_path / "policy.pt"
    save_policy(path, cfg, student)
    closed = []
    actions = []
    infer = playback.deterministic_actions

    def check_inference(actor, obs, device, history_normalizer):
        assert not actor.training and not torch.is_grad_enabled()
        if student:
            assert not history_normalizer.training and history_normalizer.count.item() == 8
        return infer(actor, obs, device, history_normalizer)

    monkeypatch.setattr(playback, "deterministic_actions", check_inference)

    def record(**kwargs):
        kwargs["step"](
            {
                "obs": np.ones((2, 147), np.float32),
                "proprio_hist" if student else "priv_info": np.ones(
                    (2, 30, 49) if student else (2, 9), np.float32
                ),
            }
        )
        assert kwargs["output_video"] == tmp_path / "play_video.mp4"
        return kwargs["output_video"]

    env = SimpleNamespace(
        run_playback_mode=record,
        close=lambda: closed.append(True),
        step=lambda action: actions.append(action) or SimpleNamespace(obs={}),
    )
    monkeypatch.setattr(playback, "SharpaTeacherEnv", lambda *_: env)
    assert playback.play_checkpoint(path) == tmp_path / "play_video.mp4"
    assert actions[0].shape == (2, 22) and np.isfinite(actions[0]).all()
    assert closed == [True]

    def fail(**kwargs):
        raise RuntimeError("recording failed")

    env.run_playback_mode = fail
    with pytest.raises(RuntimeError, match="recording failed"):
        playback.play_checkpoint(path)
    assert closed == [True, True]


def test_disabled_playback_does_not_create_environment(tmp_path, monkeypatch):
    cfg = compose_config("ppo", "mujoco", [])
    path = tmp_path / "policy.pt"
    monkeypatch.setattr(playback, "SharpaTeacherEnv", lambda *_: pytest.fail("Playback disabled"))
    for no_play, mode in [(True, "record"), (False, "none")]:
        cfg.training.no_play, cfg.training.play_render_mode = no_play, mode
        save_policy(path, cfg)
        assert playback.play_checkpoint(path) is None
