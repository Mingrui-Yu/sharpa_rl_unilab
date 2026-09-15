from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from uni_rl.algos.common.normalization import EmpiricalNormalization

from sharpa_rl_unilab.algos.hora.teacher import TeacherActor, TeacherFlashActor
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import CONTRACT_VERSION, OBS_SHAPES
from sharpa_rl_unilab.training import playback


def save_checkpoint(tmp_path, *, algo="ppo", student=False, overrides=()):
    cfg = compose_config(
        algo,
        "mujoco",
        ["training.device=cpu", "training.play_env_num=2", "training.play_steps=3", *overrides],
    )
    actor_cls = TeacherFlashActor if algo == "flashsac" else TeacherActor
    actor = actor_cls(OmegaConf.to_container(cfg.model, resolve=True), student=student).eval()
    snapshot = {
        "contract": CONTRACT_VERSION,
        "algorithm": algo,
        "stage": "student" if student else "teacher",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "actor": actor.state_dict(),
    }
    if student:
        normalizer = EmpiricalNormalization((30, 49), "cpu")
        normalizer(torch.ones(8, 30, 49))
        snapshot["history_normalizer"] = normalizer.state_dict()
    checkpoint = tmp_path / "final.pt"
    torch.save(snapshot, checkpoint)
    return checkpoint


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
@pytest.mark.parametrize("student", [False, True])
def test_checkpoint_policy_and_normalization_in_playback(tmp_path, monkeypatch, algo, student):
    checkpoint = save_checkpoint(tmp_path, algo=algo, student=student)
    closed = []
    actions = []
    counts = []
    infer = playback.deterministic_actions

    def check_inference(actor, obs, device, history_normalizer):
        assert not actor.training and not torch.is_grad_enabled()
        assert (history_normalizer is not None) == student
        if student:
            assert not history_normalizer.training
            counts.append(history_normalizer.count.item())
        return infer(actor, obs, device, history_normalizer)

    class Env:
        def __init__(self, cfg, num_envs):
            assert num_envs == 2
            self.obs = {
                key: np.ones((num_envs, *shape), np.float32) for key, shape in OBS_SHAPES.items()
            }

        def reset(self, *, seed):
            return self.obs, {}

        def step(self, action):
            actions.append(action)
            return SimpleNamespace(obs=self.obs)

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "record"
            assert kwargs["output_video"] == checkpoint.parent / "play_video.mp4"
            obs = kwargs["initialize"]()
            for _ in range(kwargs["play_steps"]):
                obs = kwargs["step"](obs)
            return str(kwargs["output_video"])

        def close(self):
            closed.append(True)

    monkeypatch.setattr(playback, "SharpaTeacherEnv", Env)
    monkeypatch.setattr(playback, "deterministic_actions", check_inference)
    assert Path(playback.play_checkpoint(checkpoint)).name == "play_video.mp4"
    assert closed == [True]
    assert len(actions) == 3
    assert actions[0].shape == (2, 22)
    assert np.isfinite(actions).all() and np.abs(actions).max() <= 1
    for action in actions[1:]:
        np.testing.assert_array_equal(action, actions[0])
    if student:
        assert counts == [8, 8, 8]


@pytest.mark.parametrize("override", ["training.no_play=true", "training.play_render_mode=none"])
def test_disabled_playback_does_not_create_environment(tmp_path, monkeypatch, override):
    checkpoint = save_checkpoint(tmp_path, overrides=[override])
    monkeypatch.setattr(
        playback, "SharpaTeacherEnv", lambda *_: pytest.fail("Playback was disabled")
    )
    assert playback.play_checkpoint(checkpoint) is None
    assert not (tmp_path / "play_video.mp4").exists()


def test_environment_closes_on_recording_failure(tmp_path, monkeypatch):
    checkpoint = save_checkpoint(tmp_path)
    closed = []

    def fail(**kwargs):
        raise RuntimeError("Encoder failed")

    monkeypatch.setattr(
        playback,
        "SharpaTeacherEnv",
        lambda *_: SimpleNamespace(
            run_playback_mode=fail,
            close=lambda: closed.append(True),
        ),
    )
    with pytest.raises(RuntimeError, match="Encoder failed"):
        playback.play_checkpoint(checkpoint)
    assert closed == [True]


@pytest.mark.parametrize("student", [False, True])
def test_cli_replays_returned_checkpoint_after_training(tmp_path, monkeypatch, student):
    import sys

    from sharpa_rl_unilab import assets, cli
    from sharpa_rl_unilab.training import student_runtime, teacher_runtime

    checkpoint = tmp_path / "final.pt"
    calls = []

    def train(*args, **kwargs):
        calls.append("train")
        return checkpoint

    def play(path, *, device):
        assert calls == ["train"]
        assert path == checkpoint and device == "cpu"
        calls.append("play")

    monkeypatch.setattr(playback, "play_checkpoint", play)
    if student:
        monkeypatch.setattr(student_runtime, "train_student", train)
        monkeypatch.setattr(
            sys, "argv", ["sharpa-distill", "--checkpoint", "teacher.pt", "--device", "cpu"]
        )
        student_runtime.main()
    else:
        monkeypatch.setattr(assets, "ensure_assets", lambda: None)
        monkeypatch.setattr(teacher_runtime, "train_teacher", train)
        cli._main(play=False, argv=["training.device=cpu"])
    assert calls == ["train", "play"]
