import numpy as np
import pytest
import torch
from tensordict import TensorDict

from sharpa_rl_unilab.algos.hora.teacher import (
    TeacherActor,
    TeacherFlashActor,
    frozen_weights,
    observe_new_samples,
    pack_actor,
)
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.evaluation import deterministic_actions
from sharpa_rl_unilab.training.student_runtime import StudentTrainer
from sharpa_rl_unilab.training.teacher_runtime import (
    Replay,
    config_dict,
    load_policy,
    make_models,
    stage_rollout,
    timeout_rewards,
)


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.set_num_threads(2)
    torch.manual_seed(11)


def observations(n=8):
    return {
        key: torch.randn(n, *shape)
        for key, shape in {
            "obs": (147,),
            "priv_info": (9,),
            "critic": (174,),
            "proprio_hist": (30, 49),
        }.items()
    }


def transitions(n=8):
    obs, nxt = observations(n), observations(n)
    data = {key: obs[key].unsqueeze(0) for key in ("obs", "priv_info", "critic")}
    data.update({"next_" + key: nxt[key].unsqueeze(0) for key in ("obs", "priv_info", "critic")})
    data.update(
        actions=torch.randn(1, n, 22).tanh(),
        rewards=torch.randn(1, n),
        actions_log_prob=torch.zeros(1, n),
        terminated=torch.zeros(1, n),
        truncated=torch.ones(1, n),
    )
    return data


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
def test_actor_information_and_value_gradient_are_independent(algo):
    cfg = compose_config(algo, "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    obs = observations()
    td = TensorDict({"actor": obs["obs"], "priv_info": obs["priv_info"]}, batch_size=8)
    mean, _ = actor.shared.policy_mean(td, prefer_student=False)
    mean.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in actor.shared.priv_encoder.parameters()
    )
    assert all(p.grad is None for p in critic.parameters())
    assert not set(actor.parameters()).intersection(critic.parameters())
    with pytest.raises(ValueError, match="explicit priv_info"):
        actor.shared.policy_mean(
            TensorDict({"actor": obs["obs"]}, batch_size=8), prefer_student=False
        )
    layers = list(actor.shared.priv_encoder.modules())
    assert sum(isinstance(layer, torch.nn.ELU) for layer in layers) == 3
    assert isinstance(layers[-1], torch.nn.ELU)


def test_replay_reencodes_both_raw_privilege_vectors_and_isolates_gradients():
    cfg = compose_config("flashsac", "mujoco", [])
    actor, critic, learner = make_models(cfg, "cpu")
    replay = Replay(8, "cpu")
    raw = transitions()
    replay.add(raw)
    assert replay.data["priv_info"].shape == (8, 9)
    assert replay.data["next_priv_info"].shape == (8, 9)
    assert not any("latent" in key for key in replay.data)
    sampled = replay.sample(8)
    torch.testing.assert_close(sampled["obs"][:, 147:], sampled["priv_info"])
    torch.testing.assert_close(sampled["next_obs"][:, 147:], sampled["next_priv_info"])
    seen = []
    handle = actor.shared.priv_encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    before_actor, before_critic = frozen_weights(actor), frozen_weights(critic)
    learner.update_critic(sampled)
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in before_actor.items())
    assert any(
        not torch.equal(value, critic.state_dict()[key]) for key, value in before_critic.items()
    )
    torch.testing.assert_close(seen[-1], sampled["next_priv_info"])
    before_critic = frozen_weights(critic)
    learner.update_actor(sampled)
    assert all(torch.equal(value, critic.state_dict()[key]) for key, value in before_critic.items())
    assert any(
        not torch.equal(value, actor.state_dict()[key])
        for key, value in before_actor.items()
        if "priv_encoder" in key
    )
    # Native FlashSAC packs current/next inputs to evaluate policy and temperature.
    torch.testing.assert_close(
        seen[-1], torch.cat((sampled["priv_info"], sampled["next_priv_info"]))
    )
    handle.remove()
    assert actor.shared.obs_normalizer.count.item() == 0
    assert all(np.isfinite(value) for value in learner.saturation_metrics.values())


def test_normalization_counts_fresh_batches_once_despite_staging_and_replay():
    cfg = compose_config("appo", "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    stages = None
    obs = {key: value.numpy() for key, value in observations().items()}
    raw = {key: value.numpy() for key, value in transitions().items()}
    for version in range(2):
        stages, size = stage_rollout(stages, raw, obs, version, actor, critic, "cpu")
        assert size == 8
    assert actor.shared.obs_normalizer.count.item() == 16
    assert critic.obs_normalizer.count.item() == 16
    before = frozen_weights(actor.shared.obs_normalizer)
    td = TensorDict({"policy": stages.batch()["observations"][:, :8].flatten(0, 1)}, batch_size=8)
    for _ in range(5):
        actor(td, stochastic_output=True)
        actor.update_normalization(td)
        critic.update_normalization(td)
    assert all(
        torch.equal(value, actor.shared.obs_normalizer.state_dict()[key])
        for key, value in before.items()
    )


def test_timeout_bootstrap_uses_final_clean_value_and_real_termination_wins():
    batch = {
        "next_critic": torch.tensor([[[5.0], [7.0], [11.0]]]),
        "rewards": torch.ones(1, 3),
        "truncated": torch.tensor([[1.0, 1.0, 0.0]]),
        "terminated": torch.tensor([[0.0, 1.0, 0.0]]),
    }
    corrected = timeout_rewards(batch, lambda td: td["critic"], 0.9)
    torch.testing.assert_close(corrected, torch.tensor([[5.5, 1.0, 1.0]]))


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
def test_student_only_trains_history_encoder_then_uses_updated_action(algo):
    cfg = compose_config(algo, "mujoco", [])
    model = config_dict(cfg.model)
    teacher = TeacherFlashActor(model) if algo == "flashsac" else TeacherActor(model)
    obs = observations()
    observe_new_samples(teacher, None, pack_actor(obs), obs["critic"])
    trainer = StudentTrainer(teacher, 0.001, "cpu")
    before = frozen_weights(trainer.actor)
    arrays = {key: value.numpy() for key, value in obs.items()}
    actions, loss = trainer.update_and_act(arrays)
    assert np.isfinite(loss)
    changed = {
        key
        for key, value in trainer.actor.state_dict().items()
        if not torch.equal(value, before[key])
    }
    assert changed and all(key.startswith("shared.adapt_tconv.") for key in changed)
    assert trainer.actor.shared.obs_normalizer.count.item() == 8
    assert trainer.history_normalizer.count.item() == 8
    # Deployment has neither privilege nor critic state.
    measurable = {key: arrays[key] for key in ("obs", "proprio_hist")}
    np.testing.assert_array_equal(
        actions, deterministic_actions(trainer.actor, measurable, "cpu", trainer.history_normalizer)
    )
    assert trainer.history_normalizer.count.item() == 8


def test_old_checkpoint_fails_closed(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="retraining"):
        load_policy(path)


def test_staging_wraparound_and_short_tail_keep_raw_targets():
    cfg = compose_config("appo", "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    obs = {k: v.numpy() for k, v in observations().items()}
    raw = {k: v.numpy().repeat(2, axis=0) for k, v in transitions().items()}
    pool = None
    for version in range(5):
        raw["priv_info"].fill(version)
        pool, size = stage_rollout(pool, raw, obs, version, actor, critic, "cpu", capacity=2)
        assert size == 16
    assert pool.active_count == 2
    batch = pool.batch()
    assert set(batch["behavior_version"].flatten().tolist()) == {3, 4}
    torch.testing.assert_close(batch["observations"][..., 147], batch["behavior_version"])
    before = batch["rewards"].clone()
    batch["rewards"] = timeout_rewards(batch, critic, 0.99)
    torch.testing.assert_close(pool.batch()["rewards"], before)
    assert actor.shared.obs_normalizer.count.item() == 80
    short = {k: v[:1] for k, v in raw.items()}
    pool, size = stage_rollout(pool, short, obs, 5, actor, critic, "cpu", capacity=2)
    assert size == 8 and pool.active_count == 1
    assert pool.batch()["observations"].shape[:2] == (1, 8)
    assert actor.shared.obs_normalizer.count.item() == 88
