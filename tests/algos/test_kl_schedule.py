import copy

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from sharpa_rl_unilab.algos.hora.kl_schedule import (
    TeacherPPO,
    TeacherRolloutStorage,
    adaptive_learning_rate,
)
from sharpa_rl_unilab.algos.hora.teacher import TeacherAPPOLearner
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.teacher_runtime import load_policy, make_models, train_teacher


@pytest.mark.parametrize(
    "kl,expected",
    [
        (0.0, 0.0015),
        (0.009, 0.0015),
        (0.01, 0.001),
        (0.04, 0.001),
        (0.041, 0.001 / 1.5),
        (-1e-8, 0.001),
        (float("nan"), 0.001),
    ],
)
def test_kl_schedule_boundaries(kl, expected):
    assert adaptive_learning_rate(0.001, kl, 0.02, 2, 1.5) == expected
    assert adaptive_learning_rate(0.01, 0, 0.02, 2, 1.5) == 0.01
    assert adaptive_learning_rate(1e-5, 1, 0.02, 2, 1.5) == 1e-5


def ppo_rollout(actor, critic, storage):
    obs = TensorDict({"policy": torch.randn(8, 156), "critic": torch.randn(8, 174)}, [8])
    with torch.no_grad():
        tr = storage.Transition()
        tr.observations = obs
        tr.actions = actor(obs, stochastic_output=True)
        tr.actions_log_prob = actor.get_output_log_prob(tr.actions)
        tr.distribution_params = actor.output_distribution_params
        tr.values = critic(obs)
        tr.rewards, tr.dones = torch.randn(8), torch.zeros(8)
        storage.add_transition(tr)
    return obs


def appo_rollout(actor):
    obs, critic = torch.randn(2, 8, 156), torch.randn(2, 8, 174)
    with torch.no_grad():
        sample = actor.policy(obs.flatten(0, 1)).sample()
    return dict(
        observations=obs,
        critic=critic,
        actions=sample.raw.reshape(2, 8, 22),
        actions_log_prob=sample.log_prob.reshape(2, 8),
        rewards=torch.randn(2, 8),
        dones=torch.zeros(2, 8),
        last_obs=obs[-1],
        last_critic=critic[-1],
    )


@pytest.mark.parametrize("algo", ["ppo", "appo"])
@pytest.mark.parametrize("parameterization", ["log", "direct"])
@pytest.mark.parametrize("schedule", ["adaptive", "fixed"])
def test_zero_kl_after_first_step_and_lr_carried_across_updates(algo, parameterization, schedule):
    cfg = compose_config(algo, "mujoco", [f"model.std_parameterization={parameterization}"])
    actor, critic, _ = make_models(cfg, "cpu")
    options = dict(num_learning_epochs=2, num_mini_batches=2, schedule=schedule, desired_kl=0.02)
    if algo == "ppo":
        obs = TensorDict({"policy": torch.zeros(8, 156), "critic": torch.zeros(8, 174)}, [8])
        storage = TeacherRolloutStorage("rl", 8, 1, obs, [22], "cpu")
        learner = TeacherPPO(actor, critic, storage, **options)
        factor = 1.5
    else:
        learner = TeacherAPPOLearner(actor=actor, critic=critic, **options)
        factor = 1.1
    rates = []

    # Exercise real update loops with zero parameter movement: KL stays exactly zero.
    def keep_parameters(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = None

    def record_rate(optimizer, args, kwargs):
        rates.append(optimizer.param_groups[0]["lr"])

    learner.optimizer.register_step_pre_hook(keep_parameters)
    learner.optimizer.register_step_post_hook(record_rate)
    expected = []
    lr = learner.learning_rate
    for _ in range(2):
        for minibatch in range(4):
            if schedule == "adaptive" and minibatch:
                lr = min(0.01, lr * factor)
            expected.append(lr)
        if algo == "ppo":
            learner.compute_returns(ppo_rollout(actor, critic, storage))
            learner.update()
            assert storage.current_batch is None
        else:
            learner.update(learner.process_batch(appo_rollout(actor)))
        assert learner.schedule == schedule
    assert rates == pytest.approx(expected)


@pytest.mark.parametrize("tau,frequency", [(1.0, 2), (0.5, 1)])
def test_appo_skips_only_after_full_target_copy(tau, frequency):
    cfg = compose_config("appo", "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    learner = TeacherAPPOLearner(
        actor=actor,
        critic=critic,
        schedule="adaptive",
        tau=tau,
        target_update_freq=frequency,
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    rates = []

    def no_movement(optimizer, args, kwargs):
        rates.append(learner.learning_rate)
        optimizer.zero_grad(set_to_none=True)

    learner.optimizer.register_step_pre_hook(no_movement)
    for _ in range(3):
        learner.update(learner.process_batch(appo_rollout(actor)))
    expected = [0.001, 0.0011, 0.0011 if tau == 1 else 0.00121]
    assert rates == pytest.approx(expected)


def test_appo_first_schedule_is_skipped_even_with_nonzero_kl():
    cfg = compose_config("appo", "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    learner = TeacherAPPOLearner(actor=actor, critic=critic, schedule="adaptive")
    learner._update_adaptive_learning_rate(1.0)
    assert learner.learning_rate == 0.001
    learner._update_adaptive_learning_rate(1.0)
    assert learner.learning_rate == 0.001 / 1.1
    learner.update_target_network()
    learner._update_adaptive_learning_rate(0.0)
    assert learner.learning_rate == 0.001 / 1.1


def test_ppo_scheduler_preserves_upstream_loss_weights_and_rng_at_fixed_lr():
    cfg = compose_config("ppo", "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    obs = TensorDict({"policy": torch.zeros(8, 156), "critic": torch.zeros(8, 174)}, [8])
    storage = TeacherRolloutStorage("rl", 8, 1, obs, [22], "cpu")
    last = ppo_rollout(actor, critic, storage)
    # One step exercises the adaptive hook, but must not change LR after reset.
    options = dict(num_learning_epochs=1, num_mini_batches=1)
    learner = TeacherPPO(actor, critic, storage, schedule="adaptive", **options)
    learner.compute_returns(last)
    reference = PPO(
        copy.deepcopy(actor),
        copy.deepcopy(critic),
        copy.deepcopy(storage),
        schedule="fixed",
        **options,
    )
    rng = torch.get_rng_state()
    expected = reference.update()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = learner.update()
    assert actual == expected
    assert torch.equal(torch.get_rng_state(), after)
    torch.testing.assert_close(actor.state_dict(), reference.actor.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(critic.state_dict(), reference.critic.state_dict(), atol=0, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ppo", "appo"])
def test_kl_schedule_in_real_training(tmp_path, monkeypatch, algo):
    cls = TeacherPPO if algo == "ppo" else TeacherAPPOLearner
    method = "_schedule_before_step" if algo == "ppo" else "_update_adaptive_learning_rate"
    original = getattr(cls, method)
    trace = []

    def record(learner, *args, **kwargs):
        skipped, before = learner._skip_kl_schedule, learner.learning_rate
        result = original(learner, *args, **kwargs)
        trace.append((skipped, before, learner.learning_rate))
        return result

    monkeypatch.setattr(cls, method, record)
    cfg = compose_config(
        algo,
        "mujoco",
        [
            "training.device=cpu",
            "training.num_envs=8",
            "training.torch_threads.learner_num_threads=2",
            "training.torch_threads.collector_num_threads=2",
            "training.no_play=true",
            "training.logger=no_print",
            f"training.log_dir={tmp_path / 'run'}",
            "algo.max_iterations=2",
            "algo.save_interval=1",
            "algo.algorithm.num_learning_epochs=2",
            "algo.algorithm.num_mini_batches=2",
            "algo.num_steps_per_env=2" if algo == "ppo" else "algo.steps_per_env=2",
        ],
    )
    path = train_teacher(cfg)
    actor, restored, snapshot = load_policy(path, configure_runtime=False)
    assert actor.kl_mode == restored.algo.algorithm.kl_mode == "exact"
    assert snapshot["counters"]["optimizer_updates"] == 8
    assert [row[0] for row in trace] == [True, False, False, False] * 2
    assert trace[0][1] == trace[0][2] == 0.001
    assert trace[4][1] == trace[4][2] == trace[3][2]
    assert snapshot["optimizer"]["param_groups"][0]["lr"] == trace[-1][2]
    torch.testing.assert_close(actor.state_dict(), snapshot["actor"], atol=0, rtol=0)
