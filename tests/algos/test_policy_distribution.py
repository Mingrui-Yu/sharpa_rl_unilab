import copy
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from uni_rl.algos.appo.learner import APPOLearner
from uni_rl.algos.flash_sac.network import FlashSACActor

from sharpa_rl_unilab.algos.hora.distribution import PolicyDistribution
from sharpa_rl_unilab.algos.hora.teacher import TeacherActor, TeacherAPPOLearner, TeacherFlashActor
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import CONTRACT_VERSION
from sharpa_rl_unilab.training.action_diagnostics import EpisodeActionDiagnostics
from sharpa_rl_unilab.training.exploration import HeldGaussianNoise
from sharpa_rl_unilab.training.teacher_runtime import algorithm_options, load_policy, make_models


@pytest.mark.parametrize("mapping", ["clip", "tanh"])
def test_sampling_density_and_pathwise_gradient(mapping):
    mean = torch.randn(16, 22, requires_grad=True)
    log_std = torch.full((22,), -0.4, requires_grad=True)
    dist = PolicyDistribution(mean, log_std.exp(), mapping)
    state = torch.random.get_rng_state()
    sample = dist.sample()
    torch.random.set_rng_state(state)
    noise = torch.randn_like(mean)
    external = dist.sample(noise)
    torch.testing.assert_close(sample.raw, external.raw, rtol=0, atol=0)
    reference = torch.distributions.Normal(mean, log_std.exp()).log_prob(sample.raw).sum(-1)
    if mapping == "tanh":
        reference -= torch.log1p(-sample.action.square()).sum(-1)
    torch.testing.assert_close(sample.log_prob, reference, atol=2e-4, rtol=2e-5)
    grads = torch.autograd.grad(
        sample.action.square().sum() + sample.log_prob.sum(), (mean, log_std), retain_graph=True
    )
    other = torch.autograd.grad(
        external.action.square().sum() + external.log_prob.sum(), (mean, log_std)
    )
    for a, b in zip(grads, other):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all() and a.abs().sum() > 0
    with torch.no_grad():
        assert not dist.sample().action.requires_grad


def test_tanh_entropy_gradient_against_quadrature_finite_difference():
    nodes, weights = np.polynomial.hermite.hermgauss(100)
    noise = torch.tensor(nodes * np.sqrt(2), dtype=torch.float64).reshape(-1, 1)
    weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float64)
    mean = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    dist = PolicyDistribution(mean.expand(100, 1), log_std.exp(), "tanh")
    entropy = (dist.entropy(noise) * weights).sum()
    gradients = torch.autograd.grad(entropy, (mean, log_std))

    def reference(mu, ls):
        z = mu + np.exp(ls) * nodes * np.sqrt(2)
        jacobian = 2 * (np.log(2) - np.logaddexp(z, -z))
        return ls + 0.5 * np.log(2 * np.pi * np.e) + np.dot(weights.numpy(), jacobian)

    h = 1e-5
    expected = [
        (reference(0.7 + h, -0.3) - reference(0.7 - h, -0.3)) / (2 * h),
        (reference(0.7, -0.3 + h) - reference(0.7, -0.3 - h)) / (2 * h),
    ]
    np.testing.assert_allclose([g.item() for g in gradients], expected, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distribution_boundary_math_uses_fp32(dtype):
    mean = torch.tensor([[20.0, -20.0]], dtype=dtype, requires_grad=True)
    std = torch.full_like(mean, math.exp(-10), requires_grad=True)
    dist = PolicyDistribution(mean, std, "tanh")
    sample = dist.sample(torch.ones_like(mean))
    assert sample.raw.dtype == torch.float32
    loss = sample.log_prob.sum() + dist.entropy().sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


@pytest.mark.parametrize("mode", ["state_independent", "state_dependent"])
def test_common_actor_paths_and_std_updates(mode):
    actors = []
    obs = torch.randn(8, 156)
    for algo in ("ppo", "appo", "flashsac"):
        cfg = compose_config(
            algo,
            "mujoco",
            [f"model.std_mode={mode}", "model.action_mapping=tanh", "model.initial_std=0.7"],
        )
        actor, _, _ = make_models(cfg, "cpu")
        actors.append(actor)
    for actor in actors[1:]:
        actor.load_state_dict(actors[0].state_dict())
    noise = torch.randn(8, 22)
    samples = [actor.policy(obs).sample(noise) for actor in actors]
    for sample in samples:
        torch.testing.assert_close(sample.std, torch.full((8, 22), 0.7))
        torch.testing.assert_close(sample.raw, samples[0].raw, rtol=0, atol=0)
        torch.testing.assert_close(sample.log_prob, samples[0].log_prob, rtol=0, atol=0)
    for actor in actors:
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
        dist = actor.policy(obs)
        # Nonuniform observations/targets exercise the initially zero output head.
        loss = -dist.log_prob(torch.arange(8).reshape(8, 1).expand(8, 22) / 5).mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(
            p.grad is not None and p.grad.abs().sum() > 0 for p in actor.std_module.parameters()
        )
        optimizer.step()
        std = actor.policy(obs).std
        assert not torch.allclose(std, samples[0].std)
        assert torch.equal(std[0], std[-1]) == (mode == "state_independent")


@pytest.mark.parametrize(
    "override",
    [
        "model.initial_std=0",
        "model.initial_std=-1",
        "model.initial_std=.nan",
        "model.std_mode=wrong",
        "model.log_std_bounds=[1,2]",
        "model.log_std_bounds=[0,0]",
    ],
)
def test_invalid_std_config(override):
    cfg = compose_config("ppo", "mujoco", [override])
    with pytest.raises(ValueError):
        make_models(cfg, "cpu")


def test_adapter_cache_isolation_and_raw_sample_score_gradient():
    cfg = compose_config("ppo", "mujoco", ["model.action_mapping=tanh"])
    actor, _, _ = make_models(cfg, "cpu")
    other = copy.deepcopy(actor)
    obs = TensorDict({"policy": torch.randn(8, 156)}, batch_size=8)
    with torch.no_grad():
        sample = actor.policy(obs["policy"]).sample()
    actor(obs)
    saved = actor.distribution
    other(TensorDict({"policy": obs["policy"] * 2}, batch_size=8))
    assert actor.distribution is saved
    torch.testing.assert_close(actor.get_output_log_prob(sample.raw), sample.log_prob)
    actor.get_output_log_prob(sample.raw).sum().backward()
    assert sample.raw.grad is None
    assert actor.shared.mu_head.weight.grad.abs().sum() > 0
    actor(TensorDict({"policy": obs["policy"] * 3}, batch_size=8))
    assert actor.distribution is not saved
    torch.testing.assert_close(saved.log_prob(sample.raw), sample.log_prob)


@pytest.mark.parametrize("mode", ["state_independent", "state_dependent"])
def test_optional_bounds_match_appo_current_target_and_sampling(mode):
    cfg = compose_config(
        "appo", "mujoco", [f"model.std_mode={mode}", "model.log_std_bounds=[-5,2]"]
    )
    actor, critic, _ = make_models(cfg, "cpu")
    learner = TeacherAPPOLearner(actor=actor, critic=critic, device="cpu")
    obs, clean = torch.randn(8, 156), torch.randn(8, 174)
    for raw, expected in [(-30.0, math.exp(-5)), (30.0, math.exp(2))]:
        with torch.no_grad():
            parameter = (
                actor.std_module.log_std
                if mode == "state_independent"
                else actor.std_module.head.bias
            )
            parameter.fill_(raw)
        learner.update_target_network()
        mean, std, _ = learner._minibatch_policy_value(obs, clean)
        torch.testing.assert_close(std, torch.full((8, 22), expected))
        dist = actor.policy(obs)
        target = learner.target_actor.policy(obs)
        torch.testing.assert_close(dist.std, target.std, rtol=0, atol=0)
        sample = dist.sample()
        torch.testing.assert_close(
            learner._gaussian_log_prob(sample.raw, mean, std), sample.log_prob
        )


@pytest.mark.parametrize("mapping", ["clip", "tanh"])
@pytest.mark.parametrize("mode", ["state_independent", "state_dependent"])
@pytest.mark.parametrize("kl_mode", ["log_epsilon", "exact"])
def test_appo_loss_uses_current_entropy_and_target_kl(mapping, mode, kl_mode):
    cfg = compose_config(
        "appo",
        "mujoco",
        [
            f"model.action_mapping={mapping}",
            f"model.std_mode={mode}",
            f"algo.algorithm.kl_mode={kl_mode}",
        ],
    )
    actor, critic, _ = make_models(cfg, "cpu")
    learner = TeacherAPPOLearner(
        actor=actor,
        critic=critic,
        device="cpu",
        **algorithm_options(cfg, APPOLearner, extra_options=("kl_mode",)),
    )
    obs, clean = torch.randn(8, 156), torch.randn(8, 174)
    with torch.no_grad():
        sample = actor.policy(obs).sample()
    args = (
        obs,
        clean,
        sample.raw,
        torch.zeros(8),
        torch.ones(8),
        sample.log_prob,
        torch.zeros(8),
        sample.log_prob,
        sample.mean,
        sample.std,
    )
    result = learner._minibatch_loss_tensors(*args)
    torch.testing.assert_close(result[-1], torch.ones(8))
    expected_kl = 0.0 if kl_mode == "exact" else 0.00022029876708984375
    torch.testing.assert_close(result[4], torch.tensor(expected_kl), atol=1e-7, rtol=0)
    before = learner.learning_rate
    learner._update_adaptive_learning_rate(result[4].item())
    assert learner.learning_rate == (before if kl_mode == "exact" else before * 1.1)
    # Only entropy can contribute a mean gradient here if advantages are zero.
    zero_adv = list(args)
    zero_adv[4] = torch.zeros(8)
    learner.optimizer.zero_grad()
    entropy_result = learner._minibatch_loss_tensors(*zero_adv)
    entropy_result[0].backward()
    if mapping == "tanh":
        assert actor.shared.mu_head.bias.grad.abs().sum() > 0
    with torch.no_grad():
        actor.shared.mu_head.bias.add_(0.2)
    result = learner._minibatch_loss_tensors(*args)
    dist = actor.policy(obs)
    reference = (
        torch.distributions.kl_divergence(
            torch.distributions.Normal(sample.mean, sample.std),
            torch.distributions.Normal(dist.mean, dist.std),
        )
        .sum(-1)
        .mean()
    )
    if kl_mode == "log_epsilon":
        reference = (
            (
                torch.log(dist.std / sample.std + 1e-5)
                + (sample.std.square() + (sample.mean - dist.mean).square())
                / (2 * dist.std.square())
                - 0.5
            )
            .sum(-1)
            .mean()
        )
    torch.testing.assert_close(result[4], reference)
    before = learner.learning_rate
    learner._update_adaptive_learning_rate(result[4].item())
    assert learner.learning_rate < before


@pytest.mark.parametrize("algo", ["ppo", "appo"])
def test_on_policy_kl_default_and_invalid_mode(algo):
    cfg = compose_config(algo, "mujoco", [])
    assert cfg.algo.algorithm.kl_mode == "log_epsilon"
    actor, critic, _ = make_models(cfg, "cpu")
    assert actor.kl_mode == "log_epsilon"
    assert TeacherAPPOLearner(actor=actor, critic=critic, device="cpu").kl_mode == "log_epsilon"
    with pytest.raises(ValueError, match="kl_mode must be exact or log_epsilon"):
        TeacherAPPOLearner(actor=actor, critic=critic, device="cpu", kl_mode="typo")
    bad = compose_config("appo", "mujoco", ["+algo.algorithm.kl_mod=exact"])
    with pytest.raises(ValueError, match="Unsupported algorithm options"):
        algorithm_options(bad, APPOLearner, extra_options=("kl_mode",))
    for invalid in ["typo", "legacy"]:
        bad = compose_config(algo, "mujoco", [f"algo.algorithm.kl_mode={invalid}"])
        with pytest.raises(ValueError, match="kl_mode must be exact or log_epsilon"):
            make_models(bad, "cpu")


@pytest.mark.parametrize("kl_mode", ["exact", "log_epsilon"])
@pytest.mark.parametrize("mapping", ["clip", "tanh"])
def test_ppo_and_appo_kl_agree_for_changed_mean_and_std(kl_mode, mapping):
    cfg = compose_config(
        "ppo", "mujoco", [f"algo.algorithm.kl_mode={kl_mode}", f"model.action_mapping={mapping}"]
    )
    actor, critic, _ = make_models(cfg, "cpu")
    learner = TeacherAPPOLearner(actor=actor, critic=critic, device="cpu", kl_mode=kl_mode)
    obs, clean = torch.randn(8, 156), torch.randn(8, 174)
    with torch.no_grad():
        sample = actor.policy(obs).sample()
        old_mean = sample.mean + torch.randn_like(sample.mean) * 0.1
        old_std = sample.std * torch.linspace(0.2, 2.0, 22)
    args = (
        obs,
        clean,
        sample.raw,
        torch.zeros(8),
        torch.ones(8),
        sample.log_prob,
        torch.zeros(8),
        sample.log_prob,
        old_mean,
        old_std,
    )
    appo_kl = learner._minibatch_loss_tensors(*args)[4]
    ppo_kl = actor.get_kl_divergence((old_mean, old_std), (sample.mean, sample.std))
    torch.testing.assert_close(ppo_kl.mean(), appo_kl, atol=0, rtol=0)
    exact = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_std),
        torch.distributions.Normal(sample.mean, sample.std),
    ).sum(-1)
    expected = (
        exact if kl_mode == "exact" else exact + torch.log1p(1e-5 * old_std / sample.std).sum(-1)
    )
    torch.testing.assert_close(ppo_kl, expected, atol=2e-6, rtol=1e-5)
    same = actor.get_kl_divergence((sample.mean, sample.std), (sample.mean, sample.std))
    torch.testing.assert_close(
        same,
        torch.full((8,), 0.0 if kl_mode == "exact" else 0.00022029876708984375),
        atol=0,
        rtol=0,
    )


def test_appo_kl_selection_preserves_loss_gradient_and_rng():
    cfg = compose_config("appo", "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    learner = TeacherAPPOLearner(actor=actor, critic=critic, device="cpu")
    obs, clean = torch.randn(8, 156), torch.randn(8, 174)
    with torch.no_grad():
        sample = actor.policy(obs).sample()
    args = (
        obs,
        clean,
        sample.raw,
        torch.randn(8),
        torch.randn(8),
        sample.log_prob,
        torch.zeros(8),
        sample.log_prob,
        sample.mean,
        sample.std,
    )
    rng = torch.get_rng_state()
    records = []
    for mode in ["log_epsilon", "exact"]:
        learner.kl_mode = mode
        learner.optimizer.zero_grad()
        torch.set_rng_state(rng)
        result = learner._minibatch_loss_tensors(*args)
        result[0].backward()
        records.append(
            (
                result,
                [p.grad.clone() for p in learner._combined_params if p.grad is not None],
                torch.get_rng_state(),
            )
        )
    for i, (a, b) in enumerate(zip(records[0][0], records[1][0])):
        if i != 4:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(records[0][1], records[1][1]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(records[0][2], records[1][2])


@pytest.mark.parametrize("kl_mode", ["exact", "log_epsilon"])
def test_ppo_update_uses_configured_kl_for_learning_rate(kl_mode):
    cfg = compose_config(
        "ppo",
        "mujoco",
        [
            f"algo.algorithm.kl_mode={kl_mode}",
            "algo.algorithm.num_learning_epochs=1",
            "algo.algorithm.num_mini_batches=1",
        ],
    )
    actor, critic, _ = make_models(cfg, "cpu")
    obs = TensorDict({"policy": torch.randn(8, 156), "critic": torch.randn(8, 174)}, batch_size=8)
    storage = RolloutStorage("rl", 8, 1, obs, [22], "cpu")
    transition = RolloutStorage.Transition()
    with torch.no_grad():
        transition.observations = obs
        transition.actions = actor(obs, stochastic_output=True)
        transition.actions_log_prob = actor.get_output_log_prob(transition.actions)
        transition.distribution_params = actor.output_distribution_params
        transition.values = critic(obs)
        transition.rewards = torch.ones(8)
        transition.dones = torch.zeros(8)
        storage.add_transition(transition)
    options = algorithm_options(cfg, PPO, extra_options=("kl_mode",))
    options.pop("kl_mode")  # The actor consumes this option, as in train_teacher.
    learner = PPO(actor, critic, storage, device="cpu", **options)
    learner.compute_returns(obs)
    before = learner.learning_rate
    learner.update()
    assert learner.learning_rate == (before if kl_mode == "exact" else before * 1.5)


@pytest.mark.parametrize("max_steps", [1, 16])
def test_held_noise_matches_upstream_per_environment(max_steps):
    class Reference(FlashSACActor):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.mean, self.std = torch.zeros(8, 22), torch.ones(8, 22)
            pmf = torch.arange(1, max_steps + 1).float().pow(-2)
            self.register_buffer("zeta_cdf", (pmf / pmf.sum()).cumsum(0))
            self.register_buffer("_noise", torch.zeros(0), persistent=False)
            self.register_buffer(
                "_repeat_count", torch.zeros(0, dtype=torch.int32), persistent=False
            )
            self.register_buffer(
                "_repeat_target", torch.zeros(0, dtype=torch.int32), persistent=False
            )

        def get_mean_and_std(self, obs, training=False):
            return self.mean, self.std

    reference = Reference()
    noise = HeldGaussianNoise(max_steps=max_steps)
    for step in range(40):
        reference.mean.fill_(step / 20)
        reference.std.fill_(0.3 + step / 40)
        dones = torch.arange(8) == step % 11
        rng = torch.random.get_rng_state()
        expected = reference.explore(None, dones)
        torch.random.set_rng_state(rng)
        epsilon = noise.sample(reference.mean, dones)
        actual = PolicyDistribution(reference.mean, reference.std, "tanh").sample(
            epsilon, log_prob=False
        )
        torch.testing.assert_close(actual.action, expected, rtol=0, atol=0)
        for a, b in [
            (noise.noise, reference._noise),
            (noise.count, reference._repeat_count),
            (noise.target, reference._repeat_target),
        ]:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
@pytest.mark.parametrize("student", [False, True])
def test_legacy_checkpoint_restores_distribution(tmp_path, algo, student):
    cfg = compose_config(algo, "mujoco", [])
    OmegaConf.set_struct(cfg, False)
    # Build an old-format fixture independently of the state migration function.
    flash = algo == "flashsac"
    cfg.model.action_mapping = "tanh" if flash else "clip"
    cfg.model.std_parameterization = "legacy_tanh" if flash else "legacy_scalar"
    cfg.model.std_mode = "state_dependent" if flash else "state_independent"
    cls = TeacherFlashActor if flash else TeacherActor
    original = cls(OmegaConf.to_container(cfg.model), student=student)
    with torch.no_grad():
        if not flash:
            original.std_module.std_param[:3] = torch.tensor([-2.0, 1e7, 0.4])
    state = dict(original.state_dict())
    if flash:
        state["std_head.weight"] = state.pop("std_module.head.weight")
        state["std_head.bias"] = state.pop("std_module.head.bias")
        state["shared.distribution.std_param"] = torch.ones(22)
        state["zeta_cdf"] = torch.ones(16)
    else:
        state["shared.distribution.std_param"] = state.pop("std_module.std_param")
    for key in ("std_parameterization", "std_mode", "initial_std", "log_std_bounds"):
        del cfg.model[key]
    del cfg.model.action_mapping
    path = tmp_path / "old.pt"
    torch.save(
        dict(
            contract=CONTRACT_VERSION,
            stage="student" if student else "teacher",
            algorithm=algo,
            config=OmegaConf.to_container(cfg, resolve=True),
            actor=state,
        ),
        path,
    )
    loaded, migrated, _ = load_policy(path, configure_runtime=False)
    inputs = {"actor": torch.randn(8, 147), "priv_info": torch.randn(8, 9)}
    if student:
        inputs["proprio_hist"] = torch.randn(8, 30, 49)
    td = TensorDict(inputs, batch_size=8)
    before, after = original.policy(td), loaded.policy(td)
    torch.testing.assert_close(before.mean, after.mean, rtol=0, atol=0)
    torch.testing.assert_close(before.std, after.std, rtol=0, atol=0)
    torch.testing.assert_close(before.deterministic(), after.deterministic(), rtol=0, atol=0)
    if not flash:
        torch.testing.assert_close(after.std[0, :3], torch.tensor([1e-6, 1e6, 0.4]), rtol=0, atol=0)
    # A resaved legacy model retains its explicit parameterization and new keys.
    torch.save(
        dict(
            contract=CONTRACT_VERSION,
            stage="student" if student else "teacher",
            algorithm=algo,
            config=OmegaConf.to_container(migrated, resolve=True),
            actor=loaded.state_dict(),
        ),
        path,
    )
    restored, _, _ = load_policy(path, configure_runtime=False)
    torch.testing.assert_close(restored.policy(td).std, before.std, rtol=0, atol=0)


def test_diagnostics_include_terminal_difference_but_not_reset_jump():
    def record(diag, a, q):
        diag.record(
            np.full((1, 22), a),
            np.ones((1, 22)),
            np.full((1, 22), q),
            np.ones((1, 22)),
            np.zeros(22),
            np.ones(22),
        )

    diag = EpisodeActionDiagnostics(np.zeros((1, 22)))
    record(diag, 0.0, 1.0)
    record(diag, 1.0, 4.0)  # Terminal step: second difference is 2.
    result = diag.result()
    assert result["q_second_difference_rms"] == 2
    assert result["action_delta_rms"] == 1
    assert result["saturation"] == 0.5
    reset = EpisodeActionDiagnostics(np.full((1, 22), 100))
    record(reset, -1.0, 101.0)
    assert reset.result()["q_second_difference_rms"] == 0
