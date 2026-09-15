"""Real rollout, original/current math, RNG, gradients, and Adam mechanism audit."""

import copy
import json

import torch
from adapter import OUT, branch, config, legacy_module, old_kl, write
from uni_rl.algos.appo.learner import APPOLearner

from sharpa_rl_unilab.algos.hora import teacher
from sharpa_rl_unilab.algos.hora.distribution import PolicyDistribution
from sharpa_rl_unilab.algos.hora.legacy import migrate_actor_state
from sharpa_rl_unilab.training.teacher_runtime import config_dict


def error(a, b):
    a, b = a.detach().double(), b.detach().double()
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "relative_l2": (d.norm() / (a.norm() + 1e-30)).item(),
        "allclose": torch.allclose(a, b, atol=1e-6, rtol=1e-5),
    }


def cpu(x):
    return {k: v.detach().cpu() for k, v in x.items()}


def seed():
    torch.manual_seed(314159)


def build(group, state, device):
    old = legacy_module()
    model = config_dict(config(group, OUT / "audit").model)
    actor = (old.TeacherActor if group == "R" else teacher.TeacherActor)(model).to(device)
    weights = state["actor"] if group == "R" else migrate_actor_state(state["actor"])
    if group in "CDN":
        weights = dict(weights)
        weights["std_module.log_std"] = weights.pop("std_module.std_param").log()
    actor.load_state_dict(weights)
    critic = teacher.CleanValue(model).to(device)
    critic.load_state_dict(state["critic"])
    base = old.TeacherAPPOLearner if group == "R" else teacher.TeacherAPPOLearner

    class OldKL(base):
        def _minibatch_loss_tensors(self, *args):
            r = super()._minibatch_loss_tensors(*args)
            if group in "AC":
                p = self._loss_distribution
                return (*r[:4], old_kl(p.mean, p.std, args[-2], args[-1]).mean(), *r[5:])
            return r

    learner = OldKL(
        actor, critic, device=device, **config_dict(config(group, OUT / "audit").algo.algorithm)
    )
    # target equals current for these controlled initial / trained-state comparisons.
    return learner


def args_from(b, device):
    ids = torch.arange(0, b["actions"].shape[0] * b["actions"].shape[1], 4, device=device)[:4096]
    adv = b["advantages"].flatten()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    fields = ["observations", "critic", "actions", "returns"]
    a = [b[k].flatten(0, 1)[ids] for k in fields]
    a += [adv[ids]] + [
        b[k].flatten()[ids] for k in ["actions_log_prob", "values", "target_log_probs"]
    ]
    a += [b[k][ids] for k in ["_old_mu", "_old_sigma"]]
    return a, ids


def gradients(learner, args):
    seed()
    learner.optimizer.zero_grad()
    result = learner._minibatch_loss_tensors(*args)
    result[0].backward()
    grads = {k: p.grad.clone() for k, p in learner.actor.named_parameters() if p.grad is not None}
    grads.update(
        {
            "critic." + k: p.grad.clone()
            for k, p in learner.critic.named_parameters()
            if p.grad is not None
        }
    )
    before = {k: p.detach().clone() for k, p in learner.actor.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(learner._combined_params, 1.0)
    # Identical frozen LR isolates density/gradient from KL scheduling.
    learner.optimizer.step()
    delta = {k: p.detach() - before[k] for k, p in learner.actor.named_parameters()}
    return result, grads, norm, delta


def main():
    torch.set_num_threads(4)
    device = "cuda:0"
    smoke = OUT / "smoke_A"
    data = torch.load(smoke / "real_batch.pt", weights_only=False, map_location=device)
    initial = torch.load(smoke / "batch_state.pt", weights_only=False, map_location=device)
    initial["actor"]["shared.distribution.std_param"] = initial["actor"].pop("std_module.std_param")
    historical = torch.load(
        json.loads((OUT / "manifest.json").read_text())["history"]["original"]["checkpoint"],
        weights_only=False,
        map_location=device,
    )
    # Collect an actual trained-policy rollout; avoid testing only very off-policy initial data.
    from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
    from sharpa_rl_unilab.training.configuration import configure_threads
    from sharpa_rl_unilab.training.teacher_runtime import collect, stage_rollout, timeout_rewards

    cfg = config("A", OUT / "audit")
    configure_threads(cfg)
    trained_learner = build("A", historical, device)
    env = SharpaTeacherEnv(cfg, 2048)
    try:
        obs, _ = env.reset(seed=2)
        seed()
        raw, last = collect(env, obs, trained_learner.actor, 8, device)
        stages, _ = stage_rollout(
            None, raw, last, 501, trained_learner.actor, trained_learner.critic, device
        )
        trained_data = stages.batch()
        trained_data["rewards"] = timeout_rewards(trained_data, trained_learner.critic, 0.99)
        historical = dict(
            historical,
            actor=trained_learner.actor.state_dict(),
            critic=trained_learner.critic.state_dict(),
        )
        historical["actor"]["shared.distribution.std_param"] = historical["actor"].pop(
            "std_module.std_param"
        )
        torch.save(
            {
                "batch": trained_data,
                "actor": historical["actor"],
                "critic": historical["critic"],
                "rng_seed": 314159,
            },
            OUT / "audit/trained_real_rollout.pt",
        )
    finally:
        env.close()
    report = {
        "numeric": {
            "dtype": "float32",
            "autocast": torch.is_autocast_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "states": {},
    }
    for name, state in [("initial", initial), ("trained", historical)]:
        learners = {g: build(g, state, device) for g in "RABCDN"}
        batches = {}
        rng = {}
        for g, learner in learners.items():
            b = {
                k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k, v in (data if name == "initial" else trained_data).items()
            }
            seed()
            learner.process_batch(b)
            rng[g] = torch.cuda.get_rng_state()
            batches[g] = b
        args, indices = args_from(batches["R"], device)
        torch.save(
            {"args": [t.cpu() for t in args], "indices": indices.cpu(), "rng_seed": 314159},
            OUT / "audit" / f"{name}_minibatch.pt",
        )
        r = {}
        r["process_R_A"] = {
            k: error(batches["R"][k], batches["A"][k])
            for k in [
                "values",
                "returns",
                "advantages",
                "target_log_probs",
                "_old_mu",
                "_old_sigma",
            ]
        }
        r["process_rng_equal"] = torch.equal(rng["R"], rng["A"])
        evaluated = {g: gradients(learner, args) for g, learner in learners.items()}

        def rename(grads):
            return {
                k.replace("shared.distribution.std_param", "std_module.std_param"): v
                for k, v in grads.items()
            }

        gr, ga = map(rename, [evaluated["R"][1], evaluated["A"][1]])
        r["loss_R_A"] = {
            n: error(a, b)
            for n, a, b in zip(
                ["loss", "policy", "value", "entropy", "kl", "log_prob", "ratio"],
                evaluated["R"][0],
                evaluated["A"][0],
            )
        }
        r["grad_R_A"] = {k: error(gr[k], ga[k]) for k in gr}
        dr, da = map(rename, [evaluated["R"][3], evaluated["A"][3]])
        r["adam_R_A"] = {k: error(dr[k], da[k]) for k in dr}
        r["norms"] = {g: float(evaluated[g][2]) for g in learners}
        r["clip_coefficients"] = {
            g: min(1.0, 1 / (float(evaluated[g][2]) + 1e-6)) for g in learners
        }
        gs, gl = evaluated["A"][1], evaluated["C"][1]
        std = state["actor"]["shared.distribution.std_param"]
        r["chain_rule"] = error(std * gs["std_module.std_param"], gl["std_module.log_std"])
        r["actual_std_step"] = {
            g: (
                evaluated[g][3]["std_module.std_param"].cpu().tolist()
                if g in "AB"
                else ((std.log() + evaluated[g][3]["std_module.log_std"]).exp() - std)
                .cpu()
                .tolist()
            )
            for g in "ABCDN"
        }
        r["trunk_adam_A_C"] = {
            k: error(evaluated["A"][3][k], evaluated["C"][3][k])
            for k in evaluated["A"][3]
            if k.startswith("shared.")
        }
        r["fixed_LR_A_B_max_update_diff"] = max(
            error(evaluated["A"][3][k], evaluated["B"][3][k])["max_abs"] for k in evaluated["A"][3]
        )
        r["fixed_LR_C_D_max_update_diff"] = max(
            error(evaluated["C"][3][k], evaluated["D"][3][k])["max_abs"] for k in evaluated["C"][3]
        )
        r["D_N_max_update_diff"] = max(
            error(evaluated["D"][3][k], evaluated["N"][3][k])["max_abs"] for k in evaluated["D"][3]
        )
        torch.save(
            {
                g: {
                    "grad": cpu(v[1]),
                    "delta": cpu(v[3]),
                    "optimizer": learners[g].optimizer.state_dict(),
                }
                for g, v in evaluated.items()
            },
            OUT / "audit" / f"{name}_gradient_adam.pt",
        )
        report["states"][name] = r
    report["sampling"] = []
    old = legacy_module()
    for device in ["cpu", "cuda:0"]:
        for scale, mean_scale in [(1.0, 1.0), (0.03, 1.0), (1.0, 20.0), (1e-6, 1.0), (1e6, 1.0)]:
            seed()
            mean = torch.randn(4096, 22, device=device) * mean_scale
            std = torch.full_like(mean, scale)
            eps = torch.randn_like(mean)
            raw = mean + std * eps
            d = PolicyDistribution(mean, std, "tanh")
            seed()
            normal = torch.distributions.Normal(mean, std).sample()
            rng1 = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
            seed()
            public = d.sample().raw
            rng2 = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
            seed()
            eold = old.tanh_entropy(mean, std)
            rng3 = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
            seed()
            enew = d.entropy()
            rng4 = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
            report["sampling"].append(
                {
                    "device": device,
                    "std": scale,
                    "mean_scale": mean_scale,
                    "shared_epsilon": error(raw, d.sample(eps).raw),
                    "native_sample": error(normal, public),
                    "native_rng_equal": torch.equal(rng1, rng2),
                    "entropy": error(eold, enew),
                    "entropy_rng_equal": torch.equal(rng3, rng4),
                    "density_old_learner": error(
                        APPOLearner._gaussian_log_prob(raw, mean, std) - old.tanh_log_det(raw),
                        d.log_prob(raw),
                    ),
                    "density_old_collector": error(
                        torch.distributions.Normal(mean, std).log_prob(raw).sum(-1)
                        - old.tanh_log_det(raw),
                        d.log_prob(raw),
                    ),
                }
            )
    mean = torch.zeros(1024, 22, device="cuda:0")
    std = torch.ones_like(mean)
    ko = old_kl(mean, std, mean, std).mean().item()
    ke = PolicyDistribution(mean, std, "tanh").kl_from(mean, std).mean().item()
    report["equal_policy_KL"] = {
        "old": ko,
        "exact": ke,
        "old_branch": branch(ko),
        "exact_branch": branch(ke),
    }
    vals = torch.tensor([0.5e-6, 1e-6, 2e-6, 0.5, 1, 2, 5e5, 1e6, 2e6], device="cuda:0")
    s = vals.clone().requires_grad_()
    log_parameter = vals.log().requires_grad_()
    ys = s.clamp(1e-6, 1e6)
    yl = log_parameter.clamp(-13.815510557964274, 13.815510557964274).exp()
    ys.sum().backward()
    yl.sum().backward()
    report["boundaries"] = {
        "input": vals.tolist(),
        "direct": ys.tolist(),
        "log": yl.tolist(),
        "direct_grad": s.grad.tolist(),
        "log_grad": log_parameter.grad.tolist(),
    }
    write(OUT / "audit/fixed_data_report.json", report)
    print(
        json.dumps(
            {
                "equal_policy_KL": report["equal_policy_KL"],
                "states": {
                    k: {
                        a: v
                        for a, v in r.items()
                        if a
                        in [
                            "process_rng_equal",
                            "chain_rule",
                            "norms",
                            "fixed_LR_A_B_max_update_diff",
                            "D_N_max_update_diff",
                        ]
                    }
                    for k, r in report["states"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
