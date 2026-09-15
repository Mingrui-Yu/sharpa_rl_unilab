"""Isolate the known original Normal.log_prob operation order in V-trace."""

import copy
import json
import types

import torch
from adapter import OUT, write
from audit import args_from, build, error, gradients, seed

from sharpa_rl_unilab.algos.hora.distribution import tanh_log_det


def main():
    torch.set_num_threads(4)
    state = torch.load(
        OUT / "audit/trained_real_rollout.pt", weights_only=False, map_location="cpu"
    )
    original = build("R", state, "cpu")
    current = build("A", state, "cpu")
    restored = build("A", state, "cpu")

    def original_density(self, raw):
        return torch.distributions.Normal(self.output_mean, self.output_std).log_prob(raw).sum(
            -1
        ) - tanh_log_det(raw)

    restored.target_actor.get_output_log_prob = types.MethodType(
        original_density, restored.target_actor
    )
    batches = []
    for learner in [original, current, restored]:
        batch = copy.deepcopy(state["batch"])
        seed()
        learner.process_batch(batch)
        batches.append(batch)
    report = {}
    for label, index in [("current", 1), ("only_target_density_restored", 2)]:
        report[label] = {
            k: error(batches[0][k], batches[index][k])
            for k in ["target_log_probs", "returns", "advantages"]
        }
    assert all(v["max_abs"] == 0 for v in report["only_target_density_restored"].values())
    steps = [
        gradients(learner, args_from(batch, "cpu")[0])
        for learner, batch in zip([original, current, restored], batches)
    ]

    def rename(values):
        return {
            k.replace("shared.distribution.std_param", "std_module.std_param"): v
            for k, v in values.items()
        }

    report["end_to_end_fixed_LR"] = {}
    for label, index in [("current", 1), ("only_target_density_restored", 2)]:
        a, b = steps[0], steps[index]
        ga, gb = rename(a[1]), rename(b[1])
        da, db = rename(a[3]), rename(b[3])
        report["end_to_end_fixed_LR"][label] = {
            "loss": error(a[0][0], b[0][0]),
            "max_gradient_absolute_error": max(error(ga[k], gb[k])["max_abs"] for k in ga),
            "max_adam_update_absolute_error": max(error(da[k], db[k])["max_abs"] for k in da),
            "gradient_norms": [float(a[2]), float(b[2])],
        }
    # Repeat the full-pipeline comparison with the historical Adam moments.
    history_path = json.loads((OUT / "manifest.json").read_text())["history"]["original"][
        "checkpoint"
    ]
    history = torch.load(history_path, weights_only=False, map_location="cpu")
    warmed = [build("R", state, "cpu"), build("A", state, "cpu")]
    orders = [
        [
            k.replace("shared.distribution.std_param", "std_module.std_param")
            for k, _ in learner.actor.named_parameters()
        ]
        + ["critic." + k for k, _ in learner.critic.named_parameters()]
        for learner in warmed
    ]
    assert orders[0] == orders[1]
    warmed_steps = []
    for learner, batch in zip(warmed, batches[:2]):
        learner.optimizer.load_state_dict(copy.deepcopy(history["optimizer"]))
        warmed_steps.append(gradients(learner, args_from(batch, "cpu")[0]))
    da, db = [rename(step[3]) for step in warmed_steps]
    report["historical_Adam_moments"] = {
        "lr": warmed[0].optimizer.param_groups[0]["lr"],
        "max_adam_update_absolute_error": max(error(da[k], db[k])["max_abs"] for k in da),
        "per_parameter_update_errors": {k: error(da[k], db[k]) for k in da},
    }
    torch.save(
        {"fresh": steps, "historical_moments": warmed_steps},
        OUT / "audit/end_to_end_density_steps.pt",
    )
    report["interpretation"] = (
        "The original Normal.log_prob operation order fully explains the fixed-input process_batch residual in this CPU replay. Default reference tolerance failures in the GPU audit remain reported, not erased."
    )
    write(OUT / "audit/density_residual.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
