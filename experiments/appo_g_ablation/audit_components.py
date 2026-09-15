"""Fixed-data one-factor probes of the bridge's remaining FP32 gradients (CPU)."""

import argparse
import json
import types

import torch
from adapter import OUT, write
from audit import args_from, build, error, gradients, seed
from uni_rl.algos.appo.learner import APPOLearner

from sharpa_rl_unilab.algos.hora.distribution import PolicyDistribution, tanh_log_det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    device = parser.parse_args().device
    torch.set_num_threads(4)
    report = {"device": device}
    for name in ["initial", "trained"]:
        if name == "initial":
            state = torch.load(
                OUT / "smoke_A/batch_state.pt", weights_only=False, map_location=device
            )
            state["actor"]["shared.distribution.std_param"] = state["actor"].pop(
                "std_module.std_param"
            )
            batch = torch.load(
                OUT / "smoke_A/real_batch.pt", weights_only=False, map_location=device
            )
        else:
            state = torch.load(
                OUT / "audit/trained_real_rollout.pt", weights_only=False, map_location=device
            )
            batch = state["batch"]
        reference = build("R", state, device)
        seed()
        reference.process_batch(batch)
        args, _ = args_from(batch, device)
        candidates = {
            "R": reference,
            "A": build("A", state, device),
            "A_dense_std_clamp": build("A", state, device),
            "A_old_learner_density": build("A", state, device),
        }
        dense = candidates["A_dense_std_clamp"]
        original = dense._minibatch_policy_value

        def dense_std(self, *args):
            mean, _, value = original(*args)
            std = self.actor.std_module.std_param.expand_as(mean).clamp(1e-6, 1e6)
            self._loss_distribution = PolicyDistribution(mean, std, "tanh")
            return mean, std, value

        dense._minibatch_policy_value = types.MethodType(dense_std, dense)

        def old_density(self, actions, mean, std):
            return APPOLearner._gaussian_log_prob(actions, mean, std) - tanh_log_det(actions)

        candidates["A_old_learner_density"]._gaussian_log_prob = types.MethodType(
            old_density, candidates["A_old_learner_density"]
        )
        steps = {k: gradients(v, args) for k, v in candidates.items()}

        def rename(x):
            return {
                k.replace("shared.distribution.std_param", "std_module.std_param"): v
                for k, v in x.items()
            }

        reference_grad = rename(steps["R"][1])
        reference_delta = rename(steps["R"][3])
        report[name] = {}
        for label, step in steps.items():
            grad = rename(step[1])
            delta = rename(step[3])
            report[name][label] = {
                "max_grad_abs": max(error(reference_grad[k], grad[k])["max_abs"] for k in grad),
                "std_grad": error(
                    reference_grad["std_module.std_param"], grad["std_module.std_param"]
                ),
                "max_adam_delta_abs": max(
                    error(reference_delta[k], delta[k])["max_abs"] for k in delta
                ),
            }
    write(OUT / f"audit/numeric_components_{device.split(':')[0]}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
