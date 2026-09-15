"""Check that both KL-only interventions preserve the fixed-LR loss and update."""

import types

import torch
from adapter import OUT, old_kl, write
from audit import build, gradients
from train_kl_bias import BIAS, perturb_loss


def main():
    torch.set_num_threads(4)
    same_policy = old_kl(torch.zeros(22), torch.ones(22), torch.zeros(22), torch.ones(22)).item()
    assert same_policy == BIAS
    report = {"same_policy_old_KL_fp32": same_policy, "states": {}}
    for state_name in ["initial", "trained"]:
        if state_name == "initial":
            state = torch.load(
                OUT / "smoke_A/batch_state.pt", weights_only=False, map_location="cpu"
            )
            state["actor"]["shared.distribution.std_param"] = state["actor"].pop(
                "std_module.std_param"
            )
        else:
            state = torch.load(
                OUT / "audit/trained_real_rollout.pt", weights_only=False, map_location="cpu"
            )
        args = torch.load(OUT / f"audit/{state_name}_minibatch.pt", weights_only=False)["args"]
        baseline = build("N", state, "cpu")
        reference = gradients(baseline, args)
        report["states"][state_name] = {}
        for mode in ["constant", "original_log"]:
            learner = build("N", state, "cpu")
            learner._minibatch_loss_tensors = types.MethodType(
                perturb_loss(type(learner)._minibatch_loss_tensors, mode), learner
            )
            result = gradients(learner, args)
            equal = {
                "non_KL_outputs": all(
                    torch.equal(a, b)
                    for i, (a, b) in enumerate(zip(reference[0], result[0]))
                    if i != 4
                ),
                "gradients": all(torch.equal(reference[1][k], result[1][k]) for k in reference[1]),
                "fixed_LR_Adam_update": all(
                    torch.equal(reference[3][k], result[3][k]) for k in reference[3]
                ),
            }
            assert all(equal.values())
            report["states"][state_name][mode] = {
                **equal,
                "exact_KL": reference[0][4].item(),
                "selected_KL": result[0][4].item(),
            }
    write(OUT / "audit/kl_bias_fixed_data.json", report)
    print(report)


if __name__ == "__main__":
    main()
