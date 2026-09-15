"""N setup with a single experiment-only perturbation to the scheduler KL."""

import argparse

import torch
from adapter import OUT, append, config, install

from sharpa_rl_unilab.training.teacher_runtime import train_teacher

BIAS = 0.00022029876708984375


def perturb_loss(original, mode, bias=BIAS):
    def loss(self, *args):
        result = original(self, *args)
        if mode == "constant":
            selected = result[4] + bias
        else:
            # Replace only the logarithmic term; retain the new normalized-square terms.
            mean, std = self._loss_distribution.mean, self._loss_distribution.std
            old_mean, old_std = args[-2:]
            with torch.no_grad():
                selected = (
                    (
                        torch.log(std / old_std + 1e-5)
                        + 0.5 * ((old_std / std).square() + ((old_mean - mean) / std).square() - 1)
                    )
                    .sum(-1)
                    .mean()
                )
        if hasattr(self, "_kl_selected"):
            self._kl_selected.append(selected.detach())
        return (*result[:4], selected, *result[5:])

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["constant", "original_log"], default="constant")
    parser.add_argument("--bias", type=float, default=BIAS)
    parser.add_argument("--iterations", type=int, default=501)
    parser.add_argument("--name", default="N_kl_bias_seed1")
    args = parser.parse_args()
    run = OUT / args.name
    if run.exists():
        raise FileExistsError(run)
    cfg = config("N", run, args.iterations)
    cfg.experiment.kl_intervention = {
        "mode": args.mode,
        "constant_bias": args.bias if args.mode == "constant" else None,
        "epsilon_inside_log": 1e-5 if args.mode == "original_log" else None,
        "scope": "scheduler KL and its metric only; optimization loss unchanged",
        "audit": "updates.jsonl retains unperturbed exact_kl and old_kl; branch/lr are applied",
    }
    learner_type = install("N", run)
    learner_type._minibatch_loss_tensors = perturb_loss(
        learner_type._minibatch_loss_tensors, args.mode, args.bias
    )
    original_update = learner_type.update

    def update(self, batch):
        self._kl_selected = []
        result = original_update(self, batch)
        append(
            run / "selected_kl.jsonl",
            {
                "iteration": self._update_counter,
                "selected_kl": torch.stack(self._kl_selected).cpu().tolist(),
            },
        )
        return result

    learner_type.update = update
    print(train_teacher(cfg), flush=True)


if __name__ == "__main__":
    main()
