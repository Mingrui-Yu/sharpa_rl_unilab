"""Conditional single-factor restoration of the original target log-density order."""

import argparse
import types

import torch
from adapter import OUT, config, install

from sharpa_rl_unilab.algos.hora.distribution import tanh_log_det
from sharpa_rl_unilab.training.teacher_runtime import train_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="A_target_density_seed1")
    parser.add_argument("--iterations", type=int, default=501)
    args = parser.parse_args()
    run = OUT / args.name
    cfg = config("A", run, args.iterations)
    cfg.experiment.restoration = (
        "target_actor.get_output_log_prob only: original Normal.log_prob operation order"
    )
    learner_type = install("A", run)
    original = learner_type.process_batch

    def density(self, raw):
        return torch.distributions.Normal(
            self.output_mean, self.output_std, validate_args=False
        ).log_prob(raw).sum(-1) - tanh_log_det(raw)

    def process(self, batch):
        if self._update_counter == 0:
            self.target_actor.get_output_log_prob = types.MethodType(density, self.target_actor)
        return original(self, batch)

    learner_type.process_batch = process
    print(train_teacher(cfg), flush=True)


if __name__ == "__main__":
    main()
