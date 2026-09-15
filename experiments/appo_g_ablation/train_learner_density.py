"""Conditional restoration of only the original learner Gaussian density order."""

import argparse

from adapter import OUT, config, install
from uni_rl.algos.appo.learner import APPOLearner

from sharpa_rl_unilab.algos.hora.distribution import tanh_log_det
from sharpa_rl_unilab.training.teacher_runtime import train_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=501)
    parser.add_argument("--name", default="A_learner_density_seed1")
    args = parser.parse_args()
    run = OUT / args.name
    cfg = config("A", run, args.iterations)
    cfg.experiment.restoration = "learner Gaussian log-probability operation order only; actor/collector/target density unchanged"
    learner_type = install("A", run)

    def density(self, actions, mean, std):
        return APPOLearner._gaussian_log_prob(actions, mean, std) - tanh_log_det(actions)

    learner_type._gaussian_log_prob = density
    print(train_teacher(cfg), flush=True)


if __name__ == "__main__":
    main()
