"""Conditional intervention: replay a predeclared main-pair LR sequence."""

import argparse
import hashlib
import json

from adapter import OUT, config, install

from sharpa_rl_unilab.training.teacher_runtime import train_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=list("ABCD"))
    parser.add_argument("--reference", required=True, choices=["A", "C"])
    parser.add_argument("--iterations", type=int, default=501)
    parser.add_argument("--name")
    args = parser.parse_args()
    declared = json.loads((OUT / "manifest.json").read_text())["conditional_LR_reference"]
    assert declared["A/B" if args.group in "AB" else "C/D"] == args.reference
    source = OUT / f"{args.reference}_seed1/updates.jsonl"
    frozen = json.loads((OUT / f"audit/lr_reference_{args.reference}.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen["source_sha256"]
    sequence = frozen["sequence"]
    assert len(sequence) == 10020
    run = OUT / (args.name or f"{args.group}_fixed_lr_seed1")
    cfg = config(args.group, run, args.iterations)
    cfg.experiment.lr_reference = str(source)
    cfg.experiment.lr_reference_sha256 = frozen["source_sha256"]
    cfg.experiment.lr_intervention = "minibatch-index replay; recorded branch is the native counterfactual, not the applied LR rule"
    learner_type = install(args.group, run)
    original = learner_type._update_adaptive_learning_rate

    def replay(self, kl):
        original(self, kl)
        index = self._update_counter * 20 + len(self._audit_lr) - 1
        self.learning_rate = sequence[index]
        for params in self.optimizer.param_groups:
            params["lr"] = self.learning_rate
        self._audit_lr[-1][1] = self.learning_rate

    learner_type._update_adaptive_learning_rate = replay
    print(train_teacher(cfg), flush=True)


if __name__ == "__main__":
    main()
