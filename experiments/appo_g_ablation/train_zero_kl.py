"""Experiment-only exact-KL scheduler: allow the first zero KL to raise LR."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sharpa_rl_unilab.algos.hora.teacher import TeacherAPPOLearner
from sharpa_rl_unilab.cli import _main


def install(run: Path, *, control: bool):
    original_init = TeacherAPPOLearner.__init__
    original_schedule = TeacherAPPOLearner._update_adaptive_learning_rate
    original_update = TeacherAPPOLearner.update

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        assert self.kl_mode == "exact"
        self._experiment_schedule = []
        initial = {
            key: {k: v.detach().cpu().clone() for k, v in getattr(self, key).state_dict().items()}
            for key in ("actor", "critic", "target_actor")
        }
        initial["rng"] = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
        torch.save(initial, run / "initial.pt")

    def schedule(self, kl):
        before, reset = self.learning_rate, self._skip_kl_schedule
        if not control:
            self._skip_kl_schedule = False
        original_schedule(self, kl)
        self._experiment_schedule.append(
            {
                "kl": kl,
                "reference_reset": reset,
                "lr_before": before,
                "lr_after": self.learning_rate,
                "skipped": bool(control and reset),
            }
        )

    def update(self, batch):
        self._experiment_schedule = []
        result = original_update(self, batch)
        record = {"iteration": self._update_counter, "minibatches": self._experiment_schedule}
        with (run / "schedule.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        return result

    TeacherAPPOLearner.__init__ = initialize
    TeacherAPPOLearner._update_adaptive_learning_rate = schedule
    TeacherAPPOLearner.update = update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--iterations", type=int, default=501)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    assert not args.run.exists(), "Do not overwrite a training run"
    install(args.run, control=args.control)
    _main(
        play=False,
        argv=[
            "--algo",
            "appo",
            f"training.log_dir={args.run}",
            "training.no_play=true",
            f"algo.max_iterations={args.iterations}",
            f"+experiment.skip_first_kl_schedule={str(args.control).lower()}",
            "+experiment.name=exact_zero_kl_increase",
        ],
    )
    (args.run / "intervention.json").write_text(
        json.dumps(
            {
                "skip_first_kl_schedule": args.control,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "definition": "Keep exact KL and all native settings; disable only the first-schedule skip flag",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
