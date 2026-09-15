import argparse

from adapter import OUT, config, install

from sharpa_rl_unilab.training.teacher_runtime import train_teacher

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=list("RABCDN"))
    p.add_argument("--iterations", type=int, default=501)
    p.add_argument("--name")
    args = p.parse_args()
    run = OUT / (args.name or f"{args.group}_seed1")
    cfg = config(args.group, run, args.iterations)
    install(args.group, run)
    print(train_teacher(cfg), flush=True)
