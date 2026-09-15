"""Train PPO, APPO and FlashSAC teachers serially with Hydra defaults."""

import subprocess
import sys


def main():
    for algorithm in ("ppo", "appo", "flashsac"):
        print(f"Training {algorithm} teacher", flush=True)
        subprocess.run(
            [sys.executable, "-m", "sharpa_rl_unilab.cli", "--algo", algorithm],
            check=True,
        )


if __name__ == "__main__":
    main()
