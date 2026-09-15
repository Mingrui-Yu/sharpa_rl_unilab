"""Evaluate each conditional run after all conditional training has finished."""

import concurrent.futures
import json
import time

from adapter import OUT, write
from run_evaluations import evaluate_group


def main():
    while not (OUT / "additional_complete.json").exists():
        time.sleep(10)
    names = json.loads((OUT / "additional_complete.json").read_text())["runs"]
    assert len(names) <= 4
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(evaluate_group, name, index) for index, name in enumerate(names)]
        for task in tasks:
            task.result()
    write(
        OUT / "evaluation/additional_complete.json",
        {"runs": names, "wall_seconds": time.time() - start},
    )


if __name__ == "__main__":
    main()
