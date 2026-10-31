"""Device-resident time/environment replay with episode-boundary masks."""

from __future__ import annotations

import torch

from .state import STATE_DIM

FIELDS: dict[str, tuple[int | tuple[()], torch.dtype]] = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (22, torch.float32),
    "xi": (55, torch.float32),
    "r": ((), torch.float32),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, steps: int, num_envs: int, device):
        self.steps = steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.data = {
            key: torch.zeros(
                (steps, num_envs) + ((shape,) if isinstance(shape, int) else shape),
                dtype=dtype,
                device=self.device,
            )
            for key, (shape, dtype) in FIELDS.items()
        }
        self.ptr = 0
        self.size = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for key in FIELDS:
            self.data[key][self.ptr].copy_(transition[key])
        self.ptr = (self.ptr + 1) % self.steps
        self.size = min(self.size + 1, self.steps)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + age) % self.steps

    def sample_windows(self, batch: int, window: int) -> dict[str, torch.Tensor]:
        if not 1 <= window <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {window}")
        start = torch.randint(0, self.size - window + 1, (batch,), device=self.device)
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        ages = start[:, None] + torch.arange(window, device=self.device)
        rows = self._row(ages)
        envs = env[:, None].expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in ("s", "s1", "a", "xi", "done")}
        alive = torch.cat(
            (torch.ones(batch, 1, device=self.device), 1 - out["done"][:, :-1].float()), -1
        )
        out["mask"] = alive.cumprod(-1)
        return out

    def sample_starts(self, batch: int, history: int):
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if history < 1:
            raise ValueError("history length must be positive")
        age = torch.randint(0, self.size, (batch,), device=self.device)
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        row = self._row(age)
        episode_step = self.data["ep_step"][row, env].long()
        back = torch.arange(history - 1, -1, -1, device=self.device)
        back = torch.minimum(back[None], episode_step[:, None])
        back = torch.minimum(back, age[:, None])
        rows = self._row(age[:, None] - back)
        envs = env[:, None].expand_as(rows)
        start = {key: self.data[key][row, env] for key in ("s", "xi")}
        histories = {key: self.data[key][rows, envs] for key in ("s", "xi")}
        return start, histories


__all__ = ["FIELDS", "ReplayBuffer"]
