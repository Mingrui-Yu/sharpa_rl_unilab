"""Mixed Muon/Adam optimizers for SuperMimic networks."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


class MuonAdam:
    """Optimize hidden matrices with Muon and all other tensors with Adam.

    Muon is intended for two-dimensional hidden-layer weight matrices. Input and
    output projections, biases, normalizers, and other vector parameters remain
    on Adam. ``match_rms_adamw`` scaling lets Muon reuse the existing Adam
    learning rate without a separate learning-rate search.
    """

    def __init__(
        self,
        model: nn.Module,
        mlps: Iterable[nn.Module],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        fused: bool = False,
        output_mlps: Iterable[nn.Module] = (),
    ) -> None:
        """Create the mixed optimizer.

        ``mlps`` are network bodies whose interior Linear weights use Muon; their
        first and final Linear layers stay on Adam. ``output_mlps`` additionally
        selects the final Linear layer. That is useful for a trunk whose output
        projection is a separate head rather than the final layer of the module.
        """

        hidden: set[int] = set()
        for mlp in mlps:
            layers = _linear_layers(mlp)
            if len(layers) < 2:
                raise ValueError("MuonAdam requires an MLP with at least two Linear layers")
            hidden.update(id(layer.weight) for layer in layers[1:-1])

        for mlp in output_mlps:
            layers = _linear_layers(mlp)
            if len(layers) < 2:
                raise ValueError("MuonAdam requires an MLP with at least two Linear layers")
            hidden.update(id(layer.weight) for layer in layers[1:])

        muon_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) in hidden and parameter.requires_grad
        ]
        adam_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in hidden and parameter.requires_grad
        ]
        if not muon_parameters:
            raise ValueError("MuonAdam requires at least one hidden Linear weight matrix")

        self.muon = torch.optim.Muon(
            muon_parameters,
            lr=lr,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.Adam(adam_parameters, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self) -> list[dict]:
        """Return live parameter groups from both wrapped optimizers."""

        return self.muon.param_groups + self.adam.param_groups

    @property
    def defaults(self) -> dict:
        """Expose defaults for optimizer metadata and lightweight introspection."""

        return {**self.muon.defaults, **self.adam.defaults}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        with torch.no_grad():
            self.muon.step()
            self.adam.step()
        return loss

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        learning_rates = [group["lr"] for group in self.param_groups]
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
        # Optimizer state follows the model parameters, while runtime settings
        # such as the configured learning rate remain owned by the new run.
        for group, learning_rate in zip(self.param_groups, learning_rates, strict=True):
            group["lr"] = learning_rate


__all__ = ["MuonAdam"]
