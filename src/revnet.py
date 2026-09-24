"""Reversible additive coupling with bounded, shape-preserving deltas."""
import torch
from torch import nn
from torch.nn import functional as F

COUPLING_MAX_NORM = 10.0


def coupling_activation(x: torch.Tensor) -> torch.Tensor:
    delta = F.leaky_relu(x)
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return delta * (COUPLING_MAX_NORM / norm).clamp(max=1.0)


class RevNet(nn.Module):
    """Even layers update x1; odd layers update x2. Weights are unshared."""

    def __init__(self, depth: int = 6, half_dim: int = 1536):
        super().__init__()
        if depth < 2 or half_dim < 1:
            raise ValueError("depth must be >= 2 and half_dim must be positive")
        self.layers = nn.ModuleList(nn.Linear(half_dim, half_dim) for _ in range(depth))

    def apply_layer(self, index, x1, x2):
        layer = self.layers[index]
        if index % 2 == 0:
            return x1 + coupling_activation(layer(x2)), x2
        return x1, x2 + coupling_activation(layer(x1))

    def forward(self, x1, x2):
        for i in range(len(self.layers)):
            x1, x2 = self.apply_layer(i, x1, x2)
        return x1, x2

    def states(self, x1, x2):
        """Return the state after each layer for local training."""
        outputs = []
        for i in range(len(self.layers)):
            x1, x2 = self.apply_layer(i, x1, x2)
            outputs.append((x1, x2))
        return outputs

    def targets(self, x1, x2):
        """Invert a terminal target into targets after each layer.

        The inverse is algebraic; floating-point subtraction is not bit-exact.
        """
        targets = [(x1, x2)]
        for i in range(len(self.layers) - 1, 0, -1):
            if i % 2 == 0:
                x1 = x1 - coupling_activation(self.layers[i](x2))
            else:
                x2 = x2 - coupling_activation(self.layers[i](x1))
            targets.append((x1, x2))
        return list(reversed(targets))
