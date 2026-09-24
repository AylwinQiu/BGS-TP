"""Shared BP, sequential target-propagation and grouped Jacobi updates."""
import torch
from torch.nn import functional as F

from .data import split_batch


class Trainer:
    def __init__(self, model, method="bgstp", lr=1e-3, layer_select="random",
                 layers_per_step=1, lr_decay_to_frac=None):
        if method not in {"bgstp", "bp", "jacobi"}:
            raise ValueError(f"unknown method: {method}")
        if layer_select not in {"random", "round-robin"}:
            raise ValueError(f"unknown layer selection: {layer_select}")
        if not 1 <= layers_per_step <= len(model.layers):
            raise ValueError("layers_per_step must be between 1 and depth")
        if lr <= 0 or (lr_decay_to_frac is not None and not 0 < lr_decay_to_frac <= 1):
            raise ValueError("lr must be positive and lr_decay_to_frac must be in (0, 1]")
        self.model, self.method, self.lr = model, method, lr
        self.layer_select, self.layers_per_step = layer_select, layers_per_step
        self.lr_decay_to_frac = lr_decay_to_frac
        self.pointer = 0
        params = [model.parameters()] if method == "bp" else [
            layer.parameters() for layer in model.layers
        ]
        self.optimizers = [torch.optim.Adam(p, lr=lr) for p in params]

    def _update(self, loss, indices):
        for i in indices:
            self.optimizers[i].zero_grad(set_to_none=True)
        loss.backward()
        for i in indices:
            self.optimizers[i].step()

    @torch.no_grad()
    def _snapshot(self, x1, x2, truth):
        outputs = self.model.states(x1, x2)
        return outputs, self.model.targets(outputs[-1][0], truth)

    def step(self, images, step=1, total_steps=1):
        if self.lr_decay_to_frac is not None:
            fraction = 1 - (1 - self.lr_decay_to_frac) * (step - 1) / max(total_steps - 1, 1)
            for optimizer in self.optimizers:
                optimizer.param_groups[0]["lr"] = self.lr * fraction
        x1, x2, truth = split_batch(images)
        if self.method == "bp":
            loss = F.mse_loss(self.model(x1, x2)[1], truth)
            self._update(loss, [0])
            return loss.item()

        depth = len(self.model.layers)
        if self.method == "bgstp":
            if self.layer_select == "round-robin":
                indices = [(self.pointer + j) % depth for j in range(self.layers_per_step)]
                self.pointer = (self.pointer + self.layers_per_step) % depth
            else:
                indices = torch.randperm(depth)[:self.layers_per_step].tolist()
            for i in indices:
                # Recompute BOTH inputs and targets after each single-layer update.
                outputs, targets = self._snapshot(x1, x2, truth)
                inputs = (x1, x2) if i == 0 else outputs[i - 1]
                actual = self.model.apply_layer(i, *inputs)
                loss = F.mse_loss(torch.cat(actual, dim=1), torch.cat(targets[i], dim=1))
                self._update(loss, [i])
        else:
            # Original two-phase baseline: adjacent pairs share a stale target snapshot.
            _, targets = self._snapshot(x1, x2, truth)
            offset = step % 2
            groups = [(0,)] if offset else []
            groups += [tuple(range(i, min(i + 2, depth))) for i in range(offset, depth, 2)]
            for group in groups:
                for i in group:
                    x1, x2 = self.model.apply_layer(i, x1, x2)
                loss = F.mse_loss(torch.cat((x1, x2), dim=1),
                                  torch.cat(targets[group[-1]], dim=1))
                self._update(loss, group)
                x1, x2 = x1.detach(), x2.detach()
        return loss.item()


@torch.no_grad()
def evaluate(model, images, n=None, batch_size=256):
    """Sample-weighted MSE, evaluated in batches to bound memory use."""
    if n is not None and n < len(images):
        images = images[torch.randperm(len(images), device=images.device)[:n]]
    device = next(model.parameters()).device
    squared_error, count = 0.0, 0
    for batch in images.split(batch_size):
        x1, x2, truth = split_batch(batch.to(device))
        squared_error += F.mse_loss(model(x1, x2)[1], truth, reduction="sum").item()
        count += truth.numel()
    return squared_error / count
