"""CIFAR-10 as channel-major RGB vectors in [0, 1]."""
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
HALF_DIM = 1536


def load_task_data(data_dir=ROOT / "data", train_samples=None, test_samples=None):
    """Download/cache CIFAR-10 locally; keep images on CPU until batched."""
    splits = []
    for train, limit in ((True, train_samples), (False, test_samples)):
        dataset = CIFAR10(root=str(data_dir), train=train, download=True)
        images = torch.from_numpy(dataset.data[:limit].copy())
        images = images.permute(0, 3, 1, 2).reshape(len(images), -1)
        splits.append(images.float().div_(255.0))
    return tuple(splits)


def split_batch(images):
    x1, target = images.chunk(2, dim=1)
    return x1, torch.zeros_like(target), target
