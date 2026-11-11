"""Reproducible sampling with exactly 712 draws per medium per epoch."""
import torch
from torch.utils.data import Sampler

from .dataset import DOMAINS


class BalancedExposureSampler(Sampler):
    """Draw equally with replacement, then deterministically shuffle all draws."""

    def __init__(self, datasets, num_samples=2136, seed=42):
        self.names = [dataset.domain for dataset in datasets]
        self.sizes = [len(dataset) for dataset in datasets]
        if (not self.names or num_samples % len(self.names)
                or any(size <= 0 for size in self.sizes)):
            raise ValueError("Nonempty domains and an exactly divisible exposure budget are required")
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        count = self.num_samples // len(self.names)
        draws, offset = [], 0
        for name, size in zip(self.names, self.sizes):
            generator = torch.Generator().manual_seed(
                self.seed * 1000003 + self.epoch * 101 + DOMAINS.index(name))
            draws.append(torch.randint(size, (count,), generator=generator) + offset)
            offset += size
        generator = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch * 101 + 99)
        order = torch.randperm(self.num_samples, generator=generator)
        return iter(torch.cat(draws)[order].tolist())
