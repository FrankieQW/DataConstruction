from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[tuple[int, int]]):
    """Map a global consumed-sample offset to a deterministic shuffled stream."""

    def __init__(
        self,
        dataset_size: int,
        start_offset: int,
        sample_count: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if start_offset < 0 or sample_count < 0:
            raise ValueError("start_offset and sample_count must be non-negative")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size 无效")
        self.dataset_size = int(dataset_size)
        self.start_offset = int(start_offset)
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        stop_offset = self.start_offset + self.sample_count
        position = self.start_offset + self.rank
        cached_epoch = -1
        permutation: list[int] = []
        while position < stop_offset:
            epoch, offset = divmod(position, self.dataset_size)
            if epoch != cached_epoch:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.seed + epoch)
                permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
                cached_epoch = epoch
            dataset_index = permutation[offset]
            sample_seed = _sample_seed(self.seed, position)
            yield dataset_index, sample_seed
            position += self.world_size

    def __len__(self) -> int:
        remaining = max(self.sample_count - self.rank, 0)
        return (remaining + self.world_size - 1) // self.world_size

    def state_dict(self, consumed_samples: int) -> dict[str, int]:
        return {
            "samples_consumed": int(consumed_samples),
            "sampler_seed": self.seed,
            "dataset_size": self.dataset_size,
            "world_size": self.world_size,
        }


def _sample_seed(seed: int, position: int) -> int:
    # NumPy accepts unsigned 64-bit seeds. SplitMix64 avoids nearby positions
    # producing visibly related attribute samples.
    value = (int(seed) + int(position) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)
