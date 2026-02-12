import math
import numpy as np
import torch
from torch.utils.data import Sampler


class DomainSampler(Sampler[list[int]]):
    """
    Yields batches with half of each domain.
    Works with DDP by sharding batches by rank.
    """
    def __init__(
        self,
        n_real: int,
        n_sim: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        assert batch_size % 2 == 0, "batch_size must be even for 50/50 balancing"
        self.n_real = n_real
        self.n_sim = n_sim
        self.bs = batch_size
        self.half = batch_size // 2
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # ----------
        # Number of batches limited by smaller domain
        # ----------
        self.num_batches = min(n_real, n_sim) // self.half
        # -- Total epoch samples per domain
        self.domain_samples = self.num_batches * self.half

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        # -- Number of batches this DDP rank will produce
        return self.num_batches // self.world_size
    
    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)

        # -- All real / sim sample indices
        real_idx = np.arange(self.n_real)
        sim_idx = np.arange(self.n_sim)

        if self.shuffle:
            g.shuffle(real_idx)
            g.shuffle(sim_idx)

        # -- Consume samples equally from both domains
        real_idx = real_idx[:self.domain_samples]
        sim_idx  = sim_idx[:self.domain_samples]

        # ----------
        # Make batches [real_samples, sim_samples]
        # ----------
        batches = []

        for i in range(self.num_batches):
            # -------------------------
            # Define real / sim sample indices
            # -- sim indices are offset by real for order in ConcatDataset([real, sim])
            # -------------------------
            real_samples = real_idx[i * self.half : (i + 1) * self.half]
            sim_samples  = sim_idx[i * self.half : (i + 1) * self.half] + self.n_real

            # -- Concatenate real / sim batch
            batch = np.concatenate([real_samples, sim_samples])

            # -- Shuffle / store batch
            if self.shuffle:
                g.shuffle(batch)

            batches.append(batch.tolist())

        # ----------
        # Shard batches by rank
        # -- [rank + 0, rank + world_size, rank + 2 * world_size, ...]
        # ----------
        for b in batches[self.rank::self.world_size]:
            yield b

        