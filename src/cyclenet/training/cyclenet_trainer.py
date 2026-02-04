import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from cyclenet.models import CycleNet
from cyclenet.diffusion import DiffusionSchedule


class CycleNetTrainer:
    def __init__(
        self,
        model: CycleNet,
        sched: DiffusionSchedule,
        dataloader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.sched = sched
        self.dataloader = dataloader
        self.device = device

    def train(self, steps: int):
        """
        
        """
        pass

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        """
        pass
