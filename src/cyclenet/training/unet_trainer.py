import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from cyclenet.models import UNet
from cyclenet.models.diffusion import DiffusionSchedule


class UNetTrainer:
    def __init__(
        self,
        model: UNet,
        sched: DiffusionSchedule,
        dataloader: DataLoader,
        device: torch.device,
        ema_decay: float
    ):
        self.model = model
        self.sched = sched
        self.dataloader = dataloader
        self.device = device
        self.ema_decay = ema_decay

        # -------------------------
        # Initialize EMA model
        # -------------------------
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.to(device)


    def train(self, steps: int):
        """
        
        """
        pass

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        """
        pass

    @torch.no_grad()
    def update_ema(self):
        """
        
        """
        for p_ema, p_model in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.mul_(self.ema_decay).add_(p_model, alpha=1.0 - self.ema_decay)

    
        