import wandb
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from cyclenet.models import UNet
from cyclenet.models.conditioning import DomainEmbedding
from cyclenet.diffusion import *
from cyclenet.diffusion.losses import unet_loss


class UNetTrainer:
    def __init__(
        self,
        model: UNet,
        ema_model: UNet,
        domain_emb: DomainEmbedding,
        sched: DiffusionSchedule,
        optimizer: Optimizer,
        dataloader: DataLoader,
        device: torch.device,
        train_dir: str,
        log_config: DictConfig,
        ema_decay: float,
        start_step: int = 1
    ):
        self.model = model
        self.ema_model = ema_model
        self.domain_emb = domain_emb
        self.sched = sched
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.train_dir = train_dir
        self.log_config = log_config
        self.ema_decay = ema_decay
        self.start_step = start_step

        self.model.train()
        self.ema_model.eval()

    def train(self, steps: int):
        """
        
        """
        step = self.start_step
        epoch = 1

        while step < steps:
            self.model.train()

            # -------------------------
            # Run training epoch
            # -------------------------
            epoch_loss = 0.0
            num_batches = 0

            for x_0, d_idx in tqdm(self.dataloader, desc=f"Epoch {epoch}, Step {step}", unit="Batch"):
                if step >= steps:
                    break

                # -------------------------
                # Perform train step
                # -------------------------
                x_0, d_idx = x_0.to(self.device), d_idx.to(self.device)
                loss = self.train_step(x_0, d_idx)

                if step % self.log_config.loss_interval == 0:
                    self.log_loss("train/batch_loss", loss.item(), step, epoch)

                if step % self.log_config.ckpt_interval == 0:
                    self.save_checkpoint(step)

                if step % self.log_config.sample_interval == 0:
                    self.generate_samples(step, epoch)

                epoch_loss += loss.item()
                num_batches += 1
                step += 1

            # -------------------------
            # Log average epoch loss
            # -------------------------
            epoch_loss /= num_batches
            print(f"Epoch {epoch} Average Loss: {epoch_loss}")
            epoch += 1

    def train_step(self, x_0: torch.Tensor, d_idx: torch.Tensor) -> torch.Tensor:
        """
        
        """
        self.optimizer.zero_grad()

        # -------------------------
        # Sample batch of timesteps
        # -------------------------
        B = x_0.shape[0]
        t = torch.randint(0, self.sched.T, (B,), device=self.device)

        # -------------------------
        # Get domain embeddings
        # -------------------------
        d_emb = self.domain_emb(d_idx)

        # -------------------------
        # Noise / forward / compute loss
        # -------------------------
        loss = unet_loss(self.model, x_0, t, d_emb, self.sched)
        loss.backward()
        self.optimizer.step()
        self.update_ema()

        return loss

    @torch.no_grad()
    def update_ema(self):
        """
        
        """
        for p_ema, p_model in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.mul_(self.ema_decay).add_(p_model, alpha=1.0 - self.ema_decay)

    def log_loss(self, label: str, loss: float, step: int, epoch: int):
        """
        Logs loss to wandb dashboard
        """
        wandb.log(
            {
                label: loss, 
                "epoch": epoch
            }, 
            step=step
        )

    def save_checkpoint(self, step: int):
        """
        Saves model checkpoint (model, EMA, DomainEmbedding)
        """
        ckpt_path = Path(self.train_dir) / "checkpoints" / f"model-step{step}.ckpt"

        torch.save({
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "domain_emb": self.domain_emb.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step
        }, ckpt_path)

    @torch.no_grad()
    def generate_samples(self, step: int, epoch: int):
        """
        Generates UNet samples and saves to figs dir
        """
        num_samples = self.log_config.sample.num_samples
        shape = self.log_config.sample.shape
        sampler = self.log_config.sample.sampler

        x_ref = torch.empty((num_samples, *shape), device=self.device, dtype=torch.float32)

        # -------------------------
        # Sample random domain embeddings
        # -------------------------
        d_probs = torch.empty((num_samples,)).uniform_(0, 1)
        d_idx = torch.bernoulli(d_probs).to(self.device)
        d_emb = self.domain_emb(d_idx)

        # -------------------------
        # Generate / save samples
        # -------------------------
        fig_dir = Path(self.train_dir) / "figs"
        # -- DDPM
        if sampler.lower() == "ddpm":
            unet_samples = unet_ddpm_loop(self.model, x_ref, d_emb, self.sched)
            ema_samples = unet_ddpm_loop(self.ema_model, x_ref, d_emb, self.sched)
        # -- DDIM
        elif sampler.lower() == "ddim":
            num_steps = self.log_config.sample.ddim_steps
            eta = self.log_config.sample.eta
            unet_samples = unet_ddim_loop(self.model, x_ref, d_emb, self.sched, num_steps, eta)
            ema_samples = unet_ddim_loop(self.ema_model, x_ref, d_emb, self.sched, num_steps, eta)
        else:
            raise ValueError("Sampler must be 'ddpm' or 'ddim'.")
        
        unet_out_path = fig_dir / f"step{step}_unet.png"
        ema_out_path = fig_dir / f"step{step}_ema.png"
        self.save_samples(unet_samples, unet_out_path)
        self.save_samples(ema_samples, ema_out_path)

        # -------------------------
        # Log samples
        # -------------------------
        wandb.log(
            {
                "figs/unet_samples": wandb.Image(unet_out_path),
                "epoch": epoch
            },
            step=step
        )

        wandb.log(
            {
                "figs/ema_samples": wandb.Image(ema_out_path),
                "epoch": epoch
            },
            step=step
        )
        
    def save_samples(self, samples: torch.Tensor, out_path: str):
        # -------------------------
        # Save sample images
        # -------------------------
        x_vis = (samples.clamp(-1, 1) + 1) / 2    # [0, 1]
        grid = make_grid(x_vis, nrow=4)
        save_image(grid, out_path)