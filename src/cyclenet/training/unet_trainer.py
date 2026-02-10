import torch
import torch.nn as nn
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.amp import GradScaler, autocast
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
        is_main: bool,
        start_step: int = 1
    ):
        self.model = model
        self.ema_model = ema_model
        self.domain_emb = domain_emb
        self.sched = sched
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.log_config = log_config
        self.ema_decay = ema_decay
        self.is_main = is_main
        self.start_step = start_step

        # -- Setup tensorboard
        self.train_dir = Path(train_dir)
        self.ckpt_dir = self.train_dir / "checkpoints"
        self.fig_dir = self.train_dir / "figs"
        self.tb_dir = self.train_dir / "tb"

        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir)) if self.is_main else None

        # -- Create GradScaler
        self.scaler = GradScaler(device="cuda")

        self.model.train()
        self.ema_model.eval()

    def train(self, steps: int):
        """
        
        """
        step = self.start_step
        epoch = 1

        while step < steps:
            self.model.train()

            # ----------
            # Set DistributedSampler epoch
            # ----------
            sampler = getattr(self.dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

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

                if self.is_main and step % self.log_config.loss_interval == 0:
                    self.log_loss("train/batch_loss", loss.item(), step, epoch)

                if self.is_main and step % self.log_config.ckpt_interval == 0:
                    self.save_checkpoint(step)
                if dist.is_initialized():
                    dist.barrier()

                if self.is_main and step % self.log_config.sample_interval == 0:
                    self.generate_samples(step, epoch)
                if dist.is_initialized():
                    dist.barrier()

                epoch_loss += loss.item()
                num_batches += 1
                step += 1

            # -------------------------
            # Log average epoch loss
            # -------------------------
            epoch_loss /= num_batches
            print(f"Epoch {epoch} Average Loss: {epoch_loss}")
            epoch += 1

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def train_step(self, x_0: torch.Tensor, d_idx: torch.Tensor) -> torch.Tensor:
        """
        
        """
        self.optimizer.zero_grad()

        # -- Mixed-precision forward & loss
        with autocast(device_type="cuda"):
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

        # -------------------------
        # Guard against NaNs / Infs
        # -------------------------
        if not torch.isfinite(loss):
            if self.is_main:
                print(
                    f"[NaN] skipping step — "
                    f"loss={loss.item()}, "
                    f"t[min,max]=({t.min().item()},{t.max().item()})",
                    flush=True,
                )
            self.optimizer.zero_grad(set_to_none=True)
            return loss.detach()

        # -- Scale, backward, step, and update
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(unwrap(self.model).parameters()) +
            list(unwrap(self.domain_emb).parameters()),
            max_norm=1.0
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_ema()

        return loss

    @torch.no_grad()
    def update_ema(self):
        """
        
        """
        for p_ema, p_model in zip(self.ema_model.parameters(), unwrap(self.model).parameters()):
            p_ema.mul_(self.ema_decay).add_(p_model, alpha=1.0 - self.ema_decay)

    def log_loss(self, label: str, loss: float, step: int, epoch: int):
        """
        Logs loss to tensorboard
        """
        if self.writer is None:
            return
        self.writer.add_scalar(label, loss, step)
        self.writer.add_scalar("train/epoch", epoch, step)

    def save_checkpoint(self, step: int):
        """
        Saves model checkpoint (model, EMA, DomainEmbedding)
        """
        ckpt_path = Path(self.train_dir) / "checkpoints" / f"model-step{step}.ckpt"

        torch.save({
            "model": unwrap(self.model).state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "domain_emb": unwrap(self.domain_emb).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step
        }, ckpt_path)

    @torch.no_grad()
    def generate_samples(self, step: int, epoch: int):
        if not self.is_main:
            return

        model = unwrap(self.model)
        domain_emb = unwrap(self.domain_emb)

        model.eval()
        self.ema_model.eval()

        # -------------------------
        # Setup
        # -------------------------
        num_samples = int(self.log_config.sample.num_samples)
        shape = self.log_config.sample.shape
        sampler = self.log_config.sample.sampler

        # -------------------------
        # d_idx = first half 0s, second half 1s
        # -------------------------
        n0 = num_samples // 2
        n1 = num_samples - n0
        d_idx = torch.cat([
            torch.zeros(n0, device=self.device, dtype=torch.long),
            torch.ones(n1, device=self.device, dtype=torch.long),
        ], dim=0)
        d_emb = domain_emb(d_idx)

        # -------------------------
        # Generate samples
        # -------------------------
        x_ref = torch.empty((num_samples, *shape), device=self.device, dtype=torch.float32)
        fig_dir = Path(self.train_dir) / "figs"

        # -- DDPM
        if sampler.lower() == "ddpm":
            unet_samples = unet_ddpm_loop(model, x_ref, d_emb, self.sched)
            ema_samples  = unet_ddpm_loop(self.ema_model, x_ref, d_emb, self.sched)
        # -- DDIM
        elif sampler.lower() == "ddim":
            num_steps = self.log_config.sample.ddim_steps
            eta = self.log_config.sample.eta
            unet_samples = unet_ddim_loop(model, x_ref, d_emb, self.sched, num_steps, eta)
            ema_samples  = unet_ddim_loop(self.ema_model, x_ref, d_emb, self.sched, num_steps, eta)
        else:
            raise ValueError("Sampler must be 'ddpm' or 'ddim'.")
        
        model.train()

        unet_out_path = fig_dir / f"step{step}_unet.png"
        ema_out_path  = fig_dir / f"step{step}_ema.png"
        unet_grid = self.save_samples(unet_samples, unet_out_path)
        ema_grid  = self.save_samples(ema_samples, ema_out_path)

        self.writer.add_image("figs/unet_samples", unet_grid, step)
        self.writer.add_image("figs/ema_samples", ema_grid, step)
        self.writer.add_scalar("train/epoch", epoch, step)
        
    def save_samples(self, samples: torch.Tensor, out_path: str):
        # -------------------------
        # Save sample images
        # -------------------------
        x_vis = (samples.clamp(-1, 1) + 1) / 2    # [0, 1]
        grid = make_grid(x_vis, nrow=4)
        save_image(grid, out_path)
        return grid
    

def unwrap(m):
    return m.module if hasattr(m, "module") else m