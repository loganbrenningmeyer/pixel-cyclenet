import torch
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from cyclenet.models import CycleNet
from cyclenet.diffusion import *
from cyclenet.diffusion.losses import cyclenet_loss


class CycleNetTrainer:
    def __init__(
        self,
        model: CycleNet,
        ema_model: CycleNet,
        sched: DiffusionSchedule,
        optimizer: Optimizer,
        dataloader: DataLoader,
        sample_loader: DataLoader,
        device: torch.device,
        train_dir: str,
        log_config: DictConfig,
        sample_config: DictConfig,
        ema_decay: float,
        is_main: bool,
        recon_weight: float = 1.0,
        cycle_weight: float = 0.01,
        consis_weight: float = 0.1,
        invar_weight: float = 0.1,
        start_step: int = 1,
        start_epoch: int = 1,
    ):
        self.model = model
        self.ema_model = ema_model
        self.sched = sched
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.sample_loader = sample_loader
        self.sample_iter = iter(sample_loader) if sample_loader is not None else None
        self.device = device
        self.log_config = log_config
        self.sample_config = sample_config
        self.ema_decay = ema_decay
        self.is_main = is_main
        self.start_step = start_step
        self.start_epoch = start_epoch

        # -- Loss weights
        self.recon_weight = recon_weight
        self.cycle_weight = cycle_weight
        self.consis_weight = consis_weight
        self.invar_weight = invar_weight

        # -- Create directories
        self.train_dir = Path(train_dir)
        self.ckpt_dir = self.train_dir / "checkpoints"
        self.fig_dir = self.train_dir / "figs"
        self.tb_dir = self.train_dir / "tb"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # -- Setup tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.tb_dir)) if self.is_main else None

        # -- Create GradScaler
        self.scaler = GradScaler()

        self.model.train()
        self.ema_model.eval()

    def train(self, steps: int):
        """
        
        """
        step = self.start_step
        epoch = self.start_epoch

        while step < steps:
            self.model.train()

            # ----------
            # Set DomainSampler epoch
            # ----------
            if hasattr(self.dataloader.batch_sampler, "set_epoch"):
                self.dataloader.batch_sampler.set_epoch(epoch)

            # ----------
            # Run training epoch
            # ----------
            epoch_loss = 0.0
            num_batches = 0

            for x_0, src_idx, tgt_idx in tqdm(self.dataloader, desc=f"Epoch {epoch}, Step {step}", unit="Batch"):
                if step >= steps:
                    break

                # ----------
                # Perform train step
                # ----------
                x_0, src_idx, tgt_idx = x_0.to(self.device), src_idx.to(self.device), tgt_idx.to(self.device)
                loss_dict, loss = self.train_step(x_0, src_idx, tgt_idx)

                if self.is_main and step % self.log_config.loss_interval == 0:
                    self.log_loss("train/batch_loss", loss_dict, step, epoch)

                if self.is_main and step % self.log_config.ckpt_interval == 0:
                    self.save_checkpoint(step, epoch)
                if dist.is_initialized():
                    dist.barrier()

                if self.is_main and step % self.log_config.sample_interval == 0:
                    self.generate_samples(step, epoch)
                if dist.is_initialized():
                    dist.barrier()

                epoch_loss += loss.item()
                num_batches += 1
                step += 1

            # ----------
            # Log average epoch loss
            # ----------
            if num_batches > 0:
                epoch_loss /= num_batches

            if self.is_main:
                print(f"Epoch {epoch} Average Loss: {epoch_loss}")

            epoch += 1

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


    def train_step(self, x_0: torch.Tensor, src_idx: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        """
        
        """
        self.optimizer.zero_grad()

        with autocast(device_type="cuda"):
            # ----------
            # Sample batch of timesteps
            # ----------
            B = x_0.shape[0]
            t = torch.randint(0, self.sched.T, (B,), device=self.device)

            # ----------
            # Noise / forward / compute loss
            # ----------
            loss_dict = cyclenet_loss(
                model=self.model, 
                x_0=x_0, 
                t=t,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                sched=self.sched
            )

            loss = (
                self.recon_weight * loss_dict["recon"]
                + self.cycle_weight * loss_dict["cycle"]
                + self.consis_weight * loss_dict["consis"]
                + self.invar_weight * loss_dict["invar"]
            )

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
            return loss_dict, torch.zeros((), device=self.device)
        
        # -- Scale, backward, step, and update
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(list(unwrap(self.model).parameters()), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_ema()

        return loss_dict, loss

    @torch.no_grad()
    def update_ema(self):
        """
        
        """
        for p_ema, p_model in zip(self.ema_model.parameters(), unwrap(self.model).parameters()):
            p_ema.mul_(self.ema_decay).add_(p_model, alpha=1.0 - self.ema_decay)        

    def log_loss(self, label: str, loss_dict: dict[str, torch.Tensor], step: int, epoch: int):
        """
        Logs loss to tensorboard
        """
        if self.writer is None:
            return
        
        self.writer.add_scalar(f"{label}/recon", self.recon_weight * loss_dict["recon"].item(), step)
        self.writer.add_scalar(f"{label}/cycle", self.cycle_weight * loss_dict["cycle"].item(), step)
        self.writer.add_scalar(f"{label}/consis", self.consis_weight * loss_dict["consis"].item(), step)
        self.writer.add_scalar(f"{label}/invar", self.invar_weight * loss_dict["invar"].item(), step)
        self.writer.add_scalar("train/epoch", epoch, step)       

    def save_checkpoint(self, step: int, epoch: int):
        """
        Saves model checkpoint (model, EMA, DomainEmbedding)
        """
        ckpt_path = Path(self.train_dir) / "checkpoints" / f"step-{step}.ckpt"

        torch.save({
            "model": unwrap(self.model).state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        }, ckpt_path)

    @torch.no_grad()
    def generate_samples(self, step: int, epoch: int):
        """
        Generates CycleNet translation samples and saves to figs dir
        """
        if not self.is_main or self.writer is None:
            return
        
        model = unwrap(self.model)

        model.eval()
        self.ema_model.eval()

        # -------------------------
        # Define source / target indices (x_src is all source)
        # -------------------------
        x_src = self._next_sample_batch()
        x_src_ctrl = ((x_src + 1.0) / 2.0).clamp(0.0, 1.0)

        B = x_src.shape[0]

        src_idx = torch.zeros((B,), device=self.device, dtype=torch.long)
        tgt_idx = torch.ones((B,), device=self.device, dtype=torch.long)

        # -------------------------
        # Save / log source images
        # -------------------------
        samples_dir = Path(self.train_dir) / "figs" / f"step-{step}"
        samples_dir.mkdir(parents=True, exist_ok=True)

        x_src_grid = self.save_samples(x_src, samples_dir / "x_src.png")
        self.writer.add_image("figs/x_src", x_src_grid, step)

        # -- Track epoch
        self.writer.add_scalar("train/epoch", epoch, step)

        # -------------------------
        # Generate samples for each noise strength / cfg_weight
        # -------------------------
        sampler = self.sample_config.sampler
        cfg_weights = self.sample_config.cfg_weights
        noise_strengths = self.sample_config.noise_strengths

        for noise_strength in noise_strengths:

            for cfg_weight in cfg_weights:
                # -- DDPM
                if sampler.lower() == "ddpm":
                    model_samples, x_noise = cyclenet_ddpm_loop(
                        model=model, 
                        x_src=x_src, 
                        src_idx=src_idx, 
                        tgt_idx=tgt_idx,
                        c_img=x_src_ctrl,
                        sched=self.sched,
                        w=cfg_weight,
                        strength=noise_strength
                    )

                    ema_samples, x_noise = cyclenet_ddpm_loop(
                        model=self.ema_model, 
                        x_src=x_src, 
                        src_idx=src_idx, 
                        tgt_idx=tgt_idx,
                        c_img=x_src_ctrl,
                        sched=self.sched,
                        w=cfg_weight,
                        strength=noise_strength
                    )

                # -- DDIM
                elif sampler.lower() == "ddim":
                    num_steps = self.sample_config.ddim_steps
                    eta = self.sample_config.eta

                    model_samples, x_noise = cyclenet_ddim_loop(
                        model=model, 
                        x_src=x_src, 
                        src_idx=src_idx, 
                        tgt_idx=tgt_idx,
                        c_img=x_src_ctrl,
                        sched=self.sched,
                        w=cfg_weight,
                        strength=noise_strength,
                        num_steps=num_steps,
                        eta=eta
                    )

                    ema_samples, x_noise = cyclenet_ddim_loop(
                        model=self.ema_model, 
                        x_src=x_src, 
                        src_idx=src_idx, 
                        tgt_idx=tgt_idx,
                        c_img=x_src_ctrl,
                        sched=self.sched,
                        w=cfg_weight,
                        strength=noise_strength,
                        num_steps=num_steps,
                        eta=eta
                    )
                else:
                    raise ValueError("Sampler must be 'ddpm' or 'ddim'.")

                # -------------------------
                # Save / log samples for noise_strength / cfg_weight pair
                # -------------------------
                cfg_str = f"cfg-{cfg_weight:.1f}"
                strength_str = f"strength-{noise_strength:.2f}"
                
                model_grid = self.save_samples(model_samples, samples_dir / strength_str / cfg_str / "model.png")
                ema_grid = self.save_samples(ema_samples, samples_dir / strength_str / cfg_str / "ema.png")

                self.writer.add_image(f"figs/model/{strength_str}_{cfg_str}", model_grid, step)
                self.writer.add_image(f"figs/ema/{strength_str}_{cfg_str}", ema_grid, step)

            # -------------------------
            # Save / log noised source images
            # -------------------------
            noise_grid = self.save_samples(x_noise, samples_dir / strength_str / "x_t.png")
            self.writer.add_image(f"figs/x_t/{strength_str}", noise_grid, step)

        # -- Put model back in train mode
        model.train()

    def save_samples(self, samples: torch.Tensor, out_path: str):
        # -------------------------
        # Save sample images
        # -------------------------
        x_vis = (samples.clamp(-1, 1) + 1) / 2    # [0, 1]
        grid = make_grid(x_vis, nrow=4)
        save_image(grid, out_path)
        return grid
    
    def _next_sample_batch(self):
        if self.sample_iter is None:
            return None
        try:
            x_src = next(self.sample_iter)
        except StopIteration:
            self.sample_iter = iter(self.sample_loader)
            x_src = next(self.sample_iter)
        return x_src.to(self.device, non_blocking=True)


def unwrap(m):
    return m.module if hasattr(m, "module") else m