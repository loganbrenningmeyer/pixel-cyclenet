import math
import torch

from cyclenet.diffusion import DiffusionSchedule
from cyclenet.diffusion.sampling import ddpm_steps_from_strength, ddim_steps_from_strength

device = torch.device("cpu")
sched = DiffusionSchedule(
    schedule="linear",
    T=1000,
    beta_start=1e-4,
    beta_end=0.02,
    device=device,
)

def retained_signal_ddpm(strength: float):
    t = ddpm_steps_from_strength(sched, strength)[-1]
    ab = float(sched.alpha_bars[t])
    return {
        "t": t,
        "alpha_bar": ab,
        "source_amplitude": math.sqrt(ab),
        "source_variance": ab,
        "noise_amplitude": math.sqrt(1.0 - ab),
        "noise_variance": 1.0 - ab,
        "snr": ab / (1.0 - ab),
    }

def retained_signal_ddim(strength: float, num_steps: int = 100):
    t = ddim_steps_from_strength(sched, num_steps, strength)[-1]
    ab = float(sched.alpha_bars[t])
    return {
        "t": t,
        "alpha_bar": ab,
        "source_amplitude": math.sqrt(ab),
        "source_variance": ab,
        "noise_amplitude": math.sqrt(1.0 - ab),
        "noise_variance": 1.0 - ab,
        "snr": ab / (1.0 - ab),
    }

strengths = [0.1, 0.2, 0.3, 0.4, 0.5]

for s in strengths:
    print(f"Strength {s}".center(40, "=") + f"\n{retained_signal_ddim(s, num_steps=100)}")

