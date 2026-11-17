import torch
from pathlib import Path
import os
from src.config.loader import ConfigLoader

path = Path(os.getcwd()) / "config" / "project.yaml"


loader = ConfigLoader()
loader.load(path)
config = loader.get()

t = torch.randint(0, config.diffusion.T, (config.training.batch_size,)).long()

betas = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.T)

alphas = 1 - betas

cum_alphas = torch.cumprod(alphas, dim=0)

print(cum_alphas[t])
