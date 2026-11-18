import torch
import warnings
from typing import Any, Optional
from pathlib import Path
from diffusers import UNet2DModel
from ..config import ConfigLoader


class DiffusionModel(torch.nn.Module):
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor

    def __init__(self):
        super().__init__()

        # Loading the configuration
        self.config = ConfigLoader().get()

        # Loading the unet backbone
        self.model = UNet2DModel(
            sample_size=32,  # CIFAR-10 images are 32x32
            in_channels=3,   # RGB images
            out_channels=3,  # Predicting noise for RGB images
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

        # Loading the diffusion process parameters (betas)
        betas = torch.linspace(self.config.diffusion.beta_start, self.config.diffusion.beta_end, self.config.diffusion.T)

        # Transforming the betas into alphas for computational efficiency
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).view(-1, 1, 1, 1)
        # Reshaped to (T, 1, 1, 1) for broadcasting during the forward pass

        self.register_buffer("betas", betas, False)
        self.register_buffer("alphas", alphas, False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, False)

    def _diffusion_params(self) -> dict:
        d = self.config.diffusion
        return {
            "T": d.T,
            "beta_start": d.beta_start,
            "beta_end": d.beta_end
        }

    @torch.no_grad()
    def generate(self, amount: int = 5) -> torch.Tensor:
        """Generates a whole new batch of images

        Args:
            amount (int): How many images will be generated. Defaults to 5
        """
        self.eval()

        device = self.alphas.device

        # Sorting the noise that the model will generate
        x_t = torch.randn((amount, 3, 32, 32), device=device)

        for t in range(self.config.diffusion.T-1, -1, -1):
            t_batch = torch.full((amount,), t, device=device, dtype=torch.long)
            z = torch.randn_like(x_t, device=device) if t > 0 else torch.zeros_like(x_t, device=device)

            generated_noise = self.model(x_t, t_batch).sample

            x_t = 1/torch.sqrt(self.alphas[t]) * (x_t - (1 - self.alphas[t])/torch.sqrt(1 - self.alphas_cumprod[t]) * generated_noise ) + torch.sqrt(self.betas[t]) * z

        return ((x_t.clamp(-1, 1) + 1) / 2).cpu()


    def forward(self, batch: torch.Tensor, noise: torch.Tensor):
        """The forward pass receives a batch of size (batch_size, 3, 32, 32) and returns a batch of the same size. The returned batch represents the
        predicted noise for each image in the input batch.

        Args:
            batch (torch.Tensor): Batch of size (B x C x H x W), wich consists in a size B list of 3D matrices representing
            the images

            noise (torch.Tensor): Batch of size (B x C x H x W) corresponding to the noise that will be applied in each
            image
        """
        # Sorting the batch of time steps that each image in the batch will be diffused
        t = torch.randint(0, self.config.diffusion.T, (batch.size(0),), device=self.alphas.device).long()

        # Selecting the noise contribution for each image based on its time step
        alpha_t = self.alphas_cumprod[t]

        # Generating the noised images based on the time steps and the noise passed
        noised_images = torch.sqrt(alpha_t) * batch + torch.sqrt(1 - alpha_t) * noise

        predicted_noise = self.model(noised_images, t).sample  # The model predicts the noise added to the image

        return predicted_noise

    def save(self, path: str|Path, optimizer: Optional[torch.optim.Optimizer] = None, step: int | None = None, trained: bool = False):
        """Saves the unet weights and the diffusion parameters. If you want to restart the training, pass the
        optimizer parameter and the step parameter. If optimizer is provided, but step is not provided, then the
        training will be restarted from the start
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "diffusion": self._diffusion_params(),
            "step": step,
            "trianed": trained
        }

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str|Path, optimizer: Optional[torch.optim.Optimizer] = None) -> tuple[int,bool]:
        checkpoint = torch.load(path, map_location=self.betas.device)

        # Checking if the betas in the checkpoint are the sames that are in the configuration file
        # if its not, then warns that the betas in the CHECKPOINT will be the preferences
        if checkpoint["diffusion"] != self._diffusion_params():
            warnings.warn(
                f"The checkpoint was trained with {checkpoint['diffusion']}, "
                f"but the current config is {self._diffusion_params()}. "
                "Sampling with a different noise schedule will give wrong results."
                "The parameters from the loaded state will be used instead of the ones in the configuration file",
                UserWarning,
                stacklevel=2,
            )

        self.model.load_state_dict(checkpoint["model"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint.get("step"), checkpoint.get("trained")
