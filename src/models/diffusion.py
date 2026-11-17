import torch
from typing import Any
from diffusers import UNet2DModel
from ..config import ConfigLoader


class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Loading the configuration
        self.config = ConfigLoader().get()

        # Loading the device
        self.__device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.betas = torch.linspace(self.config.diffusion.beta_start, self.config.diffusion.beta_end, self.config.diffusion.T).to(self.__device)

        # Transforming the betas into alphas for computational efficiency
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).view(-1, 1, 1, 1)
        # Reshaped to (T, 1, 1, 1) for broadcasting during the forward pass

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)
        self.__device = args[0] if args else kwargs.get("device", self.__device)
        return self

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
        t = torch.randint(0, self.config.diffusion.T, (batch.size(0),), device=self.__device).long()

        # Selecting the noise contribution for each image based on its time step
        alpha_t = self.alphas_cumprod[t]

        # Generating the noised images based on the time steps and the noise passed
        noised_images = torch.sqrt(alpha_t) * batch + torch.sqrt(1 - alpha_t) * noise

        predicted_noise = self.model(noised_images, t).sample  # The model predicts the noise added to the image

        return predicted_noise
