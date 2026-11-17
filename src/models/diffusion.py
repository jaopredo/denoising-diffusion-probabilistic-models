import torch
from typing import Any
from diffusers import UNet2DModel
from ..config import ConfigLoader


class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ConfigLoader().get()
        self.__device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet2DModel(
            sample_size=32,  # CIFAR-10 images are 32x32
            in_channels=3,   # RGB images
            out_channels=3,  # Predicting noise for RGB images
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)
        self.__device = args[0] if args else kwargs.get("device", self.__device)
        return self

    def forward(self, batch: torch.Tensor):
        """The forward pass receives a batch of size (batch_size, 3, 32, 32) and returns a batch of the same size. The returned batch represents the
        predicted noise for each image in the input batch.

        Args:
            batch (_type_): 
        """

        # Sorting the batch of time steps that each image in the batch will be diffused
        t = torch.randint(0, self.config.diffusion.T, (batch.size(0),), device=self.__device).long()
