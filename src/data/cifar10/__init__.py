import torchvision
import torch
from typing import Optional, cast
from pathlib import Path
from src.config.schema import Config
from torch.utils.data import DataLoader, Dataset


class DataloaderBuilderCIFAR10:
    def __init__(self):
        self.root: Optional[str|Path] = None
        self.train: bool = False
        self.download: Optional[bool] = None
        self.batch_size: Optional[int] = None
        self.shuffle: bool = False
        self.transform = torchvision.transforms.ToTensor()

    def set_root(self, root: str|Path):
        self.root = root
        return self

    def set_train_to_true(self):
        self.train = True
        return self

    def set_download(self, download: bool):
        self.download = download
        return self

    def set_from_config(self, config: Config):
        self.root = config.data.root
        self.download = config.data.download
        self.batch_size = config.training.batch_size
        self.shuffle = config.training.shuffle
        return self

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        return self

    def set_shuffle_to_true(self):
        self.shuffle = True
        return self

    def build(self):
        if self.root is None:
            raise ValueError("Root directory must be set before building the dataset.")
        if self.download is None:
            raise ValueError("Download flag must be set before building the dataset.")
        if self.batch_size is None:
            raise ValueError("Batch size must be set before building the dataset.")

        data = cast(
            Dataset[tuple[torch.Tensor, int]],
            torchvision.datasets.CIFAR10(
                root=self.root,
                train=self.train,
                download=self.download,
                transform=self.transform
            )
        )

        loader = DataLoader(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            dataset=data
        )

        return loader
