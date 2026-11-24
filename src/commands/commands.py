import csv

from os.path import exists
from loguru import logger

from torch._prims_common import check

from ..config.schema import Config
import torch
from pathlib import Path
from typing import Any
from datetime import datetime
from torchvision.utils import save_image
from abc import ABC, abstractmethod
from ..models import DiffusionModel
from .registry import CommandRegistry
from ..config import ConfigLoader
from ..data import DataloaderBuilderCIFAR10
from ..utils import get_outputs_paths


class Command(ABC, metaclass=CommandRegistry):
    """Abstract base class for all commands in the application.
    """
    
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any):
        """Execute the command with the given arguments.
        """
        pass

    @classmethod
    @abstractmethod
    def get_arguments(cls) -> dict[str, Any]:
        """Returns the argumnts dict structure passed to the argparser
        """
        pass


class TrainingCommand(Command):
    """Concrete command class for training the model.
    """
    _arguments = {
        "checkpoint": {
            "type": str,
            "required": False
        }
    }

    def __init__(self):
        self.config = ConfigLoader().get()

        self.paths = get_outputs_paths()

    def execute(self, *args: Any, **kwargs: Any):
        logger.info(f"Executing training command with arguments: args {args}, kwargs {kwargs}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg_train = self.config.training
        cfg_logging = self.config.logging

        # Loading the dataset
        dataloader = self._load_dataset_loader()

        model = DiffusionModel().to(device)  # Replace with actual model initialization

        # Initializing the optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),  # Replace with model parameters
            lr=cfg_train.lr
        )

        # Initializing the criterium
        criterion = torch.nn.MSELoss()  # To take the difference between the noise and the predicted noise

        num_steps = cfg_train.num_steps
        step = 1
        trained = False

        # Getting the last checkpoint if the user passes in the kwargs
        if kwargs.get("checkpoint"):
            checkpoint_name = kwargs["checkpoint"]
            checkpoint_path = self.paths["checkpoints"] / checkpoint_name

            if checkpoint_path.exists():
                step, trained = model.load(checkpoint_path, optimizer)
                logger.info(f"Loading train from step {step}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{self.config.experiment.name}-{timestamp}"
            checkpoint_path = self.paths["checkpoints"] / checkpoint_name


        loss_history: list[float] = []


        while step <= num_steps and not trained:
            # Saving checkpoint for possible training interruption
            if step % cfg_train.checkpoint_every == 0:
                logger.info(f"Saving checkpoint at step {step}...")
                model.save(checkpoint_path, optimizer, step, trained)

            if step % cfg_logging.show_steps_every == 0:
                logger.info(f"Step {step}/{num_steps}")

            for x, _ in dataloader:
                # Training logic here
                model.train()

                # Zeros the gradients
                optimizer.zero_grad()

                x = x.to(device)

                noise = torch.randn_like(x)

                epsilon = model.forward(x, noise)  # Forward pass, the model predicts the noise added to the image

                loss = criterion(epsilon, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg_train.grad_clip)
                loss_history.append(loss.item())
                optimizer.step()

            if step % cfg_logging.log_every == 0:
                logger.info(f"Registered Loss: {loss_history[-1]}")
                self._save_loss_history(loss_history, self.paths["loss"])

            if step % cfg_logging.sample_every == 0:
                logger.info(f"Generating new samples on the actual weights")
                image_batch = model.generate(cfg_logging.num_samples)

                # One grid with all the images together
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_image(image_batch, self.paths["samples"] / f"{timestamp}_grid.png", nrow=cfg_logging.num_samples)


            if step > num_steps:
                trained = True
                break
            step += 1
    
    def _save_loss_history(self, loss_history: list[float], path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss_file_name = f"loss-{timestamp}.csv"

        with open(path / loss_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "loss"])

            for epoch, loss in enumerate(loss_history, start=1):
                writer.writerow([epoch, loss])


    def _load_dataset_loader(self):
        dataset_builder = DataloaderBuilderCIFAR10().set_from_config(self.config).set_train_to_true()
        dataset = dataset_builder.build()
        return dataset

    @classmethod
    def get_arguments(cls) -> dict[str, Any]:
        return cls._arguments


class GenerateCommand(Command):
    """Concrete command class for generating samples from the model.
    """

    _arguments = {
        "checkpoint": {
            "type": str,
            "required": True
        },

        "amount": {
            "type": int,
            "required": False,
            "default": 5
        }
    }

    def __init__(self):
        self.config = ConfigLoader().get()

        self.paths = get_outputs_paths()
    
    def execute(self, *args: Any, **kwargs: Any):
        logger.info(f"Executing generate command with arguments: args {args}, kwargs {kwargs}")

        cfg_train = self.config.training

        model = DiffusionModel()

        # Implement the generation logic here
        

        checkpoint_name = kwargs["checkpoint"]
        checkpoint_path = self.paths["checkpoints"] / checkpoint_name

        _ = model.load(checkpoint_path)

        image_batch_size = kwargs["amount"]
        image_batch = model.generate(image_batch_size)

        # Folder where the images will be saved (created if it does not exist)
        output_dir = Path(self.config.experiment.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        # The timestamp avoids overwriting images from previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # One file per image
        for i, img in enumerate(image_batch):
            save_image(img, output_dir / f"{timestamp}_{i:03d}.png")

        # One grid with all the images together
        save_image(image_batch, output_dir / f"{timestamp}_grid.png", nrow=image_batch_size)

        logger.success(f"Diffusion Model generated all {image_batch_size} images successfully")
        logger.success(f"Images stored in path {output_dir.resolve()}")

    @classmethod
    def get_arguments(cls) -> dict[str, Any]:
        return cls._arguments


class EvaluationCommand(Command):
    """Concrete command class for evaluating the model.
    """

    _arguments = {}
    
    def execute(self, *args: Any, **kwargs: Any):
        # Implement the evaluation logic here
        print(f"Executing evaluation command with arguments: args {args}, kwargs {kwargs}")

    @classmethod
    def get_arguments(cls) -> dict[str, Any]:
        return cls._arguments
