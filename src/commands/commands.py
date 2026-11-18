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
    _arguments = {}

    def __init__(self):
        self.config = ConfigLoader().get()

    def execute(self, *args: Any, **kwargs: Any):
        print(f"Executing training command with arguments: args {args}, kwargs {kwargs}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg_train = self.config.training

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
        step = 0
        trained = False

        # Checking if there isn't a checkpoint that didn't finish the training
        path = Path(cfg_train.checkpoint_path) / cfg_train.checkpoint_name
        if path.exists():
            step, trained = model.load(path, optimizer)
            print(f"Loading train from step {step}")


        while step <= num_steps and not trained:
            print(f"Training step {step}/{num_steps}...")

            # Saving checkpoint for possible training interruption
            if step % cfg_train.checkpoint_every == 0:
                print(f"Saving checkpoint at step {step}...")
                model.save(path, optimizer, step, trained)

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
                optimizer.step()

            if step > num_steps:
                trained = True
                break
            step += 1

        print("Train finished")
        

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
        "test_argument": {
            "type": str,
            "required": True
        }
    }

    def __init__(self):
        self.config = ConfigLoader().get()
    
    def execute(self, *args: Any, **kwargs: Any):
        print(f"Executing generate command with aarguments: args {args}, kwargs {kwargs}")

        cfg_train = self.config.training

        model = DiffusionModel()

        # Implement the generation logic here
        checkpoint_path = Path(cfg_train.checkpoint_path) / cfg_train.checkpoint_name
        if checkpoint_path.exists():
            _ = model.load(checkpoint_path)

        image_batch = model.generate()

        # Folder where the images will be saved (created if it does not exist)
        output_dir = Path(self.config.experiment.output_dir) / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        # The timestamp avoids overwriting images from previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # One file per image
        for i, img in enumerate(image_batch):
            save_image(img, output_dir / f"{timestamp}_{i:03d}.png")

        # One grid with all the images together
        save_image(image_batch, output_dir / f"{timestamp}_grid.png", nrow=5)

    @classmethod
    def get_arguments(cls) -> dict[str, Any]:
        return cls._arguments


class EvaluationCommand(Command):
    """Concrete command class for evaluating the model.
    """

    _arguments = {
        "test_argument": {
            "type": str,
            "required": True
        }
    }
    
    def execute(self, *args: Any, **kwargs: Any):
        # Implement the evaluation logic here
        print(f"Executing evaluation command with arguments: args {args}, kwargs {kwargs}")

    @classmethod
    def get_arguments(cls) -> dict[str, Any]:
        return cls._arguments
