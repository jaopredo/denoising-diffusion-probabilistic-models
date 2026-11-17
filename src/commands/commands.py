import torch
from abc import ABC, abstractmethod
from ..models import DiffusionModel
from .registry import CommandRegistry
from ..config import ConfigLoader
from ..data import DataloaderBuilderCIFAR10


class Command(ABC, metaclass=CommandRegistry):
    """Abstract base class for all commands in the application.
    """
    
    @abstractmethod
    def execute(self):
        """Execute the command with the given arguments.
        """
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Return the priority of the command. Lower values indicate higher priority.
        """
        pass


class TrainingCommand(Command):
    """Concrete command class for training the model.
    """
    __priority = 1  # Comando de treino deve ser executado antes de qualquer outro comando

    def __init__(self):
        self.config = ConfigLoader().get()

    def execute(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loading the dataset
        dataloader = self._load_dataset_loader()

        model = DiffusionModel().to(device)  # Replace with actual model initialization

        # Initializing the optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),  # Replace with model parameters
            lr=self.config.training.lr
        )

        # Initializing the criterium
        criterion = torch.nn.MSELoss()  # To take the difference between the noise and the predicted noise

        num_steps = self.config.training.num_steps
        step = num_steps

        while step <= num_steps:
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
                break
            step += 1
        

    def _load_dataset_loader(self):
        dataset_builder = DataloaderBuilderCIFAR10().set_from_config(self.config).set_train_to_true()
        dataset = dataset_builder.build()
        return dataset

    @property
    def priority(self) -> int:
        return self.__priority


class GenerateCommand(Command):
    """Concrete command class for generating samples from the model.
    """
    __priority = 2  # Comando de geração deve ser executado após o comando de treino
    
    def execute(self):
        # Implement the generation logic here
        print("Executing generate command")

    @property
    def priority(self) -> int:
        return self.__priority


class EvaluationCommand(Command):
    """Concrete command class for evaluating the model.
    """
    __priority = 3  # Comando de avaliação deve ser executado após o comando de geração
    
    def execute(self):
        # Implement the evaluation logic here
        print("Executing evaluation command")

    @property
    def priority(self) -> int:
        return self.__priority
