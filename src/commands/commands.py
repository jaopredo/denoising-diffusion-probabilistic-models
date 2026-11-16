from typing import Any
from abc import ABC, abstractmethod
from .registry import CommandRegistry


class Command(ABC, metaclass=CommandRegistry):
    """Abstract base class for all commands in the application.
    """
    
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any):
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
    
    def execute(self, *args: Any, **kwargs: Any):
        # Implement the training logic here
        print("Executing training command with args:", args, "and kwargs:", kwargs)

    @property
    def priority(self) -> int:
        return self.__priority


class GenerateCommand(Command):
    """Concrete command class for generating samples from the model.
    """
    __priority = 2  # Comando de geração deve ser executado após o comando de treino
    
    def execute(self, *args: Any, **kwargs: Any):
        # Implement the generation logic here
        print("Executing generate command with args:", args, "and kwargs:", kwargs)

    @property
    def priority(self) -> int:
        return self.__priority


class EvaluationCommand(Command):
    """Concrete command class for evaluating the model.
    """
    __priority = 3  # Comando de avaliação deve ser executado após o comando de geração
    
    def execute(self, *args: Any, **kwargs: Any):
        # Implement the evaluation logic here
        print("Executing evaluation command with args:", args, "and kwargs:", kwargs)

    @property
    def priority(self) -> int:
        return self.__priority
