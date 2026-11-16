from typing import Any
from abc import ABCMeta


class CommandRegistry(ABCMeta):
    """Class responsible for registering commands in a dictionary for later retrieval.
    """
    registry: dict[str, type] = {}

    def __init__(self, name: str, bases: tuple[type, ...], dict: dict[str, Any], /, **kwds: Any) -> None:
        if name != "Command":
            # Separando o factory para manter apenas o nome do canal como
            # chave do dicionário
            normalized_name = name.replace("Command", "").lower()
            CommandRegistry.registry[normalized_name] = self
        super().__init__(name, bases, dict, **kwds)
