# from typing import Any
from .commands import Command
from .registry import CommandRegistry


GenericCommandListType = list[str]


class CommandManager:
    """Class responsible for managing all commands in the application
    """
    def __init__(self):
        self.commands_to_execute: list[Command] = []

    def add_commands(self, commands: GenericCommandListType):
        """Receives a list of tuples with 3 elements. The first element consists in the
        key name of the command, and the others are the arguments and keyword arguments,
        respectively, to be passed to the command's constructor.

        Args:
            commands (list[tuple[str, tuple, dict]]): List of tuples with the command's key name, arguments, and keyword arguments.
        """
        self.commands_to_execute += self._parse_commands(commands)

    def clear_commands(self):
        self.commands_to_execute.clear()

    def _parse_commands(self, commands: GenericCommandListType) -> list[Command]:
        """Given a list of tuples with the command's key name, arguments, and keyword arguments,
        this method creates instances of the corresponding command classes and returns them as a list.

        Args:
            commands (GenericCommandListType): List of tuples with the command's key name, arguments, and keyword arguments.

        Raises:
            ValueError: Raises a ValueError if a command's key name is not registered in the CommandRegistry.

        Returns:
            list[Command]: List of instances of the corresponding command classes.
        """
        # List to hold the command instances to be returned
        commands_to_return: list[Command] = []

        # Looping through the provided commands to create instances of the corresponding command classes
        for command_name in commands:
            # Raises error if the command is not registered in the CommandRegistry
            if command_name not in CommandRegistry.registry:
                raise ValueError(f"Command '{command_name}' is not registered.")
            commands_to_return.append(CommandRegistry.registry[command_name]())

        # Sorting the commands based on their priority attribute
        commands_to_return.sort(key=lambda cmd: cmd.priority)

        return commands_to_return

    def execute_commands(self):
        """Executes all commands in the commands_to_execute list in order of their priority.
        """
        for command in self.commands_to_execute:
            command.execute()
