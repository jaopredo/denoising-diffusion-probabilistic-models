from typing import Any
from argparse import ArgumentParser
from .commands import Command
from .registry import CommandRegistry


class CommandManager:
    """Class responsible for managing all commands in the application
    """
    def __init__(self, parser: ArgumentParser):
        self.parser = parser

    def configure_commands(self):
        subparsers = self.parser.add_subparsers(dest="command", required=True)

        registry = CommandRegistry.registry

        for command_name, command in registry.items():
            command_parser = subparsers.add_parser(command_name)

            command = registry[command_name]
            for argument, kwargs in command.get_arguments().items():
                command_parser.add_argument(f"--{argument}", **kwargs)

    def run(self, **kwargs: Any):
        registry = CommandRegistry.registry

        command_name = kwargs["command"]

        arguments_only = kwargs.copy()
        del arguments_only["command"]

        registry[command_name]().execute(**arguments_only)
