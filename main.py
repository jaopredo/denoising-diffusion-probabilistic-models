import argparse
import os
from pathlib import Path
from src.commands import CommandRegistry, CommandManager
from src.config import ConfigLoader


class Program:
    def __init__(self, name: str, parser: argparse.ArgumentParser, command_manager: CommandManager):
        self.name = name
        self.parser = parser
        self.command_manager = command_manager

        self.configure_arguments()

    def configure_arguments(self):
        for key in CommandRegistry.registry.keys():
            self.parser.add_argument(f'--{key}', action='store_true', help=f'Execute the {key} command')

    def retrieve_arguments(self):
        args = self.parser.parse_args()
        return args

    def run(self):
        print(f"Running {self.name}...")
        args = self.retrieve_arguments()
        self.command_manager.add_commands([
            key for key, value in vars(args).items() if value
        ])
        self.command_manager.execute_commands()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Denoising Diffusion Probabilistic Models")
    command_manager = CommandManager()

    # Load configuration
    root = Path(os.getcwd())
    config_loader = ConfigLoader()
    config_loader.load(root / "config" / "project.yaml")
    config = config_loader.get()

    # Run the program with the loaded configuration
    program = Program(config.experiment.name, argument_parser, command_manager)
    program.run()
