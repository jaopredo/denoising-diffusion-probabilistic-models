import argparse
from src.commands import CommandRegistry, CommandManager


class Program:
    def __init__(self, parser: argparse.ArgumentParser, command_manager: CommandManager):
        self.name = "Denoising Diffusion Probabilistic Models"
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

    program = Program(argument_parser, command_manager)
    program.run()
