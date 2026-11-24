import argparse
import os
from pathlib import Path
from src.commands import CommandRegistry, CommandManager
from src.config import ConfigLoader
from loguru import logger


if __name__ == "__main__":
    logger.info("Program initialized")
    argument_parser = argparse.ArgumentParser(description="Denoising Diffusion Probabilistic Models")

    command_manager = CommandManager(argument_parser)
    command_manager.configure_commands()
    logger.success("Commands configured")

    args = argument_parser.parse_args()
    args_dict = vars(args)

    # Load configuration
    root = Path(os.getcwd())
    config_loader = ConfigLoader()
    config_loader.load(root / "config" / "project.yaml")
    config = config_loader.get()

    logger.info("Starting passed command")
    command_manager.run(**args_dict)
