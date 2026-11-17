import yaml
from pathlib import Path
from typing import Optional
from ..core import SingletonMeta
from .schema import *


class ConfigLoader(metaclass=SingletonMeta):
    def __init__(self):
        self.config: Optional[Config] = None

    def load(self, config_path: str|Path) -> None:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
            self.config = Config(
                experiment=Experiment(**config_dict["experiment"]),
                data=Data(**config_dict["data"]),
                diffusion=Diffusion(**config_dict["diffusion"]),
                model=Model(**config_dict["model"]),
                training=Training(**config_dict["training"]),
                logging=Logging(**config_dict["logging"]),
            )

    def get(self) -> Config:
        if self.config is None:
            raise ValueError("Configuration has not been loaded yet.")
        return self.config
