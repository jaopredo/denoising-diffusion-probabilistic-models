import yaml
from typing import Optional
from core import SingletonMeta
from .schema import Config


class ConfigLoader(metaclass=SingletonMeta):
    def __init__(self):
        self.config: Optional[Config] = None

    def load(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
            self.config = Config(
                experiment=config_dict["experiment"],
                data=config_dict["data"],
                diffusion=config_dict["diffusion"],
                model=config_dict["model"],
                training=config_dict["training"],
                logging=config_dict["logging"],
                sampling=config_dict["sampling"],
            )

    def get(self) -> Config:
        if self.config is None:
            raise ValueError("Configuration has not been loaded yet.")
        return self.config
