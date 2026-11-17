from pathlib import Path
import os
from src.config.loader import ConfigLoader

path = Path(os.getcwd()) / "config" / "project.yaml"


loader1 = ConfigLoader()
loader1.load(path)

loader2 = ConfigLoader()
print(loader2.get())  # Should print the root directory from the config
