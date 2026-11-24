import os
from pathlib import Path
from ..config import ConfigLoader


def get_outputs_paths():
	"""Returns an dictionary containing the formated paths for each important folder of outputs
	"""
	config = ConfigLoader().get()

	root = Path(os.getcwd())

	output_path = root / config.experiment.output_dir

	paths = {
        "samples": output_path / config.experiment.samples_dir_name,
        "checkpoints": output_path / config.experiment.checkpoints_dir_name,
        "loss": output_path / config.experiment.loss_dir_name
    }

    # Checking if each directory exists and, if not, create them
	for path in paths.values():
		path.mkdir(parents=True, exist_ok=True)

	return paths
