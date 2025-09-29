from pathlib import Path
import yaml


def load_config(path: str) -> dict:
	cfg_path = Path(path)
	with cfg_path.open("r") as f:
		cfg = yaml.safe_load(f)
	return cfg

