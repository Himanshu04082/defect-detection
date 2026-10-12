"""Load project configuration from config/config.yaml."""
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/config.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)