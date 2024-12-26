import yaml
from typing import Dict


def load_config(config_file: str) -> Dict:
    with open(config_file, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config