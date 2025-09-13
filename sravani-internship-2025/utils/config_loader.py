import yaml
import os



def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_path_registry(path="config/paths.yaml"):
    return load_yaml_config(path)
