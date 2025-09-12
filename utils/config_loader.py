from pathlib import Path
import yaml

def load_config(file_path=None):
    """
    Load configuration from a YAML file.

    :param file_path: Path to the YAML configuration file.
    :return: Parsed configuration as a dictionary.
    """
    if file_path is None:
        # Automatically resolve path relative to project root
        file_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        print(f"Configuration loaded from {file_path}: {config}")
    return config


if __name__ == "__main__":
    config = load_config()
    print(config)