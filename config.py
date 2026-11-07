"""Shared configuration loading without importing the ML dependencies."""

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.toml")


def load_config(path=DEFAULT_CONFIG):
    """Load TOML and resolve configured filesystem paths beside the config file."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)

    def resolve(value):
        return str((path.parent / Path(value).expanduser()).resolve())

    config["paths"] = {key: resolve(value) for key, value in config["paths"].items()}
    config["inference"]["adapter_path"] = resolve(config["inference"]["adapter_path"])
    if config["dataset"]["path"] == "json":
        files = config["dataset"]["data_files"]
        config["dataset"]["data_files"] = (
            [resolve(value) for value in files] if isinstance(files, list) else resolve(files)
        )
    return config


def config_from_cli():
    parser = argparse.ArgumentParser(description="Run with shared project configuration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a TOML config file")
    return load_config(parser.parse_args().config)
