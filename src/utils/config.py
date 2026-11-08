from pathlib import Path
from typing import Any

import yaml

DIRECTORY_PATH_KEYS = [
    "data_raw",
    "data_metadata",
    "data_splits",
    "data_processed",
    "checkpoints",
    "outputs_checks",
    "outputs_generated_samples",
    "outputs_evaluations"
]



def get_project_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root



def resolve_directory_paths(raw_path_section: dict[str, str], project_root: Path) -> dict[str, Any]:
    resolved_paths: dict[str, Any] = {}
    
    
    for key, value in raw_path_section.items():
        if key in DIRECTORY_PATH_KEYS:
            resolved_paths[key] = project_root / value
        else:
            resolved_paths[key] = value
            
    return resolved_paths



def default_config_path() -> Path:
    return get_project_root() / "configs" / "config.yaml"



def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = default_config_path()
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
        
    
    project_root = get_project_root()
    
    
    resolved_config: dict[str, Any] = dict(raw_config)
    resolved_config["project_root"] = project_root
    resolved_config["paths"] = resolve_directory_paths(raw_config["paths"], project_root)
    
    metadata_dir = resolved_config["paths"]["data_metadata"]
    csv_filename = raw_config["paths"]["csv_filename"]
    resolved_config["paths"]["csv_path"] = metadata_dir / csv_filename
    
    return resolved_config



def pick(cli_value: Any, config_value: Any) -> Any:
    """Command line value if it was given, otherwise the config value."""
    return config_value if cli_value is None else cli_value



def resolve_path(cli_value: str | Path | None, config_value: Path) -> Path:
    return Path(pick(cli_value, config_value))



if __name__ == "__main__":
    config = load_config()
    
    print(f"Project root: {config["project_root"]}")
    
    print("\nResolved paths:")
    for key, value in config["paths"].items():
        print(f"    {key}: {value}")
        
    print("\nData settings:")
    for key, value in config["data"].items():
        print(f"    {key}: {value}")
        
    print(f"\nSeed: {config["seed"]}")
