"""Configuration loader and dataset configuration parser."""
import os
import json
from typing import Tuple

def load_config(dataset_name: str) -> Tuple[str, dict]:
    """
    Loads the dataset configuration based on the dataset name.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'ImageCHD_dataset')
        
    Returns:
        Tuple[str, dict]: (dataset_dir, config_dict)
    """
    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, "datasets", dataset_name)
    config_path = os.path.join(dataset_dir, f"{dataset_name}.json")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Label config not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    return dataset_dir, config

