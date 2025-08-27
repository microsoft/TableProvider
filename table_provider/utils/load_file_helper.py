# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import yaml
from pathlib import Path

def load_args_from_yaml(yaml_file_path):
    """
    Load configuration from a YAML file.

    Parameters:
    - yaml_file_path: Path to the YAML file.

    Returns:
    - config: A dictionary containing the configuration.
    """
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_llm_config(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "model": {
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "GPT_MODEL": "text-davinci-003",
            },
            "api_key": "",
            "batch_size": 16,
            "total_tokens": 4000,
            "max_truncate_tokens": 1400,
            "example_token_limit": {
                "tabfact": 627,
                "hybridqa": 1238,
                "sqa": 1439,
                "totto": 889,
                "feverous": 1261,
            },
            "augmentation_token_limit": 1000,
            "max_rows": 50,
            "max_columns": 10,
        }

    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
