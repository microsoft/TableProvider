# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os

config_directory = os.path.dirname(os.path.dirname(__file__))
DEFAULT_LIMIT_CONFIG_PATH = f"{config_directory}/default_limit_config.json"

class TokenLimitConfig:
    def __init__(self, config_map = None):
        with open(DEFAULT_LIMIT_CONFIG_PATH) as config_file:
            default_config = json.load(config_file)
        config = config_map

        if not config:
            self.task_example_token_limit = default_config["task_example_token_limit"]
            self.max_truncate_tokens = default_config["max_truncate_tokens"]
            self.augmentation_token_limit = default_config["augmentation_token_limit"]
            self.max_rows = default_config["max_rows"]
            self.max_columns = default_config["max_columns"]
        else:
            self.task_example_token_limit = default_config["task_example_token_limit"] if not config.get("task_example_token_limit") else config["task_example_token_limit"]
            self.max_truncate_tokens = default_config["max_truncate_tokens"] if not config.get("max_truncate_tokens") else config["max_truncate_tokens"]
            self.augmentation_token_limit = default_config["augmentation_token_limit"] if not config.get("augmentation_token_limit") else config["augmentation_token_limit"]
            self.max_rows = default_config["max_rows"] if not config.get("max_rows") else config["max_rows"]
            self.max_columns = default_config["max_columns"] if not config.get("max_columns") else config["max_columns"]


    def get_task_token_limit(self, task_name):
        if self.task_example_token_limit.get(task_name):
            return self.task_example_token_limit[task_name]
        else:
            raise ValueError(f"Task name {task_name} does not have token limit")