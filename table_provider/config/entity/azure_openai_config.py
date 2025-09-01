# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from openai import AzureOpenAI

config_directory = os.path.dirname(os.path.dirname(__file__))
DEFAULT_CALL_LLM_CONFIG_PATH = f"{config_directory}/default_azure_openai_config.json"

class AzureOpenAIConfig:
    def __init__(self):
        self.endpoint = None
        self.deployment_name = None
        self.model_name = None

    def __init__(self, endpoint, deployment_name, model_name):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.model_name = model_name


class AzureOpenAIClientConfig:
    def __init__(self, config: AzureOpenAIConfig):
        self.token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        if not config.endpoint or not config.deployment_name:
            raise ValueError("AzureOpenAIConfig is not properly initialized, endpoint or deployment_name is missing")

        self.client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            azure_ad_token_provider=self.token_provider,
            api_version="2024-05-01-preview",
        )

    def get_client(self):
        return self.client

