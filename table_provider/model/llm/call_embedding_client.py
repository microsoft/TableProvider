# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from typing import List

from table_provider import logger
from table_provider.config.entity.azure_openai_config import AzureOpenAIConfig, AzureOpenAIClientConfig
from table_provider.model.llm.call_llm_client import DEFAULT_CALL_LLM_CONFIG_PATH


class CallEmbeddingClient:
    def __init__(self, config: AzureOpenAIConfig = None):
        if config is None:
            with open(DEFAULT_CALL_LLM_CONFIG_PATH) as config_file:
                json_config = json.load(config_file)
            self.config = AzureOpenAIConfig(json_config["call_embedding_config"]["endpoint"],
                                       json_config["call_embedding_config"]["deployment_name"],
                                       json_config["call_embedding_config"]["model_name"])
            self.model_name = json_config["call_embedding_config"]["model_name"]
        else:
            self.config = config
            self.model_name = config.model_name
        self.client_config = AzureOpenAIClientConfig(self.config)
        self.client = self.client_config.client
        self.embedding_type = self.config.model_name
        self.logger = logger

    def call_embeddings(
        self,
        user_query: str,
        row_column_list: List[str],
    ):
        logger.info(f"Embedding type: {self.embedding_type}")
        if self.embedding_type != self.model_name:
            return None, None

        # generate column embeddings
        # if row_column_list's len > 2048, get embeddings in a loop
        value_list_embeddings_list = []
        for i in range(0, len(row_column_list), 2048):
            value_list_embeddings = self.client.embeddings.create(
                model=self.config.deployment_name,
                input=row_column_list[i : i + 2048],
            ).data
            value_list_embeddings_list.extend(value_list_embeddings)

        pure_embeddings = []
        for value_list_embedding in value_list_embeddings_list:
            pure_embeddings.append(value_list_embedding.embedding)
        value_list_embeddings = pure_embeddings

        if user_query != "":
            user_query_embedding = self.client.embeddings.create(
                model=self.config.deployment_name,
                input=[user_query],
            ).data[0].embedding
        else:
            user_query_embedding = None

        return value_list_embeddings, user_query_embedding