# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datasets import load_dataset
import json

from table_provider.config.entity.azure_openai_config import AzureOpenAIConfig
from table_provider.model.table_provider_enum import TaskName, TableSerializationType
from table_provider.config.entity.task_token_limit_config import TokenLimitConfig
from table_provider.model.llm.call_llm_client import CallLLMClient
from table_provider.service.table_serializer_service import TableDataSerializer


class TableLoaderService:
    def __init__(self, task_name: str,
                 split: str = "None",
                 use_small_sample_list: bool = False,
                 azure_openai_config: AzureOpenAIConfig = None,
                 token_limit_config: TokenLimitConfig = None):
        """
        Load table from dataset
        Args:
            task_name (str): valid task name should be selected from ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
            split (str): train, validation, or test
        """
        if task_name not in [task.value for task in TaskName]:
            raise ValueError(f"Task name {task_name} is not supported")

        self.task_name = task_name
        self.split = split
        self.call_llm_client = CallLLMClient(azure_openai_config)
        self.dataset = self.load_table(split, use_small_sample_list)
        self.token_limit_config = TokenLimitConfig(token_limit_config)

    def load_table(self, split: str, use_small_sample_list: bool):
        # Load table from dataset
        if self.task_name == TaskName.json_convert.value:
            return None

        dataset = load_dataset(
            f"table_provider/task/script/{self.task_name}.py",
            split=split if split != "None" else None,
            verification_mode="no_checks",
        )
        if use_small_sample_list and len(dataset) >= 1000:
            shuffled_dataset = dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(1000))
        else:
            return dataset

    def parse_task_table(self, _example: dict) -> dict:
        # Parse table to the format of each task
        if self.task_name == TaskName.feverous.value:
            label = "2" if str(_example["label"]) == "NOT ENOUGH INFO" else "0" if str(
                _example["label"]) == "REFUTES" else "1"
            return {
                "title": "",
                "context": _example["context"],
                "table": {
                    "header": _example['table']['header'][0],
                    "rows": _example['table']['rows'][0],
                    "caption": "",
                },
                "query": _example["statement"],
                "label": label,
            }
        elif self.task_name == TaskName.hybridqa.value:
            return {
                "title": "",
                "context": [_example["context"], _example["passage"]],
                "table": {
                    "header": _example['table']['header'],
                    "rows": _example['table']['rows'],
                    "caption": "",
                },
                "query": _example["question"],
                "label": _example["answer_text"],
            }
        elif self.task_name == TaskName.tabfact.value:
            label = str(_example["label"]) if str(_example["label"]) in ["0", "1"] else "2"
            return {
                "title": "",
                "context": "",
                "table": {
                    "header": _example['table']['header'],
                    "rows": _example['table']['rows'],
                    "caption": _example['table']['caption'],
                },
                "query": _example["statement"],
                "label": label,
            }
        elif self.task_name == TaskName.totto.value:
            return {
                "title": _example['table_page_title'],
                "context": "",
                "table": {
                    "header": _example['table_rows'][0],
                    "rows": _example['table_rows'][1:],
                    "caption": _example['table_section_title'],
                    "header_hierarchy": _example['table_header_hierarchy'],
                },
                "query": f"Produce a one-sentence description for each highlighted cells ({str(_example['highlighted_cells'])}) of the table.",
                "label": _example["final_sentences"],
            }
        elif self.task_name == TaskName.json_convert.value:
            return {
                "title": "",
                "context": "",
                "table": {
                    "header": _example['header'],
                    "rows": _example['rows'],
                    "caption": "",
                },
                "query": _example["query"],
                "label": _example["label"],
            }
        else:
            raise ValueError(f"Task name {self.task_name} is not supported")

    @staticmethod
    def parse_input_table(table_name, table_header, table_data, user_query):
        parsed_tables = []
        if isinstance(table_name, str):
            if isinstance(user_query, list):
                user_query = " ".join(user_query)
            parsed_table = {
                "title": table_name,
                "context": "",
                "table": {
                    "header": table_header,
                    "rows": table_data,
                    "caption": "",
                },
                "query": user_query,
            }
            parsed_tables.append(parsed_table)
        else:
            if user_query is None:
                user_query = ["" for _ in range(len(table_name))]
            elif isinstance(user_query, str):
                user_query = [user_query for _ in range(len(table_name))]

            if len(table_name) != len(table_header) or len(table_name) != len(table_data) or len(table_name) != len(user_query):
                raise ValueError("The length of table name, table header, and table data should be the same")
            for (single_table_name, single_table_header, single_table_data, single_user_query) \
                    in zip(table_name, table_header, table_data, user_query):
                parsed_table = {
                    "title": single_table_name,
                    "context": "",
                    "table": {
                        "header": single_table_header,
                        "rows": single_table_data,
                        "caption": "",
                    },
                    "query": single_user_query,
                }
                parsed_tables.append(parsed_table)

        return parsed_tables

    @staticmethod
    def serialization(_example: dict, serialization_function=TableSerializationType.html):
        # Linearize table
        table_data_serializer = TableDataSerializer()
        return table_data_serializer.retrieve_linear_function(
            serialization_function, structured_data_dict=_example
        )

    def get_k_shot_examples(self, task_name: str, k: int, linearization_function=TableSerializationType.html):
        # Get K shot examples
        k_shot_examples = []
        for i in range(len(self.dataset)):
            shot_info = self.parse_task_table(self.dataset[i])
            shot_example = "\n".join([
                "Example table is:",
                self.serialization(shot_info, serialization_function=linearization_function),
                "Example query of the following table is: ",
                shot_info["query"],
                "Example answer is: ",
                "|".join(shot_info["label"]) if isinstance(shot_info["label"], list) else shot_info["label"],
            ])
            k_shot_token_limit = self.token_limit_config.get_task_token_limit(task_name)
            if self.call_llm_client.num_tokens(shot_example) < k_shot_token_limit:
                k_shot_examples.append(shot_example)
                if len(k_shot_examples) == k:
                    break

            if i == len(self.dataset) - 1 and len(k_shot_examples) < k:
                print(f"Warning: There is no example with less than {k_shot_token_limit} tokens.")
                k_shot_examples.append(
                    self.call_llm_client.truncated_string(shot_example, k_shot_token_limit)
                )

        return k_shot_examples

    @staticmethod
    def load_local_table(task_name):
        with open(f"table_provider/task/dataset/{task_name}.jsonl", "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def dynamic_load_task_table(self, task_name, load_local_dataset):
        if load_local_dataset:
            tables = self.load_local_table(task_name)
        else:
            parsed_tables = []
            for table in self.dataset:
                parsed_table = self.parse_task_table(table)
                parsed_tables.append(parsed_table)
            tables = parsed_tables

        return tables
