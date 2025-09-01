# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import time

import pandas as pd
from langchain_community.retrievers import WikipediaRetriever

from table_provider.config.entity.azure_openai_config import AzureOpenAIConfig
from table_provider.model.table_provider_enum import TableAugmentationType, TableSerializationType
from table_provider.config.entity.task_token_limit_config import TokenLimitConfig
from table_provider.model.llm.call_llm_client import CallLLMClient
from table_provider.model.metadata import MetadataApi
from table_provider.prompt.augmentation_prompt.augmentation_prompt import TableAugmentationPrompt, TableAugmentationPromptType
from table_provider.service.table_serializer_service import TableDataSerializer


class TableAugmentationService:
    def __init__(
        self,
        azure_openai_config: AzureOpenAIConfig = None,
        token_limit_config: TokenLimitConfig = None
    ):
        self.serializer = TableDataSerializer()
        self.call_llm_client = CallLLMClient(azure_openai_config)
        self.token_limit_config = TokenLimitConfig(token_limit_config)


    def augment(self, parsed_example, table_augmentation_type):
        """
        Run the table augmentation.
        Args:
            parsed_example: Input parsed example which is table data
            table_augmentation_type: The type of table augmentation to run
        Returns:
            the augmented table
        """
        if table_augmentation_type not in [
            augmentation_type.value for augmentation_type in TableAugmentationType
        ]:
            raise ValueError(
                f"Table Augmentation Type {table_augmentation_type} is not supported"
            )

        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, "Table has no rows"
        assert len(parsed_example["table"]["header"]) > 0, "Table has no header"

        # Run the row filter
        if table_augmentation_type == "header_field_categories":
            augmentation_info = self.func_set()["metadata"](
                parsed_example, only_return_categories=True
            )
        else:
            augmentation_info = self.func_set()[table_augmentation_type](
                parsed_example
            )

        if (
            self.call_llm_client.num_tokens(augmentation_info)
            < self.token_limit_config.augmentation_token_limit
        ):
            return "augmentation info for the table:\n" + augmentation_info
        else:
            return (
                "augmentation info for the table:\n"
                + self.call_llm_client.truncated_string(
                    augmentation_info,
                    self.token_limit_config.augmentation_token_limit,
                    print_warning=False,
                )
            )

    @staticmethod
    def _pre_k_rows_of_parsed_example(parsed_example: dict, pre_k: int = 5):
        sampled_table = {
            "title": parsed_example["title"],
            "context": parsed_example["context"],
            "table": {
                "header": parsed_example["table"]["header"],
                "rows": parsed_example["table"]["rows"][:pre_k],
                "caption": parsed_example["table"]["caption"],
            },
        }

        return sampled_table

    @staticmethod
    def get_table_size(parsed_example: dict) -> str:
        """
        Get the table size
        Args:
            parsed_example: the parsed example
        Returns:
            the table size
        """
        return json.dumps(
            {
                "table_size": [
                    len(parsed_example["table"]["header"]),
                    len(parsed_example["table"]["rows"]),
                ]
            },
            indent=4,
            sort_keys=True,
        )

    @staticmethod
    def get_header_hierarchy(parsed_example: dict) -> str:
        """
        Get the header hierarchy
        Args:
            parsed_example: the parsed example
        Returns:
            the header hierarchy
        """
        return json.dumps(
            parsed_example["table"]["header_hierarchy"], indent=4, sort_keys=True
        )

    def get_metadata(
        self, parsed_example: dict, only_return_categories: bool = False
    ) -> str:
        """
        Get the metadata
        Args:
            parsed_example: the parsed example
            only_return_categories: whether to only return the categories
        Returns:
            the metadata
        """
        parsed_example = self._pre_k_rows_of_parsed_example(parsed_example)
        df = pd.DataFrame(
            parsed_example["table"]["rows"], columns=parsed_example["table"]["header"]
        )
        metadata_api = MetadataApi('table_provider/model/metadata/model/model/metadata_tapas_202202_d0e0.pt')
        emb = metadata_api.embedding(df)
        predict = metadata_api.predict(df, emb)
        if only_return_categories:
            return str(predict["Msr_type_res"])
        else:
            return json.dumps(
                {
                    "measure_dimension_type": predict["Msr_res"],
                    "aggregation_type": predict["Agg_score_res"],
                    "measure_type": predict["Msr_type_res"],
                },
                indent=4,
                sort_keys=True,
            )

    @staticmethod
    def get_header_categories(parsed_example: dict) -> str:
        df = pd.DataFrame(
            parsed_example["table"]["rows"], columns=parsed_example["table"]["header"]
        )
        metadata_api = MetadataApi('table_provider/model/metadata/model/model/metadata_tapas_202202_d0e0.pt')
        emb = metadata_api.embedding(df)
        predict = metadata_api.predict(df, emb)
        return str(predict["Msr_type_res"])

    def get_intermediate_NL_reasoning_steps(self, parsed_example: dict) -> str:
        sampled_table = self._pre_k_rows_of_parsed_example(parsed_example)
        sampled_table = sampled_table["table"]

        prompt = TableAugmentationPrompt.get_prompt(
            TableAugmentationPromptType.GET_INTERMEDIATE_NL_REASONING_STEPS,
            sampled_table=sampled_table,
        )

        output = self.call_llm_client.generate_text(prompt)
        if isinstance(output, list):
            return " ".join(output)
        elif isinstance(output, str):
            return output

    def get_trunk_summarization(self, parsed_example: dict) -> str:
        sampled_table = {
            "title": parsed_example["title"],
            "context": parsed_example["context"],
            "table": {
                "header": parsed_example["table"]["header"],
                "rows": parsed_example["table"]["rows"][:5],
                "caption": parsed_example["table"]["caption"],
            },
        }
        serialized_table = self.serializer.retrieve_linear_function(
            TableSerializationType.html, structured_data_dict=sampled_table
        )

        # TODO Why use serialized table?
        prompt = TableAugmentationPrompt.get_prompt(
            TableAugmentationPromptType.GET_TRUNK_SUMMARIZATION,
            serialized_table=serialized_table,
        )

        output = self.call_llm_client.generate_text(prompt)
        if isinstance(output, list):
            return " ".join(output)
        elif isinstance(output, str):
            return output

    def get_term_explanations(self, parsed_example):
        # TODO: refactor pre_k to make it as the basic power to call llm, truncate table when necessary
        sampled_table = self._pre_k_rows_of_parsed_example(parsed_example)
        sampled_table = json.dumps(sampled_table, indent=4)
        prompt = TableAugmentationPrompt.get_prompt(TableAugmentationPromptType.GET_TERM_EXPLANATIONS, sampled_table=sampled_table)

        self.call_llm_client.generate_text(prompt)

        raise NotImplementedError("This get_term_explanations function is not supported yet")

    @staticmethod
    def get_docs_references(parsed_example: dict) -> str:
        """
        Wikipedia is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. Wikipedia is the largest and most-read reference work in history.
        This function is used to retrieve wiki pages from wikipedia,org
        """
        retriever = WikipediaRetriever(lang="en", load_max_docs=2)
        if parsed_example["title"] != "":
            retriever_input = parsed_example["title"]
        elif parsed_example["context"] != "" and len(parsed_example["context"]) != 0:
            if isinstance(parsed_example["context"], list):
                retriever_input = " ".join(parsed_example["context"])
            else:
                retriever_input = parsed_example["context"]
        else:
            print(
                f"No title or context found, use header instead: {parsed_example['table']['header']}"
            )
            retriever_input = " ".join(parsed_example["table"]["header"])

        docs = retriever.get_relevant_documents(retriever_input)
        time.sleep(15)
        if len(docs) == 0:
            raise ValueError("No Wikipedia Search Documents were found, empty augmentation info")
        return json.dumps(docs[0].metadata, indent=4)

    def assemble_neural_symbolic_augmentation(
        self, parsed_example: dict
    ):
        parsed_example = self._pre_k_rows_of_parsed_example(parsed_example)
        return "\n".join(
            [
                self.get_table_size(parsed_example),
                self.get_header_hierarchy(parsed_example),
                self.get_metadata(parsed_example),
                self.get_intermediate_NL_reasoning_steps(parsed_example),
            ]
        )

    def assemble_retrieval_based_augmentation(
        self, parsed_example: dict
    ):
        parsed_example = self._pre_k_rows_of_parsed_example(parsed_example)
        return "\n".join(
            [
                self.get_term_explanations(parsed_example),
                self.get_docs_references(parsed_example),
            ]
        )

    def columns_analysis(self, parsed_example: dict):
        parsed_example = self._pre_k_rows_of_parsed_example(parsed_example)

        system_prompt = TableAugmentationPrompt.get_system_prompt(
            TableAugmentationPromptType.GET_COLUMNS_ANALYSIS
        )

        sampled_table = {
            "title": parsed_example["title"],
            "context": parsed_example["context"],
            "table": {
                "header": parsed_example["table"]["header"],
                "rows": parsed_example["table"]["rows"][:5],
                "caption": parsed_example["table"]["caption"],
            },
        }
        serialized_table = self.serializer.retrieve_linear_function(
            TableSerializationType.html, structured_data_dict=sampled_table
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Table: \n{serialized_table}"""},
        ]

        response = self.call_llm_client.generate_text_with_messages(messages)
        if "```JSON" in response:
            analysis_result = response.split("```JSON")[1].split("```")[0] if "```" in response else response.strip(".")
        elif "```json" in response:
            analysis_result = response.split("```json")[1].split("```")[0] if "```" in response else response.strip(".")
        else:
            analysis_result = response

        return analysis_result


    def func_set(self) -> dict:
        return {
            TableAugmentationType.table_size.value: self.get_table_size,
            TableAugmentationType.header_hierarchy.value: self.get_header_hierarchy,
            TableAugmentationType.metadata.value: self.get_metadata,
            TableAugmentationType.intermediate_NL_reasoning_steps.value: self.get_intermediate_NL_reasoning_steps,
            TableAugmentationType.trunk_summary.value: self.get_trunk_summarization,
            TableAugmentationType.term_explanations.value: self.get_term_explanations,
            TableAugmentationType.docs_references.value: self.get_docs_references,
            TableAugmentationType.columns_analysis.value: self.columns_analysis,
            TableAugmentationType.neural_symbolic_augmentation.value: self.assemble_neural_symbolic_augmentation,
            TableAugmentationType.retrieval_based_augmentation.value: self.assemble_retrieval_based_augmentation,
        }
