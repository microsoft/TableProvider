# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tqdm import tqdm

from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse
from table_provider.orchestrator.extension_base import ExtensionBase
from table_provider.service.table_sampling_service import TableSamplingService


class TableSamplingExtension(ExtensionBase):
    def __init__(self):
        super().__init__()

    def is_required(self) -> bool:
        return True

    def should_execute(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        if request.enable_sampling:
            return True

        return False

    @staticmethod
    def _save_sampled_table_info(context: TableProviderContext, table_sampling_type, sampled_table_info):
        sampled_table_info_list = context.sampled_table_info_map.get(table_sampling_type, [])
        sampled_table_info_list.append(sampled_table_info)
        context.sampled_table_info_map[table_sampling_type] = sampled_table_info_list


    def _execute_internal(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        tables = context.tables
        input_table_sampling_type = request.table_sampling.sampling_type

        table_sampling_types = []
        if isinstance(input_table_sampling_type, str):
            table_sampling_types.append(input_table_sampling_type)
        elif isinstance(input_table_sampling_type, list):
            table_sampling_types.extend(input_table_sampling_type)

        table_samping_service = TableSamplingService(
            split = request.table_sampling.split,
            n_cluster = request.table_sampling.n_cluster,
            top_k = request.table_sampling.top_k,
            whether_column_grounding = request.table_sampling.whether_column_grounding,
            llm_azure_openai_config = request.llm_azure_openai_config,
            embedding_azure_openai_config = request.embedding_azure_openai_config,
            token_limit_config = request.table_sampling.token_limit_config
        )

        for i in tqdm(range(len(tables)), desc="Table sampling..."):
            table = tables[i]
            for table_sampling_type in table_sampling_types:
                query = table["query"] \
                    if table["query"] and table["query"] != "" \
                    else ""

                sampled_table_info = table_samping_service.sample(
                    query = query,
                    parsed_example = table,
                    table_sampling_type = table_sampling_type
                )

                self._save_sampled_table_info(context, table_sampling_type, sampled_table_info)