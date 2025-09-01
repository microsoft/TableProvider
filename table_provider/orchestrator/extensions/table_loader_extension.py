# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from table_provider.model.table_provider_enum import TableSource
from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse
from table_provider.orchestrator.extension_base import ExtensionBase
from table_provider.service.table_loader_service import TableLoaderService


class TableLoaderExtension(ExtensionBase):
    def __init__(self):
        super().__init__()

    def is_required(self) -> bool:
        return True

    def should_execute(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        return True

    def _execute_internal(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        if request.table_loader.table_source == TableSource.task.value:
            table_loader_service = TableLoaderService(
                request.table_loader.task_name,
                request.table_loader.split,
                request.table_loader.use_small_sample_list,
                request.llm_azure_openai_config
            )

            context.tables = table_loader_service.dynamic_load_task_table(
                task_name=request.table_loader.task_name,
                load_local_dataset=request.table_loader.load_local_dataset
            )
        else:
            table_name = request.table_loader.table_name
            table_header = request.table_loader.table_header
            table_data = request.table_loader.table_content
            user_query = request.table_loader.user_query

            context.tables = TableLoaderService.parse_input_table(
                table_name=table_name,
                table_header=table_header,
                table_data=table_data,
                user_query=user_query
            )
