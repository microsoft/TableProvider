# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from table_provider.model.table_provider_enum import TableSource
from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest, TableLoaderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse
from table_provider.orchestrator.extension_base import ExtensionBase


class InputValidationExtension(ExtensionBase):
    def __init__(self):
        super().__init__()

    def is_required(self) -> bool:
        return True

    def should_execute(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        return True

    def _execute_internal(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        self._validate_table_loader(request.table_loader)

    @staticmethod
    def _validate_table_loader(request: TableLoaderRequest):
        if request.table_source == TableSource.input.value:
            if not request.table_name:
                raise ValueError("Table name is required")
            if not request.table_header:
                raise ValueError("Table header is required")
            if not request.table_content:
                raise ValueError("Table content is required")
        elif request.table_source == "task":
            if not request.task_name:
                raise ValueError("Task name is required")
        else:
            raise ValueError("Invalid table source")