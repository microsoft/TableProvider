# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from typing_extensions import Optional
from table_provider import logger
from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse


class ExtensionBase:
    def __init__(self):
        self.dependencies: Optional[List[type]] = None
        self.logger = logger

    def is_required(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    def should_execute(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        return True

    def _execute_internal(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        raise NotImplementedError("Subclasses must implement this method")

    def execute(self, request, context, response):
        logger.info(f"======[EXTENSION]: {self.__class__.__name__} start execution======")

        if not self.should_execute(request, context, response):
            logger.info(f"======[EXTENSION]: {self.__class__.__name__} skip execution======")
            return

        self._execute_internal(request, context, response)
        logger.info(f"======[EXTENSION]: {self.__class__.__name__} end execution======")