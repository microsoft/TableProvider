# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from flask import abort, request
from flask_restful import Resource
from pydantic import ValidationError

from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse
from table_provider.orchestrator.orchestrators.table_provider_orchestrator import TableProviderOrchestrator

class TableProviderController:
    @staticmethod
    def process_tables(table_provider_request: TableProviderRequest):
        table_provider_orchestrator = TableProviderOrchestrator()
        table_provider_context = TableProviderContext()
        table_provider_response = TableProviderResponse()
        table_provider_orchestrator.execute(table_provider_request, table_provider_context, table_provider_response)
        return table_provider_response.result

class TableProviderExecutionApi(Resource):
    def __init__(self):
        self.table_provider_orchestrator = TableProviderOrchestrator()

    def post(self):
        user_input = request.get_json()
        if not isinstance(user_input, dict):
            abort(400, description="Expected a JSON object")
        try:
            table_provider_request = TableProviderRequest(**user_input)
        except ValidationError:
            abort(400, description="Invalid TableProvider request")
        table_provider_context = TableProviderContext()
        table_provider_response = TableProviderResponse()
        self.table_provider_orchestrator.execute(table_provider_request, table_provider_context, table_provider_response)
        return table_provider_response.result