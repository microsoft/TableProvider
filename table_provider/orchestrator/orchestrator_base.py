# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from table_provider import logger


class OrchestratorBase:
    def __init__(self):
        self.extensions = []
        self.logger = logger

    def add_extension(self, extension):
        self.extensions.append(extension)

    def execute(self, request, context, response):
        logger.info(f"============[ORCHESTRATOR]: {self.__class__.__name__} start execution============")
        self._execute_all_extensions(request, context, response)
        logger.info(f"============[ORCHESTRATOR]: {self.__class__.__name__} end execution============")

    def _execute_all_extensions(self, request, context, response):
        if not self.extensions or len(self.extensions) == 0:
            raise ValueError("No extensions added to the orchestrator")

        for extension in self.extensions:
            try:
                extension.execute(request, context, response)
            except Exception as e:
                if extension.is_required():
                    raise Exception(f"Error executing extension {extension.__class__.__name__}: {e}")
                else:
                    logger.warn(f"Error executing extension {extension.__class__.__name__}: {e}")
                    continue

        return