# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from table_provider.orchestrator.extensions.input_validation_extension import InputValidationExtension
from table_provider.orchestrator.extensions.table_augmentation_extension import TableAugmentationExtension
from table_provider.orchestrator.extensions.table_format_extension import TableFormatExtension
from table_provider.orchestrator.extensions.table_loader_extension import TableLoaderExtension
from table_provider.orchestrator.extensions.table_sampling_extension import TableSamplingExtension
from table_provider.orchestrator.orchestrator_base import OrchestratorBase


class TableProviderOrchestrator(OrchestratorBase):
    def __init__(self):
        super().__init__()
        self.add_extension(InputValidationExtension())
        self.add_extension(TableLoaderExtension())
        self.add_extension(TableSamplingExtension())
        self.add_extension(TableAugmentationExtension())
        self.add_extension(TableFormatExtension())
