# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tqdm import tqdm

from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse
from table_provider.orchestrator.extension_base import ExtensionBase
from table_provider.service.table_augmentation_service import TableAugmentationService


class TableAugmentationExtension(ExtensionBase):
    def __init__(self):
        super().__init__()

    def is_required(self) -> bool:
        return True

    def should_execute(self, request: TableProviderRequest, context: TableProviderContext,
                       response: TableProviderResponse):
        if request.enable_augmentation:
            return True

        return False

    @staticmethod
    def _save_table_augmentation_info(context: TableProviderContext, table_augmentation_type, augmented_table_info):
        augmented_table_info_list = context.table_augmentation_info_map.get(table_augmentation_type, [])
        augmented_table_info_list.append(augmented_table_info)
        context.table_augmentation_info_map[table_augmentation_type] = augmented_table_info_list

    def _execute_internal(self, request: TableProviderRequest, context: TableProviderContext, response: TableProviderResponse):
        tables = context.tables
        input_table_augmentation_type = request.table_augmentation.augmentation_type

        table_augmentation_types = []
        if isinstance(input_table_augmentation_type, str):
            table_augmentation_types.append(input_table_augmentation_type)
        elif isinstance(input_table_augmentation_type, list):
            table_augmentation_types.extend(input_table_augmentation_type)

        table_augmentation_service = TableAugmentationService(
            request.llm_azure_openai_config,
            request.table_augmentation.token_limit_config
        )

        for i in tqdm(range(len(tables)), desc="Table augmenting..."):
            table = tables[i]
            for table_augmentation_type in table_augmentation_types:
                augmented_table_info = table_augmentation_service.augment(
                    parsed_example=table,
                    table_augmentation_type=table_augmentation_type
                )

                self._save_table_augmentation_info(context, table_augmentation_type, augmented_table_info)
