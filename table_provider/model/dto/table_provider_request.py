# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

from pydantic import BaseModel

from table_provider.model.dto.table_loader_request import TableLoaderRequest
from table_provider.model.dto.table_sampling_request import TableSamplingRequest
from table_provider.model.dto.table_augmentation_request import TableAugmentationRequest
from table_provider.model.dto.table_format_request import TableFormatRequest

class TableProviderRequest(BaseModel):
    table_loader: TableLoaderRequest
    table_sampling: TableSamplingRequest
    table_augmentation: TableAugmentationRequest
    table_format: TableFormatRequest
    llm_azure_openai_config: Optional[dict] = None
    embedding_azure_openai_config: Optional[dict] = None
    enable_sampling: bool = False
    enable_augmentation: bool = False
