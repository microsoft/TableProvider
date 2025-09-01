# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union
from pydantic import BaseModel, model_validator

from table_provider.model.table_provider_enum import TableSamplingType


class TableSamplingRequest(BaseModel):
    sampling_type: Union[str, list]
    split: Optional[str] = 'train'
    n_cluster: Optional[int] = 3
    top_k: Optional[int] = 5
    whether_column_grounding: Optional[bool] = False
    token_limit_config: Optional[dict] = None

    @staticmethod
    def validate_single_sampling_type(single_sampling_type):
        valid_sampling_type = [sampling_type.value for sampling_type in TableSamplingType]
        if single_sampling_type not in valid_sampling_type:
            raise ValueError(f'Invalid sampling type: {single_sampling_type}. Must be one of {TableSamplingType}')

    @model_validator(mode='before')
    @classmethod
    def validate_table_sampling(cls, values: dict) -> dict:
        sampling_type = values.get('sampling_type')
        if isinstance(sampling_type, str):
            cls.validate_single_sampling_type(sampling_type)
        elif isinstance(sampling_type, list):
            for single_sampling_type in sampling_type:
                cls.validate_single_sampling_type(single_sampling_type)
        else:
            raise ValueError("Sampling type must be string or list")

        return values