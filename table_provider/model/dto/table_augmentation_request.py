# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union

from pydantic import BaseModel, model_validator

from table_provider.model.table_provider_enum import TableAugmentationType


class TableAugmentationRequest(BaseModel):
    token_limit_config: Optional[dict] = None
    augmentation_type: Union[str, list]

    @staticmethod
    def validate_single_augmentation_type(single_augmentation_type):
        valid_augmentation_list = [augmentation_type.value for augmentation_type in TableAugmentationType]
        if single_augmentation_type not in valid_augmentation_list:
            raise ValueError(f'Invalid augmentation type: {single_augmentation_type}. Must be one of {valid_augmentation_list}')

    @model_validator(mode='before')
    @classmethod
    def validate_table_augmentation(cls, values: dict) -> dict:
        augmentation_type = values.get('augmentation_type')
        if isinstance(augmentation_type, str):
            cls.validate_single_augmentation_type(augmentation_type)
        elif isinstance(augmentation_type, list):
            for single_augmentation_type in augmentation_type:
                cls.validate_single_augmentation_type(single_augmentation_type)
        else:
            raise ValueError("Augmentation type must be string or list")

        return values