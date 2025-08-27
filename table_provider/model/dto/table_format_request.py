# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union, List
from pydantic import BaseModel, model_validator

# 假设 TableFormatType 是一个枚举类，定义了所有有效的格式类型
from table_provider.model.table_provider_enum import TableFormatType

class TableFormatRequest(BaseModel):
    format_type: Union[str, List[str]] = None
    enable_composition: bool = False
    enable_separate: bool = False
    unique_format_id: Optional[str] = None

    @staticmethod
    def validate_single_format_type(single_format_type):
        valid_format_list = [format_type.value for format_type in TableFormatType]
        if single_format_type not in valid_format_list:
            raise ValueError(f'Invalid format type: {single_format_type}. Must be one of {valid_format_list}')

    @model_validator(mode='before')
    @classmethod
    def validate_table_format(cls, values: dict) -> dict:
        format_type = values.get('format_type')
        if isinstance(format_type, str):
            cls.validate_single_format_type(format_type)
        elif isinstance(format_type, list):
            for single_format_type in format_type:
                cls.validate_single_format_type(single_format_type)
        else:
            raise ValueError("Format type must be string or list")

        return values