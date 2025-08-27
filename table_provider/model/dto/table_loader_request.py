# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from table_provider.model.table_provider_enum import TableSource


class TableLoaderRequest(BaseModel):
    table_source: str
    table_name: Optional[Union[str, list]] = Field(default="")
    table_header: Optional[Union[str, list]] = Field(default="")
    table_content: Optional[Union[str, list]] = Field(default="")
    user_query: Optional[str] = Field(default="")
    task_name: Optional[str] = Field(default="")
    split: Optional[str] = Field(default="None")
    load_local_dataset: Optional[bool] = Field(default=False)
    use_small_sample_list: Optional[bool] = Field(default=False)

    @model_validator(mode='before')
    @classmethod
    def validate_table_source(cls, values: dict) -> dict:
        table_source = values.get('table_source')
        table_name = values.get('table_name')
        table_header = values.get('table_header')
        table_content = values.get('table_content')
        task_name = values.get('task_name')

        valid_table_source_list = [table_source.value for table_source in TableSource]
        if table_source not in valid_table_source_list:
            raise ValueError(f'Invalid table_source, table source must be in {valid_table_source_list}')

        if table_source == TableSource.input.value:
            if not table_name:
                raise ValueError("Table name is required for input table source")
            if not table_header:
                raise ValueError("Table header is required for input table source")
            if not table_content:
                raise ValueError("Table content is required for input table source")
        elif table_source == TableSource.task.value:
            if not task_name:
                raise ValueError("Task name is required for task table source")

        return values